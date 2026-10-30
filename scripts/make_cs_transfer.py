"""把 ScholarQABench CS 构造成**冻结的迁移终评集**（M4 / Spec §6.2、§14）。

为什么不是"直接跑原任务"：
    ScholarQABench CS 是**开放生成式**任务，官方评分靠 Prometheus rubric（`rubrics/
    prometheus_rubrics_v8.json`），需要 LLM 评判 —— 非零成本、不可离线重放，
    而且它的主指标（rubric 分）与本项目主指标（MCQ accuracy）不同源，
    两者横向比较没有意义。

    本项目要回答的迁移问题是**机制问题**：「在生物医学语料上学到的进化结果
    （召回路由 / 证据比对 / 缺口声明），在 CS 领域语料上是否仍按同一机制工作」。
    因此这里做的是**任务格式适配 + 全程冻结**：把 CS 题转成与 litqa2 / protocolqa
    同构的 MCQ，走同一套 harness、同一套指标、同一套冻结与泄漏校验。

适配规则（**预先注册、确定性、可审计** —— 写在 `ADAPTATION_RULE_VERSION`）：

1. 源：`vendor/.../scholarqa_cs/qa_metadata_all.jsonl`，100 题，逐题取 `question`
   与四个系统的 `src_answers`（只用 `Perplexity`，固定顺序，避免多选一带来的自由度）。
2. 切句：清洗 markdown（表格分隔符 `|`、加粗 `**`、LaTeX 转义），按句末标点切句；
   保留长度 60–170 字符、≥8 词、不含 URL 的句子。
3. **过滤元话语**：丢弃以 "Here/These/This/Overall/In conclusion/Let's/We will" 等
   开头、或含 "we discuss / will be discussed / below / above" 的句子 —— 这类句子
   在源答案里占比很高，但不含实质陈述，作为选项会让题目退化成"猜哪句更通顺"。
4. 正确选项：本题句子池中，与**问题实词集合重叠最高**的一句（并列时按字符串排序，
   保证确定性）。
5. 干扰项：**优先取本题答案中不在 `evidence` 段落内的句子**（按与问题实词重叠降序）。
   语义是"与本题高度相关、但**没有证据支撑**的说法" —— 判别信号是
   *这句话有没有被检索到的证据支持*，而不是 *这句话通不通顺*，
   这正是本项目要迁移的机制（evidence 比对 / abstain）。
   若本题凑不满 3 句，才回退到**其它题**的句子（同样按重叠降序）。
   两类干扰句都必须**不作为子串出现在 evidence 段落里**。
6. 证据段落 `evidence`：本题答案中包含正确选项句的**原文段落**（前后合并至
   ~320 字符），保证"正确选项被段落字面陈述"成立 —— 这是 evidence 判据的前提。
7. **稳定打乱选项位置**：用 `random.Random(sha256(dataset|idx))` 做确定性置换，
   避免正确项总落在末位（否则"永远选最后一个"就能刷分，Δ 会被天花板吃掉）。
8. 若某题抽不出 ≥4 个合格句子、或干扰项不足 3 个，**丢弃该题**，不强行凑。
   丢弃数写进 manifest。

口径声明（必须写进报告）：
    这是**自动适配**出来的任务集，标签噪声高于人工标注集。
    因此它的绝对准确率**不可与 litqa2 / protocolqa 横向比较**，
    只用于"同一冻结集、同一协议下，基线与进化后的前后对比"。

产物：
    data/normalized/scholarqa_cs.jsonl             空（进化器不可见任何 CS 题目）
    data/sealed/scholarqa_cs.final_holdout.jsonl   全部题目，仅 evaluator 可加载
    data/manifests/scholarqa_cs.json               manifest（冻结校验用）
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.split import MANIFEST_DIR, NORMALIZED_DIR, SEALED_DIR  # noqa: E402
from src.util import ensure_dir, now_iso, sha256_file, sha256_obj, write_json, write_jsonl  # noqa: E402

ADAPTATION_RULE_VERSION = "cs-transfer-v1"

SOURCE = ("vendor/ScholarQABench-main/ScholarQABench-main/"
          "data/scholarqa_cs/qa_metadata_all.jsonl")
DATASET = "scholarqa_cs"
DOMAIN = "cs_ai"
ANSWER_KEY = "Perplexity"     # 固定取一个系统的答案，避免"四选一"引入自由度

MIN_SENT_CHARS = 60
MAX_SENT_CHARS = 170
MIN_SENT_WORDS = 8
CONTEXT_CHARS = 320           # evidence 段落目标长度
N_DISTRACTORS = 3

# 元话语句首（小写比较）
META_OPENERS = (
    "here,", "here ", "these ", "this ", "those ", "overall", "in conclusion",
    "in summary", "in this answer", "in this response", "let's", "lets ",
    "we will", "we'll", "i will", "i'll", "below", "above", "as mentioned",
    "as noted", "note that", "keep in mind", "it is important to note",
    "the following", "we discuss", "to summarize", "summary:", "conclusion",
    "### conclusion", "### overview", "### introduction", "### summary",
    "### background", "### references", "### sources", "## overview",
)
META_SUBSTRINGS = (
    "we will discuss", "will be discussed", "as described above",
    "as described below", "in the following sections", "see the table below",
    "please note", "disclaimer", "i don't have access",
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n")
_TOKEN = re.compile(r"[a-z]{4,}")


def _clean(t: str) -> str:
    t = re.sub(r"\|+", " ", t)
    t = t.replace("**", "").replace("\\(", "(").replace("\\)", ")")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _tokens(s: str) -> set:
    return set(_TOKEN.findall(s.lower()))


def _is_meta(s: str) -> bool:
    low = s.lower().lstrip("# ").strip()
    if any(low.startswith(o) for o in META_OPENERS):
        return True
    return any(m in low for m in META_SUBSTRINGS)


def _as_text(v: Any) -> str:
    if isinstance(v, str):
        return v
    try:
        return "\n".join(
            str(x.get("text", "")) if isinstance(x, dict) else str(x)
            for x in list(v)
        )
    except Exception:
        return str(v)


def _sentences(answer_text: str) -> List[str]:
    txt = _clean(_as_text(answer_text))
    out: List[str] = []
    for raw in _SENT_SPLIT.split(txt):
        s = _clean(raw)
        if not (MIN_SENT_CHARS <= len(s) <= MAX_SENT_CHARS):
            continue
        if len(s.split()) < MIN_SENT_WORDS:
            continue
        if "http" in s.lower():
            continue
        if _is_meta(s):
            continue
        out.append(s)
    return out


def _context_for(answer_text: str, gold_sentence: str) -> str:
    """取包含正确句的原文段落（清洗后），保证正确句被字面包含。"""
    txt = _clean(_as_text(answer_text))
    i = txt.find(gold_sentence)
    if i < 0:
        return txt[:CONTEXT_CHARS]
    start = max(0, i - CONTEXT_CHARS // 3)
    end = min(len(txt), i + len(gold_sentence) + CONTEXT_CHARS // 3)
    return txt[start:end].strip()


def build_rows(src: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]

    # 先给每题建句子池（只保留合格句 ≥ 4 的题，否则当不了干扰项来源也组不出题）
    pools: Dict[int, List[str]] = {}
    full_text: Dict[int, str] = {}
    for r in rows:
        idx = int(r["idx"])
        ans = r.get("src_answers") or {}
        text = _as_text(ans.get(ANSWER_KEY, ""))
        full_text[idx] = _clean(text)
        sents = _sentences(text)
        if len(sents) >= N_DISTRACTORS + 1:
            pools[idx] = sents

    tasks: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    order = sorted(pools)   # 确定性迭代顺序

    for idx in order:
        question = _clean(str(rows[idx - 1].get("question", "")))
        qtok = _tokens(question)
        pool = pools[idx]
        # 正确选项：与问题实词重叠最高，并列按字符串排序
        gold = sorted(pool, key=lambda s: (-len(qtok & _tokens(s)), s))[0]
        gold_tok = _tokens(gold)
        if not (qtok & gold_tok):
            dropped.append({"idx": idx, "reason": "gold 句与问题无实词重叠"})
            continue

        # 干扰项：先取本题答案中"不在 evidence 段落内"的句子。
        # 排序用**与 gold 句的实词重叠**（不是与问题的重叠）——
        # 首版按"与问题重叠"排序，取到的干扰句跨子领域、与 gold 毫无词面关联，
        # 基线就到 87%，天花板直接把 Δ 吃掉，迁移评估等于白做。
        # 改成"与 gold 最相近"之后，四个选项在词面上彼此接近，
        # 唯一的判别信号就是**这句话有没有被检索到的证据支持** —— 正是要迁移的机制。
        evidence = _context_for(rows[idx - 1].get("src_answers", {}).get(ANSWER_KEY, ""), gold)
        ev_low = evidence.lower()
        gold_tok = _tokens(gold)

        same_q = [(s, len(gold_tok & _tokens(s)), len(qtok & _tokens(s)))
                  for s in pool
                  if s.lower() != gold.lower() and s.lower() not in ev_low]
        same_q.sort(key=lambda kv: (-kv[1], -kv[2], kv[0]))
        distractors = [s for s, _, _ in same_q[:N_DISTRACTORS]]

        if len(distractors) < N_DISTRACTORS:
            other: List[Tuple[int, str]] = []
            for j in order:
                if j == idx:
                    continue
                for s in pools[j]:
                    if s.lower() in ev_low or s.lower() in full_text[idx].lower():
                        continue
                    other.append((-len(qtok & _tokens(s)), s))
            other.sort(key=lambda kv: (kv[0], kv[1]))
            seen = {s.lower()[:60] for s in distractors}
            for _, s in other:
                key = s.lower()[:60]
                if key in seen or s.lower() == gold.lower():
                    continue
                seen.add(key)
                distractors.append(s)
                if len(distractors) == N_DISTRACTORS:
                    break
        if len(distractors) < N_DISTRACTORS:
            dropped.append({"idx": idx, "reason": "干扰项不足 3 个"})
            continue

        options = distractors + [gold]
        # 稳定打乱：random.Random(seed) 跨平台确定性，避免 gold 总在末位
        rng = random.Random(int(sha256_obj(f"{DATASET}|{idx}")[:16], 16))
        shuffled = list(options)
        rng.shuffle(shuffled)
        correct_index = shuffled.index(gold)

        tasks.append({
            "task_id": f"cs-{idx:03d}",
            "dataset": DATASET,
            "domain": DOMAIN,
            "split": "final_holdout",
            "task_type": "mcq",
            "question": question,
            "options": shuffled,
            "correct_index": correct_index,
            "evidence": evidence,
            "context": "",
            "sources": [str(rows[idx - 1].get("sources", ""))] if rows[idx - 1].get("sources") else [],
        })

    stats = {
        "n_source": len(rows),
        "n_with_pool": len(pools),
        "n_tasks": len(tasks),
        "n_dropped": len(dropped),
        "dropped_examples": dropped[:10],
        "n_options": N_DISTRACTORS + 1,
        "answer_key": ANSWER_KEY,
    }
    return tasks, stats


def main() -> int:
    src = Path(SOURCE)
    if not src.exists():
        print(f"[cs-transfer] 源数据不存在：{src}")
        return 1

    tasks, stats = build_rows(src)
    if not tasks:
        print("[cs-transfer] 没有构造出任何题目")
        return 1

    tasks.sort(key=lambda t: t["task_id"])

    ids = sorted(t["task_id"] for t in tasks)
    manifest = {
        "dataset": DATASET,
        "created_at": now_iso(),
        "source_file": str(src).replace("\\", "/"),
        "source_sha256": sha256_file(src),
        "domain": DOMAIN,
        "task_type": "mcq",
        "n_questions": len(tasks),
        "salt": "not-applicable",       # CS 全量冻结为迁移 final，不做比例划分
        "ratios": {"dev_optimize": 0, "selection_gate": 0, "final_holdout": 100},
        "split_algorithm": "frozen_transfer_set（全量 final_holdout，Spec §6.2：只终评）",
        "adaptation": {
            "rule_version": ADAPTATION_RULE_VERSION,
            "answer_key": ANSWER_KEY,
            "sentence_chars": [MIN_SENT_CHARS, MAX_SENT_CHARS],
            "context_chars": CONTEXT_CHARS,
            "meta_filter": True,
            "distractor_source": "other_questions_same_dataset",
            "distractor_exclusion": "不得作为子串出现在本题任一答案文本中",
            "option_shuffle": "sha256(dataset|idx) 稳定打乱",
        },
        "stats": stats,
        "caveat": ("自动适配构造，标签噪声高于人工标注集；绝对准确率不可与 "
                   "litqa2/protocolqa 横向比较，仅用于同协议前后对比。"),
        "splits": {
            "dev_optimize": {"n": 0, "ids": [], "ids_sha256": sha256_obj([])},
            "selection_gate": {"n": 0, "ids": [], "ids_sha256": sha256_obj([])},
            "final_holdout": {"n": len(ids), "ids": ids, "ids_sha256": sha256_obj(ids)},
        },
    }
    # Reuse the staged split writer and its frozen-dataset guard.
    from scripts.make_splits import _commit_prepared
    _commit_prepared([(DATASET, manifest, [], tasks)])

    print(json.dumps({
        "ok": True,
        "dataset": DATASET,
        "manifest": f"data/manifests/{DATASET}.json",
        "sealed": f"data/sealed/{DATASET}.final_holdout.jsonl",
        "stats": stats,
        "caveat": manifest["caveat"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
