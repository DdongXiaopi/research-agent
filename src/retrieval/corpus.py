"""真实语料池（R0）。

为什么要有这个模块
    R0 的发现：真实模型的对错主要由夹具里那句**模板假话**决定——
    `Related work on <题干>. Under the conditions described, <错误选项> was reported.`
    干扰段落字面断言一个错误选项是真的。离线模拟器按词重叠打分，对它几乎免疫；
    真实 LLM 几乎没抵抗力（protocolqa：有它 9.1% / 没它 61.5%）。
    于是"改提示词涨了多少分"衡量的是**能不能识别那句模板假话**，不是会不会推理。

    本模块把语料的**每一个字**都换成数据集自带的真实论文文本：
      - gold 段落    = 该题的真实证据（LitQA2 key-passage / ProtocolQA protocol /
                       SuppQA 全文），**不再追加** `Reported finding: <答案>`
      - 干扰段落      = 同数据集**其它题**的真实证据，按与本题题干的词面相关度
                       取最靠前的几条——这正是真实检索器会捞回来的混淆项
      - 背景段落      = 相关度最低的真实段落
    难度因此回到"读懂证据再判断"，而不是"识破模板"。

泄漏边界（重要）
    语料池默认**排除 final_holdout**。否则 dev/gate 的语料里会出现封存题的
    段落文本，等于把封存材料提前曝光。封存集自己评测时再显式放开。

版权
    SuppQA 全文由 `scripts/fetch_suppqa_fulltext.py` 抓取，落在
    `data/cache/`（.gitignore 已排除，**不随仓库分发**）。
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from ..util import path, read_jsonl, sha256_file

REAL_CORPUS_VERSION = "real-passage-v1"

# 长证据（ProtocolQA 的 protocol 中位 4159 字符）切成窗口，取最相关的那一段。
# 真实检索系统也是这么做的：召回文档 → 段落级重排 → 只把 top 段落喂给模型。
WINDOW_CHARS = 1400
WINDOW_STRIDE = 700

STOP_WORDS = {
    "the", "and", "for", "that", "with", "this", "from", "have", "has", "been",
    "are", "was", "were", "which", "what", "when", "into", "using", "used",
    "may", "can", "should", "following", "these", "those", "than", "then",
    "you", "your", "their", "not", "but", "all", "any", "how", "why", "who",
    "its", "of", "in", "on", "to", "as", "at", "by", "or", "is", "be", "if",
    "a", "an", "we", "our", "it", "they", "there", "here", "such", "other",
    "more", "most", "some", "each", "per", "via", "had", "did", "does",
}


def tokens(text: str) -> Set[str]:
    """实词集合（供词面相关度打分）。"""
    return {t for t in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", (text or "").lower())
            if t not in STOP_WORDS}


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


def windows(text: str, size: int = WINDOW_CHARS,
            stride: int = WINDOW_STRIDE) -> List[str]:
    """把长文本切成重叠窗口（按句子边界累积，不硬切单词）。"""
    sents = _sentences(text)
    if not sents:
        return []
    if sum(len(s) for s in sents) <= size:
        return [" ".join(sents)]
    out: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for s in sents:
        cur.append(s)
        cur_len += len(s) + 1
        if cur_len >= size:
            out.append(" ".join(cur))
            # 回退 stride 长度作为下一个窗口的起点
            back = 0
            drop: List[str] = []
            for prev in reversed(cur):
                if back >= stride:
                    break
                drop.insert(0, prev)
                back += len(prev) + 1
            cur = list(drop)
            cur_len = sum(len(x) + 1 for x in cur)
    if cur and " ".join(cur) not in out:
        out.append(" ".join(cur))
    return out or [" ".join(sents)]


def best_window(text: str, query_tokens: Set[str], rank: int = 0) -> str:
    """取与查询词面最相关的第 `rank` 个窗口（rank=0 = 最相关）。

    `rank=1` 用来模拟"检索捞到了文档但位置不对"——真实的检索失败模式，
    替代旧夹具里靠改写制造的 `trap` 桶。
    """
    ws = windows(text)
    if len(ws) <= 1:
        return ws[0] if ws else ""
    scored = sorted(
        range(len(ws)),
        key=lambda i: (-len(tokens(ws[i]) & query_tokens), i),
    )
    return ws[scored[min(rank, len(scored) - 1)]]


def _suppqa_fulltext(doi: str, key: str) -> str:
    """SuppQA 全文来自抓取缓存；缺失则返回空串（由调用方显式处理）。"""
    d = path("data", "cache", "suppqa_fulltext")
    fp = d / f"{key}.json"
    if not fp.exists():
        return ""
    try:
        rec = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return (rec.get("text") or "") if rec.get("ok") else ""


def _sha1_16(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def evidence_of(task: Dict[str, Any]) -> str:
    """取该题的真实证据文本：优先全文缓存，其次归一化 evidence 字段。"""
    ev = (task.get("evidence") or "").strip()
    if task.get("dataset") == "suppqa":
        # 必须与 fetch_suppqa_fulltext.py 的 key 算法**逐字一致**（含 .strip()），
        # 否则算出的哈希对不上缓存文件，池会静默变空。
        doi = str((task.get("sources") or [""])[0]).replace("https://doi.org/", "").strip()
        ft = _suppqa_fulltext(doi, _sha1_16(doi or task.get("task_id", "")))
        if len(ft) >= 400:
            return ft
    return ev


_POOL_CACHE: Dict[tuple, List[Dict[str, Any]]] = {}


def load_pool(dataset: str, exclude_splits: Sequence[str] = ("final_holdout",),
              allow_all: bool = False) -> List[Dict[str, Any]]:
    """加载某数据集的真实证据段落池（每个条目 = 一篇/一段真实文献文本）。

    返回条目字段：`task_id / text / title / source / url / tokens`。
    `tokens` 只在内存里，不会写进任何产物。
    """
    from ..data.split import FinalHoldoutSealed, _validate_dataset_name
    _validate_dataset_name(dataset)
    if type(allow_all) is not bool:
        raise ValueError("allow_all must be a boolean")
    if isinstance(exclude_splits, (str, bytes)):
        raise ValueError("exclude_splits must be a collection of split names")
    exclude_splits = tuple(exclude_splits)
    if any(not isinstance(value, str) for value in exclude_splits):
        raise ValueError("exclude_splits must contain only split names")
    if allow_all and os.environ.get("RSI_ALLOW_FINAL") != "1":
        raise FinalHoldoutSealed("sealed corpus requires the evaluator final-holdout unlock")
    fp = path("data", "normalized", f"{dataset}.jsonl")
    sealed = path("data", "sealed", f"{dataset}.final_holdout.jsonl")
    inputs = [fp] + ([sealed] if allow_all else [])
    if dataset == "suppqa":
        inputs.extend(sorted(path("data", "cache", "suppqa_fulltext").glob("*.json")))
    identities = tuple((str(item), sha256_file(item) if item.is_file() else None) for item in inputs)
    scope = tuple(sorted(set(exclude_splits))) if not allow_all else None
    key = (dataset, scope, identities)
    if key in _POOL_CACHE:
        return copy.deepcopy(_POOL_CACHE[key])

    rows = read_jsonl(fp)
    # SuppQA 是**整集封存**：归一化文件里一行都没有，题目全在 data/sealed/。
    # 只有 evaluator（allow_all=True）才把封存区并入语料池 —— 默认不并，
    # 否则 dev/gate 的干扰段落会用到封存题的材料，等于提前曝光。
    if allow_all:
        rows = rows + read_jsonl(sealed)
    if not rows:
        _POOL_CACHE[key] = []
        return []
    bad = set(exclude_splits) | {"final_holdout"} if not allow_all else set()
    pool: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("split") in bad:
            continue
        text = evidence_of(r)
        if len(text.strip()) < 40:
            continue
        src = (r.get("sources") or [""])[0]
        pool.append({
            "task_id": r.get("task_id", ""),
            "question": r.get("question", ""),
            "text": text,
            "title": (r.get("question", "") or "")[:80],
            "source": src or "corpus://local",
            "url": src or "",
            "tokens": tokens(text),
        })
    for old_key in list(_POOL_CACHE):
        if old_key[0:2] == key[0:2]:
            del _POOL_CACHE[old_key]
    _POOL_CACHE[key] = pool
    return copy.deepcopy(pool)


def rank_others(task: Dict[str, Any], pool: Sequence[Dict[str, Any]],
                query_tokens: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """把池里**其它题**的段落按与本题的相关度排序（真实 hard negative）。

    相关度只用题干 + 全部选项的实词重合，**不看 correct_index**。
    排在最前面的就是真实检索器最容易误捞的混淆段落。
    """
    qt = query_tokens if query_tokens is not None else tokens(
        (task.get("question", "")) + " " + " ".join(task.get("options", [])))
    me = task.get("task_id", "")
    scored = []
    for p in pool:
        if p["task_id"] == me:
            continue
        s = len(p["tokens"] & qt)
        scored.append((-s, p["task_id"], p))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [x[2] for x in scored]


def self_test() -> Dict[str, Any]:
    """语料自检（供 `rsi.py verify --strict`）：钉死 R0 修掉的东西不会悄悄回来。

    三条：
      1. v3 语料里不许再出现合成模板句（答案直给 / 断言错误选项的假话）；
      2. top_k 内不许出现重复段落（重复等于把 k 降了）；
      3. 语料池必须非空，且不含封存题（否则等于提前曝光封存材料）。
    """
    from .gateway import REAL_CORPUS, RetrievalGateway  # 局部导入避免循环

    checks: List[Dict[str, Any]] = []
    gw = RetrievalGateway(corpus_version=REAL_CORPUS, top_k=3)
    bad_template, bad_dup = 0, 0
    n = 0
    for ds in ("litqa2", "protocolqa"):
        try:
            from ..data.split import load_tasks
            tasks = load_tasks(ds, splits=["dev_optimize"])[:20]
        except Exception:
            continue
        for t in tasks:
            n += 1
            try:
                docs = gw.search(t, skill_recalled=True, rerank=True)
            except Exception:
                continue
            texts = [d.get("text", "") for d in docs]
            if any("Reported finding:" in x or "Related work on " in x for x in texts):
                bad_template += 1
            if len(set(texts)) != len(texts):
                bad_dup += 1

    checks.append({
        "check": "v3_corpus_has_no_synthetic_template",
        "passed": bad_template == 0,
        "detail": f"{n} 题里有 {bad_template} 题的语料仍含合成模板句",
    })
    checks.append({
        "check": "v3_topk_has_no_duplicate",
        "passed": bad_dup == 0,
        "detail": f"{n} 题里有 {bad_dup} 题的 top_k 出现重复段落",
    })
    pool_ok = all(len(load_pool(ds)) > 0 for ds in ("litqa2", "protocolqa"))
    checks.append({
        "check": "real_corpus_pool_nonempty",
        "passed": bool(pool_ok),
        "detail": "真实语料池为空 → v3 无法构造（检查 data/normalized 与全文缓存）",
    })
    sealed = sum(1 for ds in ("litqa2", "protocolqa")
                 for p in load_pool(ds) if p.get("split") == "final_holdout")
    checks.append({
        "check": "pool_excludes_final_holdout",
        "passed": sealed == 0,
        "detail": f"语料池里有 {sealed} 条封存题段落（会提前曝光封存材料）",
    })
    return {"ok": all(c["passed"] for c in checks), "checks": checks, "tasks": n}


def pool_coverage(dataset: str, *, allow_all: bool = False) -> Dict[str, Any]:
    """池的健康度自检：多少题有真实证据、SuppQA 全文命中率。"""
    pool = load_pool(dataset, allow_all=allow_all)
    total = len(pool)
    if dataset == "suppqa":
        rows = read_jsonl(path("data", "normalized", "suppqa.jsonl"))
        with_ft = sum(1 for r in rows
                      if len(_suppqa_fulltext(
                          str((r.get("sources") or [""])[0]).replace("https://doi.org/", ""),
                          _sha1_16(str((r.get("sources") or [""])[0])
                                   .replace("https://doi.org/", "") or r.get("task_id", "")))) >= 400)
        return {"dataset": dataset, "pool": total, "normalized": len(rows),
                "with_fulltext": with_ft}
    return {"dataset": dataset, "pool": total}
