"""检索网关（薄适配层，Spec §3.1）。

M1 使用 **offline fixture**：由数据集自带的证据段落构造稳定的本地语料，
保证零网络、零成本、可重放。实验性的真实 PubMed 论文快照在
`pubmed_snapshot.py` 单独实现。可选 PaperQAGateway 接收调用方提供的
PaperQA Doc/Text 记录并做本地片段检索；不启动其 agent，也不把 clients
当成自然语言问题到论文的搜索器。
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Sequence, Set

from ..util import sha256_text
from . import chunking  # 【P8】证据分块 + 相关性筛选
from . import corpus as realcorpus  # R0：真实语料池（v3）


class RetrievalError(RuntimeError):
    pass


class DependencyMissing(RetrievalError):
    pass


FILLERS = [
    "Background: methodological considerations for reproducibility in laboratory workflows.",
    "Review: common sources of variance in measurement and reporting across studies.",
    "Editorial: standards for documenting materials, reagents, and instrument settings.",
]


# ---- 内容派生打分权重（【A1】不再允许用 is_gold 排序）----
#
# 为什么必须改：旧实现是 `lead = [d for d in docs if d.get("is_gold")]`，
# 也就是"命中 Skill → 领域过滤"这一步**直接读答案键**把 gold 提到首位。
# 那样测出来的 description 维度收益，本质是"进化学会触发一个 oracle 过滤器"，
# 真实检索里没有 is_gold 这个字段 → 结论不可外推（M1 +24.79pp / M2 +7.69pp 皆如此）。
#
# 改成内容派生后，三个信号都由段落**自身内容**算出：
#   q_topical    段落与 question 的实词重合 —— 通用检索器的默认信号（奖励复述题干的干扰项）
#   conclusion   段落是否字面陈述了某个候选结论 —— 领域 Skill 才有的过滤器
#   opt_overlap  段落与 options 的实词重合 —— 显式重排用的**弱**信号
# 三者权重拉开档次（1 : 100 : 3），保证"召回 > 重排 > 无"这个顺序是内容决定的。
CONCLUSION_WEIGHT = 100          # 命中 Skill：偏好"给出具体结论"的段落
OPTION_OVERLAP_WEIGHT = 3        # 显式重排：只按"与候选选项的词面相关"微调（弱信号）

# 【N3】检索夹具判据版本 —— 进入 final 的"协议指纹"。改排序规则必须递增。
# v1 = is_gold 置顶（A1 前的 oracle 夹具）；v2 = 内容派生打分 + hard negative（A1 后）。
RETRIEVAL_RULE_VERSION = "offline-fixture-content-v2"
HARDNEG_BUCKET = 15              # 额外 15% 的题：干扰项也字面陈述一个错误结论
TRAP_BUCKET = 10                 # 10% 的题：gold 只以改写出现 + 干扰项字面陈述（召回会失手）

# ---- 语料版本（R0）----
#
# v2（旧协议，历史结论专用）：段落由模板合成，gold 追加 `Reported finding: <答案>`，
#   干扰项在 hardneg 桶里**字面断言一个错误选项是真的**。
#   真实模型实测：protocolqa 有那句假话 9.1% / 没有 61.5% —— 测的是"识破模板"，
#   不是"读懂证据"。离线模拟器按词重叠打分对它免疫，两台仪器测的不是一回事。
#   ⇒ 只保留给 `configs/experiment.legacy-v2.yaml` 复现历史数字用。
# v3（**默认**）：gold = 该题真实论文证据（不追加答案句）；
#   干扰 = 同数据集其它题的真实段落，按词面相关度取最靠前的（真实 hard negative）；
#   背景 = 相关度最低的真实段落。详见 `src/retrieval/corpus.py`。
#
# 2026-09-26 升级：默认从 v2 切到 v3。理由见 docs/rebuild-2026-09-26.md ——
# v2 测的是"识破模板假话"，protocolqa/dev 的进化收益方向在 v2 与 v3 下相反
# （−3.28pp vs +9.84pp）。**注意**：离线模拟器（fake）在 v3 上是随机水平
# （词面重叠打不出靶子），所以 v3 下的结论必须来自真实模型。
SYNTHETIC_CORPUS = "synthetic-v2"
REAL_CORPUS = "real-v3"
DEFAULT_CORPUS_VERSION = REAL_CORPUS

# ---- passage 标识符方案（【P6/P7】封存题 ID 不应出现在任何落盘产物里）----
#
# 背景：doc_id 原样形如 `cand::suppqa-037::1`，**把 task_id 编进了标识符**。
# 模型回引时会把它写进响应正文（`ANSWER: 3 … cand::suppqa-037::1`），
# 而 HttpProvider 默认把原始响应落盘到 `artifacts/http_cache` —— 封存题 ID 于是
# 出现在"进化路径可读"的共享缓存里，verify 的泄漏扫描因此变红（2026-09-26 实测：
# 命中 cs-095、suppqa-037/038/039/046/058）。
#
# 两个方案：
#   task_scoped  （现状）`cand::<task_id>::<i>` —— 含 task_id，会污染落盘产物；
#   index_scoped （推荐）`cand::<i>`           —— 题内唯一即可，不含任何题标识。
#
# 判定 gold 用的是文档上的 `is_gold` 布尔字段（runner.py:211），不是 id 字符串，
# 所以换方案**不影响 citation_precision 的计算口径**。但换方案会改变模型看到的输入
# 与同分排序的 tie-break（`sha256(doc_id)`），因此它**属于协议**：
# 一旦切换而未重跑封存基线，`check_protocol` / 进化护栏会直接 raise（fail-loud）。
DOC_ID_TASK_SCOPED = "task_scoped"
DOC_ID_INDEX_SCOPED = "index_scoped"
DEFAULT_DOC_ID_SCHEME = DOC_ID_TASK_SCOPED


def doc_id(kind: str, task_id: str, index: Optional[int] = None,
           scheme: str = DEFAULT_DOC_ID_SCHEME) -> str:
    """生成 passage 标识符。`kind` ∈ {gold, cand, filler}。

    只认两个白名单方案：写错必须炸，不能"看着像就按默认跑"。
    """
    if scheme not in (DOC_ID_TASK_SCOPED, DOC_ID_INDEX_SCOPED):
        raise RetrievalError(
            f"未知 doc_id 方案 '{scheme}'（可用：{DOC_ID_TASK_SCOPED} / {DOC_ID_INDEX_SCOPED}）")
    if scheme == DOC_ID_INDEX_SCOPED:
        return f"{kind}::{index}" if index is not None else f"{kind}::0"
    return f"{kind}::{task_id}::{index}" if index is not None else f"{kind}::{task_id}"


def _bucket(task_id: str, tag: str) -> int:
    return int(sha256_text(f"{tag}|{task_id}")[:8], 16) % 100


def _paraphrase(option: str) -> str:
    """把干扰项改写成"部分支持"的文献表述：只保留一半实词。

    真实检索里，干扰项往往也有文献部分支持；这里保证它的词汇重合度**低于** gold 文档，
    否则"逐项证据比对"策略反而会被完美匹配的干扰项带偏。
    """
    words = [w for w in option.split() if len(w) > 3]
    kept = words[::2] or words[:1] or ["this"]
    return " ".join(kept)


def _topic(question: str) -> str:
    words = [w for w in question.replace("?", " ").split() if len(w) > 3]
    return " ".join(words[:5]) or "this system"


MIN_REAL_EVIDENCE = 80         # 真实短段落（LitQA2 部分 key-passage 仅百余字）可接受
MIN_FULLTEXT = 400             # SuppQA 未抓到全文时只有"标题+DOI"，必须拒绝


def _pick(text: str, qtok: Set[str], opttok: Set[str],
          chunk: Optional[Dict[str, Any]], exclude_top: int = 0) -> str:
    """从一篇文档里挑出要进 prompt 的那段正文。

    `chunk=None`（默认，协议 v2）= 旧行为：单个最相关滑动窗口（`best_window`）。
    `chunk=配置字典`（P8 启用）= 按步骤/句边界切块 → 逐块与"问题+选项"打分 →
    预算内取高分块（详见 `chunking.py`）。预算由构造保证 ≤ `max_doc_chars`。
    """
    if not chunk or not chunk.get("enabled"):
        return realcorpus.best_window(text, qtok | opttok, rank=exclude_top)
    out, _stats = chunking.assemble(
        text, qtok, opttok,
        budget_chars=int(chunk["budget_chars"]),
        max_chunks=int(chunk["max_chunks"]),
        target_chars=int(chunk["target_chars"]),
        max_chars=int(chunk["max_chars"]),
        option_weight=int(chunk["option_weight"]),
        exclude_top=exclude_top,
    )
    return out


def _dedup_by_text(ranked: List[Dict[str, Any]],
                   qt: Set[str], opttok: Optional[Set[str]] = None,
                   chunk: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """按**取窗后的实际文本**去重。

    不去重会出问题：ProtocolQA 里多个 protocol 共享完全相同的标准步骤段落，
    实测 top_k 三条里出现两段一字不差的 "Add 70 µL of RNase free water…"，
    白白占掉两个位置，等于把 top_k 降到 2。
    """
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for p in ranked:
        w = _pick(p["text"], qt, opttok if opttok is not None else set(), chunk)
        key = sha256_text(w)[:16]
        if key in seen:
            continue
        seen.add(key)
        q = dict(p)
        q["_window"] = w
        out.append(q)
    return out


def build_real_corpus(task: Dict[str, Any],
                      pool: Optional[Sequence[Dict[str, Any]]] = None,
                      doc_id_scheme: str = DEFAULT_DOC_ID_SCHEME,
                      chunk: Optional[Dict[str, Any]] = None
                      ) -> List[Dict[str, Any]]:
    """【R0 / v3】全部由数据集自带的**真实论文文本**构成的语料。

    与 v2 的三点差别（每一点都对应一个实测出的问题）：
    1. gold 段落**不再追加** `Reported finding: <答案>` —— 那是答案直给；
    2. 干扰段落是**同数据集其它题的真实段落**（按词面相关度取最靠前的，
       即真实检索器最容易误捞的），不再有"字面断言错误选项"的模板假话；
    3. `trap` 桶改为"gold 取次优窗口"（检索捞对文档但位置不对），
       不再靠改写制造。

    `hard` 桶保留（~15% 的题语料里根本没有答案），这是真实检索失败模式。
    """
    task_id = task["task_id"]
    ds = task.get("dataset", "")
    # 封存题（终评）需要用**含封存区**的池；dev/gate 必须用不含封存区的池。
    # SuppQA 是整集封存，池只可能来自 data/sealed/，这里必须放行。
    allow_all = str(task.get("split")) == "final_holdout"
    pool = list(pool) if pool is not None else list(
        realcorpus.load_pool(ds, allow_all=allow_all))
    # 【P8】分块打分需要把"题干词"与"选项词"分开（选项权重更高）；
    # 旧口径 `qt` 是两者的并集，继续用于 `rank_others` 的干扰项排序，保持不变。
    qtok = realcorpus.tokens(task.get("question", ""))
    opttok = realcorpus.tokens(" ".join(task.get("options", [])))
    qt = qtok | opttok

    ranked = _dedup_by_text(realcorpus.rank_others(task, pool, qt), qtok, opttok, chunk)
    if len(ranked) < 3:
        raise RetrievalError(
            f"数据集 '{ds}' 的真实语料池只有 {len(ranked)} 条可用的其它题段落，"
            "不足以构造 v3 语料。请先确认 data/normalized 与 SuppQA 全文缓存。"
        )
    ev = realcorpus.evidence_of(task)
    # SuppQA 的坑：parquet 里只有 paper-title + DOI，没有正文。
    # 未抓到全文时 evidence 就是那百来字的题名出处 —— 它不是证据，必须拦掉，
    # 否则整集会静默退化为"用标题猜答案"。
    # 因此 dev/gate 等非封存集**强制先抓全文**（缺则报错，逼着跑 fetch 脚本）；
    # 但封存集（final_holdout）是固定封死的，个别题若真抓不到全文，应降级为
    # evidence_missing（无 gold → 模型应弃权），而不是让整轮崩溃。
    if ds == "suppqa" and len(ev.strip()) < MIN_FULLTEXT and not allow_all:
        raise RetrievalError(
            f"任务 {task_id}（suppqa）没有抓到全文（长度 {len(ev.strip())} "
            f"< {MIN_FULLTEXT}）。v3 语料不允许用标题/出处冒充正文；"
            "请先跑 scripts/fetch_suppqa_fulltext.py。"
        )
    if ds == "suppqa" and len(ev.strip()) < MIN_FULLTEXT and allow_all:
        # 封存集里个别题真缺全文：走 no_evidence → gold_missing 路径，
        # 语料里没有答案，模型应弃权（真实检索失败模式，不算崩溃）。
        pass

    trap = _bucket(task_id, "trap") < TRAP_BUCKET
    # 证据过短（LitQA2 有 ~5% 的 key-passage 不到 80 字符）在真实里就是
    # "检索不到能用的证据" —— 归入 hard 桶，而不是硬造一段出来。
    no_evidence = len(ev.strip()) < MIN_REAL_EVIDENCE
    gold_missing = (_bucket(task_id, "hard") < 15) or no_evidence

    docs: List[Dict[str, Any]] = []
    # 干扰项：相关度最高的其它题段落（真实 hard negative —— 检索器真会捞回来）
    for i, p in enumerate(ranked[:3]):
        docs.append({
            "doc_id": doc_id("cand", task_id, i, doc_id_scheme),
            "title": f"Related study {i}",
            "text": p.get("_window") or _pick(p["text"], qtok, opttok, chunk),
            "source": p["source"], "url": p["url"], "is_gold": False,
        })
    # 背景段落：相关度最低的真实段落（不是通用模板句）。
    # 必须从已用作干扰项的前 3 条**之外**取，否则 top_k 里会出现重复文本。
    rest = ranked[3:]
    tail = list(reversed(rest))[:2]
    for i, p in enumerate(tail):
        docs.append({
            "doc_id": doc_id("filler", task_id, i, doc_id_scheme),
            "title": f"Background {i}",
            "text": p.get("_window") or _pick(p["text"], qtok, opttok, chunk),
            "source": p["source"], "url": p["url"], "is_gold": False,
        })

    if gold_missing:
        for d in docs:
            d["evidence_missing"] = True
        return docs  # 语料里根本没有答案 → 归因 skill.missing / capability_limit

    gold_text = _pick(ev, qtok, opttok, chunk, exclude_top=1 if trap else 0)
    # 若某条干扰段落的窗口文本与 gold 撞车（ProtocolQA 里多个 protocol 共享
    # 标准步骤，实测有此情况），剔除它 —— top_k 里出现两段一样的内容等于降 k。
    gkey = sha256_text(gold_text)[:16]
    docs = [d for d in docs if sha256_text(d["text"])[:16] != gkey]

    gold_doc = {
        "doc_id": doc_id("gold", task_id, None, doc_id_scheme),
        "title": f"Evidence for: {task['question'][:60]}",
        "text": gold_text,
        "source": (task.get("sources") or ["corpus://local"])[0],
        "url": (task.get("sources") or [""])[0],
        "is_gold": True,
    }
    base: List[Dict[str, Any]] = []
    distractors = [d for d in docs if d["doc_id"].startswith("cand::")]
    fillers = [d for d in docs if d["doc_id"].startswith("filler::")]
    if fillers:
        base.append(fillers[0])
    if distractors:
        base.append(distractors[0])
    if len(fillers) > 1:
        base.append(fillers[1])
    base.append(gold_doc)
    if len(distractors) > 1:
        base.append(distractors[1])
    return base


def build_fixture_corpus(task: Dict[str, Any],
                         corpus_version: str = DEFAULT_CORPUS_VERSION,
                         doc_id_scheme: str = DEFAULT_DOC_ID_SCHEME,
                         chunk: Optional[Dict[str, Any]] = None
                         ) -> List[Dict[str, Any]]:
    """为单个任务构造语料。

    `corpus_version`:
      - `real-v3`（默认）：真实论文语料（R0）。离线模拟器在它上面是随机水平，
        结论必须来自真实模型。
      - `synthetic-v2`：合成夹具，仅供 `configs/experiment.legacy-v2.yaml`
        复现 2026-09-26 之前的历史结论。

    `chunk`：P8 证据分块配置（None = 旧的单窗口行为）。
    """
    # 只认两个白名单值：写错版本号必须炸，不能"看着像 real 就按 real 跑"。
    if str(corpus_version) == REAL_CORPUS:
        return build_real_corpus(task, doc_id_scheme=doc_id_scheme, chunk=chunk)
    if str(corpus_version) == SYNTHETIC_CORPUS:
        return _build_synthetic_corpus(task, doc_id_scheme=doc_id_scheme)
    raise RetrievalError(
        f"未知语料版本 '{corpus_version}'（可用：{SYNTHETIC_CORPUS} / {REAL_CORPUS}）")


def _build_synthetic_corpus(task: Dict[str, Any],
                            doc_id_scheme: str = DEFAULT_DOC_ID_SCHEME
                            ) -> List[Dict[str, Any]]:
    """v2 合成夹具（原实现，逐字保留，仅重命名）。"""
    task_id = task["task_id"]
    gold_option = task["options"][int(task["correct_index"])]
    wrong = [o for i, o in enumerate(task["options"]) if i != task["correct_index"]]
    evidence = (task.get("evidence") or "").strip() or task["question"]
    sources = task.get("sources") or []

    trap = _bucket(task_id, "trap") < TRAP_BUCKET
    # 【A1】hard negative：干扰项**也**字面陈述一个错误结论，且同样复述题干主题词。
    # 没有它，evidence/召回 策略在本夹具里没有任何失败模式（实测 broke=0），
    # 收益会被系统性高估。有了它，"段落陈述了结论"不再等于"陈述的是对的结论"。
    hardneg = _bucket(task_id, "hardneg") < HARDNEG_BUCKET
    gold_tail = f"{_paraphrase(gold_option)} was the result observed in the reported experiment." \
        if trap else f"Reported finding: {gold_option}."
    gold_doc = {
        "doc_id": doc_id("gold", task_id, None, doc_id_scheme),
        "title": f"Evidence for: {task['question'][:60]}",
        "text": f"{evidence} {gold_tail}",
        "source": sources[0] if sources else "fixture://local-corpus",
        "url": sources[0] if sources else "",
        "is_gold": True,
    }
    docs: List[Dict[str, Any]] = []
    topic = _topic(task["question"])
    for i, opt in enumerate(wrong):
        if (trap or hardneg) and i == 0:
            # 对抗性干扰项：**同时**复述题干主题词 + 字面陈述一个错误结论。
            # 于是它在 q_topical 上不输 gold、在 conclusion 上也不输 gold，
            # 召回/证据策略在这批题上会真的失手（不再有"只要命中就赢"的天花板）。
            text = f"Related work on {topic}. Under the conditions described, {opt.rstrip('.')} was reported."
        else:
            text = f"Related work on {topic} examined different experimental conditions."
        docs.append(
            {
                "doc_id": doc_id("cand", task_id, i, doc_id_scheme),
                "title": f"Related study {i}",
                "text": text,
                "source": "fixture://related",
                "url": "",
                "is_gold": False,
            }
        )
    for i, f in enumerate(FILLERS):
        docs.append(
            {
                "doc_id": doc_id("filler", task_id, i, doc_id_scheme),
                "title": f"Background {i}",
                "text": f,
                "source": "fixture://filler",
                "url": "",
                "is_gold": False,
            }
        )

    gold_missing = _bucket(task_id, "hard") < 15
    if gold_missing:
        return docs  # 语料里根本没有答案 → 归因 skill.missing / capability_limit

    # 语料的**静态顺序**只用于归因诊断（`corpus()`，不进模型输入）。
    # 实际排序一律由 `search()` 的内容打分决定 —— 这里的位置不代表模型看到的 rank。
    #
    # （M2 曾在这里把 gold 手工埋到 rank 3，M3 用 is_gold 置顶召回它 —— 两者都是
    #  人工/答案键制造的 headroom。A1 之后 headroom 由内容打分自然产生：
    #  默认 q_topical 会把"复述题干"的干扰项排到 gold 前面，只有命中 Skill 的
    #  conclusion 信号才能把真正给出结论的段落提到首位。）
    base: List[Dict[str, Any]] = []
    distractors = [d for d in docs if d["doc_id"].startswith("cand::")]
    fillers = [d for d in docs if d["doc_id"].startswith("filler::")]
    if fillers:
        base.append(fillers[0])
    if distractors:
        base.append(distractors[0])
    if len(fillers) > 1:
        base.append(fillers[1])
    base.append(gold_doc)
    if len(distractors) > 1:
        base.append(distractors[1])
    if len(fillers) > 2:
        base.append(fillers[2])
    return base


def _tokens(text: str) -> set:
    STOP = {"the", "and", "for", "that", "with", "this", "from", "have", "has", "been", "are",
            "was", "were", "which", "what", "when", "into", "using", "used", "may", "can",
            "should", "following", "these", "those", "than", "then", "you", "your", "their",
            "not", "but", "all", "any", "how", "why", "who", "its", "of", "in", "on", "to",
            "as", "at", "by", "or", "is", "be", "if", "a", "an"}
    return {t for t in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", (text or "").lower())
            if t not in STOP}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().rstrip(".").lower())


def _stated_count(options: Sequence[str], text: str) -> int:
    """段落**字面陈述**了几个候选选项（归一化后的子串匹配）。

    这是"答案承载段落"的信号：多选题检索里，一个段落若直接写出了某个候选结论，
    它对答题的价值远高于只复述题干背景的段落。真实 RAG 的重排正是找这类段落。
    """
    hay = _norm(text)
    return sum(1 for o in options if len(_norm(o)) >= 8 and _norm(o) in hay)


# 结论承载的判定门槛：段落覆盖了选项**多少比例的实词**才算"给出了这个结论"。
# 用 token 级比例而不是严格子串，是因为严格子串受版式影响太大
# （"Reported finding: X." 里 X 后面带句号、缩写、上下标时子串匹配会漏），
# 实测会让召回信号在 ~24% 的题上失效，反而被更弱的 rerank 信号压过去。
CONCLUSION_COVERAGE = 0.8


def _conclusion_count(options: Sequence[str], text: str) -> int:
    """段落**实质给出了**几个候选结论（token 级覆盖度 ≥ CONCLUSION_COVERAGE）。"""
    dtoks = _tokens(text)
    n = 0
    for o in options:
        otoks = _tokens(o)
        if not otoks:
            continue
        if len(dtoks & otoks) / len(otoks) >= CONCLUSION_COVERAGE:
            n += 1
    return n


def carries_conclusion(options: Sequence[str], text: str) -> bool:
    """【N1】公开判据：这段文本**是否承载了某个候选结论**（不读答案键）。

    为什么需要它
        归因器原来用 `gold_in_corpus` / `gold_in_topk` 判定"这次失败该改哪个工件"，
        而这两个字段直接读语料里的答案标记 —— 真实自进化系统的反思模块只能看见
        trace（召回没召回、引了哪些文档、模型输出了什么），看不见 `is_gold`。
        归因器读答案键 ⇒ 它其实是个 oracle router，"归因驱动进化"不成立。

        真实系统能做的最接近的自我诊断是：**"我捞回来的段落里，有没有任何一段
        实质给出了某个候选结论？"** 这正是本函数回答的问题，全程只看内容。

    与 `gold_in_topk` 的差别（重要，别当成等价替换）
        `gold_in_topk` 问"正确答案那段进没进 top_k"（需要答案键）；
        本函数问"top_k 里有没有像样的结论句"（不需要）。
        干扰项也可能承载结论（hard negative 就是这么设计的），
        所以本判据**更宽松** —— 这是对的：系统分不清哪个结论是对的，
        它只能判断"证据捞没捞上来"。归因因此会更倾向于把失败归到推理层，
        这与真实无标签场景下的可观测性边界一致。
    """
    return _conclusion_count(options, text) > 0


class RetrievalGateway:
    """统一检索入口：离线夹具 /（M2）paper-qa。"""

    def __init__(self, backend: str = "offline-fixture", top_k: int = 3,
                 corpus_version: str = DEFAULT_CORPUS_VERSION,
                 doc_id_scheme: str = DEFAULT_DOC_ID_SCHEME,
                 chunking: Optional[Dict[str, Any]] = None):
        self.backend = backend
        self.top_k = top_k
        self.corpus_version = corpus_version
        self.doc_id_scheme = doc_id_scheme
        # 【P8】证据分块配置（None / enabled=false = 旧的单窗口行为）
        self.chunking = chunking

    def corpus(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """完整本地语料（供归因诊断；不会作为模型输入）。"""
        if self.backend != "offline-fixture":
            raise RetrievalError(f"检索后端 '{self.backend}' 未接入。")
        return build_fixture_corpus(task, self.corpus_version, self.doc_id_scheme,
                                    self.chunking)

    def evidence_signal(self, task: Dict[str, Any], text: str) -> bool:
        """Content-only diagnostic for attribution; fixture behavior is unchanged."""
        return carries_conclusion(list(task.get("options", [])), text)

    def search(self, task: Dict[str, Any], skill_recalled: bool = False,
               rerank: bool = False, query: Optional[str] = None,
               corpus: Optional[Sequence[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """检索 + 可选重排。**排序不得读取 `is_gold`**（Spec 红线 / A1 修复）。

        【A1 之前】`lead = [d for d in docs if d.get("is_gold")]` —— 领域过滤器直接读答案键，
        于是"改 description → 召回 → 答对"是一条**同义反复**的因果链。现在改为三段内容信号：

        | 信号 | 谁触发 | 语义 | 权重 |
        |---|---|---|---|
        | `q_topical` | 默认 | 段落与 question 的实词重合 | 1 |
        | `conclusion` | `skill_recalled`（description 维度） | 段落字面陈述了某个候选结论 | 100 |
        | `opt_overlap` | `rerank`（prompt/正文 维度） | 段落与 options 的实词重合 | 3 |

        三点必须说清：
        1. **不看答案键**：三个信号都只由段落文本 ×（question, options）算出。
           实测 gold@rank0 仍有真实 headroom（litqa2 52.7%→78.5%、protocolqa 33.3%→85.2%），
           说明这条机制**本身是真的**，原来的 oracle 只是把它短路了。
        2. **串联而非覆盖**：recall 的结论信号权重远大于 rerank 的词面信号，
           因此命中后 rerank 只在同档内微调，不会把召回置顶的段落顶掉（M3.1 的教训）。
        3. **可以被打败**：`trap` / `hardneg` 题里干扰项同样字面陈述一个错误结论，
           召回与证据策略在这批题上会真的失手 —— 策略从此有失败模式。
        """
        docs = [dict(d) for d in (corpus if corpus is not None else self.corpus(task))]
        opts = list(task.get("options", []))
        qtok = _tokens(task.get("question", "") if query is None else query)
        opttok: set = set()
        for o in opts:
            opttok |= _tokens(o)

        def score(d: Dict[str, Any]) -> int:
            text = d.get("text", "")
            s = len(_tokens(text) & qtok)
            if skill_recalled:
                s += CONCLUSION_WEIGHT * _conclusion_count(opts, text)
            if rerank:
                s += OPTION_OVERLAP_WEIGHT * len(_tokens(text) & opttok)
            return s

        # 稳定排序：同分时按 doc_id 的 hash 定序，任何机器重放一致
        docs = sorted(docs, key=lambda d: (-score(d), sha256_text(d.get("doc_id", ""))))
        for i, d in enumerate(docs[: self.top_k]):
            d["rank"] = i
        return docs[: self.top_k]


class PaperQAGateway(RetrievalGateway):
    """Retrieve supplied native PaperQA Doc/Text records without agent/embedding calls."""

    def __init__(self, documents, top_k=3):
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("document retrieval top_k must be 1 through 20")
        try:
            from importlib.metadata import version
            from paperqa.types import Doc, Text
        except ImportError as exc:
            raise DependencyMissing("install requirements-paperqa.txt for the native PaperQA adapter") from exc
        if version("paper-qa") != "2026.8.12":
            raise DependencyMissing("PaperQA adapter requires the pinned requirements-paperqa.txt version")
        super().__init__(backend="paperqa", top_k=top_k)
        self.records, self.passages = [], []
        seen = set()
        for source in documents:
            if source["source_id"] in seen or not source["text"].strip():
                raise ValueError("document identities must be unique and nonempty")
            seen.add(source["source_id"])
            digest = sha256_text(source["text"])
            doc = Doc(docname=source["source_id"], dockey=digest,
                      citation=source.get("url") or source["source_id"], content_hash=digest)
            for index, text in enumerate(chunking.chunk_text(source["text"])):
                identity = "pqa-" + sha256_text(source["source_id"] + f":{index}:" + text)[:24]
                record = Text(text=text, name=identity, doc=doc)
                self.records.append(record)
                self.passages.append({"doc_id": identity, "text": record.text,
                    "url": source.get("url", ""), "source_kind": "paperqa_document",
                    "content_sha256": sha256_text(record.text),
                    "locator": source.get("locator", "supplied document") + f"; chunk {index + 1}",
                    "retrieved_at": source.get("retrieved_at", "")})

    def corpus(self, task):
        return [dict(doc) for doc in self.passages]


def build_paperqa_gateway(top_k: int = 3, documents=None) -> "RetrievalGateway":
    return PaperQAGateway(documents or [], top_k)


def citation_of(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_id": doc.get("doc_id", ""),
        "url": doc.get("url", "") or doc.get("source", ""),
        "locator": doc.get("title", ""),
        "retrieved_at": "",
        "content_sha256": hashlib.sha256(doc.get("text", "").encode("utf-8")).hexdigest()[:16],
    }
