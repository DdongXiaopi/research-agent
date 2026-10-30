"""【P8】证据分块 + 相关性筛选（paper-qa 范式）。

## 为什么要有这个模块

N4 只修好了 `context` 侧（阈值 + 边界截断）。**检索到的 passage 侧**仍是
"一条 passage = 一个 1400 字符的滑动窗口，取与题目词面重叠最高的那一个"
（`corpus.best_window`）。这个做法有两个结构性缺陷：

1. **只取一块**：长 protocol 里答案常横跨两个不相邻的步骤（如"第 2 步配错了
   浓度 → 第 8 步离心力不够"），单个连续窗口拿不全；
2. **窗口是连续区间**：高相关区域被低相关区域隔开时，为了把它们圈进同一个
   窗口，只能把中间那堆无关文字一起带上 —— 相关性被稀释。

paper-qa 的做法（`src/paperqa/docs.py`）正是针对这两点：
**文档→滑窗分块→逐块 relevance score→只把高分块拼进上下文**。

## 预算中性（本模块的第一原则）

默认 `budget_chars` = 现状的 `max_doc_chars`（1500），即**字符总量不变，只改变
这 1500 个字符选哪里**。这样：

- 效应来自"选得对不对"，不是"给得多不多" —— 单一变量；
- token 成本基本不变，不会被成本门禁误伤；
- **短证据自动 no-op**：文本本身 ≤ 预算时原样返回（逐字符不变），
  于是 litqa2（evidence p50 292）与 pubmedqa（p50 1352）天然成为**结构性
  阴性对照** —— 分块对它们不存在，它们若变化就说明实现有 bug。

## 不读答案键

打分只用 chunk 文本 × (question, options) 的实词重合，与
`gateway.search()` 的打分口径同源（q_topical + OPTION_WEIGHT × opt_overlap）。
`is_gold` 一律不参与（Spec 红线 / A1）。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set

# 【N4 之后的第二个证据呈现判据版本】进协议指纹（仅在启用时拼接，见
# `src/eval/final_eval.py`）。改切块/筛选规则必须递增：
#   none-v0      = 旧行为（`best_window` 单窗口），分块未启用
#   step-aware-v1= 按步骤/行/句边界切块 + 预算内取高分块（本模块）
CHUNK_RULE_VERSION = "render-budget-v2"
CHUNK_RULE_NONE = "none-v0"

# 默认切块参数
DEFAULT_TARGET_CHARS = 700      # 一块的目标长度（软目标）
DEFAULT_MAX_CHARS = 1000        # 一块的硬上限（超过即强制收口）
DEFAULT_MAX_CHUNKS = 4          # 一条 passage 最多保留几块
DEFAULT_OPTION_WEIGHT = 3       # 选项词重合权重（与 gateway.OPTION_OVERLAP_WEIGHT 同源）

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;。；])\s+")

# 行首看着像"结构标记"的模式：Step N / Part N / Day N / 1. / a) / ### / - 。
# 命中即优先作为切块边界（protocolqa 的 protocol 是一行一步）。
_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"step\s*\d+|part\s*\d+|day\s*\d+|section\s*\d+|"
    r"\d+[.)]\s|[a-z][.)]\s|#{1,6}\s|[-*]\s"
    r")",
    re.IGNORECASE,
)


def _sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT_RE.split((text or "").replace("\n", " ").strip())
    return [p.strip() for p in parts if p.strip()]


def atomic_blocks(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> List[str]:
    """把文本切成**不可再切的原子块**：优先按行（protocol 一行一步），
    过长的行再按句切。绝不从单词中间切。
    """
    if max_chars < 1:
        raise ValueError("chunk max_chars must be positive")
    raw = [b.strip() for b in re.split(r"\n+", (text or "")) if b.strip()]
    out: List[str] = []
    for b in raw:
        if len(b) <= max_chars:
            out.append(b)
            continue
        cur = ""
        for s in _sentences(b):
            if len(s) > max_chars:
                if cur:
                    out.append(cur)
                    cur = ""
                # Bound an unbroken sentence/token as well. Small overlap retains
                # nearby context; offsets are not represented as exact quotations.
                step = max(1, max_chars - min(80, max_chars // 5))
                out.extend(s[start:start + max_chars]
                           for start in range(0, len(s), step))
                continue
            if cur and len(cur) + 1 + len(s) > max_chars:
                out.append(cur)
                cur = s
            else:
                cur = (cur + " " + s).strip()
        if cur:
            out.append(cur)
    return out


def chunk_text(text: str, target_chars: int = DEFAULT_TARGET_CHARS,
               max_chars: int = DEFAULT_MAX_CHARS) -> List[str]:
    """把文本切成块：贪心累积原子块到 `target_chars`，硬顶 `max_chars`。

    保证：
      - 任一块都不跨过"行/句"边界（`atomic_blocks` 的性质）；
      - 输入短于 `target_chars` 时返回**单块**（→ 上层预算内原样返回，no-op）。
    """
    blocks = atomic_blocks(text, max_chars)
    if not blocks:
        return []
    total = sum(len(b) for b in blocks) + max(0, len(blocks) - 1)
    if total <= target_chars:
        return ["\n".join(blocks)]

    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for b in blocks:
        add = len(b) + (1 if cur else 0)
        if cur and cur_len + add > target_chars:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
            add = len(b)
        cur.append(b)
        cur_len += add
        if cur_len >= max_chars:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def chunk_score(chunk: str, q_tokens: Set[str], opt_tokens: Set[str],
                option_weight: int = DEFAULT_OPTION_WEIGHT) -> int:
    """块与"问题 + 选项"的词面相关度（内容派生，不读答案键）。

    口径与 `gateway.search()` 的重排信号同源：题干重合 ×1 + 选项重合 ×
    `option_weight`。选项权重更高，因为"提到候选结论"的块比"复述题干"的块
    对答题更有价值 —— 这与 gateway 里 `opt_overlap` 的定位一致。
    """
    from .corpus import tokens

    ctoks = tokens(chunk)
    return len(ctoks & q_tokens) + option_weight * len(ctoks & opt_tokens)


def select_chunks(text: str, q_tokens: Set[str], opt_tokens: Set[str],
                  budget_chars: int = 1500, max_chunks: int = DEFAULT_MAX_CHUNKS,
                  target_chars: int = DEFAULT_TARGET_CHARS,
                  max_chars: int = DEFAULT_MAX_CHARS,
                  option_weight: int = DEFAULT_OPTION_WEIGHT,
                  exclude_top: int = 0) -> Dict[str, Any]:
    """按相关性筛选块，返回选中的块（保持原文顺序）。

    `exclude_top=k`：**故意**丢掉得分最高的 k 个块再选 —— 用来复现
    `gateway` 里 `trap` 桶的语义（"文档捞对了，但取到的部位不对"）。
    没有它，P8 会把这个失败模式一起修掉，收益里就混进了"少了一种失败模式"
    这个与分块无关的贡献（混淆变量）。
    """
    if budget_chars < 1 or max_chunks < 1 or target_chars < 1:
        raise ValueError("chunk budgets must be positive")
    content_budget = max(1, budget_chars - 96)
    chunks = chunk_text(text, min(target_chars, content_budget), min(max_chars, content_budget))
    scores = [chunk_score(c, q_tokens, opt_tokens, option_weight) for c in chunks]
    ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
    if exclude_top > 0:
        ranked = ranked[exclude_top:]

    keep: List[int] = []
    used = 0
    for i in ranked:
        if scores[i] <= 0:
            break                       # 零相关的块一律不要（这是"筛选"的实质）
        if len(keep) >= max_chunks:
            break
        if len(_render_chunks(chunks, sorted(keep + [i]))[0]) > budget_chars:
            continue                    # 这块装不下，看下一块（可能更小且同样相关）
        keep.append(i)
        used += len(chunks[i])
    keep.sort()

    dropped = [i for i in range(len(chunks)) if i not in set(keep)]
    return {
        "chunks": chunks,
        "scores": scores,
        "kept": keep,
        "dropped": dropped,
        "kept_chars": sum(len(chunks[i]) for i in keep),
        "dropped_chars": sum(len(chunks[i]) for i in dropped),
        "score_captured": sum(scores[i] for i in keep),
        "score_total": sum(scores),
    }


def _render_chunks(chunks: List[str], keep: List[int]) -> tuple[str, int]:
    parts, gap_marked = [], 0
    previous = None
    for index in keep:
        if previous is not None and index != previous + 1:
            gap = sum(len(chunks[j]) for j in range(previous + 1, index))
            gap_marked += gap
            parts.append(f"[… 省略 {gap} 字符（与本题相关性较低的部分）]")
        parts.append(chunks[index])
        previous = index
    if keep:
        tail = sum(len(chunk) for chunk in chunks[keep[-1] + 1:])
        if tail:
            parts.append(f"[… 另有 {tail} 字符未纳入（相关性较低）]")
    return "\n".join(parts), gap_marked


def assemble(text: str, q_tokens: Set[str], opt_tokens: Set[str],
             budget_chars: int = 1500, max_chunks: int = DEFAULT_MAX_CHUNKS,
             target_chars: int = DEFAULT_TARGET_CHARS,
             max_chars: int = DEFAULT_MAX_CHARS,
             option_weight: int = DEFAULT_OPTION_WEIGHT,
             exclude_top: int = 0) -> tuple:
    """分块 → 筛选 → 拼回文本。返回 `(文本, 统计)`。

    **预算内的短文本逐字符原样返回**（`passthrough`）—— 这是 litqa2 /
    pubmedqa 能当阴性对照的前提，也是"本改动只在长证据上生效"的保证。
    """
    t = (text or "").strip()
    if not t:
        return "", {"method": "empty", "n_chunks": 0}
    if len(t) <= budget_chars:
        return t, {"method": "passthrough", "n_chunks": 1,
                   "kept_chars": len(t), "dropped_chars": 0,
                   "score_captured": chunk_score(t, q_tokens, opt_tokens, option_weight),
                   "score_total": chunk_score(t, q_tokens, opt_tokens, option_weight)}

    sel = select_chunks(t, q_tokens, opt_tokens, budget_chars, max_chunks,
                        target_chars, max_chars, option_weight, exclude_top)
    chunks = sel["chunks"]
    keep = sel["kept"]
    if not keep:
        # 极端情况：所有块零相关。宁可给"预算内的开头若干块"也不给空串 ——
        # 空 passage 会让 rerank/citation/abstain 通道全部空转（N4 探针实测过）。
        head: List[str] = []
        used = 0
        for c in chunks:
            if used + len(c) + bool(head) > budget_chars:
                break
            head.append(c)
            used += len(c) + (1 if len(head) > 1 else 0)
        out_head = "\n".join(head) or t[:budget_chars]
        stats0 = dict(sel)
        stats0.update({"method": "fallback_head", "kept_chars": len(out_head),
                       "assembled_chars": len(out_head)})
        return out_head, stats0

    out, gap_marked = _render_chunks(chunks, keep)
    stats = dict(sel)
    stats["method"] = "chunked"
    stats["assembled_chars"] = len(out)
    stats["gap_marked_chars"] = gap_marked
    return out, stats


def chunking_enabled(cfg: Optional[Dict[str, Any]]) -> bool:
    """配置里是否启用了证据分块（默认关 —— 保持与已封存基线同一协议）。"""
    chunk = ((cfg or {}).get("retrieval") or {}).get("chunking") or {}
    if not isinstance(chunk, dict):
        return False
    return bool(chunk.get("enabled", False))


def config_of(cfg: Dict[str, Any], fallback_budget: int = 1500) -> Dict[str, Any]:
    """取出分块参数；`budget_chars` 缺省时回落到 `max_doc_chars`。

    为什么默认回落到 `max_doc_chars`：分块的产出还要被 provider 的
    `max_doc_chars` 二次截断。预算 > 该阈值 ⇒ 选出来又被切掉，等于白选且
    静默丢失。回落到它，两个阈值**由构造保证一致**，不会出现这种哑火。
    """
    c = dict(((cfg or {}).get("retrieval") or {}).get("chunking") or {})
    return {
        "enabled": bool(c.get("enabled", False)),
        "budget_chars": min(int(c.get("budget_chars") or fallback_budget), fallback_budget),
        "target_chars": int(c.get("target_chars") or DEFAULT_TARGET_CHARS),
        "max_chars": int(c.get("max_chars") or DEFAULT_MAX_CHARS),
        "max_chunks": int(c.get("max_chunks") or DEFAULT_MAX_CHUNKS),
        "option_weight": int(c.get("option_weight") or DEFAULT_OPTION_WEIGHT),
    }
