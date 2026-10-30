"""GEPA 式**整段 prompt 改写**（2026-09-27）。

## 为什么需要这个模块

`reflection.py` 的两个 prompt 提案路径（模板 / 反思后端）最终都是
`new = old.rstrip() + "\\n" + directive` —— **追加一句**。
这意味着候选空间只有"加哪句触发词"，量级约 2^4。

对照 vendor：
- **GEPA**（`vendor/gepa-main/src/gepa/strategies/instruction_proposal.py`）
  输出的是**一整条新的 instruction**，且提示词显式要求
  "if the assistant may have utilized a generalizable strategy to solve the task,
  include that in the instruction as well"；
- **DSPy MIPROv2**（`vendor/dspy-main/dspy/teleprompt/mipro_optimizer_v2.py`）
  为**每个 predictor** 生成 N 条候选 instruction 的自由文本改写。

本模块把本项目的提案形态从"追加一句"升级为"整段改写"，并把
"写进可泛化的解题策略"作为**硬性要求**写进提案提示词。

## 为什么保留 patch 模式

`patch` 是历史所有 release 的产出形态，切换会让历史产物不可比。
故用配置 `evolution.proposer` 控制（默认 `patch`），`rewrite` 需显式开启。

## 校验为什么必须包含"输出契约"

本项目靠 `ANSWER:` / `CITED:` / `INSUFFICIENT` 解析模型输出。
若改写后的 prompt 丢掉这些关键词，评测会整体退化成"解析失败"，
而这不是 prompt 质量问题，是产出物不合格 —— 必须在提案阶段拦掉。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

REWRITE_SYSTEM = (
    "You are an expert at writing instructions for a scientific question-answering "
    "assistant. You will be given the assistant's current instructions and examples "
    "of cases where it failed, each with feedback on what went wrong.\n"
    "Your job is to write a NEW, COMPLETE set of instructions that fixes the "
    "recurring problems."
)

REWRITE_USER = """The assistant is currently given these instructions:
```
{current}
```

Here are {n} cases where the assistant failed, with feedback:
```
{cases}
```
{demos}
Write a NEW, COMPLETE version of the instructions (rewrite the whole thing; do NOT output a patch or a diff).

Requirements:
1. Identify the RECURRING failure pattern across the cases above. Do not write rules for one-off mistakes.
2. For each recurring pattern, write a specific, checkable rule the assistant can follow.
3. If there is a generalizable STRATEGY that would help — for example, how to compare each option against the evidence before deciding, or how to decide when the evidence is insufficient — SPELL IT OUT as concrete steps inside the instructions.
4. PRESERVE everything in the current instructions that is already working.
5. You MUST preserve the output contract: the assistant must still reply with an `ANSWER:` line (and the `CITED:` / `INSUFFICIENT` behaviour if the current instructions require it).
6. Be concise. Do NOT pad with generic advice like "be careful", "think step by step", or "be accurate" without concrete content. Generic advice is worse than the current instructions.
7. DO NOT trade precision for coverage. Telling the assistant to "try to answer" / "attempt an answer" without also telling it HOW to decide that the evidence is sufficient will raise coverage and LOWER precision — that is a REGRESSION, not an improvement. Any rule that makes the assistant answer more often MUST be paired with an explicit sufficiency test.
8. If the current instructions have a way to signal "insufficient evidence" (e.g. `INSUFFICIENT`), keep using that EXACT wording. Do NOT invent alternatives such as "None", "N/A", "unknown", or free-form explanations — a different wording breaks the grader.

Output ONLY the new instructions inside a single ``` fenced block, and nothing else.
"""


@dataclass
class RewriteResult:
    """整段改写的产出。"""
    prompt: str = ""
    rationale: str = ""
    raw: str = ""
    rejected: Optional[str] = None
    attempts: List[Dict[str, Any]] = field(default_factory=list)


# --- 提取 -----------------------------------------------------------------

_FENCE = re.compile(r"```(?:[a-zA-Z]*)\s*\n(.*?)```", re.S)


def extract_rewritten_prompt(raw: str) -> str:
    """从模型输出里取出 ``` 围栏内的正文。

    没有围栏时退回整段文本 —— 有些模型不按格式输出，
    但只要内容是有效指令就仍然可用（后续校验会拦掉真正不合格的）。
    """
    m = _FENCE.search(raw or "")
    if m:
        return (m.group(1) or "").strip()
    return (raw or "").strip()


# --- 校验 -----------------------------------------------------------------

# 输出契约关键词：丢了它们，评测解析会整体失败（不是质量问题，是不合格）
CONTRACT_MARKERS = ("ANSWER",)

# 弃权表述的替代词（带引号出现时）。实测教训：某候选写了 respond with "None"
# 来替代 `INSUFFICIENT`，评测口径被破坏 —— 它的 coverage 因此 −2.5pp。
# 只匹配**带引号**的写法，避免把普通行文里的 "none" 误伤掉。
_QUOTED_ALT_ABSTAIN = re.compile(r"""["']\s*(none|n/a|unknown|no answer)\s*["']""",
                                 re.I)


def validate_rewrite(new: str, old: str, cfg: Optional[Dict[str, Any]] = None
                     ) -> Optional[str]:
    """返回 None 表示通过；否则返回被拒原因（写进候选留痕）。"""
    cfg = cfg or {}
    if not new or not new.strip():
        return "改写结果为空"
    if len(new.strip()) < 40:
        return f"改写结果过短（{len(new.strip())} 字符），不像完整指令"

    gate = dict(cfg.get("gates", {}).get("static", {}) or {})
    max_chars = int(gate.get("max_prompt_chars", 8000) or 8000)
    if len(new) > max_chars:
        return f"改写结果超过 max_prompt_chars（{len(new)} > {max_chars}）"

    up = new.upper()
    for marker in CONTRACT_MARKERS:
        if marker not in up:
            return f"改写结果丢失输出契约关键词 '{marker}'"

    # 弃权表述：若用带引号的 "None"/"N/A" 替代了 INSUFFICIENT，评测口径会坏掉。
    # 只在新 prompt 完全没有 INSUFFICIENT 语义时才判 —— 两者并存不算问题。
    if "insufficient" not in new.lower():
        m = _QUOTED_ALT_ABSTAIN.search(new)
        if m:
            return (f"疑似用 '{m.group(1)}' 替换了 INSUFFICIENT 弃权表述"
                    f"（会破坏评测口径；实测该写法使 coverage −2.5pp）")

    # 零信息量：与旧 prompt 几乎相同 → 等于没改
    a = re.sub(r"\s+", " ", new.strip().lower())
    b = re.sub(r"\s+", " ", (old or "").strip().lower())
    if a == b:
        return "改写结果与当前 prompt 实质相同（零信息量）"
    return None


# --- 主入口 ---------------------------------------------------------------

def _format_cases(failure_rows: Optional[Sequence[Dict[str, Any]]], n: int = 4
                  ) -> str:
    rows = list(failure_rows or [])[:n]
    if not rows:
        return "(no failure cases provided)"
    # Harness failure rows intentionally contain IDs and scores, not question
    # text. Resolve only the visible training split here; otherwise the model
    # receives empty Question/Options fields and can only invent generic rules.
    from .reflection import _task_index

    tasks = _task_index([str(r.get("dataset", "")) for r in rows if r.get("dataset")])
    parts: List[str] = []
    for i, r in enumerate(rows, 1):
        task = tasks.get(str(r.get("task_id", ""))) or {}
        q = str(r.get("question") or task.get("question") or "")[:300]
        opts = r.get("options") or task.get("options") or []
        opts_s = "; ".join(str(o)[:80] for o in opts[:6])
        material = str(r.get("evidence") or task.get("evidence")
                       or task.get("context") or "")[:500]
        gold = r.get("correct_index", r.get("gold_index", "?"))
        got = r.get("answer_index", "?")
        fb = str(r.get("feedback", "") or r.get("attributed_class", "") or "")[:200]
        parts.append(
            f"[{i}] Question: {q or '(unavailable)'}\n"
            f"    Options: {opts_s or '(unavailable)'}\n"
            f"    Task-provided material: {material or '(none)'}\n"
            f"    Correct option: {gold}\n    Assistant answered: {got}\n"
            f"    Feedback: {fb or '(none)'}")
    return "\n\n".join(parts)


def propose_rewrite(provider: Any, cfg: Dict[str, Any], cls: str,
                    current_prompt: str,
                    failure_rows: Optional[Sequence[Dict[str, Any]]] = None,
                    demos_block: str = "",
                    max_attempts: int = 2) -> RewriteResult:
    """调用反思模型产出整条新 prompt。

    与 `reflection_backend.LlmBackend.propose` 的差别：
    - 产出是**整条 prompt**，不是追加补丁；
    - 提示词硬性要求写入"可泛化的解题策略"；
    - 校验包含输出契约与"零信息量"检查。
    """
    res = RewriteResult()
    n = min(len(failure_rows or []), 4)
    user = REWRITE_USER.format(
        current=(current_prompt or "")[:3000],
        n=n,
        cases=_format_cases(failure_rows, n),
        demos=(("\nHere are examples the assistant answered CORRECTLY, for reference:\n```\n"
                + demos_block + "\n```\n") if demos_block else ""),
    )
    for i in range(max(1, int(max_attempts))):
        try:
            if hasattr(provider, "chat"):
                raw = provider.chat(REWRITE_SYSTEM, user)
            else:  # pragma: no cover - 兼容无 chat 的 provider
                out = provider.complete(system_prompt=REWRITE_SYSTEM, question=user,
                                        options=[], docs=[],
                                        task_id=f"rewrite:{cls}")
                raw = getattr(out, "raw", "") or getattr(out, "answer_text", "") or ""
        except Exception as exc:
            res.attempts.append({"attempt": i + 1, "error": str(exc)[:200]})
            continue
        text = extract_rewritten_prompt(str(raw))
        reason = validate_rewrite(text, current_prompt, cfg)
        res.attempts.append({"attempt": i + 1, "raw": str(raw)[:300],
                             "parsed_head": text[:200], "rejected": reason})
        if reason is None:
            res.prompt = text
            res.raw = str(raw)
            res.rationale = (
                f"归因 {cls}：GEPA 式整段改写（参考 {n} 条失败样本"
                + ("，含 few-shot 正确样例" if demos_block else "")
                + "）")
            return res
    res.rejected = "整段改写未产出合格候选（见 attempts）"
    return res
