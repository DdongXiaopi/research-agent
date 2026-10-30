"""Few-shot 示例引导（bootstrap demos）—— MIPROv2 的核心维度（2026-09-27）。

## 为什么需要

对照 `vendor/dspy-main`：`MIPROv2` 的候选空间是
**instruction 变体 × few-shot 示例组合** 的笛卡尔积。
本项目此前**完全没有 few-shot 维度** —— 候选只在"改哪句指令"上搜索，
而"给模型看几个答对的示范"是另一个完全没被探索的轴。

## 做法（dspy BootstrapFewShot 的思路，按本项目约束简化）

1. 在 **dev_optimize** 上跑当前 release，拿到逐题结果；
2. 挑出**答对**的题作为 demo 池（答错的不能当示范）；
3. 候选 prompt 里嵌入 k 条 demo（question / options / evidence / 正确选项）。

## 数据隔离

demos 只从 `dev_optimize` 引导 —— 那是**进化可见区**。
`final_holdout` 由 `src/data/split.py` 物理封存，即便本模块显式
`allow_final=True` 也会被 `FinalHoldoutSealed` 拦下，所以这里**不可能**误取封存题。

## 代价

demo 会占用 prompt 长度（`gates.static.max_prompt_chars`），
所以 k 必须小（默认 3），并做字符预算裁剪。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_K = 3
DEFAULT_MAX_DEMO_CHARS = 400       # 单条 demo 的证据/题干裁剪上限
DEFAULT_MAX_BLOCK_CHARS = 2000     # 整块 few-shot 的字符预算
BUNDLE_VERSION = 1


def serialize_demos_bundle(demos: Sequence[Dict[str, Any]]) -> str:
    """Serialize a reproducible task-demo artifact with source IDs and labels."""
    normalized = []
    for item in demos:
        task = dict(item.get("task") or {})
        task_id = str(item.get("task_id") or task.get("task_id") or "")
        correct_index = int(item.get("correct_index", task.get("correct_index", -1)))
        if task_id != str(task.get("task_id") or ""):
            raise ValueError("demo task_id must match the embedded task")
        if correct_index != int(task.get("correct_index", -2)):
            raise ValueError("demo correct_index must match the embedded task")
        normalized.append({"task_id": task_id, "correct_index": correct_index,
                           "task": task})
    return json.dumps({"version": BUNDLE_VERSION, "demos": normalized},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_demos_bundle(content: str) -> List[Dict[str, Any]]:
    """Parse and validate a task-demo artifact without loading benchmark data."""
    try:
        bundle = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid task-demo JSON: {exc}") from exc
    if not isinstance(bundle, dict) or bundle.get("version") != BUNDLE_VERSION:
        raise ValueError("unsupported task-demo bundle version")
    demos = bundle.get("demos")
    if not isinstance(demos, list) or len(demos) > 4:
        raise ValueError("task-demo bundle must contain at most four demos")
    seen = set()
    for item in demos:
        if not isinstance(item, dict) or not isinstance(item.get("task"), dict):
            raise ValueError("invalid task-demo entry")
        task = item["task"]
        task_id = str(item.get("task_id") or "")
        if (not task_id or task_id in seen or task_id != str(task.get("task_id") or "")
                or task.get("split") != "dev_optimize"):
            raise ValueError("task-demo IDs must be unique visible dev_optimize tasks")
        seen.add(task_id)
        correct = int(item.get("correct_index", -1))
        if correct != int(task.get("correct_index", -2)):
            raise ValueError("task-demo answer index mismatch")
        options = task.get("options") or []
        if not options or not 0 <= correct < len(options):
            raise ValueError("task-demo answer index is outside its options")
        if not str(task.get("question") or "").strip():
            raise ValueError("task-demo question is empty")
    return demos


def _fmt_option(i: int, o: Any) -> str:
    return f"{i}. {str(o)}"


def render_demo(task: Dict[str, Any], correct_index: Any,
                max_chars: int = DEFAULT_MAX_DEMO_CHARS) -> str:
    """把一道题渲染成一条 few-shot 示范。"""
    ev = str(task.get("evidence", "") or task.get("context", "") or "")
    if len(ev) > max_chars:
        half = max(1, (max_chars - 18) // 2)
        ev = ev[:half] + "\n[... omitted ...]\n" + ev[-half:]
    q = str(task.get("question", "") or "")
    opts = task.get("options") or []
    lines = []
    if ev:
        lines.append("Evidence: " + ev)
    lines.append("Question: " + q)
    lines.append("Options:")
    lines.extend(_fmt_option(i, o) for i, o in enumerate(opts))
    lines.append(f"ANSWER: {correct_index}")
    return "\n".join(lines)


def render_demos_block(demos: Sequence[Dict[str, Any]],
                       max_block_chars: int = DEFAULT_MAX_BLOCK_CHARS,
                       max_demo_chars: int = DEFAULT_MAX_DEMO_CHARS) -> str:
    """渲染可嵌入 prompt 的 few-shot 块；超预算时保留前缀示例。"""
    block, _ = render_demos_block_with_ids(
        demos, max_block_chars=max_block_chars, max_demo_chars=max_demo_chars)
    return block


def render_demos_block_with_ids(
        demos: Sequence[Dict[str, Any]],
        max_block_chars: int = DEFAULT_MAX_BLOCK_CHARS,
        max_demo_chars: int = DEFAULT_MAX_DEMO_CHARS) -> tuple[str, List[str]]:
    """Render demos and return the exact train IDs that fit in the block budget."""
    if not demos:
        return "", []
    prefix = ("Below are examples of questions this assistant answered correctly. "
              "Follow the same standard.\n\n")
    parts: List[str] = []
    rendered_ids: List[str] = []
    used = len(prefix)
    for d in demos:
        s = render_demo(d.get("task") or {}, d.get("correct_index"),
                        max_chars=max_demo_chars)
        addition = len(s) + (2 if parts else 0)
        if used + addition > max_block_chars:
            break
        parts.append(s)
        rendered_ids.append(str(d.get("task_id") or
                                (d.get("task") or {}).get("task_id") or ""))
        used += addition
    if not parts:
        return "", []
    return prefix + "\n\n".join(parts), rendered_ids


def bootstrap_demos(cfg: Dict[str, Any], state: Any, dataset: str,
                    split: str = "dev_optimize",
                    k: int = DEFAULT_K, seed: int = 20260924,
                    limit: Optional[int] = None,
                    run_tag: str = "fewshot-bootstrap") -> List[Dict[str, Any]]:
    """跑一次当前 release，挑答对的题作为 demo 池。

    返回 `[{task_id, task, correct_index}]`；失败（跑不通 / 无答对题）返回空列表 ——
    **静默降级为空 block**，不阻塞进化。
    """
    if k <= 0:
        return []
    try:
        from ..eval.harness import evaluate
        from ..data.split import load_tasks
        from ..eval.options import inject_all
    except Exception:                                   # pragma: no cover
        return []

    # ⚠️ 不要先用 limit 截断 tasks 再建索引：`load_tasks` 的顺序与 `evaluate`
    # 实际跑的题**不是同一批**（实测：load_tasks[:30] 首题 21645374，而
    # evaluate 的 per_task 首题 10135926）。截断会让 task_id 全部落空 ⇒ demos=0。
    # limit 只决定**评测跑多少题**，索引始终用完整题表。
    try:
        tasks = load_tasks(dataset, [split])
        if cfg.get("evaluation", {}).get("insufficient_option", True):
            tasks = inject_all(tasks)
    except Exception:
        return []
    if not tasks:
        return []

    try:
        res = evaluate(cfg, state, dataset, split, seed=seed, limit=limit,
                       run_tag=run_tag, write_per_task=True,
                       cfg_fp="fewshot-bootstrap")
    except Exception:
        return []

    per_task = list(getattr(res, "per_task", None) or [])
    by_id = {str(t.get("task_id")): t for t in tasks}
    demos: List[Dict[str, Any]] = []
    for row in per_task:
        if not row.get("correct"):
            continue
        tid = str(row.get("task_id"))
        task = by_id.get(tid)
        if not task:
            continue
        if int(row.get("gold_index", -1)) != int(task.get("correct_index", -2)):
            # A demonstration with mismatched option order teaches the wrong
            # answer. Fail closed on any split or option-protocol mismatch.
            continue
        demos.append({
            "task_id": tid,
            "task": task,
            "correct_index": task["correct_index"],
        })
        if len(demos) >= int(k):
            break
    return demos
