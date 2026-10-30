"""选项口径：注入官方的"信息不足"选项（R2）。

为什么
    LAB-Bench 系（LitQA2 / ProtocolQA / SuppQA）的官方评测里，每道题的选项集
    都额外注入一个 "Insufficient information to answer the question"
    （asta-bench `astabench/evals/labbench/litqa2/task.py:60-93`，
    原注释直接指向 LAB-Bench 官方实现）。本项目的归一化数据把它**丢掉了**，
    于是：
      1. 与官方数字不可比 —— 官方分母里有一部分是"主动弃权"，我们没有；
      2. 更糟的是"没证据就别猜"这个能力**在本项目里没有出口**：
         模型要么硬选一个错的，要么输出文本弃权（真实模型 506 次里只有 4 次）。
         有了这个选项，弃权变成可选项，coverage × precision 才有了意义。

    官方口径（对齐 `litqa2/task.py:114-139`）：
      - 选了它 → accuracy 记错、coverage 记弃权、precision 的分母排除它；
      - 它的位置是**随机**的（不是永远挂最后一个，否则模型会学位置偏见）。
        本项目用 task_id 的稳定 hash 决定位置，保证可重放。

开关
    `evaluation.insufficient_option`（布尔）。**2026-09-26 起默认为 true** ——
    官方口径本就带这个选项；打开会改变选项数量与协议指纹，因此历史基线
    （用 `configs/experiment.legacy-v2.yaml` 复现）不可与新结果直接比较。

    实测影响（真实模型、real-v3 语料、n=30/集）：注入后 litqa2 的
    accuracy 78.63 → 63.33、coverage 99.15 → 70，但 **precision 79.31 → 90.48** ——
    单看 accuracy 会以为变差，实际是"没证据时不再硬猜"这件事第一次有了出口。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..util import sha256_text

INSUFFICIENT_OPTION = "Insufficient information to answer the question"
OPTION_RULE_VERSION = "official-insufficient-v1"

# 默认开关（2026-09-26 升级为"默认注入"，对齐官方口径）。
# 关掉它的唯一正当理由：用 `configs/experiment.legacy-v2.yaml` 复现历史数字。
INSUFFICIENT_OPTION_DEFAULT = True


def is_insufficient(text: Any) -> bool:
    """该选项是否为"信息不足"选项（容忍大小写与前后空白）。"""
    return str(text or "").strip().lower().startswith("insufficient information")


def explicit_information_gap_index(options: List[str]) -> Optional[int]:
    """Return the index of a task's labeled information-gap class, if present.

    A labeled class is a normal answer in three-way claim-verification tasks;
    it must not be confused with the separate runtime abstention option used by
    benchmarks such as LAB-Bench. Supported canonical labels include SciFact's
    NOT_ENOUGH_INFO and CliniFact's NEI.
    """
    for index, option in enumerate(options):
        normalized = re.sub(r"[^a-z0-9]+", "_", str(option).strip().lower()).strip("_")
        if normalized in {"not_enough_info", "not_enough_information", "nei"}:
            return index
    return None


def insufficient_position(task_id: str, n_options: int, dataset: str = "") -> int:
    """注入位置：由 task_id 稳定决定，任何机器重放一致。"""
    if n_options <= 0:
        return 0
    return int(sha256_text(f"insufficient|{dataset}|{task_id}")[:8], 16) % (n_options + 1)


def inject_insufficient(task: Dict[str, Any]) -> Dict[str, Any]:
    """返回注入后的新 task（不修改原对象）。

    `correct_index` 会随插入位置右移；`insufficient_index` 记录它的落点，
    供归因与报告使用。已经注入过的题不会重复注入。
    """
    opts: List[str] = list(task.get("options", []))
    if any(is_insufficient(o) for o in opts):
        return dict(task)
    ci = int(task.get("correct_index", 0))
    pos = insufficient_position(task.get("task_id", ""), len(opts),
                                task.get("dataset", ""))
    new_opts = opts[:pos] + [INSUFFICIENT_OPTION] + opts[pos:]
    new_ci = ci + 1 if ci >= pos else ci
    out = dict(task)
    out["options"] = new_opts
    out["correct_index"] = new_ci
    out["insufficient_index"] = pos
    out["option_rule_version"] = OPTION_RULE_VERSION
    return out


def inject_all(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [inject_insufficient(t) for t in tasks]


def resolve_abstain(answer_index: Optional[int], options: List[str]) -> int:
    """把"选了 Insufficient 选项"归一化为弃权（-1），对齐官方计分口径。

    放在这里而不是 provider 里，是因为 fake / http 两条通道都要遵守同一口径。
    """
    if answer_index is None:
        return -1
    i = int(answer_index)
    if i >= 0 and i < len(options) and is_insufficient(options[i]):
        return -1
    return i
