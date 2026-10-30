"""Pareto 多目标前沿（2026-09-27）。

## 为什么需要

单次改写 + "挑主判据最优"会选出**偏科生**。实测：候选 2 在 dev 上主判据 +5.24pp，
但 holdout 上 `coverage +3.68pp` 而 `answered_precision −2.33pp` —— 它把
"更敢答"和"答得更准"做成了取舍，净收益被抵消（McNemar p=0.79）。

GEPA 的关键设计之一正是：**返回一组在不同子目标上互补的 prompt（Pareto 前沿），
而不是单一最优**，避免过早收敛到局部最优。本模块实现这个。

## 目标向量的选择

三个目标，全部"越大越好"：

1. `answered_precision` —— 主判据（敢答时的正确率，不受敢不敢答影响）
2. `coverage` —— 防止靠"保守弃权"刷 precision
3. `-unsupported_answer_rate` —— 接地性（无支撑主张越少越好）

**刻意不纳入 accuracy**：accuracy = precision × coverage，是前两个目标的乘积，
不独立；纳入会让支配判定偏向"两头都不差"的平庸解，削弱前沿的多样性。

## 噪声与容差

严格支配在噪声下会产生大量"假支配"（Δ 小于噪声地板却没有实质差别），
于是前沿退化成只剩一个点。故支配判定带容差 `tol`（默认 1pp，约等于
selection_gate 的噪声地板 0.89pp）：只有当某目标**超出容差**才算真更好。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# 目标名（顺序固定），全部"越大越好"
OBJECTIVES: Tuple[str, ...] = ("answered_precision", "coverage",
                               "neg_unsupported_answer_rate")

# 默认容差（比例，非百分点）：1pp
DEFAULT_TOL = 0.01


def objectives_of(metrics: Optional[Dict[str, Any]]) -> Optional[List[float]]:
    """从一次评测的聚合指标里取出目标向量；缺字段返回 None（不进前沿）。"""
    if not metrics:
        return None
    prec = metrics.get("answered_precision")
    cov = metrics.get("coverage")
    uns = metrics.get("unsupported_answer_rate")
    # Missing grounding labels are unknown, not a perfect zero failure rate.
    # In particular, real PubMed articles do not carry fixture is_gold flags.
    if prec is None or cov is None or uns is None:
        return None
    return [float(prec), float(cov), -float(uns)]


def dominates(a: Sequence[float], b: Sequence[float],
              tol: float = DEFAULT_TOL) -> bool:
    """a 是否 Pareto 支配 b（带容差）。

    条件：所有目标上 a 不比 b 差超过 tol，且至少一个目标 a 比 b 好超过 tol。
    """
    if len(a) != len(b):
        raise ValueError("目标向量长度不一致")
    not_worse = all(x >= y - tol for x, y in zip(a, b))
    strictly_better = any(x > y + tol for x, y in zip(a, b))
    return not_worse and strictly_better


def pareto_frontier(items: Iterable[Dict[str, Any]],
                    tol: float = DEFAULT_TOL) -> List[Dict[str, Any]]:
    """返回非支配解集。

    `items` 每项需含 `objectives`（三元组）。返回保持输入顺序的子集。
    """
    rows = [it for it in items if it.get("objectives")]
    front: List[Dict[str, Any]] = []
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            if dominates(b["objectives"], a["objectives"], tol=tol):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front


def select_seed(frontier: Sequence[Dict[str, Any]],
                primary_idx: int = 0,
                strategy: str = "best_primary",
                round_index: int = 0) -> Optional[Dict[str, Any]]:
    """从前沿里选一个作为下一轮的种子（对应 GEPA 的 candidate_selector）。

    - `best_primary`（默认）：主判据最优 —— 收敛快，但可能反复扎在同一处；
    - `round_robin`：轮转 —— 保持前沿多样性，更贴近 GEPA 的做法。
    """
    if not frontier:
        return None
    if strategy == "round_robin":
        return frontier[round_index % len(frontier)]
    return max(frontier, key=lambda it: it["objectives"][primary_idx])


def summarize_frontier(frontier: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把前沿压成可读摘要（报告用）。"""
    out = []
    for it in frontier:
        o = it["objectives"]
        out.append({
            "id": it.get("id"),
            "metrics": {k: it.get("metrics", {}).get(k) for k in
                        ("accuracy", "coverage", "answered_precision",
                         "abstain_rate", "unsupported_answer_rate")},
            "objectives": dict(zip(OBJECTIVES, [round(x, 4) for x in o])),
        })
    return out
