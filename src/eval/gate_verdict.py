"""决策 1（R1）：把**常规判定**下沉到 `selection_gate`，不碰封存集。

## 为什么需要这条通路
Spec §418 的揭盲预算是"基线封存后 + 最终各一次"。实测 suppqa 已经 2/2 用满，
`unblind_budget` 启用后它已**退役** —— 再想用封存集做"A/B 谁更好"的判定，
系统现在会硬拒绝（这是对的）。但判定需求并没有消失。

hermes（vendor, `PLAN.md:636`）给的解法正是：**在进化集之外单独设一个可反复跑的
回归门**（TBLite，100 题、1–2 小时、独立阈值），回归判定不花封存预算。
本项目的 `selection_gate` 就是这个角色 —— 它本来就设计成"只回传聚合、不给 CI"，
代价要说清楚：

| 集 | 有 CI | 能反复看 | 用途 |
|---|---|---|---|
| dev_optimize | ✅ | ✅ | 进化集，用它下结论 = Goodhart |
| selection_gate | ❌（设计如此）| ✅ | **本模块**：方向性判定 / 发布门 |
| final_holdout | ✅ | ❌（预算耗尽） | 已退役，只保留那一次的最终声明 |

所以这里出的结论**只能称"方向性"**，不得称"显著改善" —— 函数会把这一点写进结果。
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Sequence

from ..governance.budget import BudgetGateError, check_tolerances
from ..governance.grounding_attestation import require_grounding_calibration
from ..governance import release as rel
from ..runtime.runner import ArtifactState
from ..runtime.channels import behavior_profile
from ..util import artifacts_path, now_iso, sha256_obj, write_json
from .harness import evaluate
from .verdict import pair_triple

GATE_VERDICT_ROOT = artifacts_path("gate_verdicts")
GATE_VERDICT_SPLIT = "selection_gate"


def _require_live_execution_guard(cfg: Dict[str, Any],
                                  execution_plan_sha256: str | None) -> None:
    provider = str((cfg.get("runtime") or {}).get("provider") or "fake").lower()
    if provider == "fake":
        return
    if not re.fullmatch(r"[0-9a-f]{64}", execution_plan_sha256 or ""):
        raise ValueError("live gate scoring requires a verified frozen execution-plan SHA-256")
    if os.environ.get("RSI_ACTIVE_EXECUTION_PLAN_SHA256") != execution_plan_sha256:
        raise PermissionError(
            "live gate scoring requires the active verified execution plan")
    if not os.environ.get("RSI_CALL_LEDGER_PATH"):
        raise ValueError("live gate scoring requires the bounded append-only request ledger")
    for name in ("RSI_CALL_LEDGER_MAX_LOGICAL", "RSI_CALL_LEDGER_MAX_PHYSICAL"):
        try:
            value = int(os.environ.get(name, "0"))
        except ValueError:
            value = 0
        if value < 1:
            raise ValueError(f"live gate scoring requires a positive {name} ceiling")


def compare_on_split(cfg: Dict[str, Any], champion: ArtifactState, challenger: ArtifactState,
                     datasets: Sequence[str], split: str = "selection_gate",
                     seeds: Optional[Sequence[int]] = None,
                     limit: Optional[int] = None,
                     execution_plan_sha256: str | None = None) -> Dict[str, Any]:
    """在两个 release 之间跑同一 split，三指标并列比较（**不碰 final_holdout**）。"""
    if split != GATE_VERDICT_SPLIT:
        raise ValueError(
            f"gate verdict only accepts {GATE_VERDICT_SPLIT}; "
            "optimize diagnostics and final evaluation use separate entrypoints")
    _require_live_execution_guard(cfg, execution_plan_sha256)
    grounding_attestation = require_grounding_calibration(cfg, datasets)
    cfg_fp = sha256_obj(cfg)
    seed_list = [int(s) for s in seeds] if seeds else [
        int(cfg.get("evaluation", {}).get("seed", 0))]
    primary = set(cfg.get("evaluation", {}).get("primary_final_datasets") or [])
    transfer = set(cfg.get("evaluation", {}).get("transfer_datasets") or [])

    per_ds: Dict[str, Any] = {}
    for ds in datasets:
        per_seed: Dict[str, Any] = {}
        for sd in seed_list:
            b = evaluate(cfg, champion, ds, split, seed=sd, limit=limit,
                         run_tag="gate-champion", write_per_task=False, cfg_fp=cfg_fp)
            c = evaluate(cfg, challenger, ds, split, seed=sd, limit=limit,
                         run_tag="gate-challenger", write_per_task=False, cfg_fp=cfg_fp)
            if b.n_tasks < 1 or c.n_tasks < 1:
                raise ValueError(
                    f"gate comparison returned no tasks for {ds}/{split}/seed={sd}")
            if b.n_tasks != c.n_tasks:
                raise ValueError(
                    f"gate comparison task-count mismatch for {ds}/{split}/seed={sd}: "
                    f"champion={b.n_tasks}, challenger={c.n_tasks}")
            base_cohort = str(getattr(b, "task_cohort_sha256", ""))
            candidate_cohort = str(getattr(c, "task_cohort_sha256", ""))
            if not base_cohort or not candidate_cohort:
                raise ValueError(
                    f"gate comparison missing task-cohort fingerprint for "
                    f"{ds}/{split}/seed={sd}")
            if base_cohort != candidate_cohort:
                raise ValueError(
                    f"gate comparison task-cohort mismatch for {ds}/{split}/seed={sd}")
            per_seed[str(sd)] = {
                "champion_run_id": b.run_id,
                "challenger_run_id": c.run_id,
                "n_tasks": b.n_tasks,
                "task_cohort_sha256": base_cohort,
                # Aggregate-only provider accounting for gate consumers.
                "cost": {"baseline": b.cost, "candidate": c.cost},
                "metrics": {"baseline": b.metric_values,
                            "candidate": c.metric_values},
                "class_metrics": {
                    "baseline": b.class_metrics,
                    "candidate": c.class_metrics,
                },
                **pair_triple(b.metric_values, c.metric_values, cfg),
            }
        per_ds[ds] = {
            "role": ("main" if ds in primary else "smoke" if ds in transfer
                     else "unregistered" if primary else "main"),
            "per_seed": per_seed,
            "primary_seed": seed_list[0],
        }

    return {
        "created_at": now_iso(),
        "split": split,
        "champion_release": champion.release_id,
        "challenger_release": challenger.release_id,
        "datasets": list(datasets),
        "grounding_calibration": grounding_attestation,
        "seeds": seed_list,
        "per_dataset": per_ds,
        "budget_note": ("本次判定跑在 %s 上，**不消耗 final_holdout 揭盲预算**"
                        "（封存集退役后这是唯一可反复跑的判定面）" % split),
        "significance_note": ("selection_gate 按设计只回传聚合、不给 CI；"
                              "这里的结论只能是方向性的，不得称「显著改善」。"
                              "显著性结论只能来自 final_holdout（已退役）或 future 重封。"),
    }


def _tolerance_report(cfg: Dict[str, Any], comparison: Dict[str, Any],
                      champion: ArtifactState,
                      challenger: ArtifactState) -> Dict[str, Any]:
    """Run configured secondary gates for every dataset and execution seed."""
    champion_effective = behavior_profile(champion, cfg).get("effective") or []
    challenger_effective = behavior_profile(challenger, cfg).get("effective") or []
    reports: Dict[str, Dict[str, Any]] = {}
    cost_runs: Dict[str, Dict[str, Any]] = {}
    cost_enabled = ((cfg.get("gates", {}).get("accept", {}).get("tolerance", {})
                     or {}).get("cost_pct") is not None)

    for dataset, block in (comparison.get("per_dataset") or {}).items():
        for seed, row in (block.get("per_seed") or {}).items():
            label = f"{dataset}/{seed}"
            metrics = row.get("metrics") or {}
            baseline = metrics.get("baseline") or {}
            candidate = metrics.get("candidate") or {}
            try:
                result = check_tolerances(
                    baseline, candidate, cfg,
                    base_effective=champion_effective,
                    cand_effective=challenger_effective)
            except BudgetGateError as exc:
                result = {"passed": False, "provider_kind": (
                    cfg.get("runtime", {}).get("provider", "fake")),
                    "not_measurable": [], "items": [], "error": str(exc)}
            reports[label] = result
            cost_items = [item for item in result.get("items", [])
                          if item.get("quantity") == "cost"]
            if cost_enabled:
                cost_passed = (len(cost_items) == 1
                               and cost_items[0].get("status") == "pass")
                cost_runs[label] = {
                    **result,
                    "passed": cost_passed,
                }

    non_cost_passed = bool(reports) and all(
        all(item.get("status") in ("pass", "n_a")
            for item in result.get("items", [])
            if item.get("quantity") != "cost")
        and "error" not in result
        for result in reports.values())
    if not cost_enabled:
        cost_passed = True
    else:
        cost_passed = bool(cost_runs) and all(
            report.get("passed") is True for report in cost_runs.values())

    return {
        "secondary_tolerances": {
            "passed": non_cost_passed,
            "per_dataset_seed": reports,
            "note": "每个 dataset/seed 独立比较；n/a 仅按 check_tolerances 的 required/strict 规则处理",
        },
        "authoritative_cost": {
            "passed": cost_passed,
            "per_execution_label": cost_runs,
            "required_by_config": cost_enabled,
        },
    }


def run_gate_verdict(cfg: Dict[str, Any], challenger_id: str,
                     champion_id: Optional[str] = None,
                     datasets: Optional[Sequence[str]] = None,
                     split: str = "selection_gate",
                     seeds: Optional[Sequence[int]] = None,
                     limit: Optional[int] = None,
                     check_protocol: bool = True,
                     execution_plan_sha256: str | None = None) -> Dict[str, Any]:
    """跑一次下沉判定，落盘到 `artifacts/gate_verdicts/`。

    `check_protocol=False` 用于离线夹具等**明确不与封存基线配对**的场景：
    判定照出，但协议项标注为"未检查"，不得被当作同源证据。
    """
    if split != GATE_VERDICT_SPLIT:
        raise ValueError(
            f"gate verdict only accepts {GATE_VERDICT_SPLIT}; "
            "optimize diagnostics and final evaluation use separate entrypoints")
    _require_live_execution_guard(cfg, execution_plan_sha256)
    ds = list(datasets or cfg.get("evaluation", {}).get("datasets", []))
    # Reject a missing/stale semantic-judge calibration before loading
    # releases or measuring candidate task outcomes.
    grounding_attestation = require_grounding_calibration(cfg, ds)
    champ_id = champion_id or rel.current_release_id()
    champion = rel.load_state(champ_id)
    challenger = rel.load_state(challenger_id)
    if champ_id == challenger_id:
        raise ValueError("champion 与 challenger 是同一个 release，判定无意义")
    rep = compare_on_split(cfg, champion, challenger, ds, split=split, seeds=seeds,
                           limit=limit,
                           execution_plan_sha256=execution_plan_sha256)
    if execution_plan_sha256:
        rep["execution_plan_sha256"] = execution_plan_sha256
    rep["grounding_calibration"] = grounding_attestation
    from ..governance.promotion import promotion_decision

    tolerance_report = _tolerance_report(cfg, rep, champion, challenger)
    rep["tolerance_report"] = tolerance_report
    rep["promotion"] = promotion_decision(
        cfg, rep, tolerance_report=tolerance_report,
        protocol_checked=check_protocol)
    # Bind the promotion summary to the exact releases measured by this gate.
    rep["promotion"].update({
        "champion_release_id": champ_id,
        "challenger_release_id": challenger_id,
    })
    p = GATE_VERDICT_ROOT / ("gate-verdict-%s-%s.json"
                             % (challenger_id, sha256_obj(rep["created_at"])[:10]))
    write_json(p, rep)
    rep["report_path"] = str(p)
    return rep
