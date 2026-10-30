"""发布门（决策 4）：challenger → champion 之前必须过的检查。

对齐 **MLflow Model Registry 的 promotion gate**（2.9+ 推荐 alias 机制）：
在 staging 上跑自动检查 —— 最小指标 + **对当前 champion 无回退** + schema + 延迟，
并**要求 sign-off**（可审计）。只靠离线指标 promote 是公认陷阱。

本项目的落地：
1. 主判据（默认 `answered_precision`）不回退 —— 三指标口径，不靠单一 accuracy；
2. `unsupported_answer_rate` 不回退（接地性，容差来自 gates.accept.tolerance）；
3. 协议同源（能与封存基线配对），否则候选出身不可比（P1 的护栏）；
4. 次指标容差门禁（`governance.budget.check_tolerances`）通过；
5. **人工 sign-off 必填**（`require_human_review`）—— 机器只给"可提交评审"，
   真正改指针是 `release.promote_challenger()`，且必须带 reviewer + reason。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .budget import check_tolerances
from .grounding_attestation import require_grounding_calibration
from ..eval.assessment_policy import seed_safety

DEFAULT_BLOCKING = (
    "缺少主数据集评测结果",
    "主判据回退",
    "无支撑主张率回退",
    "协议不同源",
    "次指标容差未通过",
)


def promotion_decision(cfg: Dict[str, Any], gate_report: Dict[str, Any],
                       protocol_ok: Optional[bool] = None,
                       tolerance_report: Optional[Dict[str, Any]] = None,
                       protocol_checked: bool = True,
                       assessment_scope: str = "formal") -> Dict[str, Any]:
    """综合判定 challenger 能不能晋升。

    `gate_report` 来自 `eval.gate_verdict.compare_on_split`（selection_gate 上的三指标比较）。
    只输出"能不能提交人工评审"，不改任何指针。

    `protocol_checked=False`（离线夹具等明确不与封存基线配对的场景）：
    协议项记为**未检查**而不是通过 —— 报告里不得写成"协议同源已验证"。
    """
    ev = cfg.get("evaluation", {}) or {}
    tol_pp = float(ev.get("primary_axis_tolerance_pp", 0.0) or 0.0)
    unsupported_tol = float((cfg.get("gates", {}).get("accept", {}).get("tolerance", {})
                             or {}).get("unsupported_answer_rate_pp", 5.0))
    accept_cfg = cfg.get("gates", {}).get("accept", {}) or {}
    require_classwise = bool(accept_cfg.get("require_classwise_non_regression", False))
    require_strict_accuracy_gain = bool(accept_cfg.get("require_strict_accuracy_gain", False))

    checks: List[Dict[str, Any]] = []
    blocking: List[str] = []
    if assessment_scope not in {"formal", "exploratory"}:
        raise ValueError("unknown promotion assessment scope")

    # Defense in depth for callers that bypass the standard gate runner.
    # The standard runner performs this check before measurement; promotion
    # repeats it against the report's exact dataset set.
    grounding_attestation = require_grounding_calibration(
        cfg, (gate_report.get("datasets") or
              (gate_report.get("per_dataset") or {}).keys()))
    checks.append({"check": "semantic_grounding_calibration",
                   "passed": True if grounding_attestation["required"] else None,
                   "detail": grounding_attestation})

    # 1) 协议同源
    if protocol_ok is None and protocol_checked:
        protocol_ok = _protocol_ok(cfg)
    if not protocol_checked:
        checks.append({"check": "protocol_same_origin", "passed": None,
                       "detail": "本次显式跳过协议校验（离线夹具/不同源场景）："
                                 "记为未检查，**不得写成已验证**"})
        if assessment_scope == "formal":
            blocking.append("协议未校验")
    else:
        checks.append({"check": "protocol_same_origin", "passed": bool(protocol_ok),
                       "detail": ("当前配置与封存 final_holdout 基线协议一致（同源可比）"
                                  if protocol_ok else
                                  "当前配置与封存基线协议不一致 —— 候选出身不可比（P1）")})
        if not protocol_ok:
            blocking.append("协议不同源")

    # 2/3) 逐数据集：主判据不回退 + unsupported 不回退 + coverage 必报
    per_ds: Dict[str, Any] = {}
    dataset_reports = gate_report.get("per_dataset")
    required = set(ev.get("primary_final_datasets") or [])
    if required and (not isinstance(dataset_reports, dict)
                     or any(ds not in dataset_reports
                            or (dataset_reports[ds] or {}).get("role") != "main"
                            for ds in required)):
        blocking.append("缺少配置要求的主数据集评测结果")
    if not isinstance(dataset_reports, dict) or not dataset_reports:
        blocking.append("缺少主数据集评测结果")
        checks.append({"check": "main_dataset_results", "passed": False,
                       "detail": "gate report has no per-dataset results"})
        dataset_reports = {}
    elif not any((blk or {}).get("role") == "main"
                 for blk in dataset_reports.values() if isinstance(blk, dict)):
        blocking.append("缺少主数据集评测结果")
        checks.append({"check": "main_dataset_results", "passed": False,
                       "detail": "gate report has no dataset with role=main"})
    else:
        checks.append({"check": "main_dataset_results", "passed": True,
                       "detail": "at least one main dataset is present"})

    for ds, blk in dataset_reports.items():
        if not isinstance(blk, dict):
            blocking.append(f"{ds}：主数据集评测结果无效")
            checks.append({"check": f"dataset:{ds}", "passed": False,
                           "detail": "dataset report is not a mapping"})
            continue
        sd = str(blk.get("primary_seed"))
        row = (blk.get("per_seed") or {}).get(sd) or {}
        if not row:
            if blk.get("role") == "main":
                blocking.append(f"{ds}：主数据集评测结果缺少 primary seed 行")
            per_ds[ds] = {"role": blk.get("role"), "passed": False,
                          "reason": "primary seed row missing"}
            checks.append({"check": f"dataset:{ds}", "passed": False,
                           "detail": per_ds[ds]})
            continue
        primary_delta = row.get("primary_delta_pp")
        primary_delta_valid = (isinstance(primary_delta, (int, float))
                               and not isinstance(primary_delta, bool)
                               and math.isfinite(float(primary_delta)))
        non_regress = (primary_delta_valid
                       and float(primary_delta) >= -tol_pp - 1e-9
                       and row.get("primary_non_regression") is True)
        cov_reported = bool(row.get("coverage_reported"))
        coverage_delta = (row.get("deltas_pp") or {}).get("coverage")
        coverage_valid = (isinstance(coverage_delta, (int, float))
                          and not isinstance(coverage_delta, bool)
                          and math.isfinite(float(coverage_delta)))
        ub = ((row.get("candidate") or {}).get("unsupported_answer_rate"),
              (row.get("baseline") or {}).get("unsupported_answer_rate"))
        if ub[0] is None or ub[1] is None:
            ub_ok, ub_detail = False, "缺 unsupported 数据（必需指标未验证）"
        elif any(not isinstance(value, (int, float)) or isinstance(value, bool)
                 or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
                 for value in ub):
            ub_ok, ub_detail = False, "unsupported 数据无效或超出 [0, 1]"
        else:
            ub_delta = (ub[0] - ub[1]) * 100.0
            ub_ok = ub_delta <= unsupported_tol + 1e-9
            ub_detail = f"上升 {ub_delta:+.2f}pp（容差 {unsupported_tol}pp）"
        ds_checks = {
            "role": blk.get("role"),
            "primary_axis": row.get("primary_axis"),
            "primary_delta_pp": primary_delta,
            "primary_delta_valid": primary_delta_valid,
            "primary_non_regression": non_regress,
            "coverage_reported": cov_reported,
            "coverage_delta_pp": coverage_delta,
            "coverage_delta_valid": coverage_valid,
            "unsupported_ok": ub_ok,
            "unsupported_detail": ub_detail,
            "verdict": row.get("verdict"),
            "caveats": row.get("caveats", []),
            "floor_effect": (row.get("floor_effect") or {}).get("triggered"),
        }
        classwise_ok: Optional[bool] = None
        classwise_detail: Dict[str, Any] = {"required": require_classwise}
        seed_rows = blk.get("per_seed") or {sd: row}
        seed_policy = {str(seed): seed_safety(
            seed_row, primary_tolerance_pp=tol_pp,
            coverage_tolerance_pp=float(accept_cfg.get("coverage_tolerance_pp", 0)),
            accuracy_tolerance_pp=float(accept_cfg.get("accuracy_tolerance_pp", 0)),
            unsupported_tolerance_pp=unsupported_tol)
            for seed, seed_row in seed_rows.items()}
        seeds_safe = bool(seed_policy) and all(x["passed"] for x in seed_policy.values())
        declared_seeds = gate_report.get("seeds")
        if declared_seeds is not None:
            seeds_safe = seeds_safe and (len(declared_seeds) == len(set(declared_seeds))
                and {str(s) for s in declared_seeds} == set(seed_policy))
        ds_checks["all_seed_safety"] = {"passed": seeds_safe, "per_seed": seed_policy}
        if require_classwise:
            per_seed: Dict[str, Any] = {}
            for seed, seed_row in seed_rows.items():
                pair = seed_row.get("class_metrics") or {}
                before = pair.get("baseline") or {}
                after = pair.get("candidate") or {}
                class_labels = set(before)
                reasons = []
                per_class: Dict[str, Any] = {}
                if not class_labels or class_labels != set(after):
                    reasons.append("class labels missing or changed")
                for label in sorted(class_labels | set(after)):
                    b, c = before.get(label) or {}, after.get(label) or {}
                    label_checks = {}
                    for metric in ("accuracy", "coverage", "answered_precision"):
                        bv, cv = b.get(metric), c.get(metric)
                        passed = (bv is not None and cv is not None
                                  and float(cv) + 1e-9 >= float(bv))
                        label_checks[metric] = {
                            "baseline": bv, "candidate": cv,
                            "delta_pp": (round((float(cv) - float(bv)) * 100, 4)
                                         if bv is not None and cv is not None else None),
                            "passed": passed,
                        }
                        if not passed:
                            reasons.append(f"{label}:{metric} regressed or missing")
                    task_count_match = b.get("n") == c.get("n")
                    label_checks["task_count_match"] = task_count_match
                    if not task_count_match:
                        reasons.append(f"{label}: task counts differ")
                    per_class[label] = label_checks
                per_seed[str(seed)] = {
                    "passed": not reasons,
                    "per_class": per_class,
                    "reasons": reasons,
                }
            classwise_ok = bool(per_seed) and all(
                result["passed"] for result in per_seed.values())
            classwise_detail = {
                "required": True, "passed": classwise_ok,
                "evaluated_all_seeds": True,
                "per_seed": per_seed,
            }
        strict_gain_ok: Optional[bool] = None
        if require_strict_accuracy_gain:
            strict_by_seed = {
                str(seed): (seed_row.get("deltas_pp") or {}).get("accuracy")
                for seed, seed_row in seed_rows.items()
            }
            strict_gain_ok = bool(strict_by_seed) and all(
                delta is not None and float(delta) > 1e-9
                for delta in strict_by_seed.values())
        ds_checks["classwise_non_regression"] = classwise_detail
        ds_checks["strict_accuracy_gain"] = {
            "required": require_strict_accuracy_gain,
            "passed": strict_gain_ok,
            "accuracy_delta_pp": (row.get("deltas_pp") or {}).get("accuracy"),
            "accuracy_delta_pp_by_seed": ({
                str(seed): (seed_row.get("deltas_pp") or {}).get("accuracy")
                for seed, seed_row in seed_rows.items()
            } if require_strict_accuracy_gain else {}),
            "evaluated_all_seeds": require_strict_accuracy_gain,
        }
        per_ds[ds] = ds_checks
        # 只有 main 角色的数据集能一票否决；smoke 集只报不算
        if blk.get("role") == "main":
            if not seeds_safe:
                blocking.append(f"{ds}：跨 seed 主指标、有效覆盖率或准确率回退")
            if not non_regress:
                blocking.append(f"{ds}：主判据回退（{primary_delta}pp）")
            if not ub_ok:
                blocking.append(f"{ds}：无支撑主张率回退（{ub_detail}）")
            if not cov_reported or not coverage_valid:
                blocking.append(f"{ds}：coverage 未同报（precision 可能靠保守刷出来）")
            if classwise_ok is False:
                blocking.append(f"{ds}：逐类指标回退或不完整")
            if strict_gain_ok is False:
                blocking.append(f"{ds}：accuracy 未严格提升")
        dataset_passed = bool(non_regress and ub_ok and cov_reported and coverage_valid
                              and seeds_safe
                              and classwise_ok is not False
                              and strict_gain_ok is not False)
        checks.append({"check": f"dataset:{ds}",
                       "passed": dataset_passed,
                       "detail": ds_checks})

    # 4) 次指标容差（调用方传入；不传则跳过，记为未检查而非通过）
    cost_exception_reviewable = False
    cost_exception_detail: Dict[str, Any] = {
        "eligible": False,
        "requires_explicit_human_cost_approval": True,
    }
    if tolerance_report is not None:
        tol_ok = all(bool(v.get("passed", True)) for v in tolerance_report.values())
        checks.append({"check": "secondary_tolerances", "passed": tol_ok,
                       "detail": tolerance_report})
        if not tol_ok:
            blocking.append("次指标容差未通过")
            cost_exception_reviewable, cost_exception_detail = _cost_exception_reviewability(
                tolerance_report, per_ds, blocking)
    else:
        checks.append({"check": "secondary_tolerances", "passed": None,
                       "detail": "本次未传次指标比较（记为未检查，不算通过）"})

    require_review = bool(cfg.get("gates", {}).get("accept", {})
                          .get("require_human_review", True))
    return {
        "promotable": not blocking,
        "assessment_scope": assessment_scope,
        "cost_overrun_exception_reviewable": cost_exception_reviewable,
        "cost_overrun_exception": cost_exception_detail,
        "blocking": blocking,
        "checks": checks,
        "per_dataset": per_ds,
        "require_human_review": require_review,
        "tolerances": {"primary_axis_tolerance_pp": tol_pp,
                       "unsupported_answer_rate_pp": unsupported_tol},
        "note": ("机器判定只输出『可提交人工评审』；真正移动 champion 指针必须走 "
                 "`rsi.py promote`（带 reviewer + reason，写入 release 血缘）"
                 if require_review else "配置已关闭人工评审要求（require_human_review=false）"),
        "split_note": gate_report.get("significance_note", ""),
    }


def _cost_exception_reviewability(tolerance_report: Dict[str, Any],
                                  per_dataset: Dict[str, Any],
                                  blocking: List[str]) -> tuple[bool, Dict[str, Any]]:
    """Allow submission for human review only when measured cost is the sole blocker.

    This never changes ``promotable``. A separate explicit approval flag is
    required at review time, and the measured overrun remains visible to the
    reviewer. Missing/unknown cost, any other failed tolerance, or a nonpositive
    main-metric gain stays fail-closed.
    """
    report = tolerance_report.get("authoritative_cost") or {}
    runs = report.get("per_execution_label") or {}
    other_tolerances = [value for key, value in tolerance_report.items()
                        if key != "authoritative_cost"]
    if (blocking != ["次指标容差未通过"] or not runs
            or report.get("passed") is not False
            or any(not isinstance(value, dict) or value.get("passed") is not True
                   for value in other_tolerances)):
        return False, {"eligible": False, "reason": "cost_is_not_the_only_blocker"}
    labels = {}
    for label, result in runs.items():
        items = list(result.get("items") or [])
        cost_items = [item for item in items if item.get("quantity") == "cost"]
        failed = [item for item in items if item.get("status") == "fail"]
        if (result.get("passed") is not False or len(cost_items) != 1
                or failed != cost_items or cost_items[0].get("status") != "fail"
                or any(item.get("status") != "pass" for item in items
                       if item.get("quantity") != "cost")
                or cost_items[0].get("base") is None
                or cost_items[0].get("cand") is None):
            return False, {"eligible": False,
                           "reason": f"cost_evidence_incomplete_or_other_failure:{label}"}
        cost = cost_items[0]
        try:
            base_value = float(cost["base"])
            candidate_value = float(cost["cand"])
            limit_pct = float(cost["limit"])
        except (TypeError, ValueError):
            return False, {"eligible": False,
                           "reason": f"cost_overrun_not_computable:{label}"}
        if base_value <= 0:
            return False, {"eligible": False,
                           "reason": f"cost_overrun_not_computable:{label}"}
        delta_pct = (candidate_value - base_value) / abs(base_value) * 100.0
        if (not math.isfinite(base_value) or not math.isfinite(candidate_value)
                or not math.isfinite(limit_pct) or delta_pct <= limit_pct + 1e-9):
            return False, {"eligible": False,
                           "reason": f"cost_overrun_measurement_inconsistent:{label}"}
        labels[str(label)] = {
            "baseline_usd": base_value,
            "candidate_usd": candidate_value,
            "limit_pct": limit_pct,
            "observed_delta_pct": round(delta_pct, 4),
        }
    main_rows = [row for row in per_dataset.values() if row.get("role") == "main"]
    if not main_rows or not all(
            row.get("primary_non_regression") is True
            and row.get("primary_delta_pp") is not None
            and float(row["primary_delta_pp"]) > 0
            for row in main_rows):
        return False, {"eligible": False, "reason": "main_metric_gain_not_positive"}
    detail = {
        "eligible": True,
        "reason": "measured_cost_is_the_only_failed_gate_and_main_metric_improved",
        "requires_explicit_human_cost_approval": True,
        "per_execution_label": labels,
    }
    return True, detail


def _protocol_ok(cfg: Dict[str, Any]) -> bool:
    """当前配置能否与封存 final_holdout 基线配对（同源）。"""
    try:
        from ..eval.final_eval import check_protocol, load_baseline

        return bool(check_protocol(cfg, load_baseline())["ok"])
    except Exception:
        # 基线不存在 / 损坏 —— 无法证明同源，按 fail-closed 判不同源
        return False
