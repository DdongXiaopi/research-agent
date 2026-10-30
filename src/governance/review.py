"""人工 review 与留痕（FR-205 / Spec §6.3）。

流程：候选 → `review_request.json`（含 diff、指标、回滚点）→ 人工决定 → `review.json`
→ 批准则发布并移动 release 指针。release 必须人工批准，不允许自动发布。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..contracts.types import Candidate, ReviewRecord, content_sha256
from ..evolution.evidence_inference_confirmation import verify_confirmation_attestation
from ..util import ensure_dir, now_iso, artifacts_path, read_json, write_json

REVIEWS_ROOT = artifacts_path("reviews")


class ReviewError(RuntimeError):
    pass


def request_sha256(req: Dict[str, Any]) -> str:
    """Fingerprint the complete review request independent of JSON whitespace."""
    canonical = json.dumps(req, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False)
    return content_sha256(canonical)


def _dir() -> Path:
    return ensure_dir(REVIEWS_ROOT)


def build_request(candidate: Candidate, optimize_metrics: Dict[str, Any],
                  gate_metrics: Dict[str, Any], rollback_point: str,
                  static_gate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "parent_release_id": candidate.parent_release_id,
        "artifact_type": candidate.artifact_type,
        "artifact_id": candidate.artifact_id,
        "attributed_class": candidate.attributed_class,
        "rationale": candidate.rationale,
        "diff": candidate.diff,
        "new_content_sha256": candidate.content_sha256,
        "candidate_artifact_sha256": candidate_artifact_sha256(candidate),
        "generator": candidate.generator,
        "static_gate": {"passed": static_gate["passed"], "results": static_gate["results"]},
        "optimize_metrics": optimize_metrics,
        "gate_metrics": gate_metrics,
        "rollback_point": rollback_point,
        "created_at": now_iso(),
        "status": "pending",
    }


def candidate_artifact_sha256(candidate: Candidate) -> str:
    """Bind body AND metadata/lineage; workflow status is not artifact content."""
    value = {key: item for key, item in candidate.to_dict().items()
             if key not in {"status", "static_gate_results"}}
    return request_sha256(value)


def write_request(req: Dict[str, Any]) -> Path:
    p = _dir() / f"{req['candidate_id']}.request.json"
    write_json(p, req)
    return p


def _aggregate_cost_failure_is_only_measured_overrun(
        cost_runs: Dict[str, Any]) -> bool:
    """Recompute per-label cost status and reject missing or tampered evidence."""
    if not cost_runs:
        return False
    saw_overrun = False
    for result in cost_runs.values():
        if result.get("passed") not in (True, False):
            return False
        items = list(result.get("items") or [])
        cost_items = [item for item in items if item.get("quantity") == "cost"]
        if len(cost_items) != 1 or any(
                item.get("status") != "pass" and item.get("quantity") != "cost"
                for item in items):
            return False
        cost = cost_items[0]
        if (cost.get("status") not in ("pass", "fail")
                or cost.get("base") is None or cost.get("cand") is None
                or cost.get("limit") is None):
            return False
        try:
            baseline, candidate = float(cost["base"]), float(cost["cand"])
            limit = float(cost["limit"])
        except (TypeError, ValueError):
            return False
        if (baseline <= 0 or not math.isfinite(baseline)
                or not math.isfinite(candidate) or not math.isfinite(limit)):
            return False
        delta_pct = (candidate - baseline) / abs(baseline) * 100.0
        recomputed_pass = delta_pct <= limit + 1e-9
        if recomputed_pass != (cost["status"] == "pass"):
            return False
        if not recomputed_pass:
            saw_overrun = True
        if result.get("passed") != recomputed_pass:
            return False
    return saw_overrun


def _cost_exception_matches_measurements(review: Dict[str, Any],
                                        cost_runs: Dict[str, Any]) -> bool:
    recorded = review.get("per_execution_label") or {}
    if set(recorded) != {str(label) for label in cost_runs}:
        return False
    for label, result in cost_runs.items():
        cost = next((item for item in result.get("items", [])
                     if item.get("quantity") == "cost"), None)
        detail = recorded.get(str(label)) or {}
        if cost is None or cost.get("base") is None or cost.get("cand") is None:
            return False
        try:
            base, candidate = float(cost["base"]), float(cost["cand"])
        except (TypeError, ValueError):
            return False
        if base <= 0 or not math.isfinite(base) or not math.isfinite(candidate):
            return False
        delta = round((candidate - base) / abs(base) * 100.0, 4)
        if (detail.get("baseline_usd") != base
                or detail.get("candidate_usd") != candidate
                or detail.get("limit_pct") != cost.get("limit")
                or detail.get("observed_delta_pct") != delta):
            return False
    return True


def _main_dataset_gains_are_positive(dataset_checks: List[Dict[str, Any]]) -> bool:
    main_checks = [check for check in dataset_checks
                   if (check.get("detail") or {}).get("role") == "main"]
    if not main_checks:
        return False
    for check in main_checks:
        delta = (check.get("detail") or {}).get("primary_delta_pp")
        try:
            delta = float(delta)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(delta) or delta <= 0:
            return False
    return True


def _exact_zero_price_cost_is_measured(dataset: str, dataset_report: Dict[str, Any],
                                       evidence: Dict[str, Any]) -> bool:
    """Accept a persisted pre-fix ``n_a`` only for an audited exact-zero tariff.

    Older round reports were written before budget.py distinguished a measured
    zero price from a zero denominator.  This narrow compatibility path keeps
    those reports usable while refusing missing, estimated, or one-sided costs.
    """
    if dataset_report.get("provider_kind") != "http":
        return False
    cost_items = [item for item in dataset_report.get("items", [])
                  if item.get("quantity") == "cost"]
    if len(cost_items) != 1 or cost_items[0].get("status") != "n_a":
        return False
    item = cost_items[0]
    if item.get("base") != 0.0 or item.get("cand") != 0.0:
        return False
    base = (evidence.get("baseline_metrics") or {}).get(dataset) or {}
    cand = (evidence.get("candidate_metrics") or {}).get(dataset) or {}
    # The tolerance report is a gate comparison, so bind the exact-zero proof
    # to that same gate arm rather than to an unrelated optimize measurement.
    base_gate = base.get("gate") if isinstance(base, dict) else None
    cand_gate = cand.get("gate") if isinstance(cand, dict) else None
    # Candidate gate metrics are wrapped as {metrics: {...}} while the
    # baseline arm stores the aggregate fields directly under ``gate``.
    if isinstance(cand_gate, dict) and isinstance(cand_gate.get("metrics"), dict):
        cand_gate = cand_gate["metrics"]
    if not isinstance(base_gate, dict) or not isinstance(cand_gate, dict):
        return False
    return all(
        metrics.get("cost_usd") == 0.0
        and metrics.get("cost_usd_nominal") == 0.0
        and metrics.get("cost_estimated") is False
        for metrics in (base_gate, cand_gate)
    )


def _round_evidence_matches_request(req: Dict[str, Any]) -> bool:
    """Bind a normal review request to its persisted, candidate-specific round."""
    round_id = str(req.get("round_id") or "")
    if not round_id or Path(round_id).name != round_id:
        return False
    report_path = artifacts_path("rounds", f"{round_id}.json")
    try:
        report = read_json(report_path)
    except (OSError, ValueError, TypeError):
        return False
    if (not isinstance(report, dict) or report.get("round_id") != round_id
            or report.get("release_id") != req.get("parent_release_id")):
        return False
    candidates = report.get("candidates") or []
    matches = [item for item in candidates
               if item.get("candidate_id") == req.get("candidate_id")]
    if len(matches) != 1:
        return False
    candidate = matches[0]
    if (candidate.get("status") != "proposed_accept"
            or candidate.get("parent_release_id") != req.get("parent_release_id")
            or candidate.get("content_sha256") != req.get("new_content_sha256")
            or candidate.get("static_gate_passed") is not True
            or (req.get("static_gate") or {}).get("passed") is not True):
        return False

    candidate_metrics = candidate.get("candidate_metrics") or {}
    expected_optimize = {dataset: (metrics.get("optimize") or {})
                         for dataset, metrics in candidate_metrics.items()}
    expected_gate = {dataset: (metrics.get("gate") or {})
                     for dataset, metrics in candidate_metrics.items()}
    if (req.get("optimize_metrics") != expected_optimize
            or req.get("gate_metrics") != expected_gate):
        return False

    evidence = req.get("evaluation_summary") or {}
    # The top-level round report adds this audit-only control limit after the
    # request summary is assembled. All other control fields must match exactly.
    report_control = dict(report.get("measurement_control") or {})
    report_control.pop("max_abs_net", None)
    expected_evidence = {
        "baseline_metrics": candidate.get("baseline_metrics"),
        "candidate_metrics": candidate.get("candidate_metrics"),
        "task_support": candidate.get("task_support"),
        "tolerance": candidate.get("tolerance"),
        "seed_requirement": candidate.get("seed_requirement"),
        "evaluation_support": candidate.get("evaluation_support"),
        "measurement_control": report_control,
    }
    return all(evidence.get(key) == value and value is not None
               for key, value in expected_evidence.items())


def read_request(candidate_id: str) -> Dict[str, Any]:
    p = _dir() / f"{candidate_id}.request.json"
    if not p.exists():
        raise ReviewError(f"review request 不存在: {candidate_id}")
    return read_json(p)


def request_effect_gate_ok(req: Dict[str, Any]) -> bool:
    """An accepted candidate must include its final measurement context."""
    if req.get("verdict") != "proposed_accept":
        return False
    if req.get("evidence_type") == "open_research_evaluation_v1":
        from .open_promotion import verify_request
        return verify_request(req)
    if req.get("evidence_type") == "engineering_fixed_task_v1":
        from .engineering_fixed import verify_request
        return verify_request(req)
    if req.get("evidence_type") == "aggregate_selection_gate_v1":
        evidence = req.get("evaluation_summary") or {}
        if not verify_confirmation_attestation(
                evidence.get("independent_confirmation") or {},
                candidate_id=str(req.get("candidate_id") or ""),
                candidate_content_sha256=str(req.get("new_content_sha256") or ""),
                source_plan_sha256=str(
                    evidence.get("source_search_plan_sha256") or ""),
                source_report_path=str(
                    evidence.get("source_report_path") or ""),
                source_report_sha256=str(
                    evidence.get("source_report_sha256") or "")):
            return False
        tolerance = evidence.get("tolerance_report") or {}
        cost_evidence = tolerance.get("authoritative_cost") or tolerance
        cost_runs = cost_evidence.get("per_execution_label") or {}
        expected_cost_labels = {str(value) for value in evidence.get("execution_labels") or []}
        priced_cost_passed = bool(cost_runs) and all(
            result.get("passed") is True and any(
                item.get("quantity") == "cost" and item.get("status") == "pass"
                for item in result.get("items", []))
            for result in cost_runs.values())
        promotion = (req.get("gate_metrics") or {}).get("promotion") or {}
        promotion_checks = promotion.get("checks") or []
        protocol_check = next((check for check in promotion_checks
                               if check.get("check") == "protocol_same_origin"), {})
        dataset_checks = [check for check in promotion_checks
                          if str(check.get("check", "")).startswith("dataset:")]
        secondary_check = next((check for check in promotion_checks
                                if check.get("check") == "secondary_tolerances"), {})
        cost_overrun_review = req.get("cost_overrun_review") or {}
        promotion_exception = promotion.get("cost_overrun_exception") or {}
        non_cost_tolerances = [value for key, value in tolerance.items()
                               if key != "authoritative_cost"]
        exception_reviewable = (
            evidence.get("cost_exception_reviewable") is True
            and cost_overrun_review.get("eligible") is True
            and cost_overrun_review.get("requires_explicit_human_cost_approval") is True
            and promotion.get("cost_overrun_exception_reviewable") is True
            and promotion_exception.get("eligible") is True
            and promotion_exception.get("per_execution_label")
            == cost_overrun_review.get("per_execution_label")
            and _aggregate_cost_failure_is_only_measured_overrun(cost_runs)
            and all(isinstance(value, dict) and value.get("passed") is True
                    for value in non_cost_tolerances)
            and _cost_exception_matches_measurements(cost_overrun_review, cost_runs))
        if exception_reviewable:
            exception_reviewable = _main_dataset_gains_are_positive(dataset_checks)
        tolerances_ok = cost_evidence.get("passed") is True and priced_cost_passed
        return bool(
            evidence.get("official_validation_read") is True
            and evidence.get("official_test_read") is False
            and evidence.get("per_task_output_written") is False
            and ((evidence.get("promotion_promotable") is True
                  and promotion.get("promotable") is True)
                 or (exception_reviewable
                     and evidence.get("promotion_promotable") is False
                     and promotion.get("promotable") is False))
            and evidence.get("protocol_checked") is False
            and (tolerances_ok or exception_reviewable)
            and expected_cost_labels == set(cost_runs)
            and protocol_check.get("passed") is None
            and bool(dataset_checks)
            and all(check.get("passed") is True for check in dataset_checks)
            and (secondary_check.get("passed") is True or
                 exception_reviewable and secondary_check.get("passed") is False)
            and (req.get("static_gate") or {}).get("passed") is True
            and evidence.get("baseline_metrics")
            and evidence.get("candidate_metrics")
        )
    # A missing round identifier used to be treated as a trusted legacy/manual
    # request. That made a request with only ``verdict=proposed_accept`` and a
    # passing static gate sufficient for approval. There is no supported
    # production path that can establish effect-gate evidence without a round.
    if not req.get("round_id"):
        return False
    if (req.get("static_gate") or {}).get("passed") is not True:
        return False
    if not isinstance(req.get("optimize_metrics"), dict) or not req["optimize_metrics"]:
        return False
    if not isinstance(req.get("gate_metrics"), dict) or not req["gate_metrics"]:
        return False
    if not _round_evidence_matches_request(req):
        return False
    evidence = req.get("evaluation_summary") or {}
    control = (evidence.get("measurement_control") or evidence.get("placebo") or {})
    seeds = evidence.get("seed_requirement") or {}
    tolerance = evidence.get("tolerance") or {}
    if tolerance.get("passed") is not True:
        return False
    # Defense in depth for M2+: real-provider round requests must include a
    # measured, passing cost item. The only compatibility exception is an old
    # report whose two arms both carry an audited exact-zero tariff; this is a
    # measured zero-cost comparison, not an unknown cost.
    per_dataset = tolerance.get("per_dataset") or {}
    for dataset, dataset_report in per_dataset.items():
        if dataset_report.get("provider_kind") not in ("http", "pi"):
            continue
        cost_items = [item for item in dataset_report.get("items", [])
                      if item.get("quantity") == "cost"]
        if not cost_items:
            return False
        if any(item.get("status") != "pass" for item in cost_items):
            if not _exact_zero_price_cost_is_measured(dataset, dataset_report, evidence):
                return False
            continue
    support = evidence.get("task_support") or {}
    cohort_support = evidence.get("evaluation_support") or {}
    if (cohort_support.get("passed") is not True
            or not cohort_support.get("per_dataset_seed")):
        return False
    try:
        paired_tasks = int(support.get("n_paired", 0))
    except (TypeError, ValueError):
        return False
    return bool(evidence.get("baseline_metrics") and
                evidence.get("candidate_metrics") and
                paired_tasks > 0 and
                seeds.get("passed") is True and
                control.get("control_type") == "exact_baseline_repeat" and
                not control.get("contaminated"))


def write_review(record: ReviewRecord) -> Path:
    p = _dir() / f"{record.subject_id or record.subject_sha256[:12]}.review.json"
    write_json(p, record.to_dict())
    return p


def decide(candidate_id: str, decision: str, reason: str = "", reviewer: str = "human",
           metrics: Optional[Dict[str, Any]] = None,
           accept_cost_overrun: bool = False) -> Dict[str, Any]:
    """人工批准 / 拒绝。批准只写 review 记录，发布由上层调用 release.publish。"""
    if decision not in ("approve", "reject"):
        raise ReviewError(f"无效 review 决定: {decision}")
    if not reviewer.strip():
        raise ReviewError("reviewer 不能为空")
    req = read_request(candidate_id)
    if decision == "approve" and not request_effect_gate_ok(req):
        raise ReviewError("候选未通过评测门，或 review 缺少负向对照与完整证据摘要")
    exception_required = bool(
        req.get("evidence_type") == "aggregate_selection_gate_v1"
        and ((req.get("evaluation_summary") or {}).get("cost_exception_reviewable") is True))
    if decision == "approve" and exception_required and not accept_cost_overrun:
        raise ReviewError(
            "该候选超过成本容差；批准时必须显式设置 accept_cost_overrun=True 并说明理由")
    if decision == "approve" and exception_required and not reason.strip():
        raise ReviewError("批准成本超容差例外时必须填写人工权衡理由")
    if accept_cost_overrun and not exception_required:
        raise ReviewError("该 review request 没有可审批的成本超支例外")
    subject_sha = req.get("new_content_sha256") or content_sha256(req.get("diff", ""))
    rec = ReviewRecord(
        action_type="release",
        subject_sha256=subject_sha,
        decision=decision,
        created_at=now_iso(),
        reason=reason,
        reviewer=reviewer,
        subject_id=candidate_id,
        metrics=metrics or {
            "optimize": req.get("optimize_metrics", {}),
            "gate": req.get("gate_metrics", {}),
        },
        rollback_point=req.get("rollback_point", ""),
        parent_release_id=req.get("parent_release_id", ""),
        diff=req.get("diff", ""),
    )
    rec.metrics = dict(rec.metrics or {})
    rec.metrics["review_request_sha256"] = request_sha256(req)
    rec.metrics["cost_overrun_exception_accepted"] = bool(
        decision == "approve" and accept_cost_overrun)
    if exception_required:
        rec.metrics["cost_overrun_exception"] = req.get("cost_overrun_review")
    path_written = write_review(rec)
    return {"review": rec.to_dict(), "path": str(path_written), "request": req}


def list_reviews() -> List[Dict[str, Any]]:
    out = []
    for p in sorted(_dir().glob("*.review.json")):
        out.append(read_json(p))
    return out
