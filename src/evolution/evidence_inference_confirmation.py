"""Verification helpers for post-preselection Evidence Inference confirmation."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..data.evidence_inference_cohort import LABELS as EI_LABELS
from .evidence_inference_promotion import (SUPPORTED_PROTOCOLS,
                                            compute_mipro_plan_sha256)
from ..util import read_json, sha256_file, sha256_obj

CONFIRMATION_PROTOCOL = "evidence-inference-independent-confirmation-v3"
MIN_CONFIRMATION_TASKS_PER_CLASS = 30


def validate_fresh_scope_independence(
        fresh_cohort: dict[str, Any], source_cohort: dict[str, Any],
        source_plan: dict[str, Any], *, candidate_id: str,
        candidate_report_sha256: str) -> None:
    """Verify post-preselection binding and article/task disjointness.

    The confirmation runner and downstream promotion verifier share this
    function so the independence contract cannot drift between execution and
    review.
    """
    source_hash = str(source_cohort.get("cohort_sha256") or "")
    unhashed_source = dict(source_cohort)
    unhashed_source.pop("cohort_sha256", None)
    if not source_hash or sha256_obj(unhashed_source) != source_hash:
        raise ValueError("candidate source cohort self-hash is invalid")
    if source_plan.get("cohort_sha256") != source_hash:
        raise ValueError("candidate source plan is not bound to its cohort")

    binding = fresh_cohort.get("independent_confirmation_for")
    expected_binding = {
        "candidate_id": candidate_id,
        "candidate_report_sha256": candidate_report_sha256,
        "source_plan_sha256": str(source_plan.get("plan_sha256") or ""),
        "source_cohort_sha256": source_hash,
        "created_after_candidate_preselection": True,
    }
    if binding != expected_binding:
        raise ValueError("fresh cohort is not frozen for this preselected candidate")
    if int(fresh_cohort.get("minimum_annotation_votes") or 0) < 2:
        raise ValueError("fresh confirmation cohort requires two annotation votes")

    excluded_hashes = fresh_cohort.get("excluded_cohort_sha256s")
    if not isinstance(excluded_hashes, list) or source_hash not in {
            str(value) for value in excluded_hashes}:
        raise ValueError("fresh cohort does not exclude the candidate source cohort")
    exclusion_policies = fresh_cohort.get("excluded_cohort_scope_policies")
    if (not isinstance(exclusion_policies, list)
            or not any(isinstance(item, dict)
                       and item.get("cohort_sha256") == source_hash
                       and item.get("scopes") == "all"
                       for item in exclusion_policies)):
        raise ValueError(
            "fresh cohort does not attest exclusion of every source scope")

    source_scopes = source_cohort.get("scopes")
    fresh_scopes = fresh_cohort.get("scopes")
    required_source = {"mipro_train", "search_validation", "internal_confirmation"}
    required_fresh = {"search_validation", "internal_confirmation"}
    if not isinstance(source_scopes, dict) or not required_source.issubset(source_scopes):
        raise ValueError("candidate source cohort lacks required nested scopes")
    if not isinstance(fresh_scopes, dict) or not required_fresh.issubset(fresh_scopes):
        raise ValueError("fresh cohort lacks required evaluation scopes")

    def identities(scope: dict[str, Any], field: str, cohort_label: str) -> set[str]:
        values = scope.get(field)
        if (not isinstance(values, list) or not values
                or any(not isinstance(value, str) or not value for value in values)
                or len(values) != len(set(values))):
            raise ValueError(f"{cohort_label} has invalid {field}")
        return set(values)

    for fresh_name in sorted(required_fresh):
        fresh_scope = fresh_scopes[fresh_name]
        fresh_articles = identities(fresh_scope, "article_ids", "fresh cohort")
        fresh_tasks = identities(fresh_scope, "task_ids", "fresh cohort")
        for source_name in sorted(required_source):
            source_scope = source_scopes[source_name]
            source_articles = identities(source_scope, "article_ids", "source cohort")
            source_tasks = identities(source_scope, "task_ids", "source cohort")
            if fresh_articles & source_articles or fresh_tasks & source_tasks:
                raise ValueError(
                    "fresh evaluation scope overlaps a prior cohort scope: "
                    f"{fresh_name} vs {source_name}")


def _validate_confirmation_metrics(
        metrics: Any, *, labels: list[str], task_count: int,
        class_counts: dict[str, int], context: str) -> None:
    if not isinstance(metrics, dict) or set(metrics) != set(labels):
        raise ValueError(f"independent confirmation {context} metrics are incomplete")
    for label in labels:
        row = metrics[label]
        per_class = row.get("per_gold_label") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or row.get("n") != task_count
                or not isinstance(per_class, dict) or set(per_class) != set(EI_LABELS)):
            raise ValueError(f"independent confirmation {context} metrics are malformed")
        for key in ("accuracy", "coverage", "answered_precision"):
            value = row.get(key)
            if (not isinstance(value, (int, float)) or not math.isfinite(value)
                    or not 0.0 <= value <= 1.0):
                raise ValueError(f"independent confirmation {context} metrics are malformed")
        answered = 0.0
        weighted_correct = 0.0
        weighted_answered = 0.0
        for gold_label in EI_LABELS:
            item = per_class[gold_label]
            if (not isinstance(item, dict)
                    or item.get("n") != class_counts[gold_label]):
                raise ValueError(
                    f"independent confirmation {context} class counts are malformed")
            for key in ("accuracy", "coverage", "answered_precision"):
                value = item.get(key)
                if (not isinstance(value, (int, float)) or not math.isfinite(value)
                        or not 0.0 <= value <= 1.0):
                    raise ValueError(
                        f"independent confirmation {context} class metrics are malformed")
            count = class_counts[gold_label]
            weighted_correct += count * float(item["accuracy"])
            class_answered = count * float(item["coverage"])
            answered += class_answered
            weighted_answered += class_answered * float(item["answered_precision"])
        expected = {
            "accuracy": weighted_correct / task_count,
            "coverage": answered / task_count,
            "answered_precision": weighted_answered / answered if answered else 0.0,
        }
        if any(not math.isclose(float(row[key]), value, rel_tol=1e-9, abs_tol=1e-9)
               for key, value in expected.items()):
            raise ValueError(
                f"independent confirmation {context} aggregate metrics disagree with classes")


def _has_class_aware_gain(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    if float(candidate["accuracy"]) <= float(baseline["accuracy"]) + 1e-12:
        return False
    for label in EI_LABELS:
        for key in ("accuracy", "coverage", "answered_precision"):
            if (float(candidate["per_gold_label"][label][key]) + 1e-12
                    < float(baseline["per_gold_label"][label][key])):
                return False
    return True


def _read_bound_cohort(path_value: Any, file_sha256: Any,
                       cohort_sha256: Any, *, context: str) -> dict[str, Any]:
    path = Path(str(path_value or "")).resolve()
    if not path.is_file():
        raise ValueError(f"{context} cohort file is missing")
    if not isinstance(file_sha256, str) or sha256_file(path) != file_sha256:
        raise ValueError(f"{context} cohort file hash mismatch")
    cohort = read_json(path)
    embedded_hash = str(cohort.get("cohort_sha256") or "")
    unsigned = dict(cohort)
    unsigned.pop("cohort_sha256", None)
    if (not embedded_hash or sha256_obj(unsigned) != embedded_hash
            or embedded_hash != cohort_sha256):
        raise ValueError(f"{context} cohort identity mismatch")
    return cohort


def load_confirmation_attestation(
        report_path: str | Path, *, candidate_id: str,
        candidate_content_sha256: str, source_plan_sha256: str,
        source_report_path: str | Path, source_report_sha256: str) -> dict[str, Any]:
    """Require a completed passing report bound to this exact candidate and search."""
    path = Path(report_path).resolve()
    if not path.is_file():
        raise ValueError("independent confirmation report is missing")
    report = read_json(path)
    plan = report.get("plan") or {}
    actual_plan_sha = sha256_obj({key: value for key, value in plan.items()
                                  if key != "plan_sha256"})
    if (plan.get("protocol") != CONFIRMATION_PROTOCOL
            or plan.get("plan_sha256") != actual_plan_sha):
        raise ValueError("independent confirmation plan identity is invalid")
    if (report.get("status") != "independent_confirmation_passed"
            or report.get("independent_confirmation_completed") is not True
            or report.get("independent_confirmation_passed") is not True
            or report.get("confirmation_eligible") is not True):
        raise ValueError("independent confirmation did not pass")
    if (report.get("candidate_id") != candidate_id
            or plan.get("candidate_id") != candidate_id
            or plan.get("candidate_content_sha256") != candidate_content_sha256):
        raise ValueError("independent confirmation belongs to different candidate content")
    schedule = plan.get("arm_order_schedule")
    task_ids = plan.get("task_ids")
    raw_labels = plan.get("execution_labels")
    labels = [str(value) for value in raw_labels or []]
    task_count = plan.get("task_count")
    class_counts = plan.get("class_counts")
    scopes = plan.get("scopes")
    if (not isinstance(task_ids, list) or len(task_ids) < len(EI_LABELS) * MIN_CONFIRMATION_TASKS_PER_CLASS
            or len(task_ids) % 2 or len(task_ids) != len(set(task_ids))
            or task_count != len(task_ids)
            or not isinstance(class_counts, dict) or set(class_counts) != set(EI_LABELS)
            or any(type(class_counts[label]) is not int
                   or class_counts[label] < MIN_CONFIRMATION_TASKS_PER_CLASS
                   for label in EI_LABELS)
            or sum(class_counts.values()) != task_count
            or not isinstance(scopes, dict)
            or scopes.get("internal_confirmation") != task_count
            or not isinstance(raw_labels, list) or not raw_labels
            or any(type(value) is not int for value in raw_labels)
            or len(labels) != len(set(labels))):
        raise ValueError("independent confirmation violates the minimum sample contract")
    if (plan.get("arm_order_policy") != "per-seed-task-balanced-block-v1"
            or not isinstance(schedule, dict)
            or sha256_obj(schedule) != plan.get("arm_order_schedule_sha256")
            or set(schedule) != set(labels)):
        raise ValueError("independent confirmation arm-order schedule is invalid")
    for label in labels:
        per_label = schedule[label]
        if (not isinstance(per_label, dict) or set(per_label) != set(task_ids)
                or any(order not in (["baseline", "candidate"],
                                    ["candidate", "baseline"])
                       for order in per_label.values())
                or sum(order[0] == "baseline" for order in per_label.values())
                != len(task_ids) // 2):
            raise ValueError("independent confirmation arm-order schedule is not balanced")
    if plan.get("source_plan_sha256") != source_plan_sha256:
        raise ValueError("independent confirmation is not bound to the candidate source plan")
    if plan.get("candidate_source_report_file_sha256") != source_report_sha256:
        raise ValueError("independent confirmation is not bound to the source report file")
    source_path = Path(source_report_path).resolve()
    if (str(source_path) != str(Path(
            plan.get("candidate_source_report_path") or "").resolve())
            or not source_path.is_file()
            or sha256_file(source_path) != source_report_sha256):
        raise ValueError("candidate source report file is missing or has changed")
    source_report = read_json(source_path)
    source_plan = source_report.get("plan") or {}
    source_protocol = str(source_report.get("protocol") or "")
    if (source_protocol not in SUPPORTED_PROTOCOLS
            or source_plan.get("protocol") != source_protocol
            or source_plan.get("plan_sha256") != source_plan_sha256
            or compute_mipro_plan_sha256(source_plan) != source_plan_sha256
            or source_report.get("status") not in {
                "completed", "recovered_trial_internal_gate_completed"}
            or source_report.get("candidate_id") != candidate_id
            or source_report.get("candidate_status") != "internal_gate_passed"
            or source_report.get("eligible_for_official_selection_gate") is not True
            or source_report.get("confirmation_eligible") is not False
            or source_report.get("selection_gate_read") is not False
            or source_report.get("official_validation_read") is not False
            or source_report.get("official_test_read") is not False):
        raise ValueError("candidate source report is not a sealed internal preselection")
    source_cohort = _read_bound_cohort(
        source_report.get("cohort_path"),
        plan.get("source_cohort_file_sha256"),
        plan.get("source_cohort_sha256"), context="candidate source")
    if source_plan.get("cohort_sha256") != source_cohort.get("cohort_sha256"):
        raise ValueError("candidate source plan and cohort differ")
    fresh_cohort = _read_bound_cohort(
        plan.get("cohort_path"), plan.get("cohort_file_sha256"),
        plan.get("cohort_sha256"), context="independent confirmation")
    validate_fresh_scope_independence(
        fresh_cohort, source_cohort, source_plan, candidate_id=candidate_id,
        candidate_report_sha256=source_report_sha256)
    fresh_internal = (fresh_cohort.get("scopes") or {}).get(
        "internal_confirmation") or {}
    if (fresh_internal.get("task_ids") != task_ids
            or fresh_internal.get("class_counts") != class_counts
            or fresh_internal.get("task_count") != task_count):
        raise ValueError("confirmation report tasks differ from its frozen cohort")
    if (report.get("selection_gate_read") is not False
            or report.get("official_validation_read") is not False
            or report.get("official_test_read") is not False
            or plan.get("selection_gate_read") is not False
            or plan.get("official_validation_read") is not False
            or plan.get("official_test_read") is not False):
        raise ValueError("independent confirmation report read a protected evaluation scope")
    if (int(plan.get("minimum_annotation_votes") or 0) < 2
            or not plan.get("cohort_sha256")
            or not plan.get("candidate_source_report_file_sha256")):
        raise ValueError("independent confirmation lacks cohort or source attestation")
    baseline_metrics = report.get("baseline_metrics_by_execution_label")
    candidate_metrics = report.get("candidate_metrics_by_execution_label")
    _validate_confirmation_metrics(
        baseline_metrics, labels=labels, task_count=task_count,
        class_counts=class_counts, context="baseline")
    _validate_confirmation_metrics(
        candidate_metrics, labels=labels, task_count=task_count,
        class_counts=class_counts, context="candidate")
    computed_pass = all(_has_class_aware_gain(
        candidate_metrics[label], baseline_metrics[label]) for label in labels)
    if (not computed_pass
            or report.get("independent_confirmation_passed") is not True
            or report.get("confirmation_eligible") is not True
            or report.get("independent_confirmation_passed") is not computed_pass
            or report.get("confirmation_eligible") is not computed_pass):
        raise ValueError("independent confirmation metrics do not support its pass status")
    accounting = report.get("request_accounting") or {}
    logical = int(accounting.get("logical_operations", -1))
    physical = int(accounting.get("physical_http_attempts", -1))
    minimum_logical = task_count * len(labels) * 2
    if (logical < minimum_logical or physical < logical
            or logical > int(plan.get("logical_budget") or 0)
            or logical > int(plan.get("maximum_logical_operations") or 0)
            or physical > int(plan.get("physical_budget") or 0)
            or physical > int(plan.get("maximum_physical_attempts") or 0)):
        raise ValueError("independent confirmation request accounting is invalid")
    return {
        "report_path": str(path),
        "report_sha256": sha256_file(path),
        "plan_sha256": actual_plan_sha,
        "candidate_id": candidate_id,
        "candidate_content_sha256": candidate_content_sha256,
        "source_plan_sha256": source_plan_sha256,
        "source_report_sha256": source_report_sha256,
        "source_report_path": str(source_path),
        "cohort_sha256": str(plan["cohort_sha256"]),
        "passed": True,
    }


def verify_confirmation_attestation(attestation: dict[str, Any], *,
                                   candidate_id: str,
                                   candidate_content_sha256: str,
                                   source_plan_sha256: str,
                                   source_report_path: str | Path,
                                   source_report_sha256: str) -> bool:
    """Re-read and hash-check the attested report at review/release time."""
    try:
        expected = load_confirmation_attestation(
            attestation.get("report_path", ""), candidate_id=candidate_id,
            candidate_content_sha256=candidate_content_sha256,
            source_plan_sha256=source_plan_sha256,
            source_report_path=source_report_path,
            source_report_sha256=source_report_sha256)
    except (OSError, ValueError, TypeError, KeyError):
        return False
    return expected == attestation
