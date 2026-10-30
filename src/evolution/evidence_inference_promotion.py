"""Fail-closed bridge from an internally accepted EI MIPRO candidate to selection_gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts.types import Candidate, content_id, content_sha256
from ..data.evidence_inference_cohort import LABELS
from ..data.evidence_inference_v2 import DATASET, MANIFEST_PATH, load_manifest
from ..evolution.task_program import parse_task_program, serialize_task_program
from ..governance.policy import static_gate
from ..evolution.task_input_gates import validate_task_demo_rendering
from ..util import read_json, sha256_file, sha256_obj

PROTOCOL = "evidence-inference-v2-abstract-mipro-task-input-v6"
SUPPORTED_PROTOCOLS = {PROTOCOL}


def compute_mipro_plan_sha256(plan: dict[str, Any]) -> str:
    """Hash frozen MIPRO semantics using the runner's canonical exclusions.

    Credential-at-launch attestation and the preflight wrapper's status are
    runtime metadata, not experiment inputs. Keep this implementation shared
    by the runner and the promotion bridge so a valid current report cannot be
    rejected under a subtly different hash contract.
    """
    canonical = dict(plan)
    canonical.pop("plan_sha256", None)
    canonical.pop("api_key_explicit_at_launch", None)
    canonical.pop("status", None)
    return sha256_obj(canonical)


def validate_search_report(report: dict[str, Any], train_rows: list[dict[str, Any]],
                           parent_state: Any) -> tuple[Candidate, dict[str, Any]]:
    """Authenticate an internally passing source report and rebuild its candidate."""
    recovered = report.get("status") == "recovered_trial_internal_gate_completed"
    if (report.get("status") != "completed" and not recovered) or report.get("protocol") not in SUPPORTED_PROTOCOLS:
        raise ValueError("MIPRO source report is incomplete or uses another protocol")
    plan = report.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("MIPRO source report has no embedded protocol plan")
    saved_plan_sha = str(plan.get("plan_sha256") or "")
    if (not saved_plan_sha
            or compute_mipro_plan_sha256(plan) != saved_plan_sha):
        raise ValueError("MIPRO source plan hash mismatch")
    if plan.get("protocol") not in SUPPORTED_PROTOCOLS or plan.get("protocol") != report.get("protocol") or plan.get("dataset") != DATASET:
        raise ValueError("MIPRO plan dataset/protocol mismatch")
    if plan.get("release_id") != parent_state.release_id:
        raise ValueError("MIPRO candidate parent release differs from protocol release")
    for flag in ("selection_gate_read", "official_validation_read", "official_test_read"):
        if report.get(flag) is not False:
            raise ValueError(f"MIPRO source report must attest {flag}=false")
    if report.get("retrieval_used") is not False:
        raise ValueError("MIPRO source report used retrieval")
    if (report.get("candidate_status") != "internal_gate_passed"
            or report.get("eligible_for_official_selection_gate") is not True):
        raise ValueError("candidate did not pass its internal gate")
    if report.get("confirmation_eligible") is not False:
        raise ValueError("exploratory MIPRO candidate cannot claim confirmation eligibility")

    if recovered:
        # A recovered trial is eligible only when its exact gate record is still
        # present in the hash-bound failed source report. This keeps recovery
        # provenance intact without pretending the interrupted search completed.
        source_path = report.get("source_failed_report")
        source_sha = str(report.get("source_failed_report_sha256") or "")
        if not source_path or not source_sha:
            raise ValueError("recovered report is missing its failed-source binding")
        source_path = Path(source_path)
        if not source_path.is_file() or sha256_file(source_path) != source_sha:
            raise ValueError("recovered report failed-source hash mismatch")
        failed_source = read_json(source_path)
        if (failed_source.get("status") != "candidate_search_failed"
                or (failed_source.get("plan") or {}).get("plan_sha256") != saved_plan_sha):
            raise ValueError("recovered report is not bound to the matching failed search")
        saved_gate = report.get("saved_program_gate")
        if not isinstance(saved_gate, dict) or not any(
                row == saved_gate and row.get("candidate_id") == report.get("candidate_id")
                and row.get("passed") is True
                for row in failed_source.get("candidate_program_gates", [])):
            raise ValueError("recovered candidate has no matching passing source gate")
        if report.get("candidate_content_sha256") != saved_gate.get("content_sha256"):
            raise ValueError("recovered candidate content hash differs from its source gate")

    labels = [int(value) for value in plan.get("execution_labels") or []]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("MIPRO report has invalid execution labels")
    baseline = report.get("baseline_internal_metrics_by_execution_label") or {}
    candidate_metrics = report.get("candidate_internal_metrics_by_execution_label") or {}
    expected_labels = {str(value) for value in labels}
    if set(baseline) != expected_labels or set(candidate_metrics) != expected_labels:
        raise ValueError("MIPRO internal metrics do not cover every execution label")
    for label in labels:
        before = baseline[str(label)]
        after = candidate_metrics[str(label)]
        if float(after.get("accuracy", -1)) <= float(before.get("accuracy", -1)) + 1e-12:
            raise ValueError("internal confirmation lacks strict accuracy gain for every label")
        before_classes = before.get("per_gold_label") or {}
        after_classes = after.get("per_gold_label") or {}
        if set(before_classes) != set(LABELS) or set(after_classes) != set(LABELS):
            raise ValueError("internal confirmation is missing one or more gold classes")
        for gold_label in LABELS:
            for metric in ("accuracy", "coverage", "answered_precision"):
                if float(after_classes[gold_label].get(metric, -1)) + 1e-12 < float(before_classes[gold_label].get(metric, -1)):
                    raise ValueError(f"internal confirmation regressed {gold_label}:{metric}")

    artifact = report.get("candidate")
    if not isinstance(artifact, dict):
        raise ValueError("internally passing report has no candidate artifact")
    content = serialize_task_program(artifact.get("system_prompt", ""),
                                     artifact.get("task_demos") or [])
    parsed = parse_task_program(content)
    train_by_id = {str(row["task_id"]): row for row in train_rows}
    for demo in parsed["task_demos"]:
        task_id = str(demo["task_id"])
        source = train_by_id.get(task_id)
        if source is None:
            raise ValueError(f"candidate demo is outside frozen dev_optimize: {task_id}")
        if demo["task"] != source or int(demo["correct_index"]) != int(source["correct_index"]):
            raise ValueError(f"candidate demo differs from frozen train row: {task_id}")

    candidate_id = content_id("cand", parent_state.release_id, "task_program", content)
    if candidate_id != report.get("candidate_id"):
        raise ValueError("candidate identity does not match the MIPRO report")
    candidate = Candidate(
        candidate_id=candidate_id, parent_release_id=parent_state.release_id,
        artifact_type="task_program", artifact_id="task_program",
        diff="MIPROv2 selected prompt × authenticated train-only labeled demos",
        rationale=("Recovered fully evaluated MIPRO trial passed the frozen internal gate; "
                   "pending one aggregate-only selection_gate evaluation." if recovered else
                   "Passed frozen nested internal gate; pending one aggregate-only selection_gate evaluation."),
        new_content=content, content_sha256=content_sha256(content),
        generator={"optimizer": "DSPy MIPROv2", "protocol": PROTOCOL,
                   "source_study_id": report.get("study_id"),
                   "source_plan_sha256": saved_plan_sha},
    )
    return candidate, plan


def selection_gate_preflight(plan: dict[str, Any]) -> dict[str, Any]:
    """Read manifest metadata only; do not materialize official validation rows."""
    manifest = load_manifest()
    entry = manifest.get("splits", {}).get("selection_gate") or {}
    n_tasks = int(entry.get("n", -1))
    labels = [int(value) for value in plan.get("execution_labels") or []]
    if n_tasks <= 0 or not labels:
        raise ValueError("selection_gate manifest size or execution labels are invalid")
    task_evaluations = n_tasks * len(labels) * 2
    router_calls = task_evaluations if plan.get("routing_mode") == "model" else 0
    return {
        "dataset": DATASET, "split": "selection_gate", "task_count": n_tasks,
        "execution_labels": labels,
        "execution_labels_are_independent_rng_seeds": False,
        "baseline_candidate_task_evaluations": task_evaluations,
        "estimated_task_model_calls": task_evaluations,
        "estimated_skill_router_calls": router_calls,
        "estimated_total_model_calls": task_evaluations + router_calls,
        "manifest_selection_ids_sha256": entry.get("ids_sha256"),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "selection_file_sha256": entry.get("file_sha256"),
        "validation_rows_loaded": False, "official_test_read": False,
    }


def candidate_static_gate(candidate: Candidate, cfg: dict[str, Any], parent_state: Any):
    result = static_gate(candidate, cfg, parent_state)
    if candidate.artifact_type != "task_program":
        return result
    try:
        artifact = parse_task_program(candidate.new_content)
    except (TypeError, ValueError) as exc:
        return {**result, "passed": False,
                "results": list(result.get("results") or []) + [{
                    "check": "task_demo_rendering", "passed": False,
                    "detail": f"invalid task_program: {exc}",
                }]}
    demo_gate = validate_task_demo_rendering(artifact.get("task_demos") or [], cfg)
    checks = list(result.get("results") or [])
    checks.append({"check": "task_demo_rendering", "passed": demo_gate["passed"],
                   "detail": demo_gate})
    return {**result, "passed": bool(result.get("passed") and demo_gate["passed"]),
            "results": checks, "task_demo_rendering": demo_gate}
