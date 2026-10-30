"""Read-only replay verification, never a production grounding attestation.

The selected-answer calibration format binds the prompt/parser and recorded
calls, but currently has no completed-provider-job provenance or runtime
dispatcher binding. Reproducing it cannot authorize a production metric.
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any

from .grounding_selected_answer_calibration import (
    REPORT_SCHEMA,
    SCOPE,
    TASK_CONTRACT,
    _sha,
    build_selected_answer_plan,
    score_selected_answer_calibration,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
PRODUCTION_BLOCKERS = (
    "selected_answer_runtime_dispatcher_not_bound",
    "completed_provider_job_receipts_not_verified",
    "real_passing_calibration_report_not_verified",
)


def production_attestation_rejection() -> str:
    """Explain why this report format cannot authorize production scoring."""
    return ("selected-answer calibration cannot attest production: "
            + ", ".join(PRODUCTION_BLOCKERS))


def _read_report(report_path: Path, expected_sha256: str) -> dict[str, Any]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key in selected-answer report")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-finite value in selected-answer report: {value}")

    try:
        raw = report_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("selected-answer report file SHA-256 is stale")
        report = json.loads(raw.decode("utf-8"),
                            object_pairs_hook=unique_object,
                            parse_constant=reject_constant)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("selected-answer report is unreadable or invalid") from exc
    if not isinstance(report, dict):
        raise ValueError("selected-answer report must be a mapping")
    return report


def verify_selected_answer_report(
    report_path: str | Path,
    *,
    expected_report_sha256: str,
    dataset: str,
    model_id: str,
    model_protocol_sha256: str,
    packet: dict[str, Any],
    reference: dict[str, Any],
    plan: dict[str, Any],
    call_records: list[dict[str, Any]],
    task_contract: str = TASK_CONTRACT,
    scope: str = SCOPE,
) -> dict[str, Any]:
    """Reproduce a passing recorded report without calls, writes or activation.

    The expected SHA identifies the report's exact file bytes; its embedded SHA
    separately binds the canonical report. All inputs are caller-supplied,
    explicitly identified calibration artifacts, never inferred production data.
    Recorded per-call receipts are replayed but are not evidence of a real
    provider job. A successful result remains ineligible for production use.
    """
    if (not isinstance(expected_report_sha256, str)
            or _SHA256.fullmatch(expected_report_sha256) is None):
        raise ValueError("expected selected-answer report file SHA-256 is required")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("selected-answer report dataset is required")
    if task_contract != TASK_CONTRACT:
        raise ValueError("selected-answer report task contract mismatch")
    if scope != SCOPE:
        raise ValueError("top3 cannot inherit single_doc selected-answer calibration")
    report_path = Path(report_path).expanduser()
    report = _read_report(report_path, expected_report_sha256)
    if (report.get("schema") != REPORT_SCHEMA
            or report.get("dataset") != dataset
            or report.get("task_contract") != task_contract
            or report.get("scope") != scope):
        raise ValueError("selected-answer report schema/dataset/task/scope mismatch")
    if report.get("report_sha256") != _sha(
            {key: value for key, value in report.items() if key != "report_sha256"}):
        raise ValueError("selected-answer report embedded SHA-256 mismatch")
    if report.get("status") != "completed" or report.get("calibration_pass") is not True:
        raise ValueError("selected-answer report did not pass its frozen thresholds")
    if not isinstance(plan, dict):
        raise ValueError("selected-answer report requires its frozen plan")
    # Use the public builder so malformed model identities, altered token limits
    # and evidence truncation cannot bypass its validation through the scorer's
    # private plan reconstruction helper.
    rebuilt_plan = build_selected_answer_plan(
        packet, reference, model_id=model_id,
        model_protocol_sha256=model_protocol_sha256,
        created_at=plan.get("created_at"),
        max_doc_chars=plan.get("max_doc_chars"),
        max_tokens=plan.get("max_tokens"))
    if _sha(plan) != _sha(rebuilt_plan) or plan.get("dataset") != dataset:
        raise ValueError("selected-answer report model/plan/scorer identity mismatch")
    reproduced = score_selected_answer_calibration(packet, reference, plan, call_records)
    # Canonical comparison also distinguishes bools from integers. Python's dict
    # equality would accept forged accounting values such as True == 1.
    if _sha(report) != _sha(reproduced):
        raise ValueError("selected-answer report does not reproduce from its recorded inputs and receipts")
    if reproduced.get("calibration_pass") is not True:
        raise ValueError("selected-answer report replay failed its frozen thresholds")

    from .grounding_calibration import runtime_scorer_sha256

    return {
        "verification_kind": "offline_selected_answer_report_replay",
        "report_reproduced": True,
        "thresholds_passed": True,
        "dataset": dataset,
        "task_contract": task_contract,
        "scope": scope,
        "report_path": str(report_path.resolve()),
        "report_file_sha256": expected_report_sha256,
        "report_content_sha256": report["report_sha256"],
        "packet_sha256": report["packet_sha256"],
        "reference_sha256": report["reference_sha256"],
        "plan_sha256": report["plan_sha256"],
        "model_id": report["model_id"],
        "model_protocol_sha256": report["model_protocol_sha256"],
        "scorer_implementation_sha256": report["scorer_implementation_sha256"],
        "runtime_binding": {
            "verified": False,
            "current_runtime_scorer_sha256": runtime_scorer_sha256(),
            "report_runtime_scorer_sha256": report.get("runtime_scorer_sha256"),
        },
        "production_attestation_eligible": False,
        "production_blockers": list(PRODUCTION_BLOCKERS),
        "model_calls_during_verification": 0,
    }
