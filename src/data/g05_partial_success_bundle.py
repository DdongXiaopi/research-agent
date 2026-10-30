"""Create a hash-bound, non-promotable bundle from individually valid G05 receipts.

This is deliberately separate from the complete top-up merge. It never changes
parent/child runs, claims, ledgers, or receipt files, and it marks the resulting
119-case packet as incomplete and calibration-ineligible until a later AI
reference-capacity check produces a separate decision.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from . import g05_task_authoring_topup as topup
from .g05_task_packet import _canonical_sha, _require_sha

BUNDLE_SCHEMA = "g05-partial-success-evidence-bundle-v1"
AUTHORING_INPUT_SCHEMA = "g05-partial-success-authoring-input-v1"
TASK_PACKET_SCHEMA = "g05-selected-answer-support-task-packet-v1"
EXPECTED_PARENT_VALID = 117
EXPECTED_PARENT_INVALID = 2
EXPECTED_PARENT_UNKNOWN = 1
EXPECTED_CHILD_VALID = 2
EXPECTED_CHILD_INVALID = 1


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_status_hash(status: dict[str, Any], *, label: str) -> str:
    status_sha = _require_sha(status.get("run_sha256"), f"{label} run hash is invalid")
    if _canonical_sha({key: value for key, value in status.items() if key != "run_sha256"}) != status_sha:
        raise ValueError(f"{label} run status hash mismatch")
    return status_sha


def _verify_historical_plan(plan: dict[str, Any]) -> None:
    """Verify an already-executed plan without granting current dispatch rights.

    This intentionally does not call the current-plan validator: that validator
    correctly rejects plans whose SPEC/protocol hashes are no longer current.
    Here those frozen document hashes are preserved as historical identity and
    compared with the current documents in the diagnostic bundle.
    """
    if not isinstance(plan, dict) or plan.get("schema") != topup.TOPUP_PLAN_SCHEMA:
        raise ValueError("unsupported historical top-up plan")
    plan_sha = _require_sha(plan.get("plan_sha256"), "historical plan hash is invalid")
    if _canonical_sha({key: value for key, value in plan.items() if key != "plan_sha256"}) != plan_sha:
        raise ValueError("historical top-up plan hash mismatch")
    cases, slots = plan.get("cases"), plan.get("replacement_slots")
    count = len(cases) if isinstance(cases, list) else -1
    if (plan.get("status") != "frozen_requires_hash_bound_approval"
            or plan.get("dispatch_authorized") is not False or plan.get("calls_allowed") is not False
            or plan.get("task_contract") != "selected_answer_support_v1"
            or plan.get("scope") != "single_doc"
            or plan.get("model_identity") != list(topup.MODEL_IDENTITY)
            or plan.get("provider_model_id") != topup.MODEL_ID
            or plan.get("provider") != "http" or plan.get("base_url") != topup.BASE_URL
            or plan.get("api_key_env") != topup.API_KEY_ENV
            or plan.get("paid_fallback") is not False or plan.get("cache_enabled") is not False
            or plan.get("max_cost_usd") != topup.MAX_COST_USD
            or count != 3 or not isinstance(slots, list) or len(slots) != count
            or plan.get("topup_logical_calls") != count
            or plan.get("max_logical_calls") != count
            or plan.get("max_physical_attempts") != count
            or plan.get("max_retries_per_call") != 0
            or plan.get("max_tokens_per_call") != topup.MAX_TOKENS_PER_CALL
            or plan.get("max_total_completion_tokens") != count * topup.MAX_TOKENS_PER_CALL
            or plan.get("max_input_chars_per_call") != topup.MAX_INPUT_CHARS_PER_CALL
            or plan.get("max_total_input_chars") != count * topup.MAX_INPUT_CHARS_PER_CALL
            or plan.get("lineage_original_plan_unchanged") is not True
            or plan.get("lineage_cumulative_physical_attempts_max") != 123):
        raise ValueError("historical plan violates its frozen scope or hard caps")
    for key in ("spec_sha256", "protocol_sha256", "topup_source_sha256", "provider_source_sha256",
                "parent_run_sha256", "parent_ledger_sha256", "parent_status_file_sha256",
                "parent_execution_plan_sha256", "parent_approval_sha256",
                "parent_authoring_plan_sha256", "sample_plan_sha256"):
        _require_sha(plan.get(key), f"historical plan {key} is invalid")
    if not all(isinstance(case, dict) for case in cases) or not all(isinstance(slot, dict) for slot in slots):
        raise ValueError("historical plan cases or replacement slots are malformed")
    if ({case.get("replaces_case_id") for case in cases} !=
            {slot.get("replaces_case_id") for slot in slots}
            or len({case.get("case_id") for case in cases}) != count
            or len({case.get("attempt_id") for case in cases}) != count
            or len({case.get("prompt_sha256") for case in cases}) != count):
        raise ValueError("historical plan slot coverage or identities are invalid")
    for case in cases:
        if (not isinstance(case, dict)
                or not isinstance(case.get("case_id"), str)
                or not case["case_id"].startswith("g05-topup-case-")
                or not isinstance(case.get("attempt_id"), str)
                or not case["attempt_id"].startswith("g05-topup-attempt-")
                or case.get("max_completion_tokens") != topup.MAX_TOKENS_PER_CALL
                or not isinstance(case.get("system"), str) or not isinstance(case.get("user"), str)
                or len(case["system"]) + len(case["user"]) > topup.MAX_INPUT_CHARS_PER_CALL
                or hashlib.sha256((case["system"] + "\0" + case["user"]).encode()).hexdigest()
                    != case.get("prompt_sha256")):
            raise ValueError("historical plan prompt or attempt identity is invalid")


def _verify_historical_approval(plan: dict[str, Any], approval: dict[str, Any]) -> None:
    if not isinstance(approval, dict) or approval.get("schema") != topup.TOPUP_APPROVAL_SCHEMA:
        raise ValueError("historical top-up approval is missing or malformed")
    approval_sha = _require_sha(approval.get("approval_sha256"), "historical approval hash is invalid")
    if _canonical_sha({key: value for key, value in approval.items()
                       if key != "approval_sha256"}) != approval_sha:
        raise ValueError("historical approval hash mismatch")
    try:
        stamp = datetime.fromisoformat(str(approval.get("approved_at", "")).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("historical approval timestamp is invalid") from exc
    if (stamp.tzinfo is None or approval.get("approved") is not True
            or not isinstance(approval.get("approved_by"), str)
            or not approval["approved_by"].strip()
            or approval.get("approval_statement") != topup.TOPUP_APPROVAL_STATEMENT):
        raise ValueError("historical approval does not contain an explicit approval identity")
    body = {key: value for key, value in approval.items() if key != "approval_sha256"}
    expected = {"schema": topup.TOPUP_APPROVAL_SCHEMA, "approved": True,
        "approved_by": approval["approved_by"].strip(), "approved_at": approval["approved_at"],
        "approval_statement": topup.TOPUP_APPROVAL_STATEMENT,
        "topup_plan_sha256": plan["plan_sha256"],
        "parent_run_sha256": plan["parent_run_sha256"],
        "parent_ledger_sha256": plan["parent_ledger_sha256"],
        "topup_run_dir": plan["topup_run_dir"], "model_identity": plan["model_identity"],
        "provider_model_id": topup.MODEL_ID,
        "max_logical_calls": plan["max_logical_calls"],
        "max_physical_attempts": plan["max_physical_attempts"], "max_retries_per_call": 0,
        "max_tokens_per_call": topup.MAX_TOKENS_PER_CALL,
        "max_total_completion_tokens": plan["max_total_completion_tokens"],
        "max_cost_usd": topup.MAX_COST_USD, "paid_fallback": False, "cache_enabled": False,
        "lineage_cumulative_physical_attempts_max": plan["lineage_cumulative_physical_attempts_max"]}
    if body != expected:
        raise ValueError("historical approval does not match the exact frozen plan and caps")


def _verify_child(topup_dir: Path, parent_snapshot: dict[str, Any], workspace: Path):
    plan = topup._read_json(topup_dir / "topup-plan.json")
    approval = topup._read_json(topup_dir / "approval.json")
    _verify_historical_plan(plan)
    _verify_historical_approval(plan, approval)
    if (plan["parent_run_sha256"] != parent_snapshot["status"]["run_sha256"]
            or plan["parent_ledger_sha256"] != parent_snapshot["status"]["call_ledger_sha256"]
            or plan["parent_status_file_sha256"] != _sha256_file(parent_snapshot["parent_dir"] / "status.json")
            or plan["parent_execution_plan_sha256"] != parent_snapshot["plan"]["plan_sha256"]
            or plan["topup_run_dir"] != topup._safe_relative(topup_dir, workspace, name="top-up run")):
        raise ValueError("top-up artifacts are not bound to this original parent run")
    if (_sha256_file(Path(topup.__file__) ) != plan.get("topup_source_sha256")
            or _sha256_file(Path(topup.__file__).resolve().parents[1] / "runtime" / "provider.py")
                != plan.get("provider_source_sha256")):
        raise ValueError("top-up or provider implementation changed after its frozen run")

    claim_dir = workspace / ".workbuddy" / "g05-topup-claims"
    claim_path = claim_dir / f"{plan['parent_run_sha256']}.json"
    claim = topup._read_json(claim_path)
    if (claim.get("parent_run_sha256") != plan["parent_run_sha256"]
            or claim.get("topup_plan_sha256") != plan["plan_sha256"]
            or claim.get("topup_run_dir") != plan["topup_run_dir"]
            or claim.get("claim_sha256") != _canonical_sha(
                {key: value for key, value in claim.items() if key != "claim_sha256"})):
        raise ValueError("one-shot top-up claim does not match the frozen child run")

    status_path = topup_dir / "status.json"
    status = topup._read_json(status_path)
    status_sha = _verify_status_hash(status, label="top-up")
    ledger_path = topup_dir / "call-ledger.jsonl"
    if (status.get("schema") != topup.TOPUP_RUN_SCHEMA
            or status.get("status") != "incomplete_or_invalid"
            or status.get("topup_plan_sha256") != plan["plan_sha256"]
            or status.get("approval_sha256") != approval["approval_sha256"]
            or status.get("test_only") is not False
            or status.get("paid_fallback") is not False
            or status.get("physical_attempts") != len(plan["cases"])
            or status.get("attempted_case_count") != len(plan["cases"])
            or status.get("max_physical_attempts") != len(plan["cases"])
            or not ledger_path.is_file()
            or _sha256_file(ledger_path) != status.get("call_ledger_sha256")):
        raise ValueError("top-up status or ledger is not the frozen complete-attempt partial run")

    events = topup._topup_events(ledger_path)
    terminal, started, orphaned = topup._topup_ledger_states(events)
    cases = {row["case_id"]: row for row in plan.get("cases", [])}
    if (orphaned or set(started) != set(cases) or set(terminal) != set(cases)
            or status.get("valid_count") != EXPECTED_CHILD_VALID
            or status.get("invalid_count") != EXPECTED_CHILD_INVALID
            or status.get("unknown_outcome_count") != 0
            or len(cases) != EXPECTED_CHILD_VALID + EXPECTED_CHILD_INVALID):
        raise ValueError("child ledger does not match the approved 2-valid/1-invalid attempt inventory")

    valid: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []
    for case_id, case in cases.items():
        shell = next((row for row in parent_snapshot["sample"]["task_shells"]
                      if row.get("case_id") == case.get("replaces_case_id")), None)
        if (shell is None or case.get("source_group_id", "").casefold()
                != shell.get("source_group_id", "").casefold()
                or case.get("source_doi", "").casefold() != shell.get("source_doi", "").casefold()
                or case.get("source_content_sha256") != shell.get("source_content_sha256")):
            raise ValueError("top-up case source identity differs from its original frozen sample shell")
        start = next(row for row in events if row.get("event") == "g05_topup_case_start"
                     and row.get("case_id") == case_id)
        end = terminal[case_id]
        if (start.get("topup_plan_sha256") != plan["plan_sha256"]
                or start.get("attempt_id") != case["attempt_id"]
                or start.get("prompt_sha256") != case["prompt_sha256"]
                or start.get("attempt") != 1
                or end.get("physical_attempts_reserved") != 1):
            raise ValueError("child ledger case identity or one-attempt reservation mismatch")
        path = topup_dir / "cases" / f"{case_id}.json"
        if end.get("status") == "valid":
            receipt_sha = _require_sha(end.get("result_sha256"), "valid child receipt hash is missing")
            if not path.is_file() or _sha256_file(path) != receipt_sha:
                raise ValueError("valid child receipt is missing or changed")
            receipt = topup._read_json(path)
            parsed, error = topup._strict_task_json(receipt.get("raw_output"))
            ledger_end = end
            if (receipt.get("schema") != "g05-task-authoring-topup-receipt-v1"
                    or receipt.get("status") != "valid" or receipt.get("test_only") is not False
                    or receipt.get("case_id") != case_id
                    or receipt.get("attempt_id") != case["attempt_id"]
                    or receipt.get("replaces_case_id") != case["replaces_case_id"]
                    or receipt.get("source_group_id") != case["source_group_id"]
                    or receipt.get("source_content_sha256") != case["source_content_sha256"]
                    or receipt.get("prompt_sha256") != case["prompt_sha256"]
                    or error is not None or parsed != receipt.get("parsed_response")
                    or receipt.get("raw_output_sha256") != _canonical_sha(receipt.get("raw_output"))
                    or receipt.get("response_model_id") != topup.MODEL_ID
                    or receipt.get("http_attempts") != 1
                    or receipt.get("provider_model_id") != topup.MODEL_ID
                    or receipt.get("reason") is not None
                    or receipt.get("cost", {}).get("usd") != 0.0
                    or receipt.get("cost", {}).get("cost_estimated") is not False
                    or receipt.get("cost", {}).get("usage_complete") is not True
                    or ledger_end.get("cost_usd") != 0.0
                    or not isinstance(receipt.get("prompt_tokens"), int)
                    or receipt.get("prompt_tokens") < 0
                    or not isinstance(receipt.get("completion_tokens"), int)
                    or receipt.get("completion_tokens") < 0):
                raise ValueError("valid child receipt fails authoring identity, schema, or usage checks")
            valid[case_id] = {"case": case, "receipt": receipt,
                "receipt_sha256": receipt_sha, "parsed": parsed}
        elif end.get("status") == "invalid":
            receipt_sha = _require_sha(end.get("result_sha256"), "invalid child receipt hash is missing")
            if not path.is_file() or _sha256_file(path) != receipt_sha:
                raise ValueError("invalid child receipt is missing or changed")
            receipt = topup._read_json(path)
            if (receipt.get("schema") != "g05-task-authoring-topup-receipt-v1"
                    or receipt.get("status") != "invalid" or receipt.get("test_only") is not False
                    or receipt.get("case_id") != case_id
                    or receipt.get("attempt_id") != case["attempt_id"]
                    or receipt.get("replaces_case_id") != case["replaces_case_id"]
                    or receipt.get("prompt_sha256") != case["prompt_sha256"]
                    or receipt.get("raw_output_sha256") != _canonical_sha(receipt.get("raw_output"))):
                raise ValueError("invalid child receipt failed identity or raw-output integrity checks")
            excluded.append({"origin": "topup", "case_id": case["replaces_case_id"],
                "attempt_id": case["attempt_id"], "topup_case_id": case_id,
                "source_group_id": case["source_group_id"], "status": "invalid",
                "reason": end.get("reason"), "receipt_sha256": receipt_sha,
                "raw_output_sha256": receipt["raw_output_sha256"],
                "raw_response_reused": False})
        else:
            raise ValueError("child run contains unsupported terminal status")

    return plan, approval, status, status_sha, _sha256_file(status_path), ledger_path, events, terminal, valid, excluded


def _parent_exclusions(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    plan_cases = {row["case_id"]: row for row in snapshot["authoring"]["cases"]}
    excluded: list[dict[str, Any]] = []
    for case_id, end in snapshot["failed"].items():
        case = plan_cases[case_id]
        row = {"origin": "parent", "case_id": case_id,
            "attempt_id": f"{case_id}:attempt-1", "source_group_id": case["source_group_id"],
            "status": end["status"], "reason": end.get("reason"), "receipt_sha256": end.get("result_sha256"),
            "raw_response_reused": False}
        if end["status"] == "invalid":
            receipt = topup._load_parent_receipt(snapshot["parent_dir"], case_id, end, case,
                require_valid=False)
            row["raw_output_sha256"] = receipt.get("raw_output_sha256")
        excluded.append(row)
    return excluded


def _packet_hash(packet: dict[str, Any]) -> str:
    return _canonical_sha({key: value for key, value in packet.items() if key != "task_packet_sha256"})


def build_g05_partial_success_bundle(
    parent_run_dir: Path, topup_run_dir: Path, *, workspace: Path,
) -> dict[str, Any]:
    """Verify and materialize only 117 parent + 2 child valid task receipts."""
    workspace = Path(workspace).resolve()
    parent_dir = Path(parent_run_dir).resolve()
    topup_dir = Path(topup_run_dir).resolve()
    snapshot = topup._load_parent_snapshot(parent_dir, workspace)
    topup._safe_relative(topup_dir, workspace, name="top-up run")
    if (len(snapshot["valid"]) != EXPECTED_PARENT_VALID
            or sum(row.get("status") == "invalid" for row in snapshot["failed"].values()) != EXPECTED_PARENT_INVALID
            or sum(row.get("status") == "failed_unknown_outcome" for row in snapshot["failed"].values())
                != EXPECTED_PARENT_UNKNOWN):
        raise ValueError("parent run must be the frozen 117-valid/2-invalid/1-unknown batch")
    child_plan, child_approval, child_status, child_status_sha, child_status_file_sha, child_ledger_path, child_events, child_terminal, child_valid, child_excluded = _verify_child(topup_dir, snapshot, workspace)
    exclusions = _parent_exclusions(snapshot) + child_excluded
    if (len(exclusions) != 4
            or sum(row["status"] == "invalid" for row in exclusions) != 3
            or sum(row["status"] == "failed_unknown_outcome" for row in exclusions) != 1):
        raise ValueError("partial-success evidence must explicitly exclude 3 invalid and 1 unknown attempts")

    shell_by_id = {row["case_id"]: row for row in snapshot["sample"]["task_shells"]}
    parent_valid = snapshot["valid"]
    topup_by_parent = {row["case"]["replaces_case_id"]: row for row in child_valid.values()}
    if len(topup_by_parent) != EXPECTED_CHILD_VALID:
        raise ValueError("two valid top-up receipts must replace two distinct parent slots")
    rows = []
    evidence_receipts = []
    for case_id, shell in shell_by_id.items():
        parent_case = next(row for row in snapshot["authoring"]["cases"] if row["case_id"] == case_id)
        if case_id in parent_valid:
            parsed = parent_valid[case_id]["parsed_response"]
            event = next(row for row in snapshot["events"] if row.get("event") == "g05_authoring_case_end"
                and row.get("case_id") == case_id)
            receipt_sha = event["result_sha256"]
            origin = "parent"
            attempt_id = f"{case_id}:attempt-1"
        elif case_id in topup_by_parent:
            item = topup_by_parent[case_id]
            parsed = item["parsed"]
            receipt_sha = item["receipt_sha256"]
            origin = "topup"
            attempt_id = item["case"]["attempt_id"]
        else:
            continue
        if parent_case["source_group_id"] != shell["source_group_id"]:
            raise ValueError("valid authoring response source differs from its frozen sample shell")
        rows.append({"case_id": case_id, "source_group_id": shell["source_group_id"],
            "source_doi": shell["source_doi"], "source_content_sha256": shell["source_content_sha256"],
            "question": parsed["question"], "selected_answer": parsed["selected_answer"]})
        evidence_receipts.append({"origin": origin, "case_id": case_id,
            "attempt_id": attempt_id, "source_group_id": shell["source_group_id"],
            "receipt_sha256": receipt_sha, "status": "valid"})
    if (len(rows) != EXPECTED_PARENT_VALID + EXPECTED_CHILD_VALID
            or len({row["source_group_id"].casefold() for row in rows}) != len(rows)):
        raise ValueError("partial-success task inventory must have exactly 119 unique DOI groups")

    authoring_input_body = {"schema": AUTHORING_INPUT_SCHEMA,
        "source_sample_plan_sha256": snapshot["sample"]["plan_sha256"],
        "complete_120": False, "calibration_eligible": False,
        "cases": rows, "receipt_manifest": evidence_receipts,
        "excluded_receipts": exclusions}
    authoring_input = {**authoring_input_body,
        "authoring_input_sha256": _canonical_sha(authoring_input_body)}

    receipt_manifest_sha = _canonical_sha({"success": evidence_receipts, "excluded": exclusions})
    packet_cases = []
    row_by_id = {row["case_id"]: row for row in rows}
    for shell in snapshot["sample"]["task_shells"]:
        authored = row_by_id.get(shell["case_id"])
        if authored is None:
            continue
        packet_cases.append({"case_id": shell["case_id"],
            "selection_role": shell["selection_role"],
            "task_contract": "selected_answer_support_v1", "scope": "single_doc",
            "source_group_id": shell["source_group_id"], "source_doi": shell["source_doi"],
            "source_content_sha256": shell["source_content_sha256"],
            "question": authored["question"], "selected_answer": authored["selected_answer"],
            "cited_document": {"title": shell["cited_document"]["title"],
                "abstract": shell["cited_document"]["abstract"]}})
    packet_body = {"schema": "g05-selected-answer-support-task-packet-v1",
        "task_contract": "selected_answer_support_v1", "scope": "single_doc",
        "source_sample_plan_sha256": snapshot["sample"]["plan_sha256"],
        "authoring_input_sha256": authoring_input["authoring_input_sha256"],
        "source_receipt_manifest_sha256": receipt_manifest_sha,
        "case_count": len(packet_cases), "source_groups_are_unique": True,
        "reference_labels_generated": False, "reference_origin": "pending_independent_ai_reference",
        "target_scorer_outputs_generated": False, "target_scorer_calls": 0, "model_calls": 0,
        "complete_120": False, "calibration_eligible": False,
        "scorer_calls_authorized": False, "cases": packet_cases}
    task_packet = {**packet_body, "task_packet_sha256": _packet_hash(packet_body)}

    spec_path = topup.PROJECT_ROOT / topup.SPEC_RELATIVE_PATH
    protocol_path = topup.PROJECT_ROOT / topup.PROTOCOL_RELATIVE_PATH
    current_spec_sha = _sha256_file(spec_path)
    current_protocol_sha = _sha256_file(protocol_path)
    body = {"schema": BUNDLE_SCHEMA, "status": "partial_success_diagnostic_only",
        "parent_run_sha256": snapshot["status"]["run_sha256"],
        "parent_ledger_sha256": snapshot["status"]["call_ledger_sha256"],
        "child_plan_sha256": child_plan["plan_sha256"],
        "child_run_sha256": child_status_sha,
        "child_status_file_sha256": child_status_file_sha,
        "child_ledger_sha256": child_status["call_ledger_sha256"],
        "child_batch_status": child_status["status"],
        "child_approval_sha256": child_approval["approval_sha256"],
        "child_claim_sha256": topup._read_json(
            workspace / ".workbuddy" / "g05-topup-claims" /
            f"{child_plan['parent_run_sha256']}.json")["claim_sha256"],
        "sample_plan_sha256": snapshot["sample"]["plan_sha256"],
        "source_pool_sha256": snapshot["sample"]["source_pool_sha256"],
        "publication_status_report_sha256": snapshot["sample"]["publication_status_report_sha256"],
        "frozen_spec_sha256": child_plan["spec_sha256"],
        "frozen_protocol_sha256": child_plan["protocol_sha256"],
        "current_spec_sha256": current_spec_sha,
        "current_protocol_sha256": current_protocol_sha,
        "frozen_documents_match_current":
            child_plan["spec_sha256"] == current_spec_sha
            and child_plan["protocol_sha256"] == current_protocol_sha,
        "success_count": len(rows), "distinct_doi_groups": len(rows),
        "complete_120": False, "calibration_eligible": False,
        "scorer_calls_authorized": False, "scorer_packet_generated": False,
        "model_calls_during_bundle": 0,
        "valid_receipts": evidence_receipts, "excluded_receipts": exclusions,
        "authoring_input": authoring_input, "task_packet": task_packet,
        "source_receipt_manifest_sha256": receipt_manifest_sha,
        "topup_status_preserved_incomplete": True,
        "old_invalid_response_reused": False,
        "topup_physical_attempts": child_status["physical_attempts"],
        "lineage_cumulative_physical_attempts":
            snapshot["status"]["physical_attempts"] + child_status["physical_attempts"],
        "lineage_cumulative_physical_attempts_max": child_plan["lineage_cumulative_physical_attempts_max"],
        "capacity_gate": {"minimum_resolved_cases_per_class": 30,
            "minimum_distinct_doi_groups_per_class": 30,
            "must_validate_with_ai_reference_capacity_gate": True}}
    bundle = {**body, "bundle_sha256": _canonical_sha(body)}
    verify_g05_partial_success_bundle(bundle)
    return bundle


def verify_g05_partial_success_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(bundle, dict) or bundle.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("unsupported G05 partial-success bundle")
    bundle_sha = _require_sha(bundle.get("bundle_sha256"), "bundle hash is missing or invalid")
    if _canonical_sha({key: value for key, value in bundle.items() if key != "bundle_sha256"}) != bundle_sha:
        raise ValueError("partial-success bundle hash mismatch")
    if (bundle.get("status") != "partial_success_diagnostic_only"
            or bundle.get("success_count") != 119
            or bundle.get("distinct_doi_groups") != 119
            or bundle.get("complete_120") is not False
            or bundle.get("calibration_eligible") is not False
            or bundle.get("scorer_calls_authorized") is not False
            or bundle.get("scorer_packet_generated") is not False
            or bundle.get("topup_status_preserved_incomplete") is not True
            or bundle.get("old_invalid_response_reused") is not False
            or bundle.get("topup_physical_attempts") != 3
            or bundle.get("lineage_cumulative_physical_attempts") != 123
            or bundle.get("model_calls_during_bundle") != 0):
        raise ValueError("partial-success bundle status or no-promotion flags are invalid")
    for key in ("frozen_spec_sha256", "frozen_protocol_sha256",
                "current_spec_sha256", "current_protocol_sha256"):
        _require_sha(bundle.get(key), f"partial bundle {key} is invalid")
    if (type(bundle.get("frozen_documents_match_current")) is not bool
            or bundle["frozen_documents_match_current"] != (
                bundle["frozen_spec_sha256"] == bundle["current_spec_sha256"]
                and bundle["frozen_protocol_sha256"] == bundle["current_protocol_sha256"])):
        raise ValueError("partial bundle historical/current document comparison is missing")
    authoring = bundle.get("authoring_input")
    packet = bundle.get("task_packet")
    if not isinstance(authoring, dict) or not isinstance(packet, dict):
        raise ValueError("partial-success bundle artifacts are missing")
    auth_body = {key: value for key, value in authoring.items() if key != "authoring_input_sha256"}
    if (authoring.get("schema") != AUTHORING_INPUT_SCHEMA
            or authoring.get("complete_120") is not False
            or authoring.get("calibration_eligible") is not False
            or authoring.get("authoring_input_sha256") != _canonical_sha(auth_body)
            or len(authoring.get("cases", [])) != 119
            or authoring.get("source_sample_plan_sha256") != bundle.get("sample_plan_sha256")):
        raise ValueError("partial authoring input identity, hash, or eligibility is invalid")
    packet_body = {key: value for key, value in packet.items() if key != "task_packet_sha256"}
    if (packet.get("schema") != TASK_PACKET_SCHEMA
            or packet.get("task_packet_sha256") != _canonical_sha(packet_body)
            or packet.get("case_count") != 119 or len(packet.get("cases", [])) != 119
            or packet.get("complete_120") is not False
            or packet.get("calibration_eligible") is not False
            or packet.get("scorer_calls_authorized") is not False
            or packet.get("source_sample_plan_sha256") != bundle.get("sample_plan_sha256")
            or packet.get("target_scorer_outputs_generated") is not False
            or packet.get("target_scorer_calls") != 0):
        raise ValueError("partial task packet hash, schema, or no-scorer flags are invalid")
    cases = packet["cases"]
    if (not all(isinstance(row, dict) for row in cases)
            or len({row.get("case_id") for row in cases}) != 119
            or len({str(row.get("source_group_id", "")).casefold() for row in cases}) != 119
            or any(row.get("reference_label") is not None or "target_scorer_output" in row for row in cases)):
        raise ValueError("partial task packet cases are duplicate or contain labels/target output")
    excluded = bundle.get("excluded_receipts")
    if (not isinstance(excluded, list) or len(excluded) != 4
            or not all(isinstance(row, dict) for row in excluded)
            or sum(row.get("status") == "invalid" for row in excluded) != 3
            or sum(row.get("status") == "failed_unknown_outcome" for row in excluded) != 1
            or any("raw_output" in row or "raw_response" in row for row in excluded)):
        raise ValueError("partial bundle exclusion evidence is incomplete or leaks failed response bodies")
    valid_receipts = bundle.get("valid_receipts")
    if (not isinstance(valid_receipts, list) or len(valid_receipts) != 119
            or sum(row.get("origin") == "parent" for row in valid_receipts) != EXPECTED_PARENT_VALID
            or sum(row.get("origin") == "topup" for row in valid_receipts) != EXPECTED_CHILD_VALID
            or len({row.get("source_group_id", "").casefold() for row in valid_receipts}) != 119
            or any(not _valid_receipt_row(row) for row in valid_receipts)):
        raise ValueError("partial bundle valid receipt manifest is incomplete or duplicated")
    manifest_sha = _canonical_sha({"success": valid_receipts, "excluded": excluded})
    if (bundle.get("source_receipt_manifest_sha256") != manifest_sha
            or packet.get("source_receipt_manifest_sha256") != manifest_sha
            or authoring.get("receipt_manifest") != valid_receipts
            or authoring.get("excluded_receipts") != excluded):
        raise ValueError("partial bundle receipt manifest hash or bindings mismatch")
    authoring_by_id = {row.get("case_id"): row for row in authoring["cases"]
        if isinstance(row, dict)}
    packet_by_id = {row.get("case_id"): row for row in cases if isinstance(row, dict)}
    receipts_by_id = {row["case_id"]: row for row in valid_receipts}
    if (set(authoring_by_id) != set(packet_by_id) or set(packet_by_id) != set(receipts_by_id)
            or any(authoring_by_id[case_id].get("source_group_id", "").casefold()
                   != packet_by_id[case_id].get("source_group_id", "").casefold()
                   or authoring_by_id[case_id].get("question") != packet_by_id[case_id].get("question")
                   or authoring_by_id[case_id].get("selected_answer") != packet_by_id[case_id].get("selected_answer")
                   or receipts_by_id[case_id].get("source_group_id", "").casefold()
                   != packet_by_id[case_id].get("source_group_id", "").casefold()
                   for case_id in packet_by_id)):
        raise ValueError("partial authoring input, packet and successful receipt manifest differ")
    return {"bundle_sha256": bundle_sha, "success_count": 119,
        "distinct_doi_groups": 119, "excluded_count": 4,
        "complete_120": False, "calibration_eligible": False,
        "scorer_calls_authorized": False}


def _valid_receipt_row(row: Any) -> bool:
    expected = {"origin", "case_id", "attempt_id", "source_group_id", "receipt_sha256", "status"}
    return (isinstance(row, dict) and set(row) == expected
        and row.get("origin") in {"parent", "topup"}
        and isinstance(row.get("case_id"), str) and bool(row["case_id"])
        and isinstance(row.get("attempt_id"), str) and bool(row["attempt_id"])
        and isinstance(row.get("source_group_id"), str) and bool(row["source_group_id"])
        and row.get("status") == "valid"
        and isinstance(row.get("receipt_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", row["receipt_sha256"]) is not None)
