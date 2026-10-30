"""Hash-bound, one-shot continuation for failed G05 authoring slots.

The original authoring run and its 120-request ledger remain immutable. This
module creates a separately approved child plan for at most three replacement
slots, with new case/attempt identities and a new ledger. Old failed response
bodies are never copied into replacement prompts or merged as task data.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .g05_task_authoring_execution import (
    API_KEY_ENV, BASE_URL, MODEL_ID, MODEL_IDENTITY, MAX_CALLS,
    MAX_INPUT_CHARS_PER_CALL, MAX_TOKENS_PER_CALL, MAX_COST_USD,
    _append_jsonl, _canonical_sha, _check_frozen_sources, _file_sha256,
    _ledger_states, _read_ledger, _run_lock, _save_once_json,
    _strict_task_json, _verify_price_evidence, _verified_authoring_plan,
    _write_once, _write_status,
    verify_execution_plan, verify_g05_authoring_approval,
)
from .g05_task_packet import _require_sha
from .g05_task_authoring_execution import PRICE_SOURCE

TOPUP_PLAN_SCHEMA = "g05-task-authoring-topup-plan-v1"
TOPUP_APPROVAL_SCHEMA = "g05-task-authoring-topup-approval-v1"
TOPUP_RUN_SCHEMA = "g05-task-authoring-topup-run-v1"
TOPUP_MERGE_SCHEMA = "g05-task-authoring-topup-merge-v1"
TOPUP_APPROVAL_STATEMENT = "I_APPROVE_G05_AUTHORING_TOPUP_CALLS"
MAX_TOPUP_CALLS = 3
MAX_TOPUP_COMPLETION_TOKENS = MAX_TOPUP_CALLS * MAX_TOKENS_PER_CALL
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC_RELATIVE_PATH = "RSI自进化项目_SPEC_v1.2.md"
PROTOCOL_RELATIVE_PATH = "docs/G05-SELECTED-ANSWER-SUPPORT-REFERENCE-PROTOCOL-2026-10-16.md"


def _protocol_document_hashes() -> dict[str, str]:
    return {"spec_sha256": _file_sha256(PROJECT_ROOT / SPEC_RELATIVE_PATH),
        "protocol_sha256": _file_sha256(PROJECT_ROOT / PROTOCOL_RELATIVE_PATH)}


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"G05 top-up artifact is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"G05 top-up artifact must be an object: {path.name}")
    return value


def _safe_relative(path: Path, workspace: Path, *, name: str) -> str:
    resolved = path.resolve()
    root = workspace.resolve()
    try:
        relative = resolved.relative_to((root / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError(f"{name} must stay inside workspace .workbuddy") from exc
    if not relative.parts:
        raise ValueError(f"{name} cannot be workspace .workbuddy itself")
    return relative.as_posix()


def _verify_parent_status(status: dict, status_path: Path, ledger_path: Path,
                          plan: dict, authoring: dict, sample: dict,
                          approval: dict) -> tuple[list[dict], dict[str, dict], dict[str, dict], set[str]]:
    status_sha = _require_sha(status.get("run_sha256"), "parent run status hash is invalid")
    if _canonical_sha({key: value for key, value in status.items()
                       if key != "run_sha256"}) != status_sha:
        raise ValueError("parent run status hash mismatch")
    if (status.get("schema") != "g05-task-authoring-execution-run-v1"
            or status.get("status") != "incomplete_or_invalid"
            or status.get("execution_plan_sha256") != plan.get("plan_sha256")
            or status.get("approval_sha256") != approval.get("approval_sha256")
            or status.get("authoring_plan_sha256") != authoring.get("plan_sha256")
            or status.get("sample_plan_sha256") != sample.get("plan_sha256")
            or status.get("test_only") is not False
            or status.get("paid_fallback") is not False):
        raise ValueError("parent run is not an eligible non-test incomplete G05 batch")
    ledger_sha = _require_sha(status.get("call_ledger_sha256"), "parent ledger hash is missing")
    if not ledger_path.is_file() or _file_sha256(ledger_path) != ledger_sha:
        raise ValueError("parent ledger hash does not match status")
    events = _read_ledger(ledger_path)
    terminal, started, orphaned = _ledger_states(events)
    cases = {row["case_id"]: row for row in _verified_authoring_plan(authoring, sample)}
    if (len(started) != MAX_CALLS or set(started) != set(cases)
            or orphaned or set(terminal) != set(cases)
            or status.get("attempted_case_count") != MAX_CALLS
            or status.get("physical_attempts") != MAX_CALLS
            or status.get("max_physical_attempts") != MAX_CALLS):
        raise ValueError("parent run is not a closed 120-case ledger")
    valid: dict[str, dict] = {}
    failed: dict[str, dict] = {}
    for case_id, call in cases.items():
        start = next(row for row in events if row.get("event") == "g05_authoring_case_start"
                     and row.get("case_id") == case_id)
        if (start.get("execution_plan_sha256") != plan["plan_sha256"]
                or start.get("prompt_sha256") != call["prompt_sha256"]
                or start.get("attempt") != 1):
            raise ValueError("parent case start does not match frozen plan")
        end = terminal[case_id]
        state = end.get("status")
        if state == "valid":
            receipt = _load_parent_receipt(parent_dir=ledger_path.parent,
                case_id=case_id, end=end, case=call)
            valid[case_id] = receipt
        elif state in {"invalid", "failed_unknown_outcome"}:
            if end.get("physical_attempts_reserved") != 1:
                raise ValueError("failed parent slot has invalid physical reservation")
            if state == "invalid":
                _load_parent_receipt(parent_dir=ledger_path.parent,
                    case_id=case_id, end=end, case=call, require_valid=False)
            elif end.get("result_sha256") is not None:
                raise ValueError("unknown parent outcome cannot contain a response receipt")
            failed[case_id] = end
        else:
            raise ValueError("parent ledger contains an unsupported terminal state")
    if (len(valid) != status.get("valid_count")
            or sum(row.get("status") == "invalid" for row in failed.values()) != status.get("invalid_count")
            or sum(row.get("status") == "failed_unknown_outcome" for row in failed.values())
                != status.get("unknown_outcome_count")
            or len(valid) + len(failed) != MAX_CALLS
            or not 1 <= len(failed) <= MAX_TOPUP_CALLS):
        raise ValueError("parent status counts disagree with its closed ledger or top-up cap")
    if len(valid) + len(failed) != MAX_CALLS:
        raise ValueError("parent batch does not preserve the original 120-case cohort")
    return events, valid, failed, started


def _load_parent_receipt(parent_dir: Path, case_id: str, end: dict, case: dict,
                         *, require_valid: bool = True) -> dict:
    result_sha = _require_sha(end.get("result_sha256"), "parent result receipt hash is missing")
    path = parent_dir / "cases" / f"{case_id}.json"
    if not path.is_file() or _file_sha256(path) != result_sha:
        raise ValueError("parent case receipt is missing or changed")
    receipt = _read_json(path)
    if (receipt.get("case_id") != case_id
            or receipt.get("prompt_sha256") != case.get("prompt_sha256")
            or receipt.get("source_group_id") != case.get("source_group_id")
            or receipt.get("source_content_sha256") != case.get("source_content_sha256")
            or receipt.get("test_only") is not False):
        raise ValueError("parent case receipt identity does not match its frozen case")
    if not require_valid:
        if receipt.get("status") != "invalid":
            raise ValueError("failed parent slot receipt is not marked invalid")
        return receipt
    parsed, error = _strict_task_json(receipt.get("raw_output"))
    if (receipt.get("status") != "valid" or error is not None
            or parsed != receipt.get("parsed_response")
            or receipt.get("raw_output_sha256") != _canonical_sha(receipt.get("raw_output"))
            or receipt.get("raw_response_sha256") != _canonical_sha(parsed)
            or receipt.get("response_model_id") != MODEL_ID
            or receipt.get("http_attempts") != 1
            or not isinstance(receipt.get("call_record"), dict)
            or receipt["call_record"].get("status") != "succeeded"
            or receipt["call_record"].get("model_id") != MODEL_ID
            or receipt["call_record"].get("request_sha256") != case.get("prompt_sha256")
            or receipt["call_record"].get("physical_attempts") != 1
            or receipt["call_record"].get("cost", {}).get("usd") != 0.0
            or receipt.get("cost", {}).get("usd") != 0.0
            or receipt.get("cost", {}).get("cost_estimated") is not False
            or receipt.get("cost", {}).get("usage_complete") is not True):
        raise ValueError("parent valid receipt does not pass authoring output validation")
    return receipt


def _load_parent_snapshot(parent_run_dir: Path, workspace: Path) -> dict:
    parent_dir = Path(parent_run_dir).resolve()
    _safe_relative(parent_dir, workspace, name="parent run")
    plan = _read_json(parent_dir / "execution-plan.json")
    approval = _read_json(parent_dir / "approval.json")
    authoring = _read_json(parent_dir / "authoring-plan.json")
    sample = _read_json(parent_dir / "sample-plan.json")
    status_path = parent_dir / "status.json"
    status = _read_json(status_path)
    verify_execution_plan(plan)
    verify_g05_authoring_approval(plan, approval)
    if (plan.get("authoring_plan_sha256") != authoring.get("plan_sha256")
            or plan.get("sample_plan_sha256") != sample.get("plan_sha256")):
        raise ValueError("parent execution plan is not bound to its source artifacts")
    _check_frozen_sources(plan)
    price_path = workspace / plan["price_evidence"]["report_path"]
    if _verify_price_evidence(price_path, workspace) != plan["price_evidence"]:
        raise ValueError("parent zero-price evidence changed or expired")
    events, valid, failed, started = _verify_parent_status(status, status_path,
        parent_dir / "call-ledger.jsonl", plan, authoring, sample, approval)
    return {"parent_dir": parent_dir, "plan": plan, "approval": approval,
        "authoring": authoring, "sample": sample, "status": status,
        "events": events, "valid": valid, "failed": failed, "started": started}


def _topup_case_id(parent_run_sha: str, seed: str, parent_case_id: str) -> tuple[str, str]:
    attempt_id = "g05-topup-attempt-" + hashlib.sha256(
        f"{parent_run_sha}\0{seed}\0{parent_case_id}\0attempt".encode()).hexdigest()[:24]
    case_id = "g05-topup-case-" + hashlib.sha256(
        f"{parent_run_sha}\0{seed}\0{parent_case_id}\0case".encode()).hexdigest()[:24]
    return case_id, attempt_id


def build_g05_task_authoring_topup_plan(
    parent_run_dir: Path, *, seed: str, run_dir: Path, workspace: Path,
    _allow_existing_run_dir: bool = False,
) -> dict:
    """Freeze a zero-dispatch child plan for only failed/invalid parent slots."""
    workspace = Path(workspace).resolve()
    snapshot = _load_parent_snapshot(Path(parent_run_dir), workspace)
    if not isinstance(seed, str) or not seed.strip():
        raise ValueError("top-up seed is required")
    run_relative = _safe_relative(Path(run_dir), workspace, name="top-up run directory")
    topup_run_dir = workspace / ".workbuddy" / Path(run_relative)
    if topup_run_dir.exists() and not _allow_existing_run_dir:
        raise FileExistsError("top-up run directory already exists; use a new frozen plan")
    parent_sha = snapshot["status"]["run_sha256"]
    shells = {row["case_id"]: row for row in snapshot["sample"]["task_shells"]}
    authoring_cases = {row["case_id"]: row for row in snapshot["authoring"]["cases"]}
    topup_cases, slots = [], []
    for parent_case_id in sorted(snapshot["failed"]):
        failed_row = snapshot["failed"][parent_case_id]
        old_case = authoring_cases[parent_case_id]
        shell = shells[parent_case_id]
        case_id, attempt_id = _topup_case_id(parent_sha, seed, parent_case_id)
        system = old_case["system"]
        user = old_case["user"] + (
            "\n\nFresh replacement authoring identity: " + attempt_id + ". "
            "This is a new task-authoring request. Create a fresh question and selected_answer "
            "from only the title and abstract above; do not continue or repair any prior response. "
            "Return exactly the required JSON fields and no correctness or evidence-support label."
        )
        if len(system) + len(user) > MAX_INPUT_CHARS_PER_CALL:
            raise ValueError("replacement authoring prompt exceeds frozen input cap")
        prompt_sha = hashlib.sha256((system + "\0" + user).encode("utf-8")).hexdigest()
        topup_cases.append({"case_id": case_id, "attempt_id": attempt_id,
            "replaces_case_id": parent_case_id,
            "source_group_id": old_case["source_group_id"],
            "source_doi": old_case["source_doi"],
            "source_content_sha256": old_case["source_content_sha256"],
            "authoring_route": old_case["authoring_route"],
            "system": system, "user": user, "prompt_sha256": prompt_sha,
            "max_completion_tokens": MAX_TOKENS_PER_CALL})
        prior_receipt_sha = failed_row.get("result_sha256")
        slots.append({"replaces_case_id": parent_case_id,
            "prior_status": failed_row["status"], "prior_reason": failed_row.get("reason"),
            "prior_receipt_sha256": prior_receipt_sha,
            "source_group_id": shell["source_group_id"],
            "source_content_sha256": shell["source_content_sha256"]})
    if len(topup_cases) != len(snapshot["failed"]):
        raise ValueError("top-up did not create exactly one new case per failed parent slot")
    body = {"schema": TOPUP_PLAN_SCHEMA,
        "status": "frozen_requires_hash_bound_approval",
        "dispatch_authorized": False, "calls_allowed": False,
        "task_contract": "selected_answer_support_v1", "scope": "single_doc",
        "seed": seed, "parent_run_dir": _safe_relative(snapshot["parent_dir"], workspace,
            name="parent run"),
        "topup_run_dir": run_relative,
        "parent_execution_plan_sha256": snapshot["plan"]["plan_sha256"],
        "parent_approval_sha256": snapshot["approval"]["approval_sha256"],
        "parent_authoring_plan_sha256": snapshot["authoring"]["plan_sha256"],
        "sample_plan_sha256": snapshot["sample"]["plan_sha256"],
        "parent_status_file_sha256": _file_sha256(snapshot["parent_dir"] / "status.json"),
        "parent_run_sha256": parent_sha,
        "parent_ledger_sha256": snapshot["status"]["call_ledger_sha256"],
        "parent_case_count": MAX_CALLS,
        "parent_valid_count": len(snapshot["valid"]),
        "replacement_slots": slots,
        "parent_valid_receipts": [{"case_id": case_id,
            "receipt_sha256": _file_sha256(snapshot["parent_dir"] / "cases" / f"{case_id}.json")}
            for case_id in sorted(snapshot["valid"])],
        "cases": topup_cases,
        "model_identity": list(MODEL_IDENTITY), "provider_model_id": MODEL_ID,
        "provider": "http", "base_url": BASE_URL, "api_key_env": API_KEY_ENV,
        "price_evidence": snapshot["plan"]["price_evidence"],
        "topup_source_sha256": _file_sha256(Path(__file__)),
        "provider_source_sha256": _file_sha256(
            Path(__file__).resolve().parents[1] / "runtime" / "provider.py"),
        **_protocol_document_hashes(),
        "paid_fallback": False, "cache_enabled": False,
        "topup_logical_calls": len(topup_cases),
        "max_logical_calls": len(topup_cases),
        "max_physical_attempts": len(topup_cases),
        "max_retries_per_call": 0,
        "max_tokens_per_call": MAX_TOKENS_PER_CALL,
        "max_total_completion_tokens": len(topup_cases) * MAX_TOKENS_PER_CALL,
        "max_input_chars_per_call": MAX_INPUT_CHARS_PER_CALL,
        "max_total_input_chars": len(topup_cases) * MAX_INPUT_CHARS_PER_CALL,
        "max_cost_usd": MAX_COST_USD,
        "lineage_original_physical_attempts": snapshot["status"]["physical_attempts"],
        "lineage_cumulative_physical_attempts_max": (
            snapshot["status"]["physical_attempts"] + len(topup_cases)),
        "lineage_original_plan_unchanged": True,
        "planning_calls": 0}
    return {**body, "plan_sha256": _canonical_sha(body)}


def verify_g05_task_authoring_topup_plan(plan: dict) -> None:
    if not isinstance(plan, dict) or plan.get("schema") != TOPUP_PLAN_SCHEMA:
        raise ValueError("unsupported G05 top-up plan")
    plan_sha = _require_sha(plan.get("plan_sha256"), "top-up plan hash is invalid")
    if _canonical_sha({key: value for key, value in plan.items()
                       if key != "plan_sha256"}) != plan_sha:
        raise ValueError("top-up plan hash mismatch")
    cases, slots = plan.get("cases"), plan.get("replacement_slots")
    count = len(cases) if isinstance(cases, list) else -1
    if (plan.get("status") != "frozen_requires_hash_bound_approval"
            or plan.get("dispatch_authorized") is not False
            or plan.get("calls_allowed") is not False
            or plan.get("task_contract") != "selected_answer_support_v1"
            or plan.get("scope") != "single_doc"
            or plan.get("model_identity") != list(MODEL_IDENTITY)
            or plan.get("provider_model_id") != MODEL_ID
            or plan.get("provider") != "http" or plan.get("base_url") != BASE_URL
            or plan.get("api_key_env") != API_KEY_ENV
            or plan.get("paid_fallback") is not False
            or plan.get("cache_enabled") is not False
            or plan.get("max_cost_usd") != MAX_COST_USD
            or not 1 <= count <= MAX_TOPUP_CALLS
            or not isinstance(slots, list) or len(slots) != count
            or plan.get("topup_logical_calls") != count
            or plan.get("max_logical_calls") != count
            or plan.get("max_physical_attempts") != count
            or plan.get("max_retries_per_call") != 0
            or plan.get("max_tokens_per_call") != MAX_TOKENS_PER_CALL
            or plan.get("max_total_completion_tokens") != count * MAX_TOKENS_PER_CALL
            or plan.get("max_input_chars_per_call") != MAX_INPUT_CHARS_PER_CALL
            or plan.get("max_total_input_chars") != count * MAX_INPUT_CHARS_PER_CALL
            or plan.get("lineage_original_plan_unchanged") is not True
            or type(plan.get("lineage_original_physical_attempts")) is not int
            or plan.get("lineage_cumulative_physical_attempts_max") !=
                plan.get("lineage_original_physical_attempts") + count):
        raise ValueError("top-up plan violates frozen scope or hard limits")
    for key, expected in _protocol_document_hashes().items():
        if plan.get(key) != expected:
            raise ValueError("top-up plan is stale against the current SPEC/protocol documents")
    if ({case.get("replaces_case_id") for case in cases} !=
            {slot.get("replaces_case_id") for slot in slots}
            or len({case.get("case_id") for case in cases}) != count
            or len({case.get("attempt_id") for case in cases}) != count
            or len({case.get("prompt_sha256") for case in cases}) != count):
        raise ValueError("top-up case/attempt identities or failed-slot coverage are invalid")
    for case in cases:
        if (not isinstance(case.get("case_id"), str)
                or not case["case_id"].startswith("g05-topup-case-")
                or not isinstance(case.get("attempt_id"), str)
                or not case["attempt_id"].startswith("g05-topup-attempt-")
                or case.get("max_completion_tokens") != MAX_TOKENS_PER_CALL
                or not isinstance(case.get("system"), str)
                or not isinstance(case.get("user"), str)
                or len(case["system"]) + len(case["user"]) > MAX_INPUT_CHARS_PER_CALL
                or hashlib.sha256((case["system"] + "\0" + case["user"]).encode()).hexdigest()
                    != case.get("prompt_sha256")):
            raise ValueError("top-up case prompt or identity contract is invalid")


def approve_g05_task_authoring_topup(
    plan: dict, *, approved_by: str, approval_statement: str,
    approved_at: str | None = None,
) -> dict:
    verify_g05_task_authoring_topup_plan(plan)
    if approval_statement != TOPUP_APPROVAL_STATEMENT:
        raise ValueError("explicit top-up approval statement is required")
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise ValueError("top-up approval reviewer identity is required")
    timestamp = approved_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("top-up approval timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("top-up approval timestamp must include a timezone")
    body = {"schema": TOPUP_APPROVAL_SCHEMA, "approved": True,
        "approved_by": approved_by.strip(), "approved_at": timestamp,
        "approval_statement": TOPUP_APPROVAL_STATEMENT,
        "topup_plan_sha256": plan["plan_sha256"],
        "parent_run_sha256": plan["parent_run_sha256"],
        "parent_ledger_sha256": plan["parent_ledger_sha256"],
        "topup_run_dir": plan["topup_run_dir"],
        "model_identity": plan["model_identity"], "provider_model_id": MODEL_ID,
        "max_logical_calls": plan["max_logical_calls"],
        "max_physical_attempts": plan["max_physical_attempts"],
        "max_retries_per_call": 0,
        "max_tokens_per_call": MAX_TOKENS_PER_CALL,
        "max_total_completion_tokens": plan["max_total_completion_tokens"],
        "max_cost_usd": MAX_COST_USD, "paid_fallback": False,
        "cache_enabled": False,
        "lineage_cumulative_physical_attempts_max":
            plan["lineage_cumulative_physical_attempts_max"]}
    return {**body, "approval_sha256": _canonical_sha(body)}


def verify_g05_topup_approval(plan: dict, approval: dict) -> None:
    verify_g05_task_authoring_topup_plan(plan)
    if not isinstance(approval, dict) or approval.get("schema") != TOPUP_APPROVAL_SCHEMA:
        raise ValueError("G05 top-up approval is missing or malformed")
    approval_sha = _require_sha(approval.get("approval_sha256"), "top-up approval hash is invalid")
    if _canonical_sha({key: value for key, value in approval.items()
                       if key != "approval_sha256"}) != approval_sha:
        raise ValueError("top-up approval hash mismatch")
    expected = approve_g05_task_authoring_topup(plan,
        approved_by=approval.get("approved_by"),
        approval_statement=approval.get("approval_statement"),
        approved_at=approval.get("approved_at"))
    if approval != expected or approval.get("approved") is not True:
        raise ValueError("top-up approval does not match the exact plan and hard caps")


def _verify_parent_for_plan(plan: dict, workspace: Path) -> dict:
    parent_dir = workspace / ".workbuddy" / Path(plan["parent_run_dir"])
    snapshot = _load_parent_snapshot(parent_dir, workspace)
    if (snapshot["status"]["run_sha256"] != plan["parent_run_sha256"]
            or snapshot["status"]["call_ledger_sha256"] != plan["parent_ledger_sha256"]
            or snapshot["plan"]["plan_sha256"] != plan["parent_execution_plan_sha256"]
            or snapshot["approval"]["approval_sha256"] != plan["parent_approval_sha256"]
            or snapshot["authoring"]["plan_sha256"] != plan["parent_authoring_plan_sha256"]
            or snapshot["sample"]["plan_sha256"] != plan["sample_plan_sha256"]
            or _file_sha256(parent_dir / "status.json") != plan["parent_status_file_sha256"]):
        raise ValueError("parent G05 run changed after top-up plan freeze")
    regenerated = build_g05_task_authoring_topup_plan(parent_dir,
        seed=plan["seed"], run_dir=workspace / ".workbuddy" / Path(plan["topup_run_dir"]),
        workspace=workspace, _allow_existing_run_dir=True)
    if regenerated != plan:
        raise ValueError("top-up plan no longer matches parent slots or source hashes")
    return snapshot


def _topup_events(path: Path) -> list[dict]:
    return _read_ledger(path)


def _topup_ledger_states(events: list[dict]) -> tuple[dict[str, dict], set[str], set[str]]:
    starts, ends = {}, {}
    for row in events:
        if row.get("event") == "g05_topup_case_start":
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or case_id in starts:
                raise ValueError("duplicate or malformed top-up start; refusing resume")
            starts[case_id] = row
        elif row.get("event") == "g05_topup_case_end":
            case_id = row.get("case_id")
            if case_id not in starts or case_id in ends:
                raise ValueError("orphan or duplicate top-up terminal; refusing resume")
            ends[case_id] = row
    return ends, set(starts), set(starts) - set(ends)


def _verify_topup_resume(run_dir: Path, plan: dict, approval: dict,
                         ledger_path: Path) -> None:
    status_path = run_dir / "status.json"
    status = _read_json(status_path)
    if (status.get("status") == "started" and not ledger_path.exists()
            and status.get("topup_plan_sha256") == plan["plan_sha256"]
            and status.get("approval_sha256") == approval["approval_sha256"]
            and status.get("model_calls") == 0):
        return
    sha = status.get("run_sha256")
    if (not isinstance(sha, str) or _canonical_sha({k: v for k, v in status.items()
            if k != "run_sha256"}) != sha):
        raise ValueError("top-up resume status hash is missing or invalid")
    if (status.get("topup_plan_sha256") != plan["plan_sha256"]
            or status.get("approval_sha256") != approval["approval_sha256"]
            or ( _file_sha256(ledger_path) if ledger_path.is_file() else None)
                != status.get("call_ledger_sha256")):
        raise ValueError("top-up ledger hash does not match its frozen status")


def execute_g05_task_authoring_topup(
    plan: dict, approval: dict | None, *, workspace: Path,
    execute: bool = False, resume: bool = False, provider=None,
    stop_after_first_error: bool = False,
) -> dict:
    """Dry-run by default; dispatch only the approved incremental slots."""
    workspace = Path(workspace).resolve()
    verify_g05_task_authoring_topup_plan(plan)
    snapshot = _verify_parent_for_plan(plan, workspace)
    current_price = _verify_price_evidence(
        workspace / plan["price_evidence"]["report_path"], workspace)
    if current_price != plan["price_evidence"]:
        raise ValueError("top-up zero-price evidence changed or expired")
    if (_file_sha256(Path(__file__)) != plan.get("topup_source_sha256")
            or _file_sha256(Path(__file__).resolve().parents[1] / "runtime" / "provider.py")
                != plan.get("provider_source_sha256")
            or _protocol_document_hashes() != {"spec_sha256": plan.get("spec_sha256"),
                "protocol_sha256": plan.get("protocol_sha256")}):
        raise ValueError("top-up or provider source changed after plan freeze")
    if not execute:
        if approval is not None:
            verify_g05_topup_approval(plan, approval)
        return {"schema": TOPUP_RUN_SCHEMA, "status": "dry_run_validated",
            "topup_plan_sha256": plan["plan_sha256"], "model_calls": 0,
            "planned_calls": len(plan["cases"]), "dispatch_authorized": False,
            "parent_run_sha256": snapshot["status"]["run_sha256"]}
    if approval is None:
        raise ValueError("top-up execution requires a separate hash-bound approval")
    verify_g05_topup_approval(plan, approval)
    if provider is not None and not os.environ.get("RSI_HOME"):
        raise ValueError("injected top-up providers are test-only and require isolated RSI_HOME")
    run_dir = workspace / ".workbuddy" / Path(plan["topup_run_dir"])
    run_relative = _safe_relative(run_dir, workspace, name="top-up run directory")
    if run_relative != plan["topup_run_dir"]:
        raise ValueError("top-up run path differs from its frozen plan")
    claim_dir = workspace / ".workbuddy" / "g05-topup-claims"
    claim_path = claim_dir / f"{plan['parent_run_sha256']}.json"
    ledger_path = run_dir / "call-ledger.jsonl"
    with _run_lock(claim_dir / plan["parent_run_sha256"]):
        if not resume:
            if claim_path.exists():
                raise FileExistsError("this parent G05 run already has a top-up claim")
            if run_dir.exists():
                raise FileExistsError("top-up run exists; resume the same frozen plan")
            run_dir.mkdir(parents=True)
            for name, row in (("topup-plan.json", plan), ("approval.json", approval)):
                _save_once_json(run_dir / name, row)
            initial = {"schema": TOPUP_RUN_SCHEMA, "status": "started",
                "topup_plan_sha256": plan["plan_sha256"], "approval_sha256": approval["approval_sha256"],
                "model_calls": 0}
            _save_once_json(run_dir / "status.json", initial)
            claim = {"parent_run_sha256": plan["parent_run_sha256"],
                "topup_plan_sha256": plan["plan_sha256"],
                "topup_run_dir": run_relative}
            claim["claim_sha256"] = _canonical_sha(claim)
            _save_once_json(claim_path, claim)
        else:
            if not run_dir.is_dir() or not claim_path.is_file():
                raise FileNotFoundError("matching top-up run and one-shot claim are required to resume")
            claim = _read_json(claim_path)
            if (claim.get("topup_plan_sha256") != plan["plan_sha256"]
                    or claim.get("topup_run_dir") != run_relative
                    or claim.get("parent_run_sha256") != plan["parent_run_sha256"]
                    or _canonical_sha({k: v for k, v in claim.items() if k != "claim_sha256"})
                        != claim.get("claim_sha256")):
                raise ValueError("top-up claim is bound to a different plan or run")
            for name, expected in (("topup-plan.json", plan), ("approval.json", approval)):
                path = run_dir / name
                if not path.is_file() or _read_json(path) != expected:
                    raise ValueError("top-up resume artifacts changed")
            _verify_topup_resume(run_dir, plan, approval, ledger_path)
        with _run_lock(run_dir):
            return _execute_topup_locked(plan, approval, run_dir, ledger_path,
                provider=provider, stop_after_first_error=stop_after_first_error,
                resume=resume)


def _execute_topup_locked(plan: dict, approval: dict, run_dir: Path,
                          ledger_path: Path, *, provider, stop_after_first_error,
                          resume: bool) -> dict:
    events = _topup_events(ledger_path)
    terminal, started, orphaned = _topup_ledger_states(events)
    cases = {row["case_id"]: row for row in plan["cases"]}
    if (set(started) - set(cases) or set(terminal) - set(cases)
            or len(started) > plan["max_physical_attempts"] or orphaned):
        raise ValueError("top-up ledger violates frozen plan or physical cap")
    for case_id, row in cases.items():
        start = next((event for event in events if event.get("event") == "g05_topup_case_start"
                      and event.get("case_id") == case_id), None)
        if start and (start.get("topup_plan_sha256") != plan["plan_sha256"]
                or start.get("attempt_id") != row["attempt_id"]
                or start.get("prompt_sha256") != row["prompt_sha256"]):
            raise ValueError("top-up start receipt does not match frozen case identity")
    if provider is None:
        if not os.environ.get(API_KEY_ENV):
            raise RuntimeError(f"required credential environment variable is missing: {API_KEY_ENV}")
        from ..runtime.provider import build_provider
        actual_provider = build_provider({
            "runtime": {"provider": "http", "http_no_cache_required": True},
            "models": {"task": {"provider": "http", "model_id": MODEL_ID,
                "base_url": BASE_URL, "api_key_env": API_KEY_ENV,
                "max_tokens": MAX_TOKENS_PER_CALL, "max_retries": 1,
                "timeout": 120, "temperature": 0, "cache": False,
                "enable_thinking": False}}})
    else:
        actual_provider = provider
    names = ("RSI_ACTIVE_EXECUTION_PLAN_SHA256", "RSI_CALL_LEDGER_PATH",
        "RSI_CALL_LEDGER_MAX_LOGICAL", "RSI_CALL_LEDGER_MAX_PHYSICAL",
        "RSI_CALL_LEDGER_MAX_COMPONENT_AI_EVALUATION", "RSI_HTTP_NO_CACHE")
    saved = {key: os.environ.get(key) for key in names}
    os.environ.update({"RSI_ACTIVE_EXECUTION_PLAN_SHA256": plan["plan_sha256"],
        "RSI_CALL_LEDGER_PATH": str(ledger_path),
        "RSI_CALL_LEDGER_MAX_LOGICAL": str(plan["max_logical_calls"]),
        "RSI_CALL_LEDGER_MAX_PHYSICAL": str(plan["max_physical_attempts"]),
        "RSI_CALL_LEDGER_MAX_COMPONENT_AI_EVALUATION": str(plan["max_physical_attempts"]),
        "RSI_HTTP_NO_CACHE": "1"})
    prompt_total = completion_total = 0
    valid = invalid = unknown = 0
    for case_id, end in terminal.items():
        if end.get("result_sha256"):
            receipt_path = run_dir / "cases" / f"{case_id}.json"
            if not receipt_path.is_file() or _file_sha256(receipt_path) != end["result_sha256"]:
                raise ValueError("top-up receipt missing or changed")
            receipt = _read_json(receipt_path)
            if receipt.get("status") != end.get("status"):
                raise ValueError("top-up receipt terminal state disagrees with its ledger")
            prompt_tokens = receipt.get("prompt_tokens")
            completion_tokens = receipt.get("completion_tokens")
            if type(prompt_tokens) is int and prompt_tokens >= 0:
                prompt_total += prompt_tokens
            if type(completion_tokens) is int and completion_tokens >= 0:
                completion_total += completion_tokens
            if receipt.get("status") == "valid":
                valid += 1
            elif end.get("status") == "invalid":
                invalid += 1
        elif end.get("status") == "invalid":
            invalid += 1
        else:
            unknown += 1
    stopped = False
    interrupted = False
    try:
        from ..runtime.call_ledger import call_component
        for case in plan["cases"]:
            case_id = case["case_id"]
            if case_id in started:
                continue
            if len(started) >= plan["max_physical_attempts"]:
                raise RuntimeError("top-up physical call cap reached")
            _append_jsonl(ledger_path, {"event": "g05_topup_case_start",
                "case_id": case_id, "attempt_id": case["attempt_id"], "attempt": 1,
                "prompt_sha256": case["prompt_sha256"],
                "topup_plan_sha256": plan["plan_sha256"],
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            started.add(case_id)
            try:
                with call_component("ai_evaluation"):
                    output = actual_provider.chat_measured(case["system"], case["user"],
                        max_tokens=plan["max_tokens_per_call"])
                prompt_tokens = getattr(output, "prompt_tokens", None)
                completion_tokens = getattr(output, "completion_tokens", None)
                model_id = getattr(output, "model_id", None)
                cost = getattr(output, "cost", None) or {}
                usage_valid = (getattr(output, "usage_available", False) is True
                    and type(prompt_tokens) is int and 0 <= prompt_tokens <= MAX_INPUT_CHARS_PER_CALL
                    and type(completion_tokens) is int and 0 <= completion_tokens <= MAX_TOKENS_PER_CALL
                    and prompt_total + prompt_tokens <= plan["max_total_input_chars"]
                    and completion_total + completion_tokens <= plan["max_total_completion_tokens"])
                cost_valid = (cost.get("usd") == 0.0 and cost.get("cost_estimated") is False
                    and cost.get("usage_complete") is True and cost.get("pricing_region") == "CN"
                    and cost.get("pricing_source") == PRICE_SOURCE
                    and cost.get("pricing_evidence_report_sha256") ==
                        plan["price_evidence"]["provider_price_evidence_sha256"])
                identity_valid = model_id == MODEL_ID and getattr(output, "http_attempts", None) == 1
                parsed, parse_error = _strict_task_json(getattr(output, "content", None))
                reason = parse_error or (None if usage_valid and cost_valid and identity_valid
                    else "usage_cost_model_or_http_identity_contract_failed")
                hard_violation = not (usage_valid and cost_valid and identity_valid)
                state = "valid" if reason is None else "invalid"
                output_text = getattr(output, "content", None)
                receipt = {"schema": "g05-task-authoring-topup-receipt-v1",
                    "case_id": case_id, "attempt_id": case["attempt_id"],
                    "replaces_case_id": case["replaces_case_id"],
                    "source_group_id": case["source_group_id"],
                    "source_content_sha256": case["source_content_sha256"],
                    "prompt_sha256": case["prompt_sha256"], "status": state,
                    "reason": reason, "provider_model_id": MODEL_ID,
                    "response_model_id": model_id,
                    "http_attempts": getattr(output, "http_attempts", None),
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "cost": cost, "raw_output": output_text,
                    "raw_output_sha256": _canonical_sha(output_text),
                    "parsed_response": parsed, "test_only": provider is not None}
                receipt_path = run_dir / "cases" / f"{case_id}.json"
                result_sha = _write_once(receipt_path, receipt)
                _append_jsonl(ledger_path, {"event": "g05_topup_case_end",
                    "case_id": case_id, "attempt_id": case["attempt_id"],
                    "status": state, "reason": reason, "result_sha256": result_sha,
                    "physical_attempts_reserved": 1,
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "cost_usd": cost.get("usd"),
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                terminal[case_id] = {"status": state, "result_sha256": result_sha}
                if type(prompt_tokens) is int and prompt_tokens >= 0:
                    prompt_total += prompt_tokens
                if type(completion_tokens) is int and completion_tokens >= 0:
                    completion_total += completion_tokens
                if state == "valid":
                    valid += 1
                else:
                    invalid += 1
                if hard_violation:
                    stopped = True
                    break
            except BaseException as exc:
                _append_jsonl(ledger_path, {"event": "g05_topup_case_end",
                    "case_id": case_id, "attempt_id": case["attempt_id"],
                    "status": "failed_unknown_outcome", "reason": type(exc).__name__,
                    "physical_attempts_reserved": 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                terminal[case_id] = {"status": "failed_unknown_outcome"}
                unknown += 1
                if stop_after_first_error:
                    stopped, interrupted = True, True
                    break
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    events = _topup_events(ledger_path)
    terminal, started, orphaned = _topup_ledger_states(events)
    if orphaned:
        unknown += len(orphaned)
    valid = invalid = 0
    for case_id, end in terminal.items():
        if end.get("status") == "valid":
            receipt_path = run_dir / "cases" / f"{case_id}.json"
            if not receipt_path.is_file() or _file_sha256(receipt_path) != end.get("result_sha256"):
                raise ValueError("top-up valid receipt hash mismatch")
            receipt = _read_json(receipt_path)
            if receipt.get("status") == "valid" and receipt.get("test_only") is (provider is not None):
                valid += 1
            else:
                raise ValueError("top-up receipt state or test-only marker mismatch")
        elif end.get("status") == "invalid":
            invalid += 1
    unknown = sum(row.get("status") == "failed_unknown_outcome" for row in terminal.values()) + len(orphaned)
    complete = (len(started) == len(cases) and valid == len(cases)
        and invalid == 0 and unknown == 0 and not stopped and provider is None)
    if complete:
        status_value = "complete"
    elif provider is not None and len(started) == len(cases) and valid == len(cases):
        status_value = "test_only_not_promotable"
    elif stopped and interrupted:
        status_value = "interrupted_incomplete"
    else:
        status_value = "incomplete_or_invalid"
    summary = {"schema": TOPUP_RUN_SCHEMA, "status": status_value,
        "topup_plan_sha256": plan["plan_sha256"], "approval_sha256": approval["approval_sha256"],
        "parent_run_sha256": plan["parent_run_sha256"],
        "parent_ledger_sha256": plan["parent_ledger_sha256"],
        "case_count": len(cases), "attempted_case_count": len(started),
        "physical_attempts": len(started), "max_physical_attempts": len(cases),
        "valid_count": valid, "invalid_count": invalid, "unknown_outcome_count": unknown,
        "model_calls": len(started), "total_prompt_tokens": prompt_total,
        "total_completion_tokens": completion_total,
        "max_total_completion_tokens": plan["max_total_completion_tokens"],
        "max_cost_usd": MAX_COST_USD, "paid_fallback": False,
        "test_only": provider is not None, "call_ledger_path": "call-ledger.jsonl",
        "call_ledger_sha256": _file_sha256(ledger_path) if ledger_path.exists() else None}
    summary["run_sha256"] = _canonical_sha(summary)
    _write_status(run_dir / "status.json", summary)
    return summary


def merge_g05_task_authoring_topup(
    parent_run_dir: Path, topup_run_dir: Path, *, workspace: Path,
) -> dict:
    """Merge only validated successful parent and child task outputs."""
    workspace = Path(workspace).resolve()
    snapshot = _load_parent_snapshot(Path(parent_run_dir), workspace)
    topup_dir = Path(topup_run_dir).resolve()
    _safe_relative(topup_dir, workspace, name="top-up run")
    plan = _read_json(topup_dir / "topup-plan.json")
    approval = _read_json(topup_dir / "approval.json")
    verify_g05_task_authoring_topup_plan(plan)
    verify_g05_topup_approval(plan, approval)
    if (plan["parent_run_sha256"] != snapshot["status"]["run_sha256"]
            or plan["parent_status_file_sha256"] != _file_sha256(snapshot["parent_dir"] / "status.json")
            or plan["parent_execution_plan_sha256"] != snapshot["plan"]["plan_sha256"]
            or plan["topup_run_dir"] != _safe_relative(topup_dir, workspace, name="top-up run")):
        raise ValueError("top-up artifacts are not bound to this original parent run")
    if (_file_sha256(Path(__file__)) != plan.get("topup_source_sha256")
            or _file_sha256(Path(__file__).resolve().parents[1] / "runtime" / "provider.py")
                != plan.get("provider_source_sha256")
            or _protocol_document_hashes() != {"spec_sha256": plan.get("spec_sha256"),
                "protocol_sha256": plan.get("protocol_sha256")}):
        raise ValueError("top-up or provider source changed after plan freeze")
    if _verify_price_evidence(workspace / plan["price_evidence"]["report_path"],
                              workspace) != plan["price_evidence"]:
        raise ValueError("top-up zero-price evidence changed or expired")
    status = _read_json(topup_dir / "status.json")
    status_sha = _require_sha(status.get("run_sha256"), "top-up run status hash is invalid")
    if _canonical_sha({k: v for k, v in status.items() if k != "run_sha256"}) != status_sha:
        raise ValueError("top-up run status hash mismatch")
    ledger_path = topup_dir / "call-ledger.jsonl"
    if (status.get("status") == "test_only_not_promotable" or status.get("test_only") is not False):
        raise ValueError("test-only top-up results cannot be merged")
    if (status.get("status") != "complete"
            or _file_sha256(ledger_path) != status.get("call_ledger_sha256")
            or status.get("topup_plan_sha256") != plan["plan_sha256"]
            or status.get("approval_sha256") != approval["approval_sha256"]
            or status.get("valid_count") != len(plan["cases"])
            or status.get("invalid_count") != 0 or status.get("unknown_outcome_count") != 0):
        raise ValueError("top-up execution is incomplete or changed")
    events = _topup_events(ledger_path)
    terminal, started, orphaned = _topup_ledger_states(events)
    cases = {row["case_id"]: row for row in plan["cases"]}
    if orphaned or set(started) != set(cases) or set(terminal) != set(cases):
        raise ValueError("top-up ledger does not cover every approved replacement case")
    replacement_outputs = {}
    replacement_map = []
    for case_id, case in cases.items():
        end = terminal[case_id]
        path = topup_dir / "cases" / f"{case_id}.json"
        if (end.get("status") != "valid" or not path.is_file()
                or _file_sha256(path) != end.get("result_sha256")):
            raise ValueError("top-up output is missing, invalid, or hash-mismatched")
        receipt = _read_json(path)
        parsed, error = _strict_task_json(receipt.get("raw_output"))
        if (receipt.get("schema") != "g05-task-authoring-topup-receipt-v1"
                or receipt.get("status") != "valid" or receipt.get("test_only") is not False
                or receipt.get("case_id") != case_id
                or receipt.get("attempt_id") != case["attempt_id"]
                or receipt.get("replaces_case_id") != case["replaces_case_id"]
                or receipt.get("prompt_sha256") != case["prompt_sha256"]
                or parsed is None or error is not None or parsed != receipt.get("parsed_response")):
            raise ValueError("top-up receipt failed output or identity validation")
        replacement_outputs[case["replaces_case_id"]] = parsed
        replacement_map.append({"replaces_case_id": case["replaces_case_id"],
            "topup_case_id": case_id, "attempt_id": case["attempt_id"],
            "topup_receipt_sha256": end["result_sha256"]})
    authored = []
    source_cases = {row["case_id"]: row for row in snapshot["authoring"]["cases"]}
    for shell in snapshot["sample"]["task_shells"]:
        parent_case_id = shell["case_id"]
        if parent_case_id in snapshot["valid"]:
            raw = snapshot["valid"][parent_case_id]["parsed_response"]
        else:
            raw = replacement_outputs.get(parent_case_id)
        if raw is None:
            raise ValueError("not every original sample slot has exactly one valid authoring response")
        if source_cases[parent_case_id]["source_group_id"] != shell["source_group_id"]:
            raise ValueError("merged source group differs from frozen sample shell")
        authored.append({"case_id": parent_case_id, "source_group_id": shell["source_group_id"],
            "question": raw["question"], "selected_answer": raw["selected_answer"]})
    if len(authored) != MAX_CALLS or len({row["case_id"] for row in authored}) != MAX_CALLS:
        raise ValueError("merged G05 authoring input does not cover exactly 120 sample slots")
    authoring_input = {"schema": "g05-task-authoring-input-v1",
        "sample_plan_sha256": snapshot["sample"]["plan_sha256"], "cases": authored}
    body = {"schema": TOPUP_MERGE_SCHEMA, "status": "complete",
        "parent_run_sha256": snapshot["status"]["run_sha256"],
        "parent_ledger_sha256": snapshot["status"]["call_ledger_sha256"],
        "topup_plan_sha256": plan["plan_sha256"],
        "topup_run_sha256": status_sha,
        "authoring_input": authoring_input,
        "authoring_input_sha256": _canonical_sha(authoring_input),
        "replacement_map": replacement_map,
        "old_invalid_response_reused": False,
        "lineage_cumulative_physical_attempts":
            snapshot["status"]["physical_attempts"] + status["physical_attempts"],
        "lineage_cumulative_physical_attempts_max":
            plan["lineage_cumulative_physical_attempts_max"],
        "model_calls_during_merge": 0}
    return {**body, "merge_sha256": _canonical_sha(body)}
