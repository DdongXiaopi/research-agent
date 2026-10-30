"""Offline, fail-closed audit for a failed synthetic reference job.

This module deliberately does not resume, retry, merge, score, or manufacture a
completed ai_jobs receipt. A response can be associated with an input file, but
the legacy request ledger does not bind operation IDs to case/payload hashes;
therefore completed rows are diagnostic evidence only.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .instruction_reviewer_cohort_reference import (
    RUBRIC_V2_VERSION, _reference_payload, reference_inputs, validate_reference_packet,
)
from .instruction_reviewer_rubric_v2 import RUBRIC_V2
from ..util import read_json, sha256_file, sha256_obj


AUDIT_SCHEMA = "instruction-reviewer-partial-run-audit-v1"


def _read_strict_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"ledger contains blank row at line {line_number}")
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError(f"ledger JSON invalid at line {line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"ledger row is not an object at line {line_number}")
                rows.append(row)
    except OSError as exc:
        raise ValueError("ledger is unavailable") from exc
    if not rows:
        raise ValueError("ledger is empty")
    return rows


def _verify_ledger(path: Path, *, expected_count: int, model_id: str,
                   plan_sha256: str) -> dict:
    events = _read_strict_jsonl(path)
    allowed = ["logical_start", "http_attempt_start", "http_attempt_end", "logical_end"]
    if len(events) != expected_count * len(allowed):
        raise ValueError("ledger event count does not match attempted calls")
    groups: dict[str, list[dict]] = {}
    sequence: list[str] = []
    seen_event_ids: set[tuple[str, str, int | None]] = set()
    for row in events:
        event = row.get("event")
        operation_id = row.get("operation_id")
        if event not in allowed or not isinstance(operation_id, str) or not operation_id:
            raise ValueError("ledger contains unknown event or missing operation identity")
        if event == "logical_start":
            if operation_id in groups:
                raise ValueError("ledger contains duplicate logical operation")
            sequence.append(operation_id)
            groups[operation_id] = []
        elif operation_id not in groups:
            raise ValueError("ledger event precedes its logical start")
        attempt = row.get("attempt") if event.startswith("http_attempt_") else None
        event_key = (operation_id, event, attempt)
        if event_key in seen_event_ids:
            raise ValueError("ledger contains duplicate event")
        seen_event_ids.add(event_key)
        groups[operation_id].append(row)
    if len(groups) != expected_count:
        raise ValueError("ledger logical operation count mismatch")
    for operation_id, group in groups.items():
        if [row.get("event") for row in group] != allowed:
            raise ValueError("ledger operation events are incomplete or out of order")
        start, http_start, http_end, end = group
        if (start.get("model_id") != model_id or start.get("component") != "ai_evaluation"
                or start.get("plan_sha256") != plan_sha256
                or start.get("provider") != "http"):
            raise ValueError("ledger operation identity does not match frozen plan")
        if (http_start.get("attempt") != 1 or http_end.get("attempt") != 1
                or http_end.get("status") not in (200, None)
                or end.get("status") not in ("success", "failed")):
            raise ValueError("ledger physical attempt status is invalid")
        if ((http_end.get("status") == 200) != (end.get("status") == "success")):
            raise ValueError("ledger HTTP and logical statuses disagree")
    return {"operation_count": len(groups),
            "physical_status_counts": dict(sorted(Counter(
                str(group[2].get("status")) for group in groups.values()).items())),
            "logical_status_counts": dict(sorted(Counter(
                str(group[3].get("status")) for group in groups.values()).items())),
            "operation_ids_in_order": sequence,
            "payload_binding_present": all("payload_sha256" in row for row in events),
            "case_index_binding_present": all("case_index" in row for row in events)}


def audit_partial_reference_run(packet: dict, frozen_plan: dict, slot: str,
                                run_dir: str | Path) -> dict:
    """Return a metadata-only audit manifest; never modify the failed run."""
    validate_reference_packet(packet)
    if slot not in {"reviewer_a", "reviewer_b"}:
        raise ValueError("invalid reviewer slot")
    plan = frozen_plan.get("plans", {}).get(slot)
    if not isinstance(plan, dict):
        raise ValueError("frozen plan is missing reviewer plan")
    body = {key: value for key, value in frozen_plan.items() if key != "plan_sha256"}
    plan_body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    role_binding = frozen_plan.get(slot, {})
    expected_inputs = reference_inputs(packet, slot)
    if (frozen_plan.get("plan_sha256") != sha256_obj(body)
            or frozen_plan.get("packet_sha256") != packet.get("packet_sha256")
            or frozen_plan.get("rubric_version") != RUBRIC_V2_VERSION
            or frozen_plan.get("rubric_sha256") != sha256_obj(RUBRIC_V2)
            or frozen_plan.get("reviewer_a", {}).get("model_id")
                == frozen_plan.get("reviewer_b", {}).get("model_id")
            or role_binding.get("model_id") != plan.get("model_id")
            or role_binding.get("plan_sha256") != plan.get("plan_sha256")
            or plan.get("plan_sha256") != sha256_obj(plan_body)
            or plan.get("schema") != "ai-evaluation-plan-v1"
            or plan.get("kind") != "instruction_reference"
            or plan.get("inputs_sha256") != sha256_obj(expected_inputs)
            or plan.get("max_calls") != len(packet.get("cases", []))
            or type(plan.get("max_physical_attempts")) is not int
            or plan.get("max_physical_attempts") < plan.get("max_calls")
            or plan.get("protocol_metadata") != {
                "rubric_version": RUBRIC_V2_VERSION,
                "rubric_sha256": sha256_obj(RUBRIC_V2)}):
        raise ValueError("frozen plan identity, inner plan hash, inputs hash, or rubric binding mismatch")

    root = Path(run_dir).resolve()
    required = ("plan.json", "inputs.json", "calls.json", "status.json", "call-ledger.jsonl")
    paths = {name: root / name for name in required}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("partial job is missing required audit artifact")
    if (root / "receipt.json").exists() or (root / "output.json").exists():
        raise ValueError("failed partial job must not contain completed receipt or output")
    disk_plan = read_json(paths["plan.json"])
    if disk_plan != plan:
        raise ValueError("run plan identity differs from frozen cohort plan")
    disk_inputs = read_json(paths["inputs.json"])
    if disk_inputs != expected_inputs:
        raise ValueError("run inputs differ from frozen blind packet")
    calls = read_json(paths["calls.json"])
    status = read_json(paths["status.json"])
    packet_cases = packet.get("cases", [])
    if not isinstance(calls, list) or not isinstance(packet_cases, list):
        raise ValueError("calls or packet cases have invalid shape")
    if len(calls) >= len(packet_cases) or not calls:
        raise ValueError("audit requires a genuinely partial job")
    if status.get("status") != "failed" or status.get("call_count") != len(calls):
        raise ValueError("run status does not describe the failed partial job")

    ledger = _verify_ledger(paths["call-ledger.jsonl"], expected_count=len(calls),
        model_id=plan["model_id"], plan_sha256=plan["plan_sha256"])
    if len(calls) != 19 or len(packet_cases) != 40:
        raise ValueError("this audit profile is frozen for the observed 19-of-40 failed run")
    states = []
    response_checks = []
    for index, case in enumerate(packet_cases):
        if index < len(calls):
            row = calls[index]
            expected_payload = _reference_payload(packet, case)
            if (row.get("index") != index or row.get("role") != "instruction_reference"
                    or row.get("input_sha256") != sha256_obj(expected_payload)):
                raise ValueError(f"call input identity mismatch at index {index}")
            if index < 18:
                call_path = root / f"call-{index}.json"
                if not call_path.is_file():
                    raise ValueError(f"completed response file missing at index {index}")
                raw = read_json(call_path)
                if (row.get("status") != "completed"
                        or raw.get("role") != "instruction_reference"
                        or raw.get("payload") != expected_payload
                        or raw.get("system") != RUBRIC_V2
                        or raw.get("user") != json.dumps(expected_payload,
                            ensure_ascii=False, sort_keys=True)):
                    raise ValueError(f"completed response payload mismatch at index {index}")
                try:
                    parsed = json.loads(raw.get("raw_output", ""))
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ValueError(f"raw response JSON invalid at index {index}") from exc
                cost = raw.get("cost")
                if (not isinstance(cost, dict) or cost.get("usd") != 0.0
                        or cost.get("usage_complete") is not True
                        or cost.get("pricing_region") != "CN"
                        or not isinstance(cost.get("prompt_tokens"), int)
                        or not isinstance(cost.get("completion_tokens"), int)
                        or row.get("cost") != cost
                        or row.get("response_sha256") != sha256_obj(raw.get("raw_output"))):
                    raise ValueError(f"response hash or usage identity mismatch at index {index}")
                response_checks.append({"index": index,
                    "call_file_sha256": sha256_file(call_path),
                    "raw_output_sha256": sha256_obj(raw["raw_output"]),
                    "parsed_response_sha256": sha256_obj(parsed),
                    "response_string_sha256": sha256_obj(raw["raw_output"]),
                    "input_payload_sha256": sha256_obj(expected_payload),
                    "usage_verified": True})
                state = "completed_response_unbound_to_operation"
            else:
                if row.get("status") != "started" or row.get("response_sha256") is not None:
                    raise ValueError("attempted unknown call must remain without response hash")
                if (root / f"call-{index}.json").exists():
                    raise ValueError("attempted unknown call unexpectedly has a response file")
                state = "attempted_unknown"
            states.append({"case_id": case["case_id"], "index": index,
                "state": state, "input_payload_sha256": sha256_obj(expected_payload),
                "source_plan_sha256": plan["plan_sha256"],
                "source_tree_sha256": plan.get("source_tree_sha256")})
        else:
            states.append({"case_id": case["case_id"], "index": index,
                "state": "never_started", "input_payload_sha256": None,
                "source_plan_sha256": plan["plan_sha256"],
                "source_tree_sha256": plan.get("source_tree_sha256")})
    counts = Counter(row["state"] for row in states)
    accounting = status.get("request_accounting", {})
    if (ledger["logical_status_counts"] != {"failed": 1, "success": 18}
            or ledger["physical_status_counts"] != {"200": 18, "None": 1}
            or accounting.get("logical_operations") != 19
            or accounting.get("physical_http_attempts") != 19
            or accounting.get("logical_status_counts") != ledger["logical_status_counts"]
            or accounting.get("physical_status_counts") != ledger["physical_status_counts"]):
        raise ValueError("status accounting differs from strict ledger audit")
    files = {name: sha256_file(path) for name, path in paths.items()}
    files.update({f"call-{index}.json": row["call_file_sha256"]
                  for index, row in enumerate(response_checks)})
    return {
        "schema": AUDIT_SCHEMA, "status": "partial_audited_fail_closed",
        "packet_sha256": packet["packet_sha256"],
        "cohort_plan_sha256": frozen_plan["plan_sha256"],
        "slot": slot, "model_id": plan["model_id"],
        "run_plan_sha256": plan["plan_sha256"],
        "source_tree_sha256": plan.get("source_tree_sha256"),
        "rubric_version": RUBRIC_V2_VERSION,
        "rubric_sha256": sha256_obj(RUBRIC_V2),
        "run_dir": str(root), "run_artifact_sha256": files,
        "ledger_audit": {key: value for key, value in ledger.items()
                         if key != "operation_ids_in_order"},
        "case_states": states, "case_state_counts": dict(sorted(counts.items())),
        "completed_response_checks": response_checks,
        "physical_operation_binding_verified": False,
        "binding_gap": "legacy ledger has no case_index or payload_sha256; calls rows have no operation_id",
        "mergeable": False, "complete_reference_eligible": False,
        "automatic_retry_allowed": False, "receipt_fabricated": False,
        "response_labels_included": False,
        "limitations": ["Completed responses are locally hash-checked but cannot be cryptographically tied to physical ledger operations.",
            "Index 18 is attempted_unknown and must never be replayed under this audit.",
            "Indexes 19-39 never started; this manifest is not a complete reference run."]}

