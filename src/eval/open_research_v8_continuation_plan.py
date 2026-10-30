"""Offline parent-bound plan for a bounded G09 v8 Europe PMC continuation.

This module only validates frozen v7 evidence and writes a deterministic plan.
It never contacts Europe PMC, creates a cohort, labels, scores, or answers tasks.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter

from ..util import sha256_obj

V8_PLAN_SCHEMA = "g09-v8-parent-bound-continuation-plan-v1"
V7_PLAN_SCHEMA = "g09-v7-query-arm-plan-v1"
V7_CAPTURE_SCHEMA = "g09-topic-source-capture-v7"
V7_ROOT_AUDIT_SCHEMA = "g09-v7-independent-root-audit-v1"
V8_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
V8_MAX_CUMULATIVE_REQUESTS = 504
V8_MIN_REQUEST_INTERVAL_SECONDS = 1.0
V8_PAGE_SIZE = 10
_IDENTITY_FIELDS = ("candidate_audit", "records", "records_added", "exclusions")
_AUDIT_TRUE_FIELDS = (
    "artifact_hash_valid", "plan_binding_valid", "implementation_hash_valid",
    "budget_valid", "reserve_gate_valid", "cursor_continuity_valid",
    "request_phase_accounting_valid", "accepted_records_compliant",
    "diagnostic_only",
)


def _sha_ok(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _payload_sha(value: dict, field: str) -> str:
    return sha256_obj({key: item for key, item in value.items() if key != field})


def _capture_sha(value: dict) -> str:
    return sha256_obj({key: item for key, item in value.items()
        if key not in {"capture_sha256", "audit"}})


def _identity(row: object) -> tuple[str, str]:
    if not isinstance(row, dict):
        return "", ""
    doi, pmcid = row.get("doi"), row.get("pmcid")
    return (doi.strip().casefold() if isinstance(doi, str) else "",
            pmcid.strip().casefold() if isinstance(pmcid, str) else "")


def _validate_parent(*, v7_plan: dict, v7_capture: dict,
        v7_capture_file_sha256: str, v7_root_audit: dict,
        v7_root_audit_file_sha256: str) -> tuple[dict, dict, dict, int]:
    if not isinstance(v7_plan, dict) or v7_plan.get("schema") != V7_PLAN_SCHEMA:
        raise ValueError("v8 requires the frozen v7 query-arm plan")
    plan_sha = _payload_sha(v7_plan, "plan_sha256")
    if v7_plan.get("plan_sha256") != plan_sha:
        raise ValueError("v7 parent plan hash mismatch")
    if not isinstance(v7_capture, dict) or v7_capture.get("schema") != V7_CAPTURE_SCHEMA:
        raise ValueError("v8 requires the completed v7 capture artifact")
    if not _sha_ok(v7_capture_file_sha256) or not _sha_ok(v7_root_audit_file_sha256):
        raise ValueError("v8 parent artifact file hashes must be lowercase SHA-256")
    capture_sha = v7_capture.get("capture_sha256")
    if capture_sha != _capture_sha(v7_capture) or \
            v7_capture.get("plan_sha256") != plan_sha:
        raise ValueError("v7 capture content hash or plan binding mismatch")
    if v7_capture.get("diagnostic_only") is not True or \
            v7_capture.get("scoreable_cohort_created") is not False:
        raise ValueError("v7 capture is not a source-discovery-only parent")

    audit = v7_root_audit
    if not isinstance(audit, dict) or audit.get("schema") != V7_ROOT_AUDIT_SCHEMA:
        raise ValueError("v8 requires the independent v7 root audit")
    if audit.get("root_audit_sha256") != _payload_sha(audit, "root_audit_sha256"):
        raise ValueError("v7 independent root-audit hash mismatch")
    if audit.get("capture_file_sha256") not in (None, v7_capture_file_sha256):
        raise ValueError("v7 root audit does not bind the supplied capture file")
    if audit.get("capture_sha256") != capture_sha or audit.get("plan_sha256") != plan_sha:
        raise ValueError("v7 root audit does not bind the capture and plan")
    if audit.get("diagnostic_only") is not True or \
            audit.get("scoreable_cohort_created") is not False:
        raise ValueError("v7 root audit is not diagnostic-only")
    capture_audit = audit.get("capture_audit")
    if not isinstance(capture_audit, dict) or any(
            capture_audit.get(key) is not True for key in _AUDIT_TRUE_FIELDS) or \
            capture_audit.get("scoreable_cohort_created") is not False:
        raise ValueError("v7 root audit did not pass all capture-integrity checks")
    if capture_audit.get("capture_sha256") != capture_sha or \
            capture_audit.get("plan_sha256") != plan_sha:
        raise ValueError("v7 capture audit binding mismatch")
    ledger_audit = audit.get("ledger_audit")
    if not isinstance(ledger_audit, dict) or ledger_audit.get("ambiguous_in_flight_count") != 0:
        raise ValueError("v7 write-ahead ledger has ambiguous in-flight requests")

    logs = v7_capture.get("request_log")
    attempts = v7_capture.get("request_attempts")
    cumulative = v7_capture.get("cumulative_request_attempts")
    hard_max = v7_plan.get("cumulative_request_hard_max")
    if not isinstance(logs, list) or type(attempts) is not int or attempts != len(logs) or \
            type(cumulative) is not int or type(hard_max) is not int or \
            cumulative != v7_plan.get("parent_cumulative_request_attempts") + attempts or \
            cumulative > hard_max or hard_max > V8_MAX_CUMULATIVE_REQUESTS:
        raise ValueError("v7 cumulative request ledger or hard ceiling is inconsistent")
    if audit.get("request_attempts") != attempts or \
            capture_audit.get("cumulative_request_attempts") != cumulative:
        raise ValueError("v7 root audit request totals mismatch")
    failures = [row for row in logs if row.get("http_status") != 200]
    if len(failures) != v7_capture.get("request_failures") or \
            len(failures) != audit.get("request_failures"):
        raise ValueError("v7 failed-request rows do not reconcile")
    remaining = hard_max - cumulative
    if remaining <= 0:
        raise ValueError("v7 cumulative request ceiling is already exhausted")
    return v7_plan, v7_capture, audit, remaining


def build_v8_continuation_plan(*, v7_plan: dict, v7_capture: dict,
        v7_capture_file_sha256: str, v7_root_audit: dict,
        v7_root_audit_file_sha256: str, max_new_requests: int | None = None) -> dict:
    """Build a one-shot, deterministic continuation plan from v7 receipts."""
    plan, capture, audit, remaining = _validate_parent(v7_plan=v7_plan,
        v7_capture=v7_capture, v7_capture_file_sha256=v7_capture_file_sha256,
        v7_root_audit=v7_root_audit,
        v7_root_audit_file_sha256=v7_root_audit_file_sha256)
    if max_new_requests is None:
        max_new_requests = remaining
    if type(max_new_requests) is not int or not 1 <= max_new_requests <= remaining:
        raise ValueError("v8 request budget must fit the unspent parent cumulative ceiling")

    arm_specs = {row["arm_id"]: row for row in plan.get("arms", [])
        if isinstance(row, dict) and isinstance(row.get("arm_id"), str)}
    topics = capture.get("topics")
    if not isinstance(topics, dict) or not arm_specs:
        raise ValueError("v7 capture is missing the frozen topic or arm manifest")
    logs_by_arm: dict[str, list[dict]] = {}
    for row in capture["request_log"]:
        if not isinstance(row, dict) or row.get("arm_id") not in arm_specs:
            raise ValueError("v7 request ledger contains an unknown arm")
        logs_by_arm.setdefault(row["arm_id"], []).append(row)

    # Preserve all v5/v6 IDs from the frozen v7 plan, plus all IDs ever audited
    # or retained by v7. These are global exclusions across every v8 query arm.
    dois = {str(value).strip().casefold() for value in plan.get("excluded_parent_dois", [])
        if isinstance(value, str) and value.strip()}
    pmcids = {str(value).strip().casefold() for value in plan.get("excluded_parent_pmcids", [])
        if isinstance(value, str) and value.strip()}
    for topic in topics.values():
        if not isinstance(topic, dict):
            raise ValueError("v7 topic record is malformed")
        for field in _IDENTITY_FIELDS:
            rows = topic.get(field, [])
            if not isinstance(rows, list):
                raise ValueError(f"v7 {field} identity ledger is malformed")
            for row in rows:
                doi, pmcid = _identity(row)
                if doi:
                    dois.add(doi)
                if pmcid:
                    pmcids.add(pmcid)

    failed_arms: set[str] = set()
    failed_cursors: list[str] = []
    available: list[dict] = []
    for arm_id, arm in sorted(arm_specs.items()):
        topic = topics.get(arm.get("topic_id"))
        if not isinstance(topic, dict):
            raise ValueError(f"v7 capture lacks topic for arm {arm_id}")
        if topic.get("slot_kind") == "conflict":
            continue
        accepted_count = topic.get("parent_record_count", 0) + \
            topic.get("v6_record_count", 0) + len(topic.get("records_added", []))
        if accepted_count >= 2:
            continue
        requests = sorted(logs_by_arm.get(arm_id, []), key=lambda row: row.get("page_index", -1))
        failures = [row for row in requests if row.get("http_status") != 200]
        if failures:
            failed_arms.add(arm_id)
            failed_cursors.extend(row.get("cursor_mark", "") for row in failures
                if isinstance(row.get("cursor_mark"), str))
            # A request with unknown receipt is never replayed or skipped over.
            continue
        page_rows = [row for row in topic.get("core_pages", [])
            if isinstance(row, dict) and row.get("arm_id") == arm_id]
        page_rows.sort(key=lambda row: row.get("page_index", -1))
        if len(page_rows) != len(requests):
            raise ValueError(f"v7 page receipts do not reconcile for arm {arm_id}")
        if not page_rows:
            continue
        for req, page in zip(requests, page_rows):
            if req.get("cursor_mark") != page.get("cursor_mark") or \
                    req.get("page_index") != page.get("page_index"):
                raise ValueError(f"v7 cursor/page request binding mismatch for arm {arm_id}")
        latest = page_rows[-1]
        cursor = latest.get("next_cursor_mark")
        previous = latest.get("cursor_mark")
        if cursor is None or not isinstance(cursor, str) or not cursor.strip():
            continue  # v7 already observed the exact terminal page state
        if cursor == previous:
            continue  # Europe PMC's echoed cursor means exhaustion
        consumed = {row.get("cursor_mark") for row in requests}
        if cursor in consumed:
            raise ValueError(f"v7 next cursor would replay a consumed page for arm {arm_id}")
        if arm.get("query_sha256") != hashlib.sha256(arm.get("query", "").encode()).hexdigest():
            raise ValueError(f"frozen v7 query hash mismatch for arm {arm_id}")
        page_cap = arm.get("max_pages_with_reserve")
        if type(page_cap) is not int:
            raise ValueError(f"v7 page cap is malformed for arm {arm_id}")
        remaining_pages = page_cap - len(requests)
        if remaining_pages <= 0:
            continue
        available.append({"arm_id": arm_id, "topic_id": arm["topic_id"],
            "arm_type": arm["arm_type"], "query": arm["query"],
            "query_sha256": arm["query_sha256"], "page_size": arm.get("page_size", V8_PAGE_SIZE),
            "start_cursor": cursor, "v7_terminal_response_sha256": latest.get("response_sha256"),
            "v7_last_successful_page_index": latest.get("page_index"),
            "v7_successful_pages": len(requests), "remaining_v7_arm_page_cap": remaining_pages})

    # Allocate round-robin at topic level. A topic gets its first safe cursor
    # extension before any topic receives a second, then continue to exhaustion
    # of the frozen per-arm caps or the global parent-bound ceiling.
    allocations = {row["arm_id"]: 0 for row in available}
    cursors = {row["arm_id"]: row["start_cursor"] for row in available}
    capacity = {row["arm_id"]: row["remaining_v7_arm_page_cap"] for row in available}
    by_topic: dict[str, list[dict]] = {}
    for row in sorted(available, key=lambda x: (x["topic_id"], x["arm_id"])):
        by_topic.setdefault(row["topic_id"], []).append(row)
    topic_ids = sorted(by_topic)
    schedule: list[dict] = []
    while len(schedule) < max_new_requests:
        progressed = False
        for topic_id in topic_ids:
            if len(schedule) >= max_new_requests:
                break
            for row in by_topic[topic_id]:
                arm_id = row["arm_id"]
                if allocations[arm_id] >= capacity[arm_id]:
                    continue
                page_number = allocations[arm_id] + 1
                schedule.append({"request_index": len(schedule) + 1, "topic_id": topic_id,
                    "arm_id": arm_id, "page_ordinal_for_v8": page_number,
                    "cursor": cursors[arm_id], "query_sha256": row["query_sha256"],
                    "page_size": row["page_size"],
                    "require_previous_response_checkpoint": page_number > 1})
                allocations[arm_id] += 1
                # Future cursors are intentionally unknown until Europe PMC
                # responds; the executor must checkpoint and use that response.
                cursors[arm_id] = None
                progressed = True
                break
        if not progressed:
            break
    if not schedule:
        raise ValueError("v7 has no safe, unexhausted cursor available for continuation")

    arms = []
    for row in available:
        count = allocations[row["arm_id"]]
        if not count:
            continue
        arms.append({**row, "max_v8_pages": count,
            "future_pages_require_checkpointed_next_cursor": count > 1})
    max_new_requests = len(schedule)
    excluded_cursors = sorted(set(failed_cursors) | {
        row.get("cursor_mark") for row in capture["request_log"]
        if isinstance(row.get("cursor_mark"), str)})
    body = {
        "schema": V8_PLAN_SCHEMA,
        "parent_v7_plan_sha256": plan["plan_sha256"],
        "parent_v7_capture_sha256": capture["capture_sha256"],
        "parent_v7_capture_file_sha256": v7_capture_file_sha256,
        "parent_v7_root_audit_sha256": audit["root_audit_sha256"],
        "parent_v7_root_audit_file_sha256": v7_root_audit_file_sha256,
        "parent_v7_cumulative_attempts": capture["cumulative_request_attempts"],
        "cumulative_hard_max": plan["cumulative_request_hard_max"],
        "unspent_parent_cumulative_budget": plan["cumulative_request_hard_max"] -
            capture["cumulative_request_attempts"],
        "max_new_requests": max_new_requests,
        "minimum_request_interval_seconds": V8_MIN_REQUEST_INTERVAL_SECONDS,
        "api_endpoint": V8_API,
        "page_size": V8_PAGE_SIZE,
        "source_acceptance_gate_relaxed": False,
        "scope": "cursor continuation only; no new query arms, no cohort, labels, answers or scores",
        "current_complete_primary_groups": capture["primary_groups_complete"],
        "required_complete_primary_groups": 48,
        "incomplete_primary_topic_ids": sorted(
            tid for tid, row in topics.items() if row.get("slot_kind") != "conflict" and
            row.get("parent_record_count", 0) + row.get("v6_record_count", 0) +
            len(row.get("records_added", [])) < 2),
        "arms": arms,
        "schedule": schedule,
        "failed_arms_excluded": sorted(failed_arms),
        "failed_cursors_excluded": sorted(set(failed_cursors)),
        "excluded_cursors": excluded_cursors,
        "excluded_dois": sorted(dois),
        "excluded_pmcids": sorted(pmcids),
        "identity_exclusion_sha256": sha256_obj({"dois": sorted(dois),
            "pmcids": sorted(pmcids)}),
        "identity_exclusion_policy": {"replay_identity": False,
            "v5_v6_id_source": "frozen v7 parent identity exclusion sets",
            "v7_id_source": "candidate audit, accepted records, additions, and exclusions",
            "enforce_before_acceptance_or_storage": True},
        "cursor_policy": {"continue_only_from_latest_successful_next_cursor": True,
            "never_replay_any_v7_request_cursor": True,
            "failed_in_flight_page_replay": False,
            "failed_arm_retry": False,
            "checkpoint_response_before_next_page": True,
            "no_retry_after_ambiguous_request_intent": True},
        "authorization_required": True,
    }
    body["plan_sha256"] = sha256_obj(body)
    return body


def summarize_v8_continuation_preflight(plan: dict) -> dict:
    """Check the stable preflight surface; makes no network or model calls."""
    if not isinstance(plan, dict) or plan.get("schema") != V8_PLAN_SCHEMA:
        raise ValueError("unsupported G09 v8 continuation plan")
    if plan.get("plan_sha256") != _payload_sha(plan, "plan_sha256"):
        raise ValueError("v8 plan hash mismatch")
    schedule = plan.get("schedule")
    arms = plan.get("arms")
    if not isinstance(schedule, list) or not schedule or not isinstance(arms, list) or not arms:
        raise ValueError("v8 plan requires a non-empty safe-cursor schedule")
    if len(schedule) != plan.get("max_new_requests") or \
            plan.get("max_new_requests") > plan.get("unspent_parent_cumulative_budget"):
        raise ValueError("v8 schedule exceeds its parent-bound request budget")
    if plan.get("parent_v7_cumulative_attempts") + len(schedule) > plan.get("cumulative_hard_max"):
        raise ValueError("v8 schedule exceeds cumulative hard maximum")
    if plan.get("source_acceptance_gate_relaxed") is not False or \
            plan.get("authorization_required") is not True:
        raise ValueError("v8 plan relaxes its source gate or lacks explicit authorization")
    if plan.get("scope") != "cursor continuation only; no new query arms, no cohort, labels, answers or scores":
        raise ValueError("v8 plan scope is not source-discovery-only")
    failed = set(plan.get("failed_cursors_excluded", []))
    consumed = set(plan.get("excluded_cursors", []))
    if failed and not failed.issubset(consumed):
        raise ValueError("v8 plan does not exclude the failed in-flight cursor")
    if any(row.get("cursor") in consumed for row in schedule):
        raise ValueError("v8 schedule replays an already-consumed cursor")
    if plan.get("identity_exclusion_policy", {}).get("replay_identity") is not False:
        raise ValueError("v8 plan allows parent source identity replay")
    if plan.get("api_endpoint") != V8_API or plan.get("page_size") != V8_PAGE_SIZE:
        raise ValueError("v8 request endpoint or page size differs from frozen interface")
    interval = plan.get("minimum_request_interval_seconds")
    if type(interval) not in (int, float) or interval < V8_MIN_REQUEST_INTERVAL_SECONDS:
        raise ValueError("v8 requests are paced below the one-second minimum")
    return {"schema": "g09-v8-continuation-preflight-v1",
        "plan_sha256": plan["plan_sha256"],
        "parent_v7_capture_sha256": plan["parent_v7_capture_sha256"],
        "current_complete_primary_groups": plan["current_complete_primary_groups"],
        "required_complete_primary_groups": plan["required_complete_primary_groups"],
        "incomplete_primary_topic_count": len(plan["incomplete_primary_topic_ids"]),
        "continuation_arm_count": len(arms), "scheduled_request_slots": len(schedule),
        "max_new_requests": plan["max_new_requests"],
        "unspent_parent_cumulative_budget": plan["unspent_parent_cumulative_budget"],
        "cumulative_hard_max": plan["cumulative_hard_max"],
        "failed_arms_excluded": plan["failed_arms_excluded"],
        "source_acceptance_gate_relaxed": False,
        "network_calls_made": 0, "model_calls_made": 0,
        "cohort_created": False, "authorization_required": True}
