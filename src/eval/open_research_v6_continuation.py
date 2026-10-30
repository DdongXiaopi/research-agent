"""Bounded, hash-bound continuation for the immutable G09 v5 source capture.

This module never rewrites v5. It produces a child artifact that records only
new requests and additions, while binding the exact frozen v5 capture, plan,
and implementation identities. All source acceptance continues to use the
strict four-dimension literal PICO gate from ``open_research_sources``.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from ..util import sha256_obj
from . import open_research_sources as sources

V6_CONTINUATION_SCHEMA = "g09-topic-source-continuation-v6"
V6_PLAN_SCHEMA = "g09-topic-source-continuation-plan-v1"
V6_AUDIT_SCHEMA = "g09-topic-source-continuation-root-audit-v1"
EXPECTED_V5_CAPTURE_SHA256 = "4fee0b4a40ea989efa34586d791fdc375cc74080f507de864d1aab703d489191"
EXPECTED_V5_CAPTURE_FILE_SHA256 = "ebb1b23d946431384eef94ae32424008fc9b2daa07b169b32fb7a29b8ec58da8"
EXPECTED_V5_PLAN_SHA256 = "06e9b3ae268c13b672fe6ce17446058f9a04132c70dde6cc0f52121ea187baf3"
EXPECTED_V5_IMPLEMENTATION_SHA256 = "3083d060988db47b7751054290d05be174dd13c1975018d4dd13f7020a0ae889"
V5_REQUEST_ATTEMPTS = 160
HARD_REQUEST_MAX = 252
V6_PLANNED_REQUEST_BUDGET = 89
V6_RESERVE_REQUEST_BUDGET = 3
V6_PAGE_SIZE = 10
V6_ALTERNATIVE_QUERY_TOPIC_IDS = frozenset({
    "regular-014", "regular-028", "replacement-002", "replacement-009"})
_PICO_FIELDS = (
    ("condition", "condition_terms"), ("population", "population_terms"),
    ("intervention", "intervention_terms"), ("outcome", "outcome_terms"))


def _capture_payload_sha256(capture: dict) -> str:
    return sha256_obj({key: value for key, value in capture.items()
        if key != "capture_sha256"})


def _plan_payload_sha256(plan: dict) -> str:
    return sha256_obj({key: value for key, value in plan.items()
        if key != "plan_sha256"})


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _canonical_doi(value: object) -> str:
    return value.strip().casefold() if isinstance(value, str) else ""


def _canonical_pmcid(value: object) -> str:
    return value.strip().casefold() if isinstance(value, str) else ""


def _all_parent_identities(capture: dict) -> tuple[set[str], set[str]]:
    dois, pmcids = set(), set()
    for topic in capture.get("topics", []):
        if not isinstance(topic, dict):
            continue
        for row in topic.get("candidate_audit", []):
            if isinstance(row, dict):
                doi, pmcid = _canonical_doi(row.get("doi")), _canonical_pmcid(row.get("pmcid"))
                if doi:
                    dois.add(doi)
                if pmcid:
                    pmcids.add(pmcid)
        for row in topic.get("records", []):
            if isinstance(row, dict):
                doi, pmcid = _canonical_doi(row.get("doi")), _canonical_pmcid(row.get("pmcid"))
                if doi:
                    dois.add(doi)
                if pmcid:
                    pmcids.add(pmcid)
    return dois, pmcids


def _parent_topic_state(capture: dict, topic_spec: dict) -> dict:
    topic_id = topic_spec["topic_id"]
    topic = next((row for row in capture["topics"]
        if isinstance(row, dict) and row.get("topic_id") == topic_id), None)
    if topic is None:
        raise ValueError(f"v5 parent topic is missing: {topic_id}")
    records = topic.get("records")
    pages = topic.get("core_pages")
    if not isinstance(records, list) or not isinstance(pages, list):
        raise ValueError(f"v5 parent topic is malformed: {topic_id}")
    count = len(records)
    if count > 2:
        raise ValueError(f"v5 parent topic exceeds the two-source cap: {topic_id}")
    last = pages[-1] if pages else None
    if not isinstance(last, dict):
        raise ValueError(f"v5 incomplete topic has no terminal page receipt: {topic_id}")
    hit_count, offset, returned = last.get("hit_count"), last.get("record_offset"), last.get("records_returned")
    if any(type(value) is not int or value < 0 for value in (hit_count, offset, returned)):
        raise ValueError(f"v5 terminal page lacks valid hit-count continuity: {topic_id}")
    consumed = offset + returned
    exhausted = consumed >= hit_count or returned == 0
    next_cursor = last.get("next_cursor_mark")
    old_cursors = {page.get("cursor_mark") for page in pages if isinstance(page, dict)}
    if not exhausted:
        if not isinstance(next_cursor, str) or not next_cursor or next_cursor in old_cursors:
            raise ValueError(f"v5 terminal cursor is missing or replayed: {topic_id}")
    return {"topic": topic, "parent_record_count": count,
        "parent_record_identities": [{"doi": row.get("doi"), "pmcid": row.get("pmcid")}
            for row in records if isinstance(row, dict)],
        "parent_terminal_page": {key: last.get(key) for key in (
            "page_index", "cursor_mark", "next_cursor_mark", "response_sha256",
            "records_returned", "hit_count", "record_offset")},
        "parent_captured_cursors": [page.get("cursor_mark") for page in pages
            if isinstance(page, dict) and isinstance(page.get("cursor_mark"), str)],
        "exhausted": exhausted,
        "query_mode": ("strict_four_dimension_alternative" if exhausted else "resume_v5_cursor"),
        "initial_cursor": "*" if exhausted else next_cursor}


def _validate_v5_parent(capture: dict, capture_file_sha256: str,
        parent_plan_manifest: dict) -> dict:
    if not isinstance(capture, dict) or capture.get("schema") != sources.TOPIC_SOURCE_CAPTURE_SCHEMA:
        raise ValueError("v6 requires the frozen v5 source capture schema")
    plan = sources.validate_frozen_topic_plan(parent_plan_manifest,
        expected_plan_sha256=EXPECTED_V5_PLAN_SHA256)
    if capture.get("capture_sha256") != EXPECTED_V5_CAPTURE_SHA256 or \
            _capture_payload_sha256(capture) != EXPECTED_V5_CAPTURE_SHA256:
        raise ValueError("parent capture identity does not match immutable v5 capture")
    if capture_file_sha256 != EXPECTED_V5_CAPTURE_FILE_SHA256:
        raise ValueError("parent capture file identity does not match immutable v5 file")
    if capture.get("frozen_topic_plan_sha256") != EXPECTED_V5_PLAN_SHA256 or \
            plan["plan_sha256"] != EXPECTED_V5_PLAN_SHA256:
        raise ValueError("parent plan identity does not match immutable v5 plan")
    if capture.get("capture_implementation_sha256") != EXPECTED_V5_IMPLEMENTATION_SHA256:
        raise ValueError("parent implementation identity does not match immutable v5 implementation")
    attempts = capture.get("request_attempts")
    if attempts != V5_REQUEST_ATTEMPTS or len(capture.get("request_log", [])) != attempts:
        raise ValueError("parent v5 request ledger does not match its frozen 160 attempts")
    if capture.get("diagnostic_only") is not True or capture.get("scoreable_cohort_created") is not False:
        raise ValueError("v5 parent must remain diagnostic-only")
    topic_ids = [row.get("topic_id") for row in capture.get("topics", []) if isinstance(row, dict)]
    expected_ids = [row["topic_id"] for row in plan["topic_specs"]]
    if len(topic_ids) != 60 or len(set(topic_ids)) != 60 or set(topic_ids) != set(expected_ids):
        raise ValueError("parent source topics do not match the frozen v5 plan")
    return plan


def build_strict_four_dimension_query(topic_spec: dict, *, capture_date: str) -> str:
    """Build a narrow alternative query using the unchanged PICO term lists."""
    base = sources.source_query(capture_date)
    dimensions = []
    for _, field in _PICO_FIELDS:
        values = sources._query_terms(topic_spec.get(field), label=field)
        field_terms = [f'(TITLE:"{term}" OR ABSTRACT:"{term}")' for term in values]
        dimensions.append("(" + " OR ".join(field_terms) + ")")
    return base + " AND " + " AND ".join(dimensions)


def build_v6_continuation_plan(*, parent_capture: dict,
        parent_capture_file_sha256: str, parent_plan_manifest: dict) -> dict:
    """Freeze a continuation plan bound to the immutable v5 parent artifacts."""
    if not _valid_sha(parent_capture_file_sha256):
        raise ValueError("parent capture file hash must be a lowercase SHA-256")
    parent_plan = _validate_v5_parent(parent_capture, parent_capture_file_sha256,
        parent_plan_manifest)
    if not isinstance(parent_capture.get("topics"), list):
        raise ValueError("parent capture topics must be an array")
    specs = {row["topic_id"]: row for row in parent_plan["topic_specs"]}
    frozen_slots = []
    seen_exhausted = set()
    for topic_id in sorted(specs):
        spec = specs[topic_id]
        if spec["slot_kind"] == "conflict":
            continue
        topic = next(row for row in parent_capture["topics"] if row["topic_id"] == topic_id)
        if topic.get("status") == "complete" and len(topic.get("records", [])) == 2:
            continue
        state = _parent_topic_state(parent_capture, spec)
        if state["exhausted"]:
            seen_exhausted.add(topic_id)
            if topic_id not in V6_ALTERNATIVE_QUERY_TOPIC_IDS:
                raise ValueError(f"no reviewed alternative query is frozen for exhausted slot {topic_id}")
            query = build_strict_four_dimension_query(spec, capture_date=parent_plan["capture_date"])
        else:
            query = topic.get("query")
            if not isinstance(query, str) or not query:
                raise ValueError(f"v5 query is missing for resumable topic {topic_id}")
        frozen_slots.append({"topic_id": topic_id, "slot_kind": spec["slot_kind"],
            "parent_record_count": state["parent_record_count"],
            "parent_record_identities": state["parent_record_identities"],
            "parent_terminal_page": state["parent_terminal_page"],
            "parent_captured_cursors": state["parent_captured_cursors"],
            "query_mode": state["query_mode"], "query": query,
            "initial_cursor": state["initial_cursor"]})
    if seen_exhausted != set(V6_ALTERNATIVE_QUERY_TOPIC_IDS):
        missing = sorted(set(V6_ALTERNATIVE_QUERY_TOPIC_IDS) - seen_exhausted)
        extra = sorted(seen_exhausted - set(V6_ALTERNATIVE_QUERY_TOPIC_IDS))
        raise ValueError(f"v5 exhausted primary slots differ from reviewed set; missing={missing}, extra={extra}")
    body = {"schema": V6_PLAN_SCHEMA,
        "parent_capture_sha256": parent_capture["capture_sha256"],
        "parent_capture_file_sha256": parent_capture_file_sha256,
        "parent_plan_sha256": parent_plan["plan_sha256"],
        "parent_implementation_sha256": parent_capture["capture_implementation_sha256"],
        "continuation_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parent_request_attempts": V5_REQUEST_ATTEMPTS,
        "hard_request_max": HARD_REQUEST_MAX,
        "planned_additional_requests": min(V6_PLANNED_REQUEST_BUDGET,
            HARD_REQUEST_MAX - V5_REQUEST_ATTEMPTS - V6_RESERVE_REQUEST_BUDGET),
        "reserve_requests": V6_RESERVE_REQUEST_BUDGET,
        "target_records_per_topic": 2, "page_size": V6_PAGE_SIZE,
        "alternative_query_topic_ids": sorted(V6_ALTERNATIVE_QUERY_TOPIC_IDS),
        "topic_specs_sha256": sha256_obj(parent_plan["topic_specs"]),
        "slots": frozen_slots,
        "selection_policy": {
            "schedule": "topic_id lexical order; one page per eligible slot per round",
            "continuation": "resume exactly from v5 terminal nextCursorMark; never replay a captured cursor",
            "reserve": "after 89 planned pages only; at most 3 pages if fewer than 48 primary groups are complete and an unfilled slot remains searchable",
            "identity_exclusion": "seed global DOI/PMCID sets from every v5 candidate_audit and accepted record; never request a previously seen source identity again",
            "acceptance": "identity plus explicit CC BY plus abstract length >=400 plus literal title/abstract evidence for all four frozen PICO dimensions",
            "rejected_abstract_retention": "retain abstract and four-dimension match evidence only after valid DOI+PMCID, explicit CC BY and minimum abstract length pass",
            "cohort": "diagnostic-only; typed cohort, rubric, assistant answers and scoring are out of scope",
        }}
    return {**body, "plan_sha256": sha256_obj(body)}


def _verify_v6_plan(plan: dict, expected_sha256: str, parent_capture: dict,
        capture_file_sha256: str, parent_plan_manifest: dict) -> dict:
    if not isinstance(plan, dict) or plan.get("schema") != V6_PLAN_SCHEMA:
        raise ValueError("unsupported v6 continuation plan")
    actual = _plan_payload_sha256(plan)
    if not _valid_sha(expected_sha256) or actual != expected_sha256 or plan.get("plan_sha256") != actual:
        raise ValueError("v6 continuation plan hash mismatch")
    parent_plan = _validate_v5_parent(parent_capture, capture_file_sha256, parent_plan_manifest)
    expected_bindings = {"parent_capture_sha256": parent_capture["capture_sha256"],
        "parent_capture_file_sha256": capture_file_sha256,
        "parent_plan_sha256": parent_plan["plan_sha256"],
        "parent_implementation_sha256": parent_capture["capture_implementation_sha256"],
        "continuation_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parent_request_attempts": V5_REQUEST_ATTEMPTS,
        "hard_request_max": HARD_REQUEST_MAX,
        "planned_additional_requests": V6_PLANNED_REQUEST_BUDGET,
        "reserve_requests": V6_RESERVE_REQUEST_BUDGET,
        "topic_specs_sha256": sha256_obj(parent_plan["topic_specs"])}
    if any(plan.get(key) != value for key, value in expected_bindings.items()):
        raise ValueError("v6 continuation plan parent/budget binding mismatch")
    derived = build_v6_continuation_plan(parent_capture=parent_capture,
        parent_capture_file_sha256=capture_file_sha256,
        parent_plan_manifest=parent_plan_manifest)
    if plan != derived:
        raise ValueError("v6 continuation plan differs from parent-derived continuation slots")
    return parent_plan


def _response_json(url: str, *, opener) -> tuple[dict, bytes]:
    status, raw = sources.bounded_source_response(
        url, opener=opener, user_agent="G09-research-assistant-evaluation-v6/1.0")
    if status != 200:
        raise RuntimeError(f"Europe PMC request failed with HTTP {status}")
    import json
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Europe PMC returned a non-object page")
    return result, raw


def _build_request_url(query: str, cursor: str) -> str:
    return sources.API_ROOT + "/search?" + urlencode({"query": query,
        "format": "json", "resulttype": "core", "pageSize": V6_PAGE_SIZE,
        "cursorMark": cursor})


def continue_v5_capture(*, parent_capture: dict, parent_capture_file_sha256: str,
        parent_plan_manifest: dict, continuation_plan: dict, verified_plan_sha256: str,
        captured_at: str, opener=urlopen, request_budget: int | None = None) -> dict:
    """Continue v5 cursors deterministically without altering the parent capture."""
    try:
        timestamp = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("captured_at must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError("captured_at must include a timezone")
    parent_plan = _verify_v6_plan(continuation_plan, verified_plan_sha256,
        parent_capture, parent_capture_file_sha256, parent_plan_manifest)
    budget = HARD_REQUEST_MAX - V5_REQUEST_ATTEMPTS if request_budget is None else request_budget
    if type(budget) is not int or not 0 <= budget <= HARD_REQUEST_MAX - V5_REQUEST_ATTEMPTS:
        raise ValueError("v6 additional request budget must be between 0 and 92")
    planned_budget = min(V6_PLANNED_REQUEST_BUDGET, budget)
    reserve_budget = min(V6_RESERVE_REQUEST_BUDGET, budget - planned_budget)
    plan_by_id = {row["topic_id"]: row for row in parent_plan["topic_specs"]}
    parent_topics = {row["topic_id"]: row for row in parent_capture["topics"]}
    frozen_slots = {row["topic_id"]: row for row in continuation_plan["slots"]}
    already_seen_dois, already_seen_pmcids = _all_parent_identities(parent_capture)
    additions_seen_dois, additions_seen_pmcids = set(), set()
    outputs = {}
    for topic in parent_capture["topics"]:
        topic_id = topic["topic_id"]
        spec = plan_by_id[topic_id]
        if topic_id in frozen_slots:
            slot = frozen_slots[topic_id]
            outputs[topic_id] = {"topic_id": topic_id, "slot_kind": spec["slot_kind"],
                "query_mode": slot["query_mode"], "query": slot["query"],
                "parent_record_count": len(topic["records"]),
                "parent_record_identities": slot["parent_record_identities"],
                "records_added": [], "core_pages": [], "candidate_audit": [],
                "exclusions": [], "status": "incomplete", "failure_reason": None,
                "search_exhausted": False}
        else:
            outputs[topic_id] = {"topic_id": topic_id, "slot_kind": spec["slot_kind"],
                "query_mode": "parent_unchanged", "query": None,
                "parent_record_count": len(topic["records"]),
                "parent_record_identities": [{"doi": row.get("doi"), "pmcid": row.get("pmcid")}
                    for row in topic["records"] if isinstance(row, dict)],
                "records_added": [], "core_pages": [], "candidate_audit": [],
                "exclusions": [], "status": "complete" if len(topic["records"]) == 2 else "not_searched",
                "failure_reason": None, "search_exhausted": False}

    states = {}
    for topic_id, slot in frozen_slots.items():
        topic = parent_topics[topic_id]
        spec = plan_by_id[topic_id]
        state = _parent_topic_state(parent_capture, spec)
        prior_pages = topic["core_pages"]
        prior_seen_cursors = {page.get("cursor_mark") for page in prior_pages}
        states[topic_id] = {"slot": slot, "spec": spec,
            "cursor": slot["initial_cursor"], "record_offset": (
                state["parent_terminal_page"]["record_offset"] +
                state["parent_terminal_page"]["records_returned"]
                if slot["query_mode"] == "resume_v5_cursor" else 0),
            "hit_count": (state["parent_terminal_page"]["hit_count"]
                if slot["query_mode"] == "resume_v5_cursor" else None),
            "seen_cursors": set(prior_seen_cursors) if slot["query_mode"] == "resume_v5_cursor" else set(),
            "page_index": len(prior_pages) if slot["query_mode"] == "resume_v5_cursor" else 0,
            "done": False}

    request_log = []
    planned_used = 0
    reserve_used = 0

    def completed_primary_groups() -> int:
        count = 0
        for topic_id, original in parent_topics.items():
            if original["slot_kind"] == "conflict":
                continue
            extra = outputs[topic_id]["records_added"]
            if len(original["records"]) + len(extra) >= 2:
                count += 1
        return count

    def available(topic_id: str) -> bool:
        original, output = parent_topics[topic_id], outputs[topic_id]
        return (not states[topic_id]["done"] and
            len(original["records"]) + len(output["records_added"]) < 2 and
            output["failure_reason"] is None)

    def request_one(topic_id: str) -> None:
        state = states[topic_id]
        slot, spec, output = state["slot"], state["spec"], outputs[topic_id]
        cursor, query = state["cursor"], slot["query"]
        if cursor in state["seen_cursors"]:
            output["failure_reason"] = "cursor_replay_prevented"
            state["done"] = True
            return
        url = _build_request_url(query, cursor)
        log = {"topic_id": topic_id, "query_mode": slot["query_mode"],
            "phase": "planned" if planned_used < planned_budget else "reserve",
            "page_index": state["page_index"] + 1, "cursor_mark": cursor,
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "method": "GET", "url": url}
        try:
            response, raw = _response_json(url, opener=opener)
        except Exception as exc:
            log.update({"http_status": None, "error_type": type(exc).__name__,
                "error": str(exc)[:500]})
            request_log.append(log)
            output["failure_reason"] = "core_search_request_failed"
            state["done"] = True
            return
        page_hash = hashlib.sha256(raw).hexdigest()
        log.update({"http_status": 200, "response_bytes": len(raw),
            "response_sha256": page_hash})
        request_log.append(log)
        rows = (response.get("resultList") or {}).get("result")
        if not isinstance(rows, list) or len(rows) > V6_PAGE_SIZE:
            log["error_type"] = "invalid_core_search_response"
            output["failure_reason"] = "invalid_core_search_response"
            state["done"] = True
            return
        next_cursor, hit_count = response.get("nextCursorMark"), response.get("hitCount")
        if type(hit_count) is not int or hit_count < 0:
            log["error_type"] = "invalid_hit_count"
            output["failure_reason"] = "invalid_hit_count"
            state["done"] = True
            return
        if state["hit_count"] is not None and hit_count != state["hit_count"]:
            log["error_type"] = "hit_count_changed_during_cursor_continuation"
            output["failure_reason"] = "hit_count_changed_during_cursor_continuation"
            state["done"] = True
            return
        page_index = state["page_index"] + 1
        page_offset = state["record_offset"]
        output["core_pages"].append({"page_index": page_index,
            "cursor_mark": cursor,
            "next_cursor_mark": next_cursor if isinstance(next_cursor, str) else None,
            "response_sha256": page_hash, "records_returned": len(rows),
            "hit_count": hit_count, "record_offset": page_offset})
        state["page_index"] = page_index
        accepted_this_page = False
        for result_index, record in enumerate(rows):
            rank = page_offset + result_index
            if not isinstance(record, dict):
                output["exclusions"].append({"reason": "invalid_core_record", "core_rank": rank})
                continue
            doi = _canonical_doi(record.get("doi"))
            pmcid = _canonical_pmcid(record.get("pmcid"))
            if not doi or not re.fullmatch(r"pmc\d+", pmcid):
                output["exclusions"].append({"doi": doi or None, "pmcid": pmcid or None,
                    "reason": "missing_source_identity", "core_rank": rank,
                    "response_sha256": page_hash})
                continue
            if any(doi.startswith(prefix) for prefix in sources.HISTORICAL_DOI_PREFIX_EXCLUSIONS):
                log["identity_error"] = "PLOS DOI family is forbidden in v6 source records"
                output["failure_reason"] = "forbidden_source_family"
                state["done"] = True
                break
            if doi in already_seen_dois or doi in additions_seen_dois or \
                    pmcid in already_seen_pmcids or pmcid in additions_seen_pmcids:
                output["exclusions"].append({"doi": doi, "pmcid": pmcid,
                    "reason": "parent_seen_source_identity" if
                    doi in already_seen_dois or pmcid in already_seen_pmcids else
                    "duplicate_source_across_v6_slots", "core_rank": rank,
                    "response_sha256": page_hash})
                continue
            # Consume every newly observed valid identity, including ineligible
            # sources, so alternative queries cannot buy another evaluation of it.
            additions_seen_dois.add(doi)
            additions_seen_pmcids.add(pmcid)
            audit = {"doi": doi, "pmcid": pmcid, "core_rank": rank,
                "page_index": page_index, "result_index": result_index,
                "core_response_sha256": page_hash}
            output["candidate_audit"].append(audit)
            if str(record.get("license") or "").strip().casefold() != "cc by":
                output["exclusions"].append({"doi": doi, "pmcid": pmcid,
                    "reason": "core_license_not_cc_by", "core_rank": rank,
                    "response_sha256": page_hash})
                continue
            abstract = record.get("abstractText")
            if not isinstance(abstract, str) or len(abstract.strip()) < 400:
                output["exclusions"].append({"doi": doi, "pmcid": pmcid,
                    "reason": "abstract_missing_or_too_short", "core_rank": rank,
                    "response_sha256": page_hash})
                continue
            relevance = sources.assess_topic_pico_relevance(record, spec)
            audit["abstract_pico_relevance"] = relevance
            if not relevance["eligible"]:
                output["exclusions"].append({"doi": doi, "pmcid": pmcid,
                    "reason": "pico_relevance_missing_dimensions", "core_rank": rank,
                    "pico_relevance": relevance, "abstract_text": abstract,
                    "license": "cc by", "abstract_length": len(abstract.strip()),
                    "response_sha256": page_hash})
                continue
            output["records_added"].append(record)
            accepted_this_page = True
            if len(parent_topics[topic_id]["records"]) + len(output["records_added"]) == 2:
                break
        state["hit_count"] = hit_count
        consumed = page_offset + len(rows)
        state["record_offset"] = consumed
        if output["failure_reason"]:
            return
        if len(parent_topics[topic_id]["records"]) + len(output["records_added"]) >= 2:
            state["done"] = True
            return
        if not rows or consumed >= hit_count:
            output["search_exhausted"] = True
            state["done"] = True
            return
        if not isinstance(next_cursor, str) or not next_cursor or \
                next_cursor == cursor or next_cursor in state["seen_cursors"]:
            output["failure_reason"] = "cursor_missing_or_repeated"
            state["done"] = True
            return
        state["seen_cursors"].add(cursor)
        state["cursor"] = next_cursor

    def spend_phase(phase_budget: int, phase: str) -> int:
        nonlocal planned_used, reserve_used
        used = 0
        eligible_ids = sorted(states)
        while used < phase_budget:
            progressed = False
            for topic_id in eligible_ids:
                if used >= phase_budget:
                    break
                if not available(topic_id):
                    continue
                if completed_primary_groups() >= 60:
                    break
                request_one(topic_id)
                used += 1
                progressed = True
                if phase == "planned":
                    planned_used += 1
                else:
                    reserve_used += 1
            if not progressed:
                break
        return used

    spend_phase(planned_budget, "planned")
    reserve_gate = {"planned_phase_complete": planned_used == planned_budget,
        "primary_groups_before_reserve": completed_primary_groups(),
        "searchable_slots_before_reserve": [topic_id for topic_id in sorted(states)
            if available(topic_id)]}
    reserve_gate["authorized"] = (planned_budget > 0 and
        reserve_gate["planned_phase_complete"] and
        reserve_gate["primary_groups_before_reserve"] < 48 and
        bool(reserve_gate["searchable_slots_before_reserve"]))
    if reserve_budget and reserve_gate["authorized"]:
        spend_phase(reserve_budget, "reserve")

    for topic_id, output in outputs.items():
        original = parent_topics[topic_id]
        total = len(original["records"]) + len(output["records_added"])
        if total >= 2:
            output["status"] = "complete"
        elif output["failure_reason"]:
            output["status"] = "incomplete_error"
        elif output["search_exhausted"]:
            output["status"] = "incomplete_exhausted"
        elif output["query_mode"] == "parent_unchanged":
            output["status"] = "complete" if total == 2 else "not_searched"
        else:
            output["status"] = "incomplete_searchable"

    cumulative = V5_REQUEST_ATTEMPTS + len(request_log)
    body = {"schema": V6_CONTINUATION_SCHEMA,
        "captured_at": captured_at, "parent_capture_sha256": parent_capture["capture_sha256"],
        "parent_capture_file_sha256": parent_capture_file_sha256,
        "parent_plan_sha256": parent_plan["plan_sha256"],
        "parent_implementation_sha256": parent_capture["capture_implementation_sha256"],
        "continuation_implementation_sha256": continuation_plan["continuation_implementation_sha256"],
        "continuation_plan_sha256": continuation_plan["plan_sha256"],
        "frozen_slots": continuation_plan["slots"],
        "hard_request_max": HARD_REQUEST_MAX, "parent_request_attempts": V5_REQUEST_ATTEMPTS,
        "planned_request_budget": planned_budget, "planned_requests_used": planned_used,
        "reserve_request_budget": reserve_budget, "reserve_requests_used": reserve_used,
        "reserve_gate": reserve_gate,
        "additional_request_budget": budget, "request_attempts": len(request_log),
        "cumulative_request_attempts": cumulative,
        "request_failures": sum(1 for row in request_log if row.get("http_status") != 200 or
            row.get("error_type") or row.get("identity_error")),
        "request_log": request_log, "topics": outputs,
        "accepted_additions": sum(len(row["records_added"]) for row in outputs.values()),
        "primary_groups_complete": completed_primary_groups(),
        "diagnostic_only": True, "scoreable_cohort_created": False,
        "license_policy": "Only explicit CC BY core records with complete DOI/PMCID and abstracts >=400 characters may be accepted or retain a rejected full abstract.",
        "pico_policy": "Literal normalized title/abstract phrase required for each of the four frozen PICO dimensions; no synonym expansion or semantic judge.",
        "source_plan_sha256": sha256_obj(parent_plan["topic_specs"])}
    body["capture_sha256"] = sha256_obj(body)
    return body


def audit_v6_continuation(result: dict, *, parent_capture: dict,
        parent_plan_manifest: dict) -> dict:
    """Create an independently recomputable, compact audit for the v6 child."""
    parent_plan = _validate_v5_parent(parent_capture,
        result.get("parent_capture_file_sha256"), parent_plan_manifest)
    artifact_hash_ok = result.get("capture_sha256") == _capture_payload_sha256(result)
    identities_doi, identities_pmcid = _all_parent_identities(parent_capture)
    new_dois, new_pmcids = set(), set()
    accepted_compliant = True
    spec_by_id = {row["topic_id"]: row for row in parent_plan["topic_specs"]}
    for topic_id, topic in result.get("topics", {}).items():
        if len(topic.get("records_added", [])) > max(0, 2 - topic.get("parent_record_count", 0)):
            accepted_compliant = False
        for record in topic.get("records_added", []):
            doi, pmcid = _canonical_doi(record.get("doi")), _canonical_pmcid(record.get("pmcid"))
            if (not doi or not re.fullmatch(r"pmc\d+", pmcid) or
                    str(record.get("license") or "").strip().casefold() != "cc by" or
                    not isinstance(record.get("abstractText"), str) or
                    len(record["abstractText"].strip()) < 400 or
                    not sources.assess_topic_pico_relevance(record, spec_by_id[topic_id])["eligible"] or
                    doi in identities_doi or pmcid in identities_pmcid or
                    doi in new_dois or pmcid in new_pmcids):
                accepted_compliant = False
            new_dois.add(doi)
            new_pmcids.add(pmcid)
    retention_compliant = True
    for topic in result.get("topics", {}).values():
        for exclusion in topic.get("exclusions", []):
            if "abstract_text" not in exclusion:
                continue
            abstract = exclusion.get("abstract_text")
            pico = exclusion.get("pico_relevance")
            if (exclusion.get("reason") != "pico_relevance_missing_dimensions" or
                    not _canonical_doi(exclusion.get("doi")) or
                    not re.fullmatch(r"pmc\d+", _canonical_pmcid(exclusion.get("pmcid"))) or
                    exclusion.get("license") != "cc by" or
                    not isinstance(abstract, str) or len(abstract.strip()) < 400 or
                    exclusion.get("abstract_length") != len(abstract.strip()) or
                    not isinstance(pico, dict) or pico.get("eligible") is not False or
                    set((pico.get("matched_terms") or {}).keys()) != set(sources.PICO_DIMENSIONS)):
                retention_compliant = False
    unique_requests = set()
    cursor_continuity_valid = True
    per_topic_last = {}
    frozen_by_id = {row["topic_id"]: row for row in result.get("frozen_slots", [])
        if isinstance(row, dict) and isinstance(row.get("topic_id"), str)}
    page_receipts = {row["topic_id"]: {page.get("page_index"): page
        for page in row.get("core_pages", [])} for row in result.get("topics", {}).values()
        if isinstance(row, dict) and row.get("query_mode") != "parent_unchanged"}
    for request in result.get("request_log", []):
        topic_id = request.get("topic_id")
        slot = frozen_by_id.get(topic_id)
        if slot is None:
            cursor_continuity_valid = False
            continue
        key = (topic_id, request.get("query_sha256"), request.get("cursor_mark"))
        if key in unique_requests:
            cursor_continuity_valid = False
        unique_requests.add(key)
        previous = per_topic_last.get(topic_id)
        expected_cursor = previous if previous is not None else slot.get("initial_cursor")
        if request.get("cursor_mark") != expected_cursor:
            cursor_continuity_valid = False
        # Each page receipt binds the returned next cursor for this request.
        page_receipt = page_receipts.get(topic_id, {}).get(request.get("page_index"))
        expected_query_sha = hashlib.sha256(slot.get("query", "").encode("utf-8")).hexdigest()
        if (page_receipt is None or
                page_receipt.get("cursor_mark") != request.get("cursor_mark") or
                page_receipt.get("response_sha256") != request.get("response_sha256") or
                request.get("query_sha256") != expected_query_sha):
            cursor_continuity_valid = False
        if previous is None and slot.get("query_mode") == "resume_v5_cursor":
            terminal = slot.get("parent_terminal_page") or {}
            if (request.get("cursor_mark") != terminal.get("next_cursor_mark") or
                    request.get("cursor_mark") == terminal.get("cursor_mark") or
                    request.get("cursor_mark") in slot.get("parent_captured_cursors", [])):
                cursor_continuity_valid = False
        if previous is None and slot.get("query_mode") == "strict_four_dimension_alternative":
            if request.get("cursor_mark") != "*":
                cursor_continuity_valid = False
            expected_query = build_strict_four_dimension_query(spec_by_id[topic_id],
                capture_date=parent_plan["capture_date"])
            if slot.get("query") != expected_query:
                cursor_continuity_valid = False
        per_topic_last[topic_id] = page_receipt.get("next_cursor_mark") if page_receipt else None
    cumulative = result.get("parent_request_attempts", -1) + result.get("request_attempts", -1)
    reserve_gate = result.get("reserve_gate")
    reserve_gate_valid = (isinstance(reserve_gate, dict) and
        (result.get("reserve_requests_used", 0) == 0 or
            (reserve_gate.get("authorized") is True and
             reserve_gate.get("planned_phase_complete") is True and
             reserve_gate.get("primary_groups_before_reserve", 60) < 48 and
             bool(reserve_gate.get("searchable_slots_before_reserve")))))
    budget_valid = (result.get("hard_request_max") == HARD_REQUEST_MAX and
        result.get("parent_request_attempts") == V5_REQUEST_ATTEMPTS and
        result.get("request_attempts") == len(result.get("request_log", [])) and
        cumulative <= HARD_REQUEST_MAX and
        result.get("planned_request_budget", 0) <= V6_PLANNED_REQUEST_BUDGET and
        result.get("reserve_request_budget", 0) <= V6_RESERVE_REQUEST_BUDGET and
        result.get("additional_request_budget") ==
            result.get("planned_request_budget", 0) + result.get("reserve_request_budget", 0) and
        result.get("cumulative_request_attempts") == cumulative and
        result.get("planned_requests_used", 0) <= result.get("planned_request_budget", -1) and
        result.get("reserve_requests_used", 0) <= result.get("reserve_request_budget", -1))
    budget_valid = budget_valid and reserve_gate_valid
    implementation_hash_valid = (result.get("continuation_implementation_sha256") ==
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    request_phases = [row.get("phase") for row in result.get("request_log", [])]
    planned_phase_count = request_phases.count("planned")
    reserve_phase_count = request_phases.count("reserve")
    phases_valid = (planned_phase_count == result.get("planned_requests_used") and
        reserve_phase_count == result.get("reserve_requests_used") and
        all(phase in {"planned", "reserve"} for phase in request_phases) and
        not ("planned" in request_phases[request_phases.index("reserve"):] if
            "reserve" in request_phases else False))
    budget_valid = budget_valid and phases_valid
    body = {"schema": V6_AUDIT_SCHEMA,
        "parent_capture_sha256": parent_capture["capture_sha256"],
        "parent_plan_sha256": parent_plan["plan_sha256"],
        "continuation_plan_sha256": result.get("continuation_plan_sha256"),
        "v6_capture_sha256": result.get("capture_sha256"),
        "artifact_hash_valid": artifact_hash_ok,
        "implementation_hash_valid": implementation_hash_valid,
        "budget_valid": budget_valid,
        "reserve_gate_valid": reserve_gate_valid,
        "cursor_continuity_valid": cursor_continuity_valid,
        "accepted_records_compliant": accepted_compliant,
        "rejected_abstract_retention_compliant": retention_compliant,
        "request_phase_accounting_valid": phases_valid,
        "parent_request_attempts": result.get("parent_request_attempts"),
        "v6_request_attempts": result.get("request_attempts"),
        "planned_requests_used": result.get("planned_requests_used"),
        "reserve_requests_used": result.get("reserve_requests_used"),
        "cumulative_request_attempts": cumulative,
        "primary_groups_complete": result.get("primary_groups_complete"),
        "diagnostic_only": result.get("diagnostic_only") is True,
        "scoreable_cohort_created": result.get("scoreable_cohort_created") is True}
    return {**body, "audit_sha256": sha256_obj(body)}
