"""Offline, hash-bound G09 v7 multi-arm source-discovery plan builder.

This module creates no HTTP requests. Query arms change candidate discovery
only; source identity, deduplication, and strict literal four-way PICO
acceptance remain the capture adapter's responsibility. License is provenance,
not an eligibility requirement for this personal project.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..util import sha256_obj
from . import open_research_sources as sources
from . import open_research_v6_continuation as v6

V7_PLAN_SCHEMA = "g09-v7-query-arm-plan-v1"
V7_AUDIT_SCHEMA = "g09-topic-source-continuation-root-audit-v1"
V7_MAX_NEW_REQUESTS = 252
V7_BASE_PAGES_PER_ARM = 5
V7_RESERVE_MAX_PER_ARM = 3
V7_MIN_REQUEST_INTERVAL_SECONDS = 1.0
V7_ARMS = {
    "condition_population": ("condition_terms", "population_terms"),
    "condition_outcome": ("condition_terms", "outcome_terms"),
}
_IDENTITY_KEYS = ("candidate_audit", "records", "records_added", "exclusions")


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _payload_sha(value: dict, field: str) -> str:
    return sha256_obj({key: item for key, item in value.items() if key != field})


def _identity(row: object) -> tuple[str, str]:
    if not isinstance(row, dict):
        return "", ""
    doi, pmcid = row.get("doi"), row.get("pmcid")
    return (
        doi.strip().casefold() if isinstance(doi, str) else "",
        pmcid.strip().casefold() if isinstance(pmcid, str) else "",
    )


def _collect_seen_identities(parent_capture: dict, v6_capture: dict) -> tuple[list[str], list[str]]:
    dois: set[str] = set()
    pmcids: set[str] = set()
    source_sets = []
    for topic in parent_capture.get("topics", []):
        if isinstance(topic, dict):
            source_sets.append(topic)
    for topic in v6_capture.get("topics", {}).values():
        if isinstance(topic, dict):
            source_sets.append(topic)
    for topic in source_sets:
        for field in _IDENTITY_KEYS:
            rows = topic.get(field, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                doi, pmcid = _identity(row)
                if doi:
                    dois.add(doi)
                if pmcid:
                    pmcids.add(pmcid)
    return sorted(dois), sorted(pmcids)


def _dimension_query(topic_spec: dict, field: str, label: str) -> str:
    values = sources._query_terms(topic_spec.get(field), label=field)
    return "(" + " OR ".join(
        f'(TITLE:"{term}" OR ABSTRACT:"{term}")' for term in values) + ")"


def _build_pair_query(topic_spec: dict, first: str, second: str, capture_date: str) -> str:
    base = sources.source_query(capture_date)
    return base + " AND " + _dimension_query(topic_spec, first, first) + \
        " AND " + _dimension_query(topic_spec, second, second)


def _verified_parent_inputs(*, parent_capture: dict, parent_capture_file_sha256: str,
        parent_plan_manifest: dict, v6_capture: dict, v6_capture_file_sha256: str,
        v6_root_audit: dict, v6_root_audit_file_sha256: str) -> dict:
    hashes = (parent_capture_file_sha256, v6_capture_file_sha256, v6_root_audit_file_sha256)
    if any(not _valid_sha(value) for value in hashes):
        raise ValueError("all frozen parent artifact file hashes must be lowercase SHA-256")
    if not isinstance(parent_capture, dict) or \
            parent_capture.get("schema") != sources.TOPIC_SOURCE_CAPTURE_SCHEMA:
        raise ValueError("v7 requires the frozen G09 topic source parent capture")
    parent_capture_sha = _payload_sha(parent_capture, "capture_sha256")
    if parent_capture.get("capture_sha256") != parent_capture_sha:
        raise ValueError("v5 parent capture hash mismatch")
    parent_plan = sources.validate_frozen_topic_plan(parent_plan_manifest,
        expected_plan_sha256=parent_capture.get("frozen_topic_plan_sha256"))
    if parent_capture_file_sha256 == "" or \
            parent_capture.get("frozen_topic_plan_sha256") != parent_plan["plan_sha256"]:
        raise ValueError("v5 parent file/plan binding mismatch")
    if parent_capture.get("request_attempts") != v6.V5_REQUEST_ATTEMPTS or \
            len(parent_capture.get("request_log", [])) != v6.V5_REQUEST_ATTEMPTS:
        raise ValueError("v5 parent request ledger is not the frozen 160-request capture")
    if parent_capture.get("diagnostic_only") is not True or \
            parent_capture.get("scoreable_cohort_created") is not False:
        raise ValueError("v5 source capture must remain diagnostic-only")

    if not isinstance(v6_capture, dict) or v6_capture.get("schema") != v6.V6_CONTINUATION_SCHEMA:
        raise ValueError("v7 requires the frozen G09 v6 continuation capture")
    v6_capture_sha = _payload_sha(v6_capture, "capture_sha256")
    if v6_capture.get("capture_sha256") != v6_capture_sha:
        raise ValueError("v6 capture hash mismatch")
    expected_v6_bindings = {
        "parent_capture_sha256": parent_capture_sha,
        "parent_capture_file_sha256": parent_capture_file_sha256,
        "parent_plan_sha256": parent_plan["plan_sha256"],
        "cumulative_request_attempts": 252,
        "request_attempts": 92,
        "request_failures": 0,
    }
    if any(v6_capture.get(key) != value for key, value in expected_v6_bindings.items()):
        raise ValueError("v6 capture parent/budget binding mismatch")
    if v6_capture_file_sha256 == "" or v6_capture.get("diagnostic_only") is not True or \
            v6_capture.get("scoreable_cohort_created") is not False:
        raise ValueError("v6 parent must remain diagnostic-only with a bound file hash")

    audit = v6_root_audit
    if not isinstance(audit, dict) or audit.get("schema") != V7_AUDIT_SCHEMA:
        raise ValueError("v7 requires the independent v6 root audit")
    audit_sha = _payload_sha(audit, "audit_sha256")
    if audit.get("audit_sha256") != audit_sha:
        raise ValueError("v6 root audit hash mismatch")
    audit_bindings = {
        "v6_capture_sha256": v6_capture_sha,
        "continuation_plan_sha256": v6_capture.get("continuation_plan_sha256"),
        "parent_capture_sha256": parent_capture_sha,
        "parent_plan_sha256": parent_plan["plan_sha256"],
        "cumulative_request_attempts": 252,
        "parent_request_attempts": 160,
        "v6_request_attempts": 92,
        "primary_groups_complete": v6_capture.get("primary_groups_complete"),
    }
    if any(audit.get(key) != value for key, value in audit_bindings.items()):
        raise ValueError("v6 root audit does not bind the frozen capture/budget")
    must_be_true = ("artifact_hash_valid", "budget_valid", "cursor_continuity_valid",
        "implementation_hash_valid", "accepted_records_compliant",
        "rejected_abstract_retention_compliant", "reserve_gate_valid",
        "request_phase_accounting_valid", "diagnostic_only")
    if any(audit.get(key) is not True for key in must_be_true) or \
            audit.get("scoreable_cohort_created") is not False:
        raise ValueError("v6 audit is not eligible for a v7 query plan")
    topic_ids = [row.get("topic_id") for row in parent_capture.get("topics", [])
        if isinstance(row, dict)]
    expected_ids = [row["topic_id"] for row in parent_plan["topic_specs"]]
    if len(topic_ids) != 60 or len(set(topic_ids)) != 60 or set(topic_ids) != set(expected_ids):
        raise ValueError("v5 parent topic IDs do not match the frozen topic plan")
    v6_topics = v6_capture.get("topics")
    if not isinstance(v6_topics, dict) or set(v6_topics) != set(expected_ids):
        raise ValueError("v6 continuation does not contain the frozen 60-topic set")
    return {"parent_plan": parent_plan, "parent_capture_sha256": parent_capture_sha,
        "v6_capture_sha256": v6_capture_sha, "audit_sha256": audit_sha,
        "v6_topics": v6_topics}


def build_v7_query_arm_plan(*, parent_capture: dict,
        parent_capture_file_sha256: str, parent_plan_manifest: dict,
        v6_capture: dict, v6_capture_file_sha256: str,
        v6_root_audit: dict, v6_root_audit_file_sha256: str,
        max_new_requests: int = V7_MAX_NEW_REQUESTS) -> dict:
    """Build a deterministic, offline-only candidate-discovery plan."""
    if type(max_new_requests) is not int or not 1 <= max_new_requests <= V7_MAX_NEW_REQUESTS:
        raise ValueError(f"max_new_requests must be between 1 and {V7_MAX_NEW_REQUESTS}")
    bound = _verified_parent_inputs(parent_capture=parent_capture,
        parent_capture_file_sha256=parent_capture_file_sha256,
        parent_plan_manifest=parent_plan_manifest, v6_capture=v6_capture,
        v6_capture_file_sha256=v6_capture_file_sha256, v6_root_audit=v6_root_audit,
        v6_root_audit_file_sha256=v6_root_audit_file_sha256)
    parent_plan, v6_topics = bound["parent_plan"], bound["v6_topics"]
    specs = {row["topic_id"]: row for row in parent_plan["topic_specs"]}
    parent_topics = {row["topic_id"]: row for row in parent_capture["topics"]}
    remaining, excluded = [], []
    complete_primary_count = 0
    for topic_id in sorted(specs):
        spec = specs[topic_id]
        parent_count = len(parent_topics[topic_id].get("records", []))
        v6_topic = v6_topics[topic_id]
        if v6_topic.get("parent_record_count") != parent_count:
            raise ValueError(f"v6 topic parent count differs from frozen v5 capture: {topic_id}")
        added = v6_topic.get("records_added", [])
        if not isinstance(added, list):
            raise ValueError(f"v6 record additions are malformed for {topic_id}")
        current_count = parent_count + len(added)
        if current_count > 2:
            raise ValueError(f"source group exceeds the two-record cap: {topic_id}")
        if spec["slot_kind"] == "conflict":
            excluded.append({"topic_id": topic_id, "reason": "conflict_candidate_not_primary"})
            continue
        if current_count == 2:
            complete_primary_count += 1
            excluded.append({"topic_id": topic_id, "reason": "already_complete_after_v6"})
            continue
        remaining.append(topic_id)

    if complete_primary_count != v6_capture.get("primary_groups_complete"):
        raise ValueError("v6 primary group count differs from record-level source groups")
    if not remaining:
        raise ValueError("no incomplete primary topics remain for a v7 discovery plan")
    arms = []
    for topic_id in remaining:
        spec, old_topic, v6_topic = specs[topic_id], parent_topics[topic_id], v6_topics[topic_id]
        for arm_id, (left, right) in V7_ARMS.items():
            query = _build_pair_query(spec, left, right, parent_plan["capture_date"])
            arms.append({"arm_id": f"{topic_id}:{arm_id}", "topic_id": topic_id,
                "arm_type": arm_id, "query": query, "query_sha256": hashlib.sha256(
                    query.encode("utf-8")).hexdigest(), "start_cursor": "*",
                "max_pages": V7_BASE_PAGES_PER_ARM,
                "page_size": v6.V6_PAGE_SIZE,
                "max_pages_with_reserve": V7_BASE_PAGES_PER_ARM + V7_RESERVE_MAX_PER_ARM,
                "rationale": "candidate discovery only; frozen local four-dimension gate is unchanged"})
        if not v6_topic.get("search_exhausted", True):
            pages = v6_topic.get("core_pages")
            if not isinstance(pages, list) or not pages:
                raise ValueError(f"open v6 search has no page/cursor receipt: {topic_id}")
            last = pages[-1]
            cursor = last.get("next_cursor_mark") if isinstance(last, dict) else None
            seen_cursors = {page.get("cursor_mark") for page in pages
                if isinstance(page, dict)}
            parent_seen_cursors = {page.get("cursor_mark") for page in old_topic.get("core_pages", [])
                if isinstance(page, dict)}
            if not isinstance(cursor, str) or not cursor or cursor in seen_cursors or \
                    cursor in parent_seen_cursors:
                raise ValueError(f"open v6 cursor is missing or replayed: {topic_id}")
            query = v6_topic.get("query")
            if not isinstance(query, str) or not query:
                raise ValueError(f"v6 resume query is missing for {topic_id}")
            arms.append({"arm_id": f"{topic_id}:condition_intervention_resume",
                "topic_id": topic_id, "arm_type": "condition_intervention_resume",
                "query": query, "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "start_cursor": cursor, "max_pages": V7_BASE_PAGES_PER_ARM,
                "max_pages_with_reserve": V7_BASE_PAGES_PER_ARM + V7_RESERVE_MAX_PER_ARM,
                "page_size": v6.V6_PAGE_SIZE,
                "parent_v6_terminal_cursor_sha256": hashlib.sha256(
                    cursor.encode("utf-8")).hexdigest(),
                "rationale": "continue only the exact unseen cursor; candidate discovery only"})

    base_budget = sum(arm["max_pages"] for arm in arms)
    if base_budget > max_new_requests:
        raise ValueError("max_new_requests is smaller than deterministic base allocation")
    dois, pmcids = _collect_seen_identities(parent_capture, v6_capture)
    parent_request_attempts = parent_capture["request_attempts"]
    parent_cumulative_attempts = v6_capture["cumulative_request_attempts"]
    cumulative_hard_max = parent_cumulative_attempts + max_new_requests
    if cumulative_hard_max > v6.HARD_REQUEST_MAX * 2:
        raise ValueError("v7 cumulative request ceiling exceeds the explicit two-round cap")
    source_gate_sha = hashlib.sha256(Path(sources.__file__).read_bytes()).hexdigest()
    v6_implementation_sha = hashlib.sha256(Path(v6.__file__).read_bytes()).hexdigest()
    missing_dimensions = {key: 0 for key in ("condition", "population", "intervention", "outcome")}
    pico_rejection_count = 0
    incomplete_slot_evidence = []
    for topic_id in remaining:
        slot = v6_topics[topic_id]
        for row in slot.get("exclusions", []):
            if not isinstance(row, dict) or row.get("reason") != "pico_relevance_missing_dimensions":
                continue
            pico_rejection_count += 1
            relevance = row.get("pico_relevance", row.get("abstract_pico_relevance", {}))
            for dimension in relevance.get("missing_dimensions", []):
                if dimension in missing_dimensions:
                    missing_dimensions[dimension] += 1
        receipts = slot.get("core_pages", [])
        incomplete_slot_evidence.append({"topic_id": topic_id,
            "query_mode": slot.get("query_mode"),
            "search_exhausted": slot.get("search_exhausted"),
            "page_count": len(receipts),
            "hit_counts": [page.get("hit_count") for page in receipts if isinstance(page, dict)],
            "records_returned": [page.get("records_returned") for page in receipts if isinstance(page, dict)],
            "pico_rejection_count": sum(1 for row in slot.get("exclusions", [])
                if isinstance(row, dict) and row.get("reason") == "pico_relevance_missing_dimensions"),
            "candidate_audit_count": len(slot.get("candidate_audit", []))})
    alternative_query_evidence = []
    for topic_id in sorted(v6.V6_ALTERNATIVE_QUERY_TOPIC_IDS):
        slot = v6_topics[topic_id]
        receipts = slot.get("core_pages", [])
        alternative_query_evidence.append({"topic_id": topic_id,
            "query_mode": slot.get("query_mode"), "search_exhausted": slot.get("search_exhausted"),
            "hit_counts": [page.get("hit_count") for page in receipts if isinstance(page, dict)],
            "records_returned": [page.get("records_returned") for page in receipts if isinstance(page, dict)],
            "exclusions": [{"doi": row.get("doi"), "pmcid": row.get("pmcid"),
                "reason": row.get("reason")} for row in slot.get("exclusions", [])
                if isinstance(row, dict) and row.get("reason") == "parent_seen_source_identity"]})
    body = {
        "schema": V7_PLAN_SCHEMA,
        "parent_capture_sha256": bound["parent_capture_sha256"],
        "parent_capture_file_sha256": parent_capture_file_sha256,
        "parent_plan_sha256": parent_plan["plan_sha256"],
        "parent_source_implementation_sha256": parent_capture.get("capture_implementation_sha256"),
        "v6_capture_sha256": bound["v6_capture_sha256"],
        "v6_capture_file_sha256": v6_capture_file_sha256,
        "v6_continuation_plan_sha256": v6_capture["continuation_plan_sha256"],
        "v6_implementation_sha256": v6_implementation_sha,
        "v6_root_audit_sha256": bound["audit_sha256"],
        "v6_root_audit_file_sha256": v6_root_audit_file_sha256,
        "topic_specs_sha256": sha256_obj(parent_plan["topic_specs"]),
        "source_gate_sha256": source_gate_sha,
        "plan_builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "capture_date": parent_plan["capture_date"],
        "current_complete_primary_groups": v6_capture["primary_groups_complete"],
        "parent_cumulative_request_attempts": parent_cumulative_attempts,
        "required_complete_primary_groups": 48,
        "incomplete_primary_topic_ids": remaining,
        "excluded_topic_ids": excluded,
        "arms": arms,
        "max_new_requests": max_new_requests,
        "minimum_request_interval_seconds": V7_MIN_REQUEST_INTERVAL_SECONDS,
        "base_request_budget": base_budget,
        "reserve_request_budget": max_new_requests - base_budget,
        "cumulative_request_hard_max": cumulative_hard_max,
        "excluded_parent_dois": dois,
        "excluded_parent_pmcids": pmcids,
        "identity_exclusion_sha256": sha256_obj({"dois": dois, "pmcids": pmcids}),
        "v6_discovery_evidence": {
            "pico_rejection_count_in_remaining_slots": pico_rejection_count,
            "missing_dimensions": missing_dimensions,
            "remaining_slot_search_receipts": incomplete_slot_evidence,
            "strict_four_dimension_alternative_slots": alternative_query_evidence,
            "basis": "v6 showed population/outcome literal matches were the dominant PICO misses; targeted arms add these dimensions to candidate retrieval while leaving local acceptance unchanged",
        },
        "identity_exclusion_policy": {
            "replay_parent_doi_or_pmcid": False,
            "seen_scope": "all v5/v6 candidate_audit, accepted records, records_added and exclusions",
            "enforcement": "global DOI+PMCID exclusion before acceptance, storage or any ID-specific source fetch; a query page may contain a prior identity, but it is discarded and never individually refetched",
        },
        "selection_policy": {
            "scheduler": "lexical topic_id, then arm_id; round-robin one page per eligible arm per pass",
            "base_arm_page_cap": V7_BASE_PAGES_PER_ARM,
            "reserve": f"after base pages only; at most {max_new_requests - base_budget} additional pages, at most {V7_RESERVE_MAX_PER_ARM} reserve pages per arm, round-robin among incomplete topics with an unexhausted arm; stop once 48 primary groups complete or no eligible arm remains",
            "query_arms": "condition+population and condition+outcome use only frozen PICO terms; resume arm uses exact v6 query and nextCursorMark",
            "acceptance_gate": "not relaxed: strict technical gate requires title/abstract literal evidence for all four frozen PICO dimensions, DOI+PMCID identity, abstract length >=400, global deduplication, forbidden-source-family exclusion, and max two accepted sources per group; license is provenance only and is not an eligibility requirement for this personal project",
            "cohort": "source-discovery plan only; no cohort, rubric, answer, annotation or score",
        },
    }
    return {**body, "plan_sha256": sha256_obj(body)}


def validate_v7_query_arm_plan(*, plan: dict, expected_plan_sha256: str,
        parent_capture: dict, parent_capture_file_sha256: str,
        parent_plan_manifest: dict, v6_capture: dict,
        v6_capture_file_sha256: str, v6_root_audit: dict,
        v6_root_audit_file_sha256: str) -> dict:
    """Verify plan bytes/hash bindings by rebuilding it from frozen inputs."""
    if not isinstance(plan, dict) or plan.get("schema") != V7_PLAN_SCHEMA:
        raise ValueError("unsupported v7 query-arm plan")
    actual_sha = _payload_sha(plan, "plan_sha256")
    if not _valid_sha(expected_plan_sha256) or actual_sha != expected_plan_sha256 or \
            plan.get("plan_sha256") != actual_sha:
        raise ValueError("v7 query-arm plan hash mismatch")
    rebuilt = build_v7_query_arm_plan(parent_capture=parent_capture,
        parent_capture_file_sha256=parent_capture_file_sha256,
        parent_plan_manifest=parent_plan_manifest, v6_capture=v6_capture,
        v6_capture_file_sha256=v6_capture_file_sha256, v6_root_audit=v6_root_audit,
        v6_root_audit_file_sha256=v6_root_audit_file_sha256,
        max_new_requests=plan.get("max_new_requests"))
    if plan != rebuilt:
        raise ValueError("v7 query-arm plan differs from parent-derived plan")
    return rebuilt
