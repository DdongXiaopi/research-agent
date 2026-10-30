"""Offline G09 v9 field-scoped AND discovery plan.

Builds a bounded, evidence-ranked Europe PMC plan from frozen G09 evidence.
It performs no network/model calls and does not create or label a cohort.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

from ..util import sha256_obj
from . import open_research_sources as sources

V9_PLAN_SCHEMA = "g09-v9-field-scoped-search-plan-v1"
V9_MAX_REQUESTS = 40
V9_TARGET_GROUPS = 11
V9_BASE_PAGES = 3
V9_PAGE_SIZE = 10
V9_INTERVAL_SECONDS = 1.0
PICO_FIELDS = ("condition_terms", "population_terms", "intervention_terms", "outcome_terms")
_IDENTITY_FIELDS = ("candidate_audit", "records", "records_added", "exclusions")


def _payload_sha(value: dict, field: str) -> str:
    return sha256_obj({key: item for key, item in value.items() if key != field})


def _file_sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _quote(term: str) -> str:
    return '"' + term.replace('"', '\\"') + '"'


def _field_clause(values: list[str], field: str) -> str:
    terms = sources._query_terms(values, label=field)
    return "(" + " OR ".join(
        f"(TITLE:{_quote(term)} OR ABSTRACT:{_quote(term)})" for term in terms) + ")"


def _query(spec: dict, capture_date: str) -> str:
    clauses = ["HAS_ABSTRACT:Y",
        f"FIRST_PDATE:[2020-01-01 TO {capture_date}]", "NOT DOI:10.1371*"]
    clauses.extend(_field_clause(spec[field], field) for field in PICO_FIELDS)
    return " AND ".join(clauses)


def _near_miss_counts(capture: dict) -> Counter:
    accepted = set()
    topics = capture.get("topics", {})
    for topic in topics.values():
        if isinstance(topic, dict):
            for row in topic.get("records_added", []):
                if isinstance(row, dict):
                    accepted.add((str(row.get("doi", "")).casefold(),
                        str(row.get("pmcid", "")).casefold()))
    counts = Counter()
    for topic_id, topic in topics.items():
        if not isinstance(topic, dict):
            continue
        seen = set()
        for row in topic.get("candidate_audit", []):
            if not isinstance(row, dict):
                continue
            relevance = row.get("abstract_pico_relevance", {})
            matched = relevance.get("matched_terms", {}) if isinstance(relevance, dict) else {}
            dimensions = sum(bool(matched.get(name)) for name in
                ("condition", "population", "intervention", "outcome"))
            identity = (str(row.get("doi", "")).casefold(), str(row.get("pmcid", "")).casefold())
            if dimensions == 3 and identity not in accepted and identity not in seen:
                counts[topic_id] += 1
                seen.add(identity)
    return counts


def _seen_identities(plan: dict, capture: dict) -> tuple[list[str], list[str]]:
    dois = {str(value).strip().casefold() for value in plan.get("excluded_parent_dois", [])
        if isinstance(value, str) and value.strip()}
    pmcids = {str(value).strip().casefold() for value in plan.get("excluded_parent_pmcids", [])
        if isinstance(value, str) and value.strip()}
    for topic in capture.get("topics", {}).values():
        if not isinstance(topic, dict):
            continue
        for field in _IDENTITY_FIELDS:
            for row in topic.get(field, []):
                if isinstance(row, dict):
                    doi, pmcid = row.get("doi"), row.get("pmcid")
                    if isinstance(doi, str) and doi.strip():
                        dois.add(doi.strip().casefold())
                    if isinstance(pmcid, str) and pmcid.strip():
                        pmcids.add(pmcid.strip().casefold())
        for row in topic.get("parent_record_identities", []):
            if isinstance(row, dict):
                doi, pmcid = row.get("doi"), row.get("pmcid")
                if isinstance(doi, str) and doi.strip():
                    dois.add(doi.strip().casefold())
                if isinstance(pmcid, str) and pmcid.strip():
                    pmcids.add(pmcid.strip().casefold())
    return sorted(dois), sorted(pmcids)


def build_v9_field_search_plan(*, v7_plan: dict, v7_capture: dict,
        v7_capture_file_sha256: str, v7_root_audit: dict,
        v7_root_audit_file_sha256: str, topic_manifest: dict,
        topic_manifest_file_sha256: str, max_requests: int = V9_MAX_REQUESTS,
        target_groups: int = V9_TARGET_GROUPS) -> dict:
    """Create a deterministic parent-bound, field-scoped plan without I/O."""
    if v7_plan.get("schema") != "g09-v7-query-arm-plan-v1":
        raise ValueError("v9 requires the frozen v7 query plan")
    if v7_plan.get("plan_sha256") != _payload_sha(v7_plan, "plan_sha256"):
        raise ValueError("v7 plan hash mismatch")
    if v7_capture.get("schema") != "g09-topic-source-capture-v7":
        raise ValueError("v9 requires the frozen v7 capture")
    if v7_capture.get("capture_sha256") != sha256_obj({key: val for key, val in
            v7_capture.items() if key not in {"capture_sha256", "audit"}}):
        raise ValueError("v7 capture hash mismatch")
    for digest in (v7_capture_file_sha256, v7_root_audit_file_sha256,
            topic_manifest_file_sha256):
        if not _file_sha(digest):
            raise ValueError("v9 parent file hashes must be lowercase SHA-256")
    if v7_capture.get("plan_sha256") != v7_plan["plan_sha256"]:
        raise ValueError("v7 capture is not bound to supplied plan")
    audit = v7_root_audit
    if audit.get("schema") != "g09-v7-independent-root-audit-v1" or \
            audit.get("root_audit_sha256") != _payload_sha(audit, "root_audit_sha256"):
        raise ValueError("v9 requires a valid independent v7 root audit")
    if audit.get("capture_sha256") != v7_capture["capture_sha256"] or \
            audit.get("plan_sha256") != v7_plan["plan_sha256"] or \
            audit.get("capture_file_sha256") != v7_capture_file_sha256 or \
            audit.get("diagnostic_only") is not True:
        raise ValueError("v7 root audit does not bind this diagnostic capture")
    capture_audit = audit.get("capture_audit")
    required_audit_fields = ("artifact_hash_valid", "plan_binding_valid",
        "implementation_hash_valid", "budget_valid", "reserve_gate_valid",
        "cursor_continuity_valid", "request_phase_accounting_valid",
        "accepted_records_compliant", "diagnostic_only")
    if not isinstance(capture_audit, dict) or any(
            capture_audit.get(field) is not True for field in required_audit_fields) or \
            capture_audit.get("scoreable_cohort_created") is not False:
        raise ValueError("v7 root audit did not pass capture integrity checks")
    ledger_audit = audit.get("ledger_audit")
    if not isinstance(ledger_audit, dict) or ledger_audit.get(
            "ambiguous_in_flight_count") != 0:
        raise ValueError("v7 write-ahead ledger contains ambiguous requests")
    if v7_capture.get("diagnostic_only") is not True or \
            v7_capture.get("scoreable_cohort_created") is not False:
        raise ValueError("v9 parent capture must remain diagnostic-only")
    if type(max_requests) is not int or not 1 <= max_requests <= V9_MAX_REQUESTS:
        raise ValueError("v9 request budget must be within 1..40")
    if type(target_groups) is not int or target_groups != V9_TARGET_GROUPS:
        raise ValueError("v9 selects exactly 11 evidence-ranked incomplete groups")
    if topic_manifest.get("plan_sha256") != v7_capture.get("parent_plan_sha256") and \
            topic_manifest.get("plan_sha256") != v7_plan.get("parent_plan_sha256"):
        # v7 binds its originating topic plan directly. Keep this check explicit.
        raise ValueError("topic manifest does not match the v5 topic-plan binding")
    manifest = sources.validate_frozen_topic_plan(topic_manifest,
        expected_plan_sha256=topic_manifest.get("plan_sha256"))
    if v7_plan.get("topic_specs_sha256") != sha256_obj(manifest["topic_specs"]):
        raise ValueError("topic specifications differ from the frozen v7 plan")

    specs = {row["topic_id"]: row for row in manifest["topic_specs"]}
    capture_topics = v7_capture.get("topics")
    if not isinstance(capture_topics, dict) or set(capture_topics) != set(specs):
        raise ValueError("v7 capture and topic manifest must contain the same groups")
    incomplete = []
    near_misses = _near_miss_counts(v7_capture)
    for topic_id, topic in capture_topics.items():
        spec = specs[topic_id]
        if spec.get("slot_kind") == "conflict":
            continue
        current = int(topic.get("parent_record_count", 0)) + int(topic.get("v6_record_count", 0)) + \
            len(topic.get("records_added", []))
        if current < 2:
            incomplete.append({"topic_id": topic_id, "current_source_count": current,
                "three_of_four_near_miss_count": near_misses[topic_id]})
    if len(incomplete) < V9_TARGET_GROUPS:
        raise ValueError("fewer than 11 incomplete primary groups remain")
    ranked = sorted(incomplete, key=lambda row: (-row["three_of_four_near_miss_count"],
        row["current_source_count"], row["topic_id"]))
    selected = ranked[:V9_TARGET_GROUPS]
    if max_requests < len(selected):
        raise ValueError("v9 must budget at least one first page per selected group")
    base_pages = min(V9_BASE_PAGES, max_requests // len(selected))
    quotas = {row["topic_id"]: base_pages for row in selected}
    remaining = max_requests - base_pages * len(selected)
    for row in selected:
        if remaining <= 0:
            break
        if quotas[row["topic_id"]] < 4:
            quotas[row["topic_id"]] += 1
            remaining -= 1
    if remaining:
        raise ValueError("v9 max request budget cannot be allocated within group caps")
    capture_date = v7_plan["capture_date"]
    excluded_dois, excluded_pmcids = _seen_identities(v7_plan, v7_capture)
    arms = []
    for rank, row in enumerate(selected, 1):
        topic_id = row["topic_id"]
        query = _query(specs[topic_id], capture_date)
        arms.append({"arm_id": f"{topic_id}:all_four_pico_fields", "topic_id": topic_id,
            "rank": rank, "query": query,
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "max_requests": quotas[topic_id], "page_size": V9_PAGE_SIZE,
            "start_cursor": "*", "three_of_four_near_miss_count": row[
                "three_of_four_near_miss_count"], "current_source_count": row[
                "current_source_count"], "rationale": "AND all four frozen PICO fields; candidate discovery only"})
    body = {"schema": V9_PLAN_SCHEMA, "status": "offline_plan_not_executed",
        "parent_v7_plan_sha256": v7_plan["plan_sha256"],
        "parent_v7_capture_sha256": v7_capture["capture_sha256"],
        "parent_v7_capture_file_sha256": v7_capture_file_sha256,
        "parent_v7_root_audit_sha256": audit["root_audit_sha256"],
        "parent_v7_root_audit_file_sha256": v7_root_audit_file_sha256,
        "topic_manifest_sha256": manifest["plan_sha256"],
        "topic_manifest_file_sha256": topic_manifest_file_sha256,
        "plan_builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "acceptance_implementation_sha256": hashlib.sha256(
            Path(__file__).with_name("open_research_v7_capture.py").read_bytes()).hexdigest(),
        "capture_date": capture_date, "query_contract": {
            "fields": list(PICO_FIELDS), "operator_between_dimensions": "AND",
            "operator_within_dimension": "OR", "requires_abstract": True,
            "open_access_filter": False, "license_filter": False,
            "excluded_doi_prefix": "10.1371*"},
        "request_budget": max_requests, "minimum_request_interval_seconds": V9_INTERVAL_SECONDS,
        "page_size": V9_PAGE_SIZE, "selected_groups": [row["topic_id"] for row in selected],
        "selection_evidence": selected, "reserve_groups": ranked[V9_TARGET_GROUPS:],
        "excluded_parent_dois": excluded_dois,
        "excluded_parent_pmcids": excluded_pmcids,
        "identity_exclusion_sha256": sha256_obj({"dois": excluded_dois,
            "pmcids": excluded_pmcids}),
        "identity_exclusion_policy": "Every DOI and PMCID seen in frozen v5/v6 exclusions and v7 candidate audit, accepted records, records_added, exclusions, or parent identities must be dropped before acceptance/storage; query-page matches are not refetched by ID.",
        "arms": arms, "acceptance_gate": "license metadata is provenance only; retain all existing strict DOI+PMCID identity, abstract length >=400, literal evidence for all four frozen PICO dimensions, deduplication, forbidden-source-family exclusion, and max two records per group gates",
        "diagnostic_only": True, "network_calls": 0, "model_calls": 0,
        "cohort_created": False}
    return {**body, "plan_sha256": sha256_obj(body)}
