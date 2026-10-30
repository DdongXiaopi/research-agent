"""Deterministic, target-blind G05 source sampling and empty task shells."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

AUTHORING_CONTEXT_POLICY = "abstract_chars_400_6000_no_replacement_char_v1"
MIN_AUTHORING_ABSTRACT_CHARS = 400
MAX_AUTHORING_ABSTRACT_CHARS = 6000


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _rank(seed: str, entry: dict[str, Any]) -> str:
    material = "\0".join((seed, str(entry["doi"]), str(entry["source_content_sha256"])))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def authoring_context_exclusion(abstract: Any) -> str | None:
    """Return the frozen authoring-context exclusion reason, if any."""
    if not isinstance(abstract, str):
        return "abstract_not_text"
    visible = abstract.strip()
    if len(visible) < MIN_AUTHORING_ABSTRACT_CHARS:
        return "abstract_below_400_chars"
    if len(visible) > MAX_AUTHORING_ABSTRACT_CHARS:
        return "abstract_above_6000_chars"
    if "\ufffd" in visible:
        return "abstract_contains_replacement_character"
    return None


def build_g05_sample_plan(
    source_pool: dict[str, Any],
    publication_status_report: dict[str, Any],
    *,
    seed: str,
    primary_size: int = 60,
    reserve_size: int = 60,
) -> dict[str, Any]:
    """Rank point-in-time verified DOI groups and emit blank case shells.

    This function has no model/scorer dependency. The seed and source pool are
    inputs, every DOI is one independent group, and unknown class capacity is
    reported honestly until reference construction has been separately approved.
    """
    if not isinstance(seed, str) or not seed.strip():
        raise ValueError("deterministic sampling seed is required")
    if primary_size < 1 or reserve_size < 0:
        raise ValueError("primary size must be positive and reserve size nonnegative")
    pool_entries = source_pool.get("entries") or []
    if source_pool.get("schema") != "g05-unlabeled-plos-source-pool-v1" or not pool_entries:
        raise ValueError("unsupported or empty G05 source pool")
    if source_pool.get("source_pool_sha256") != _canonical_sha(pool_entries):
        raise ValueError("source pool hash mismatch")
    if source_pool.get("candidate_count") != len(pool_entries):
        raise ValueError("source pool candidate count mismatch")
    if not publication_status_report.get("entries"):
        raise ValueError("publication status report is empty")
    if publication_status_report.get("schema") != "g05-publication-status-report-v1":
        raise ValueError("unsupported publication status report")
    if publication_status_report.get("source_pool_sha256") != source_pool["source_pool_sha256"]:
        raise ValueError("publication status report is not bound to this source pool")
    if not _valid_sha(publication_status_report.get("status_plan_sha256")):
        raise ValueError("publication status plan hash is missing or invalid")
    unsigned_status_report = {
        key: value for key, value in publication_status_report.items() if key != "report_sha256"
    }
    if (not _valid_sha(publication_status_report.get("report_sha256"))
            or publication_status_report.get("report_sha256") != _canonical_sha(unsigned_status_report)):
        raise ValueError("publication status report hash mismatch")

    pool_by_doi: dict[str, dict[str, Any]] = {}
    for entry in pool_entries:
        doi = str(entry.get("doi") or "").strip().casefold()
        group = str(entry.get("source_group_id") or "").strip().casefold()
        if not doi or doi != group or doi in pool_by_doi:
            raise ValueError("source pool DOI groups must be present and unique")
        if not _valid_sha(entry.get("source_content_sha256")):
            raise ValueError("source content hash is missing or invalid")
        pool_by_doi[doi] = entry

    status_by_doi: dict[str, dict[str, Any]] = {}
    for row in publication_status_report["entries"]:
        doi = str(row.get("doi") or "").strip().casefold()
        if not doi or doi in status_by_doi or doi not in pool_by_doi:
            raise ValueError("publication status DOI is duplicate or outside the frozen source pool")
        status_by_doi[doi] = row

    eligible: list[dict[str, Any]] = []
    status_verified_count = 0
    authoring_context_excluded_count = 0
    excluded_counts: dict[str, int] = {}
    for doi, status in status_by_doi.items():
        source = pool_by_doi[doi]
        if status.get("source_content_sha256") != source["source_content_sha256"]:
            excluded_counts["source_identity_mismatch"] = excluded_counts.get("source_identity_mismatch", 0) + 1
            continue
        if (status.get("disposition") != "verified_current_as_of_capture"
                or status.get("eligible_for_status_screen") is not True):
            excluded_counts["status_unverified_or_excluded"] = excluded_counts.get(
                "status_unverified_or_excluded", 0) + 1
            continue
        raw_hashes = status.get("raw_response_sha256") or {}
        if (not _valid_sha(raw_hashes.get("crossref"))
                or not _valid_sha(raw_hashes.get("crossmark"))):
            excluded_counts["status_raw_response_hash_missing"] = excluded_counts.get(
                "status_raw_response_hash_missing", 0) + 1
            continue
        status_verified_count += 1
        context_reason = authoring_context_exclusion(source.get("abstract"))
        if context_reason:
            excluded_counts[context_reason] = excluded_counts.get(context_reason, 0) + 1
            authoring_context_excluded_count += 1
            continue
        eligible.append(source)

    ranked = sorted(eligible, key=lambda row: (_rank(seed, row), row["doi"]))
    selected = ranked[:primary_size + reserve_size]
    primary = selected[:primary_size]
    reserve = selected[primary_size:]
    shells = []
    for index, source in enumerate(selected):
        shells.append({
            "case_id": f"g05-case-shell-{index + 1:04d}",
            "selection_role": "primary" if index < len(primary) else "reserve",
            "task_contract": "selected_answer_support_v1",
            "scope": "single_doc",
            "source_group_id": source["source_group_id"],
            "source_content_sha256": source["source_content_sha256"],
            "source_doi": source["doi"],
            "question": None,
            "selected_answer": None,
            "cited_document": {"title": source.get("title"), "abstract": source.get("abstract")},
            "reference_label": None,
            "rationale_spans": [],
            "target_scorer_output": None,
        })

    unsigned = {
        "schema": "g05-deterministic-sample-plan-v2",
        "authoring_context_policy": AUTHORING_CONTEXT_POLICY,
        "source_pool_sha256": source_pool["source_pool_sha256"],
        "status_plan_sha256": publication_status_report["status_plan_sha256"],
        "publication_status_report_sha256": publication_status_report.get("report_sha256"),
        "seed": seed,
        "ranking": "sha256(seed NUL doi NUL source_content_sha256), ascending; DOI tie-break",
        "task_contract": "selected_answer_support_v1",
        "scope": "single_doc",
        "status_scope": "point_in_time_only",
        "eligible_source_count": len(eligible),
        "status_verified_source_count": status_verified_count,
        "authoring_context_excluded_count": authoring_context_excluded_count,
        "status_rows_seen": len(status_by_doi),
        "excluded_counts": dict(sorted(excluded_counts.items())),
        "primary_count": len(primary),
        "reserve_count": len(reserve),
        "ordered_dois": [row["doi"] for row in selected],
        "task_shells": shells,
        "class_capacity": "unknown_not_labeled",
        "labels_generated": False,
        "model_calls": 0,
        "target_scorer_calls": 0,
        "calibration_eligible": False,
    }
    unsigned["plan_sha256"] = _canonical_sha(unsigned)
    return unsigned
