"""Build an unlabeled, source-grounded G05 pool from licensed PLOS metadata.

This is a provenance and exposure screen only. It does not create claim labels,
gold references, calibration packets, or promotion evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any

from .plos_clinical_intake import EXCLUDED_TITLE_TERMS, classify_plos_copyright_statement


DOI_RE = re.compile(r"^10\.1371/journal\.[a-z0-9]+\.[0-9]{7}$", re.I)
NO_UPDATE = "no_crossref_update_found_not_status_proof"


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def build_g05_unlabeled_source_pool(
    records: list[dict[str, Any]],
    history_report: dict[str, Any],
    crossref_report: dict[str, Any],
    source_manifest: dict[str, Any],
    *,
    source_index_sha256: str,
    source_capture_plan_sha256: str,
    source_capture_receipt_sha256: str,
    source_manifest_file_sha256: str,
    history_report_file_sha256: str,
    crossref_report_file_sha256: str,
) -> dict[str, Any]:
    """Create a frozen source pool with strict provenance and group guards.

    A row is included only when the article record has an explicit article-level
    CC BY/CC0 statement, is a primary research article, did not exactly overlap
    the project-history scan, and had no Crossref update link in the frozen
    audit. Absence of an update is explicitly *not* treated as proof of current
    publication status. Each DOI is one independent source group.
    """
    if not records:
        raise ValueError("source capture is empty")
    if not re.fullmatch(r"[0-9a-f]{64}", source_index_sha256):
        raise ValueError("source index SHA-256 is invalid")
    if not history_report.get("complete") or history_report.get("unreadable_files") != 0:
        raise ValueError("historical exposure scan must be complete and readable")
    if history_report.get("source_candidate_index_sha256") != source_index_sha256:
        raise ValueError("historical scan is not bound to the frozen source index")
    if history_report.get("source_capture_plan_sha256") != source_capture_plan_sha256:
        raise ValueError("historical scan is not bound to the source capture plan")
    if history_report.get("candidate_count") != len(records):
        raise ValueError("historical scan candidate count does not match source index")
    if not source_manifest or source_manifest.get("labels_generated") is not False:
        raise ValueError("candidate source manifest must confirm labels were not generated")
    if not source_manifest.get("candidate_entries"):
        raise ValueError("candidate source manifest has no audited entries")
    if source_manifest.get("candidate_set_sha256") != _canonical_sha(
            source_manifest["candidate_entries"]):
        raise ValueError("candidate source manifest content hash mismatch")
    if not crossref_report.get("candidate_entries"):
        raise ValueError("Crossref update audit has no candidate entries")
    if crossref_report.get("candidate_manifest_sha256") != source_manifest_file_sha256:
        raise ValueError("Crossref update audit is not bound to the source manifest file")

    overlaps = history_report.get("overlap_candidates") or {}
    updates = {str(row.get("doi", "")).casefold(): row.get("update_disposition")
               for row in crossref_report["candidate_entries"]}
    licensed = {str(row.get("doi", "")).casefold(): row for row in source_manifest["candidate_entries"]}
    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    exclusions: dict[str, int] = {}
    license_counts: dict[str, int] = {}

    for row in records:
        doi = str(row.get("doi") or "").strip().casefold()
        reasons: list[str] = []
        if not DOI_RE.fullmatch(doi):
            reasons.append("invalid_plos_doi")
        if not doi or doi in seen:
            reasons.append("duplicate_or_missing_doi")
        seen.add(doi)
        if doi in overlaps:
            reasons.append("exact_project_history_overlap")
        license_class = classify_plos_copyright_statement(
            str(row.get("copyright_statement") or ""))
        if license_class is None:
            reasons.append("article_level_license_not_explicit")
        article_type = _norm(str(row.get("article_type") or ""))
        if article_type != "research article":
            reasons.append("not_primary_research_article")
        title = _norm(str(row.get("title") or ""))
        if any(term in title for term in EXCLUDED_TITLE_TERMS):
            reasons.append("excluded_title_type")
        if updates.get(doi) != NO_UPDATE:
            reasons.append("crossref_update_present_or_unknown")
        prior = licensed.get(doi)
        if prior is None or prior.get("disposition") != "pending_retraction_and_trial_evidence_review":
            reasons.append("prior_source_manifest_not_eligible")
        if prior and prior.get("license_classification") != license_class:
            reasons.append("license_classification_changed")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("source_response_sha256") or "")):
            reasons.append("raw_source_page_hash_missing")
        if not str(row.get("abstract") or "").strip():
            reasons.append("abstract_missing")
        for reason in reasons:
            exclusions[reason] = exclusions.get(reason, 0) + 1
        if reasons:
            continue

        license_counts[license_class] = license_counts.get(license_class, 0) + 1
        source = {
            "doi": doi,
            "title": str(row.get("title") or "").strip(),
            "abstract": str(row.get("abstract") or "").strip(),
            "publication_date": row.get("publication_date"),
            "article_type": row.get("article_type"),
            "source_response_sha256": row["source_response_sha256"],
            "copyright_statement_sha256": _text_sha(str(row.get("copyright_statement") or "")),
            "license_classification": license_class,
        }
        entries.append({
            "source_group_id": doi,
            "source_content_sha256": _canonical_sha(source),
            **source,
            "crossref_status": "no_update_found_status_unverified",
            "history_exposure_screen": "no_exact_match_in_scanned_history",
            "label": None,
            "claim": None,
            "selected_answer": None,
            "rationale_spans": [],
        })

    if len({entry["source_group_id"] for entry in entries}) != len(entries):
        raise ValueError("source group IDs must be unique")
    if len(entries) < 30:
        raise ValueError("fewer than 30 independent source articles remain")
    result: dict[str, Any] = {
        "schema": "g05-unlabeled-plos-source-pool-v1",
        "source_index_sha256": source_index_sha256,
        "source_capture_plan_sha256": source_capture_plan_sha256,
        "source_capture_receipt_sha256": source_capture_receipt_sha256,
        "history_report_sha256": history_report_file_sha256,
        "history_scan_complete": True,
        "history_exclusions": history_report.get("excluded_roots", []),
        "crossref_report_sha256": crossref_report_file_sha256,
        "source_manifest_file_sha256": source_manifest_file_sha256,
        "source_record_count": len(records),
        "candidate_count": len(entries),
        "independent_source_group_count": len(entries),
        "license_counts": dict(sorted(license_counts.items())),
        "exclusion_counts_nonexclusive": dict(sorted(exclusions.items())),
        "labels_generated": False,
        "model_calls": 0,
        "target_scorer_calls": 0,
        "calibration_packet_created": False,
        "confirmation_eligible": False,
        "publication_status_verified": False,
        "source_pool_sha256": _canonical_sha(entries),
        "entries": entries,
    }
    return result
