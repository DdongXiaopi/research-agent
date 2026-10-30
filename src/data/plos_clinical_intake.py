"""Fail-closed intake checks for PLOS clinical-result discovery records.

This module screens provenance and explicit evidence supplied by an isolated
source-review step. It never infers a CliniFact label or treats a search hit as
an eligible confirmation task.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any


DOI_RE = re.compile(r"^10\.1371/journal\.[a-z0-9]+\.[0-9]{7}$", re.I)
NCT_RE = re.compile(r"^NCT[0-9]{8}$", re.I)
CC_BY_URLS = {
    "https://creativecommons.org/licenses/by/4.0/",
    "https://creativecommons.org/licenses/by/3.0/",
}
CC_BY_STATEMENT = "creative commons attribution license"
CC0_STATEMENT = "creative commons cc0 public domain dedication"
EXCLUDED_TITLE_TERMS = (
    "retracted", "retraction", "correction", "erratum", "protocol",
    "systematic review", "meta-analysis", "scoping review",
    "secondary analysis", "post hoc analysis", "exploratory analysis",
)
EXCLUDED_TYPE_TERMS = (
    "review", "protocol", "correction", "retraction", "editorial", "meta-analysis",
)


def _norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def classify_plos_copyright_statement(statement: str) -> str | None:
    """Classify only explicit article-level CC BY or CC0 copyright language."""
    normalized = _norm(statement)
    if CC0_STATEMENT in normalized:
        return "CC0_EXPLICIT_ARTICLE_STATEMENT"
    if CC_BY_STATEMENT in normalized or "creativecommons.org/licenses/by/" in normalized:
        return "CC_BY_EXPLICIT_ARTICLE_STATEMENT_VERSION_UNSPECIFIED"
    return None


def screen_plos_clinical_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return an auditable discovery disposition, never a task label.

    Required source-review evidence is explicit and quote-bound to the frozen
    abstract. A ready result means only that independent AI labeling may start.
    """
    reasons: list[str] = []
    doi = str(record.get("doi") or record.get("id") or "").strip()
    if doi.lower().startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    title = str(record.get("title") or record.get("title_display") or "").strip()
    abstract = str(record.get("abstract") or "").strip()
    article_type = _norm(str(record.get("article_type") or ""))
    license_url = str(record.get("license_url") or "").strip().rstrip("/") + "/"
    license_source = str(record.get("license_source_url") or "").strip()
    copyright_statement = str(record.get("copyright_statement") or "").strip()
    license_classification = classify_plos_copyright_statement(copyright_statement)
    status = _norm(str(record.get("publication_status") or ""))

    if not DOI_RE.fullmatch(doi):
        reasons.append("unsupported_or_missing_plos_doi")
    if not title:
        reasons.append("missing_title")
    title_norm = _norm(title)
    if any(term in title_norm for term in EXCLUDED_TITLE_TERMS):
        reasons.append("excluded_title_type")
    if any(term in article_type for term in EXCLUDED_TYPE_TERMS):
        reasons.append("excluded_article_type")
    elif not article_type:
        reasons.append("article_type_unverified")
    elif "research" not in article_type and "clinical trial" not in article_type:
        reasons.append("article_type_not_primary_research")
    if status in {"retracted", "withdrawn", "corrected", "correction"}:
        reasons.append("excluded_publication_status")
    elif status not in {"published", "active"}:
        reasons.append("publication_status_unverified")
    response_sha = str(record.get("source_response_sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", response_sha):
        reasons.append("frozen_source_response_hash_missing")
    if not abstract:
        reasons.append("missing_abstract")
    url_license = license_url in {url.rstrip("/") + "/" for url in CC_BY_URLS}
    if not url_license and license_classification is None:
        reasons.append("article_specific_reuse_license_unverified")
    if not license_source.startswith("https://") or doi.casefold() not in license_source.casefold():
        reasons.append("license_evidence_not_bound_to_article")

    identity = record.get("trial_identity")
    if not isinstance(identity, dict):
        identity = {}
    nct_id = str(identity.get("nct_id") or "").strip().upper()
    if not NCT_RE.fullmatch(nct_id):
        reasons.append("original_human_trial_identity_unverified")
    nct_quote = str(identity.get("nct_evidence_quote") or "").strip()
    if (not nct_quote or not abstract or _norm(nct_quote) not in _norm(abstract)
            or nct_id.casefold() not in nct_quote.casefold()):
        reasons.append("trial_registry_identity_not_quote_bound_to_abstract")
    for field in ("intervention", "comparator", "primary_outcome", "timepoint"):
        evidence = str(identity.get(f"{field}_evidence_quote") or "").strip()
        if not evidence or not abstract or _norm(evidence) not in _norm(abstract):
            reasons.append(f"{field}_not_quote_bound_to_abstract")

    stats = record.get("primary_outcome_statistics")
    if not isinstance(stats, dict):
        stats = {}
    stat_quote = str(stats.get("evidence_quote") or "").strip()
    if not stat_quote or not abstract or _norm(stat_quote) not in _norm(abstract):
        reasons.append("primary_outcome_statistics_not_quote_bound")
    if not str(stats.get("measure") or "").strip():
        reasons.append("primary_outcome_measure_unverified")
    if not str(stats.get("value") or "").strip():
        reasons.append("primary_outcome_value_unverified")
    elif _norm(str(stats.get("value"))) not in _norm(stat_quote):
        reasons.append("primary_outcome_value_not_in_evidence_quote")
    if not str(stats.get("timepoint") or "").strip():
        reasons.append("primary_outcome_timepoint_unverified")

    source_payload = {
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "article_type": article_type,
        "license_url": license_url,
        "license_source_url": license_source,
        "copyright_statement_sha256": _sha256(copyright_statement) if copyright_statement else None,
        "publication_status": status,
        "trial_identity": identity,
        "primary_outcome_statistics": stats,
    }
    return {
        "schema": "plos-clinical-intake-screen-v1",
        "doi": doi,
        "license_classification": (license_classification or
                                   ("CC_BY_EXPLICIT_URL" if url_license else None)),
        "source_record_sha256": _sha256(json.dumps(
            {"response_sha256": response_sha, "record": source_payload},
            sort_keys=True, ensure_ascii=False, separators=(",", ":"))),
        "article_text_sha256": _sha256(abstract),
        "disposition": "ready_for_dual_ai_labeling" if not reasons else "exclude_or_resolve",
        "reasons": sorted(set(reasons)),
        "confirmation_eligible": False,
        "task_label": None,
        "labeling_started": False,
        "ai_reference_is_human_gold": False,
    }
