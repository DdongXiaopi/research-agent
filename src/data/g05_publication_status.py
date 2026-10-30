"""Fail-closed, point-in-time publication status checks for G05 sources.

This module is pure: the collector owns network access and raw-response
retention, while this interface only compares captured publisher metadata.
"""

from __future__ import annotations

import html
import hashlib
import json
import re
import unicodedata
from datetime import date
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote, unquote, urlparse


def _norm(value: str) -> str:
    visible = _visible_text(value)
    return " ".join(unicodedata.normalize("NFKC", visible).casefold().split())


class _VisibleTitleParser(HTMLParser):
    """Extract inline HTML title text without dropping sup/sub content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _visible_text(value: str) -> str:
    if not re.search(r"<[A-Za-z][^>]*>", value):
        return html.unescape(value)
    parser = _VisibleTitleParser()
    try:
        parser.feed(value)
        parser.close()
    except Exception:
        return html.unescape(re.sub(r"</?[A-Za-z][^>]*>", "", value))
    return html.unescape("".join(parser.parts))


def _expected_date(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", value.strip())
    if not match:
        return None
    try:
        return date(*(int(part) for part in match.groups())).isoformat()
    except ValueError:
        return None


def _crossref_date(payload: dict[str, Any]) -> str | None:
    for field in ("published", "published-print", "published-online"):
        date_parts = (payload.get(field) or {}).get("date-parts")
        if not date_parts or not isinstance(date_parts[0], list):
            continue
        parts = date_parts[0]
        if len(parts) < 3:
            continue
        try:
            return date(int(parts[0]), int(parts[1]), int(parts[2])).isoformat()
        except (TypeError, ValueError):
            return None
    return None


class _CrossmarkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self.non_metadata_text: list[str] = []
        self.hrefs: list[str] = []
        self.status_classes: list[str] = []
        self.change_classes: list[str] = []
        self.metadata_text: list[str] = []
        self._metadata_tag: str | None = None
        self._metadata_depth = 0
        self._metadata_parts: list[str] = []
        self.version_record_text: list[str] = []
        self._version_tag: str | None = None
        self._version_depth = 0
        self._version_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = (values.get("class") or "").split()
        if any("document-status" in item for item in classes):
            self.status_classes.extend(classes)
        if any(term in item.casefold() for item in classes
               for term in ("retraction", "correction", "expression-of-concern", "document-update")):
            self.change_classes.extend(classes)
        if self._metadata_tag is not None and tag not in {"br", "img", "hr", "meta", "link", "input"}:
            self._metadata_depth += 1
        if self._version_tag is not None and tag not in {"br", "img", "hr", "meta", "link", "input"}:
            self._version_depth += 1
        if "document-metadata__text" in classes and self._metadata_tag is None:
            self._metadata_tag = tag
            self._metadata_depth = 1
            self._metadata_parts = []
        if "records__item__content__text" in classes and self._version_tag is None:
            self._version_tag = tag
            self._version_depth = 1
            self._version_parts = []
        href = values.get("href")
        if href:
            self.hrefs.append(html.unescape(href))

    def handle_endtag(self, tag: str) -> None:
        if self._metadata_tag is not None:
            self._metadata_depth -= 1
            if self._metadata_depth <= 0 and tag == self._metadata_tag:
                value = " ".join(self._metadata_parts).strip()
                if value:
                    self.metadata_text.append(value)
                self._metadata_tag = None
                self._metadata_depth = 0
                self._metadata_parts = []
        if self._version_tag is not None:
            self._version_depth -= 1
            if self._version_depth <= 0 and tag == self._version_tag:
                value = " ".join(self._version_parts).strip()
                if value:
                    self.version_record_text.append(value)
                self._version_tag = None
                self._version_depth = 0
                self._version_parts = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.text.append(data.strip())
            if self._metadata_tag is not None:
                self._metadata_parts.append(data.strip())
            else:
                self.non_metadata_text.append(data.strip())
            if self._version_tag is not None:
                self._version_parts.append(data.strip())


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def build_status_collection_plan(
    source_pool: dict[str, Any], *, seed: str, sample_size: int,
    created_at: str,
) -> dict[str, Any]:
    """Freeze deterministic, metadata-only Crossref/Crossmark requests."""
    entries = source_pool.get("entries") or []
    if source_pool.get("schema") != "g05-unlabeled-plos-source-pool-v1" or not entries:
        raise ValueError("unsupported or empty G05 source pool")
    if source_pool.get("source_pool_sha256") != _canonical_sha(entries):
        raise ValueError("source pool hash mismatch")
    if not seed.strip() or sample_size < 1 or not created_at.strip():
        raise ValueError("seed, positive sample size, and capture timestamp are required")
    if source_pool.get("candidate_count") != len(entries):
        raise ValueError("source pool candidate count mismatch")
    by_doi: dict[str, dict[str, Any]] = {}
    for row in entries:
        doi = str(row.get("doi") or "").strip().casefold()
        group = str(row.get("source_group_id") or "").strip().casefold()
        if not doi or doi != group or doi in by_doi:
            raise ValueError("source pool DOI groups must be present and unique")
        source_sha = str(row.get("source_content_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", source_sha):
            raise ValueError("source content hash is missing or invalid")
        publication_date = _expected_date(row.get("publication_date"))
        if not publication_date:
            continue
        by_doi[doi] = row
    ranked = sorted(by_doi.values(), key=lambda row: (
        hashlib.sha256("\0".join((seed, row["doi"], row["source_content_sha256"])).encode()).hexdigest(),
        row["doi"],
    ))
    selected = ranked[:sample_size]
    requests = []
    for rank, row in enumerate(selected, start=1):
        doi = row["doi"]
        pub_date = _expected_date(row.get("publication_date"))
        requests.append({
            "rank": rank,
            "doi": doi,
            "source_group_id": row["source_group_id"],
            "source_content_sha256": row["source_content_sha256"],
            "expected_identity": {
                "doi": doi,
                "title": row["title"],
                "publication_date": pub_date,
            },
            "crossref_url": f"https://api.crossref.org/works/{quote(doi, safe='')}",
            "crossmark_url": (
                "https://crossmark.crossref.org/dialog-content?"
                f"date_stamp={pub_date}&doi={quote(doi, safe='')}&domain=pdf"),
        })
    plan: dict[str, Any] = {
        "schema": "g05-publication-status-plan-v1",
        "source_pool_sha256": source_pool["source_pool_sha256"],
        "seed": seed,
        "created_at": created_at,
        "sample_size_requested": sample_size,
        "candidate_count": len(entries),
        "planned_count": len(requests),
        "ranking": "sha256(seed NUL doi NUL source_content_sha256), ascending; DOI tie-break",
        "transport": {
            "sequential": True,
            "min_interval_seconds": 6,
            "timeout_seconds": 30,
            "max_response_bytes": 2097152,
            "retry_count": 0,
        },
        "requests": requests,
        "labels_generated": False,
        "model_calls": 0,
        "target_scorer_calls": 0,
    }
    plan["plan_sha256"] = _canonical_sha(plan)
    return plan


def build_status_extension_plan(
    source_pool: dict[str, Any], parent_plan: dict[str, Any],
    parent_report: dict[str, Any], parent_sample_plan: dict[str, Any], *,
    created_at: str,
) -> dict[str, Any]:
    """Freeze Crossref/Crossmark checks for the next two authorable DOI ranks.

    The candidate choice is the next two source-pool ranks under the already
    frozen status-screen seed, excluding the 120 DOI groups already checked.
    This is a plan builder only; it performs no network or model calls.
    """
    entries = source_pool.get("entries") or []
    if source_pool.get("schema") != "g05-unlabeled-plos-source-pool-v1" or not entries:
        raise ValueError("unsupported or empty G05 source pool")
    if source_pool.get("source_pool_sha256") != _canonical_sha(entries):
        raise ValueError("source pool hash mismatch")
    if source_pool.get("candidate_count") != len(entries):
        raise ValueError("source pool candidate count mismatch")
    parent_plan_sha = parent_plan.get("plan_sha256") if isinstance(parent_plan, dict) else None
    if (not isinstance(parent_plan, dict)
            or parent_plan.get("schema") != "g05-publication-status-plan-v1"
            or not re.fullmatch(r"[0-9a-f]{64}", str(parent_plan_sha or ""))
            or parent_plan_sha != _canonical_sha({key: value for key, value in parent_plan.items()
                                                   if key != "plan_sha256"})):
        raise ValueError("parent status plan is invalid or tampered")
    if parent_plan.get("source_pool_sha256") != source_pool["source_pool_sha256"]:
        raise ValueError("parent status plan is not bound to this source pool")
    if (parent_plan.get("sample_size_requested") != 120
            or parent_plan.get("planned_count") != 120
            or len(parent_plan.get("requests") or []) != 120):
        raise ValueError("extension requires the frozen 120-DOI parent status plan")
    if (not isinstance(parent_report, dict)
            or parent_report.get("schema") != "g05-publication-status-report-v1"
            or parent_report.get("source_pool_sha256") != source_pool["source_pool_sha256"]
            or parent_report.get("status_plan_sha256") != parent_plan_sha
            or parent_report.get("verified_current_count") != 120
            or parent_report.get("unverified_excluded_count") != 0
            or len(parent_report.get("entries") or []) != 120):
        raise ValueError("parent status report is not a complete 120-DOI verified capture")
    report_sha = parent_report.get("report_sha256")
    if (not isinstance(report_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", report_sha)
            or report_sha != _canonical_sha({key: value for key, value in parent_report.items()
                                               if key != "report_sha256"})):
        raise ValueError("parent status report hash mismatch")
    if (not isinstance(parent_sample_plan, dict)
            or parent_sample_plan.get("schema") not in {
                "g05-deterministic-sample-plan-v1", "g05-deterministic-sample-plan-v2"}
            or parent_sample_plan.get("source_pool_sha256") != source_pool["source_pool_sha256"]
            or parent_sample_plan.get("status_plan_sha256") != parent_plan_sha
            or parent_sample_plan.get("publication_status_report_sha256") != report_sha):
        raise ValueError("parent sample plan is not bound to the captured status evidence")
    if (parent_sample_plan.get("schema") == "g05-deterministic-sample-plan-v2"
            and parent_sample_plan.get("authoring_context_policy")
            != "abstract_chars_400_6000_no_replacement_char_v1"):
        raise ValueError("parent v2 sample plan has an unsupported authoring-context policy")
    sample_sha = parent_sample_plan.get("plan_sha256")
    if (not isinstance(sample_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sample_sha)
            or sample_sha != _canonical_sha({key: value for key, value in parent_sample_plan.items()
                                              if key != "plan_sha256"})
            or len(parent_sample_plan.get("task_shells") or []) != 120):
        raise ValueError("parent sample plan hash or shell inventory is invalid")

    by_doi: dict[str, dict[str, Any]] = {}
    for row in entries:
        doi = str(row.get("doi") or "").strip().casefold()
        if not doi or doi in by_doi:
            raise ValueError("source pool DOI inventory is duplicate or incomplete")
        by_doi[doi] = row
    expected_parent_dois = {str(row["doi"]).casefold() for row in parent_plan["requests"]}
    report_dois = {str(row.get("doi") or "").casefold() for row in parent_report["entries"]}
    sample_dois = {str(row.get("source_doi") or "").casefold()
                   for row in parent_sample_plan["task_shells"]}
    if (len(expected_parent_dois) != 120 or report_dois != expected_parent_dois
            or sample_dois != report_dois):
        raise ValueError("parent plan, report, and sample plan DOI coverage disagree")

    seed = parent_plan.get("seed")
    if not isinstance(seed, str) or not seed.strip() or not isinstance(created_at, str) or not created_at.strip():
        raise ValueError("extension seed or creation timestamp is missing")
    ranked_rows = []
    for doi, row in by_doi.items():
        pub_date = _expected_date(row.get("publication_date"))
        if not pub_date:
            continue
        rank_hash = hashlib.sha256("\0".join((seed, doi, row["source_content_sha256"])).encode()).hexdigest()
        ranked_rows.append((rank_hash, doi, row, pub_date))
    ranked_rows.sort(key=lambda item: (item[0], item[1]))
    ordered = [item for item in ranked_rows if item[1] not in expected_parent_dois]
    choices = []
    for rank, item in enumerate(ranked_rows, start=1):
        _, doi, row, pub_date = item
        if doi in expected_parent_dois:
            continue
        abstract = str(row.get("abstract") or "").strip()
        if len(abstract) < 400 or len(abstract) > 6000 or "\ufffd" in abstract:
            continue
        choices.append((rank, doi, row, pub_date))
        if len(choices) == 2:
            break
    if len(choices) != 2:
        raise ValueError("fewer than two unverified authorable DOI candidates remain")
    # The documented v1 replacement must be exactly the next rank window, not
    # a later candidate chosen after inspecting model outputs.
    if [row[0] for row in choices] != [121, 122]:
        raise ValueError("rank-121/122 replacement candidates fail source-context preflight")

    doi_candidates, requests = [], []
    for rank, doi, row, pub_date in choices:
        source_sha = row.get("source_content_sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", str(source_sha or "")):
            raise ValueError("replacement source content hash is invalid")
        identity = {"doi": doi, "title": row["title"], "publication_date": pub_date}
        doi_candidates.append({"rank": rank, "doi": doi,
            "source_group_id": row["source_group_id"], "source_content_sha256": source_sha,
            "expected_identity": identity, "abstract_chars": len(row["abstract"].strip()),
            "license_classification": row.get("license_classification")})
        encoded = quote(doi, safe="")
        common = {"rank": rank, "doi": doi, "source_group_id": row["source_group_id"],
            "source_content_sha256": source_sha, "expected_identity": identity}
        requests.extend((
            {**common, "request_index": len(requests) + 1, "service": "crossref",
             "url": f"https://api.crossref.org/works/{encoded}"},
            {**common, "request_index": len(requests) + 2, "service": "crossmark",
             "url": ("https://crossmark.crossref.org/dialog-content?"
                     f"date_stamp={pub_date}&doi={encoded}&domain=pdf")},
        ))
    plan: dict[str, Any] = {
        "schema": "g05-publication-status-extension-plan-v1",
        "source_pool_sha256": source_pool["source_pool_sha256"],
        "parent_status_plan_sha256": parent_plan_sha,
        "parent_status_report_sha256": report_sha,
        "parent_sample_plan_sha256": sample_sha,
        "seed": seed,
        "selection_rule": "same parent status seed/rank; first two authorable DOI groups after ranks 1-120",
        "parent_rank_count": 120,
        "rank_window": [121, 122],
        "doi_candidates": doi_candidates,
        "requests": requests,
        "doi_candidate_count": 2,
        "physical_request_count": 4,
        "max_physical_attempts": 4,
        "retry_count": 0,
        "transport": {"sequential": True, "min_interval_seconds": 6,
            "timeout_seconds": 30, "max_response_bytes": 2097152},
        "model_calls": 0,
        "target_scorer_calls": 0,
        "labels_generated": False,
    }
    plan["plan_sha256"] = _canonical_sha(plan)
    return plan


def merge_status_extension_reports(
    source_pool: dict[str, Any], parent_plan: dict[str, Any],
    parent_report: dict[str, Any], parent_sample_plan: dict[str, Any],
    extension_plan: dict[str, Any], extension_report: dict[str, Any], *,
    source_pool_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a new composite status plan/report without changing v3 inputs."""
    expected_extension = build_status_extension_plan(
        source_pool, parent_plan, parent_report, parent_sample_plan,
        # v1 extension plans predate a serialized created_at field. Selection
        # does not depend on it, so use a stable sentinel and preserve v1 hash.
        created_at="extension-plan-v1-timestamp-not-in-schema",
    )
    if extension_plan != expected_extension:
        raise ValueError("extension plan differs from deterministic parent-bound plan")
    if (extension_report.get("schema") != "g05-publication-status-extension-report-v1"
            or extension_report.get("source_pool_sha256") != source_pool.get("source_pool_sha256")
            or extension_report.get("source_pool_file_sha256") != source_pool_file_sha256
            or extension_report.get("parent_status_plan_sha256") != parent_plan.get("plan_sha256")
            or extension_report.get("parent_status_report_sha256") != parent_report.get("report_sha256")
            or extension_report.get("parent_sample_plan_sha256") != parent_sample_plan.get("plan_sha256")
            or extension_report.get("extension_plan_sha256") != extension_plan.get("plan_sha256")
            or extension_report.get("planned_count") != 2
            or extension_report.get("physical_request_count") != 4
            or extension_report.get("model_calls") != 0
            or extension_report.get("target_scorer_calls") != 0
            or extension_report.get("labels_generated") is not False):
        raise ValueError("extension report is incomplete or bound to different provenance")
    ext_sha = extension_report.get("report_sha256")
    if (not isinstance(ext_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", ext_sha)
            or ext_sha != _canonical_sha({k: v for k, v in extension_report.items()
                                          if k != "report_sha256"})):
        raise ValueError("extension report hash mismatch")
    expected_dois = {row["doi"] for row in extension_plan["doi_candidates"]}
    actual_dois = [str(row.get("doi") or "").casefold()
                   for row in extension_report.get("entries") or []]
    if len(actual_dois) != 2 or set(actual_dois) != expected_dois or len(set(actual_dois)) != 2:
        raise ValueError("extension report DOI coverage does not match the frozen plan")
    if (extension_report.get("verified_current_count") != sum(
            bool(row.get("eligible_for_status_screen")) for row in extension_report["entries"])
            or extension_report.get("unverified_excluded_count") != sum(
                not bool(row.get("eligible_for_status_screen")) for row in extension_report["entries"])):
        raise ValueError("extension report status counts do not match its entries")
    for row in extension_report["entries"]:
        if row.get("scope") != "point_in_time_only":
            raise ValueError("extension status must remain point-in-time only")
        receipts = row.get("transport") or {}
        if set(receipts) != {"crossref", "crossmark"}:
            raise ValueError("extension entry is missing endpoint receipts")
        for endpoint in ("crossref", "crossmark"):
            receipt = receipts[endpoint]
            raw_sha = (row.get("raw_response_sha256") or {}).get(endpoint)
            if (not receipt.get("raw_path") or receipt.get("response_sha256") != raw_sha
                    or not re.fullmatch(r"[0-9a-f]{64}", str(raw_sha or ""))):
                raise ValueError("extension entry lacks hash-bound raw response receipt")

    parent_dois = [str(row.get("doi") or "").casefold() for row in parent_report.get("entries") or []]
    if len(parent_dois) != 120 or len(set(parent_dois)) != 120 or set(parent_dois) & set(actual_dois):
        raise ValueError("parent and extension status DOI groups overlap or parent coverage is incomplete")
    component_plan_shas = [parent_plan["plan_sha256"], extension_plan["plan_sha256"]]
    composite_plan: dict[str, Any] = {
        "schema": "g05-publication-status-composite-plan-v1",
        "source_pool_sha256": source_pool["source_pool_sha256"],
        "source_pool_file_sha256": source_pool_file_sha256,
        "component_plan_sha256s": component_plan_shas,
        "component_report_sha256s": [parent_report["report_sha256"], ext_sha],
        "doi_group_count": 122,
        "selection_rule": "immutable parent 120 DOI screen plus deterministic rank-121/122 extension",
        "scope": "point_in_time_only",
        "model_calls": 0,
        "target_scorer_calls": 0,
        "labels_generated": False,
    }
    composite_plan["plan_sha256"] = _canonical_sha(composite_plan)
    entries = list(parent_report["entries"]) + list(extension_report["entries"])
    report: dict[str, Any] = {
        "schema": "g05-publication-status-report-v1",
        "source_pool_file_sha256": source_pool_file_sha256,
        "source_pool_sha256": source_pool["source_pool_sha256"],
        "status_plan_sha256": composite_plan["plan_sha256"],
        "status_plan_file_sha256": None,
        "component_provenance": {
            "parent_status_plan_sha256": parent_plan["plan_sha256"],
            "parent_status_report_sha256": parent_report["report_sha256"],
            "extension_plan_sha256": extension_plan["plan_sha256"],
            "extension_report_sha256": ext_sha,
            "composite_plan_sha256": composite_plan["plan_sha256"],
        },
        "capture_started_at": parent_report.get("capture_started_at"),
        "capture_completed_at": extension_report.get("capture_completed_at"),
        "planned_count": len(entries),
        "verified_current_count": sum(bool(row.get("eligible_for_status_screen")) for row in entries),
        "unverified_excluded_count": sum(not bool(row.get("eligible_for_status_screen")) for row in entries),
        "scope": "point_in_time_only",
        "entries": entries,
        "labels_generated": False,
        "model_calls": 0,
        "target_scorer_calls": 0,
    }
    report["report_sha256"] = _canonical_sha(report)
    return composite_plan, report


def classify_publication_status(
    expected_source: dict[str, Any],
    crossref_payload: dict[str, Any] | None,
    crossmark_html: str | None,
) -> dict[str, Any]:
    """Classify one captured Crossref/Crossmark pair without network access.

    Only exact DOI/title/date identity and an explicit Crossmark current state
    pass. Missing update metadata is recorded as an observation at capture;
    it is never interpreted as permanent absence of an update.
    """
    doi = str(expected_source.get("doi") or "").strip().casefold()
    title = str(expected_source.get("title") or "").strip()
    expected_date = _expected_date(expected_source.get("publication_date"))
    checks: dict[str, bool] = {
        "expected_identity_complete": bool(doi and title and expected_date),
        "crossref_doi_match": False,
        "crossref_title_match": False,
        "crossref_publication_date_match": False,
        "crossref_is_journal_article": False,
        "crossref_update_absent_at_capture": False,
        "crossmark_explicit_current": False,
        "crossmark_doi_match": False,
        "crossmark_title_match": False,
        "crossmark_publication_date_match": False,
        "crossmark_no_change_notice": False,
    }
    observation = "unknown_or_not_observed"

    if isinstance(crossref_payload, dict):
        checks["crossref_doi_match"] = str(crossref_payload.get("DOI") or "").strip().casefold() == doi
        titles = crossref_payload.get("title") or []
        checks["crossref_title_match"] = bool(titles and _norm(str(titles[0])) == _norm(title))
        checks["crossref_publication_date_match"] = (
            _crossref_date(crossref_payload) == expected_date and expected_date is not None)
        checks["crossref_is_journal_article"] = crossref_payload.get("type") == "journal-article"
        update_to = crossref_payload.get("update-to")
        if update_to is None:
            checks["crossref_update_absent_at_capture"] = True
            observation = "no_update_observed_at_capture"
        elif isinstance(update_to, list) and not update_to:
            checks["crossref_update_absent_at_capture"] = True
            observation = "no_update_observed_at_capture"
        else:
            observation = "update_observed_at_capture"

    if isinstance(crossmark_html, str) and crossmark_html.strip():
        parser = _CrossmarkParser()
        try:
            parser.feed(crossmark_html)
            parser.close()
        except Exception:
            parser = _CrossmarkParser()
        body = _norm(" ".join(parser.text))
        status = {item.casefold() for item in parser.status_classes}
        checks["crossmark_explicit_current"] = (
            "document-status--current" in status and "document is current" in body)
        def is_expected_doi_link(link: str) -> bool:
            parsed = urlparse(unquote(link))
            return (parsed.hostname in {"doi.org", "www.doi.org", "dx.doi.org"}
                    and parsed.path.lstrip("/").casefold() == doi)

        checks["crossmark_doi_match"] = any(is_expected_doi_link(link) for link in parser.hrefs)
        metadata = [_norm(value) for value in parser.metadata_text]
        checks["crossmark_title_match"] = _norm(title) in metadata
        month_names = ("january", "february", "march", "april", "may", "june", "july",
                       "august", "september", "october", "november", "december")
        expected_text_date = None
        if expected_date:
            parsed = date.fromisoformat(expected_date)
            expected_text_date = f"{month_names[parsed.month - 1]} {parsed.day}, {parsed.year}"
        # Crossmark's records section is not consistently well-formed HTML.
        # Its element-depth parser can lose sync on real publisher pages, so
        # find the exact status phrase in the complete visible body instead.
        # The date token is then parsed as a real calendar date and must match
        # the frozen Crossref date exactly.
        version_marker = re.search(r"\bversion of record\s+valid from\b", body)
        version_marker_present = version_marker is not None
        version_dates: list[str] = []
        if version_marker:
            tail = body[version_marker.end():]
            date_match = re.match(r"\s+(\d{4}-\d{2}-\d{2})\b", tail)
            if date_match:
                try:
                    version_dates.append(date.fromisoformat(date_match.group(1)).isoformat())
                except ValueError:
                    pass
        if version_marker_present:
            checks["crossmark_publication_date_match"] = bool(
                expected_date and version_dates and all(d == expected_date for d in version_dates))
        else:
            checks["crossmark_publication_date_match"] = bool(
                expected_text_date and any(expected_text_date in value for value in metadata))
        status_variants = {cls for cls in status if cls.startswith("document-status--")}
        notice_text = _norm(" ".join(parser.non_metadata_text))
        plain_change_notice = bool(re.search(
            r"\b(?:change|update)\s+(?:date|type)\b|"
            r"\b(?:retraction|retracted|correction|corrected|expression of concern)\b",
            notice_text,
        ))
        checks["crossmark_no_change_notice"] = (
            not parser.change_classes and not plain_change_notice
            and status_variants == {"document-status--current"})

    identity_checks = (
        "expected_identity_complete", "crossref_doi_match", "crossref_title_match",
        "crossref_publication_date_match", "crossref_is_journal_article",
        "crossref_update_absent_at_capture", "crossmark_explicit_current",
        "crossmark_doi_match", "crossmark_title_match",
        "crossmark_publication_date_match", "crossmark_no_change_notice",
    )
    eligible = all(checks[key] for key in identity_checks)
    return {
        "doi": doi or None,
        "disposition": "verified_current_as_of_capture" if eligible else "unverified_exclude",
        "eligible_for_status_screen": eligible,
        "scope": "point_in_time_only",
        "crossref_update_observation": observation,
        "checks": checks,
    }
