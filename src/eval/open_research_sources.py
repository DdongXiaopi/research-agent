"""Reproducible, license-aware Europe PMC source capture for G09.

The frozen v5 entrypoint searches bounded core pages and checks returned
abstracts locally. A legacy metadata-first helper remains for module diagnostics
only; production plans must pass through the content-bound frozen entrypoint.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from ..util import sha256_obj
from .open_research_cohort import HISTORICAL_DOI_PREFIX_EXCLUSIONS

API_ROOT = "https://www.ebi.ac.uk/europepmc/webservices/rest"
MAX_SOURCE_RESPONSE_BYTES = 8 * 1024 * 1024
HISTORICALLY_EXPOSED_DOIS = (
    "10.1371/journal.pone.0302385",
    "10.1371/journal.pone.0297184",
    "10.1371/journal.pone.0000332",
)
TOPIC_SLOT_CORE_CAPS = {"regular": 4, "conflict": 8, "replacement": 2}
FROZEN_TOPIC_CORE_PAGE_CAPS = {"regular": 5, "conflict": 2, "replacement": 3}
FROZEN_TOPIC_CORE_PAGE_SIZE = 10
FROZEN_TOPIC_CORE_REQUEST_RESERVE = 3
MAX_TOPIC_METADATA_REQUESTS = 60
MAX_TOPIC_METADATA_PAGE_SIZE = 1000
MAX_TOPIC_CORE_REQUESTS = 252
FROZEN_TOPIC_SLOT_ALLOCATION = {"regular": 39, "conflict": 9, "replacement": 12}
TOPIC_SELECTION_POLICY_VERSION = "europe-pmc-core-search-pico-pages-v5"
TOPIC_SELECTION_POLICY = {
    "version": TOPIC_SELECTION_POLICY_VERSION,
    "query_scope": "condition and intervention are field-scoped to TITLE or ABSTRACT for recall; population and outcome are checked on returned core abstracts",
    "candidate_order": "retain Europe PMC relevance order through cursor pagination; do not rank lite metadata by title-only evidence",
    "core_page_size": FROZEN_TOPIC_CORE_PAGE_SIZE,
    "core_page_caps_by_slot_kind": FROZEN_TOPIC_CORE_PAGE_CAPS,
    "core_request_reserve": FROZEN_TOPIC_CORE_REQUEST_RESERVE,
    "core_eligibility": "every source must contain a literal normalized title-or-abstract phrase for all four PICO dimensions",
    "synonym_policy": "no stemming, embeddings, inferred synonyms, or semantic-model judgment",
    "refill": "skip ineligible identity/license/abstract/PICO candidates and continue pages only within the frozen per-slot page cap",
    "cohort_gate": "only 48+ independently curated complete topic groups can proceed; otherwise diagnostic-only",
}
FROZEN_TOPIC_PLAN_SCHEMA = "g09-frozen-topic-plan-v5"
TOPIC_SOURCE_CAPTURE_SCHEMA = "europe-pmc-topic-stratified-source-capture-v5"
PICO_DIMENSIONS = {
    "condition": "condition_terms",
    "population": "population_terms",
    "intervention": "intervention_terms",
    "outcome": "outcome_terms",
}


def source_query(capture_date: str) -> str:
    if not isinstance(capture_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", capture_date):
        raise ValueError("capture date must be YYYY-MM-DD")
    try:
        date.fromisoformat(capture_date)
    except ValueError as exc:
        raise ValueError("capture date must be YYYY-MM-DD") from exc
    return ("OPEN_ACCESS:Y AND LICENSE:cc-by AND HAS_ABSTRACT:Y AND "
            f"FIRST_PDATE:[2020-01-01 TO {capture_date}] NOT DOI:10.1371*")


def _doi(row: dict) -> str:
    value = row.get("doi")
    return value.strip().casefold() if isinstance(value, str) else ""


def select_lite_candidates(rows: list[dict], *, target_records: int = 36,
                           max_per_journal: int = 4) -> list[dict]:
    """Select a stable, diversified metadata-only candidate set."""
    if type(target_records) is not int or target_records < 24:
        raise ValueError("source capture must target at least 24 article groups")
    if type(max_per_journal) is not int or max_per_journal < 1:
        raise ValueError("journal cap must be positive")
    excluded = {value.casefold() for value in HISTORICALLY_EXPOSED_DOIS}
    usable = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        doi = _doi(row)
        pmcid = row.get("pmcid")
        journal = row.get("journalTitle")
        title = row.get("title")
        if (not doi or doi in excluded or any(doi.startswith(prefix)
                for prefix in HISTORICAL_DOI_PREFIX_EXCLUSIONS)):
            continue
        if doi in seen:
            continue
        if (not isinstance(pmcid, str) or not pmcid.startswith("PMC") or
                not isinstance(journal, str) or not journal.strip() or
                not isinstance(title, str) or len(title.strip()) < 8):
            continue
        seen.add(doi)
        usable.append({"doi": doi, "pmcid": pmcid, "journal": journal.strip(),
                       "title": title.strip(), "pubYear": str(row.get("pubYear") or "")})
    usable.sort(key=lambda row: hashlib.sha256(row["doi"].encode()).hexdigest())
    selected, journal_counts = [], {}
    for row in usable:
        journal = row["journal"]
        if journal_counts.get(journal, 0) >= max_per_journal:
            continue
        selected.append(row)
        journal_counts[journal] = journal_counts.get(journal, 0) + 1
        if len(selected) >= target_records and len(journal_counts) >= 12:
            break
    if len(selected) < target_records or len(journal_counts) < 12:
        raise ValueError("metadata pool cannot satisfy record and journal diversity minimum")
    return selected


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def bounded_source_response(url: str, *, opener, user_agent: str) -> tuple[int, bytes]:
    """Only fetch bounded Europe PMC API responses; production redirects are rejected."""
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname != "www.ebi.ac.uk"
            or parsed.username is not None or parsed.password is not None
            or parsed.port not in (None, 443)
            or not parsed.path.startswith("/europepmc/webservices/rest/")):
        raise ValueError("source capture requires the approved Europe PMC HTTPS API")
    request = Request(url, headers={"User-Agent": user_agent})
    transport = build_opener(_NoRedirect()).open if opener is urlopen else opener
    with transport(request, timeout=45) as response:
        status = getattr(response, "status", 200)
        body = response.read(MAX_SOURCE_RESPONSE_BYTES + 1)
    if len(body) > MAX_SOURCE_RESPONSE_BYTES:
        raise ValueError("Europe PMC response exceeds the byte limit")
    return status, body


def _request_json(url: str, *, opener=urlopen) -> tuple[dict, bytes]:
    status, body = bounded_source_response(
        url, opener=opener, user_agent="G09-research-assistant-evaluation/1.0")
    if status != 200:
        raise RuntimeError(f"Europe PMC request failed with HTTP {status}")
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Europe PMC returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Europe PMC JSON response must be an object")
    return value, body


def capture_europe_pmc_source_pool(*, capture_date: str, captured_at: str,
        target_records: int = 36, max_per_journal: int = 4, opener=urlopen) -> dict:
    """Capture metadata first, then fetch core only for eligible non-PLOS articles."""
    try:
        timestamp = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("captured_at must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError("captured_at must include a timezone")
    query = source_query(capture_date)
    search_url = API_ROOT + "/search?" + urlencode({
        "query": query, "format": "json", "resulttype": "lite", "pageSize": 1000, "page": 1})
    lite, lite_raw = _request_json(search_url, opener=opener)
    rows = ((lite.get("resultList") or {}).get("result"))
    if not isinstance(rows, list):
        raise ValueError("Europe PMC metadata response lacks resultList.result")
    candidates = select_lite_candidates(rows, target_records=max(target_records * 2, target_records + 16),
                                        max_per_journal=max_per_journal)
    captured_records, core_payloads, core_exclusions = [], [], []
    core_attempts = 0
    for candidate in candidates:
        core_url = API_ROOT + "/search?" + urlencode({
            "query": "PMCID:" + candidate["pmcid"], "format": "json",
            "resulttype": "core", "pageSize": 2, "page": 1})
        core_search, raw = _request_json(core_url, opener=opener)
        core_attempts += 1
        core_rows = ((core_search.get("resultList") or {}).get("result"))
        if not isinstance(core_rows, list) or len(core_rows) != 1:
            raise ValueError("Europe PMC PMCID lookup did not return exactly one core record")
        core = core_rows[0]
        doi = _doi(core)
        license_value = str(core.get("license") or "").strip().casefold()
        if doi != candidate["doi"]:
            raise ValueError("Europe PMC core DOI differs from metadata candidate")
        if str(core.get("pmcid") or "").strip().casefold() != candidate["pmcid"].casefold():
            raise ValueError("Europe PMC core PMCID differs from metadata candidate")
        if any(doi.startswith(prefix) for prefix in HISTORICAL_DOI_PREFIX_EXCLUSIONS):
            raise ValueError("PLOS DOI family is forbidden in fetched source records")
        if license_value != "cc by":
            core_exclusions.append({"doi": doi, "pmcid": candidate["pmcid"],
                "reason": "core_license_not_cc_by", "response_sha256": hashlib.sha256(raw).hexdigest()})
            continue
        if not isinstance(core.get("abstractText"), str) or len(core["abstractText"].strip()) < 400:
            core_exclusions.append({"doi": doi, "pmcid": candidate["pmcid"],
                "reason": "abstract_missing_or_too_short", "response_sha256": hashlib.sha256(raw).hexdigest()})
            continue
        captured_records.append(core)
        core_payloads.append({"pmcid": candidate["pmcid"], "response_sha256": hashlib.sha256(raw).hexdigest()})
        if len(captured_records) == target_records:
            break
    if len(captured_records) != target_records:
        raise ValueError("core eligibility filtering could not refill the exact licensed source target")
    body = {"schema": "europe-pmc-ccby-core-capture-v1", "endpoint": API_ROOT,
        "query": query, "capture_date": capture_date, "captured_at": captured_at,
        "search_request_url": search_url,
        "search_response_sha256": hashlib.sha256(lite_raw).hexdigest(),
        "search_hit_count": (lite.get("hitCount")), "search_metadata_records": len(rows),
        "search_resulttype": "lite", "target_records": target_records,
        "max_per_journal": max_per_journal, "core_attempts": core_attempts,
        "core_record_exclusions": core_exclusions,
        "core_record_sha256": core_payloads,
        "records": captured_records,
        "license_policy": "Every fetched core record explicitly reports license=cc by; source-family exclusion applied before core fetch.",
        "excluded_doi_prefixes": list(HISTORICAL_DOI_PREFIX_EXCLUSIONS),
        "historically_exposed_doi_identities": list(HISTORICALLY_EXPOSED_DOIS),
        "historic_exposure_scope": "DOI/URL identity only; no old final abstract, rubric, answer or score fetched or read."}
    body["capture_sha256"] = sha256_obj(body)
    return body


def _query_terms(value: object, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > 8:
        raise ValueError(f"{label} query terms must be a non-empty list of at most 8 values")
    result = []
    for term in value:
        if not isinstance(term, str):
            raise ValueError(f"{label} query term must be a string")
        normalized = " ".join(term.split())
        if (not normalized or len(normalized) > 128 or
                any(ord(char) < 32 or ord(char) == 127 for char in term)):
            raise ValueError(f"invalid {label} query term")
        # Europe PMC receives these as quoted phrases; escape phrase delimiters
        # before urlencode encodes the query parameter.
        result.append(normalized.replace("\\", "\\\\").replace('"', '\\"'))
    return result


def _normalized_phrase_text(value: str) -> str:
    """Normalize punctuation and whitespace for conservative phrase matching."""
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def assess_topic_pico_relevance(record: dict, topic_spec: dict) -> dict:
    """Require a literal title/abstract phrase for every registered PICO dimension.

    This is a conservative eligibility filter, not a semantic relevance judge.
    Missing even one dimension fails closed; no stemming, embedding similarity,
    or inferred synonym expansion is applied.
    """
    if not isinstance(record, dict) or not isinstance(topic_spec, dict):
        raise ValueError("PICO relevance requires source record and topic specification objects")
    title = record.get("title") if isinstance(record.get("title"), str) else ""
    abstract = record.get("abstractText") if isinstance(record.get("abstractText"), str) else ""
    text = _normalized_phrase_text(title + " " + abstract)
    matched_terms, missing = {}, []
    for dimension, field in PICO_DIMENSIONS.items():
        values = _query_terms(topic_spec.get(field), label=field)
        matches = []
        for term in values:
            normalized = _normalized_phrase_text(term)
            if normalized and f" {normalized} " in f" {text} ":
                matches.append(term.replace('\\"', '"'))
        matched_terms[dimension] = matches
        if not matches:
            missing.append(dimension)
    # Guard the common HFpEF/HFrEF phenotype inversion even where a source
    # mentions a generic ejection-fraction phrase elsewhere in its abstract.
    population_terms = " ".join(
        _normalized_phrase_text(term)
        for term in topic_spec.get("population_terms", [])
        if isinstance(term, str))
    if ("reduced ejection fraction" in population_terms and
            "preserved ejection fraction" in text and
            "reduced ejection fraction" not in text):
        if "population" not in missing:
            missing.append("population")
            matched_terms["population"] = []
    return {"eligible": not missing, "matched_terms": matched_terms,
        "missing_dimensions": missing}


def build_topic_query(topic_spec: dict, *, capture_date: str) -> str:
    """Build a bounded Europe PMC query from one explicit topic/PICO slot."""
    source_query(capture_date)  # validate the date using the base query contract
    if not isinstance(topic_spec, dict):
        raise ValueError("topic query specification must be an object")
    expected_fields = {"topic_id", "topic", "slot_kind", "condition_terms",
        "population_terms", "intervention_terms", "outcome_terms"}
    if set(topic_spec) != expected_fields:
        raise ValueError("topic query specification has missing or unsupported fields")
    topic_id = topic_spec.get("topic_id")
    if not isinstance(topic_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", topic_id):
        raise ValueError("topic_id must be a stable lowercase slug")
    topic = topic_spec.get("topic")
    if (not isinstance(topic, str) or not topic.strip() or len(topic.strip()) > 240 or
            any(ord(char) < 32 or ord(char) == 127 for char in topic)):
        raise ValueError("topic must be a non-empty bounded plain-text label")
    slot_kind = topic_spec.get("slot_kind")
    if slot_kind not in TOPIC_SLOT_CORE_CAPS:
        raise ValueError("topic slot_kind must be regular, conflict or replacement")
    prepared_terms = {field: _query_terms(topic_spec.get(field), label=field)
        for field in PICO_DIMENSIONS.values()}
    dimensions = []
    # Core-search pages expose abstracts, so use condition+intervention for
    # bounded recall and apply the unchanged strict four-dimension local gate
    # before any source is accepted.
    for field in ("condition_terms", "intervention_terms"):
        values = prepared_terms[field]
        # Europe PMC bare terms also search indexed full text. G09 sends only
        # title/abstract material to the assistant, so scope every retrieval
        # phrase to those same fields to avoid full-text-only false positives.
        dimensions.append("(" + " OR ".join(
            f'(TITLE:"{term}" OR ABSTRACT:"{term}")' for term in values) + ")")
    base = source_query(capture_date)
    return base + " AND " + " AND ".join(dimensions)


def select_topic_candidates(rows: list[dict], *, topic_id: str, max_candidates: int,
        max_per_journal: int = 1, metadata_response_sha256: str | None = None,
        topic_spec: dict | None = None) -> list[dict]:
    """Keep Europe PMC rank while applying a bounded deterministic title PICO rerank."""
    if not isinstance(topic_id, str) or not topic_id.strip():
        raise ValueError("topic_id is required for stable topic selection")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    if type(max_per_journal) is not int or max_per_journal < 1:
        raise ValueError("max_per_journal must be a positive integer")
    if metadata_response_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", metadata_response_sha256):
        raise ValueError("metadata response identity must be a lowercase SHA-256")
    excluded = {value.casefold() for value in HISTORICALLY_EXPOSED_DOIS}
    candidates, seen_dois, seen_pmcids = [], set(), set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        doi = _doi(row)
        pmcid = row.get("pmcid")
        journal = row.get("journalTitle")
        title = row.get("title")
        if (not doi or doi in excluded or any(doi.startswith(prefix)
                for prefix in HISTORICAL_DOI_PREFIX_EXCLUSIONS)):
            continue
        if (not isinstance(pmcid, str) or not pmcid.startswith("PMC") or
                not isinstance(journal, str) or not journal.strip() or
                not isinstance(title, str) or len(title.strip()) < 8):
            continue
        pmcid_key = pmcid.strip().casefold()
        if doi in seen_dois or pmcid_key in seen_pmcids:
            continue
        seen_dois.add(doi)
        seen_pmcids.add(pmcid_key)
        candidate = {"doi": doi, "pmcid": pmcid.strip(),
            "journal": journal.strip(), "title": title.strip(),
            "pubYear": str(row.get("pubYear") or ""),
            "metadata_rank": row.get("_europe_pmc_rank", row_index)}
        if metadata_response_sha256 is not None:
            candidate["metadata_response_sha256"] = metadata_response_sha256
        if topic_spec is not None:
            assessment = assess_topic_pico_relevance(candidate, topic_spec)
            candidate["title_pico_dimensions_matched"] = (
                len(PICO_DIMENSIONS) - len(assessment["missing_dimensions"]))
        else:
            candidate["title_pico_dimensions_matched"] = 0
        candidates.append(candidate)
    candidates.sort(key=lambda row: (-row["title_pico_dimensions_matched"],
        row["metadata_rank"], row["doi"]))
    selected, journal_counts = [], {}
    for row in candidates:
        journal = row["journal"]
        if journal_counts.get(journal, 0) >= max_per_journal:
            continue
        selected.append(row)
        journal_counts[journal] = journal_counts.get(journal, 0) + 1
        if len(selected) >= max_candidates:
            break
    return selected


def capture_topic_stratified_source_pool(*, topic_specs: list[dict], capture_date: str,
        captured_at: str, target_records_per_topic: int = 2, metadata_page_size: int = 100,
        opener=urlopen) -> dict:
    """Capture per-topic CC BY records with hard per-slot request caps and audit logs.

    Identity mismatches terminate the whole capture immediately. Per-record
    license and abstract deficiencies are logged and skipped while the topic's
    bounded core request budget remains. This function only creates a source
    pool; a separate typed cohort builder must validate topic and conflict
    evidence before any rubric or assistant evaluation is authorized.
    """
    source_query(capture_date)
    try:
        timestamp = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("captured_at must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError("captured_at must include a timezone")
    if (not isinstance(topic_specs, list) or not topic_specs or
            len(topic_specs) > MAX_TOPIC_METADATA_REQUESTS):
        raise ValueError("source capture allows 1 to 60 topic metadata requests")
    if type(target_records_per_topic) is not int or target_records_per_topic != 2:
        raise ValueError("each topic group must target exactly two distinct source records")
    if type(metadata_page_size) is not int or not 1 <= metadata_page_size <= MAX_TOPIC_METADATA_PAGE_SIZE:
        raise ValueError("metadata page size must be between 1 and 100")

    normalized_specs = []
    seen_topic_ids = set()
    for spec in topic_specs:
        query = build_topic_query(spec, capture_date=capture_date)
        topic_id = spec["topic_id"]
        if topic_id in seen_topic_ids:
            raise ValueError("topic_id values must be unique within one capture")
        seen_topic_ids.add(topic_id)
        normalized_specs.append((spec, query))
    requested_core_budget = sum(TOPIC_SLOT_CORE_CAPS[spec["slot_kind"]]
        for spec, _ in normalized_specs)
    if requested_core_budget > MAX_TOPIC_CORE_REQUESTS:
        raise ValueError("topic capture core-request budget exceeds the frozen 252-request maximum")

    request_log, topic_results = [], []
    used_dois, used_pmcids = set(), set()
    fatal_identity_error = None

    def fetch(url: str, *, topic_id: str, phase: str):
        entry = {"topic_id": topic_id, "phase": phase, "url": url, "method": "GET"}
        try:
            value, raw = _request_json(url, opener=opener)
        except Exception as exc:
            entry.update({"http_status": None, "error_type": type(exc).__name__,
                "error": str(exc)[:500]})
            request_log.append(entry)
            return None, None
        entry.update({"http_status": 200, "response_bytes": len(raw),
            "response_sha256": hashlib.sha256(raw).hexdigest()})
        request_log.append(entry)
        return value, raw

    for spec, query in normalized_specs:
        topic_id, slot_kind = spec["topic_id"], spec["slot_kind"]
        cap = TOPIC_SLOT_CORE_CAPS[slot_kind]
        topic = {"topic_id": topic_id, "topic": " ".join(spec["topic"].split()),
            "slot_kind": slot_kind, "query": query, "records": [],
            "core_request_cap": cap, "core_attempts": 0,
            "exclusions": [], "candidate_audit": [], "status": "incomplete"}
        metadata_url = API_ROOT + "/search?" + urlencode({
            "query": query, "format": "json", "resulttype": "lite",
            "pageSize": metadata_page_size, "page": 1})
        lite, metadata_raw = fetch(metadata_url, topic_id=topic_id, phase="metadata")
        if lite is None:
            topic["failure_reason"] = "metadata_request_failed"
            topic_results.append(topic)
            continue
        rows = ((lite.get("resultList") or {}).get("result"))
        if not isinstance(rows, list):
            request_log[-1].update({"error_type": "invalid_metadata_response",
                "error": "Europe PMC metadata response lacks resultList.result"})
            topic["failure_reason"] = "invalid_metadata_response"
            topic_results.append(topic)
            continue
        topic["metadata_response_sha256"] = hashlib.sha256(metadata_raw).hexdigest()
        reported_hit_count = lite.get("hitCount")
        topic["metadata_hit_count"] = (reported_hit_count
            if type(reported_hit_count) is int and reported_hit_count >= 0 else None)
        topic["metadata_rows_returned"] = len(rows)
        # Remove identities already consumed by earlier topic slots before
        # applying the per-topic candidate cap, so duplicates do not crowd out
        # eligible refill candidates.
        unique_rows = []
        for metadata_rank, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            row_doi = _doi(row)
            row_pmcid = row.get("pmcid")
            row_pmcid_key = row_pmcid.strip().casefold() if isinstance(row_pmcid, str) else ""
            if row_doi in used_dois or (row_pmcid_key and row_pmcid_key in used_pmcids):
                topic["exclusions"].append({"doi": row_doi, "pmcid": row_pmcid,
                    "reason": "duplicate_source_across_topic_groups",
                    "stage": "preselection"})
                continue
            unique_rows.append({**row, "_europe_pmc_rank": metadata_rank})
        candidates = select_topic_candidates(unique_rows, topic_id=topic_id,
            max_candidates=cap, max_per_journal=1,
            metadata_response_sha256=topic["metadata_response_sha256"],
            topic_spec=spec)
        topic["core_candidate_slots_selected"] = len(candidates)
        for candidate in candidates:
            if len(topic["records"]) >= target_records_per_topic:
                break
            doi, pmcid = candidate["doi"], candidate["pmcid"]
            if doi in used_dois or pmcid.casefold() in used_pmcids:
                topic["exclusions"].append({"doi": doi, "pmcid": pmcid,
                    "reason": "duplicate_source_across_topic_groups"})
                continue
            core_url = API_ROOT + "/search?" + urlencode({
                "query": "PMCID:" + pmcid, "format": "json", "resulttype": "core",
                "pageSize": 2, "page": 1})
            core_search, raw = fetch(core_url, topic_id=topic_id, phase="core")
            topic["core_attempts"] += 1
            audit = {"doi": doi, "pmcid": pmcid,
                "metadata_rank": candidate["metadata_rank"],
                "metadata_response_sha256": candidate["metadata_response_sha256"],
                "title_pico_dimensions_matched": candidate["title_pico_dimensions_matched"],
                "core_checked": core_search is not None}
            topic["candidate_audit"].append(audit)
            if core_search is None:
                topic["exclusions"].append({"doi": doi, "pmcid": pmcid,
                    "reason": "core_request_failed"})
                continue
            core_rows = ((core_search.get("resultList") or {}).get("result"))
            if not isinstance(core_rows, list) or len(core_rows) != 1:
                fatal_identity_error = "Europe PMC PMCID lookup did not return exactly one core record"
                request_log[-1].update({"identity_error": fatal_identity_error})
                break
            core = core_rows[0]
            doi_core, pmcid_core = _doi(core), str(core.get("pmcid") or "").strip()
            if doi_core != doi:
                fatal_identity_error = "Europe PMC core DOI differs from metadata candidate"
            elif pmcid_core.casefold() != pmcid.casefold():
                fatal_identity_error = "Europe PMC core PMCID differs from metadata candidate"
            elif any(doi_core.startswith(prefix) for prefix in HISTORICAL_DOI_PREFIX_EXCLUSIONS):
                fatal_identity_error = "PLOS DOI family is forbidden in fetched source records"
            if fatal_identity_error:
                request_log[-1].update({"identity_error": fatal_identity_error})
                break
            license_value = str(core.get("license") or "").strip().casefold()
            response_hash = hashlib.sha256(raw).hexdigest()
            if license_value != "cc by":
                topic["exclusions"].append({"doi": doi, "pmcid": pmcid,
                    "reason": "core_license_not_cc_by", "response_sha256": response_hash})
                continue
            abstract = core.get("abstractText")
            if not isinstance(abstract, str) or len(abstract.strip()) < 400:
                topic["exclusions"].append({"doi": doi, "pmcid": pmcid,
                    "reason": "abstract_missing_or_too_short", "response_sha256": response_hash})
                continue
            relevance = assess_topic_pico_relevance(core, spec)
            audit.update({"abstract_pico_relevance": relevance,
                "core_response_sha256": hashlib.sha256(raw).hexdigest()})
            if not relevance["eligible"]:
                topic["exclusions"].append({"doi": doi, "pmcid": pmcid,
                    "reason": "pico_relevance_missing_dimensions",
                    "missing_dimensions": relevance["missing_dimensions"],
                    "matched_terms": relevance["matched_terms"],
                    "response_sha256": hashlib.sha256(raw).hexdigest(),
                    "metadata_rank": candidate["metadata_rank"]})
                continue
            topic["records"].append(core)
            used_dois.add(doi)
            used_pmcids.add(pmcid.casefold())
            topic.setdefault("core_record_sha256", []).append({
                "doi": doi, "pmcid": pmcid, "response_sha256": response_hash})
        if fatal_identity_error:
            topic["status"] = "failed_identity_mismatch"
            topic["failure_reason"] = fatal_identity_error
            topic_results.append(topic)
            break
        if len(topic["records"]) == target_records_per_topic:
            topic["status"] = "complete"
        topic_results.append(topic)

    request_failures = sum(1 for entry in request_log
        if entry.get("http_status") != 200 or entry.get("error_type") or entry.get("identity_error"))
    complete_topics = sum(1 for topic in topic_results if topic["status"] == "complete")
    status = ("failed_identity_mismatch" if fatal_identity_error else
        "source_pool_incomplete" if complete_topics < 48 else
        "awaiting_typed_cohort_validation")
    body = {"schema": "europe-pmc-topic-stratified-source-capture-legacy-diagnostic",
        "endpoint": API_ROOT, "capture_date": capture_date, "captured_at": captured_at,
        "target_records_per_topic": target_records_per_topic,
        "metadata_page_size": metadata_page_size,
        "metadata_request_cap": MAX_TOPIC_METADATA_REQUESTS,
        "core_request_budget_max": MAX_TOPIC_CORE_REQUESTS,
        "core_request_budget_planned": requested_core_budget,
        "core_caps_by_slot_kind": dict(TOPIC_SLOT_CORE_CAPS),
        "candidate_selection_policy": {"version": "legacy-lite-metadata-candidate-helper",
            "production_use": False},
        "topic_slots_requested": len(topic_specs), "topic_slots_completed": complete_topics,
        "status": status, "diagnostic_only": True,
        "scoreable_cohort_created": False,
        "failure_reason": fatal_identity_error,
        "request_attempts": len(request_log), "request_failures": request_failures,
        "request_log": request_log, "topics": topic_results,
        "license_policy": "Every accepted source must have core license=cc by; PLOS source family excluded before core fetch.",
        "historically_exposed_doi_identities": list(HISTORICALLY_EXPOSED_DOIS),
        "historic_exposure_scope": "DOI/URL identity only; no old final abstract, rubric, answer or score fetched or read."}
    body["capture_sha256"] = sha256_obj(body)
    return body




def capture_topic_core_search_pool(*, topic_specs: list[dict], capture_date: str,
        captured_at: str, opener=urlopen) -> dict:
    """Capture bounded core-search pages, applying strict PICO to abstracts inline."""
    source_query(capture_date)
    try:
        timestamp = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("captured_at must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError("captured_at must include a timezone")
    if not isinstance(topic_specs, list) or not topic_specs or len(topic_specs) > MAX_TOPIC_METADATA_REQUESTS:
        raise ValueError("source capture allows 1 to 60 topic searches")

    prepared, seen_ids = [], set()
    for spec in topic_specs:
        query = build_topic_query(spec, capture_date=capture_date)
        if spec["topic_id"] in seen_ids:
            raise ValueError("topic_id values must be unique within one capture")
        seen_ids.add(spec["topic_id"])
        prepared.append((spec, query))
    planned_requests = sum(FROZEN_TOPIC_CORE_PAGE_CAPS[spec["slot_kind"]]
        for spec, _ in prepared)
    if planned_requests > MAX_TOPIC_CORE_REQUESTS - FROZEN_TOPIC_CORE_REQUEST_RESERVE:
        raise ValueError("topic capture core-page budget exceeds the frozen 249-request plan")

    request_log, topics, used_dois, used_pmcids = [], [], set(), set()
    fatal_identity_error = None
    for spec, query in prepared:
        topic_id, slot_kind = spec["topic_id"], spec["slot_kind"]
        cap = FROZEN_TOPIC_CORE_PAGE_CAPS[slot_kind]
        topic = {"topic_id": topic_id, "topic": " ".join(spec["topic"].split()),
            "slot_kind": slot_kind, "query": query, "records": [],
            "core_request_cap": cap, "core_attempts": 0, "core_pages": [],
            "candidate_audit": [], "exclusions": [], "status": "incomplete"}
        cursor, seen_cursors, page_index, record_offset = "*", {"*"}, 0, 0
        while page_index < cap and len(topic["records"]) < 2:
            page_index += 1
            url = API_ROOT + "/search?" + urlencode({"query": query,
                "format": "json", "resulttype": "core",
                "pageSize": FROZEN_TOPIC_CORE_PAGE_SIZE, "cursorMark": cursor})
            log = {"topic_id": topic_id, "phase": "core_search", "page_index": page_index,
                "cursor_mark": cursor, "method": "GET", "url": url}
            topic["core_attempts"] += 1
            try:
                response, raw = _request_json(url, opener=opener)
            except Exception as exc:
                log.update({"http_status": None, "error_type": type(exc).__name__,
                    "error": str(exc)[:500]})
                request_log.append(log)
                topic["failure_reason"] = "core_search_request_failed"
                break
            page_hash = hashlib.sha256(raw).hexdigest()
            log.update({"http_status": 200, "response_bytes": len(raw),
                "response_sha256": page_hash})
            request_log.append(log)
            rows = ((response.get("resultList") or {}).get("result"))
            if not isinstance(rows, list):
                log.update({"error_type": "invalid_core_search_response"})
                topic["failure_reason"] = "invalid_core_search_response"
                break
            if len(rows) > FROZEN_TOPIC_CORE_PAGE_SIZE:
                log.update({"error_type": "core_page_exceeds_frozen_page_size"})
                topic["failure_reason"] = "core_page_exceeds_frozen_page_size"
                break
            next_cursor = response.get("nextCursorMark")
            hit_count = response.get("hitCount")
            topic["core_pages"].append({"page_index": page_index, "cursor_mark": cursor,
                "next_cursor_mark": next_cursor if isinstance(next_cursor, str) else None,
                "response_sha256": page_hash, "records_returned": len(rows),
                "hit_count": hit_count, "record_offset": record_offset})
            for result_index, record in enumerate(rows):
                rank = record_offset + result_index
                if not isinstance(record, dict):
                    topic["exclusions"].append({"reason": "invalid_core_record", "core_rank": rank})
                    continue
                doi, pmcid = _doi(record), str(record.get("pmcid") or "").strip()
                if not doi or not re.fullmatch(r"PMC\d+", pmcid, re.IGNORECASE):
                    topic["exclusions"].append({"doi": doi or None, "pmcid": pmcid or None,
                        "reason": "missing_source_identity", "core_rank": rank})
                    continue
                if any(doi.startswith(prefix) for prefix in HISTORICAL_DOI_PREFIX_EXCLUSIONS):
                    fatal_identity_error = "PLOS DOI family is forbidden in fetched source records"
                    log["identity_error"] = fatal_identity_error
                    break
                if doi in used_dois or pmcid.casefold() in used_pmcids:
                    topic["exclusions"].append({"doi": doi, "pmcid": pmcid,
                        "reason": "duplicate_source_across_topic_groups", "core_rank": rank})
                    continue
                audit = {"doi": doi, "pmcid": pmcid, "core_rank": rank,
                    "page_index": page_index, "result_index": result_index,
                    "core_response_sha256": page_hash}
                topic["candidate_audit"].append(audit)
                if str(record.get("license") or "").strip().casefold() != "cc by":
                    topic["exclusions"].append({"doi": doi, "pmcid": pmcid,
                        "reason": "core_license_not_cc_by", "core_rank": rank,
                        "response_sha256": page_hash})
                    continue
                abstract = record.get("abstractText")
                if not isinstance(abstract, str) or len(abstract.strip()) < 400:
                    topic["exclusions"].append({"doi": doi, "pmcid": pmcid,
                        "reason": "abstract_missing_or_too_short", "core_rank": rank,
                        "response_sha256": page_hash})
                    continue
                relevance = assess_topic_pico_relevance(record, spec)
                audit["abstract_pico_relevance"] = relevance
                if not relevance["eligible"]:
                    topic["exclusions"].append({"doi": doi, "pmcid": pmcid,
                        "reason": "pico_relevance_missing_dimensions", "core_rank": rank,
                        "missing_dimensions": relevance["missing_dimensions"],
                        "matched_terms": relevance["matched_terms"],
                        "response_sha256": page_hash})
                    continue
                topic["records"].append(record)
                used_dois.add(doi)
                used_pmcids.add(pmcid.casefold())
                topic.setdefault("core_record_sha256", []).append({"doi": doi,
                    "pmcid": pmcid, "core_rank": rank, "response_sha256": page_hash})
                if len(topic["records"]) == 2:
                    break
            if fatal_identity_error or len(topic["records"]) == 2:
                break
            record_offset += len(rows)
            if not rows:
                break
            if type(hit_count) is int and hit_count >= 0 and record_offset >= hit_count:
                break
            if (not isinstance(next_cursor, str) or not next_cursor or
                    next_cursor == cursor or next_cursor in seen_cursors):
                topic["failure_reason"] = "core_search_cursor_missing_or_repeated"
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        topic["status"] = "complete" if len(topic["records"]) == 2 else "incomplete"
        if fatal_identity_error:
            topic.update({"status": "failed_identity_mismatch", "failure_reason": fatal_identity_error})
        topics.append(topic)
        if fatal_identity_error:
            break

    failures = sum(1 for row in request_log if row.get("http_status") != 200 or
        row.get("error_type") or row.get("identity_error"))
    complete = sum(1 for row in topics if row["status"] == "complete")
    status = ("failed_identity_mismatch" if fatal_identity_error else
        "source_pool_incomplete" if complete < 48 else "awaiting_typed_cohort_validation")
    result = {"schema": TOPIC_SOURCE_CAPTURE_SCHEMA, "endpoint": API_ROOT,
        "capture_date": capture_date, "captured_at": captured_at,
        "target_records_per_topic": 2, "core_page_size": FROZEN_TOPIC_CORE_PAGE_SIZE,
        "core_request_budget_max": MAX_TOPIC_CORE_REQUESTS,
        "core_request_budget_reserved": FROZEN_TOPIC_CORE_REQUEST_RESERVE,
        "core_request_budget_planned": planned_requests,
        "core_caps_by_slot_kind": dict(FROZEN_TOPIC_CORE_PAGE_CAPS),
        "candidate_selection_policy": TOPIC_SELECTION_POLICY,
        "topic_slots_requested": len(topic_specs), "topic_slots_completed": complete,
        "status": status, "diagnostic_only": True, "scoreable_cohort_created": False,
        "failure_reason": fatal_identity_error, "request_attempts": len(request_log),
        "request_failures": failures, "request_log": request_log, "topics": topics,
        "license_policy": "Every accepted source must have core license=cc by; PLOS DOI families are excluded by the query and checked again.",
        "historically_exposed_doi_identities": list(HISTORICALLY_EXPOSED_DOIS),
        "historic_exposure_scope": "DOI/URL identity only; no old final abstract, rubric, answer or score fetched or read."}
    result["capture_sha256"] = sha256_obj(result)
    return result


def _validate_frozen_topic_specs(topic_specs: list[dict], *, capture_date: str) -> dict:
    if not isinstance(topic_specs, list) or len(topic_specs) != MAX_TOPIC_METADATA_REQUESTS:
        raise ValueError("frozen production plan requires exactly 60 topic slots")
    counts = {slot_kind: 0 for slot_kind in TOPIC_SLOT_CORE_CAPS}
    seen_topic_ids = set()
    for spec in topic_specs:
        build_topic_query(spec, capture_date=capture_date)
        topic_id = spec["topic_id"]
        if topic_id in seen_topic_ids:
            raise ValueError("topic_id values must be unique within the frozen plan")
        seen_topic_ids.add(topic_id)
        counts[spec["slot_kind"]] += 1
    if counts != FROZEN_TOPIC_SLOT_ALLOCATION:
        raise ValueError("frozen production plan requires 39 regular, 9 conflict and 12 replacement slots")
    core_budget = sum(FROZEN_TOPIC_CORE_PAGE_CAPS[kind] * count for kind, count in counts.items())
    if core_budget != MAX_TOPIC_CORE_REQUESTS - FROZEN_TOPIC_CORE_REQUEST_RESERVE:
        raise ValueError("frozen production plan core-page budget must be 249 with 3 reserved")
    return {"slot_counts": counts, "topic_searches": len(topic_specs),
        "core_request_budget": core_budget,
        "core_request_reserve": FROZEN_TOPIC_CORE_REQUEST_RESERVE,
        "core_page_size": FROZEN_TOPIC_CORE_PAGE_SIZE}


def build_frozen_topic_plan_manifest(topic_specs: list[dict], *, capture_date: str) -> dict:
    """Create a content-bound manifest for the reviewed 60-slot production plan."""
    _validate_frozen_topic_specs(topic_specs, capture_date=capture_date)
    # Round-trip through JSON to detach the manifest from mutable caller objects
    # and ensure only portable JSON values are included in the content hash.
    try:
        bound_specs = json.loads(json.dumps(topic_specs, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError("frozen topic specs must contain only JSON values") from exc
    validated = _validate_frozen_topic_specs(bound_specs, capture_date=capture_date)
    payload = {"schema": FROZEN_TOPIC_PLAN_SCHEMA, "capture_date": capture_date,
        "topic_specs": bound_specs, "candidate_selection_policy": TOPIC_SELECTION_POLICY,
        **validated}
    return {**payload, "plan_sha256": sha256_obj(payload)}


def validate_frozen_topic_plan(plan_manifest: dict, *, expected_plan_sha256: str) -> dict:
    """Verify exact manifest schema, content hash, slot allocation and frozen hash."""
    if not isinstance(plan_manifest, dict):
        raise ValueError("frozen plan manifest must be an object")
    expected_keys = {"schema", "capture_date", "topic_specs", "candidate_selection_policy", "slot_counts",
        "topic_searches", "core_request_budget", "core_request_reserve", "core_page_size", "plan_sha256"}
    if set(plan_manifest) != expected_keys or plan_manifest.get("schema") != FROZEN_TOPIC_PLAN_SCHEMA:
        raise ValueError("frozen plan manifest has unsupported schema or fields")
    if plan_manifest.get("candidate_selection_policy") != TOPIC_SELECTION_POLICY:
        raise ValueError("frozen plan candidate-selection policy differs from this reviewed implementation")
    if (not isinstance(expected_plan_sha256, str) or
            not re.fullmatch(r"[0-9a-f]{64}", expected_plan_sha256)):
        raise ValueError("verified plan hash must be a lowercase SHA-256")
    payload = {key: value for key, value in plan_manifest.items() if key != "plan_sha256"}
    actual_hash = sha256_obj(payload)
    if plan_manifest.get("plan_sha256") != actual_hash or expected_plan_sha256 != actual_hash:
        raise ValueError("frozen topic plan does not match verified plan hash")
    derived = _validate_frozen_topic_specs(plan_manifest.get("topic_specs"),
        capture_date=plan_manifest.get("capture_date"))
    if any(plan_manifest.get(key) != value for key, value in derived.items()):
        raise ValueError("frozen topic plan summary does not match exact topic inputs")
    return {**derived, "candidate_selection_policy": TOPIC_SELECTION_POLICY,
        "capture_date": plan_manifest["capture_date"],
        "topic_specs": plan_manifest["topic_specs"], "plan_sha256": actual_hash}


def capture_frozen_topic_stratified_source_pool(*, plan_manifest: dict,
        verified_plan_sha256: str, captured_at: str,
        opener=urlopen) -> dict:
    """Production entrypoint guarded by the reviewed exact slot allocation."""
    verified = validate_frozen_topic_plan(plan_manifest,
        expected_plan_sha256=verified_plan_sha256)
    result = capture_topic_core_search_pool(topic_specs=verified["topic_specs"],
        capture_date=verified["capture_date"], captured_at=captured_at, opener=opener)
    result["frozen_topic_plan_sha256"] = verified["plan_sha256"]
    result["capture_implementation_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    result["capture_sha256"] = sha256_obj({key: value for key, value in result.items()
        if key != "capture_sha256"})
    return result
