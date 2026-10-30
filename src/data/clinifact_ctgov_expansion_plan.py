"""Freeze a bounded, source-only CT.gov expansion plan from captured official schemas.

No network or model calls occur here. The plan continues a previously captured
official hasResults/interventional query and screens fresh NCT/RESULT-PMID
identities before any outcome-detail or PubMed fetch is considered.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import yaml


ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://clinicaltrials.gov/api/v2/studies"
OAS_URL = "https://clinicaltrials.gov/api/oas/v2"
SCHEMA = "clinifact-ctgov-expansion-plan-v1"
MAX_PAGE_SIZE = 100
MAX_STUDY_PAGES = 15
MAX_STUDIES = MAX_PAGE_SIZE * MAX_STUDY_PAGES
MAX_RESULT_RECORDS = 1500
MAX_RESULT_BATCH_SIZE = 100
MAX_RESULT_REQUESTS = MAX_RESULT_RECORDS // MAX_RESULT_BATCH_SIZE
MAX_RESULT_EDGES = 3000
MAX_UNIQUE_PMIDS = 2000
MAX_PMID_BATCH_SIZE = 100
MAX_PMID_REQUESTS = MAX_UNIQUE_PMIDS // MAX_PMID_BATCH_SIZE
MAX_RESPONSE_BYTES_PER_REQUEST = 16 * 1024 * 1024
MAX_DISCOVERY_BYTES = MAX_STUDY_PAGES * MAX_RESPONSE_BYTES_PER_REQUEST
MAX_RESULT_BYTES = MAX_RESULT_REQUESTS * MAX_RESPONSE_BYTES_PER_REQUEST
MAX_PUBMED_BYTES = MAX_PMID_REQUESTS * MAX_RESPONSE_BYTES_PER_REQUEST
TIMEOUT_SECONDS = 45
TARGET_INDEPENDENT_COMPONENTS = 144
REQUIRED_OAS_PARAMETERS = {"filter.advanced", "fields", "format", "pageSize", "pageToken", "countTotal"}
EXPECTED_FILTER = "AREA[HasResults]true AND AREA[StudyType]INTERVENTIONAL"
EXPECTED_DISCOVERY_FIELDS = "NCTId,HasResults,StudyType,PrimaryOutcome,ReferencesModule"
DETAIL_FIELDS = "protocolSection,resultsSection"
NCT_RE = re.compile(r"^NCT\d{8}$")
NCT_HISTORY_RE = re.compile(r"^NCT0*(\d{1,8})$", re.IGNORECASE)
PMID_RE = re.compile(r"^\d+$")


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _under_workbuddy(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to((root / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("CT.gov expansion artifacts must stay under project .workbuddy") from exc
    return resolved


def _schema_ref(document: Mapping[str, Any], ref: str) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError("OAS response schema must be a local reference")
    current: Any = document
    for part in ref[2:].split("/"):
        current = current[part]
    if not isinstance(current, Mapping):
        raise ValueError("OAS schema reference does not point to an object")
    return current


def validate_oas_contract(document: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the captured official OAS exposes the exact query/pagination route."""
    if document.get("openapi") != "3.0.3" or (document.get("info") or {}).get("title") != "ClinicalTrials.gov REST API":
        raise ValueError("unexpected ClinicalTrials.gov OAS identity")
    servers = {str(item.get("url", "")).rstrip("/") for item in document.get("servers", [])}
    if "https://clinicaltrials.gov/api/v2" not in servers:
        raise ValueError("official ClinicalTrials.gov API v2 server is missing")
    operation = ((document.get("paths") or {}).get("/studies") or {}).get("get")
    if not isinstance(operation, Mapping):
        raise ValueError("OAS GET /studies operation is missing")
    parameters = {str(p.get("name")): p for p in operation.get("parameters", [])}
    missing = REQUIRED_OAS_PARAMETERS - set(parameters)
    if missing:
        raise ValueError(f"OAS query/pagination parameters are missing: {', '.join(sorted(missing))}")
    expected_types = {"filter.advanced": "string", "fields": "array", "format": "string",
                      "pageToken": "string", "countTotal": "boolean"}
    for name, expected_type in expected_types.items():
        if (parameters[name].get("schema") or {}).get("type") != expected_type:
            raise ValueError(f"OAS {name} parameter type is incompatible with the selector")
    page_size = parameters["pageSize"].get("schema") or {}
    if page_size.get("type") != "integer" or int(page_size.get("minimum", 0)) > MAX_PAGE_SIZE:
        raise ValueError("OAS pageSize schema is incompatible with the bounded selector")
    page_description = str(parameters["pageSize"].get("description") or "")
    if "1,000" not in page_description and "1000" not in page_description:
        raise ValueError("OAS does not confirm the documented 1,000-record page cap")
    fmt = parameters["format"].get("schema") or {}
    if fmt.get("type") != "string" or "json" not in fmt.get("enum", []):
        raise ValueError("OAS format parameter does not support JSON")
    response = ((operation.get("responses") or {}).get("200") or {}).get("content", {}).get("application/json", {})
    response_schema_ref = (response.get("schema") or {}).get("$ref")
    if not response_schema_ref:
        raise ValueError("OAS 200 JSON response schema is missing")
    paged = _schema_ref(document, str(response_schema_ref))
    properties = paged.get("properties") or {}
    if not isinstance(properties.get("studies"), Mapping) or not isinstance(properties.get("nextPageToken"), Mapping):
        raise ValueError("OAS paged response does not define studies and nextPageToken")
    studies_schema = properties["studies"]
    if studies_schema.get("$ref"):
        studies_schema = _schema_ref(document, str(studies_schema["$ref"]))
    if studies_schema.get("type") != "array" or properties["nextPageToken"].get("type") != "string":
        raise ValueError("OAS pagination response fields have incompatible types")
    return {
        "oas_version": str((document.get("info") or {}).get("version") or ""),
        "operation_id": operation.get("operationId"),
        "endpoint": ENDPOINT,
        "validated_parameters": sorted(REQUIRED_OAS_PARAMETERS),
        "response_schema_ref": response_schema_ref,
        "max_page_size_documented": 1000,
    }


def _read_oas(oas_path: Path, receipt_path: Path) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    raw = oas_path.read_bytes()
    receipt = _load_json(receipt_path)
    if (receipt.get("status") != 200 or receipt.get("truncated") is not False
            or receipt.get("response_sha256") != sha_bytes(raw)
            or receipt.get("response_bytes_stored") != len(raw)):
        raise ValueError("official OAS receipt does not match a complete raw schema response")
    if receipt.get("url") != OAS_URL:
        raise ValueError("OAS receipt URL is not the official ClinicalTrials.gov schema endpoint")
    document = yaml.safe_load(raw)
    if not isinstance(document, dict):
        raise ValueError("official OAS is not a YAML object")
    validated = validate_oas_contract(document)
    return document, raw, {**validated, "sha256": sha_bytes(raw)}


def _validate_history(exposure: Mapping[str, Any]) -> tuple[set[str], set[str], list[str]]:
    raw_ncts = [str(value).upper() for value in exposure.get("nct_ids", [])]
    pmids = [str(value) for value in exposure.get("pmids", [])]
    if (not {".workbuddy", "artifacts"}.issubset(set(exposure.get("scan_roots", [])))
            or exposure.get("errors") != []
            or exposure.get("nct_count") != len(raw_ncts)
            or exposure.get("pmid_count") != len(pmids)
            or len(raw_ncts) != len(set(raw_ncts)) or len(pmids) != len(set(pmids))
            or any(not PMID_RE.fullmatch(value) for value in pmids)):
        raise ValueError("historical exposure inventory is incomplete or malformed")
    normalized_ncts: set[str] = set()
    invalid_ncts: list[str] = []
    for value in raw_ncts:
        match = NCT_HISTORY_RE.fullmatch(value)
        if match:
            normalized_ncts.add("NCT" + match.group(1).zfill(8))
        else:
            invalid_ncts.append(value)
    return normalized_ncts, set(pmids), sorted(invalid_ncts)


def _study_nct(study: Mapping[str, Any]) -> str:
    value = (((study.get("protocolSection") or {}).get("identificationModule") or {}).get("nctId"))
    if not isinstance(value, str) or not NCT_RE.fullmatch(value.upper()):
        raise ValueError("captured study has an invalid NCTId")
    return value.upper()


def _result_pmids(study: Mapping[str, Any]) -> list[str]:
    refs = (((study.get("protocolSection") or {}).get("referencesModule") or {}).get("references") or [])
    return sorted({str(row.get("pmid")) for row in refs
                   if row.get("type") == "RESULT" and isinstance(row.get("pmid"), (str, int))
                   and PMID_RE.fullmatch(str(row.get("pmid")))})


def select_fresh_candidates(studies: Sequence[Mapping[str, Any]], *, page_number: int,
                            response_sha256: str, excluded_nct_ids: set[str],
                            excluded_pmids: set[str], already_selected_nct_ids: set[str],
                            already_selected_pmids: set[str],
                            already_selected_result_edges: int = 0,
                            max_candidates: int = MAX_STUDIES) -> dict[str, Any]:
    """Select metadata-only NCT/PMID candidates; never evaluates outcomes or labels."""
    if type(page_number) is not int or not 1 <= page_number <= MAX_STUDY_PAGES:
        raise ValueError("page number is outside the frozen discovery cap")
    if not re.fullmatch(r"[0-9a-f]{64}", str(response_sha256)):
        raise ValueError("response SHA-256 binding is invalid")
    if (len(studies) > MAX_PAGE_SIZE or type(max_candidates) is not int
            or not 1 <= max_candidates <= MAX_STUDIES
            or type(already_selected_result_edges) is not int
            or already_selected_result_edges < 0
            or already_selected_result_edges > MAX_RESULT_EDGES):
        raise ValueError("study page or candidate count exceeds its frozen cap")
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    ncts_seen: set[str] = set()
    pmids_seen: set[str] = set()
    result_edges = already_selected_result_edges
    for row_index, study in enumerate(studies):
        nct = _study_nct(study)
        if nct in ncts_seen or nct in already_selected_nct_ids:
            raise ValueError(f"duplicate NCT identity across expansion pages: {nct}")
        ncts_seen.add(nct)
        base = {"page_number": page_number, "row_index": row_index, "nct_id": nct,
                "source_response_sha256": response_sha256,
                "study_sha256": sha_bytes(canonical(study))}
        if nct in excluded_nct_ids:
            exclusions.append({**base, "reason": "historical_nct_exposure"})
            continue
        if study.get("hasResults") is not True:
            exclusions.append({**base, "reason": "has_results_false"})
            continue
        design = (study.get("protocolSection") or {}).get("designModule") or {}
        if design.get("studyType") != "INTERVENTIONAL":
            exclusions.append({**base, "reason": "not_interventional"})
            continue
        outcomes = ((study.get("protocolSection") or {}).get("outcomesModule") or {}).get("primaryOutcomes") or []
        if len(outcomes) != 1:
            exclusions.append({**base, "reason": "primary_outcome_not_unique"})
            continue
        fresh_pmids = [pmid for pmid in _result_pmids(study) if pmid not in excluded_pmids]
        if not fresh_pmids:
            exclusions.append({**base, "reason": "no_fresh_result_pmid"})
            continue
        if len(candidates) >= max_candidates:
            exclusions.append({**base, "reason": "candidate_cap_reached"})
            continue
        result_edges += len(fresh_pmids)
        if result_edges > MAX_RESULT_EDGES:
            raise ValueError("fresh RESULT reference edge cap exceeded")
        if len(pmids_seen | already_selected_pmids | set(fresh_pmids)) > MAX_UNIQUE_PMIDS:
            raise ValueError("fresh unique PMID cap exceeded")
        pmids_seen.update(fresh_pmids)
        candidates.append({**base, "result_pmids": fresh_pmids,
                           "candidate_status": "metadata_screen_only",
                           "requires_results_section_and_pubmed_identity_review": True})
    return {
        "page_number": page_number,
        "source_response_sha256": response_sha256,
        "candidates": candidates,
        "exclusions": exclusions,
        "selected_nct_ids": [row["nct_id"] for row in candidates],
        "selected_result_pmids": sorted(pmids_seen),
        "candidate_search_authorized": False,
        "labels_generated": False,
    }


def build_expansion_plan(*, oas_path: Path, oas_receipt_path: Path,
                         parent_page_path: Path, parent_receipt_path: Path,
                         exposure_path: Path,
                         prior_census_pages: Sequence[tuple[Path, Path]] = (),
                         root: Path = ROOT) -> dict[str, Any]:
    """Build a no-dispatch plan continuing the validated historical census query."""
    root = root.resolve()
    input_paths = [oas_path, oas_receipt_path, parent_page_path, parent_receipt_path, exposure_path]
    inputs = [_under_workbuddy(Path(path), root) for path in input_paths]
    oas_file, oas_receipt_file, page_file, page_receipt_file, exposure_file = inputs
    _document, _oas_raw, oas_info = _read_oas(oas_file, oas_receipt_file)
    parent_receipt = _load_json(page_receipt_file)
    page_raw = page_file.read_bytes()
    page = json.loads(page_raw)
    params = {str(k): str(v) for k, v in (parent_receipt.get("parameters") or {}).items()}
    if (parent_receipt.get("status") != 200 or parent_receipt.get("truncated") is not False
            or parent_receipt.get("response_sha256") != sha_bytes(page_raw)
            or parent_receipt.get("response_file") != page_file.name):
        raise ValueError("parent census page receipt does not match its complete raw response")
    if (params.get("filter.advanced") != EXPECTED_FILTER
            or params.get("fields") != EXPECTED_DISCOVERY_FIELDS
            or params.get("format") != "json" or params.get("pageSize") != "1000"):
        raise ValueError("parent page does not use the frozen official hasResults/interventional query")
    token = page.get("nextPageToken")
    if not isinstance(token, str) or not token:
        raise ValueError("parent census has no continuation token for a new bounded page range")
    studies = page.get("studies")
    if not isinstance(studies, list) or len(studies) != 1000:
        raise ValueError("parent page must contain the complete frozen 1,000-record page")
    prior_ncts = {_study_nct(study) for study in studies}
    prior_pmids = {pmid for study in studies for pmid in _result_pmids(study)}
    exposure = _load_json(exposure_file)
    historic_ncts, historic_pmids, invalid_history_ncts = _validate_history(exposure)
    seen_page_hashes: list[dict[str, str]] = []
    all_seen_ncts, all_seen_pmids = set(prior_ncts), set(prior_pmids)
    seen_pairs = [(_under_workbuddy(Path(p), root), _under_workbuddy(Path(r), root))
                  for p, r in prior_census_pages]
    seen_pairs.append((page_file, page_receipt_file))
    page_numbers: list[int] = []
    for seen_path, seen_receipt_path in seen_pairs:
        raw_seen = seen_path.read_bytes()
        receipt_seen = _load_json(seen_receipt_path)
        seen_payload = json.loads(raw_seen)
        seen_params = {str(k): str(v) for k, v in (receipt_seen.get("parameters") or {}).items()}
        comparable_params = {k: v for k, v in seen_params.items() if k != "pageToken"}
        if (receipt_seen.get("status") != 200 or receipt_seen.get("truncated") is not False
                or receipt_seen.get("response_sha256") != sha_bytes(raw_seen)
                or receipt_seen.get("response_file") != seen_path.name
                or comparable_params != {k: v for k, v in params.items() if k != "pageToken"}):
            raise ValueError("a previously captured census page is incomplete or query-incompatible")
        page_number = receipt_seen.get("page_number")
        if type(page_number) is not int or page_number < 1:
            raise ValueError("previous census page is missing its page number")
        page_numbers.append(page_number)
        seen_studies = seen_payload.get("studies")
        if not isinstance(seen_studies, list) or len(seen_studies) != 1000:
            raise ValueError("previous census page is not a complete 1,000-study page")
        current_ids = [_study_nct(study) for study in seen_studies]
        if len(current_ids) != len(set(current_ids)):
            raise ValueError("previous census page contains duplicate NCT identities")
        all_seen_ncts.update(current_ids)
        all_seen_pmids.update(pmid for study in seen_studies for pmid in _result_pmids(study))
        seen_page_hashes.append({"page_sha256": sha_bytes(raw_seen),
                                 "receipt_sha256": sha_file(seen_receipt_path)})
    if page_numbers != sorted(set(page_numbers)):
        raise ValueError("previous census pages must be unique and ordered")
    if page_numbers[-1] != parent_receipt.get("page_number"):
        raise ValueError("continuation cursor must come from the final prior census page")
    historic_ncts |= all_seen_ncts
    historic_pmids |= all_seen_pmids
    query_params = {k: v for k, v in params.items() if k not in {"pageSize", "pageToken"}}
    query_params["pageSize"] = str(MAX_PAGE_SIZE)
    query_params["pageToken"] = token
    input_bindings = {
        "oas_sha256": oas_info["sha256"],
        "oas_receipt_sha256": sha_file(oas_receipt_file),
        "parent_page_sha256": sha_bytes(page_raw),
        "parent_receipt_sha256": sha_file(page_receipt_file),
        "exposure_inventory_sha256": sha_file(exposure_file),
        "seen_census_pages": seen_page_hashes,
        "expansion_module_sha256": sha_file(Path(__file__)),
        "source_adapter_sha256": sha_file(Path(__file__).with_name("clinifact_ctgov_pubmed_adapter_v1.py")),
    }
    plan: dict[str, Any] = {
        "schema": SCHEMA,
        "endpoint": ENDPOINT,
        "method": "GET",
        "oas": oas_info,
        "input_bindings": input_bindings,
        "query_selector": {
            "kind": "official_registry_pagination_continuation",
            "filter.advanced": EXPECTED_FILTER,
            "fields": EXPECTED_DISCOVERY_FIELDS,
            "continuation_parent_page_sha256": input_bindings["parent_page_sha256"],
            "start_page_token": token,
            "no_requery_or_restart": True,
        },
        "first_request_params": query_params,
        "page_policy": {
            "max_pages": MAX_STUDY_PAGES,
            "page_size": MAX_PAGE_SIZE,
            "max_studies": MAX_STUDIES,
            "follow_only_returned_nextPageToken": True,
            "duplicate_nct_across_pages": "terminal_failure",
        },
        "historical_identity_exclusions": {
            "scan_roots": sorted(set(exposure.get("scan_roots", []))),
            "nct_ids": sorted(historic_ncts),
            "pmids": sorted(historic_pmids),
            "nct_count": len(historic_ncts),
            "pmid_count": len(historic_pmids),
            "raw_nct_token_count": len(exposure.get("nct_ids", [])),
            "noncanonical_nct_tokens": invalid_history_ncts,
            "parent_page_nct_count": len(prior_ncts),
            "parent_page_result_pmid_count": len(prior_pmids),
            "all_parent_census_page_count": len(seen_pairs),
        },
        "downstream_caps": {
            "max_result_records": MAX_RESULT_RECORDS,
            "max_result_ids_per_request": MAX_RESULT_BATCH_SIZE,
            "max_result_requests": MAX_RESULT_REQUESTS,
            "max_result_reference_edges": MAX_RESULT_EDGES,
            "max_unique_pmids": MAX_UNIQUE_PMIDS,
            "max_pmids_per_request": MAX_PMID_BATCH_SIZE,
            "max_pubmed_requests": MAX_PMID_REQUESTS,
            "target_post_dedup_independent_components": TARGET_INDEPENDENT_COMPONENTS,
            "capacity_pass_threshold": TARGET_INDEPENDENT_COMPONENTS,
            "below_threshold_disposition": "stop_and_report_insufficient_no_gate_relaxation",
        },
        "response_limits": {
            "max_bytes_per_request": MAX_RESPONSE_BYTES_PER_REQUEST,
            "max_discovery_bytes_total": MAX_DISCOVERY_BYTES,
            "max_result_bytes_total": MAX_RESULT_BYTES,
            "max_pubmed_bytes_total": MAX_PUBMED_BYTES,
            "timeout_seconds": TIMEOUT_SECONDS,
            "redirects": 0,
            "retries": 0,
        },
        "write_ahead": {
            "create_exclusive_plan_directory": True,
            "fsync_plan_and_receipt_before_request": True,
            "persist_raw_response_before_parse": True,
            "reserve_attempt_receipt_before_http": True,
            "existing_directory_or_attempt_is_terminal": True,
        },
        "model_calls": 0,
        "labels_generated": False,
        "confirmation_cohort_created": False,
        "candidate_search_authorized": False,
        "dispatch_authorized": False,
        "live_capture_performed": False,
    }
    plan["plan_sha256"] = sha_bytes(canonical(plan))
    return plan


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != SCHEMA:
        raise ValueError("unexpected CT.gov expansion plan schema")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != sha_bytes(canonical(unsigned)):
        raise ValueError("CT.gov expansion plan hash mismatch")
    if plan.get("dispatch_authorized") is not False or plan.get("model_calls") != 0:
        raise ValueError("expansion plan must remain no-dispatch/no-model")
    if plan.get("page_policy", {}).get("max_studies") != MAX_STUDIES:
        raise ValueError("expansion plan study cap mismatch")
    return dict(plan)


def reload_preflight(plan_dir: Path, *, oas_path: Path, oas_receipt_path: Path,
                     parent_page_path: Path, parent_receipt_path: Path,
                     exposure_path: Path, prior_census_pages: Sequence[tuple[Path, Path]] = (),
                     root: Path = ROOT) -> dict[str, Any]:
    """Recompute source bindings and verify the no-dispatch write-ahead artifacts."""
    root = root.resolve()
    plan_dir = _under_workbuddy(plan_dir, root)
    expected_names = {"plan.json", "write-ahead.json", "preflight.json"}
    if not plan_dir.is_dir() or {item.name for item in plan_dir.iterdir()} != expected_names:
        raise ValueError("plan directory is incomplete or contains execution artifacts")
    plan = validate_plan(_load_json(plan_dir / "plan.json"))
    recomputed = build_expansion_plan(
        oas_path=oas_path, oas_receipt_path=oas_receipt_path,
        parent_page_path=parent_page_path, parent_receipt_path=parent_receipt_path,
        exposure_path=exposure_path, prior_census_pages=prior_census_pages, root=root)
    if plan != recomputed:
        raise ValueError("persisted plan differs from recomputed OAS/census/history inputs")
    wal = _load_json(plan_dir / "write-ahead.json")
    wal_unsigned = {key: value for key, value in wal.items() if key != "write_ahead_sha256"}
    if (wal.get("plan_sha256") != plan["plan_sha256"]
            or wal.get("next_request_state") != "not_dispatched"
            or wal.get("dispatch_authorized") is not False
            or wal.get("write_ahead_sha256") != sha_bytes(canonical(wal_unsigned))):
        raise ValueError("write-ahead ledger is corrupt or indicates a prior dispatch")
    preflight = _load_json(plan_dir / "preflight.json")
    preflight_unsigned = {key: value for key, value in preflight.items() if key != "preflight_sha256"}
    if (preflight.get("plan_sha256") != plan["plan_sha256"]
            or preflight.get("status") != "passed_zero_call_plan_integrity"
            or preflight.get("dispatch_authorized") is not False
            or preflight.get("network_calls") != 0 or preflight.get("model_calls") != 0
            or preflight.get("preflight_sha256") != sha_bytes(canonical(preflight_unsigned))):
        raise ValueError("zero-call preflight does not match the frozen no-dispatch plan")
    return {"status": "passed_zero_call_plan_integrity", "plan_sha256": plan["plan_sha256"],
            "oas_sha256": plan["oas"]["sha256"], "dispatch_authorized": False,
            "network_calls": 0, "model_calls": 0}


def persist_write_ahead(plan: Mapping[str, Any], plan_dir: Path, *, root: Path = ROOT) -> dict[str, Any]:
    """Create an exclusive immutable plan/writer contract; no network execution."""
    plan = validate_plan(plan)
    plan_dir = _under_workbuddy(plan_dir, root)
    plan_dir.mkdir(parents=True, exist_ok=False)
    payload = canonical(plan)
    plan_path = plan_dir / "plan.json"
    with plan_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    write_ahead = {
        "schema": "clinifact-ctgov-expansion-write-ahead-v1",
        "plan_sha256": plan["plan_sha256"],
        "planned_requests": {
            "discovery_pages_max": MAX_STUDY_PAGES,
            "detail_batches_max": MAX_RESULT_REQUESTS,
            "pubmed_batches_max": MAX_PMID_REQUESTS,
        },
        "next_request_state": "not_dispatched",
        "model_calls": 0,
        "dispatch_authorized": False,
    }
    write_ahead["write_ahead_sha256"] = sha_bytes(canonical(write_ahead))
    wal_path = plan_dir / "write-ahead.json"
    with wal_path.open("xb") as handle:
        handle.write(canonical(write_ahead))
        handle.flush()
        os.fsync(handle.fileno())
    preflight = {
        "schema": "clinifact-ctgov-expansion-preflight-v1",
        "status": "passed_zero_call_plan_integrity",
        "plan_sha256": plan["plan_sha256"],
        "oas_sha256": plan["oas"]["sha256"],
        "history_nct_count": plan["historical_identity_exclusions"]["nct_count"],
        "history_pmid_count": plan["historical_identity_exclusions"]["pmid_count"],
        "max_discovery_studies": MAX_STUDIES,
        "max_result_records": MAX_RESULT_RECORDS,
        "max_unique_pmids": MAX_UNIQUE_PMIDS,
        "dispatch_authorized": False,
        "model_calls": 0,
        "network_calls": 0,
    }
    preflight["preflight_sha256"] = sha_bytes(canonical(preflight))
    preflight_path = plan_dir / "preflight.json"
    with preflight_path.open("xb") as handle:
        handle.write(canonical(preflight))
        handle.flush()
        os.fsync(handle.fileno())
    return {"plan_path": str(plan_path), "plan_sha256": plan["plan_sha256"],
            "write_ahead_path": str(wal_path), "write_ahead_sha256": write_ahead["write_ahead_sha256"],
            "preflight_path": str(preflight_path), "preflight_sha256": preflight["preflight_sha256"],
            "dispatch_authorized": False, "next_request_state": "not_dispatched"}
