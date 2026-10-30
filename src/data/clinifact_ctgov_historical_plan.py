"""Freeze and execute a bounded, metadata-only ClinicalTrials.gov v2 capture.

This module is deliberately separate from cohort construction and model dispatch.
Registry fields can support later human/AI adjudication; they never become labels.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from src.data.plos_capacity_probe import summarize_plos_capacity
from scripts.audit_clinifact_historical_plos_source import summarize_source_gates

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://clinicaltrials.gov/api/v2/studies"
MAX_RECORDS_PER_REQUEST = 100
API_MAX_PAGE_SIZE = 1000
MAX_PHYSICAL_REQUESTS = 5
MAX_RESPONSE_BYTES_PER_REQUEST = 64 * 1024 * 1024
MAX_RESPONSE_BYTES_TOTAL = MAX_PHYSICAL_REQUESTS * MAX_RESPONSE_BYTES_PER_REQUEST
TIMEOUT_SECONDS = 45
MIN_REQUEST_START_INTERVAL_SECONDS = 1.0
EXPECTED_SOURCE_ARTICLES = 611
EXPECTED_REVIEW_ARTICLES = 485
EXPECTED_NCT_IDS = 477
FIELDS = "protocolSection,resultsSection"
PLAN_SCHEMA = "clinifact-ctgov-historical-registry-plan-v1"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _inside_workbuddy(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("G01 CT.gov plans and captures must stay under project .workbuddy") from exc
    return resolved


def _validate_source(audit_path: Path, *, expected_capture: Path | None = None) -> tuple[dict, list[dict], Path]:
    audit_path = audit_path.resolve()
    audit = _load(audit_path)
    if audit.get("schema") != "clinifact-plos-historical-source-audit-v1":
        raise ValueError("unexpected historical source audit schema")
    capture_dir = Path(audit.get("source_capture_dir", "")).resolve()
    if expected_capture is not None and capture_dir != expected_capture.resolve():
        raise ValueError("audit is not bound to the expected PLOS capture directory")
    capture_dir = _inside_workbuddy(capture_dir)
    plan_path, report_path = capture_dir / "plan.json", capture_dir / "report.json"
    index_path = capture_dir / "metadata-index.json"
    source_plan, source_report, rows = _load(plan_path), _load(report_path), _load(index_path)
    source_plan_hash = source_plan.get("plan_sha256")
    unsigned = {key: value for key, value in source_plan.items() if key != "plan_sha256"}
    if source_plan_hash != sha_bytes(canonical(unsigned)):
        raise ValueError("PLOS metadata source plan hash mismatch")
    if (source_report.get("status") != "metadata_capture_complete"
            or source_report.get("plan_sha256") != source_plan_hash
            or source_report.get("report_source_sha256") != source_plan.get("source_code_sha256")
            or source_plan.get("source_code_sha256") != sha_file(ROOT / "scripts/fetch_clinifact_historical_plos_metadata.py")
            or len(rows) != source_plan.get("expected_num_found")
            or len(rows) != source_report.get("article_count")
            or source_report.get("unique_doi_count") != len(rows)):
        raise ValueError("PLOS metadata source capture is incomplete or inconsistent")
    page_hashes: set[str] = set()
    for page_index, page in enumerate(source_report.get("pages") or []):
        raw_page = capture_dir / f"page-{page_index:02d}" / "response.bin"
        if page.get("http_status") != 200 or not raw_page.is_file() or sha_file(raw_page) != page.get("response_sha256"):
            raise ValueError("PLOS source page receipt/response hash mismatch")
        page_hashes.add(page["response_sha256"])
    if not page_hashes or any(row.get("source_page_response_sha256") not in page_hashes for row in rows):
        raise ValueError("PLOS metadata rows are not bound to the frozen raw responses")
    if (audit.get("source_capture_plan_sha256") != source_plan_hash
            or audit.get("source_capture_report_sha256") != sha_file(report_path)
            or audit.get("source_metadata_index_sha256") != sha_file(index_path)
            or audit.get("source_article_count") != len(rows)):
        raise ValueError("historical source audit is not bound to the supplied PLOS capture")
    audit_script = ROOT / "scripts/audit_clinifact_historical_plos_source.py"
    if audit.get("audit_source_sha256") != sha_file(audit_script):
        raise ValueError("historical source audit was produced by different audit code")
    history = audit.get("exact_history_scan") or {}
    if (history.get("complete") is not True or history.get("unreadable_files") != 0
            or history.get("source_metadata_index_sha256") != sha_file(index_path)):
        raise ValueError("historical exact-text scan is incomplete or not bound to the source index")
    expected_gates = summarize_source_gates(rows, history.get("overlap_candidates") or {})
    for key, value in expected_gates.items():
        if audit.get(key) != value:
            raise ValueError(f"historical source gate mismatch: {key}")
    if (len(rows) != EXPECTED_SOURCE_ARTICLES
            or expected_gates["records_remaining_for_retraction_and_trial_evidence_review"] != EXPECTED_REVIEW_ARTICLES
            or expected_gates["unique_nct_count_after_license_type_title_history_gates"] != EXPECTED_NCT_IDS):
        raise ValueError("frozen source capacity changed; rebuild and review the upstream audit")
    if audit.get("labels_generated") is not False or audit.get("confirmation_cohort_created") is not False:
        raise ValueError("source audit unexpectedly contains task labels or a cohort")
    return audit, rows, capture_dir


def build_plan(audit_path: Path) -> dict:
    audit_path = _inside_workbuddy(audit_path)
    audit, rows, capture_dir = _validate_source(audit_path)
    history = audit["exact_history_scan"]
    overlap = history.get("overlap_candidates") or {}
    all_dois = {str(row.get("doi", "")).casefold() for row in rows}
    eligible: set[str] = set()
    for row in rows:
        doi = str(row.get("doi", "")).casefold()
        # Repeat the same transparent source gates before extracting NCT IDs.
        from src.data.plos_clinical_intake import classify_plos_copyright_statement, EXCLUDED_TITLE_TERMS
        reasons = not classify_plos_copyright_statement(row.get("copyright_statement", ""))
        reasons = reasons or str(row.get("article_type") or "").casefold().strip() != "research article"
        title = " ".join(str(row.get("title") or "").casefold().split())
        reasons = reasons or any(term in title for term in EXCLUDED_TITLE_TERMS)
        reasons = reasons or bool(overlap.get(doi))
        if not reasons:
            eligible.add(doi)
    capacity = summarize_plos_capacity(rows, eligible)
    nct_ids = sorted({nct.upper() for ids in capacity.get("nct_by_doi", {}).values() for nct in ids})
    if len(eligible) != EXPECTED_REVIEW_ARTICLES or len(nct_ids) != EXPECTED_NCT_IDS:
        raise ValueError("recomputed eligible DOI/NCT set does not match frozen audit counts")
    if any(len(nct) != 11 or not nct.startswith("NCT") or not nct[3:].isdigit() for nct in nct_ids):
        raise ValueError("invalid NCT identifier in source metadata")
    batches = [nct_ids[start:start + MAX_RECORDS_PER_REQUEST]
               for start in range(0, len(nct_ids), MAX_RECORDS_PER_REQUEST)]
    if not batches or len(batches) > MAX_PHYSICAL_REQUESTS or any(
            not batch or len(batch) > MAX_RECORDS_PER_REQUEST for batch in batches):
        raise ValueError("NCT list cannot be represented within the frozen five-request cap")
    capture_plan = _load(capture_dir / "plan.json")
    capture_report = _load(capture_dir / "report.json")
    index_path = capture_dir / "metadata-index.json"
    audit_path = audit_path.resolve()
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "endpoint": ENDPOINT,
        "method": "GET",
    "params_shared": {"fields": FIELDS, "format": "json", "pageSize": API_MAX_PAGE_SIZE},
        "source": {
            "audit_path_relative": str(audit_path.relative_to(ROOT)).replace("\\", "/"),
            "audit_sha256": sha_file(audit_path),
            "audit_code_sha256": sha_file(ROOT / "scripts/audit_clinifact_historical_plos_source.py"),
            "source_capture_dir_relative": str(capture_dir.relative_to(ROOT)).replace("\\", "/"),
            "source_capture_plan_sha256": capture_plan["plan_sha256"],
            "source_capture_report_sha256": sha_file(capture_dir / "report.json"),
            "source_metadata_index_sha256": sha_file(index_path),
            "source_capture_report_claimed_sha256": capture_report.get("report_source_sha256"),
            "source_articles": len(rows),
            "license_type_title_history_gated_articles": EXPECTED_REVIEW_ARTICLES,
            "unique_nct_ids": len(nct_ids),
        },
        "eligible_article_nct_map": {doi: ids for doi, ids in sorted(capacity["nct_by_doi"].items())},
        "request_batches": [
            {"batch_index": index, "filter.ids": ids, "expected_count": len(ids)}
            for index, ids in enumerate(batches)
        ],
        "max_records_per_request": MAX_RECORDS_PER_REQUEST,
        "max_api_page_size": API_MAX_PAGE_SIZE,
        "max_physical_requests": MAX_PHYSICAL_REQUESTS,
        "max_response_bytes_per_request": MAX_RESPONSE_BYTES_PER_REQUEST,
        "max_response_bytes_total": MAX_RESPONSE_BYTES_TOTAL,
        "timeout_seconds": TIMEOUT_SECONDS,
        "minimum_request_start_interval_seconds": MIN_REQUEST_START_INTERVAL_SECONDS,
        "redirects": 0,
        "retries": 0,
        "model_calls": 0,
        "task_labels_generated": False,
        "confirmation_cohort_created": False,
        "candidate_search_authorized": False,
        "purpose": "Read-only registry metadata for later human/AI adjudication of trial identity and registered primary outcomes; registry/article co-occurrence is not a label or proof of outcome correspondence.",
        "collector_source_sha256": sha_file(Path(__file__)),
    }
    plan["plan_sha256"] = sha_bytes(canonical(plan))
    return plan


def validate_plan(plan: dict, audit_path: Path) -> dict:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unexpected CT.gov historical metadata plan schema")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != sha_bytes(canonical(unsigned)):
        raise ValueError("CT.gov historical metadata plan hash mismatch")
    expected = build_plan(audit_path)
    if plan != expected:
        raise ValueError("CT.gov plan differs from the recomputed frozen source and request contract")
    return expected


def preflight(plan_dir: Path, audit_path: Path) -> dict:
    plan_dir = _inside_workbuddy(plan_dir)
    if not plan_dir.is_dir():
        raise ValueError("plan directory does not exist")
    plan = _load(plan_dir / "plan.json")
    validated = validate_plan(plan, audit_path)
    if any((plan_dir / name).exists() for name in ("receipt.json", "response.bin", "run")):
        raise ValueError("plan directory contains unexpected execution outputs")
    return {
        "schema": "clinifact-ctgov-historical-registry-preflight-v1",
        "status": "passed",
        "plan_sha256": validated["plan_sha256"],
        "request_count": len(validated["request_batches"]),
        "nct_count": validated["source"]["unique_nct_ids"],
        "max_response_bytes_total": validated["max_response_bytes_total"],
        "model_calls": 0,
        "labels_generated": False,
        "confirmation_cohort_created": False,
        "candidate_search_authorized": False,
    }


def _extract_nct_ids(payload: Any) -> list[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("studies"), list):
        raise ValueError("invalid_studies_response")
    ids: list[str] = []
    for study in payload["studies"]:
        ident = study.get("protocolSection", {}).get("identificationModule", {})
        nct = ident.get("nctId")
        if not isinstance(nct, str):
            raise ValueError("study_response_missing_nct_id")
        ids.append(nct.upper())
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_nct_id_in_response")
    return ids


def execute(plan_dir: Path, audit_path: Path, output_dir: Path, *,
            opener_factory: Callable[[], Any] | None = None,
            sleep: Callable[[float], None] = time.sleep,
            clock: Callable[[], float] = time.monotonic) -> dict:
    plan_dir = _inside_workbuddy(plan_dir)
    output_dir = _inside_workbuddy(output_dir)
    static = preflight(plan_dir, audit_path)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse CT.gov capture directory: {output_dir}")
    plan = _load(plan_dir / "plan.json")
    opener = (opener_factory or (lambda: build_opener(_NoRedirect())))()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    receipts: list[dict] = []
    total_bytes = 0
    prior_start: float | None = None
    fatal: str | None = None
    for batch in plan["request_batches"]:
        if prior_start is not None:
            elapsed = clock() - prior_start
            if elapsed < plan["minimum_request_start_interval_seconds"]:
                sleep(plan["minimum_request_start_interval_seconds"] - elapsed)
        directory = output_dir / f"request-{batch['batch_index']:02d}"
        directory.mkdir()
        params = {**plan["params_shared"], "filter.ids": ",".join(batch["filter.ids"])}
        request = Request(plan["endpoint"] + "?" + urlencode(params), headers={
            "Accept": "application/json", "User-Agent": "RSI-G01-Source-Audit/1.0",
        })
        prior_start = clock()
        try:
            with opener.open(request, timeout=plan["timeout_seconds"]) as response:
                status = response.status
                body = response.read(plan["max_response_bytes_per_request"] + 1)
        except HTTPError as exc:
            status, body = exc.code, exc.read(plan["max_response_bytes_per_request"] + 1)
        except (URLError, TimeoutError, OSError) as exc:
            status, body = 0, type(exc).__name__.encode("ascii")
        body_path = directory / "response.bin"
        body_path.write_bytes(body)
        total_bytes += len(body)
        error: str | None = None
        returned_ids: list[str] = []
        pagination_token_present = False
        if len(body) > plan["max_response_bytes_per_request"]:
            error = "response_byte_limit_exceeded"
        elif status != 200:
            error = f"http_status_{status}"
        else:
            try:
                payload = json.loads(body)
                returned_ids = _extract_nct_ids(payload)
                pagination_token_present = bool(payload.get("nextPageToken"))
                expected = set(batch["filter.ids"])
                if not expected.issubset(set(returned_ids)):
                    error = "requested_nct_id_not_returned"
            except (json.JSONDecodeError, ValueError) as exc:
                error = f"invalid_response:{str(exc)}"
        receipt = {
            "batch_index": batch["batch_index"],
            "request_nct_count": batch["expected_count"],
            "returned_nct_count": len(returned_ids),
            "returned_nct_ids": sorted(returned_ids),
            "additional_alias_match_nct_ids": sorted(set(returned_ids) - set(batch["filter.ids"])),
            "pagination_token_present": pagination_token_present,
            "http_status": status,
            "response_bytes": len(body),
            "response_sha256": sha_file(body_path),
            "error": error,
        }
        (directory / "receipt.json").write_bytes(canonical(receipt))
        receipts.append(receipt)
        if error:
            fatal = fatal or f"batch_{batch['batch_index']}:{error}"
        elif total_bytes > plan["max_response_bytes_total"]:
            fatal = "total_response_byte_limit_exceeded"
            break
    expected_all = {nct for batch in plan["request_batches"] for nct in batch["filter.ids"]}
    returned_all = {nct for receipt in receipts for nct in receipt["returned_nct_ids"]}
    complete = fatal is None and len(receipts) == static["request_count"] and expected_all.issubset(returned_all)
    if not complete and fatal is None:
        fatal = "incomplete_capture"
    report = {
        "schema": "clinifact-ctgov-historical-registry-capture-v1",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": plan["plan_sha256"],
        "status": "complete" if complete else "incomplete",
        "requests_attempted": len(receipts),
        "expected_requests": static["request_count"],
        "expected_nct_count": len(expected_all),
        "returned_unique_nct_count": len(returned_all),
        "missing_nct_ids": sorted(expected_all - returned_all),
        "additional_alias_match_nct_ids": sorted(returned_all - expected_all),
        "requested_id_coverage_complete": expected_all.issubset(returned_all),
        "response_bytes_total": total_bytes,
        "request_receipts": receipts,
        "model_calls": 0,
        "task_labels_generated": False,
        "confirmation_cohort_created": False,
        "candidate_search_authorized": False,
        "error": fatal,
    }
    (output_dir / "report.json").write_bytes(canonical(report))
    return report
