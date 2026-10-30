"""Parent-bound, exact-ID CT.gov lookup for v3's two unresolved NCT records."""
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

from src.data.clinifact_ctgov_historical_plan import canonical, sha_bytes, sha_file

ROOT = Path(__file__).resolve().parents[2]


def _inside_workbuddy(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("G01 CT.gov direct lookup plans and captures must stay under project .workbuddy") from exc
    return resolved

MAX_PHYSICAL_REQUESTS = 2
MAX_RESPONSE_BYTES_PER_REQUEST = 64 * 1024 * 1024
MAX_RESPONSE_BYTES_TOTAL = MAX_PHYSICAL_REQUESTS * MAX_RESPONSE_BYTES_PER_REQUEST
TIMEOUT_SECONDS = 45
MIN_REQUEST_START_INTERVAL_SECONDS = 1.0
FIELDS = "protocolSection,resultsSection"
EXPECTED_MISSING_IDS = ("NCT01169766", "NCT03253888")
PLAN_SCHEMA = "clinifact-ctgov-unresolved-direct-lookup-plan-v1"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_parent(plan_path: Path, report_path: Path) -> tuple[dict, dict]:
    plan_path = _inside_workbuddy(plan_path)
    report_path = _inside_workbuddy(report_path)
    plan, report = _load(plan_path), _load(report_path)
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != sha_bytes(canonical(unsigned)):
        raise ValueError("parent v3 plan hash mismatch")
    if (report.get("schema") != "clinifact-ctgov-historical-registry-capture-v1"
            or report.get("plan_sha256") != plan.get("plan_sha256")
            or report.get("status") != "incomplete"
            or report.get("requests_attempted") != MAX_REQUESTS_PARENT
            or report.get("expected_nct_count") != 477
            or report.get("missing_nct_ids") != list(EXPECTED_MISSING_IDS)
            or report.get("task_labels_generated") is not False
            or report.get("confirmation_cohort_created") is not False
            or report.get("model_calls") != 0):
        raise ValueError("parent v3 report is not the expected two-ID unresolved capture")
    if (len(plan.get("request_batches") or []) != MAX_REQUESTS_PARENT
            or len(report.get("request_receipts") or []) != MAX_REQUESTS_PARENT):
        raise ValueError("parent v3 request count does not match its receipt")
    capture_dir = report_path.parent
    for index, receipt in enumerate(report["request_receipts"]):
        body_path = capture_dir / f"request-{index:02d}" / "response.bin"
        receipt_path = capture_dir / f"request-{index:02d}" / "receipt.json"
        if (not body_path.is_file() or not receipt_path.is_file()
                or sha_file(body_path) != receipt.get("response_sha256")
                or _load(receipt_path) != receipt):
            raise ValueError("parent v3 raw response/receipt hash mismatch")
    return plan, report


MAX_REQUESTS_PARENT = 5


def build_plan(parent_plan_path: Path, parent_report_path: Path) -> dict:
    parent_plan, parent_report = _verify_parent(parent_plan_path, parent_report_path)
    requests = [{"nct_id": nct, "path": f"/api/v2/studies/{nct}", "method": "GET"}
                for nct in EXPECTED_MISSING_IDS]
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "endpoint_root": "https://clinicaltrials.gov",
        "params_shared": {"fields": FIELDS, "format": "json"},
        "parent": {
            "plan_path_relative": str(_inside_workbuddy(parent_plan_path).relative_to(ROOT)).replace("\\", "/"),
            "plan_sha256": parent_plan["plan_sha256"],
            "plan_file_sha256": sha_file(_inside_workbuddy(parent_plan_path)),
            "report_path_relative": str(_inside_workbuddy(parent_report_path).relative_to(ROOT)).replace("\\", "/"),
            "report_sha256": sha_file(_inside_workbuddy(parent_report_path)),
            "missing_nct_ids": parent_report["missing_nct_ids"],
        },
        "requests": requests,
        "max_physical_requests": MAX_PHYSICAL_REQUESTS,
        "max_response_bytes_per_request": MAX_RESPONSE_BYTES_PER_REQUEST,
        "max_response_bytes_total": MAX_RESPONSE_BYTES_TOTAL,
        "timeout_seconds": TIMEOUT_SECONDS,
        "minimum_request_start_interval_seconds": MIN_REQUEST_START_INTERVAL_SECONDS,
        "redirects": 0,
        "retries": 0,
        "pagination": False,
        "model_calls": 0,
        "task_labels_generated": False,
        "confirmation_cohort_created": False,
        "candidate_search_authorized": False,
        "purpose": "Resolve only the two parent-bound missing NCT identifiers via exact official study paths; capture registry metadata only.",
        "collector_source_sha256": sha_file(Path(__file__)),
    }
    plan["plan_sha256"] = sha_bytes(canonical(plan))
    return plan


def validate_plan(plan: dict, parent_plan_path: Path, parent_report_path: Path) -> dict:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("schema") != PLAN_SCHEMA or plan.get("plan_sha256") != sha_bytes(canonical(unsigned)):
        raise ValueError("direct lookup plan schema/hash mismatch")
    expected = build_plan(parent_plan_path, parent_report_path)
    if plan != expected:
        raise ValueError("direct lookup plan differs from the frozen parent report")
    return expected


def preflight(plan_dir: Path, parent_plan_path: Path, parent_report_path: Path) -> dict:
    plan_dir = _inside_workbuddy(plan_dir)
    if not plan_dir.is_dir():
        raise ValueError("direct lookup plan directory does not exist")
    plan = validate_plan(_load(plan_dir / "plan.json"), parent_plan_path, parent_report_path)
    if any((plan_dir / name).exists() for name in ("run", "response.bin", "receipt.json")):
        raise ValueError("plan directory contains unexpected execution outputs")
    return {
        "schema": "clinifact-ctgov-unresolved-direct-lookup-preflight-v1",
        "status": "passed",
        "plan_sha256": plan["plan_sha256"],
        "request_count": len(plan["requests"]),
        "nct_ids": [request["nct_id"] for request in plan["requests"]],
        "model_calls": 0,
        "task_labels_generated": False,
        "confirmation_cohort_created": False,
        "candidate_search_authorized": False,
    }


def execute(plan_dir: Path, parent_plan_path: Path, parent_report_path: Path, output_dir: Path, *,
            opener_factory: Callable[[], Any] | None = None,
            sleep: Callable[[float], None] = time.sleep,
            clock: Callable[[], float] = time.monotonic) -> dict:
    plan_dir, output_dir = _inside_workbuddy(plan_dir), _inside_workbuddy(output_dir)
    static = preflight(plan_dir, parent_plan_path, parent_report_path)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse direct lookup output: {output_dir}")
    plan = _load(plan_dir / "plan.json")
    opener = (opener_factory or (lambda: build_opener(_NoRedirect())))()
    output_dir.mkdir(parents=True, exist_ok=False)
    receipts: list[dict] = []
    total_bytes = 0
    prior_start: float | None = None
    for request_spec in plan["requests"]:
        if prior_start is not None:
            elapsed = clock() - prior_start
            if elapsed < plan["minimum_request_start_interval_seconds"]:
                sleep(plan["minimum_request_start_interval_seconds"] - elapsed)
        nct_id = request_spec["nct_id"]
        request_dir = output_dir / nct_id
        request_dir.mkdir()
        url = plan["endpoint_root"] + request_spec["path"] + "?" + urlencode(plan["params_shared"])
        req = Request(url, headers={"Accept": "application/json", "User-Agent": "RSI-G01-Source-Audit/1.0"})
        prior_start = clock()
        try:
            with opener.open(req, timeout=plan["timeout_seconds"]) as response:
                status, body = response.status, response.read(plan["max_response_bytes_per_request"] + 1)
        except HTTPError as exc:
            status, body = exc.code, exc.read(plan["max_response_bytes_per_request"] + 1)
        except (URLError, TimeoutError, OSError) as exc:
            status, body = 0, type(exc).__name__.encode("ascii")
        (request_dir / "response.bin").write_bytes(body)
        total_bytes += len(body)
        error = None
        returned = None
        if len(body) > plan["max_response_bytes_per_request"]:
            error = "response_byte_limit_exceeded"
        elif status != 200:
            error = f"http_status_{status}"
        else:
            try:
                payload = json.loads(body)
                returned = payload["protocolSection"]["identificationModule"]["nctId"].upper()
                if returned != nct_id:
                    error = "requested_nct_id_mismatch"
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                error = "invalid_or_incomplete_study_record"
        receipt = {
            "requested_nct_id": nct_id,
            "returned_nct_id": returned,
            "http_status": status,
            "response_bytes": len(body),
            "response_sha256": sha_file(request_dir / "response.bin"),
            "status": "matched" if error is None else "unresolved",
            "error": error,
        }
        (request_dir / "receipt.json").write_bytes(canonical(receipt))
        receipts.append(receipt)
    matched = {receipt["requested_nct_id"] for receipt in receipts if receipt["status"] == "matched"}
    complete = len(receipts) == static["request_count"] and matched == set(static["nct_ids"])
    report = {
        "schema": "clinifact-ctgov-unresolved-direct-lookup-capture-v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": plan["plan_sha256"],
        "status": "complete" if complete else "incomplete",
        "requests_attempted": len(receipts),
        "expected_nct_ids": list(static["nct_ids"]),
        "matched_nct_ids": sorted(matched),
        "unresolved_nct_ids": sorted(set(static["nct_ids"]) - matched),
        "response_bytes_total": total_bytes,
        "request_receipts": receipts,
        "model_calls": 0,
        "task_labels_generated": False,
        "confirmation_cohort_created": False,
        "candidate_search_authorized": False,
    }
    (output_dir / "report.json").write_bytes(canonical(report))
    return report
