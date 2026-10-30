"""Fetch a bounded PLOS metadata-only subset for source eligibility triage.

No full text, task labels, candidate evaluation, or model calls are performed.
Every run creates a new write-once directory under .workbuddy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://api.plos.org/search"
EXPECTED_COUNT = 611
PAGE_SIZE = 100
PAGE_STARTS = (0, 100, 200, 300, 400, 500, 600)
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
TIMEOUT_SECONDS = 30
MIN_REQUEST_INTERVAL_SECONDS = 6
QUERY = (
    'doc_type:full AND publication_date:[2010-01-01T00:00:00Z TO '
    '2024-12-31T23:59:59Z] AND (title:"randomized controlled trial" OR '
    'title:"randomised controlled trial" OR title:"randomized clinical trial" '
    'OR title:"randomised clinical trial") NOT (title:protocol OR title:review '
    'OR title:"meta-analysis" OR title:retracted OR title:correction) '
    'AND abstract:NCT* AND (abstract:p* OR abstract:"confidence interval")'
)
FIELDS = ("id", "title_display", "abstract", "publication_date", "article_type", "copyright")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = build_opener(_NoRedirect())


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _text(value: Any, field: str) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return " ".join(item.strip() for item in value if item.strip())
    raise ValueError(f"invalid_metadata_text_shape:{field}")


def build_plan() -> dict:
    plan = {
        "schema": "clinifact-plos-historical-metadata-plan-v1",
        "endpoint": ENDPOINT,
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "capacity_receipt_sha256": hashlib.sha256(
            (ROOT / ".workbuddy/clinifact-plos-historical-capacity-20261017-v1/receipt.json").read_bytes()
        ).hexdigest(),
        "capacity_plan_sha256": "742c32b6d1eaa93a7cbc1f40bb18edcc16898e5a9f1c4245c223204059ef3eb4",
        "query": QUERY,
        "expected_num_found": EXPECTED_COUNT,
        "fields": list(FIELDS),
        "page_size": PAGE_SIZE,
        "page_starts": list(PAGE_STARTS),
        "max_physical_requests": len(PAGE_STARTS),
        "max_response_bytes_per_request": MAX_RESPONSE_BYTES,
        "timeout_seconds": TIMEOUT_SECONDS,
        "minimum_request_start_interval_seconds": MIN_REQUEST_INTERVAL_SECONDS,
        "maximum_requests_per_minute": 10,
        "redirects": 0,
        "retries": 0,
        "model_calls": 0,
        "full_text_downloaded": False,
        "labels_generated": False,
        "candidate_cohort_created": False,
        "purpose": "Metadata-only source eligibility triage; no article/task eligibility is asserted.",
    }
    plan["plan_sha256"] = _sha(_canonical(plan))
    return plan


def parse_page(body: bytes, *, start: int) -> tuple[int, list[dict[str, str]]]:
    if len(body) > MAX_RESPONSE_BYTES:
        raise ValueError("response_byte_limit_exceeded")
    try:
        payload = json.loads(body.decode("utf-8"))
        response = payload["response"]
        num_found = response["numFound"]
        docs = response["docs"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("invalid_plos_page_schema") from exc
    if type(num_found) is not int or num_found != EXPECTED_COUNT or not isinstance(docs, list):
        raise ValueError("historical_count_drift_or_invalid_docs")
    expected_rows = min(PAGE_SIZE, max(0, num_found - start))
    if len(docs) != expected_rows:
        raise ValueError("page_row_count_mismatch")
    rows = []
    for index, doc in enumerate(docs):
        try:
            row = {"doi": _text(doc["id"], "id"),
                   "title": _text(doc["title_display"], "title_display"),
                   "abstract": _text(doc["abstract"], "abstract"),
                   "publication_date": _text(doc["publication_date"], "publication_date"),
                   "article_type": _text(doc["article_type"], "article_type"),
                   "copyright_statement": _text(doc["copyright"], "copyright")}
        except (KeyError, ValueError) as exc:
            raise ValueError(f"required_metadata_invalid_at_{start + index}") from exc
        row.update({"page_start": start, "page_index": index})
        rows.append(row)
    return num_found, rows


def _get_page(start: int) -> tuple[int, bytes]:
    params = {"q": QUERY, "fl": ",".join(FIELDS),
              "sort": "publication_date asc,id asc", "start": str(start),
              "rows": str(PAGE_SIZE), "wt": "json"}
    request = Request(ENDPOINT + "?" + urlencode(params), headers={
        "Accept": "application/json",
        "User-Agent": "RSI-CliniFact-Source-Discovery/1.0 (bounded metadata audit)",
    })
    try:
        with _OPENER.open(request, timeout=TIMEOUT_SECONDS) as response:
            return int(response.status), response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        return int(exc.code), exc.read(MAX_RESPONSE_BYTES + 1)
    except (URLError, TimeoutError, OSError) as exc:
        return 0, type(exc).__name__.encode("ascii")


def freeze_plan(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("capture output must stay under project .workbuddy") from exc
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse capture directory: {output_dir}")
    plan = build_plan()
    output_dir.mkdir(parents=True)
    (output_dir / "plan.json").write_bytes(_canonical(plan))
    return plan


def _load_frozen_plan(output_dir: Path) -> dict:
    plan_path = output_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    unsigned = dict(plan)
    digest = unsigned.pop("plan_sha256", None)
    if (not digest or digest != _sha(_canonical(unsigned))
            or plan != build_plan()):
        raise ValueError("saved plan is stale, altered, or does not match current source")
    return plan


def run(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("capture output must stay under project .workbuddy") from exc
    if not output_dir.is_dir() or not (output_dir / "plan.json").is_file():
        raise ValueError("execute requires a pre-frozen write-once plan")
    if (output_dir / "report.json").exists():
        raise FileExistsError("refusing to overwrite an existing report")
    plan = _load_frozen_plan(output_dir)
    if any((output_dir / f"page-{index:02d}").exists() for index in range(len(PAGE_STARTS))):
        raise FileExistsError("refusing to resume or overwrite partial page outputs")
    page_receipts, rows = [], []
    terminal_error = None
    last_started = None
    for page_no, start in enumerate(PAGE_STARTS):
        if last_started is not None:
            remaining = MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - last_started)
            if remaining > 0:
                time.sleep(remaining)
        started_at = datetime.now(timezone.utc).isoformat()
        last_started = time.monotonic()
        status, body = _get_page(start)
        page_dir = output_dir / f"page-{page_no:02d}"
        page_dir.mkdir()
        (page_dir / "response.bin").write_bytes(body)
        receipt = {"start": start, "started_at": started_at, "http_status": status,
                   "bytes": len(body), "response_sha256": _sha(body)}
        if status != 200:
            receipt["error"] = f"http_status_{status}"
            terminal_error = receipt["error"]
            page_receipts.append(receipt)
            break
        try:
            num_found, page_rows = parse_page(body, start=start)
        except ValueError as exc:
            receipt["error"] = str(exc)
            terminal_error = receipt["error"]
            page_receipts.append(receipt)
            break
        receipt["num_found"] = num_found
        receipt["rows"] = len(page_rows)
        page_receipts.append(receipt)
        for row in page_rows:
            row["source_page_response_sha256"] = receipt["response_sha256"]
            row["license_status"] = "not_independently_verified"
            row["history_status"] = "not_scanned"
            row["trial_outcome_identity_status"] = "not_reviewed"
            row["confirmation_eligible"] = False
        rows.extend(page_rows)
    dois = [row["doi"].casefold() for row in rows]
    if len(set(dois)) != len(dois):
        terminal_error = terminal_error or "duplicate_doi_across_pages"
    complete = terminal_error is None and len(rows) == EXPECTED_COUNT and len(set(dois)) == EXPECTED_COUNT
    if rows:
        (output_dir / "metadata-index.json").write_bytes(_canonical(rows))
    report = {
        "schema": "clinifact-plos-historical-metadata-report-v1",
        "status": "metadata_capture_complete" if complete else "stopped_incomplete",
        "plan_sha256": plan["plan_sha256"],
        "pages": page_receipts,
        "article_count": len(rows),
        "unique_doi_count": len(set(dois)),
        "terminal_error": terminal_error,
        "model_calls": 0,
        "full_text_downloaded": False,
        "labels_generated": False,
        "candidate_cohort_created": False,
        "all_rows_confirmation_eligible": False,
        "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output_dir / "report.json").write_bytes(_canonical(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan", help="freeze a zero-call metadata capture plan")
    plan_parser.add_argument("--output-dir", required=True, type=Path)
    fetch_parser = sub.add_parser("fetch", help="execute the exact saved plan")
    fetch_parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        result = freeze_plan(args.output_dir)
        print(json.dumps({"status": "frozen_zero_call_plan", "plan_sha256": result["plan_sha256"],
                          "output_dir": str(args.output_dir.resolve()),
                          "model_calls": 0, "max_physical_requests": result["max_physical_requests"]},
                         ensure_ascii=False, indent=2))
        return 0
    report = run(args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "metadata_capture_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
