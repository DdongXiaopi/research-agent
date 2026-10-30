"""Freeze bounded PLOS metadata pages for later article-level review.

The collector stores exact responses and never creates labels, task rows, or a
confirmation cohort. Each invocation needs a new, non-existing output folder.
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
QUERY = (
    'doc_type:full AND publication_date:[2025-01-01T00:00:00Z TO '
    '2026-10-15T23:59:59Z] AND (title:"randomized controlled trial" OR '
    'title:"randomised controlled trial" OR title:"randomized clinical trial" '
    'OR title:"randomised clinical trial") NOT (title:protocol OR title:review '
    'OR title:"meta-analysis" OR title:retracted OR title:correction)'
)
EXPECTED_COUNT = 321
PAGE_SIZE = 100
PAGE_STARTS = (0, 100, 200, 300)
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
TIMEOUT_SECONDS = 30
MIN_REQUEST_INTERVAL_SECONDS = 6
REQUIRED_FIELDS = (
    "id", "title_display", "abstract", "publication_date", "article_type", "copyright",
)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = build_opener(_NoRedirect())


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _metadata_text(value: Any, field: str) -> str:
    """Normalize Solr multi-valued text fields without serializing Python reprs."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return " ".join(item.strip() for item in value if item.strip())
    raise ValueError(f"invalid_metadata_text_shape:{field}")


def _atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    if temp.exists():
        temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    with temp.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def build_plan() -> dict[str, Any]:
    plan = {
        "schema": "plos-clinical-metadata-capture-plan-v1",
        "endpoint": ENDPOINT,
        "collector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": sys.version.split()[0],
        "query": QUERY,
        "expected_count_from_prior_count_only_receipt": EXPECTED_COUNT,
        "field_list": list(REQUIRED_FIELDS),
        "format": "json",
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
        "purpose": "Bounded metadata discovery only; no labels, tasks, confirmation, or promotion.",
        "article_level_license_and_status_review_required": True,
        "history_deduplication_performed": False,
    }
    plan["plan_sha256"] = _sha(_json_bytes(plan))
    return plan


def _get_page(start: int) -> tuple[int, bytes, dict[str, str]]:
    params = {
        "q": QUERY,
        "fl": ",".join(REQUIRED_FIELDS),
        "sort": "publication_date asc,id asc",
        "start": str(start),
        "rows": str(PAGE_SIZE),
        "wt": "json",
    }
    request = Request(ENDPOINT + "?" + urlencode(params), headers={
        "Accept": "application/json",
        "User-Agent": "RSI-PLOS-Source-Discovery/1.0 (research metadata audit)",
    })
    try:
        with _OPENER.open(request, timeout=TIMEOUT_SECONDS) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
            return response.status, body, {
                "content_type": response.headers.get("Content-Type", ""),
                "content_length": response.headers.get("Content-Length", ""),
            }
    except HTTPError as exc:
        body = exc.read(MAX_RESPONSE_BYTES + 1)
        return exc.code, body, {"content_type": exc.headers.get("Content-Type", "")}
    except (URLError, TimeoutError, OSError) as exc:
        return 0, str(type(exc).__name__).encode("ascii"), {"transport_error": type(exc).__name__}


def collect(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("capture output must stay under this project's .workbuddy directory") from exc
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse existing capture directory: {output_dir}")
    output_dir.mkdir(parents=True)
    plan = build_plan()
    _atomic_create(output_dir / "plan.json", _json_bytes(plan))
    started = datetime.now(timezone.utc).isoformat()
    pages: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    terminal_error: str | None = None
    last_request_started = 0.0

    for page_index, start in enumerate(PAGE_STARTS):
        since_last = time.monotonic() - last_request_started
        if last_request_started and since_last < MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(MIN_REQUEST_INTERVAL_SECONDS - since_last)
        request_started_at = datetime.now(timezone.utc).isoformat()
        last_request_started = time.monotonic()
        status, body, headers = _get_page(start)
        page_dir = output_dir / f"page-{page_index:02d}"
        page_dir.mkdir()
        _atomic_create(page_dir / "response.bin", body)
        page = {
            "start": start,
            "request_started_at": request_started_at,
            "response_received_at": datetime.now(timezone.utc).isoformat(),
            "http_status": status,
            "bytes": len(body),
            "response_sha256": _sha(body),
            "headers": headers,
        }
        pages.append(page)
        if status != 200:
            terminal_error = f"http_status_{status}"
            break
        if len(body) > MAX_RESPONSE_BYTES:
            terminal_error = "response_byte_limit_exceeded"
            break
        try:
            payload = json.loads(body)
            result = payload["response"]
            docs = result["docs"]
            num_found = int(result["numFound"])
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            terminal_error = "invalid_plos_json_schema"
            break
        if num_found != EXPECTED_COUNT:
            terminal_error = "prior_count_drift"
            break
        expected_rows = min(PAGE_SIZE, max(0, num_found - start))
        if len(docs) != expected_rows:
            terminal_error = "page_row_count_mismatch"
            break
        for index, doc in enumerate(docs):
            if any(field not in doc for field in REQUIRED_FIELDS):
                terminal_error = "required_metadata_missing"
                break
            records.append({
                "doi": _metadata_text(doc["id"], "id"),
                "title": _metadata_text(doc["title_display"], "title_display"),
                "abstract": _metadata_text(doc["abstract"], "abstract"),
                "publication_date": _metadata_text(doc["publication_date"], "publication_date"),
                "article_type": _metadata_text(doc["article_type"], "article_type"),
                "copyright_statement": _metadata_text(doc["copyright"], "copyright"),
                "source_page_index": page_index,
                "source_page_start": start,
                "source_doc_index": index,
                "source_response_sha256": page["response_sha256"],
                "article_license_status": "not_yet_verified",
                "retraction_status": "not_yet_verified",
                "confirmation_eligible": False,
            })
            if terminal_error:
                break
        if terminal_error:
            break
        _atomic_create(output_dir / "status.json", _json_bytes({
            "schema": "plos-clinical-metadata-capture-status-v1",
            "state": "running",
            "pages_completed": len(pages),
            "records_captured": len(records),
            "next_start": start + PAGE_SIZE,
            "resume_supported": False,
        }))

    doi_values = [record["doi"].casefold() for record in records]
    if len(set(doi_values)) != len(doi_values):
        terminal_error = terminal_error or "duplicate_doi_across_pages"
    completed = terminal_error is None and len(records) == EXPECTED_COUNT
    if records:
        _atomic_create(output_dir / "candidate-index.json", _json_bytes(records))
    receipt = {
        "schema": "plos-clinical-metadata-capture-receipt-v1",
        "state": "completed_metadata_only" if completed else "stopped_incomplete",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": plan["plan_sha256"],
        "pages": pages,
        "records_captured": len(records),
        "unique_doi_count": len(set(doi_values)),
        "terminal_error": terminal_error,
        "model_calls": 0,
        "history_deduplication_performed": False,
        "article_license_review_completed": False,
        "labels_generated": False,
        "confirmation_cohort_created": False,
    }
    _atomic_create(output_dir / "receipt.json", _json_bytes(receipt))
    _atomic_create(output_dir / "status.json", _json_bytes(receipt))
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = collect(args.output_dir)
    except Exception as exc:
        print(f"capture failed before completion: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["state"] == "completed_metadata_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
