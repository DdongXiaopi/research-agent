"""Fetch a bounded Crossref update inventory for a frozen PLOS DOI cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://api.crossref.org/works"
FILTER = "prefix:10.1371,is-update:true,from-pub-date:2025-01-01,until-pub-date:2026-10-15"
EXPECTED_COUNT = 7018
PAGE_SIZE = 1000
MAX_PAGES = 8
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
TIMEOUT_SECONDS = 45
MIN_REQUEST_INTERVAL_SECONDS = 6
SELECT = "DOI,type,update-to,published,created,relation"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _atomic_replace(path: Path, raw: bytes) -> None:
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    with temp.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def collect(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("Crossref capture must stay under project .workbuddy") from exc
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse Crossref capture directory: {output_dir}")
    output_dir.mkdir(parents=True)
    plan = {
        "schema": "crossref-plos-update-capture-plan-v1",
        "endpoint": ENDPOINT,
        "filter": FILTER,
        "select": SELECT,
        "expected_total_records_from_count_probe": EXPECTED_COUNT,
        "page_size": PAGE_SIZE,
        "max_physical_requests": MAX_PAGES,
        "max_response_bytes_per_page": MAX_RESPONSE_BYTES,
        "timeout_seconds": TIMEOUT_SECONDS,
        "minimum_request_start_interval_seconds": MIN_REQUEST_INTERVAL_SECONDS,
        "cursor_pagination": True,
        "retries": 0,
        "redirects": 0,
        "model_calls": 0,
        "purpose": "Find deposited update-to links targeting the frozen PLOS candidate DOIs; absence is not proof of active status.",
        "collector_source_sha256": _sha(Path(__file__).read_bytes()),
    }
    plan["plan_sha256"] = _sha(_json_bytes(plan))
    (output_dir / "plan.json").write_bytes(_json_bytes(plan))

    pages: list[dict] = []
    cursor = "*"
    records: list[dict] = []
    seen_dois: set[str] = set()
    last_request_started = 0.0
    terminal_error = None
    for page_index in range(MAX_PAGES):
        since_last = time.monotonic() - last_request_started
        if last_request_started and since_last < MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(MIN_REQUEST_INTERVAL_SECONDS - since_last)
        request_started_at = datetime.now(timezone.utc).isoformat()
        last_request_started = time.monotonic()
        params = {"filter": FILTER, "select": SELECT, "rows": str(PAGE_SIZE), "cursor": cursor}
        request = Request(ENDPOINT + "?" + urlencode(params), headers={
            "Accept": "application/json",
            "User-Agent": "RSI-Source-Audit/1.0",
        })
        try:
            with build_opener(_NoRedirect()).open(request, timeout=TIMEOUT_SECONDS) as response:
                status, body = response.status, response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            status, body = exc.code, exc.read(MAX_RESPONSE_BYTES + 1)
        except (URLError, TimeoutError, OSError) as exc:
            status, body = 0, type(exc).__name__.encode("ascii")
        page_dir = output_dir / f"page-{page_index:02d}"
        page_dir.mkdir()
        (page_dir / "response.bin").write_bytes(body)
        page = {
            "page_index": page_index,
            "request_started_at": request_started_at,
            "response_received_at": datetime.now(timezone.utc).isoformat(),
            "request_cursor": cursor,
            "http_status": status,
            "response_bytes": len(body),
            "response_sha256": _sha(body),
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
            if payload.get("status") != "ok" or payload.get("message-type") != "work-list":
                raise ValueError("unexpected Crossref response envelope")
            message = payload["message"]
            total = int(message["total-results"])
            items = message["items"]
            next_cursor = message.get("next-cursor")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            terminal_error = f"invalid_response:{type(exc).__name__}"
            break
        if total != EXPECTED_COUNT:
            terminal_error = "total_count_changed_since_probe"
            break
        expected_rows = min(PAGE_SIZE, total - len(records))
        if len(items) != expected_rows:
            terminal_error = "page_size_mismatch"
            break
        for item in items:
            doi = str(item.get("DOI") or "").casefold()
            if not doi or doi in seen_dois:
                terminal_error = "missing_or_duplicate_update_doi"
                break
            seen_dois.add(doi)
            records.append({
                "update_doi": doi,
                "type": item.get("type"),
                "update_to": item.get("update-to") or [],
                "published": item.get("published"),
                "created": item.get("created"),
                "relation": item.get("relation") or {},
                "source_page_index": page_index,
                "source_response_sha256": page["response_sha256"],
            })
        if terminal_error:
            break
        cursor = str(next_cursor or "")
        status_body = {
            "schema": "crossref-plos-update-capture-progress-v1",
            "pages_completed": len(pages),
            "records_captured": len(records),
            "next_cursor_sha256": _sha(cursor.encode()) if cursor else None,
            "resume_supported": False,
        }
        _atomic_replace(output_dir / "progress.json", _json_bytes(status_body))
        if len(records) == total:
            break
        if not cursor:
            terminal_error = "missing_next_cursor_before_total"
            break

    completed = terminal_error is None and len(records) == EXPECTED_COUNT
    if records:
        (output_dir / "update-index.json").write_bytes(_json_bytes(records))
    receipt = {
        "schema": "crossref-plos-update-capture-receipt-v1",
        "state": "completed" if completed else "stopped_incomplete",
        "started_at": pages[0]["request_started_at"] if pages else None,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": plan["plan_sha256"],
        "pages": pages,
        "records_captured": len(records),
        "unique_update_doi_count": len(seen_dois),
        "terminal_error": terminal_error,
        "model_calls": 0,
        "targeted_candidate_updates": None,
        "candidate_status_verified": False,
    }
    (output_dir / "receipt.json").write_bytes(_json_bytes(receipt))
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = collect(args.output_dir)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if receipt["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
