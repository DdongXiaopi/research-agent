"""Bounded count-only probe for Crossref corrections/retractions in PLOS records."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://api.crossref.org/works"
FILTER = "prefix:10.1371,is-update:true,from-pub-date:2025-01-01,until-pub-date:2026-10-15"
MAX_RESPONSE_BYTES = 64 * 1024
TIMEOUT_SECONDS = 30


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _write_new(path: Path, value: dict) -> None:
    path = path.resolve()
    try:
        path.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("Crossref probe output must stay under project .workbuddy") from exc
    if path.exists():
        raise FileExistsError(f"refusing to overwrite probe artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()


def _write_bytes_new(path: Path, body: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite source response: {path}")
    with path.open("xb") as stream:
        stream.write(body)
        stream.flush()


def run(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("Crossref probe directory must stay under project .workbuddy") from exc
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse probe directory: {output_dir}")
    output_dir.mkdir(parents=True)
    plan = {
        "schema": "crossref-plos-update-count-plan-v1",
        "endpoint": ENDPOINT,
        "filter": FILTER,
        "params": {"rows": "0", "select": "DOI,type,update-to"},
        "max_physical_requests": 1,
        "max_response_bytes": MAX_RESPONSE_BYTES,
        "timeout_seconds": TIMEOUT_SECONDS,
        "redirects": 0,
        "retries": 0,
        "model_calls": 0,
        "purpose": "Count possible PLOS correction/retraction/update records only; do not infer candidate status.",
        "collector_source_sha256": _sha(Path(__file__).read_bytes()),
    }
    plan["plan_sha256"] = _sha(_json_bytes(plan))
    _write_new(output_dir / "plan.json", plan)
    params = {"filter": FILTER, "rows": "0", "select": "DOI,type,update-to"}
    request = Request(ENDPOINT + "?" + urlencode(params), headers={
        "Accept": "application/json",
        "User-Agent": "RSI-Source-Audit/1.0",
    })
    started = datetime.now(timezone.utc).isoformat()
    try:
        with build_opener(_NoRedirect()).open(request, timeout=TIMEOUT_SECONDS) as response:
            status, body = response.status, response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        status, body = exc.code, exc.read(MAX_RESPONSE_BYTES + 1)
    except (URLError, TimeoutError, OSError) as exc:
        status, body = 0, type(exc).__name__.encode()
    result_count = None
    error = None
    if status != 200:
        error = f"http_status_{status}"
    elif len(body) > MAX_RESPONSE_BYTES:
        error = "response_byte_limit_exceeded"
    else:
        try:
            payload = json.loads(body)
            if payload.get("status") != "ok" or payload.get("message-type") != "work-list":
                raise ValueError("unexpected Crossref response envelope")
            result_count = int(payload["message"]["total-results"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            error = f"invalid_response:{type(exc).__name__}"
    _write_bytes_new(output_dir / "response.bin", body)
    receipt = {
        "schema": "crossref-plos-update-count-receipt-v1",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": plan["plan_sha256"],
        "http_status": status,
        "response_bytes": len(body),
        "response_sha256": _sha(body),
        "total_update_records": result_count,
        "error": error,
        "model_calls": 0,
        "candidate_status_verified": False,
    }
    _write_new(output_dir / "receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run(args.output_dir)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if receipt["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
