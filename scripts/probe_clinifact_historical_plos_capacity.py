"""Bounded count-only probe for historical licensed CliniFact source capacity.

This probes the PLOS search index for older randomized clinical-trial articles.
It reads only `numFound` counts (rows=0), does not download article metadata,
assign labels, or call a model. Counts are discovery hints, not eligibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://api.plos.org/search"
BASE_QUERY = (
    'doc_type:full AND publication_date:[2010-01-01T00:00:00Z TO '
    '2024-12-31T23:59:59Z] AND (title:"randomized controlled trial" OR '
    'title:"randomised controlled trial" OR title:"randomized clinical trial" '
    'OR title:"randomised clinical trial") NOT (title:protocol OR title:review '
    'OR title:"meta-analysis" OR title:retracted OR title:correction)'
)
QUERIES = {
    "historical_trial_title_candidates": BASE_QUERY,
    "historical_nct_and_statistics_indicators": (
        BASE_QUERY + ' AND abstract:NCT* AND (abstract:p* OR abstract:"confidence interval")'
    ),
}
MAX_RESPONSE_BYTES = 64 * 1024
TIMEOUT_SECONDS = 30
MIN_REQUEST_INTERVAL_SECONDS = 6


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_plan() -> dict:
    plan = {
        "schema": "clinifact-plos-historical-count-plan-v1",
        "endpoint": ENDPOINT,
        "collector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "queries": QUERIES,
        "query_parameters": {"rows": 0, "wt": "json"},
        "max_physical_requests": len(QUERIES),
        "max_response_bytes_per_request": MAX_RESPONSE_BYTES,
        "timeout_seconds": TIMEOUT_SECONDS,
        "minimum_request_start_interval_seconds": MIN_REQUEST_INTERVAL_SECONDS,
        "retries": 0,
        "redirects": "urllib default HTTPS behavior only",
        "model_calls": 0,
        "article_text_downloaded": False,
        "labels_generated": False,
        "cohort_created": False,
        "purpose": "Historical source-volume census only; numFound is not a label or eligibility claim.",
    }
    plan["plan_sha256"] = _sha_bytes(_canonical(plan))
    return plan


def validate_count_response(body: bytes) -> int:
    if len(body) > MAX_RESPONSE_BYTES:
        raise ValueError("response_byte_limit_exceeded")
    try:
        payload = json.loads(body.decode("utf-8"))
        response = payload["response"]
        count = response["numFound"]
        docs = response["docs"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("invalid_count_response_schema") from exc
    if type(count) is not int or count < 0 or docs != []:
        raise ValueError("count_only_response_must_have_nonnegative_numFound_and_no_docs")
    return count


def _fetch(query: str) -> tuple[int, bytes]:
    params = {"q": query, "rows": "0", "wt": "json"}
    request = Request(ENDPOINT + "?" + urlencode(params), headers={
        "Accept": "application/json",
        "User-Agent": "RSI-CliniFact-Source-Capacity/1.0 (count-only research audit)",
    })
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return int(response.status), response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        return int(exc.code), exc.read(MAX_RESPONSE_BYTES + 1)
    except (URLError, TimeoutError, OSError) as exc:
        return 0, type(exc).__name__.encode("ascii")


def run(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("output must stay under project .workbuddy") from exc
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    plan = build_plan()
    output_dir.mkdir(parents=True)
    (output_dir / "plan.json").write_bytes(_canonical(plan))
    rows, errors = [], []
    prior_started = None
    for name, query in QUERIES.items():
        if prior_started is not None:
            remaining = MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - prior_started)
            if remaining > 0:
                time.sleep(remaining)
        started = datetime.now(timezone.utc).isoformat()
        prior_started = time.monotonic()
        status, body = _fetch(query)
        (output_dir / f"{name}.response.bin").write_bytes(body)
        entry = {"name": name, "request_started_at": started,
                 "http_status": status, "response_bytes": len(body),
                 "response_sha256": _sha_bytes(body)}
        if status != 200:
            entry["error"] = f"http_status_{status}"
        else:
            try:
                entry["num_found"] = validate_count_response(body)
            except ValueError as exc:
                entry["error"] = str(exc)
        if "error" in entry:
            errors.append(entry["error"])
        rows.append(entry)
    receipt = {
        "schema": "clinifact-plos-historical-count-receipt-v1",
        "plan_sha256": plan["plan_sha256"],
        "results": rows,
        "model_calls": 0,
        "article_text_downloaded": False,
        "labels_generated": False,
        "cohort_created": False,
        "status": "complete" if not errors else "incomplete",
        "errors": errors,
    }
    (output_dir / "receipt.json").write_bytes(_canonical(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    receipt = run(args.output_dir)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
