"""Collect bounded Crossref and Crossmark status responses for a frozen G05 plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.g05_publication_status import (  # noqa: E402
    build_status_collection_plan,
    build_status_extension_plan,
    classify_publication_status,
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _write_new(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite status artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _fetch(url: str, *, timeout: int, max_bytes: int, interval: float,
           previous_request_at: float | None) -> tuple[dict, bytes, float]:
    if previous_request_at is not None:
        delay = interval - (time.monotonic() - previous_request_at)
        if delay > 0:
            time.sleep(delay)
    started = time.monotonic()
    requested_at = datetime.now(timezone.utc).isoformat()
    request = Request(url, headers={
        "User-Agent": "G05PublicationStatusCollector/1.0 (+https://crossmark.crossref.org/)",
        "Accept": "application/json,text/html;q=0.9,*/*;q=0.1",
    })
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read(max_bytes + 1)
            record = {
                "requested_url": url,
                "final_url": response.geturl(),
                "http_status": int(response.status),
                "headers": {key: value for key, value in response.headers.items()},
                "requested_at": requested_at,
                "truncated": len(body) > max_bytes,
                "transport_error": None,
            }
    except HTTPError as exc:
        body = exc.read(max_bytes + 1)
        record = {
            "requested_url": url,
            "final_url": exc.geturl(),
            "http_status": int(exc.code),
            "headers": {key: value for key, value in exc.headers.items()} if exc.headers else {},
            "requested_at": requested_at,
            "truncated": len(body) > max_bytes,
            "transport_error": f"http_error_{exc.code}",
        }
    except (URLError, TimeoutError, OSError) as exc:
        body = b""
        record = {
            "requested_url": url,
            "final_url": None,
            "http_status": None,
            "headers": {},
            "requested_at": requested_at,
            "truncated": False,
            "transport_error": type(exc).__name__,
        }
    captured_body = body[:max_bytes]
    record["response_sha256"] = _sha_bytes(captured_body)
    record["response_bytes"] = len(captured_body)
    requested_origin = urlparse(url)
    final_origin = urlparse(record["final_url"]) if record["final_url"] else None

    def origin_parts(parsed):
        if parsed is None:
            return None
        default_port = 443 if parsed.scheme.casefold() == "https" else 80
        try:
            port = parsed.port or default_port
        except ValueError:
            return None
        return parsed.scheme.casefold(), (parsed.hostname or "").casefold(), port

    record["final_origin_matches"] = origin_parts(requested_origin) == origin_parts(final_origin)
    record["usable"] = (
        record["http_status"] == 200 and record["transport_error"] is None
        and not record["truncated"] and record["final_origin_matches"])
    record["elapsed_ms"] = round((time.monotonic() - started) * 1000)
    return record, captured_body, time.monotonic()


def collect(plan_path: Path, source_pool_path: Path, output_dir: Path) -> dict:
    plan_path, source_pool_path, output_dir = (
        path.resolve() for path in (plan_path, source_pool_path, output_dir))
    try:
        output_dir.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("status collector output must stay under project .workbuddy") from exc
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse status collector directory: {output_dir}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    pool = json.loads(source_pool_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    expected_plan_sha = hashlib.sha256(json.dumps(
        unsigned, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    if (plan.get("schema") != "g05-publication-status-plan-v1"
            or plan.get("plan_sha256") != expected_plan_sha
            or plan.get("source_pool_sha256") != pool.get("source_pool_sha256")):
        raise ValueError("publication status plan is invalid or not bound to the source pool")
    expected_plan = build_status_collection_plan(
        pool, seed=str(plan.get("seed") or ""),
        sample_size=int(plan.get("sample_size_requested") or 0),
        created_at=str(plan.get("created_at") or ""),
    )
    if plan != expected_plan:
        raise ValueError("publication status plan requests do not match deterministic source-derived plan")
    if plan.get("planned_count") != len(plan.get("requests") or []):
        raise ValueError("publication status plan request count mismatch")
    pool_by_doi = {row["doi"].casefold(): row for row in pool.get("entries") or []}
    for item in plan["requests"]:
        source = pool_by_doi.get(item["doi"].casefold())
        if not source or source.get("source_content_sha256") != item.get("source_content_sha256"):
            raise ValueError("status plan contains an unknown or changed source")

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_new(output_dir / "plan.json", plan_path.read_bytes())
    receipts: dict[str, dict[str, dict]] = {}
    raw_bytes: dict[str, dict[str, bytes]] = {}
    previous_at = None
    transport = plan["transport"]
    for item in plan["requests"]:
        key = hashlib.sha256(item["doi"].encode()).hexdigest()[:16]
        receipts[item["doi"]] = {}
        raw_bytes[item["doi"]] = {}
        for endpoint, url_key in (("crossref", "crossref_url"), ("crossmark", "crossmark_url")):
            record, body, previous_at = _fetch(
                item[url_key], timeout=int(transport["timeout_seconds"]),
                max_bytes=int(transport["max_response_bytes"]),
                interval=float(transport["min_interval_seconds"]),
                previous_request_at=previous_at,
            )
            _write_new(output_dir / "raw" / key / f"{endpoint}.response.bin", body)
            record["raw_path"] = f"raw/{key}/{endpoint}.response.bin"
            receipts[item["doi"]][endpoint] = record
            raw_bytes[item["doi"]][endpoint] = body

    entries = []
    for item in plan["requests"]:
        doi = item["doi"]
        endpoint_receipts = receipts[doi]
        crossref_receipt, crossmark_receipt = endpoint_receipts["crossref"], endpoint_receipts["crossmark"]
        crossref_payload = None
        crossmark_html = None
        if crossref_receipt["usable"]:
            try:
                envelope = json.loads(raw_bytes[doi]["crossref"].decode("utf-8"))
                crossref_payload = envelope.get("message") if envelope.get("status") == "ok" else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
        if crossmark_receipt["usable"]:
            try:
                crossmark_html = raw_bytes[doi]["crossmark"].decode("utf-8")
            except UnicodeDecodeError:
                pass
        verdict = classify_publication_status(item["expected_identity"], crossref_payload, crossmark_html)
        if not crossref_receipt["usable"] or not crossmark_receipt["usable"]:
            verdict["disposition"] = "unverified_exclude"
            verdict["eligible_for_status_screen"] = False
        entries.append({
            **verdict,
            "source_group_id": item["source_group_id"],
            "source_content_sha256": item["source_content_sha256"],
            "capture_timestamp": max(crossref_receipt["requested_at"], crossmark_receipt["requested_at"]),
            "source_urls": {
                "crossref": crossref_receipt["final_url"] or crossref_receipt["requested_url"],
                "crossmark": crossmark_receipt["final_url"] or crossmark_receipt["requested_url"],
            },
            "raw_response_sha256": {
                "crossref": crossref_receipt["response_sha256"],
                "crossmark": crossmark_receipt["response_sha256"],
            },
            "transport": endpoint_receipts,
        })
    report: dict = {
        "schema": "g05-publication-status-report-v1",
        "source_pool_file_sha256": _sha_file(source_pool_path),
        "source_pool_sha256": pool["source_pool_sha256"],
        "status_plan_sha256": plan["plan_sha256"],
        "status_plan_file_sha256": _sha_file(output_dir / "plan.json"),
        "capture_started_at": plan["created_at"],
        "capture_completed_at": datetime.now(timezone.utc).isoformat(),
        "planned_count": len(plan["requests"]),
        "verified_current_count": sum(row["eligible_for_status_screen"] for row in entries),
        "unverified_excluded_count": sum(not row["eligible_for_status_screen"] for row in entries),
        "scope": "point_in_time_only",
        "entries": entries,
        "labels_generated": False,
        "model_calls": 0,
        "target_scorer_calls": 0,
    }
    unsigned_report = dict(report)
    report["report_sha256"] = hashlib.sha256(json.dumps(
        unsigned_report, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    _write_new(output_dir / "report.json", (json.dumps(
        report, sort_keys=True, ensure_ascii=False, indent=2
    ) + "\n").encode("utf-8"))
    _write_new(output_dir / "receipt.json", (json.dumps({
        "state": "completed", "plan_sha256": plan["plan_sha256"],
        "report_sha256": report["report_sha256"], "request_count": 2 * len(plan["requests"]),
        "model_calls": 0, "target_scorer_calls": 0,
    }, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return report


def collect_extension(
    plan_path: Path, source_pool_path: Path, parent_plan_path: Path,
    parent_report_path: Path, parent_sample_plan_path: Path, output_dir: Path,
) -> dict:
    """Collect the frozen two-DOI extension using four bounded requests.

    This path is intentionally separate from the immutable v3 capture. The
    plan is reconstructed from the frozen source pool and parent evidence
    before the first request; retries are never performed.
    """
    paths = tuple(path.resolve() for path in (
        plan_path, source_pool_path, parent_plan_path, parent_report_path,
        parent_sample_plan_path, output_dir))
    plan_path, source_pool_path, parent_plan_path, parent_report_path, parent_sample_plan_path, output_dir = paths
    try:
        output_dir.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("status collector output must stay under project .workbuddy") from exc
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse status collector directory: {output_dir}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    pool = json.loads(source_pool_path.read_text(encoding="utf-8"))
    parent_plan = json.loads(parent_plan_path.read_text(encoding="utf-8"))
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    parent_sample = json.loads(parent_sample_plan_path.read_text(encoding="utf-8"))
    expected_plan = build_status_extension_plan(
        pool, parent_plan, parent_report, parent_sample,
        # The first frozen extension schema omitted its creation time. It is
        # deliberately not reconstructed from the wall clock; this stable
        # non-empty sentinel preserves the original plan hash.
        created_at="extension-plan-v1-timestamp-not-in-schema",
    )
    if plan != expected_plan:
        raise ValueError("extension plan differs from deterministic parent-bound plan")
    if plan.get("physical_request_count") != 4 or len(plan.get("requests") or []) != 4:
        raise ValueError("extension collector requires exactly four frozen requests")

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_new(output_dir / "extension-plan.json", plan_path.read_bytes())
    receipts: dict[str, dict[str, dict]] = {item["doi"]: {} for item in plan["doi_candidates"]}
    raw_bytes: dict[str, dict[str, bytes]] = {item["doi"]: {} for item in plan["doi_candidates"]}
    previous_at = None
    transport = plan["transport"]
    for item in plan["requests"]:
        record, body, previous_at = _fetch(
            item["url"], timeout=int(transport["timeout_seconds"]),
            max_bytes=int(transport["max_response_bytes"]),
            interval=float(transport["min_interval_seconds"]),
            previous_request_at=previous_at,
        )
        doi_key = hashlib.sha256(item["doi"].encode()).hexdigest()[:16]
        endpoint = item["service"]
        _write_new(output_dir / "raw" / doi_key / f"{endpoint}.response.bin", body)
        record["raw_path"] = f"raw/{doi_key}/{endpoint}.response.bin"
        receipts[item["doi"]][endpoint] = record
        raw_bytes[item["doi"]][endpoint] = body

    entries = []
    for item in plan["doi_candidates"]:
        doi = item["doi"]
        endpoint_receipts = receipts[doi]
        crossref_receipt = endpoint_receipts["crossref"]
        crossmark_receipt = endpoint_receipts["crossmark"]
        crossref_payload = None
        crossmark_html = None
        if crossref_receipt["usable"]:
            try:
                envelope = json.loads(raw_bytes[doi]["crossref"].decode("utf-8"))
                crossref_payload = envelope.get("message") if envelope.get("status") == "ok" else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
        if crossmark_receipt["usable"]:
            try:
                crossmark_html = raw_bytes[doi]["crossmark"].decode("utf-8")
            except UnicodeDecodeError:
                pass
        verdict = classify_publication_status(item["expected_identity"], crossref_payload, crossmark_html)
        if not crossref_receipt["usable"] or not crossmark_receipt["usable"]:
            verdict["disposition"] = "unverified_exclude"
            verdict["eligible_for_status_screen"] = False
        entries.append({
            **verdict,
            "source_group_id": item["source_group_id"],
            "source_content_sha256": item["source_content_sha256"],
            "capture_timestamp": max(crossref_receipt["requested_at"], crossmark_receipt["requested_at"]),
            "source_urls": {
                "crossref": crossref_receipt["final_url"] or crossref_receipt["requested_url"],
                "crossmark": crossmark_receipt["final_url"] or crossmark_receipt["requested_url"],
            },
            "raw_response_sha256": {
                "crossref": crossref_receipt["response_sha256"],
                "crossmark": crossmark_receipt["response_sha256"],
            },
            "transport": endpoint_receipts,
        })
    report: dict = {
        "schema": "g05-publication-status-extension-report-v1",
        "source_pool_file_sha256": _sha_file(source_pool_path),
        "source_pool_sha256": pool["source_pool_sha256"],
        "parent_status_plan_sha256": plan["parent_status_plan_sha256"],
        "parent_status_report_sha256": plan["parent_status_report_sha256"],
        "parent_sample_plan_sha256": plan["parent_sample_plan_sha256"],
        "extension_plan_sha256": plan["plan_sha256"],
        "extension_plan_file_sha256": _sha_file(output_dir / "extension-plan.json"),
        # The frozen v1 extension plan omitted a creation timestamp. Record
        # actual request timestamps instead of inventing one from wall clock.
        "capture_started_at": min(r["requested_at"] for pair in receipts.values()
                                   for r in pair.values()),
        "capture_completed_at": max(r["requested_at"] for pair in receipts.values()
                                     for r in pair.values()),
        "planned_count": len(plan["doi_candidates"]),
        "physical_request_count": 4,
        "verified_current_count": sum(row["eligible_for_status_screen"] for row in entries),
        "unverified_excluded_count": sum(not row["eligible_for_status_screen"] for row in entries),
        "scope": "point_in_time_only",
        "entries": entries,
        "labels_generated": False,
        "model_calls": 0,
        "target_scorer_calls": 0,
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(
        report, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    _write_new(output_dir / "report.json", (json.dumps(
        report, sort_keys=True, ensure_ascii=False, indent=2
    ) + "\n").encode("utf-8"))
    _write_new(output_dir / "receipt.json", (json.dumps({
        "state": "completed", "plan_sha256": plan["plan_sha256"],
        "report_sha256": report["report_sha256"], "request_count": 4,
        "model_calls": 0, "target_scorer_calls": 0,
    }, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extension", action="store_true",
                        help="collect a frozen two-DOI extension; requires parent evidence paths")
    parser.add_argument("--parent-plan", type=Path)
    parser.add_argument("--parent-report", type=Path)
    parser.add_argument("--parent-sample-plan", type=Path)
    args = parser.parse_args(argv)
    if args.extension:
        if not (args.parent_plan and args.parent_report and args.parent_sample_plan):
            parser.error("--extension requires --parent-plan, --parent-report, and --parent-sample-plan")
        report = collect_extension(args.plan, args.source_pool, args.parent_plan,
                                   args.parent_report, args.parent_sample_plan, args.output_dir)
    else:
        if any((args.parent_plan, args.parent_report, args.parent_sample_plan)):
            parser.error("parent evidence paths are valid only with --extension")
        report = collect(args.plan, args.source_pool, args.output_dir)
    print(json.dumps({
        "status": "written",
        "output_dir": str(args.output_dir.resolve()),
        "planned_dois": report["planned_count"],
        "physical_requests": report.get("physical_request_count", 2 * report["planned_count"]),
        "verified_current_as_of_capture": report["verified_current_count"],
        "unverified_excluded": report["unverified_excluded_count"],
        "target_scorer_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
