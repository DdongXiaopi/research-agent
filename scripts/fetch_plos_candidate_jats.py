"""Fetch a small DOI-bound PLOS JATS subset through the documented file endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://journals.plos.org/plosone/article/file"
MAX_DOIS = 10
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
TIMEOUT_SECONDS = 45
MIN_REQUEST_START_INTERVAL_SECONDS = 6
REDIRECT_HOST = "storage.googleapis.com"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _get(url: str) -> tuple[int, bytes, str | None, str | None]:
    request = Request(url, headers={
        "Accept": "application/xml",
        "User-Agent": "RSI-Source-Audit/1.0",
    })
    try:
        with build_opener(_NoRedirect()).open(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, response.read(MAX_RESPONSE_BYTES + 1), response.headers.get("Content-Type"), None
    except HTTPError as exc:
        return exc.code, exc.read(MAX_RESPONSE_BYTES + 1), exc.headers.get("Content-Type"), exc.headers.get("Location")
    except (URLError, TimeoutError, OSError) as exc:
        return 0, type(exc).__name__.encode("ascii"), None, None


def _safe_redirect_location(location: str, doi: str) -> bool:
    parsed = urlparse(location)
    expected_prefix = f"/plos-corpus-prod/{doi}/"
    return (
        parsed.scheme == "https"
        and parsed.hostname == REDIRECT_HOST
        and parsed.port is None
        and parsed.username is None
        and parsed.password is None
        and parsed.path.startswith(expected_prefix)
        and parsed.path.lower().endswith(".xml")
    )


def _parse_jats_sections(body: bytes, expected_doi: str) -> dict[str, list[str]]:
    root = ET.fromstring(body)
    article_dois = {
        (element.text or "").strip().casefold()
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "article-id"
        and element.attrib.get("pub-id-type") == "doi"
    }
    if expected_doi.casefold() not in article_dois:
        raise ValueError("JATS DOI does not match requested candidate")
    all_text = " ".join(root.itertext())
    section_text: dict[str, list[str]] = {"abstract": [], "body": [], "references": []}
    for element in root.iter():
        local = element.tag.rsplit("}", 1)[-1]
        if local == "ref-list":
            local = "references"
        if local in section_text:
            section_text[local].append(" ".join(element.itertext()))
    def ids(texts: list[str]) -> list[str]:
        text = " ".join(texts)
        return sorted(set(re.findall(r"(?<![A-Z0-9])NCT\d{8}(?!\d)", text)))
    abstract_ids = ids(section_text["abstract"])
    body_ids = ids(section_text["body"])
    reference_ids = ids(section_text["references"])
    all_ids = ids([all_text])
    return {
        "abstract": abstract_ids,
        "body": body_ids,
        "references": reference_ids,
        "trial_identity_candidates": sorted(set(abstract_ids) | set(body_ids)),
        "fulltext_all_sections": all_ids,
    }


def _parse_jats(body: bytes, expected_doi: str) -> tuple[list[str], bool]:
    sections = _parse_jats_sections(body, expected_doi)
    return sections["fulltext_all_sections"], True


def collect(dois: list[str], output_dir: Path) -> dict:
    canonical_dois = sorted({doi.strip().casefold() for doi in dois})
    if len(canonical_dois) != len(dois) or not canonical_dois:
        raise ValueError("DOIs must be nonempty and unique")
    if len(canonical_dois) > MAX_DOIS:
        raise ValueError(f"at most {MAX_DOIS} DOI articles may be fetched per capture")
    if any(not re.fullmatch(r"10\.1371/journal\.[a-z0-9]+\.\d{7}", doi) for doi in canonical_dois):
        raise ValueError("all DOI values must be PLOS journal DOI identifiers")
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("PLOS JATS capture must stay under project .workbuddy") from exc
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse PLOS JATS capture directory: {output_dir}")
    output_dir.mkdir(parents=True)
    plan = {
        "schema": "plos-candidate-jats-capture-plan-v1",
        "endpoint": ENDPOINT,
        "doi_ids": canonical_dois,
        "max_articles": MAX_DOIS,
        "max_physical_requests": 2 * len(canonical_dois),
        "max_response_bytes_per_request": MAX_RESPONSE_BYTES,
        "timeout_seconds": TIMEOUT_SECONDS,
        "minimum_article_request_start_interval_seconds": MIN_REQUEST_START_INTERVAL_SECONDS,
        "max_redirects_per_article": 1,
        "redirect_allowlist": {
            "https_host": REDIRECT_HOST,
            "path_prefix_pattern": "/plos-corpus-prod/{exact_doi}/",
            "path_suffix": ".xml",
        },
        "retries": 0,
        "model_calls": 0,
        "purpose": "Retrieve JATS for a small, already screened DOI subset to verify trial and outcome evidence; source capture is not a task label or cohort.",
        "collector_source_sha256": _sha(Path(__file__).read_bytes()),
    }
    plan["plan_sha256"] = _sha(_json_bytes(plan))
    (output_dir / "plan.json").write_bytes(_json_bytes(plan))

    receipts = []
    article_index = []
    last_start = 0.0
    for index, doi in enumerate(canonical_dois):
        wait = MIN_REQUEST_START_INTERVAL_SECONDS - (time.monotonic() - last_start)
        if last_start and wait > 0:
            time.sleep(wait)
        started = datetime.now(timezone.utc).isoformat()
        last_start = time.monotonic()
        article_dir = output_dir / f"article-{index:02d}"
        article_dir.mkdir()
        source_url = ENDPOINT + "?" + urlencode({"id": doi, "type": "manuscript"})
        first_status, first_body, first_type, location = _get(source_url)
        (article_dir / "origin-response.bin").write_bytes(first_body)
        response_status, body, content_type, redirect_host = first_status, first_body, first_type, None
        redirect_location_sha256 = None
        redirect_rejected = False
        physical_requests = 1
        if first_status in (301, 302, 303, 307, 308):
            if not location or not _safe_redirect_location(location, doi):
                redirect_rejected = True
            else:
                redirect_host = urlparse(location).hostname
                redirect_location_sha256 = _sha(location.encode())
                response_status, body, content_type, _ = _get(location)
                physical_requests += 1
        (article_dir / "response.xml").write_bytes(body)
        ncts: list[str] = []
        nct_ids_by_section: dict[str, list[str]] = {}
        xml_ok = False
        error = None
        if redirect_rejected:
            error = "redirect_target_rejected"
        elif response_status != 200:
            error = f"http_status_{response_status}"
        elif len(body) > MAX_RESPONSE_BYTES:
            error = "response_byte_limit_exceeded"
        else:
            try:
                nct_ids_by_section = _parse_jats_sections(body, doi)
                ncts = nct_ids_by_section["fulltext_all_sections"]
                xml_ok = True
            except (ET.ParseError, ValueError) as exc:
                error = f"invalid_jats:{type(exc).__name__}"
        receipt = {
            "doi": doi,
            "request_started_at": started,
            "origin_http_status": first_status,
            "origin_content_type": first_type,
            "origin_response_bytes": len(first_body),
            "origin_response_sha256": _sha(first_body),
            "redirect_followed": physical_requests == 2,
            "redirect_host": redirect_host,
            "redirect_location_sha256": redirect_location_sha256,
            "redirect_rejected": redirect_rejected,
            "final_http_status": response_status,
            "final_content_type": content_type,
            "final_response_bytes": len(body),
            "final_response_sha256": _sha(body),
            "valid_jats_for_requested_doi": xml_ok,
            "nct_ids_in_fulltext": ncts,
            "nct_ids_by_section": nct_ids_by_section,
            "physical_requests": physical_requests,
            "model_calls": 0,
            "error": error,
        }
        (article_dir / "receipt.json").write_bytes(_json_bytes(receipt))
        receipts.append(receipt)
        if xml_ok:
            article_index.append({
                "doi": doi,
                "response_sha256": receipt["final_response_sha256"],
                "nct_ids_in_fulltext": ncts,
                "full_text_review_required": True,
                "task_label": None,
                "cohort_member": False,
            })
    report = {
        "schema": "plos-candidate-jats-capture-report-v1",
        "plan_sha256": plan["plan_sha256"],
        "complete": len(receipts) == len(canonical_dois) and all(row["valid_jats_for_requested_doi"] for row in receipts),
        "article_count": len(receipts),
        "valid_jats_count": len(article_index),
        "physical_requests": sum(row["physical_requests"] for row in receipts),
        "model_calls": 0,
        "articles": receipts,
        "review_index": article_index,
    }
    (output_dir / "report.json").write_bytes(_json_bytes(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doi", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = collect(args.doi, args.output_dir)
    print(json.dumps({key: report[key] for key in (
        "complete", "article_count", "valid_jats_count", "physical_requests", "model_calls"
    )}, ensure_ascii=False, sort_keys=True))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
