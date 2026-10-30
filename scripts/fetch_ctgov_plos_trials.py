"""Capture ClinicalTrials.gov records for NCT IDs found in the frozen PLOS cohort."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_plos_candidate_jats import _parse_jats_sections
from src.data.plos_capacity_probe import normalize_abstract_text


ENDPOINT = "https://clinicaltrials.gov/api/v2/studies"
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
TIMEOUT_SECONDS = 45
PAGE_SIZE = 100


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_hash(value: dict, hash_key: str = "plan_sha256") -> None:
    unsigned = {key: item for key, item in value.items() if key != hash_key}
    if value.get(hash_key) != _sha(_json_bytes(unsigned)):
        raise ValueError(f"canonical hash mismatch for {hash_key}")


def collect(
    manifest_path: Path,
    candidate_index_path: Path,
    crossref_audit_path: Path,
    capacity_report_path: Path,
    output_dir: Path,
    jats_capture_paths: list[Path] | None = None,
) -> dict:
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("ClinicalTrials.gov capture must stay under project .workbuddy") from exc
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse ClinicalTrials.gov capture directory: {output_dir}")

    manifest = _load_json(manifest_path)
    source_rows = _load_json(candidate_index_path)
    source_by_doi = {str(row.get("doi", "")).casefold(): row for row in source_rows}
    if not source_by_doi:
        raise ValueError("PLOS source index is empty")
    crossref = _load_json(crossref_audit_path)
    capacity = _load_json(capacity_report_path)
    manifest_sha = _sha(manifest_path.read_bytes())
    source_index_sha = _sha(candidate_index_path.read_bytes())
    crossref_sha = _sha(crossref_audit_path.read_bytes())
    capacity_sha = _sha(capacity_report_path.read_bytes())
    if capacity.get("candidate_manifest_sha256") != manifest_sha:
        raise ValueError("capacity probe is not bound to the supplied candidate manifest")
    if capacity.get("crossref_audit_sha256") != crossref_sha:
        raise ValueError("capacity probe is not bound to the supplied Crossref audit")
    if crossref.get("candidate_manifest_sha256") != manifest_sha:
        raise ValueError("Crossref audit is not bound to the supplied candidate manifest")
    if capacity.get("task_labels_generated") is not False or capacity.get("cohort_or_label_created") is not False:
        raise ValueError("capacity probe already contains labels or a cohort")

    entries = {str(row.get("doi", "")).casefold(): row for row in manifest.get("candidate_entries", [])}
    crossref_entries = {str(row.get("doi", "")).casefold(): row for row in crossref.get("candidate_entries", [])}
    nct_by_doi = capacity.get("nct_by_doi") or {}
    mapping: dict[str, list[str]] = {}
    nct_ids: set[str] = set()
    for raw_doi, raw_ids in nct_by_doi.items():
        doi = str(raw_doi).casefold()
        entry, update = entries.get(doi), crossref_entries.get(doi)
        if entry is None or update is None:
            raise ValueError(f"capacity DOI absent from frozen candidate/Crossref indexes: {doi}")
        if entry.get("disposition") != "pending_retraction_and_trial_evidence_review":
            raise ValueError(f"candidate is not eligible for trial evidence review: {doi}")
        source = source_by_doi.get(doi)
        if (source is None or entry.get("source_row_index") is None
                or source.get("source_response_sha256") != entry.get("source_page_response_sha256")
                or source.get("title") is None or source.get("abstract") is None):
            raise ValueError(f"candidate does not match the frozen PLOS metadata index: {doi}")
        if update.get("update_disposition") != "no_crossref_update_found_not_status_proof":
            raise ValueError(f"candidate has a known Crossref update: {doi}")
        ids = sorted({str(value).upper() for value in raw_ids})
        if not ids or any(len(value) != 11 or not value.startswith("NCT") or not value[3:].isdigit()
                          for value in ids):
            raise ValueError(f"invalid NCT ID set for {doi}")
        mapping[doi] = ids
        nct_ids.update(ids)
    if len(nct_ids) != int(capacity.get("unique_nct_count", -1)):
        raise ValueError("abstract NCT count does not match the capacity report")
    jats_capture_hashes: list[str] = []
    fulltext_ids_by_doi: dict[str, set[str]] = {}
    body_ids_by_doi: dict[str, set[str]] = {}
    reference_ids_by_doi: dict[str, set[str]] = {}
    for capture_path in jats_capture_paths or []:
        capture_path = capture_path.resolve()
        plan_path, report_path = capture_path / "plan.json", capture_path / "report.json"
        plan, jats_report = _load_json(plan_path), _load_json(report_path)
        _verify_hash(plan)
        if jats_report.get("plan_sha256") != plan.get("plan_sha256") or not jats_report.get("complete"):
            raise ValueError(f"JATS capture is incomplete or not bound to its plan: {capture_path}")
        if jats_report.get("model_calls") != 0:
            raise ValueError("JATS source capture must not contain model calls")
        jats_capture_hashes.append(_sha(report_path.read_bytes()))
        for index, row in enumerate(jats_report.get("articles") or []):
            article_dir = capture_path / f"article-{index:02d}"
            raw = (article_dir / "response.xml").read_bytes()
            disk_receipt = _load_json(article_dir / "receipt.json")
            if (_sha(raw) != row.get("final_response_sha256")
                    or disk_receipt != row or not row.get("valid_jats_for_requested_doi")):
                raise ValueError(f"JATS source receipt or response mismatch: {article_dir}")
            nct_sections = _parse_jats_sections(raw, str(row.get("doi") or ""))
            if row.get("nct_ids_by_section") != nct_sections:
                raise ValueError(f"JATS section-level NCT index mismatch: {article_dir}")
            doi = str(row.get("doi") or "").casefold()
            if doi not in mapping:
                raise ValueError(f"JATS DOI is not an eligible candidate with explicit abstract NCT: {doi}")
            fulltext_ncts = nct_sections["trial_identity_candidates"]
            reference_ncts = nct_sections["references"]
            fulltext_ids_by_doi.setdefault(doi, set()).update(fulltext_ncts)
            body_ids_by_doi.setdefault(doi, set()).update(nct_sections["body"])
            reference_ids_by_doi.setdefault(doi, set()).update(reference_ncts)
            nct_ids.update(fulltext_ncts)
    if not nct_ids or len(nct_ids) > PAGE_SIZE:
        raise ValueError("NCT batch is empty or exceeds the frozen one-request page size")

    output_dir.mkdir(parents=True)
    plan = {
        "schema": "ctgov-plos-trial-capture-plan-v1",
        "endpoint": ENDPOINT,
        "params": {"filter.ids": ",".join(sorted(nct_ids)), "format": "json", "pageSize": PAGE_SIZE},
        "candidate_manifest_sha256": manifest_sha,
        "plos_source_index_sha256": source_index_sha,
        "crossref_audit_sha256": crossref_sha,
        "capacity_report_sha256": capacity_sha,
        "jats_capture_report_sha256s": jats_capture_hashes,
        "doi_to_abstract_nct_ids": dict(sorted(mapping.items())),
        "doi_to_nct_ids": {
            doi: sorted(set(ids) | fulltext_ids_by_doi.get(doi, set()))
            for doi, ids in sorted(mapping.items())
        },
        "doi_to_fulltext_nct_ids": {
            doi: sorted(ids) for doi, ids in sorted(fulltext_ids_by_doi.items())
        },
        "doi_to_reference_nct_ids": {
            doi: sorted(ids) for doi, ids in sorted(reference_ids_by_doi.items())
        },
        "doi_to_jats_body_nct_ids": {
            doi: sorted(ids) for doi, ids in sorted(body_ids_by_doi.items())
        },
        "expected_unique_nct_count": len(nct_ids),
        "max_physical_requests": 1,
        "max_response_bytes": MAX_RESPONSE_BYTES,
        "timeout_seconds": TIMEOUT_SECONDS,
        "redirects": 0,
        "retries": 0,
        "model_calls": 0,
        "purpose": "Retrieve first-party trial registry records for explicit NCT mentions; co-occurrence is not proof of trial identity or outcome-label correctness.",
        "collector_source_sha256": _sha(Path(__file__).read_bytes()),
    }
    plan["plan_sha256"] = _sha(_json_bytes(plan))
    (output_dir / "plan.json").write_bytes(_json_bytes(plan))

    started = datetime.now(timezone.utc).isoformat()
    request = Request(ENDPOINT + "?" + urlencode(plan["params"]), headers={
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
    (output_dir / "response.bin").write_bytes(body)
    error = None
    studies: list[dict] = []
    received_ids: list[str] = []
    missing_ids: list[str] = []
    unexpected_ids: list[str] = []
    if status != 200:
        error = f"http_status_{status}"
    elif len(body) > MAX_RESPONSE_BYTES:
        error = "response_byte_limit_exceeded"
    else:
        try:
            payload = json.loads(body)
            studies = payload["studies"]
            received_ids = [study["protocolSection"]["identificationModule"]["nctId"]
                            for study in studies]
            if len(received_ids) != len(set(received_ids)):
                raise ValueError("duplicate NCT ID in response")
            missing_ids = sorted(nct_ids - set(received_ids))
            unexpected_ids = sorted(set(received_ids) - nct_ids)
            if missing_ids or unexpected_ids:
                error = "response_nct_id_set_mismatch"
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            error = f"invalid_or_incomplete_response:{type(exc).__name__}"

    if status == 200 and len(body) <= MAX_RESPONSE_BYTES and studies:
        by_id = {
            study["protocolSection"]["identificationModule"]["nctId"]: study
            for study in studies
        }
        compact_studies = []
        for nct_id, study in sorted(by_id.items()):
            protocol = study.get("protocolSection", {})
            ident = protocol.get("identificationModule", {})
            design = protocol.get("designModule", {})
            outcomes = protocol.get("outcomesModule", {})
            status_module = protocol.get("statusModule", {})
            compact_studies.append({
                "nct_id": nct_id,
                "brief_title": ident.get("briefTitle"),
                "official_title": ident.get("officialTitle"),
                "study_type": design.get("studyType"),
                "phases": design.get("phases", []),
                "overall_status": status_module.get("overallStatus"),
                "primary_outcomes": outcomes.get("primaryOutcomes", []),
                "secondary_outcomes": outcomes.get("secondaryOutcomes", []),
                "registry_record_sha256": _sha(_json_bytes(study)),
            })
        (output_dir / "study-index.json").write_bytes(_json_bytes(compact_studies))
        candidate_review_rows = []
        for doi, ids in sorted(mapping.items()):
            article = source_by_doi[doi]
            linked = []
            all_ids = sorted(set(ids) | fulltext_ids_by_doi.get(doi, set()))
            for nct_id in all_ids:
                study = by_id.get(nct_id)
                if study is None:
                    linked.append({
                        "nct_id": nct_id,
                        "registry_lookup_status": "not_returned_by_batch",
                        "registry_record_sha256": None,
                    })
                    continue
                protocol = study.get("protocolSection", {})
                ident = protocol.get("identificationModule", {})
                design = protocol.get("designModule", {})
                outcomes = protocol.get("outcomesModule", {})
                status_module = protocol.get("statusModule", {})
                linked.append({
                    "nct_id": nct_id,
                    "registry_lookup_status": "returned_by_batch",
                    "brief_title": ident.get("briefTitle"),
                    "official_title": ident.get("officialTitle"),
                    "study_type": design.get("studyType"),
                    "phases": design.get("phases", []),
                    "overall_status": status_module.get("overallStatus"),
                    "primary_outcomes": outcomes.get("primaryOutcomes", []),
                    "secondary_outcomes": outcomes.get("secondaryOutcomes", []),
                    "registry_record_sha256": _sha(_json_bytes(study)),
                })
            candidate_review_rows.append({
                "doi": doi,
                "article_title": article.get("title"),
                "article_abstract": normalize_abstract_text(article.get("abstract")),
                "explicit_nct_ids_in_abstract": ids,
                "nct_ids_found_in_jats_fulltext": sorted(fulltext_ids_by_doi.get(doi, set())),
                "nct_ids_found_in_references": sorted(reference_ids_by_doi.get(doi, set())),
                "identity_review_status": (
                    "conflicting_abstract_body_nct_ids"
                    if set(ids) and body_ids_by_doi.get(doi)
                    and set(ids) != body_ids_by_doi[doi]
                    else "explicit_nct_id_present_registry_link_pending"
                ),
                "registry_records": linked,
                "disposition": "pending_full_article_trial_identity_and_outcome_adjudication",
                "task_label": None,
                "cohort_member": False,
            })
        (output_dir / "candidate-trial-index.json").write_bytes(_json_bytes(candidate_review_rows))
    receipt = {
        "schema": "ctgov-plos-trial-capture-receipt-v1",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": plan["plan_sha256"],
        "http_status": status,
        "response_bytes": len(body),
        "response_sha256": _sha(body),
        "expected_nct_count": len(nct_ids),
        "returned_study_count": len(studies),
        "returned_unique_nct_count": len(set(received_ids)),
        "complete_exact_id_match": error is None,
        "missing_nct_ids": missing_ids,
        "unexpected_nct_ids": unexpected_ids,
        "candidate_count_with_explicit_nct": len(mapping) if studies else 0,
        "labels_generated": False,
        "confirmation_cohort_created": False,
        "model_calls": 0,
        "error": error,
    }
    (output_dir / "receipt.json").write_bytes(_json_bytes(receipt))
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--crossref-audit", type=Path, required=True)
    parser.add_argument("--capacity-report", type=Path, required=True)
    parser.add_argument("--jats-capture", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = collect(
        args.manifest, args.candidate_index, args.crossref_audit, args.capacity_report, args.output_dir,
        args.jats_capture,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0 if receipt["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
