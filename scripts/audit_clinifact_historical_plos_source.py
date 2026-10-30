"""Offline history/license/indicator audit for frozen PLOS metadata candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.plos_capacity_probe import summarize_plos_capacity  # noqa: E402
from src.data.plos_clinical_intake import (  # noqa: E402
    EXCLUDED_TITLE_TERMS, classify_plos_copyright_statement,
)
from src.data.plos_history_scan import scan_plos_history  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def summarize_source_gates(rows: list[dict], overlap: dict[str, list[str]]) -> dict:
    exclusion = Counter()
    licenses = Counter()
    eligible_dois = set()
    nct_to_dois: dict[str, set[str]] = defaultdict(set)
    all_dois = {str(row["doi"]).casefold() for row in rows}
    all_capacity = summarize_plos_capacity(rows, all_dois)
    for row in rows:
        doi = str(row["doi"]).casefold()
        reasons = []
        license_class = classify_plos_copyright_statement(row.get("copyright_statement", ""))
        if license_class:
            licenses[license_class] += 1
        else:
            reasons.append("article_license_not_explicitly_recognized")
        article_type = " ".join(str(row.get("article_type") or "").casefold().split())
        title = " ".join(str(row.get("title") or "").casefold().split())
        if article_type != "research article":
            reasons.append("not_research_article")
        if any(term in title for term in EXCLUDED_TITLE_TERMS):
            reasons.append("excluded_title_term")
        if overlap.get(doi):
            reasons.append("exact_project_history_overlap")
        for reason in reasons:
            exclusion[reason] += 1
        if not reasons:
            eligible_dois.add(doi)
        for nct in all_capacity.get("nct_by_doi", {}).get(doi, []):
            nct_to_dois[nct].add(doi)
    clean_capacity = summarize_plos_capacity(rows, eligible_dois)
    return {
        "unique_doi_count": len(all_dois),
        "unique_nct_count_before_source_gates": all_capacity["unique_nct_count"],
        "metadata_indicators_before_source_gates": all_capacity["indicator_counts"],
        "source_license_counts": dict(sorted(licenses.items())),
        "nonexclusive_exclusion_counts": dict(sorted(exclusion.items())),
        "records_remaining_for_retraction_and_trial_evidence_review": len(eligible_dois),
        "unique_nct_count_after_license_type_title_history_gates": clean_capacity["unique_nct_count"],
        "metadata_indicators_after_license_type_title_history_gates": clean_capacity["indicator_counts"],
        "nct_mapped_to_multiple_dois_count_after_gates": sum(
            len(values) > 1 for values in nct_to_dois.values()
            if any(doi in eligible_dois for doi in values)),
        "task_class_counts": {"EVIDENCE": 0, "INCONCLUSIVE": 0, "NEI": 0},
    }


def audit(capture_dir: Path, *, history_roots: list[Path] | None = None) -> dict:
    capture_dir = capture_dir.resolve()
    plan_path = capture_dir / "plan.json"
    report_path = capture_dir / "report.json"
    index_path = capture_dir / "metadata-index.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    unsigned = dict(plan)
    plan_sha = unsigned.pop("plan_sha256", None)
    if (not plan_sha or hashlib.sha256(_canonical(unsigned)).hexdigest() != plan_sha
            or plan.get("source_code_sha256") != _sha(ROOT / "scripts/fetch_clinifact_historical_plos_metadata.py")
            or plan.get("capacity_receipt_sha256") != _sha(ROOT / ".workbuddy/clinifact-plos-historical-capacity-20261017-v1/receipt.json")
            or source_report.get("plan_sha256") != plan_sha
            or source_report.get("status") != "metadata_capture_complete"
            or source_report.get("report_source_sha256") != plan.get("source_code_sha256")
            or len(rows) != plan.get("expected_num_found")
            or source_report.get("article_count") != len(rows)
            or source_report.get("unique_doi_count") != len(rows)):
        raise ValueError("capture plan/report/index provenance is invalid")
    page_hashes = set()
    for page_no, page in enumerate(source_report.get("pages") or []):
        body_path = capture_dir / f"page-{page_no:02d}" / "response.bin"
        if _sha(body_path) != page.get("response_sha256"):
            raise ValueError("raw PLOS response hash mismatch")
        if page.get("http_status") != 200:
            raise ValueError("capture contains an unsuccessful page")
        page_hashes.add(page["response_sha256"])
    if not page_hashes or any(row.get("source_page_response_sha256") not in page_hashes for row in rows):
        raise ValueError("metadata rows are not bound to raw source pages")
    roots = [Path(p).resolve() for p in (history_roots or [
        ROOT / "artifacts", ROOT / "data", ROOT / "docs", ROOT / "outputs", ROOT / ".workbuddy"])]
    history = scan_plos_history(rows, roots, exclude_roots=[capture_dir])
    overlap = history.get("overlap_candidates") or {}
    gates = summarize_source_gates(rows, overlap)
    history["source_capture_plan_sha256"] = plan_sha
    history["source_capture_report_sha256"] = _sha(report_path)
    history["source_metadata_index_sha256"] = _sha(index_path)
    return {
        "schema": "clinifact-plos-historical-source-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_capture_dir": str(capture_dir),
        "source_capture_plan_sha256": plan_sha,
        "source_capture_report_sha256": _sha(report_path),
        "source_metadata_index_sha256": _sha(index_path),
        "source_article_count": len(rows),
        **gates,
        "exact_history_scan": history,
        "class_counts": gates["task_class_counts"],
        "source_components_with_verified_trial_outcome_identity": 0,
        "retraction_correction_review_completed": False,
        "trial_outcome_identity_review_completed": False,
        "labels_generated": False,
        "ai_reference_started": False,
        "confirmation_cohort_created": False,
        "confirmation_eligible": False,
        "promotion_or_candidate_search_authorized": False,
        "limitations": [
            "PLOS index counts and abstract markers are discovery indicators only.",
            "Crossref update/retraction status has not been checked for this historical DOI range.",
            "NCT links, primary outcome identity, statistical relation, and task labels have not been reviewed.",
            "Exact text history scanning does not prove semantic independence.",
        ],
        "audit_source_sha256": _sha(Path(__file__)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        output.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("audit output must stay under project .workbuddy") from exc
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output}")
    report = audit(args.capture_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(_canonical(report))
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        "status": "complete" if report["exact_history_scan"]["complete"] else "incomplete",
        "output": str(output),
        "source_article_count": report["source_article_count"],
        "history_files_scanned": report["exact_history_scan"]["files_scanned"],
        "history_bytes_scanned": report["exact_history_scan"]["bytes_scanned"],
        "history_overlap_doi_count": len(report["exact_history_scan"]["overlap_candidates"]),
        "license_counts": report["source_license_counts"],
        "remaining_for_article_evidence_review": report["records_remaining_for_retraction_and_trial_evidence_review"],
        "class_counts": report["class_counts"],
        "confirmation_eligible": report["confirmation_eligible"],
    }, ensure_ascii=False))
    return 0 if report["exact_history_scan"]["complete"] else 1


if __name__ == "__main__":
    import os
    raise SystemExit(main())
