"""Join immutable PLOS metadata captures and apply source-only candidate gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.plos_clinical_intake import (
    EXCLUDED_TITLE_TERMS,
    classify_plos_copyright_statement,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _load_capture(root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    plan_path, receipt_path, index_path = root / "plan.json", root / "receipt.json", root / "candidate-index.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != _canonical_hash(unsigned):
        raise ValueError(f"capture plan hash mismatch: {root}")
    if (receipt.get("state") != "completed_metadata_only"
            or receipt.get("plan_sha256") != plan.get("plan_sha256")
            or receipt.get("records_captured") != len(rows)
            or receipt.get("unique_doi_count") != len(rows)):
        raise ValueError(f"capture receipt/index mismatch: {root}")
    response_hashes: set[str] = set()
    for page_index, page in enumerate(receipt.get("pages") or []):
        response = root / f"page-{page_index:02d}" / "response.bin"
        if _sha(response) != page.get("response_sha256"):
            raise ValueError(f"raw source response hash mismatch: {response}")
        response_hashes.add(page["response_sha256"])
    if not rows or any(row.get("source_response_sha256") not in response_hashes for row in rows):
        raise ValueError(f"candidate rows are not bound to raw source pages: {root}")
    return plan, receipt, rows


def build_candidate_manifest(
    capture_v1: Path, capture_v2: Path, history_report_path: Path,
) -> dict[str, Any]:
    plan1, receipt1, rows1 = _load_capture(capture_v1)
    plan2, receipt2, rows2 = _load_capture(capture_v2)
    history = json.loads(history_report_path.read_text(encoding="utf-8"))
    if not history.get("complete"):
        raise ValueError("historical scan is incomplete")
    if history.get("source_candidate_index_sha256") != _sha(capture_v1 / "candidate-index.json"):
        raise ValueError("historical scan is not bound to capture v1")
    if history.get("source_capture_plan_sha256") != plan1.get("plan_sha256"):
        raise ValueError("historical scan plan identity mismatch")
    if len(rows1) != len(rows2):
        raise ValueError("capture row counts differ")
    fields = ("doi", "title", "abstract", "publication_date", "article_type")
    if any(any(left.get(field) != right.get(field) for field in fields)
           for left, right in zip(rows1, rows2)):
        raise ValueError("metadata changed between v1 and copyright capture")
    if receipt2["pages"]:
        min_gap = int(plan2.get("minimum_request_start_interval_seconds", 0))
        starts = [datetime.fromisoformat(page["request_started_at"]) for page in receipt2["pages"]]
        if any((right - left).total_seconds() + 0.1 < min_gap
               for left, right in zip(starts, starts[1:])):
            raise ValueError("copyright capture violated frozen API pacing")

    history_overlap = history.get("overlap_candidates") or {}
    reviewed: list[dict[str, Any]] = []
    exclusion_counts: dict[str, int] = {}
    license_counts: dict[str, int] = {}
    for index, row in enumerate(rows2):
        reasons: list[str] = []
        article_type = " ".join(str(row.get("article_type") or "").casefold().split())
        title = " ".join(str(row.get("title") or "").casefold().split())
        license_class = classify_plos_copyright_statement(
            str(row.get("copyright_statement") or ""))
        if article_type != "research article":
            reasons.append("not_primary_research_article")
        if any(term in title for term in EXCLUDED_TITLE_TERMS):
            reasons.append("excluded_title_type")
        if license_class is None:
            reasons.append("article_specific_license_unrecognized")
        if history_overlap.get(str(row.get("doi") or "").casefold()):
            reasons.append("exact_project_history_overlap")
        for reason in reasons:
            exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
        if license_class:
            license_counts[license_class] = license_counts.get(license_class, 0) + 1
        reviewed.append({
            "source_row_index": index,
            "doi": row["doi"],
            "source_page_response_sha256": row["source_response_sha256"],
            "copyright_statement_sha256": hashlib.sha256(
                str(row.get("copyright_statement") or "").encode("utf-8")).hexdigest(),
            "license_classification": license_class,
            "article_type": row.get("article_type"),
            "history_overlap_kinds": sorted(history_overlap.get(str(row.get("doi") or "").casefold(), [])),
            "disposition": "exclude" if reasons else "pending_retraction_and_trial_evidence_review",
            "reasons": sorted(reasons),
            "task_label": None,
            "confirmation_eligible": False,
        })
    selected = [row for row in reviewed if not row["reasons"]]
    return {
        "schema": "plos-clinical-source-candidate-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "capture_v1_plan_sha256": plan1["plan_sha256"],
        "capture_v1_receipt_sha256": _sha(capture_v1 / "receipt.json"),
        "capture_v1_index_sha256": _sha(capture_v1 / "candidate-index.json"),
        "capture_v2_plan_sha256": plan2["plan_sha256"],
        "capture_v2_receipt_sha256": _sha(capture_v2 / "receipt.json"),
        "capture_v2_index_sha256": _sha(capture_v2 / "candidate-index.json"),
        "history_report_sha256": _sha(history_report_path),
        "candidate_set_sha256": hashlib.sha256(json.dumps(
            reviewed, sort_keys=True, ensure_ascii=False,
            separators=(",", ":")).encode()).hexdigest(),
        "source_record_count": len(reviewed),
        "license_classification_counts": dict(sorted(license_counts.items())),
        "exclusion_counts_nonexclusive": dict(sorted(exclusion_counts.items())),
        "remaining_for_retraction_and_trial_evidence_review": len(selected),
        "labels_generated": False,
        "ai_labeling_started": False,
        "confirmation_cohort_created": False,
        "candidate_entries": reviewed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-v1", required=True, type=Path)
    parser.add_argument("--capture-v2", required=True, type=Path)
    parser.add_argument("--history-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = build_candidate_manifest(args.capture_v1, args.capture_v2, args.history_report)
    output = args.output.resolve()
    try:
        output.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("candidate manifest must stay under project's .workbuddy") from exc
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "status": "written",
        "output": str(output),
        "records": report["source_record_count"],
        "license_counts": report["license_classification_counts"],
        "exclusion_counts_nonexclusive": report["exclusion_counts_nonexclusive"],
        "pending_trial_evidence_review": report["remaining_for_retraction_and_trial_evidence_review"],
        "labels_generated": False,
        "confirmation_cohort_created": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
