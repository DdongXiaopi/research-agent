"""Match Crossref update-to metadata against the frozen PLOS candidate manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def match_update_targets(candidate_dois: set[str], update_records: list[dict]) -> dict[str, list[dict]]:
    matches: dict[str, list[dict]] = {}
    for record in update_records:
        update_doi = str(record.get("update_doi") or "").casefold()
        for relation in record.get("update_to") or []:
            target = str(relation.get("DOI") or "").casefold()
            if target not in candidate_dois:
                continue
            match = {
                "update_doi": update_doi,
                "update_type": relation.get("type"),
                "label": relation.get("label"),
                "source": relation.get("source"),
                "updated": relation.get("updated"),
                "update_response_sha256": record.get("source_response_sha256"),
            }
            matches.setdefault(target, []).append(match)
    for target, items in matches.items():
        unique = {json.dumps(item, sort_keys=True, ensure_ascii=False): item for item in items}
        matches[target] = [unique[key] for key in sorted(unique)]
    return matches


def build_report(candidate_manifest_path: Path, crossref_capture_dir: Path) -> dict:
    manifest_path = candidate_manifest_path.resolve()
    capture = crossref_capture_dir.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan_path, receipt_path, index_path = capture / "plan.json", capture / "receipt.json", capture / "update-index.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    update_records = json.loads(index_path.read_text(encoding="utf-8"))
    unsigned_plan = {key: value for key, value in plan.items() if key != "plan_sha256"}
    expected_plan_hash = hashlib.sha256((json.dumps(
        unsigned_plan, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()).hexdigest()
    if plan.get("plan_sha256") != expected_plan_hash:
        raise ValueError("Crossref capture plan hash mismatch")
    if (receipt.get("state") != "completed"
            or receipt.get("plan_sha256") != plan.get("plan_sha256")
            or len(update_records) != plan.get("expected_total_records_from_count_probe")
            or receipt.get("records_captured") != len(update_records)):
        raise ValueError("Crossref capture is incomplete or does not match its plan")
    for page_index, page in enumerate(receipt.get("pages") or []):
        raw = capture / f"page-{page_index:02d}" / "response.bin"
        if _sha(raw) != page.get("response_sha256"):
            raise ValueError("Crossref raw page hash mismatch")
    candidates = {str(row["doi"]).casefold(): row for row in manifest["candidate_entries"]}
    matches = match_update_targets(set(candidates), update_records)
    output = []
    for doi, candidate in sorted(candidates.items()):
        links = matches.get(doi, [])
        types = {str(link.get("update_type") or "").casefold() for link in links}
        if "retraction" in types:
            disposition = "retraction_recorded_exclude_pending_primary_source_check"
        elif "correction" in types or "expression_of_concern" in types:
            disposition = "correction_or_concern_review_required"
        elif "new_version" in types:
            disposition = "new_version_review_required"
        elif links:
            disposition = "other_update_review_required"
        else:
            disposition = "no_crossref_update_found_not_status_proof"
        output.append({
            "doi": doi,
            "prior_source_disposition": candidate["disposition"],
            "update_disposition": disposition,
            "update_links": links,
            "candidate_label": None,
            "confirmation_eligible": False,
        })
    return {
        "schema": "plos-crossref-update-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_manifest_sha256": _sha(manifest_path),
        "crossref_plan_sha256": plan["plan_sha256"],
        "crossref_receipt_sha256": _sha(receipt_path),
        "crossref_update_index_sha256": _sha(index_path),
        "crossref_update_records": len(update_records),
        "candidate_doi_count": len(candidates),
        "candidate_dois_with_update_links": len(matches),
        "candidate_entries": output,
        "no_update_found_means_active": False,
        "current_status_fully_verified": False,
        "labels_generated": False,
        "confirmation_cohort_created": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--crossref-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(args.candidate_manifest, args.crossref_capture)
    output = args.output.resolve()
    try:
        output.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("Crossref audit output must stay under project .workbuddy") from exc
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Crossref audit report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")
    by_status: dict[str, int] = {}
    for row in report["candidate_entries"]:
        by_status[row["update_disposition"]] = by_status.get(row["update_disposition"], 0) + 1
    print(json.dumps({
        "status": "written",
        "output": str(output),
        "candidate_count": report["candidate_doi_count"],
        "with_update_links": report["candidate_dois_with_update_links"],
        "update_disposition_counts": by_status,
        "current_status_fully_verified": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
