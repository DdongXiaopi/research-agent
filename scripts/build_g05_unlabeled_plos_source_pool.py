"""Freeze a rights-screened, history-screened but still unlabeled G05 source pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.g05_source_pool import build_g05_unlabeled_source_pool  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256((json.dumps(
        value, sort_keys=True, ensure_ascii=False, indent=2
    ) + "\n").encode("utf-8")).hexdigest()


def _load_verified_capture(capture: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    plan_path = capture / "plan.json"
    receipt_path = capture / "receipt.json"
    index_path = capture / "candidate-index.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    if plan.get("plan_sha256") != _canonical_sha(
            {key: value for key, value in plan.items() if key != "plan_sha256"}):
        raise ValueError("source capture plan hash mismatch")
    if (receipt.get("state") != "completed_metadata_only"
            or receipt.get("plan_sha256") != plan.get("plan_sha256")
            or receipt.get("records_captured") != len(rows)
            or receipt.get("unique_doi_count") != len(rows)):
        raise ValueError("source capture receipt does not bind the index")
    response_hashes: set[str] = set()
    for page_index, page in enumerate(receipt.get("pages") or []):
        response_path = capture / f"page-{page_index:02d}" / "response.bin"
        if _sha(response_path) != page.get("response_sha256"):
            raise ValueError("raw source response hash mismatch")
        response_hashes.add(str(page.get("response_sha256")))
    if not rows or any(row.get("source_response_sha256") not in response_hashes for row in rows):
        raise ValueError("source index rows are not bound to raw response pages")
    return plan, receipt, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--history-report", type=Path, required=True)
    parser.add_argument("--crossref-report", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    capture = args.capture_dir.resolve()
    plan, receipt, rows = _load_verified_capture(capture)
    history = json.loads(args.history_report.read_text(encoding="utf-8"))
    crossref = json.loads(args.crossref_report.read_text(encoding="utf-8"))
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    report = build_g05_unlabeled_source_pool(
        rows, history, crossref, manifest,
        source_index_sha256=_sha(capture / "candidate-index.json"),
        source_capture_plan_sha256=plan["plan_sha256"],
        source_capture_receipt_sha256=_sha(capture / "receipt.json"),
        source_manifest_file_sha256=_sha(args.source_manifest),
        history_report_file_sha256=_sha(args.history_report),
        crossref_report_file_sha256=_sha(args.crossref_report),
    )
    output = args.output.resolve()
    try:
        output.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("frozen source pool output must stay under .workbuddy") from exc
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen source pool: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "status": "frozen_unlabeled_source_pool",
        "output": str(output),
        "candidate_count": report["candidate_count"],
        "independent_source_group_count": report["independent_source_group_count"],
        "license_counts": report["license_counts"],
        "labels_generated": False,
        "target_scorer_calls": 0,
        "confirmation_eligible": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
