"""Scan historical project text for exact reuse of a frozen PLOS candidate set."""

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

from src.data.plos_history_scan import scan_plos_history  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, value: dict) -> None:
    path = path.resolve()
    try:
        path.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("history audit output must stay under project's .workbuddy") from exc
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-root", type=Path, action="append", default=[],
        help="additional source-only metadata roots to omit to avoid self-matches; may repeat",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, action="append")
    args = parser.parse_args(argv)
    capture = args.capture_dir.resolve()
    plan_path = capture / "plan.json"
    receipt_path = capture / "receipt.json"
    index_path = capture / "candidate-index.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    records = json.loads(index_path.read_text(encoding="utf-8"))
    if (receipt.get("state") != "completed_metadata_only"
            or receipt.get("plan_sha256") != plan.get("plan_sha256")
            or receipt.get("records_captured") != len(records)
            or receipt.get("unique_doi_count") != len(records)):
        raise ValueError("capture artifacts do not form a complete, unique metadata set")
    roots = args.history_root or [
        ROOT / "artifacts", ROOT / "data", ROOT / "docs", ROOT / "outputs", ROOT / ".workbuddy"]
    extra_excludes = [path.resolve() for path in args.exclude_root]
    # Exclusions are permitted only for source-discovery metadata stored under
    # .workbuddy. Never let a caller hide task traces, evaluations, or reports
    # from the historical exposure scan.
    workbuddy = (ROOT / ".workbuddy").resolve()
    for excluded in extra_excludes:
        try:
            excluded.relative_to(workbuddy)
        except ValueError as exc:
            raise ValueError("additional exclusions must be inside .workbuddy") from exc
    report = scan_plos_history(records, roots, exclude_roots=[capture, *extra_excludes])
    report.update({
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_capture_plan_sha256": plan["plan_sha256"],
        "source_capture_receipt_sha256": _sha(receipt_path),
        "source_candidate_index_sha256": _sha(index_path),
        "source_capture_dir": str(capture),
        "candidate_set_sha256": hashlib.sha256(
            json.dumps(records, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest(),
        "promotion_or_confirmation_authorized": False,
    })
    _write_new(args.output, report)
    print(json.dumps({
        "status": "written" if report["complete"] else "written_incomplete",
        "output": str(args.output.resolve()),
        "candidate_count": report["candidate_count"],
        "files_scanned": report["files_scanned"],
        "bytes_scanned": report["bytes_scanned"],
        "overlap_counts": {key: len(value) for key, value in report["overlap_by_kind"].items()},
        "complete": report["complete"],
    }, ensure_ascii=False))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
