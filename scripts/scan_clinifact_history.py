#!/usr/bin/env python3
"""Exact-match screen CliniFact train/validation against prior project artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.clinifact_history_scan import scan_clinifact_history  # noqa: E402
from src.util import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path,
                        default=ROOT / "data/raw/clinifact/train_set.csv")
    parser.add_argument("--validation", type=Path,
                        default=ROOT / "data/raw/clinifact/validation_set.csv")
    parser.add_argument("--history-root", type=Path, action="append",
                        help="root to scan; repeat to union multiple independent history roots")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/probe/clinifact_history_scan_20261008.json")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    history_roots = args.history_root or [ROOT / "artifacts"]
    report = scan_clinifact_history(
        args.train, args.validation, history_roots)
    if args.write:
        write_json(args.output, report)
    print(json.dumps({
        "status": "written" if args.write else "validated_not_written",
        "path": str(args.output),
        "pattern_count": report["pattern_count"],
        "files_scanned": report["files_scanned"],
        "bytes_scanned": report["bytes_scanned"],
        "matched_rows_by_kind": report["matched_rows_by_kind"],
        "matched_rows_total": report["matched_rows_total"],
        "complete": report["complete"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
