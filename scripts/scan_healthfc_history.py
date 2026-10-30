#!/usr/bin/env python3
"""Scan prior project artifacts for exact HealthFC claim/evidence exposure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.healthfc_history_scan import scan_healthfc_history  # noqa: E402
from src.util import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="HealthFC CSV source")
    parser.add_argument("--history-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = scan_healthfc_history(args.source, args.history_root)
    write_json(args.output, report)
    print(json.dumps({
        "complete": report["complete"],
        "files_scanned": report["files_scanned"],
        "bytes_scanned": report["bytes_scanned"],
        "claim_rows_with_exact_prior_phrase": report[
            "claim_rows_with_exact_prior_phrase"],
        "evidence_rows_with_exact_prior_phrase": report[
            "evidence_rows_with_exact_prior_phrase"],
        "rows_with_any_exact_prior_phrase": report[
            "rows_with_any_exact_prior_phrase"],
        "files_with_any_exact_prior_phrase": report[
            "files_with_any_exact_prior_phrase"],
        "unreadable_files": report["unreadable_files"],
        "report": str(args.output),
    }, ensure_ascii=False))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
