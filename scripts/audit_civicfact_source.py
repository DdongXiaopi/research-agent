#!/usr/bin/env python3
"""Audit CIViC-Fact source metadata without emitting claim/evidence text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.civicfact_source_audit import audit_civicfact_build  # noqa: E402
from src.util import write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="upstream JSONL or JSONL.GZ build")
    parser.add_argument("--output", type=Path,
                        help="optional path for the metadata-only audit JSON")
    args = parser.parse_args()
    report = audit_civicfact_build(args.source)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
