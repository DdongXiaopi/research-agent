#!/usr/bin/env python3
"""Audit CliniFact train/validation metadata without touching the test file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.clinifact_source_audit import audit_clinifact_source  # noqa: E402
from src.util import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path,
                        default=ROOT / "data/raw/clinifact/train_set.csv")
    parser.add_argument("--validation", type=Path,
                        default=ROOT / "data/raw/clinifact/validation_set.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/probe/clinifact_source_audit_20261008.json")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    audit = audit_clinifact_source(args.train, args.validation)
    if args.write:
        write_json(args.output, audit)
    print(json.dumps({
        "status": "written" if args.write else "validated_not_written",
        "path": str(args.output),
        "train": audit["splits"]["train"],
        "validation": audit["splits"]["validation"],
        "train_validation_group_overlap": audit["train_validation_group_overlap"],
        "official_test_downloaded_or_read": audit["official_test_downloaded_or_read"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
