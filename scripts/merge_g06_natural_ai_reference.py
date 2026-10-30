#!/usr/bin/env python3
"""Merge historical G06 references with a blinded AI natural-source supplement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.natural_ai_reference_merge import merge_references, validate_merged_reference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-packet", type=Path, required=True)
    parser.add_argument("--historical-reference", type=Path, required=True)
    parser.add_argument("--external-packet", type=Path, required=True)
    parser.add_argument("--external-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("merged G06 reference is write-once")
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    merged = merge_references(read(args.historical_packet), read(args.historical_reference),
                              read(args.external_packet), read(args.external_reference))
    validate_merged_reference(merged)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "merged_ai_capacity_reference",
                      "output": str(output),
                      "reference_sha256": merged["reference_sha256"],
                      "case_count": merged["case_count"],
                      "coverage": merged["reference_class_coverage"],
                      "reference_class_coverage_sufficient": merged["reference_class_coverage_sufficient"],
                      "production_gate_eligible": merged["production_gate_eligible"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
