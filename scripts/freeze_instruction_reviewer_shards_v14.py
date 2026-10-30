#!/usr/bin/env python3
"""Verify two completed v14 reviewer aggregates and freeze v12 references once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import instruction_reviewer_cohort_v14 as cohort  # noqa: E402
from src.eval import instruction_reviewer_sharded_freeze as freeze  # noqa: E402
from src.util import read_json  # noqa: E402


def _read_json_document(path: Path, label: str) -> dict:
    lexical = Path(path).expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise ValueError(f"{label} must be an existing regular file")
    value = read_json(lexical)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--schedule-a", type=Path, required=True)
    parser.add_argument("--state-a", type=Path, required=True)
    parser.add_argument("--aggregate-a", type=Path, required=True)
    parser.add_argument("--schedule-b", type=Path, required=True)
    parser.add_argument("--state-b", type=Path, required=True)
    parser.add_argument("--aggregate-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new frozen-reference JSON path below .workbuddy")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_inputs(parser)
    args = parser.parse_args(argv)

    paths = {
        "packet": args.packet,
        "schedule_a": args.schedule_a,
        "aggregate_a": args.aggregate_a,
        "schedule_b": args.schedule_b,
        "aggregate_b": args.aggregate_b,
    }
    docs = {name: _read_json_document(path, name) for name, path in paths.items()}
    packet = docs["packet"]
    if packet.get("schema") != cohort.SCHEMA:
        raise ValueError("v14 freeze CLI accepts only the current v14 cohort packet")

    output = args.output.expanduser().absolute()
    if output.resolve(strict=False) in {
            path.expanduser().resolve() for path in paths.values()}:
        raise ValueError("frozen-reference output must not overwrite an input document")
    state_a = str(args.state_a.expanduser().resolve())
    state_b = str(args.state_b.expanduser().resolve())

    frozen = freeze.freeze_synthetic_diagnostic_reference(
        packet, docs["schedule_a"], state_a, docs["aggregate_a"],
        docs["schedule_b"], state_b, docs["aggregate_b"])
    if (frozen.get("schema") != freeze.FROZEN_SCHEMA_V12
            or frozen.get("packet_sha256") != packet.get("packet_sha256")
            or frozen.get("synthetic_only") is not True
            or frozen.get("natural_distribution") is not False
            or frozen.get("promotion_authorized") is not False):
        raise ValueError("v14 freeze must produce a correctly scoped frozen v12 reference")
    output_file_sha256 = freeze.write_synthetic_diagnostic_reference_once(output, frozen)
    print(json.dumps({
        "action": "freeze-v14-references",
        "output": str(output),
        "output_file_sha256": output_file_sha256,
        "frozen_reference_sha256": frozen["frozen_reference_sha256"],
        "packet_sha256": packet["packet_sha256"],
        "schema": frozen["schema"],
        "case_count": len(frozen.get("cases", [])),
        "disagreement_count": len(frozen.get("disagreements", [])),
        "reference_class_coverage": frozen.get("reference_class_coverage"),
        "reference_class_coverage_sufficient": frozen.get(
            "reference_class_coverage_sufficient"),
        "synthetic_only": True,
        "promotion_authorized": False,
        "model_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
