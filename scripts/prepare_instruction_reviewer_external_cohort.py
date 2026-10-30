#!/usr/bin/env python3
"""Build a zero-call external instruction-reviewer packet.

The input is a compact extract prepared from Natural Instructions, MUFFIN
rematching, or LLMBar.  This command only normalizes records and writes an
immutable packet; it never contacts a model and never treats upstream labels
as reviewer gold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.instruction_reviewer_external_source import (  # noqa: E402
    SOURCE_POLICIES,
    build_packet,
    file_source_manifest,
    read_records,
    validate_packet,
)


def _write_once(path: Path, value: object) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"refusing to overwrite immutable output: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="compact JSON/JSONL external instruction extract")
    parser.add_argument("--source-kind", choices=sorted(SOURCE_POLICIES), required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--prior-packet", type=Path, action="append", default=[],
                        help="previous packet to exact-deduplicate; may repeat")
    args = parser.parse_args(argv)
    if args.max_cases is not None and args.max_cases <= 0:
        parser.error("--max-cases must be positive")
    records = read_records(args.input)
    manifest = file_source_manifest(args.input, source_url=args.source_url,
                                    repository=args.repository,
                                    revision=args.revision)
    prior_packets = []
    for prior_path in args.prior_packet:
        try:
            prior = json.loads(prior_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read prior packet: {prior_path}") from exc
        prior_packets.append((prior_path.as_posix(), prior))
    packet = build_packet(records, source_kind=args.source_kind,
                          source_manifest=manifest, max_cases=args.max_cases,
                          prior_packets=prior_packets)
    validate_packet(packet)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_once(output, packet)
    print(json.dumps({"status": "planned_zero_call",
                      "packet": str(output),
                      "packet_sha256": packet["packet_sha256"],
                      "source_kind": packet["source_kind"],
                      "case_count": packet["case_count"],
                      "natural_distribution": packet["natural_distribution"],
                      "production_gate_eligible": packet["production_gate_eligible"],
                      "model_calls": 0, "network_calls": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
