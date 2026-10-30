#!/usr/bin/env python3
"""Build a zero-call, immutable G06 rubric-v5 challenge packet (v7)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import instruction_reviewer_reference_shards as shards  # noqa: E402
from src.eval.instruction_reviewer_cohort_v7 import (  # noqa: E402
    OUTPUT_DIRECTORY_ID, build_packet, validate_packet,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=ROOT / ".workbuddy" / OUTPUT_DIRECTORY_ID / "packet.json",
        help="write once below a controlled .workbuddy directory")
    args = parser.parse_args(argv)

    output = shards._controlled_work_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen packet: {output}")
    packet = build_packet(ROOT)
    validate_packet(packet, ROOT)
    packet_file_sha256 = shards._write_once(output, packet)
    audit = packet["exposure_audit"]
    print(json.dumps({
        "packet": str(output),
        "packet_sha256": packet["packet_sha256"],
        "packet_file_sha256": packet_file_sha256,
        "protocol_hash": packet["reference_protocol"]["protocol_hash"],
        "splitter_hash": packet["reference_protocol"]["splitter_hash"],
        "case_count": packet["case_count"],
        "construction_cell_counts": packet["construction_protocol"]["cell_counts"],
        "prior_packet_count": audit["prior_packet_count"],
        "exposed_candidate_text_count": audit["exposed_candidate_text_count"],
        "overlap_count": audit["overlap_count"],
        "model_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
