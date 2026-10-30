#!/usr/bin/env python3
"""Build and write-once a zero-call v15 G06 packet (rubric-v7 / target-v10)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import instruction_reviewer_cohort_v15 as cohort  # noqa: E402
from src.eval import instruction_reviewer_reference_shards as shards  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=ROOT / ".workbuddy" / cohort.OUTPUT_DIRECTORY_ID / "packet.json",
        help="write once below a controlled .workbuddy directory")
    args = parser.parse_args(argv)
    output = shards._controlled_work_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen packet: {output}")
    packet = cohort.build_packet(ROOT)
    cohort.validate_packet_integrity(packet, ROOT)
    file_hash = shards._write_once(output, packet)
    print(json.dumps({
        "packet": str(output),
        "packet_sha256": packet["packet_sha256"],
        "packet_file_sha256": file_hash,
        "reference_protocol_hash": packet["reference_protocol"]["protocol_hash"],
        "reference_protocol_schema": packet["reference_protocol"]["protocol_schema"],
        "target_protocol_version": packet["target_protocol"]["version"],
        "target_protocol_hash": packet["target_protocol"]["protocol_hash"],
        "dynamic_schema_cases": len(packet["reference_packet_manifest"]),
        "case_count": packet["case_count"],
        "construction_cell_counts": packet["construction_protocol"]["cell_counts"],
        "prior_packet_count": packet["exposure_audit"]["prior_packet_count"],
        "exposed_candidate_text_count": packet["exposure_audit"]["exposed_candidate_text_count"],
        "overlap_count": packet["exposure_audit"]["overlap_count"],
        "model_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
