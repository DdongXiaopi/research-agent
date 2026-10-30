#!/usr/bin/env python3
"""Build a zero-call immutable span-atomic G06 challenge packet (v10)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import instruction_reviewer_cohort_v10 as cohort  # noqa: E402
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
    cohort.validate_packet(packet, ROOT)
    file_hash = shards._write_once(output, packet)
    print(json.dumps({
        "packet": str(output), "packet_sha256": packet["packet_sha256"],
        "packet_file_sha256": file_hash,
        "reference_protocol_hash": packet["reference_protocol"]["protocol_hash"],
        "target_protocol_version": packet["target_protocol"]["version"],
        "target_protocol_response_format_sha256": packet["target_protocol"]["response_format_sha256"],
        "case_count": packet["case_count"],
        "construction_cell_counts": packet["construction_protocol"]["cell_counts"],
        "prior_packet_count": packet["exposure_audit"]["prior_packet_count"],
        "exposed_candidate_text_count": packet["exposure_audit"]["exposed_candidate_text_count"],
        "overlap_count": packet["exposure_audit"]["overlap_count"], "model_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
