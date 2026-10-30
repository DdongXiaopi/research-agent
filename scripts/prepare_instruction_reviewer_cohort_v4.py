"""Build a zero-call, immutable rubric-v2 instruction-reviewer challenge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.instruction_reviewer_cohort_v4 import (  # noqa: E402
    OUTPUT_DIRECTORY_ID, build_packet, validate_packet,
)
from src.util import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / ".workbuddy" / OUTPUT_DIRECTORY_ID / "packet.json")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen packet: {output}")
    packet = build_packet(ROOT)
    validate_packet(packet, ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, packet)
    print(json.dumps({
        "packet": str(output),
        "packet_sha256": packet["packet_sha256"],
        "rubric_version": packet["reference_protocol"]["rubric_version"],
        "rubric_sha256": packet["reference_protocol"]["rubric_sha256"],
        "case_count": packet["case_count"],
        "prior_packet_count": packet["exposure_audit"]["prior_packet_count"],
        "exposed_candidate_text_count": packet["exposure_audit"]["exposed_candidate_text_count"],
        "overlap_count": packet["exposure_audit"]["overlap_count"],
        "model_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
