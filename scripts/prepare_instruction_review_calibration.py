"""Freeze a zero-call synthetic challenge packet and blind annotation template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.instruction_review_calibration import (  # noqa: E402
    build_blind_annotation_template, build_blind_review_copy, build_calibration_packet,
    validate_annotation_template, validate_calibration_packet,
)
from src.util import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--annotations", type=Path,
                        help="combined blank template (legacy; prefer two separate copies)")
    parser.add_argument("--reviewer-a", type=Path,
                        help="separate reviewer A copy; requires --reviewer-b")
    parser.add_argument("--reviewer-b", type=Path,
                        help="separate reviewer B copy; requires --reviewer-a")
    args = parser.parse_args(argv)
    packet = build_calibration_packet()
    validate_calibration_packet(packet)
    if bool(args.reviewer_a) != bool(args.reviewer_b):
        parser.error("--reviewer-a and --reviewer-b must be supplied together")
    if bool(args.annotations) == bool(args.reviewer_a):
        parser.error("provide --annotations or the pair --reviewer-a/--reviewer-b")
    if args.annotations:
        copies = [(args.annotations, build_blind_annotation_template(packet))]
        validate_annotation_template(copies[0][1], packet)
    else:
        copies = [
            (args.reviewer_a, build_blind_review_copy(packet, "reviewer_a")),
            (args.reviewer_b, build_blind_review_copy(packet, "reviewer_b")),
        ]
    targets = (args.packet, *(path for path, _ in copies))
    if len({str(target.resolve()).casefold() for target in targets}) != len(targets):
        parser.error("packet and reviewer outputs must use distinct paths")
    for target in targets:
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing file: {target}")
    args.packet.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.packet, packet)
    for target, template in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, template)
    print(json.dumps({
        "packet": str(args.packet),
        "reviewer_a": str(args.reviewer_a),
        "reviewer_b": str(args.reviewer_b),
        "packet_sha256": packet["packet_sha256"],
        "case_count": packet["case_count"],
        "model_calls": 0,
        "independent_calibration_eligible": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
