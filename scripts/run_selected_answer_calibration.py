#!/usr/bin/env python3
"""Run one approved selected-answer calibration plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.selected_answer_calibration_runner import execute  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    approval = json.loads(args.approval.read_text(encoding="utf-8"))
    result = execute(packet, reference, plan, ledger_path=args.ledger,
                     output_path=args.out, timeout=args.timeout, approval=approval)
    print(json.dumps({"status": result["status"], "calibration_pass": result["calibration_pass"],
                      "model_calls": result["model_calls"], "ledger_sha256": result["ledger_sha256"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
