"""Recover only verified completed rows from a failed grounding adjudication job."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.ai_grounding_reference import recover_partial_adjudications  # noqa: E402
from src.util import read_json, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet-a", "packet-b", "responses-a", "responses-b", "run-dir", "output", "audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    previous, audit = recover_partial_adjudications(
        read_json(args.packet_a), read_json(args.packet_b),
        read_json(args.responses_a), read_json(args.responses_b), args.run_dir)
    write_json(args.output, previous)
    write_json(args.audit, audit)
    print(json.dumps({"output": str(args.output), "audit": str(args.audit),
                      "recovered_call_count": audit["recovered_call_count"],
                      "unresolved_call_indices": audit["unresolved_call_indices"],
                      "model_requests_during_recovery": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
