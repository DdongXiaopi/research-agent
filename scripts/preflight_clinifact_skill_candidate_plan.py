#!/usr/bin/env python3
"""Run the zero-call preflight for a frozen CliniFact Skill GEPA plan."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.clinifact_skill_candidate_preflight import preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--ledger-path", type=Path, required=True)
    parser.add_argument("--approval", type=Path)
    args = parser.parse_args(argv)
    result = preflight(args.plan, expected_plan_sha256=args.expected_plan_sha256,
                       ledger_path=args.ledger_path, approval_path=args.approval)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
