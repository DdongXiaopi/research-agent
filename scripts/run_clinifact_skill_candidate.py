#!/usr/bin/env python3
"""Execute one approved, hash-bound CliniFact Skill GEPA plan."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.clinifact_skill_candidate_execute import execute


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--approval", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    result = execute(args.plan, expected_plan_sha256=args.plan_sha256,
                     approval_path=args.approval, ledger_path=args.ledger,
                     output_path=args.out)
    print(json.dumps({k: result.get(k) for k in
                      ("status", "candidate_id", "ledger_sha256")},
                     ensure_ascii=False))
    return 0 if result["status"] in {"skill_candidate_generated", "no_skill_candidate"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

