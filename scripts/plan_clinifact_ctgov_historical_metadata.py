"""Freeze, preflight, and optionally capture bounded CT.gov registry metadata."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.clinifact_ctgov_historical_plan import (  # noqa: E402
    _inside_workbuddy, build_plan, canonical, execute, preflight,
)


def _write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="create a write-once, zero-call plan")
    prepare.add_argument("--audit", required=True, type=Path)
    prepare.add_argument("--plan-dir", required=True, type=Path)
    check = commands.add_parser("preflight", help="recompute and verify the frozen plan")
    check.add_argument("--audit", required=True, type=Path)
    check.add_argument("--plan-dir", required=True, type=Path)
    run = commands.add_parser("execute", help="perform only the plan-bound metadata requests")
    run.add_argument("--audit", required=True, type=Path)
    run.add_argument("--plan-dir", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        plan_dir = _inside_workbuddy(args.plan_dir)
        if plan_dir.exists():
            raise FileExistsError(f"refusing to reuse CT.gov plan directory: {plan_dir}")
        plan = build_plan(args.audit)
        plan_dir.mkdir(parents=True, exist_ok=False)
        _write_once(plan_dir / "plan.json", canonical(plan))
        result = {"status": "prepared_zero_call", "plan_dir": str(plan_dir),
                  "plan_sha256": plan["plan_sha256"], "request_count": len(plan["request_batches"]),
                  "nct_count": plan["source"]["unique_nct_ids"], "model_calls": 0}
    elif args.command == "preflight":
        result = preflight(args.plan_dir, args.audit)
        _write_once(_inside_workbuddy(args.plan_dir) / "preflight.json", canonical(result))
    else:
        result = execute(args.plan_dir, args.audit, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") in {"prepared_zero_call", "passed", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
