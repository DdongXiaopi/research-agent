"""Freeze/preflight and exact direct lookups for only v3's unresolved CT.gov IDs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.clinifact_ctgov_unresolved_lookup import (  # noqa: E402
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
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "preflight"):
        command = sub.add_parser(name)
        command.add_argument("--parent-plan", required=True, type=Path)
        command.add_argument("--parent-report", required=True, type=Path)
        command.add_argument("--plan-dir", required=True, type=Path)
    run = sub.add_parser("execute")
    run.add_argument("--parent-plan", required=True, type=Path)
    run.add_argument("--parent-report", required=True, type=Path)
    run.add_argument("--plan-dir", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        directory = _inside_workbuddy(args.plan_dir)
        if directory.exists():
            raise FileExistsError(f"refusing to reuse direct lookup plan directory: {directory}")
        plan = build_plan(args.parent_plan, args.parent_report)
        directory.mkdir(parents=True, exist_ok=False)
        _write_once(directory / "plan.json", canonical(plan))
        result = {"status": "prepared_zero_call", "plan_dir": str(directory),
                  "plan_sha256": plan["plan_sha256"], "request_count": len(plan["requests"]),
                  "nct_ids": [item["nct_id"] for item in plan["requests"]], "model_calls": 0}
    elif args.command == "preflight":
        result = preflight(args.plan_dir, args.parent_plan, args.parent_report)
        _write_once(_inside_workbuddy(args.plan_dir) / "preflight.json", canonical(result))
    else:
        result = execute(args.plan_dir, args.parent_plan, args.parent_report, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") in {"prepared_zero_call", "passed", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
