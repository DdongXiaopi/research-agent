"""Freeze, approve, execute or merge a one-shot G05 authoring top-up.

`run` is dry-run by default. Provider dispatch requires both `--execute` and a
separate approval artifact bound to the exact top-up plan. No credentials are
accepted on the command line.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.g05_task_authoring_topup import (  # noqa: E402
    TOPUP_APPROVAL_STATEMENT,
    approve_g05_task_authoring_topup,
    build_g05_task_authoring_topup_plan,
    execute_g05_task_authoring_topup,
    merge_g05_task_authoring_topup,
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, value) -> None:
    path = path.resolve()
    try:
        path.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("G05 top-up artifacts must stay under project .workbuddy") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="build a zero-call top-up plan from a closed parent run")
    freeze.add_argument("--parent-run", type=Path, required=True)
    freeze.add_argument("--seed", required=True)
    freeze.add_argument("--run-dir", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    approve = sub.add_parser("approve", help="write a top-up plan-bound approval; no dispatch")
    approve.add_argument("--plan", type=Path, required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--confirm", required=True,
        help=f"must equal {TOPUP_APPROVAL_STATEMENT}")
    approve.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="validate by default; dispatch only with --execute")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--approval", type=Path)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--resume", action="store_true")
    merge = sub.add_parser("merge", help="merge only complete validated parent and top-up receipts")
    merge.add_argument("--parent-run", type=Path, required=True)
    merge.add_argument("--topup-run", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        plan = build_g05_task_authoring_topup_plan(args.parent_run, seed=args.seed,
            run_dir=args.run_dir, workspace=ROOT)
        _write_new(args.output, plan)
        print(json.dumps({"status": plan["status"], "plan_sha256": plan["plan_sha256"],
            "replacement_calls": plan["max_physical_attempts"],
            "lineage_physical_attempts_max": plan["lineage_cumulative_physical_attempts_max"],
            "dispatch_authorized": False}, ensure_ascii=False))
        return 0
    if args.command == "approve":
        approval = approve_g05_task_authoring_topup(_read(args.plan),
            approved_by=args.approved_by, approval_statement=args.confirm)
        _write_new(args.output, approval)
        print(json.dumps({"status": "approved_for_exact_topup_plan",
            "approval_sha256": approval["approval_sha256"], "dispatch_authorized": True},
            ensure_ascii=False))
        return 0
    if args.command == "run":
        result = execute_g05_task_authoring_topup(_read(args.plan),
            _read(args.approval) if args.approval else None,
            workspace=ROOT, execute=args.execute, resume=args.resume)
        print(json.dumps({key: result.get(key) for key in (
            "status", "topup_plan_sha256", "model_calls", "attempted_case_count",
            "valid_count", "invalid_count", "unknown_outcome_count", "physical_attempts",
            "max_cost_usd", "lineage_cumulative_physical_attempts") if key in result},
            ensure_ascii=False))
        return 0 if result.get("status") in {"dry_run_validated", "complete"} else 2
    merged = merge_g05_task_authoring_topup(args.parent_run, args.topup_run, workspace=ROOT)
    _write_new(args.output, merged)
    print(json.dumps({"status": merged["status"],
        "authoring_input_sha256": merged["authoring_input_sha256"],
        "merge_sha256": merged["merge_sha256"], "cases": len(merged["authoring_input"]["cases"])},
        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
