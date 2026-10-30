"""Freeze, approve, dry-run, or explicitly execute G05 task authoring.

No subcommand contacts a provider except `run --execute` with a valid
hash-bound approval file. Credentials are read by HttpProvider from its
configured environment variable and are never accepted on the command line.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.g05_task_authoring_execution import (  # noqa: E402
    APPROVAL_STATEMENT,
    approve_g05_task_authoring_execution,
    execute_g05_task_authoring,
    freeze_g05_task_authoring_execution_plan,
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, value) -> None:
    path = path.resolve()
    try:
        path.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("G05 plan and approval files must stay under project .workbuddy") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze", help="write a zero-call execution plan")
    freeze.add_argument("--authoring-plan", type=Path, required=True)
    freeze.add_argument("--sample-plan", type=Path, required=True)
    freeze.add_argument("--pricing-evidence", type=Path, default=ROOT / ".workbuddy/g06-instruction-reviewer-v3-ai-reference-20261016/pricing-evidence.json")
    freeze.add_argument("--output", type=Path, required=True)

    approve = sub.add_parser("approve", help="write a human-confirmed plan-bound approval; no dispatch")
    approve.add_argument("--execution-plan", type=Path, required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--confirm", required=True,
        help=f"must equal {APPROVAL_STATEMENT}")
    approve.add_argument("--output", type=Path, required=True)

    run = sub.add_parser("run", help="validate by default; dispatch only with --execute")
    run.add_argument("--execution-plan", type=Path, required=True)
    run.add_argument("--authoring-plan", type=Path, required=True)
    run.add_argument("--sample-plan", type=Path, required=True)
    run.add_argument("--approval", type=Path)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--execute", action="store_true",
        help="required alongside a valid approval to dispatch provider calls")
    run.add_argument("--resume", action="store_true",
        help="continue only unstarted cases in the same immutable run")

    args = parser.parse_args(argv)
    if args.command == "freeze":
        plan = freeze_g05_task_authoring_execution_plan(
            _read(args.authoring_plan), _read(args.sample_plan),
            pricing_evidence_path=args.pricing_evidence, workspace=ROOT)
        _write_new(args.output, plan)
        print(json.dumps({"status": plan["status"], "plan_sha256": plan["plan_sha256"],
            "output": str(args.output.resolve()), "calls": 0,
            "dispatch_authorized": False, "max_physical_attempts": plan["max_physical_attempts"],
            "max_tokens_per_call": plan["max_tokens_per_call"],
            "max_cost_usd": plan["max_cost_usd"], "paid_fallback": False}, ensure_ascii=False))
        return 0
    if args.command == "approve":
        approval = approve_g05_task_authoring_execution(
            _read(args.execution_plan), approved_by=args.approved_by,
            approval_statement=args.confirm)
        _write_new(args.output, approval)
        print(json.dumps({"status": "approved_for_frozen_scope_only",
            "approval_sha256": approval["approval_sha256"],
            "output": str(args.output.resolve()), "paid_fallback": False}, ensure_ascii=False))
        return 0
    approval = _read(args.approval) if args.approval else None
    result = execute_g05_task_authoring(
        _read(args.execution_plan), _read(args.authoring_plan), _read(args.sample_plan),
        approval, args.run_dir, execute=args.execute, resume=args.resume, workspace=ROOT)
    print(json.dumps({key: result.get(key) for key in (
        "status", "execution_plan_sha256", "approval_validated", "planned_calls",
        "model_calls", "attempted_case_count", "valid_count", "invalid_count",
        "unknown_outcome_count", "physical_attempts", "total_completion_tokens",
        "max_cost_usd", "paid_fallback", "test_only") if key in result}, ensure_ascii=False))
    return 0 if result.get("status") in {"dry_run_validated", "complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
