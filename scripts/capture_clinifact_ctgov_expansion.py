"""Prepare or run the one-shot CT.gov expansion capture.

Default mode only prepares a separate pending approval record. `--execute`
requires an out-of-band root approval bound to the exact frozen plan/output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data import clinifact_ctgov_expansion_capture as capture
from src.data import clinifact_ctgov_expansion_plan as frozen

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _bindings(path: Path):
    value = _read(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True,
                        help="JSON object mapping frozen artifact roles to paths")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--execute", action="store_true", help="dispatch only after separate root approval validates")
    parser.add_argument("--pending-approval-output", type=Path)
    args = parser.parse_args()
    plan = _read(args.plan)
    frozen.validate_plan(plan)
    if not args.execute:
        if args.pending_approval_output is None:
            parser.error("dry-run requires --pending-approval-output")
        pending = capture.pending_root_approval(plan, output_dir=str(args.output_dir.resolve()))
        args.pending_approval_output.parent.mkdir(parents=True, exist_ok=True)
        with args.pending_approval_output.open("xb") as handle:
            handle.write(frozen.canonical(pending))
            handle.flush()
            import os
            os.fsync(handle.fileno())
        print(json.dumps({"status": "pending_root_approval", "approval_path": str(args.pending_approval_output),
                          "approval_sha256": pending["approval_sha256"], "dispatch_authorized": False,
                          "network_calls": 0}, indent=2))
        return 0
    if not args.approval:
        parser.error("--execute requires --approval")
    result = capture.execute_capture(plan=plan, plan_path=args.plan, bindings=_bindings(args.bindings),
        output_dir=args.output_dir, approval=_read(args.approval), root=ROOT)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
