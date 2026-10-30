#!/usr/bin/env python3
"""Build or preflight a read-only, parent-bound G09 v8 continuation plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.open_research_v8_continuation_plan import (  # noqa: E402
    build_v8_continuation_plan,
    summarize_v8_continuation_preflight,
)


def _read(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true",
        help="build an offline plan; output must not exist")
    mode.add_argument("--preflight", action="store_true",
        help="rebuild and verify the exact plan from its immutable v7 parent artifacts")
    parser.add_argument("--v7-plan", type=Path, required=True)
    parser.add_argument("--v7-capture", type=Path, required=True)
    parser.add_argument("--v7-root-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path,
        help="write-once plan output (required for --build)")
    parser.add_argument("--plan", type=Path,
        help="existing plan input (required for --preflight)")
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--max-new-requests", type=int)
    args = parser.parse_args(argv)
    if args.build and (args.output is None or args.plan is not None):
        parser.error("--build requires --output and forbids --plan")
    if args.preflight and (args.plan is None or args.output is not None or
            args.expected_plan_sha256 is None):
        parser.error("--preflight requires --plan and --expected-plan-sha256")

    parent_plan = _read(args.v7_plan)
    capture = _read(args.v7_capture)
    root_audit = _read(args.v7_root_audit)
    inputs = {"v7_plan": parent_plan, "v7_capture": capture,
        "v7_capture_file_sha256": _sha(args.v7_capture),
        "v7_root_audit": root_audit,
        "v7_root_audit_file_sha256": _sha(args.v7_root_audit)}
    if args.preflight:
        frozen = _read(args.plan)
        if frozen.get("plan_sha256") != args.expected_plan_sha256:
            raise ValueError("supplied expected plan hash differs from frozen v8 plan")
        rebuilt = build_v8_continuation_plan(**inputs,
            max_new_requests=frozen.get("unspent_parent_cumulative_budget"))
        if frozen != rebuilt:
            raise ValueError("v8 plan differs from deterministic parent-bound rebuild")
        print(json.dumps({"status": "preflight_validated_no_network",
            **summarize_v8_continuation_preflight(frozen)}, ensure_ascii=False,
            sort_keys=True))
        return 0

    plan = build_v8_continuation_plan(**inputs,
        max_new_requests=args.max_new_requests)
    summary = summarize_v8_continuation_preflight(plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(plan, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "plan_created_no_network", **summary,
        "plan_file_sha256": _sha(args.output)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
