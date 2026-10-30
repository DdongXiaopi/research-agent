#!/usr/bin/env python3
"""Preflight or execute one explicitly approved G09 v7 source capture.

Preflight is read-only and makes zero network/model calls. Capture requires an
exact plan hash, root approval artifact, write-ahead ledger, and fresh output
path; it is intentionally a one-shot operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.open_research_v7_capture import capture_v7_query_arm_plan  # noqa: E402
from src.eval.open_research_v7_preflight import summarize_v7_capture_plan  # noqa: E402
from src.eval.open_research_v7_query_plan import validate_v7_query_arm_plan  # noqa: E402


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(args: argparse.Namespace) -> tuple[dict, dict, dict, dict, dict, dict]:
    return (_read_json(args.plan), _read_json(args.parent_capture),
        _read_json(args.parent_plan), _read_json(args.v6_capture),
        _read_json(args.v6_root_audit), _read_json(args.approval)
        if args.approval else {})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true",
        help="verify the frozen parent→plan chain without network access")
    mode.add_argument("--execute", action="store_true",
        help="execute the one-shot source capture; requires a root approval artifact")
    for name in ("plan", "parent-capture", "parent-plan", "v6-capture", "v6-root-audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--review-package", type=Path)
    parser.add_argument("--expected-approval-sha256")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--captured-at")
    args = parser.parse_args(argv)

    if args.execute:
        missing = [name for name, value in (
            ("--approval", args.approval),
            ("--review-package", args.review_package),
            ("--expected-approval-sha256", args.expected_approval_sha256),
            ("--ledger", args.ledger), ("--output", args.output),
            ("--captured-at", args.captured_at)) if value is None]
        if missing:
            parser.error("--execute requires " + ", ".join(missing))
    elif any((args.approval, args.review_package, args.expected_approval_sha256, args.ledger,
            args.output, args.captured_at)):
        parser.error("capture-only arguments cannot be used with --preflight")

    plan, parent, parent_plan, v6_capture, v6_audit, approval = _inputs(args)
    validated = validate_v7_query_arm_plan(plan=plan,
        expected_plan_sha256=args.expected_plan_sha256,
        parent_capture=parent, parent_capture_file_sha256=_file_sha256(args.parent_capture),
        parent_plan_manifest=parent_plan, v6_capture=v6_capture,
        v6_capture_file_sha256=_file_sha256(args.v6_capture), v6_root_audit=v6_audit,
        v6_root_audit_file_sha256=_file_sha256(args.v6_root_audit))
    summary = summarize_v7_capture_plan(validated)
    if args.preflight:
        print(json.dumps({"status": "preflight_validated_no_network", **summary},
            ensure_ascii=False, sort_keys=True))
        return 0

    review_package = _read_json(args.review_package)
    if (_file_sha256(args.review_package) != approval.get("review_package_sha256") or
            review_package.get("plan_sha256") != validated["plan_sha256"]):
        raise ValueError("root approval does not bind the supplied review package and exact plan")

    result = capture_v7_query_arm_plan(plan=plan,
        expected_plan_sha256=args.expected_plan_sha256,
        parent_capture=parent, parent_capture_file_sha256=_file_sha256(args.parent_capture),
        parent_plan_manifest=parent_plan, v6_capture=v6_capture,
        v6_capture_file_sha256=_file_sha256(args.v6_capture), v6_root_audit=v6_audit,
        v6_root_audit_file_sha256=_file_sha256(args.v6_root_audit), approval=approval,
        expected_approval_sha256=args.expected_approval_sha256,
        captured_at=args.captured_at, ledger_path=args.ledger,
        capture_output_path=args.output)
    print(json.dumps({"status": result.get("status"),
        "capture_sha256": result.get("capture_sha256"),
        "primary_groups_complete": result.get("primary_groups_complete"),
        "required_primary_groups": result.get("required_primary_groups"),
        "request_attempts": result.get("request_attempts"),
        "request_failures": result.get("request_failures"),
        "diagnostic_only": result.get("diagnostic_only"),
        "scoreable_cohort_created": result.get("scoreable_cohort_created")},
        ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
