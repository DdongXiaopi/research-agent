#!/usr/bin/env python3
"""Preflight, approve, and optionally dispatch one G06 v17 reference role.

The frozen v17 protocol has exactly two blind roles. It resolves disagreements
to ``uncertain`` and has no adjudicator; requesting that role is rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import instruction_reviewer_cohort_v17 as cohort  # noqa: E402
from src.eval import instruction_reviewer_v17_execution as execution  # noqa: E402
from src.eval import instruction_reviewer_v17_runner as runner  # noqa: E402
from src.util import sha256_obj  # noqa: E402

BUNDLE = ROOT / ".workbuddy" / cohort.OUTPUT_DIRECTORY_ID
EVIDENCE = ROOT / ".workbuddy" / "g05-medaesqa-calibration-20261017-v5" / "zero-price-evidence.json"
DEFAULT_OUTPUT = ROOT / ".workbuddy" / "g06-v17-role-preflight-20261018"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen(bundle_dir: Path, evidence_path: Path) -> tuple[dict[str, Any], ...]:
    """Verify frozen bundle bytes and rebuild its packet/plan from current code."""
    manifest = _read(bundle_dir / "bundle-manifest.json")
    if manifest.get("schema") != "instruction-reviewer-v17-plan-only-bundle-v1":
        raise ValueError("unexpected G06 bundle schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or manifest.get("bundle_sha256") != sha256_obj(files):
        raise ValueError("G06 bundle manifest hash is invalid")
    for name, expected in files.items():
        path = bundle_dir / name
        if path.parent != bundle_dir or not path.is_file() or _file_sha(path) != expected:
            raise ValueError(f"G06 frozen artifact is missing or changed: {name}")
    packet = _read(bundle_dir / "packet.json")
    reference_plan = _read(bundle_dir / "reference-plan-only.json")
    cohort.validate_packet_integrity(packet, ROOT)
    cohort.validate_reference_plan(packet, reference_plan)
    rebuilt_packet = cohort.build_packet(ROOT)
    rebuilt_plan = cohort.build_reference_plan(rebuilt_packet)
    if packet != rebuilt_packet or reference_plan != rebuilt_plan:
        raise ValueError("frozen v17 packet/reference plan differs from current deterministic rebuild")
    if (manifest.get("packet_sha256") != packet.get("packet_sha256")
            or manifest.get("plan_sha256") != reference_plan.get("plan_sha256")
            or manifest.get("dispatch_authorized") is not False
            or manifest.get("model_calls") != 0 or manifest.get("network_calls") != 0):
        raise ValueError("G06 bundle manifest does not match its no-dispatch packet/plan")
    evidence = _read(evidence_path)
    plan = execution.build_execution_plan(packet, reference_plan, evidence)
    execution.validate_execution_plan(packet, reference_plan, evidence, plan)
    return packet, reference_plan, evidence, plan, manifest


def _approval_template(plan: dict[str, Any], role: str) -> dict[str, Any]:
    if role not in execution.ROLES:
        raise ValueError("v17 supports reviewer_a and reviewer_b only; it defines no adjudicator role")
    count = plan["planned_calls"][role]
    body = {"schema": "g06-v17-role-dispatch-approval-v1",
        "execution_plan_sha256": plan["execution_plan_sha256"],
        "reference_plan_sha256": plan["reference_plan_sha256"], "role": role,
        "max_calls": count, "max_physical_attempts": count,
        "hard_cost_usd": "0.00", "dispatch_authorized": False,
        "root_review_complete": False, "source_tree_sha256": runner.source_tree_fingerprint()}
    return {**body, "approval_sha256": sha256_obj(body)}


def preflight(*, role: str, bundle_dir: Path, evidence_path: Path,
              output_dir: Path) -> dict[str, Any]:
    if role not in execution.ROLES:
        raise ValueError("v17 supports reviewer_a and reviewer_b only; disagreements resolve to uncertain")
    packet, reference_plan, evidence, plan, manifest = load_frozen(bundle_dir, evidence_path)
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("G06 preflight output must be under project .workbuddy") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / f"execution-plan-{role}.json"
    approval_path = output_dir / f"pending-approval-{role}.json"
    runner._atomic(plan_path, plan)
    approval = _approval_template(plan, role)
    runner._atomic(approval_path, approval)
    result = {"schema": "g06-v17-zero-call-role-preflight-v1",
        "status": "passed_zero_call_role_preflight", "role": role,
        "packet_sha256": packet["packet_sha256"],
        "bundle_manifest_sha256": _file_sha(bundle_dir / "bundle-manifest.json"),
        "bundle_sha256": manifest["bundle_sha256"],
        "reference_plan_sha256": reference_plan["plan_sha256"],
        "zero_price_evidence_sha256": sha256_obj(evidence),
        "zero_price_evidence_file_sha256": _file_sha(evidence_path),
        "execution_plan_sha256": plan["execution_plan_sha256"],
        "source_tree_sha256": runner.source_tree_fingerprint(),
        "planned_calls": plan["planned_calls"][role],
        "hard_cost_usd": "0.00", "network_calls": 0, "model_calls": 0,
        "dispatch_authorized": False, "role_registry_reserved": False,
        "pending_approval_template_sha256": approval["approval_sha256"],
        "execution_plan_path": str(plan_path), "pending_approval_path": str(approval_path)}
    result_path = output_dir / f"preflight-{role}.json"
    runner._atomic(result_path, result)
    return result


def dispatch(*, role: str, bundle_dir: Path, evidence_path: Path,
             plan_path: Path, approval_path: Path, run_dir: Path,
             role_registry_dir: Path | None = None) -> dict[str, Any]:
    if role not in execution.ROLES:
        raise ValueError("v17 supports reviewer_a and reviewer_b only; adjudicator is not in the frozen protocol")
    packet, reference_plan, evidence, rebuilt, _manifest = load_frozen(bundle_dir, evidence_path)
    plan = _read(plan_path)
    execution.validate_execution_plan(packet, reference_plan, evidence, plan)
    approval = _read(approval_path)
    runner.validate_approval(plan, role, approval, plan["planned_calls"][role])
    return runner.dispatch_role(packet=packet, reference_plan=reference_plan,
        execution_plan=plan, zero_price_evidence=evidence, role=role,
        approval=approval, run_dir=run_dir, role_registry_dir=role_registry_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("reviewer_a", "reviewer_b", "adjudicator"))
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true", help="dispatch only with a valid exact role approval")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--role-registry-dir", type=Path,
                        help="fresh one-shot reservation directory for this dispatch")
    args = parser.parse_args(argv)
    if args.role == "adjudicator":
        parser.error("frozen G06 v17 has no adjudicator; disagreement is resolved as uncertain")
    if args.execute:
        if not all((args.plan, args.approval, args.run_dir)):
            parser.error("--execute requires --plan, --approval, and --run-dir")
        result = dispatch(role=args.role, bundle_dir=args.bundle, evidence_path=args.evidence,
            plan_path=args.plan, approval_path=args.approval, run_dir=args.run_dir,
            role_registry_dir=args.role_registry_dir)
    else:
        result = preflight(role=args.role, bundle_dir=args.bundle,
            evidence_path=args.evidence, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
