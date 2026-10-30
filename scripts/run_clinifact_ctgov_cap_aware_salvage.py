"""Verify or execute the frozen downstream-only G01/G02 salvage package."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import clinifact_ctgov_expansion_salvage_execute as runner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--v3-plan", type=Path, required=True)
    parser.add_argument("--v3-approval", type=Path, required=True)
    parser.add_argument("--v3-capture", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--v4-plan", type=Path, required=True)
    parser.add_argument("--v4-capture", type=Path, required=True)
    parser.add_argument("--v4-execution-manifest", type=Path, required=True)
    parser.add_argument("--v4-execution-approval", type=Path, required=True)
    parser.add_argument("--v4-root-approval", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path)
    parser.add_argument("--execution-approval", type=Path)
    args = parser.parse_args()

    verified = runner.verify_v5_inputs(plan_dir=args.plan_dir,
        v3_plan_path=args.v3_plan, v3_approval_path=args.v3_approval,
        v3_capture_dir=args.v3_capture, bindings_path=args.bindings,
        v4_plan_path=args.v4_plan, v4_capture_dir=args.v4_capture,
        v4_execution_manifest_path=args.v4_execution_manifest,
        v4_execution_approval_path=args.v4_execution_approval,
        v4_root_approval_path=args.v4_root_approval, root=ROOT)
    plan = verified["plan"]
    output_dir = Path(plan["output_dir"])

    if args.preflight:
        if args.execution_manifest or args.execution_approval:
            parser.error("execution artifacts are accepted only with --execute")
        manifest = runner.write_pending_manifest(plan=plan, verified=verified,
            plan_dir=args.plan_dir)
        report = {"status": "passed_zero_call_downstream_preflight",
            "plan_sha256": plan["plan_sha256"],
            "verified_inputs_sha256": verified["verified_inputs_sha256"],
            "safe_prefix_sha256": verified["safe_prefix_sha256"],
            "candidate_records_sha256": verified["candidate_records_sha256"],
            "selected_pmids_sha256": verified["selected_pmids_sha256"],
            "manifest_sha256": runner._canonical_sha(manifest),
            "output_dir": str(output_dir),
            "candidate_ncts": len(verified["candidate_records"]),
            "selected_pmids": len(verified["selected_pmids"]),
            "discovery_requests": 0, "results_requests_max": 12,
            "pubmed_requests_max": 20, "physical_requests_max": 32,
            "network_calls": 0, "model_calls": 0, "labels_generated": False,
            "dispatch_authorized": False}
        print(json.dumps(report, indent=2))
        return 0

    if not args.execution_manifest or not args.execution_approval:
        parser.error("--execute requires both --execution-manifest and --execution-approval")
    manifest = json.loads(args.execution_manifest.read_text(encoding="utf-8"))
    approval = json.loads(args.execution_approval.read_text(encoding="utf-8"))
    summary = runner.execute_downstream(verified=verified, manifest=manifest,
        approval=approval, approval_path=args.execution_approval,
        output_dir=output_dir, root=ROOT)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
