"""Zero-call preflight for the durable v4 cap-aware CT.gov salvage prefix."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import clinifact_ctgov_expansion_salvage as salvage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-plan", type=Path, required=True)
    parser.add_argument("--v3-approval", type=Path, required=True)
    parser.add_argument("--v3-capture", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--v4-plan", type=Path, required=True)
    parser.add_argument("--v4-capture", type=Path, required=True)
    parser.add_argument("--v4-execution-manifest", type=Path, required=True)
    parser.add_argument("--v4-execution-approval", type=Path, required=True)
    parser.add_argument("--v4-root-approval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new future downstream capture location under .workbuddy")
    parser.add_argument("--plan-dir", type=Path, required=True,
                        help="new write-once v5 plan/preflight location under .workbuddy")
    args = parser.parse_args()

    plan, details = salvage.build_offline_salvage(v3_plan_path=args.v3_plan,
        v3_approval_path=args.v3_approval, v3_capture_dir=args.v3_capture,
        bindings_path=args.bindings, v4_plan_path=args.v4_plan,
        v4_capture_dir=args.v4_capture,
        v4_execution_manifest_path=args.v4_execution_manifest,
        v4_execution_approval_path=args.v4_execution_approval,
        v4_root_approval_path=args.v4_root_approval,
        output_dir=args.output_dir, preflight_cli_path=Path(__file__), root=ROOT)
    salvage.persist_offline_bundle(plan, details, plan_dir=args.plan_dir)
    report = {"status": details["preflight"]["status"],
        "plan_path": str((args.plan_dir / "plan.json").resolve()),
        "preflight_path": str((args.plan_dir / "preflight.json").resolve()),
        "approval_contract_path": str((args.plan_dir / "downstream-approval.pending.json").resolve()),
        "plan_sha256": plan["plan_sha256"],
        "preflight_sha256": details["preflight"]["preflight_sha256"],
        "verified_inputs_sha256": details["preflight"]["verified_inputs_sha256"],
        "selection": plan["selection"], "downstream_only": plan["downstream_only"],
        "network_calls": 0, "model_calls": 0, "dispatch_authorized": False}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
