"""Offline-only, receipt-verified rebuild of the versioned G01/G02 bundle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import clinifact_ctgov_salvage_bundle_rebuild as rebuild


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--v3-plan", type=Path, required=True)
    parser.add_argument("--v3-approval", type=Path, required=True)
    parser.add_argument("--v3-capture", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--v4-plan", type=Path, required=True)
    parser.add_argument("--v4-capture", type=Path, required=True)
    parser.add_argument("--v4-execution-manifest", type=Path, required=True)
    parser.add_argument("--v4-execution-approval", type=Path, required=True)
    parser.add_argument("--v4-root-approval", type=Path, required=True)
    args = parser.parse_args()

    verified = rebuild.verify_v5_inputs_for_rebuild(plan_dir=args.plan_dir,
        v3_plan_path=args.v3_plan, v3_approval_path=args.v3_approval,
        v3_capture_dir=args.v3_capture, bindings_path=args.bindings,
        v4_plan_path=args.v4_plan, v4_capture_dir=args.v4_capture,
        v4_execution_manifest_path=args.v4_execution_manifest,
        v4_execution_approval_path=args.v4_execution_approval,
        v4_root_approval_path=args.v4_root_approval, root=ROOT)
    receipt = rebuild.rebuild_from_captured_receipts(plan_dir=args.plan_dir,
        capture_dir=args.capture_dir, verified=verified, root=ROOT)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
