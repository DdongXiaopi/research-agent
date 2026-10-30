"""Audit saved v3 discovery and freeze an offline continuation plan; no calls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import clinifact_ctgov_expansion_continuation as continuation
from src.data.clinifact_ctgov_expansion_capture import _canonical, _write_new
from src.data import clinifact_ctgov_expansion_plan as frozen

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--additional-pages", type=int, default=180)
    args = parser.parse_args()
    audit = continuation.audit_discovery_capture(plan_path=args.plan, capture_dir=args.capture_dir,
        bindings_path=args.bindings, root=ROOT)
    future_capture = args.output_root / "capture-v1"
    plan = continuation.build_continuation_plan(audit, output_dir=future_capture,
        additional_discovery_pages=args.additional_pages)
    result = continuation.persist_continuation_bundle(plan,
        plan_dir=args.output_root / "plan", approval_path=args.output_root / "root-approval.pending.json")
    report = {"status": "passed_zero_call_continuation_preflight",
        "plan_sha256": plan["plan_sha256"], "source_studies": audit["study_count"],
        "source_candidate_ncts": len(audit["candidate_nct_ids"]),
        "source_result_edges": audit["candidate_edges"],
        "source_unique_fresh_pmids": len(audit["candidate_pmids"]),
        "captured_nct_exclusions": len(plan["identity_exclusions"]["nct_ids"]),
        "captured_pmid_exclusions": len(plan["identity_exclusions"]["pmids"]),
        "projected_independent_components": plan["capacity_gate"]["projected_independent_components"],
        "projection_is_not_a_guarantee": True, "dispatch_authorized": False,
        "network_calls": 0, "model_calls": 0, **result}
    _write_new(args.output_root / "offline-capacity.json", _canonical(report))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
