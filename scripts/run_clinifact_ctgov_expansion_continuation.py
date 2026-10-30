"""Zero-call preflight by default; dispatch requires a separate exact approval."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import clinifact_ctgov_expansion_continuation_execute as runner
from src.data.clinifact_ctgov_expansion_capture import _canonical, _sha, _write_new


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--parent-approval", type=Path, required=True)
    parser.add_argument("--parent-capture", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--continuation-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--execute", action="store_true", help="dispatch only with valid separate root approval")
    args = parser.parse_args()

    verified = runner.verify_execution_inputs(parent_plan_path=args.parent_plan,
        parent_approval_path=args.parent_approval, parent_capture_dir=args.parent_capture,
        bindings_path=args.bindings, continuation_plan_path=args.continuation_plan,
        output_dir=args.output_dir, root=ROOT)
    inputs_sha = runner.execution_inputs_sha256(verified)
    manifest = runner.execution_approval_body(plan_sha256=verified["continuation_plan_sha256"],
        inputs_sha256=inputs_sha,
        executor_sha256=_sha(Path(runner.__file__).read_bytes()),
        cli_sha256=_sha(Path(__file__).read_bytes()), output_dir=args.output_dir)
    manifest["status"] = "pending_root_approval"
    manifest["dispatch_authorized"] = False
    manifest["approval_sha256"] = _sha(_canonical(manifest))
    report = {"status": "passed_zero_call_continuation_preflight",
        "execution_manifest": manifest,
        "source_candidate_ncts": len(verified["continuation_plan"]["source_capture"]["capture_candidate_nct_ids"]),
        "source_candidate_result_edges": verified["continuation_plan"]["source_capture"]["capture_candidate_edge_count"],
        "source_candidate_pmids": len(verified["continuation_plan"]["source_capture"]["capture_candidate_pmids"]),
        "discovery_pages": verified["continuation_plan"]["continuation_query"]["max_pages"],
        "page_size": verified["continuation_plan"]["continuation_query"]["page_size"],
        "combined_result_records_cap": verified["continuation_plan"]["remaining_downstream_budget"]["max_results_records"],
        "combined_result_edges_cap": verified["continuation_plan"]["remaining_downstream_budget"]["max_result_edges_including_saved_capture"],
        "combined_unique_pmid_cap": verified["continuation_plan"]["remaining_downstream_budget"]["max_unique_pmids_including_saved_capture"],
        "network_calls": 0, "model_calls": 0, "dispatch_authorized": False}
    if not args.execute:
        print(json.dumps(report, indent=2))
        return 0
    if not args.approval:
        parser.error("--execute requires --approval with a separately approved exact manifest")
    approval = json.loads(args.approval.read_text(encoding="utf-8"))
    runner.validate_execution_approval(approval, manifest=manifest, approval_path=args.approval)
    # Replace manifest's pending-only status with the exact approved attestation.
    result = runner.execute_continuation(verified=verified, approval=approval,
        approval_path=args.approval, manifest=manifest, output_dir=args.output_dir, root=ROOT)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
