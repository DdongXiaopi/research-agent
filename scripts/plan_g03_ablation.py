from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.g03_ablation_plan import build_g03_ablation_plan  # noqa: E402
from src.util import read_json, sha256_file, write_json  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--confirmation-report", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-protocol", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--logical-limit", type=int, required=True)
    parser.add_argument("--physical-limit", type=int, required=True)
    parser.add_argument("--ledger-path", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261017)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    ledger = args.ledger_path.resolve()
    output = args.out.resolve()
    if ledger == output:
        raise ValueError("G03 ledger path and plan output path must be distinct")
    if ledger.exists():
        raise FileExistsError(f"G03 request ledger already exists: {ledger}")
    if output.exists():
        raise FileExistsError(f"G03 plan output already exists: {output}")
    bundle_path = args.candidate_bundle.resolve()
    report_path = args.confirmation_report.resolve()
    cohort_path = args.cohort.resolve()
    plan = build_g03_ablation_plan(
        read_json(bundle_path), read_json(report_path), read_json(cohort_path),
        candidate_bundle_file_sha256=sha256_file(bundle_path),
        confirmation_report_file_sha256=sha256_file(report_path),
        cohort_file_sha256=sha256_file(cohort_path),
        target_model_id=args.target_model,
        target_protocol=args.target_protocol,
        seeds=args.seeds,
        logical_limit=args.logical_limit,
        physical_limit=args.physical_limit,
        ledger_path=str(ledger),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, plan)
    print(json.dumps({"status": plan["status"], "plan_sha256": plan["plan_sha256"],
                      "plan_path": str(output), "model_calls": 0,
                      "minimum_logical_operations": plan["minimum_logical_operations"],
                      "logical_limit": plan["logical_limit"],
                      "physical_limit": plan["physical_limit"],
                      "arms": plan["arms"], "seeds": plan["seeds"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
