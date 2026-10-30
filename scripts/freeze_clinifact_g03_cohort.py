#!/usr/bin/env python3
"""Freeze a post-selection, development-only G03 task cohort."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.clinifact_task_input import load_tasks  # noqa: E402
from src.util import read_json, sha256_obj, write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-cohort", type=Path, required=True)
    parser.add_argument("--confirmation-cohort", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.count < 6:
        raise ValueError("G03 cohort requires at least six tasks")
    if args.out.exists():
        raise FileExistsError(f"G03 cohort already exists: {args.out}")
    bundle = read_json(args.bundle.resolve())
    source = read_json(args.source_cohort.resolve())
    confirmation = read_json(args.confirmation_cohort.resolve())
    source_ids = {str(task_id) for scope in (source.get("scopes") or {}).values()
                  for task_id in scope.get("task_ids", [])}
    confirmation_ids = {str(task_id) for task_id in confirmation.get("task_ids", [])}
    excluded = source_ids | confirmation_ids
    candidates = [row for row in load_tasks(splits=["dev_optimize"])
                  if str(row["task_id"]) not in excluded]
    by_component = {}
    for row in candidates:
        by_component.setdefault(str(row.get("group_id") or row["task_id"]), []).append(row)
    representatives = [min(rows, key=lambda row: sha256_obj(["g03", row["task_id"]]))
                        for rows in by_component.values()]
    representatives.sort(key=lambda row: sha256_obj(["g03-order", row["task_id"]]))
    selected = representatives[:args.count]
    if len(selected) != args.count:
        raise ValueError("not enough post-selection disjoint task components")
    rows = [{"task_id": str(row["task_id"]), "material_sha256": sha256_obj(row),
             "component_id": str(row.get("group_id") or row["task_id"])} for row in selected]
    cohort = {
        "schema": "g03-frozen-development-cohort-v1",
        "cohort_id": "g03-dev-" + sha256_obj([bundle["candidate_bundle_sha256"], [r["task_id"] for r in rows]])[:16],
        "candidate_bundle_sha256": bundle["candidate_bundle_sha256"],
        "dataset": "clinifact_fixed_abstract", "split": "development",
        "seed_semantics": "distinct_provider_requests_uncontrolled_sampling",
        "excluded_cohort_sha256s": [str(source["cohort_sha256"]), str(confirmation["cohort_sha256"])],
        "excluded_task_ids_sha256": sha256_obj(sorted(excluded)),
        "task_rows": rows, "task_count": len(rows),
        "official_test_read": False, "official_validation_read": False,
        "selection_gate_read": False, "final_holdout_read": False,
        "source_policy": "post-selection disjoint from G01 search source and G02 bundle confirmation",
    }
    cohort["cohort_sha256"] = sha256_obj(cohort)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out.resolve(), cohort)
    print(json.dumps({"status": "g03_cohort_frozen", "cohort_sha256": cohort["cohort_sha256"],
                      "task_count": len(rows), "excluded_source": source["cohort_sha256"],
                      "excluded_confirmation": confirmation["cohort_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
