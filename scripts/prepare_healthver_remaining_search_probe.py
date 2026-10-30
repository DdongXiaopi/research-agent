"""Freeze unused HealthVer search-scope components for fixed-candidate probing."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.healthver_task_input import DATASET, LABELS, load_tasks  # noqa: E402
from src.util import read_json, sha256_file, sha256_obj, write_json  # noqa: E402

PRIOR_COHORTS = (
    ROOT / "artifacts/probe/healthver_mipro_cohort_candidate.json",
    ROOT / "artifacts/probe/healthver_mipro_fresh_component_cohort.json",
)
SOURCE_SCOPE = "search_validation_candidate"
FAILED_STUDY_REPORT = ROOT / (
    "artifacts/sandboxes/healthver-mipro-fresh-components-20261007/probe/"
    "healthver_mipro/05a1f50d2fc0437d/report.json"
)
DEFAULT_OUTPUT = ROOT / "artifacts/probe/healthver_remaining_search_probe.json"


def build_cohort() -> dict[str, Any]:
    tasks = load_tasks(["dev_optimize"])
    by_id = {str(row["task_id"]): row for row in tasks}
    used_components: set[str] = set()
    prior_hashes = {}
    for path in PRIOR_COHORTS:
        cohort = read_json(path)
        prior_hashes[path.name] = cohort["cohort_sha256"]
        for name in ("search_validation", "internal_candidate"):
            for task_id in cohort["scopes"][name]["task_ids"]:
                row = by_id.get(str(task_id))
                if row is None:
                    raise ValueError(f"previous evaluation task missing: {task_id}")
                used_components.add(str(row["group_id"]))
    remaining = [
        row for row in tasks
        if row.get("cohort_candidate_scope") == SOURCE_SCOPE
        and str(row["group_id"]) not in used_components
    ]
    component_ids = sorted({str(row["group_id"]) for row in remaining})
    labels = Counter(str(row["options"][int(row["correct_index"])]) for row in remaining)
    if len(remaining) != 37 or len(component_ids) != 2:
        raise ValueError(
            f"remaining search probe changed: {len(remaining)} tasks / {len(component_ids)} components")
    if set(labels) != set(LABELS):
        raise ValueError("remaining search probe does not cover each HealthVer label")

    report = read_json(FAILED_STUDY_REPORT)
    if report.get("candidate_status") != "internal_gate_rejected":
        raise ValueError("frozen candidate is not the expected internally rejected MIPRO result")
    if report.get("confirmation_eligible") is not False:
        raise ValueError("HealthVer MIPRO candidate must remain ineligible for confirmation")
    manifest = ROOT / "data/manifests/healthver_task_input.json"
    cohort = {
        "schema_version": "healthver-remaining-search-diagnostic-v1",
        "status": "EXPLORATORY_NEW_TO_PROJECT_NOT_CONFIRMATION",
        "dataset": DATASET,
        "protocol_version": "healthver-paired-candidate-transfer-diagnostic-v1",
        "source_manifest_sha256": sha256_file(manifest),
        "prior_evaluation_cohort_sha256": prior_hashes,
        "prior_evaluation_components_excluded": sorted(used_components),
        "candidate_study_plan_sha256": report["plan"]["plan_sha256"],
        "candidate_id": report["candidate_id"],
        "candidate_status_at_freeze": report["candidate_status"],
        "task_ids": sorted(str(row["task_id"]) for row in remaining),
        "task_count": len(remaining),
        "label_counts": {label: labels[label] for label in LABELS},
        "component_ids": component_ids,
        "component_count": len(component_ids),
        "source_scope": SOURCE_SCOPE,
        "candidate_runtime_uses_live_retrieval": False,
        "fixed_task_evidence": True,
        "source_upstream_test_materialized": True,
        "source_test_is_blind_confirmation": False,
        "eligible_for_confirmation": False,
        "selection_or_confirmation_evidence": False,
        "project_selection_gate_or_confirmation_read": False,
        "sample_policy": (
            "all task pairs from whole components remaining in the original search scope "
            "after excluding every previously scored search/internal evaluation component; "
            "fixed-candidate comparison only, no selection or prompt tuning"),
    }
    cohort["cohort_sha256"] = sha256_obj(cohort)
    return cohort


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    cohort = build_cohort()
    if args.write:
        write_json(args.output, cohort)
    print(json.dumps({
        "status": "written" if args.write else "validated_not_written",
        "path": str(args.output),
        "cohort_sha256": cohort["cohort_sha256"],
        "candidate_id": cohort["candidate_id"],
        "tasks": cohort["task_count"],
        "components": cohort["component_count"],
        "label_counts": cohort["label_counts"],
        "eligible_for_confirmation": cohort["eligible_for_confirmation"],
        "model_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
