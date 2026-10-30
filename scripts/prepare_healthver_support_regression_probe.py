"""Freeze remaining unseen HealthVer SUPPORT components for a diagnostic probe.

This is a post-selection, public-source exploration probe. It is not a
confirmation cohort and it contains only SUPPORT-labeled rows by design.
"""

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
FAILED_STUDY_REPORT = ROOT / (
    "artifacts/sandboxes/healthver-mipro-fresh-components-20261007/probe/"
    "healthver_mipro/05a1f50d2fc0437d/report.json"
)
DEFAULT_OUTPUT = ROOT / "artifacts/probe/healthver_remaining_support_probe.json"
EVAL_SCOPES = ("search_validation", "internal_candidate")


def build_cohort() -> dict[str, Any]:
    tasks = load_tasks(["dev_optimize"])
    by_id = {str(row["task_id"]): row for row in tasks}
    used_components: set[str] = set()
    prior_hashes = {}
    for cohort_path in PRIOR_COHORTS:
        cohort = read_json(cohort_path)
        prior_hashes[cohort_path.name] = cohort["cohort_sha256"]
        for scope_name in EVAL_SCOPES:
            scope = (cohort.get("scopes") or {}).get(scope_name)
            if not scope:
                raise ValueError(f"prior cohort lacks evaluation scope: {scope_name}")
            for task_id in scope["task_ids"]:
                row = by_id.get(str(task_id))
                if row is None:
                    raise ValueError(f"prior evaluation task is missing: {task_id}")
                used_components.add(str(row["group_id"]))

    remaining = [
        row for row in tasks
        if row.get("cohort_candidate_scope") == "internal_candidate"
        and str(row["group_id"]) not in used_components
    ]
    component_ids = sorted({str(row["group_id"]) for row in remaining})
    if len(component_ids) != 1:
        raise ValueError(
            f"expected one remaining whole internal component; found {len(component_ids)}")
    labels = Counter(str(row["options"][int(row["correct_index"])]) for row in remaining)
    if not remaining or set(labels) != {"SUPPORT"}:
        raise ValueError("remaining internal component is not a SUPPORT-only diagnostic group")

    selected_report = read_json(FAILED_STUDY_REPORT)
    if selected_report.get("candidate_status") != "internal_gate_rejected":
        raise ValueError("the frozen MIPRO candidate is not an internally rejected candidate")
    candidate_id = str(selected_report.get("candidate_id") or "")
    if not candidate_id:
        raise ValueError("the failed MIPRO study has no candidate ID")

    manifest_path = ROOT / "data/manifests/healthver_task_input.json"
    cohort = {
        "schema_version": "healthver-remaining-support-diagnostic-v1",
        "status": "EXPLORATORY_TARGETED_DIAGNOSTIC_NOT_CONFIRMATION",
        "dataset": DATASET,
        "protocol_version": "healthver-paired-support-diagnostic-v1",
        "source_manifest_sha256": sha256_file(manifest_path),
        "prior_evaluation_cohort_sha256": prior_hashes,
        "prior_evaluation_components_excluded": sorted(used_components),
        "candidate_study_plan_sha256": selected_report["plan"]["plan_sha256"],
        "candidate_id": candidate_id,
        "candidate_status_at_freeze": selected_report["candidate_status"],
        "task_ids": sorted(str(row["task_id"]) for row in remaining),
        "task_count": len(remaining),
        "label_counts": {label: labels[label] for label in LABELS},
        "component_ids": component_ids,
        "component_count": len(component_ids),
        "task_scope": "remaining, previously unscored internal_candidate component",
        "candidate_runtime_uses_live_retrieval": False,
        "fixed_task_evidence": True,
        "source_upstream_test_materialized": True,
        "source_test_is_blind_confirmation": False,
        "eligible_for_confirmation": False,
        "selection_or_confirmation_evidence": False,
        "project_selection_gate_or_confirmation_read": False,
        "selection_policy": (
            "freeze all tasks from one complete unused component in the prior internal "
            "candidate pool; selected MIPRO candidate and baseline are compared as a "
            "diagnostic only, with no optimizer or gate tuning"),
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
        "labels": cohort["label_counts"],
        "components": cohort["component_count"],
        "eligible_for_confirmation": cohort["eligible_for_confirmation"],
        "model_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
