"""Freeze fresh-to-project HealthVer eval groups from untouched components.

HealthVer's upstream test split is public and is already materialized by the
adapter. This cohort is exploratory only; it is not blind confirmation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_healthver_mipro_cohort import (  # noqa: E402
    _stratified_component_sample,
)
from src.data.healthver_task_input import (  # noqa: E402
    DATASET, LABELS, MANIFEST_PATH, load_tasks,
)
from src.util import read_json, sha256_file, sha256_obj, write_json  # noqa: E402

DEFAULT_OUTPUT = ROOT / "artifacts/probe/healthver_mipro_fresh_component_cohort.json"
PRIOR_COHORT = ROOT / "artifacts/probe/healthver_mipro_cohort_candidate.json"
SOURCE_EVAL_SCOPE = "reserved_confirmation_candidate"


def _balanced_component_partition(rows: list[dict[str, Any]], *, seed: int
                                  ) -> tuple[set[str], set[str], dict[str, Any]]:
    by_component: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        component = str(row["group_id"])
        label = str(row["options"][int(row["correct_index"])])
        by_component[component][label] += 1
    ordered = sorted(by_component,
                     key=lambda value: sha256_obj([seed, "component-order", value]))
    if len(ordered) < 2:
        raise ValueError("HealthVer untouched eval pool has too few components")

    candidates = []
    target_sizes = {len(ordered) // 2, (len(ordered) + 1) // 2}
    for size in target_sizes:
        for left_tuple in itertools.combinations(ordered, size):
            left = set(left_tuple)
            right = set(ordered) - left
            if len(right) != (len(ordered) - size):
                continue
            left_counts = sum((by_component[c] for c in left), Counter())
            right_counts = sum((by_component[c] for c in right), Counter())
            if not all(left_counts[label] >= 8 and right_counts[label] >= 8
                       for label in LABELS):
                continue
            imbalance = sum(abs(left_counts[label] - right_counts[label])
                            for label in LABELS)
            tie = sha256_obj([seed, sorted(left)])
            candidates.append((imbalance, tie, left, right,
                               left_counts, right_counts))
    if not candidates:
        raise ValueError("untouched HealthVer components cannot support balanced 24-row scopes")
    _, _, left, right, left_counts, right_counts = min(candidates,
                                                       key=lambda row: (row[0], row[1]))
    audit = {
        "component_count": len(ordered),
        "left_component_count": len(left),
        "right_component_count": len(right),
        "left_available_label_counts": {label: left_counts[label] for label in LABELS},
        "right_available_label_counts": {label: right_counts[label] for label in LABELS},
        "partition_objective": "minimize absolute label-count imbalance; deterministic hash tie-break",
    }
    return left, right, audit


def build_cohort(seed: int = 20261008) -> dict[str, Any]:
    rows = load_tasks(["dev_optimize"])
    prior = read_json(PRIOR_COHORT)
    prior_eval_components: set[str] = set()
    by_id = {str(row["task_id"]): row for row in rows}
    for scope_name in ("search_validation", "internal_candidate"):
        prior_scope = (prior.get("scopes") or {}).get(scope_name)
        if not prior_scope:
            raise ValueError(f"previous MIPRO cohort lacks {scope_name}")
        for task_id in prior_scope["task_ids"]:
            row = by_id.get(str(task_id))
            if row is None:
                raise ValueError(f"previous HealthVer evaluation task disappeared: {task_id}")
            prior_eval_components.add(str(row["group_id"]))

    train_pool = [row for row in rows
                  if row.get("cohort_candidate_scope") == "mipro_train_candidate"]
    eval_pool = [row for row in rows
                 if row.get("cohort_candidate_scope") == SOURCE_EVAL_SCOPE]
    eval_components = {str(row["group_id"]) for row in eval_pool}
    if eval_components & prior_eval_components:
        raise ValueError("fresh HealthVer pool overlaps a previously evaluated component")
    search_components, internal_components, partition_audit = (
        _balanced_component_partition(eval_pool, seed=seed))

    def component_rows(component_ids: set[str]) -> list[dict[str, Any]]:
        return [row for row in eval_pool if str(row["group_id"]) in component_ids]

    selections = {
        "mipro_train": _stratified_component_sample(train_pool, count=30, seed=seed),
        "search_validation": _stratified_component_sample(
            component_rows(search_components), count=24, seed=seed + 1),
        "internal_candidate": _stratified_component_sample(
            component_rows(internal_components), count=24, seed=seed + 2),
    }
    scopes: dict[str, Any] = {}
    chosen_components: dict[str, set[str]] = {}
    for name, selected in selections.items():
        counts = Counter(str(row["options"][int(row["correct_index"])])
                         for row in selected)
        if len(selected) not in (24, 30) or set(counts) != set(LABELS):
            raise ValueError(f"HealthVer fresh cohort scope is invalid: {name}")
        components = {str(row["group_id"]) for row in selected}
        chosen_components[name] = components
        scopes[name] = {
            "candidate_scope": ("mipro_train_candidate" if name == "mipro_train"
                                else SOURCE_EVAL_SCOPE),
            "task_ids": [str(row["task_id"]) for row in selected],
            "task_count": len(selected),
            "class_counts": {label: counts[label] for label in LABELS},
            "component_count": len(components),
            "article_count": len(components),
            "component_ids_sha256": sha256_obj(sorted(components)),
        }
    names = list(scopes)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            if chosen_components[left_name] & chosen_components[right_name]:
                raise ValueError("fresh HealthVer MIPRO scopes share a component")
    eval_groups = (chosen_components["search_validation"]
                   | chosen_components["internal_candidate"])
    if eval_groups & prior_eval_components:
        raise ValueError("new HealthVer evaluation cohort repeats prior evaluation components")

    manifest = read_json(MANIFEST_PATH)
    cohort = {
        "schema_version": "healthver-fresh-component-mipro-cohort-v1",
        "status": "EXPLORATORY_FRESH_TO_PROJECT_NOT_PREREGISTERED",
        "dataset": DATASET,
        "protocol_version": "healthver-component-disjoint-mipro-v2",
        "source_manifest_sha256": sha256_file(MANIFEST_PATH),
        "source_task_ids_sha256": manifest["task_ids_sha256"],
        "source_train_ids_sha256": manifest["task_ids_sha256"],
        "source_upstream_test_materialized": True,
        "source_test_is_blind_confirmation": False,
        "eligible_for_confirmation": False,
        "selection_or_confirmation_evidence": False,
        "project_selection_gate_or_confirmation_read": False,
        "candidate_runtime_uses_live_retrieval": False,
        "fixed_task_evidence": True,
        "seed": seed,
        "labels": list(LABELS),
        "previous_project_evaluation_cohort_sha256": prior["cohort_sha256"],
        "prior_project_eval_components_excluded": sorted(prior_eval_components),
        "source_eval_candidate_scope": SOURCE_EVAL_SCOPE,
        "source_eval_component_partition": partition_audit,
        "sample_policy": (
            "Train reuses the prior train-only component pool. Search and internal candidate "
            "are deterministically split across untouched source evaluation components, then "
            "sampled 8/class round-robin over components. Entire HealthVer source is public; "
            "exploratory only, not blind confirmation."),
        "scopes": scopes,
    }
    cohort["cohort_sha256"] = sha256_obj(cohort)
    return cohort


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    cohort = build_cohort(args.seed)
    if args.write:
        write_json(args.output, cohort)
    print(json.dumps({
        "status": "written" if args.write else "validated_not_written",
        "path": str(args.output), "cohort_sha256": cohort["cohort_sha256"],
        "source_upstream_test_materialized": cohort["source_upstream_test_materialized"],
        "eligible_for_confirmation": cohort["eligible_for_confirmation"],
        "scopes": {name: {"n": entry["task_count"],
                          "labels": entry["class_counts"],
                          "components": entry["component_count"]}
                   for name, entry in cohort["scopes"].items()},
        "partition": cohort["source_eval_component_partition"],
        "model_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
