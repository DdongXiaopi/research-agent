"""Create a deterministic, exploratory HealthVer MIPRO cohort candidate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.healthver_task_input import (  # noqa: E402
    DATASET, LABELS, MANIFEST_PATH, load_tasks,
)
from src.util import read_json, sha256_file, sha256_obj, write_json  # noqa: E402

DEFAULT_OUTPUT = ROOT / "artifacts/probe/healthver_mipro_cohort_candidate.json"
SCOPE_MAP = {
    "mipro_train": ("mipro_train_candidate", 30),
    "search_validation": ("search_validation_candidate", 24),
    "internal_candidate": ("internal_candidate", 24),
}


def _stratified_component_sample(rows: list[dict[str, Any]], *, count: int,
                                 seed: int) -> list[dict[str, Any]]:
    by_label_component: dict[str, dict[str, list[dict[str, Any]]]] = {
        label: defaultdict(list) for label in LABELS
    }
    for row in rows:
        label = str(row["options"][int(row["correct_index"])])
        by_label_component[label][str(row["group_id"])].append(row)
    per_label, remainder = divmod(count, len(LABELS))
    selected: list[dict[str, Any]] = []
    for label_index, label in enumerate(LABELS):
        components = by_label_component[label]
        ordered_components = sorted(
            components,
            key=lambda component: sha256_obj([seed, label, component]),
        )
        candidates = []
        for component in ordered_components:
            candidates.extend(sorted(
                components[component],
                key=lambda row: sha256_obj([seed, label, row["task_id"]]),
            ))
        wanted = per_label + int(label_index < remainder)
        if len(candidates) < wanted:
            raise ValueError(f"HealthVer scope has too few {label} pairs: {len(candidates)} < {wanted}")
        # Round-robin over components so one large connected component cannot
        # dominate the selected task sample.
        buckets = {component: list(components[component]) for component in ordered_components}
        for component in buckets:
            buckets[component].sort(
                key=lambda row: sha256_obj([seed, label, row["task_id"]]))
        chosen: list[dict[str, Any]] = []
        while len(chosen) < wanted:
            advanced = False
            for component in ordered_components:
                if buckets[component] and len(chosen) < wanted:
                    chosen.append(buckets[component].pop(0))
                    advanced = True
            if not advanced:
                break
        selected.extend(chosen)
    return sorted(selected, key=lambda row: sha256_obj([seed, row["task_id"]]))


def build_cohort(seed: int = 20261007) -> dict[str, Any]:
    tasks = load_tasks(["dev_optimize"])
    by_candidate_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tasks:
        by_candidate_scope[str(row["cohort_candidate_scope"])].append(row)
    scopes: dict[str, Any] = {}
    selected_groups: dict[str, set[str]] = {}
    for scope_name, (candidate_scope, count) in SCOPE_MAP.items():
        selected = _stratified_component_sample(
            by_candidate_scope[candidate_scope], count=count,
            seed=seed + list(SCOPE_MAP).index(scope_name),
        )
        counts = Counter(str(row["options"][int(row["correct_index"])])
                         for row in selected)
        if set(counts) != set(LABELS):
            raise ValueError(f"HealthVer cohort scope lacks a label: {scope_name}")
        groups = {str(row["group_id"]) for row in selected}
        selected_groups[scope_name] = groups
        scopes[scope_name] = {
            "candidate_scope": candidate_scope,
            "task_ids": [str(row["task_id"]) for row in selected],
            "task_count": len(selected),
            "class_counts": {label: counts[label] for label in LABELS},
            "component_count": len(groups),
            "article_count": len(groups),
            "component_ids_sha256": sha256_obj(sorted(groups)),
        }
    names = list(scopes)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if selected_groups[left] & selected_groups[right]:
                raise ValueError("HealthVer cohort scopes share a claim-abstract component")
    manifest = read_json(MANIFEST_PATH)
    cohort = {
        "schema_version": "healthver-exploratory-mipro-cohort-v1",
        "status": "EXPLORATORY_CANDIDATE_NOT_PREREGISTERED",
        "dataset": DATASET,
        "protocol_version": "healthver-component-disjoint-mipro-v1",
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
        "sample_policy": (
            "Deterministic class-stratified samples from preassigned pairwise component-"
            "disjoint candidate windows; round-robin across components; exploratory only."),
        "scopes": scopes,
    }
    cohort["cohort_sha256"] = sha256_obj(cohort)
    return cohort


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    cohort = build_cohort(args.seed)
    if args.write:
        write_json(args.output, cohort)
    print(json.dumps({
        "status": "written" if args.write else "validated_not_written",
        "path": str(args.output), "cohort_sha256": cohort["cohort_sha256"],
        "scopes": {name: {"n": entry["task_count"],
                          "labels": entry["class_counts"],
                          "components": entry["component_count"]}
                   for name, entry in cohort["scopes"].items()},
        "eligible_for_confirmation": cohort["eligible_for_confirmation"],
        "model_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
