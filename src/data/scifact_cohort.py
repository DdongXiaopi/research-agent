"""Deterministic, group-disjoint SciFact cohorts for exploration only."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .scifact_task_input import DATASET, LABELS, PROTOCOL_VERSION, SOURCE_SHA256
from ..util import sha256_obj


SCOPE_TARGETS = {
    "mipro_train": 10,
    "search_validation": 6,
    "internal_candidate": 6,
}


def build_nested_cohort(
    rows: list[dict[str, Any]],
    *,
    salt: str = "scifact-mipro-exploration-v1",
    excluded_component_ids: set[str] | None = None,
    excluded_task_ids: set[str] | None = None,
    scope_targets_per_class: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build balanced scopes without reusing a cited-document component.

    Every task in a selected scope is drawn from a different connected
    claim/document component. Components touched by historical model outputs
    are removed before selection.
    """
    if not str(salt).strip():
        raise ValueError("SciFact cohort salt must be non-empty")
    excluded_components = {str(value) for value in (excluded_component_ids or set())}
    excluded_tasks = {str(value) for value in (excluded_task_ids or set())}
    task_to_component: dict[str, str] = {}
    candidates: dict[str, dict[str, list[dict[str, Any]]]] = {
        label: defaultdict(list) for label in LABELS
    }
    seen_tasks: set[str] = set()
    for row in rows:
        task_id = str(row.get("task_id") or "")
        group_id = str(row.get("group_id") or "")
        if (row.get("dataset") != DATASET or row.get("split") != "dev_optimize"
                or not task_id or not group_id):
            raise ValueError("SciFact cohort rows must be valid fixed-evidence optimize tasks")
        if task_id in seen_tasks:
            raise ValueError(f"duplicate SciFact task ID: {task_id}")
        seen_tasks.add(task_id)
        task_to_component[task_id] = group_id
        options = row.get("options")
        answer_index = row.get("correct_index")
        if (not isinstance(options, list) or type(answer_index) is not int
                or not 0 <= answer_index < len(options)):
            raise ValueError(f"invalid SciFact answer row: {task_id}")
        label = str(options[answer_index])
        if label not in LABELS or row.get("gold_label") != label:
            raise ValueError(f"unexpected SciFact label in {task_id}: {label}")
        if group_id not in excluded_components and task_id not in excluded_tasks:
            candidates[label][group_id].append(row)

    excluded_components.update(
        task_to_component[task_id]
        for task_id in excluded_tasks if task_id in task_to_component
    )
    if excluded_components:
        for label in LABELS:
            for group_id in list(candidates[label]):
                if group_id in excluded_components:
                    del candidates[label][group_id]

    targets = dict(SCOPE_TARGETS if scope_targets_per_class is None
                   else scope_targets_per_class)
    if (set(targets) != set(SCOPE_TARGETS)
            or any(type(value) is not int or value < 1 for value in targets.values())):
        raise ValueError("SciFact scope targets must be positive integers for every scope")
    targets = {scope_name: targets[scope_name] for scope_name in SCOPE_TARGETS}

    component_scope: dict[str, str] = {}
    if scope_targets_per_class is None:
        # Preserve the established 80/10/10 assignment for existing cohorts.
        all_components = {group for by_group in candidates.values() for group in by_group}
        for group_id in all_components:
            rank = int(sha256_obj([salt, "component-scope", group_id])[:8], 16) % 100
            component_scope[group_id] = (
                "mipro_train" if rank < 80 else
                "search_validation" if rank < 90 else
                "internal_candidate"
            )
    else:
        # Small remaining cohorts can have a rare class that cannot tolerate a
        # probabilistic 80/10/10 split. Allocate exact, disjoint class quotas
        # after excluding touched components; fail closed when the source lacks
        # enough components for the requested scopes.
        for label in LABELS:
            ordered_groups = sorted(
                candidates[label],
                key=lambda group_id: sha256_obj([salt, label, group_id]),
            )
            required = sum(targets.values())
            if len(ordered_groups) < required:
                raise ValueError(
                    f"SciFact has too few independent components for {label}: "
                    f"{len(ordered_groups)} < {required}; reduce exact scope targets")
            offset = 0
            for scope_name, target in targets.items():
                for group_id in ordered_groups[offset:offset + target]:
                    component_scope[group_id] = scope_name
                offset += target

    scopes: dict[str, Any] = {}
    selected_component_ids: dict[str, set[str]] = {}
    selected_task_ids: set[str] = set()
    for label in LABELS:
        for scope_name, target in targets.items():
            ordered_groups = sorted(
                (group_id for group_id in candidates[label]
                 if component_scope.get(group_id) == scope_name),
                key=lambda group_id: sha256_obj([salt, label, group_id]),
            )
            if len(ordered_groups) < target:
                raise ValueError(
                    f"SciFact has too few independent components for {label} in "
                    f"{scope_name}: {len(ordered_groups)} < {target}; try another salt")
            selected = []
            groups = []
            for group_id in ordered_groups[:target]:
                group_rows = sorted(
                    candidates[label][group_id],
                    key=lambda row: sha256_obj([salt, label, row["task_id"]]),
                )
                selected.append(group_rows[0])
                groups.append(group_id)
            prior = scopes.setdefault(scope_name, {
                "task_ids": [], "class_counts": {name: 0 for name in LABELS},
                "component_ids": [],
            })
            prior["task_ids"].extend(str(row["task_id"]) for row in selected)
            prior["class_counts"][label] = len(selected)
            prior["component_ids"].extend(groups)
            selected_component_ids.setdefault(scope_name, set()).update(groups)
            selected_task_ids.update(str(row["task_id"]) for row in selected)

    for scope_name, entry in scopes.items():
        entry["task_ids"].sort()
        entry["component_ids"] = sorted(set(entry["component_ids"]))
        entry["task_count"] = len(entry["task_ids"])
        entry["component_count"] = len(entry["component_ids"])
        entry["article_count"] = entry["component_count"]
    scope_names = list(scopes)
    for index, left in enumerate(scope_names):
        for right in scope_names[index + 1:]:
            if selected_component_ids[left] & selected_component_ids[right]:
                raise ValueError("SciFact cohort scopes share a cited-document component")
    if len(selected_task_ids) != sum(entry["task_count"] for entry in scopes.values()):
        raise ValueError("SciFact cohort task IDs overlap across scopes")

    all_task_ids = sorted(seen_tasks)
    cohort = {
        "schema_version": 1,
        "status": "EXPLORATORY_CANDIDATE_NOT_PREREGISTERED",
        "dataset": DATASET,
        "protocol_version": PROTOCOL_VERSION,
        "source_archive_sha256": SOURCE_SHA256,
        "source_task_ids_sha256": sha256_obj(all_task_ids),
        "source_task_count": len(all_task_ids),
        "source_split_policy": "labeled official train+dev are pooled then re-split by connected cited-document component",
        "grouping_policy": "claims sharing any cited/evidence document are assigned to one connected component",
        "candidate_runtime_uses_live_retrieval": False,
        "fixed_task_evidence": True,
        "official_test_read": False,
        "source_test_is_blind_confirmation": False,
        "eligible_for_confirmation": False,
        "selection_or_confirmation_evidence": False,
        "project_selection_gate_or_confirmation_read": False,
        "salt": salt,
        "scope_targets_per_class": targets,
        "component_assignment_policy": (
            "hashed_proportional_80_10_10_v1" if scope_targets_per_class is None
            else "exact_class_stratified_quotas_v1"),
        "excluded_task_count": len(excluded_tasks),
        "excluded_task_ids_sha256": sha256_obj(sorted(excluded_tasks)),
        "excluded_component_count": len(excluded_components),
        "excluded_component_ids_sha256": sha256_obj(sorted(excluded_components)),
        "labels": list(LABELS),
        "scopes": scopes,
    }
    cohort["cohort_sha256"] = sha256_obj(cohort)
    return cohort
