"""Deterministic sharding for bounded AI rubric-authoring jobs.

The shared AI-job ledger caps one authoring job at 500 logical calls.  Large
open-research cohorts therefore need several independently hash-bound author
jobs, followed by a task-identity checked merge.  This module never reads
evaluator sidecars and never performs model calls.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..util import sha256_obj, sha256_text


def _validate_task_only(dataset: dict[str, Any]) -> None:
    if set(dataset) != {"schema", "provenance", "partitions"} or \
            dataset["schema"] != "open-research-unannotated-dataset-v1":
        raise ValueError("author sharding requires an unannotated task-only dataset")
    partitions = dataset["partitions"]
    if set(partitions) != {"dev_optimize", "selection_gate", "final_holdout"}:
        raise ValueError("author sharding requires all three explicit partitions")
    seen: set[str] = set()
    owners: dict[str, str] = {}
    for split, cases in partitions.items():
        if not isinstance(cases, list) or not cases:
            raise ValueError("author sharding requires non-empty partitions")
        for case in cases:
            if set(case) - {"request", "group_id", "evaluation_role"} or \
                    not {"request", "group_id"} <= set(case):
                raise ValueError("author sharding cannot accept answers or rubrics")
            task_id = str(case["request"].get("task_id") or "")
            group_id = str(case.get("group_id") or "")
            if not task_id or not group_id or task_id in seen:
                raise ValueError("author sharding requires unique task and group identities")
            if group_id in owners and owners[group_id] != split:
                raise ValueError("a source group crosses authoring partitions")
            seen.add(task_id)
            owners[group_id] = split


def shard_task_dataset(dataset: dict[str, Any], *, max_tasks: int = 250) -> list[dict[str, Any]]:
    """Return deterministic group-preserving shards suitable for author_plan."""
    _validate_task_only(dataset)
    if isinstance(max_tasks, bool) or not isinstance(max_tasks, int) or max_tasks < 3:
        raise ValueError("max_tasks must be an integer >= 3")
    by_group: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for split, cases in dataset["partitions"].items():
        for case in cases:
            by_group[str(case["group_id"])].append((split, case))
    total_tasks = sum(len(rows) for rows in dataset["partitions"].values())
    initial_count = max(1, (total_tasks + max_tasks - 1) // max_tasks)
    split_groups = {
        split: sorted({str(case["group_id"]) for case in dataset["partitions"][split]},
                      key=sha256_text)
        for split in ("dev_optimize", "selection_gate", "final_holdout")
    }
    max_count = min(len(groups) for groups in split_groups.values())
    for shard_count in range(initial_count, max_count + 1):
        by_shard: list[list[str]] = [[] for _ in range(shard_count)]
        for groups in split_groups.values():
            for index, group in enumerate(groups):
                by_shard[index % shard_count].append(group)
        shards = [_make_shard(dataset, by_group, groups) for groups in by_shard]
        if all(sum(len(rows) for rows in shard["partitions"].values()) <= max_tasks
               for shard in shards):
            return shards
    raise ValueError("group-preserving shard exceeds max_tasks; lower shard size or use more shards")


def _make_shard(dataset, by_group, groups):
    partitions = {split: [] for split in ("dev_optimize", "selection_gate", "final_holdout")}
    for group in groups:
        for split, case in by_group[group]:
            partitions[split].append(case)
    if any(not rows for rows in partitions.values()):
        raise ValueError("a shard must retain all three non-empty partitions")
    return {"schema": dataset["schema"], "provenance": dict(dataset["provenance"]),
            "partitions": partitions}


def merge_authored_datasets(shards: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge completed author outputs after strict task/rubric identity checks."""
    if not shards:
        raise ValueError("at least one authored shard is required")
    base = shards[0]
    if base.get("schema") != "open-research-dataset-v1":
        raise ValueError("authored shard schema is invalid")
    provenance = dict(base.get("provenance") or {})
    partitions = {split: [] for split in ("dev_optimize", "selection_gate", "final_holdout")}
    seen: set[str] = set()
    for shard in shards:
        if shard.get("schema") != "open-research-dataset-v1" or shard.get("provenance") != provenance:
            raise ValueError("authored shard provenance/schema differs")
        if set(shard.get("partitions", {})) != set(partitions):
            raise ValueError("authored shard partitions differ")
        for split, cases in shard["partitions"].items():
            for case in cases:
                request = case.get("request") if isinstance(case, dict) else None
                task_id = str(request.get("task_id") or "") if isinstance(request, dict) else ""
                if not task_id or "rubric" not in case or task_id in seen:
                    raise ValueError("authored merge requires one rubric per unique task")
                seen.add(task_id)
                partitions[split].append(case)
    return {"schema": "open-research-dataset-v1", "provenance": provenance,
            "partitions": partitions,
            "shard_count": len(shards),
            "merged_task_ids_sha256": sha256_obj(sorted(seen))}
