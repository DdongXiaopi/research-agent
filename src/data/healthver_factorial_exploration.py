"""Historical HealthVer factorial cohort; its proposed gate is now invalidated.

The 30-row MIPRO train sample remains useful for offline exploration. The
24-row ``internal_candidate`` scope was already evaluated by the earlier
HealthVer MIPRO protocol, so it must not be reused as a new factorial
selection gate. No replacement balanced, component-disjoint HealthVer gate is
available. This adapter refuses selection-gate reads rather than presenting
reused rows as fresh candidate evidence.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import healthver_task_input as source_adapter
from .healthver_task_input import DATASET as SOURCE_DATASET
from .healthver_task_input import SOURCE_SHA256
from .healthver_task_input import load_tasks as load_source_tasks
from ..util import read_json, sha256_file, sha256_obj

ROOT = Path(__file__).resolve().parents[2]
DATASET = "healthver_factorial_exploration"
COHORT_PATH = ROOT / "artifacts/probe/healthver_mipro_fresh_component_cohort.json"
MANIFEST_PATH = ROOT / "data/manifests/healthver_factorial_exploration.json"
LABELS = ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")
SCOPE_TO_SPLIT = {"mipro_train": "dev_optimize",
                  "internal_candidate": "selection_gate"}
SELECTION_GATE_ELIGIBLE = False
SELECTION_GATE_INVALIDATION_REASON = (
    "the 24 IDs are identical to the internal_candidate rows evaluated by "
    "the HealthVer MIPRO report 05a1f50d2fc0437d; no untouched balanced "
    "three-class replacement remains in the frozen HealthVer component pool"
)


def _cohort_and_tasks() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    identity = (sha256_file(COHORT_PATH), sha256_file(source_adapter.SOURCE_ARCHIVE),
                sha256_file(source_adapter.SPLIT_CANDIDATE))
    return copy.deepcopy(_cohort_and_tasks_cached(identity))


@lru_cache(maxsize=1)
def _cohort_and_tasks_cached(identity: tuple[str, str, str]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    cohort = json.loads(COHORT_PATH.read_text(encoding="utf-8"))
    recorded_hash = str(cohort.get("cohort_sha256") or "")
    unhashed = dict(cohort)
    unhashed.pop("cohort_sha256", None)
    if not recorded_hash or sha256_obj(unhashed) != recorded_hash:
        raise ValueError("HealthVer exploration cohort checksum mismatch")
    if cohort.get("dataset") != SOURCE_DATASET:
        raise ValueError("HealthVer exploration cohort references an unexpected source")
    if cohort.get("eligible_for_confirmation") is not False:
        raise ValueError("HealthVer cohort must remain ineligible for confirmation")
    if cohort.get("candidate_runtime_uses_live_retrieval") is not False:
        raise ValueError("HealthVer factorial must keep its paired abstract as fixed evidence")

    rows = load_source_tasks(["dev_optimize"])
    by_id = {str(row["task_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate task IDs in HealthVer source pool")
    selected: dict[str, list[dict[str, Any]]] = {}
    used_components: dict[str, set[str]] = {}
    for scope, split in SCOPE_TO_SPLIT.items():
        scope_doc = (cohort.get("scopes") or {}).get(scope)
        if not isinstance(scope_doc, dict):
            raise ValueError(f"HealthVer exploration cohort missing {scope}")
        task_ids = [str(value) for value in scope_doc.get("task_ids", [])]
        if len(task_ids) != int(scope_doc.get("task_count", -1)):
            raise ValueError(f"HealthVer {scope} task count mismatch")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError(f"duplicate task IDs in HealthVer {scope}")
        try:
            scope_rows = [by_id[task_id] for task_id in task_ids]
        except KeyError as exc:
            raise ValueError(f"HealthVer cohort task absent from source: {exc}") from exc
        counts = Counter(str(row["options"][int(row["correct_index"])])
                         for row in scope_rows)
        expected_counts = scope_doc.get("class_counts") or {}
        if any(counts[label] != int(expected_counts.get(label, -1))
               for label in LABELS):
            raise ValueError(f"HealthVer {scope} class counts differ from frozen cohort")
        scoped_rows = []
        for source_row in scope_rows:
            row = dict(source_row)
            row["dataset"] = DATASET
            row["split"] = split
            row["metadata"] = {
                **dict(source_row.get("metadata") or {}),
                "exploration_cohort_sha256": recorded_hash,
                "exploration_scope": scope,
            }
            scoped_rows.append(row)
        selected[split] = sorted(scoped_rows,
                                 key=lambda row: str(row["task_id"]))
        used_components[split] = {str(row.get("group_id") or "") for row in scope_rows}
        if "" in used_components[split]:
            raise ValueError(f"HealthVer {scope} row is missing its component ID")
    if used_components["dev_optimize"] & used_components["selection_gate"]:
        raise ValueError("HealthVer optimize and selection-gate components overlap")
    selected["final_holdout"] = []
    return cohort, selected


def load_tasks(splits: list[str] | tuple[str, ...] | None = None,
               limit: int | None = None) -> list[dict[str, Any]]:
    requested = set(splits) if splits is not None else {"dev_optimize", "selection_gate"}
    allowed = {"dev_optimize", "selection_gate"}
    if requested - allowed:
        raise PermissionError(
            "HealthVer factorial exploration has no final_holdout or confirmation split")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if "selection_gate" in requested and not SELECTION_GATE_ELIGIBLE:
        raise PermissionError(
            "HealthVer factorial selection gate is invalidated: "
            + SELECTION_GATE_INVALIDATION_REASON
        )
    _, grouped = _cohort_and_tasks()
    rows = [row for split in ("dev_optimize", "selection_gate")
            if split in requested for row in grouped[split]]
    return rows if limit is None else rows[:limit]


def build_manifest() -> dict[str, Any]:
    cohort, grouped = _cohort_and_tasks()
    splits = {}
    for split in ("dev_optimize", "selection_gate", "final_holdout"):
        ids = sorted(str(row["task_id"]) for row in grouped[split])
        splits[split] = {"n": len(ids), "ids": ids,
                         "ids_sha256": sha256_obj(ids)}
    return {
        "dataset": DATASET,
        "source_file": "data/raw/healthver/data.tar.gz",
        "source_sha256": SOURCE_SHA256,
        "salt": str(cohort["cohort_sha256"]),
        "ratios": {"dev_optimize": 56, "selection_gate": 44,
                   "final_holdout": 0},
        "split_algorithm": "healthver-mipro-fresh-component-cohort-v1",
        "n_questions": sum(info["n"] for info in splits.values()),
        "protocol_version": "healthver-factorial-exploration-v1",
        "source_cohort_sha256": str(cohort["cohort_sha256"]),
        "eligible_for_confirmation": False,
        "selection_gate_eligible": SELECTION_GATE_ELIGIBLE,
        "selection_gate_invalidation_reason": SELECTION_GATE_INVALIDATION_REASON,
        "source_upstream_test_materialized": True,
        "license_status": "CC-BY-NC-2.0 declared upstream; local noncommercial exploration only pending rights review",
        "scope_note": (
            "COVID-focused dataset; not blind confirmation; fixed paired abstracts; "
            "no live retrieval; historical selection-gate rows were reused by MIPRO "
            "and are blocked from factorial candidate selection"),
        "excluded_scope": "search_validation (24 rows); not used by this factorial adapter",
        "splits": splits,
    }


def load_manifest() -> dict[str, Any]:
    """Load the materialized manifest and fail if the cohort moved afterward."""
    expected = build_manifest()
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"HealthVer factorial manifest missing: {MANIFEST_PATH}; "
            "run scripts/prepare_healthver_factorial_exploration.py")
    actual = read_json(MANIFEST_PATH)
    if actual != expected:
        raise ValueError("HealthVer factorial manifest differs from its source cohort")
    return actual
