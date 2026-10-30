"""Frozen execution plans for the canonical two-split baseline scorer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..config import config_fingerprint
from ..data.freeze import require_verified
from ..governance import release
from ..util import path, sha256_file, sha256_obj


_SCORING_SOURCES = (
    "scripts/rsi.py", "src/eval/baseline_plan.py", "src/eval/harness.py",
    "src/eval/run_manifest.py", "src/eval/metrics.py", "src/eval/options.py",
    "src/data/split.py", "src/data/freeze.py", "src/runtime/runner.py",
    "src/runtime/provider.py", "src/runtime/call_ledger.py",
    "src/runtime/skills.py", "src/runtime/skill_routes.py",
    "src/runtime/tool_policy.py",
)
_BASELINE_SPLITS = ("dev_optimize", "selection_gate")


def _release_fingerprint(release_id: str) -> dict[str, str]:
    root = release.RELEASES_ROOT / release_id
    if not root.is_dir():
        raise ValueError(f"release does not exist: {release_id}")
    return {item.relative_to(root).as_posix(): sha256_file(item)
            for item in sorted(root.rglob("*")) if item.is_file()}


def _manifest_split(dataset: str, split: str, limit: int | None) -> dict[str, Any]:
    """Bind a visible cohort using the manifest and normalized file only."""
    manifest_path = path("data", "manifests", f"{dataset}.json")
    if not manifest_path.is_file():
        raise ValueError(
            f"{dataset} has no file-backed split manifest for baseline preflight")
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    entry = (manifest.get("splits") or {}).get(split)
    if not isinstance(entry, dict):
        raise ValueError(f"{dataset} manifest has no {split} scope")
    ids = entry.get("ids")
    count = int(entry.get("n", len(ids) if isinstance(ids, list) else 0))
    if count < 1 or not isinstance(ids, list) or len(ids) != count:
        raise ValueError(f"{dataset}/{split} needs a complete ID list and positive count")
    ordered_ids = sorted(str(value) for value in ids)
    if any(not value for value in ordered_ids) or len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError(f"{dataset}/{split} manifest IDs are empty or duplicated")
    if entry.get("ids_sha256") and entry["ids_sha256"] != sha256_obj(ordered_ids):
        raise ValueError(f"{dataset}/{split} manifest ID hash mismatch")
    selected_ids = ordered_ids[:limit] if limit is not None else ordered_ids
    data_path = path("data", "normalized", f"{dataset}.jsonl").resolve()
    if not data_path.is_file():
        raise ValueError(f"{dataset} normalized visible-data file is missing")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "data_file_sha256": sha256_file(data_path),
        "available_task_count": count,
        "planned_task_count": len(selected_ids),
        "planned_task_ids_sha256": sha256_obj(selected_ids),
        "scope": split,
    }


def build_baseline_execution_plan(cfg: dict[str, Any], *, release_id: str,
                                  datasets: Sequence[str], seeds: Sequence[int],
                                  limit: int | None) -> dict[str, Any]:
    """Freeze the canonical baseline run without materializing task rows."""
    if not datasets or not seeds:
        raise ValueError("baseline plan requires datasets and seeds")
    if len(set(datasets)) != len(datasets) or len(set(seeds)) != len(seeds):
        raise ValueError("baseline plan datasets and seeds must be unique")
    if limit is not None and limit < 1:
        raise ValueError("baseline plan limit must be positive")
    require_verified(datasets)
    state = release.load_state(release_id)
    scopes: dict[str, Any] = {}
    total_task_runs = 0
    for dataset in datasets:
        scopes[dataset] = {}
        for split in _BASELINE_SPLITS:
            metadata = _manifest_split(dataset, split, limit)
            scopes[dataset][split] = metadata
            total_task_runs += metadata["planned_task_count"] * len(seeds)

    runtime = cfg.get("runtime") or {}
    model_ops_per_task = int(runtime.get("max_model_calls", 0))
    if model_ops_per_task < 1:
        raise ValueError("runtime.max_model_calls must be positive")
    logical_cap = total_task_runs * model_ops_per_task
    provider = str(runtime.get("provider") or "fake").lower()
    if provider == "pi":
        pi_retries = max(0, min(4, int(runtime.get("pi_provider_max_retries", 4))))
        bridge_attempts = max(1, min(3, int(runtime.get("pi_timeout_max_attempts", 2))))
        attempts_per_operation = (pi_retries + 1) * bridge_attempts
    else:
        task_cfg = (cfg.get("models") or {}).get("task") or {}
        attempts_per_operation = max(1, int(task_cfg.get("max_retries", 1)))
        pi_retries, bridge_attempts = 0, 1
    physical_cap = logical_cap * attempts_per_operation
    unsigned = {
        "schema": "rsi-baseline-execution-plan-v1",
        "purpose": "one-shot baseline evaluation on visible development splits",
        "configuration_sha256": sha256_obj(cfg),
        "protocol_fingerprint": config_fingerprint(cfg),
        "scoring_sources_sha256": {
            name: sha256_file(path(name)) for name in _SCORING_SOURCES
        },
        "release_id": state.release_id,
        "release_artifacts": _release_fingerprint(state.release_id),
        "datasets": list(datasets),
        "seeds": [int(seed) for seed in seeds],
        "limit_per_dataset_split_seed": limit,
        "split_scopes": scopes,
        "split_policy": list(_BASELINE_SPLITS),
        "result_policy": {
            "dev_optimize": "per-task traces and summary",
            "selection_gate": "aggregate-only-no-per-task-persistence",
        },
        "provider": provider,
        "task_model_id": str(((cfg.get("models") or {}).get("task") or {}).get("model_id") or ""),
        "max_model_operations_per_task": model_ops_per_task,
        "http_attempts_per_operation_upper_bound": attempts_per_operation,
        "pi_provider_retries_after_first": pi_retries,
        "pi_bridge_attempts_upper_bound": bridge_attempts,
        "logical_request_cap": logical_cap,
        "physical_request_cap": physical_cap,
        "request_ledger_required": True,
        "selection_gate_task_rows_read_during_preflight": False,
        "final_holdout_read": False,
        "model_calls_during_preflight": 0,
    }
    return {**unsigned, "plan_sha256": sha256_obj(unsigned)}


def verify_baseline_execution_plan(plan: dict[str, Any], expected_sha256: str) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    actual = sha256_obj(unsigned)
    if plan.get("plan_sha256") != actual:
        raise ValueError("frozen baseline plan hash is internally inconsistent")
    if expected_sha256 != actual:
        raise ValueError("expected SHA-256 does not match the current baseline plan")
