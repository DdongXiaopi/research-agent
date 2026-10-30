"""Frozen request-bounded plans for canonical round and campaign execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ..config import config_fingerprint
from ..data.freeze import require_verified
from ..governance import release
from ..util import path, sha256_file, sha256_obj

_SPLITS = ("dev_optimize", "selection_gate")
_SOURCES = (
    "scripts/rsi.py", "src/evolution/round_plan.py",
    "src/evolution/round.py", "src/evolution/campaign.py",
    "src/evolution/attribution.py", "src/evolution/reflection.py",
    "src/evolution/reflection_backend.py", "src/evolution/gepa_search.py",
    "src/evolution/prompt_rewrite.py",
    "src/evolution/task_demo_search.py", "src/evolution/counterfactual.py",
    "src/eval/harness.py", "src/eval/metrics.py", "src/eval/options.py",
    "src/eval/run_manifest.py", "src/data/split.py", "src/data/freeze.py",
    "src/runtime/runner.py", "src/runtime/provider.py",
    "src/runtime/pi_bridge.py", "src/runtime/call_ledger.py",
    "src/runtime/skills.py", "src/runtime/skill_routes.py",
    "src/runtime/tool_policy.py",
)


def _release_fingerprint(release_id: str) -> dict[str, str]:
    root = release.RELEASES_ROOT / release_id
    if not root.is_dir():
        raise ValueError(f"release does not exist: {release_id}")
    return {item.relative_to(root).as_posix(): sha256_file(item)
            for item in sorted(root.rglob("*")) if item.is_file()}


def _visible_split(dataset: str, split: str, limit: int | None) -> dict[str, Any]:
    manifest_path = path("data", "manifests", f"{dataset}.json")
    if not manifest_path.is_file():
        raise ValueError(f"{dataset} has no file-backed split manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    entry = (manifest.get("splits") or {}).get(split)
    if not isinstance(entry, dict):
        raise ValueError(f"{dataset} manifest has no {split} scope")
    ids = entry.get("ids")
    count = int(entry.get("n", len(ids) if isinstance(ids, list) else 0))
    if count < 1 or not isinstance(ids, list) or len(ids) != count:
        raise ValueError(f"{dataset}/{split} requires complete manifest IDs")
    ordered = sorted(str(value) for value in ids)
    if any(not value for value in ordered) or len(set(ordered)) != len(ordered):
        raise ValueError(f"{dataset}/{split} contains empty or duplicate IDs")
    if entry.get("ids_sha256") and entry["ids_sha256"] != sha256_obj(ordered):
        raise ValueError(f"{dataset}/{split} manifest ID hash mismatch")
    selected = ordered[:limit] if limit is not None else ordered
    data_path = path("data", "normalized", f"{dataset}.jsonl").resolve()
    if not data_path.is_file():
        raise ValueError(f"{dataset} normalized data file is missing")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "data_file_sha256": sha256_file(data_path),
        "available_task_count": count,
        "planned_task_count": len(selected),
        "planned_task_ids_sha256": sha256_obj(selected),
    }


def build_round_execution_plan(
        cfg: dict[str, Any], *, command: str, release_id: str,
        datasets: Sequence[str], seeds: Sequence[int], limit: int | None,
        rounds: int, max_candidates: int, logical_cap: int, physical_cap: int,
        execution_options: dict[str, Any]) -> dict[str, Any]:
    """Bind source/data/release scope and caller-approved hard request ceilings.

    Dynamic candidate count and nested optimizer behavior are intentionally not
    guessed. The explicit caps are enforced by the shared ledger; hitting either
    ceiling aborts the run before dispatching another provider operation.
    """
    if command not in {"round", "campaign"}:
        raise ValueError("round plan command must be round or campaign")
    if not datasets or not seeds or len(set(datasets)) != len(datasets) \
            or len(set(seeds)) != len(seeds):
        raise ValueError("round plan datasets and seeds must be non-empty and unique")
    if (limit is not None and limit < 1) or rounds < 1 or max_candidates < 1:
        raise ValueError("round plan task, round, and candidate limits must be positive")
    if logical_cap < 1 or physical_cap < 1:
        raise ValueError("live round execution requires positive logical and physical caps")
    require_verified(datasets)
    state = release.load_state(release_id)
    scopes: dict[str, Any] = {}
    for dataset in datasets:
        scopes[dataset] = {
            split: _visible_split(dataset, split, limit) for split in _SPLITS
        }
    unsigned = {
        "schema": "rsi-round-execution-plan-v1",
        "purpose": "one-shot request-bounded evolution round or campaign",
        "command": command,
        "configuration_sha256": sha256_obj(cfg),
        "protocol_fingerprint": config_fingerprint(cfg),
        "source_sha256": {name: sha256_file(path(name)) for name in _SOURCES},
        "release_id": state.release_id,
        "release_artifacts": _release_fingerprint(state.release_id),
        "datasets": list(datasets),
        "seeds": [int(seed) for seed in seeds],
        "limit_per_dataset_split_seed": limit,
        "visible_split_scopes": scopes,
        "selection_policy": {
            "candidate_selection_splits": list(_SPLITS),
            "selection_gate_per_task_output": False,
            "final_holdout_read": False,
        },
        "round_count": int(rounds),
        "max_candidates_per_round": int(max_candidates),
        "execution_options": execution_options,
        "provider": str((cfg.get("runtime") or {}).get("provider") or "fake").lower(),
        "task_model_id": str(((cfg.get("models") or {}).get("task") or {}).get("model_id") or ""),
        "logical_request_cap": int(logical_cap),
        "physical_request_cap": int(physical_cap),
        "request_ledger_required": True,
        "model_calls_during_preflight": 0,
        "visible_task_rows_materialized_during_preflight": False,
        "selection_gate_rows_materialized_during_preflight": False,
        "final_holdout_read_during_preflight": False,
        "explicit_user_approval_required_before_execution": True,
    }
    return {**unsigned, "plan_sha256": sha256_obj(unsigned)}


def verify_round_execution_plan(plan: dict[str, Any], expected_sha256: str) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    actual = sha256_obj(unsigned)
    if plan.get("plan_sha256") != actual:
        raise ValueError("frozen round plan hash is internally inconsistent")
    if expected_sha256 != actual:
        raise ValueError("expected SHA-256 does not match the current round plan")
