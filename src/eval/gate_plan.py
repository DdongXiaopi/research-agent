"""Frozen, zero-call execution plans for official aggregate gate verdicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..config import config_fingerprint
from ..data.split import load_manifest
from ..governance import release
from ..util import path, repo_root, sha256_file, sha256_obj


def _release_fingerprint(release_id: str) -> dict[str, str]:
    root = release.RELEASES_ROOT / release_id
    if not root.is_dir():
        raise ValueError(f"release does not exist: {release_id}")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in files}


def _split_metadata(dataset: str) -> tuple[int, str, str, str]:
    manifest = load_manifest(dataset)
    split = (manifest.get("splits") or {}).get("selection_gate")
    if not isinstance(split, dict):
        raise ValueError(f"{dataset} manifest has no selection_gate metadata")
    ids = split.get("ids")
    count = int(split.get("n", len(ids) if isinstance(ids, list) else 0))
    if count < 1:
        raise ValueError(f"{dataset} selection_gate manifest count must be positive")
    if isinstance(ids, list):
        if len(ids) != count:
            raise ValueError(f"{dataset} selection_gate IDs/count mismatch")
        cohort_sha = sha256_obj(ids)
    else:
        cohort_sha = str(split.get("ids_sha256") or "")
        if len(cohort_sha) != 64:
            raise ValueError(f"{dataset} selection_gate manifest lacks an ID fingerprint")
    manifest_path = path("data", "manifests", f"{dataset}.json")
    manifest_sha = sha256_file(manifest_path) if manifest_path and manifest_path.exists() else sha256_obj(manifest)
    data_rel = str(split.get("path") or f"data/normalized/{dataset}.jsonl")
    data_path = (repo_root() / data_rel).resolve()
    if not data_path.is_relative_to(repo_root().resolve()) or not data_path.is_file():
        raise ValueError(f"{dataset} selection_gate data file is missing or outside the repository")
    return count, cohort_sha, manifest_sha, sha256_file(data_path)


def build_gate_execution_plan(cfg: dict[str, Any], *, challenger_id: str,
                              champion_id: str, datasets: Sequence[str],
                              seeds: Sequence[int], limit: int | None,
                              protocol_check: bool = True) -> dict[str, Any]:
    """Bind a scoring run using manifest metadata only; never load task rows."""
    from .gate_verdict import GATE_VERDICT_SPLIT

    if not datasets or not seeds:
        raise ValueError("gate execution plan requires datasets and seeds")
    required = set((cfg.get("evaluation") or {}).get("primary_final_datasets") or [])
    if not required <= set(datasets):
        raise ValueError("formal gate plan omits configured main datasets")
    if len(set(datasets)) != len(datasets) or len(set(seeds)) != len(seeds):
        raise ValueError("gate execution plan datasets/seeds must be unique")
    if limit is not None and limit < 1:
        raise ValueError("gate execution plan limit must be positive")

    # Loading release snapshots verifies their embedded artifact checksums and
    # does not construct an evaluator or provider.
    champ = release.load_state(champion_id)
    challenger = release.load_state(challenger_id)
    if champion_id == challenger_id:
        raise ValueError("champion and challenger must differ")

    dataset_plan: dict[str, Any] = {}
    total_comparisons = 0
    for dataset in datasets:
        count, cohort_sha, manifest_sha, data_sha = _split_metadata(dataset)
        planned_tasks = min(count, limit) if limit is not None else count
        dataset_plan[dataset] = {
            "manifest_sha256": manifest_sha,
            "data_file_sha256": data_sha,
            "selection_gate_count": count,
            "selection_gate_ids_sha256": cohort_sha,
            "planned_tasks_per_arm_seed": planned_tasks,
            "aggregate_only": True,
        }
        total_comparisons += planned_tasks * len(seeds) * 2

    runtime = cfg.get("runtime") or {}
    model_calls = int(runtime.get("max_model_calls", 0))
    if model_calls < 1:
        raise ValueError("runtime.max_model_calls must be positive for bounded gate execution")
    task_model = (cfg.get("models") or {}).get("task") or {}
    router_model = (cfg.get("models") or {}).get("router") or {}
    http_attempts = max(1, int(task_model.get("max_retries", 1)),
                        int(router_model.get("max_retries", 1)))
    pi_retries = max(0, min(4, int(runtime.get("pi_provider_max_retries", 4))))
    pi_subprocess_attempts = max(1, min(3, int(runtime.get("pi_timeout_max_attempts", 2))))
    per_operation_physical = max(http_attempts, (pi_retries + 1) * pi_subprocess_attempts)
    logical_cap = total_comparisons * model_calls
    physical_cap = logical_cap * per_operation_physical
    unsigned = {
        "schema": "official-gate-verdict-execution-plan-v1",
        "purpose": "one-shot aggregate-only selection_gate comparison",
        "config_fingerprint": config_fingerprint(cfg),
        "gate_protocol_source_sha256": sha256_file(path("src", "eval", "gate_verdict.py")),
        "plan_builder_source_sha256": sha256_file(Path(__file__)),
        "cli_source_sha256": sha256_file(path("scripts", "rsi.py")),
        "champion_release_id": champ.release_id,
        "champion_artifacts": _release_fingerprint(champion_id),
        "challenger_release_id": challenger.release_id,
        "challenger_artifacts": _release_fingerprint(challenger_id),
        "datasets": list(datasets),
        "dataset_manifest_metadata": dataset_plan,
        "split": GATE_VERDICT_SPLIT,
        "seeds": [int(seed) for seed in seeds],
        "limit_per_dataset_seed_arm": limit,
        "evaluation_arms": ["champion", "challenger"],
        "result_policy": "aggregate_only_no_per_task_persistence",
        "protocol_check": bool(protocol_check),
        "provider": str(runtime.get("provider") or ""),
        "task_model_id": str(task_model.get("model_id") or ""),
        "max_model_operations_per_task": model_calls,
        "http_attempts_per_operation_upper_bound": http_attempts,
        "pi_provider_retries_after_first": pi_retries,
        "pi_subprocess_attempts_upper_bound": pi_subprocess_attempts,
        "logical_request_cap": logical_cap,
        "physical_request_cap": physical_cap,
        "request_ledger_required": True,
        "model_calls_during_preflight": 0,
        "selection_gate_task_rows_read_during_preflight": False,
        "final_holdout_read": False,
        "live_authorized": False,
        "explicit_user_approval_required_before_execution": True,
    }
    return {**unsigned, "plan_sha256": sha256_obj(unsigned)}


def verify_gate_execution_plan(plan: dict[str, Any], expected_sha256: str) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    actual = sha256_obj(unsigned)
    if plan.get("plan_sha256") != actual:
        raise ValueError("frozen gate execution plan hash is internally inconsistent")
    if expected_sha256 != actual:
        raise ValueError("expected plan hash does not match current gate execution plan")
