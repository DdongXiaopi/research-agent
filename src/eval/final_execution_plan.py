"""Frozen, zero-call plans for the evaluator-only final_holdout CLI."""

from __future__ import annotations

from typing import Any, Sequence

from ..config import config_fingerprint
from ..data import freeze
from ..governance import release
from ..util import path, sha256_file, sha256_obj
from . import final_eval, unblind_budget


_FINAL_SCORING_SOURCES = (
    "scripts/rsi.py",
    "src/contracts/types.py",
    "src/eval/final_execution_plan.py",
    "src/eval/final_eval.py",
    "src/eval/harness.py",
    "src/eval/run_manifest.py",
    "src/eval/metrics.py",
    "src/eval/options.py",
    "src/eval/grounding.py",
    "src/eval/stats.py",
    "src/eval/verdict.py",
    "src/data/split.py",
    "src/data/freeze.py",
    "src/runtime/runner.py",
    "src/runtime/provider.py",
    "src/runtime/call_ledger.py",
    "src/runtime/skills.py",
    "src/runtime/skill_routes.py",
    "src/runtime/tool_policy.py",
    "src/runtime/channels.py",
    "src/runtime/input_evidence.py",
    "src/runtime/trace.py",
    "src/runtime/execution.py",
    "src/retrieval/chunking.py",
    "src/retrieval/gateway.py",
    "src/retrieval/intent.py",
    "src/retrieval/corpus.py",
    "src/eval/unblind_budget.py",
    "src/config.py",
)


def _release_fingerprint(release_id: str) -> dict[str, str]:
    root = release.RELEASES_ROOT / release_id
    if not root.is_dir():
        raise ValueError(f"release does not exist: {release_id}")
    return {item.relative_to(root).as_posix(): sha256_file(item)
            for item in sorted(root.rglob("*")) if item.is_file()}


def _attempts_per_operation(cfg: dict[str, Any]) -> dict[str, int]:
    runtime = cfg.get("runtime") or {}
    if str(runtime.get("provider") or "fake").lower() == "pi":
        provider_retries = max(0, min(4, int(runtime.get("pi_provider_max_retries", 4))))
        bridge_attempts = max(1, min(3, int(runtime.get("pi_timeout_max_attempts", 2))))
        return {"attempts": (provider_retries + 1) * bridge_attempts,
                "provider_retries": provider_retries,
                "bridge_attempts": bridge_attempts}
    task = (cfg.get("models") or {}).get("task") or {}
    attempts = max(1, int(task.get("max_retries", 1)))
    return {"attempts": attempts, "provider_retries": 0, "bridge_attempts": 1}


def build_final_execution_plan(cfg: dict[str, Any], *, purpose: str,
                               release_id: str, datasets: Sequence[str],
                               seeds: Sequence[int], limit: int | None,
                               force: bool = False, actor: str = "evaluator",
                               reason: str = "", strict_protocol: bool = True,
                               budget_override: dict[str, Any] | None = None
                               ) -> dict[str, Any]:
    """Bind one final baseline or assessment without loading final task rows.

    Freeze verification may hash sealed files to verify their recorded digest; this
    function never parses or materializes final task rows.
    """
    if purpose not in {"final-baseline", "final-eval"}:
        raise ValueError("purpose must be final-baseline or final-eval")
    if not datasets or not seeds:
        raise ValueError("final execution plan requires datasets and seeds")
    if len(set(datasets)) != len(datasets) or len(set(seeds)) != len(seeds):
        raise ValueError("final execution plan datasets and seeds must be unique")
    if limit is not None and limit < 1:
        raise ValueError("final execution plan limit must be positive")

    evaluation = cfg.get("evaluation") or {}
    allowed = set(evaluation.get("primary_final_datasets") or [])
    allowed.update(evaluation.get("transfer_datasets") or [])
    if not allowed:
        allowed.update(evaluation.get("datasets") or [])
    unexpected = sorted(set(datasets) - allowed)
    if unexpected:
        raise ValueError("datasets are not configured for final evaluation: "
                         + ", ".join(unexpected))

    baseline_present = final_eval.baseline_exists()
    if purpose == "final-eval" and not baseline_present:
        raise ValueError("final-eval plan requires an existing sealed baseline report")
    if purpose == "final-baseline" and baseline_present and not force:
        raise ValueError("final-baseline already exists; use an explicitly planned --force reseal")

    frozen_report = freeze.require_verified(list(datasets))
    frozen = freeze.load_freeze()
    protocol_sha = final_eval.protocol_fingerprint(cfg)
    scopes: dict[str, dict[str, Any]] = {}
    planned_task_runs = 0
    budget_snapshots: dict[str, Any] = {}
    for dataset in datasets:
        dataset_freeze = (frozen.get("datasets") or {}).get(dataset) or {}
        final_split = (dataset_freeze.get("splits") or {}).get("final_holdout") or {}
        count = int(final_split.get("n", 0))
        ids_sha = str(final_split.get("recomputed_ids_sha256") or "")
        sealed_sha = str((frozen.get("sealed_sha256") or {}).get(dataset) or "")
        if count < 1 or len(ids_sha) != 64 or len(sealed_sha) != 64:
            raise ValueError(f"{dataset} lacks complete frozen final_holdout metadata")
        manifest = path("data", "manifests", f"{dataset}.json")
        if not manifest.is_file():
            raise ValueError(f"{dataset} manifest is missing")
        planned_count = min(count, limit) if limit is not None else count
        scopes[dataset] = {
            "split": "final_holdout",
            "available_task_count": count,
            "planned_task_count": planned_count,
            "final_task_ids_sha256": ids_sha,
            "manifest_sha256": sha256_file(manifest),
            "sealed_file_sha256": sealed_sha,
            "sealed_rows_materialized_during_preflight": False,
        }
        planned_task_runs += planned_count * len(seeds)
        budget_snapshots[dataset] = unblind_budget.budget_state(dataset, protocol_sha)

    runtime = cfg.get("runtime") or {}
    model_ops = int(runtime.get("max_model_calls", 0))
    if model_ops < 1:
        raise ValueError("runtime.max_model_calls must be positive for bounded final execution")
    attempts = _attempts_per_operation(cfg)
    logical_cap = planned_task_runs * model_ops
    physical_cap = logical_cap * attempts["attempts"]
    state = release.load_state(release_id)
    baseline_sha = sha256_file(final_eval.BASELINE_FILE) if purpose == "final-eval" else None
    source_hashes = {
        name: sha256_file(path(*name.split("/"))) for name in _FINAL_SCORING_SOURCES
    }
    unsigned = {
        "schema": "rsi-final-execution-plan-v1",
        "purpose": purpose,
        "configuration_sha256": sha256_obj(cfg),
        "config_fingerprint": config_fingerprint(cfg),
        "final_protocol_fingerprint": protocol_sha,
        "scoring_sources_sha256": source_hashes,
        "release_id": state.release_id,
        "release_artifacts": _release_fingerprint(state.release_id),
        "baseline_report_sha256": baseline_sha,
        "datasets": list(datasets),
        "seeds": [int(seed) for seed in seeds],
        "limit_per_dataset_seed": limit,
        "split_scopes": scopes,
        "freeze_sha256": str(frozen_report.get("freeze_sha256") or ""),
        "unblind_budget_snapshot": budget_snapshots,
        "force_reseal": bool(force),
        "actor": str(actor),
        "reason": str(reason),
        "budget_override": budget_override,
        "strict_protocol": bool(strict_protocol),
        "provider": str(runtime.get("provider") or "fake").lower(),
        "task_model_id": str(((cfg.get("models") or {}).get("task") or {}).get("model_id") or ""),
        "max_model_operations_per_task": model_ops,
        "http_attempts_per_operation_upper_bound": attempts["attempts"],
        "pi_provider_retries_after_first": attempts["provider_retries"],
        "pi_bridge_attempts_upper_bound": attempts["bridge_attempts"],
        "logical_request_cap": logical_cap,
        "physical_request_cap": physical_cap,
        "request_ledger_required": True,
        "final_holdout_task_rows_read_during_preflight": False,
        "selection_gate_read": False,
        "model_calls_during_preflight": 0,
        "live_authorized": False,
        "explicit_user_approval_required_before_execution": True,
    }
    return {**unsigned, "plan_sha256": sha256_obj(unsigned)}


def verify_final_execution_plan(plan: dict[str, Any], expected_sha256: str) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    actual = sha256_obj(unsigned)
    if plan.get("plan_sha256") != actual:
        raise ValueError("frozen final execution plan hash is internally inconsistent")
    if expected_sha256 != actual:
        raise ValueError("expected SHA-256 does not match the current final execution plan")
