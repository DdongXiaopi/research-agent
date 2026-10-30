"""Bridge to the pinned vendor GEPA engine for Prompt/Skill text search.

GEPA stays in ``vendor/`` and owns reflection, mutation and Pareto selection.
This module supplies our task evaluator and keeps selection_gate/final data out
of the optimizer. It deliberately requires Python >=3.10, as pinned GEPA does.
"""

from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List


class DependencyMissing(RuntimeError):
    pass


class TaskExecutionFailure(RuntimeError):
    """A non-scored task execution failure that must abort optimizer scoring."""

    def __init__(self, task_id: str, status: str):
        self.failure = {
            "task_id": str(task_id),
            "status": str(status),
            "score_valid": False,
        }
        super().__init__(
            f"GEPA task evaluation failed without a valid score: task={task_id} "
            f"status={status}")


_VENDOR_SRC = (Path(__file__).resolve().parents[2] / "vendor/gepa-main/gepa-main/src")


def require_gepa() -> Any:
    if sys.version_info < (3, 10):
        raise DependencyMissing("Pinned GEPA requires Python >=3.10")
    if not _VENDOR_SRC.exists():
        raise DependencyMissing(f"Pinned GEPA source missing: {_VENDOR_SRC}")
    source = str(_VENDOR_SRC)
    if source in sys.path:
        sys.path.remove(source)
    sys.path.insert(0, source)
    loaded = sys.modules.get("gepa")
    loaded_path = getattr(loaded, "__file__", None) if loaded is not None else None
    if loaded_path:
        try:
            Path(loaded_path).resolve().relative_to((_VENDOR_SRC / "gepa").resolve())
            loaded_is_pinned = True
        except ValueError:
            loaded_is_pinned = False
        if not loaded_is_pinned:
            # Another dependency may have imported a separately installed
            # package with the same top-level name. Drop that package tree so
            # Python loads the repository-pinned vendor snapshot consistently.
            for name in list(sys.modules):
                if name == "gepa" or name.startswith("gepa."):
                    del sys.modules[name]
    importlib.invalidate_caches()
    try:
        from gepa import gepa_launcher
    except ImportError as exc:
        raise DependencyMissing(f"Pinned GEPA import failed: {exc}") from exc
    return gepa_launcher


def is_available() -> bool:
    try:
        require_gepa()
        return True
    except DependencyMissing:
        return False


def build_metric(primary: str = "accuracy", *, multi_objective: bool = False) -> Callable[[Any], tuple]:
    """Turn a task outcome into GEPA's score + actionable side information."""
    if primary != "accuracy":
        raise ValueError("Only task accuracy is supported in this bridge")

    def metric(outcome: Any) -> tuple:
        status = str(getattr(outcome, "status", "error"))
        if status not in {"ok", "abstain"}:
            raise TaskExecutionFailure(
                str(getattr(outcome, "task_id", "unknown")), status)
        correct = bool(getattr(outcome, "correct", False))
        answer_index = int(getattr(outcome, "answer_index", -1))
        answered = answer_index >= 0
        side = {"Feedback": "correct" if correct else
                ("abstained" if status == "abstain" else "incorrect or error"),
                "Status": status,
                "AnswerIndex": answer_index,
                "InputMode": getattr(outcome, "input_mode", "retrieval")}
        if multi_objective:
            # Per-example utility vectors let GEPA retain candidates that
            # improve answer coverage or reduce wrong confident answers while
            # the outer evaluator still enforces its aggregate hard gates.
            side["scores"] = {
                "accuracy": 1.0 if correct else 0.0,
                "coverage": 1.0 if answered else 0.0,
                "answer_quality": (1.0 if correct else -1.0) if answered else 0.0,
            }
        return (1.0 if correct else 0.0, side)

    return metric


def build_optimizer(cfg: Dict[str, Any], reflection_lm: Callable,
                    run_dir: str, max_metric_calls: int = 120) -> Any:
    """Build a bounded, sequential GEPA search using the pinned vendor API."""
    if max_metric_calls < 1:
        raise ValueError("max_metric_calls must be positive")
    acceptance_criterion = str(cfg.get("acceptance_criterion", "strict_improvement"))
    multi_objective = bool(cfg.get("multi_objective", False))
    if acceptance_criterion == "any_objective_improved" and not multi_objective:
        raise ValueError("any_objective_improved requires multi_objective=true")
    valid_criteria = {"strict_improvement", "improvement_or_equal"}
    if multi_objective:
        valid_criteria.add("any_objective_improved")
    if acceptance_criterion not in valid_criteria:
        raise ValueError(
            "unsupported acceptance_criterion for configured objective mode")
    gepa = require_gepa()
    use_merge = bool(cfg.get("use_merge", False))
    merge_max_invocations = int(cfg.get("merge_max_invocations", 5))
    merge_val_overlap_floor = int(cfg.get("merge_val_overlap_floor", 5))
    if merge_max_invocations < 1 or merge_val_overlap_floor < 1:
        raise ValueError("merge limits must be positive")
    if acceptance_criterion == "any_objective_improved":
        class AnyObjectiveImproved:
            """Accept local moves that improve at least one per-task objective."""

            @staticmethod
            def should_accept(proposal: Any, _state: Any) -> bool:
                before = getattr(getattr(proposal, "eval_before", None),
                                 "objective_scores", None)
                after = getattr(getattr(proposal, "eval_after", None),
                                "objective_scores", None)
                if not before or not after:
                    old = sum(getattr(proposal, "subsample_scores_before", []) or [])
                    new = sum(getattr(proposal, "subsample_scores_after", []) or [])
                    return new > old
                names = set().union(*(x.keys() for x in before + after))
                return any(
                    sum(float(x.get(name, 0.0)) for x in after) >
                    sum(float(x.get(name, 0.0)) for x in before)
                    for name in names)

        engine_acceptance: Any = AnyObjectiveImproved()
    else:
        engine_acceptance = acceptance_criterion
    return gepa.GEPAConfig(
        engine=gepa.EngineConfig(
            run_dir=run_dir, seed=int(cfg.get("seed", 20260929)),
            max_metric_calls=max_metric_calls,
            max_candidate_proposals=int(cfg.get("max_candidate_proposals", 8)),
            acceptance_criterion=engine_acceptance,
            # Budget exhaustion must abort the iteration; a swallowed evaluator
            # exception could leave an incomplete batch looking like real scores.
            raise_on_exception=True,
            parallel=False, max_workers=1,
            cache_evaluation=True, display_progress_bar=False,
            write_agent_state=True),
        reflection=gepa.ReflectionConfig(
            reflection_lm=reflection_lm,
            reflection_minibatch_size=int(cfg.get("reflection_minibatch_size", 5))),
        merge=(gepa.MergeConfig(
            max_merge_invocations=merge_max_invocations,
            merge_val_overlap_floor=merge_val_overlap_floor,
        ) if use_merge else None),
    )


def summarize_vendor_proposals(run_log_path: str | Path) -> Dict[str, Any]:
    """Return a compact proposal ledger without copying task text into reports."""
    path = Path(run_log_path)
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"proposal_log_available": False}
    if not isinstance(entries, list):
        return {"proposal_log_available": False}

    trace = []
    accepted = rejected = unclassified = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        decision = entry.get("proposal_accepted")
        if decision is True:
            accepted += 1
        elif decision is False:
            rejected += 1
        else:
            unclassified += 1
        old_scores = entry.get("subsample_scores") or []
        new_scores = entry.get("new_subsample_scores") or []
        numeric_old = [float(x) for x in old_scores if isinstance(x, (int, float))]
        numeric_new = [float(x) for x in new_scores if isinstance(x, (int, float))]
        trace.append({
            "iteration": entry.get("i"),
            "iteration_id": entry.get("iteration_id"),
            "subsample_size": len(entry.get("subsample_ids") or []),
            "old_subsample_mean": (sum(numeric_old) / len(numeric_old)
                                   if numeric_old else None),
            "new_subsample_mean": (sum(numeric_new) / len(numeric_new)
                                   if numeric_new else None),
            "accepted": decision if isinstance(decision, bool) else None,
        })
    return {
        "proposal_log_available": True,
        "proposal_attempts": len(trace),
        "accepted_proposals": accepted,
        "rejected_proposals": rejected,
        "unclassified_proposals": unclassified,
        "proposal_trace": trace,
    }


def summarize_chat_usage(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate measured reflection calls without treating nominal cost as USD."""
    costs = [row.get("cost") or {} for row in calls]
    usage_complete = all(bool(row.get("usage_available", True)) for row in calls)
    fully_priced = bool(costs) and usage_complete and all(
        cost.get("usd") is not None and not cost.get("cost_estimated", False)
        for cost in costs)
    bases = sorted({str(cost["cost_basis"]) for cost in costs
                    if cost.get("cost_basis")})
    return {
        "calls": len(calls),
        "cache_hits": sum(bool(row.get("cache_hit")) for row in calls),
        "usage_complete": usage_complete,
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in calls),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in calls),
        "latency_ms": sum(int(row.get("latency_ms") or 0) for row in calls),
        "cost_usd": (round(sum(float(cost["usd"]) for cost in costs), 6)
                     if fully_priced else None),
        "cost_usd_nominal": (
            round(sum(float(cost["usd_nominal"]) for cost in costs), 6)
            if usage_complete and all(cost.get("usd_nominal") is not None
                                      for cost in costs) else None),
        "cost_estimated": not fully_priced,
        "cost_basis": "; ".join(bases),
    }


def run_reflection(*, seed_candidate: str, evaluator: Callable,
                   trainset: List[Dict[str, Any]], valset: List[Dict[str, Any]],
                   config: Any, objective: str, background: str = "") -> Any:
    """Run real GEPA on visible dev examples only; caller owns every split."""
    if not trainset or not valset:
        raise ValueError("GEPA requires nonempty dev train and validation examples")
    if any(x.get("split") != "dev_optimize" for x in trainset + valset):
        raise ValueError("GEPA may only see dev_optimize examples")

    def identities(rows: List[Dict[str, Any]], name: str) -> set[str]:
        found: set[str] = set()
        for row in rows:
            task_id = str(row.get("task_id") or "").strip()
            if not task_id:
                raise ValueError(f"GEPA {name} examples require task_id")
            if task_id in found:
                raise ValueError(f"GEPA {name} examples contain duplicate task IDs")
            found.add(task_id)
        return found

    train_ids = identities(trainset, "train")
    val_ids = identities(valset, "validation")
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(
            "GEPA train and validation task IDs must be disjoint: "
            + ", ".join(sorted(overlap)))
    # Pinned GEPA logs complete proposed text through its stdout tee. Keep
    # Unicode proposals from failing on Windows consoles configured as GBK.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")
    gepa = require_gepa()
    return gepa.optimize_anything(
        seed_candidate=seed_candidate, evaluator=evaluator,
        dataset=trainset, valset=valset, objective=objective,
        background=background, config=config)
