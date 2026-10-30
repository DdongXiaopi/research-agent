"""Run a bounded Prompt-only GEPA search on visible dev_optimize tasks.

This entry point deliberately stops at a draft candidate. It does not run an
outer selection gate, access final_holdout, or publish a release.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.evolution.gepa_search import search_prompt_candidate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.util import artifacts_path, read_json, sha256_obj  # noqa: E402


def load_gepa_cohort_scopes(path: str | Path) -> tuple[str, list[str], list[str], str]:
    """Load only the optimizer train + search scopes from a frozen nested cohort."""
    cohort = read_json(path)
    recorded_sha = str(cohort.get("cohort_sha256") or "")
    hash_payload = dict(cohort)
    hash_payload.pop("cohort_sha256", None)
    if not recorded_sha or sha256_obj(hash_payload) != recorded_sha:
        raise ValueError("frozen cohort content hash is missing or invalid")
    if cohort.get("eligible_for_confirmation") is not False:
        raise ValueError("development cohort cannot be marked confirmation eligible")
    dataset = str(cohort.get("dataset") or "")
    if not dataset:
        raise ValueError("frozen cohort dataset is missing")
    scopes = cohort.get("scopes") or {}
    train_scope = scopes.get("mipro_train") or scopes.get("train")
    search_scope = scopes.get("search_validation")
    internal_scope = scopes.get("internal_confirmation")
    if not all(isinstance(scope, dict)
               for scope in (train_scope, search_scope, internal_scope)):
        raise ValueError("cohort must provide train, search_validation, and internal_confirmation scopes")
    train_ids = list(map(str, train_scope.get("task_ids") or []))
    search_ids = list(map(str, search_scope.get("task_ids") or []))
    internal_ids = list(map(str, internal_scope.get("task_ids") or []))
    if not train_ids or not search_ids or not internal_ids:
        raise ValueError("cohort scopes must each contain task IDs")
    if any(not task_id for task_id in train_ids + search_ids + internal_ids):
        raise ValueError("cohort scope contains an empty task ID")
    if (len(set(train_ids + search_ids + internal_ids))
            != len(train_ids) + len(search_ids) + len(internal_ids)):
        raise ValueError("cohort train, search, and internal task IDs must be unique and disjoint")
    return dataset, train_ids, search_ids, recorded_sha


def build_search_config(base: dict, *, task_provider: str, task_offset: int,
                        metric_calls: int, proposals: int, minibatch: int,
                        acceptance: str | None, reflection_model: str | None,
                        feedback_signals: bool, multiobjective: bool,
                        visible_train_task_ids: list[str] | None = None,
                        visible_validation_task_ids: list[str] | None = None,
                        cohort_sha256: str | None = None) -> dict:
    if task_offset < 0 or metric_calls <= 0 or proposals <= 0 or minibatch <= 0:
        raise ValueError("offset must be nonnegative and search limits must be positive")
    if acceptance is None:
        acceptance = "any_objective_improved" if multiobjective else "strict_improvement"
    if acceptance == "any_objective_improved" and not multiobjective:
        raise ValueError("any_objective_improved requires multiobjective=true")
    cfg = copy.deepcopy(base)
    cfg.setdefault("runtime", {}).update({
        "provider": task_provider,
        "input_mode": "task_input",
        "task_input_max_chars": 30000,
    })
    cfg.setdefault("evaluation", {}).update({
        "main_metric": "accuracy",
        "primary_axis": "accuracy",
        "sealed_pairing": False,
    })
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg["tools"].get("active", [])
        if name != "academic.search"]
    reflection = cfg.setdefault("models", {}).setdefault("reflection", {})
    if reflection_model:
        reflection["model_id"] = reflection_model
    gp = cfg.setdefault("evolution", {}).setdefault("gepa", {})
    gp.update({
        "enabled": True,
        "task_offset": task_offset,
        "max_metric_calls": metric_calls,
        "max_candidate_proposals": proposals,
        "reflection_minibatch_size": minibatch,
        "acceptance_criterion": acceptance,
        "feedback_evidence_signals": feedback_signals,
        "multi_objective": multiobjective,
    })
    if visible_train_task_ids is not None or visible_validation_task_ids is not None:
        if task_offset:
            raise ValueError("exact cohort IDs cannot be combined with a task offset")
        if not visible_train_task_ids or not visible_validation_task_ids:
            raise ValueError("exact train and validation task ID lists must both be non-empty")
        gp["visible_train_task_ids"] = list(visible_train_task_ids)
        gp["visible_validation_task_ids"] = list(visible_validation_task_ids)
        gp["cohort_sha256"] = cohort_sha256
    return cfg


def main() -> int:
    raise SystemExit(
        "retired_unplanned_entrypoint: this legacy GEPA Prompt-search CLI has no "
        "caller-challenged execution plan or request ledger; use a controlled, "
        "source-bound runner after its exact plan and request cap are approved")
    # The pinned vendor persists full prompt text with locale-default open().
    # Re-exec in UTF-8 mode on Windows so non-ASCII punctuation is safe to save.
    if os.name == "nt" and not sys.flags.utf8_mode:
        env = os.environ.copy()
        env["RSI_GEPA_UTF8_REEXEC"] = "1"
        return subprocess.run(
            [sys.executable, "-X", "utf8", *sys.argv], env=env, check=False).returncode
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--release", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--cohort", help=("optional frozen nested cohort JSON; GEPA sees only "
                        "mipro_train + search_validation task IDs and keeps internal_confirmation hidden"))
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261052)
    parser.add_argument("--task-provider", choices=["pi", "http"], default="pi")
    parser.add_argument("--metric-calls", type=int, default=90)
    parser.add_argument("--proposals", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=5)
    parser.add_argument("--acceptance", choices=["strict_improvement",
                                                    "improvement_or_equal",
                                                    "any_objective_improved"],
                        default=None)
    parser.add_argument("--reflection-model", default=None)
    parser.add_argument("--feedback-signals", action="store_true")
    parser.add_argument("--multiobjective", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("GEPA dev search requires an explicit isolated RSI_HOME")
    if args.limit < 3:
        raise ValueError("GEPA visible task limit must be at least 3")
    datasets = list(args.datasets or ["pubmedqa"])
    cohort_selected = bool(args.cohort)
    visible_train_ids = visible_validation_ids = cohort_sha = None
    if args.cohort:
        cohort_dataset, visible_train_ids, visible_validation_ids, cohort_sha = (
            load_gepa_cohort_scopes(args.cohort))
        if args.datasets and datasets != [cohort_dataset]:
            raise ValueError("--datasets must match the dataset recorded by --cohort")
        datasets = [cohort_dataset]
    cfg = build_search_config(
        load_config(args.config), task_provider=args.task_provider,
        task_offset=args.offset, metric_calls=args.metric_calls,
        proposals=args.proposals, minibatch=args.minibatch,
        acceptance=args.acceptance, reflection_model=args.reflection_model,
        feedback_signals=args.feedback_signals, multiobjective=args.multiobjective,
        visible_train_task_ids=visible_train_ids,
        visible_validation_task_ids=visible_validation_ids,
        cohort_sha256=cohort_sha)
    release_id = args.release or (
        "task-input-evidence-inference-v1" if cohort_selected
        else "task-input-baseline-v1")
    state = load_state(release_id)
    before = {p.resolve() for p in artifacts_path("search", "gepa").glob("*/report.json")}
    error = None
    try:
        candidate = search_prompt_candidate(
            cfg, state, datasets, args.limit, args.seed, failure_ids=[])
    except Exception as exc:
        candidate = None
        error = {"type": type(exc).__name__, "detail_omitted": True}
    reports = [p for p in artifacts_path("search", "gepa").glob("*/report.json")
               if p.resolve() not in before]
    report_path = max(reports, key=lambda p: p.stat().st_mtime) if reports else None
    print(json.dumps({
        "candidate_id": candidate.candidate_id if candidate else None,
        "candidate_status": candidate.status if candidate else "no_change",
        "report": str(report_path.resolve()) if report_path else None,
        "search_error": error,
        "selection_gate_read": False,
        "release_published": False,
    }, ensure_ascii=False))
    return 2 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
