"""Replay an already-seen PubMedQA slice to diagnose a baseline label-contract fix.

This is a diagnostic-only paired replay. It intentionally reuses questions that
were already inspected, never produces an optimizer input, and can never qualify
as discovery, confirmation, or a candidate-selection result.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_fingerprint, load_config  # noqa: E402
from src.data.split import load_tasks  # noqa: E402
from src.eval.task_rows import validate_task_answer_row  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.provider import build_provider  # noqa: E402
from src.runtime.runner import TaskRunner  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.util import (  # noqa: E402
    artifacts_path, ensure_dir, read_json, read_jsonl, sha256_file, sha256_obj,
    write_json, write_jsonl,
)

OLD_PROTOCOL = "pubmedqa-threeway-v2-skill-routing-baseline-v2"
NEW_RELEASES = {
    "task-input-baseline-v2-label-compatible",
    "task-input-baseline-v2-minimal-label-compatible",
}


def compare_rows(old_rows: list[dict], new_rows: list[dict]) -> dict:
    """Compare exact task/seed pairs and count unique-question transitions."""
    key = lambda row: (int(row["seed"]), str(row["task_id"]))
    old = {key(row): row for row in old_rows}
    new = {key(row): row for row in new_rows}
    if (not old or len(old) != len(old_rows) or len(new) != len(new_rows)
            or set(old) != set(new)):
        raise ValueError("old and new diagnostic rows must cover identical task/seed pairs")
    transitions = {"wrong_to_correct": [], "correct_to_wrong": [],
                   "same_correct": [], "same_wrong": []}
    for pair in sorted(old):
        was = bool(old[pair]["correct"])
        now = bool(new[pair]["correct"])
        name = ("wrong_to_correct" if not was and now else
                "correct_to_wrong" if was and not now else
                "same_correct" if was else "same_wrong")
        transitions[name].append(pair[1])
    task_sets = {name: sorted(set(ids)) for name, ids in transitions.items()}
    return {
        "paired_rows": len(old),
        "old_correct": sum(bool(row["correct"]) for row in old.values()),
        "new_correct": sum(bool(row["correct"]) for row in new.values()),
        "old_accuracy": sum(bool(row["correct"]) for row in old.values()) / len(old),
        "new_accuracy": sum(bool(row["correct"]) for row in new.values()) / len(new),
        "transitions_by_call": {name: len(ids) for name, ids in transitions.items()},
        "unique_task_ids_by_transition": {name: ids for name, ids in task_sets.items()},
    }


def _metrics(rows: list[dict], task_ids: list[str]) -> dict:
    selected = [row for row in rows if str(row["task_id"]) in set(task_ids)]
    if (len(selected) != len(task_ids)
            or len({str(row["task_id"]) for row in selected}) != len(task_ids)):
        raise ValueError("diagnostic rows do not cover the exact task scope")
    n = len(selected)
    answered = sum(int(row.get("answer_index", -1)) >= 0 for row in selected)
    correct = sum(bool(row.get("correct")) for row in selected)
    per_label = {}
    for label in sorted({str(row["gold_label"]).strip().casefold()
                         for row in selected}):
        subset = [row for row in selected
                  if str(row["gold_label"]).strip().casefold() == label]
        label_answered = sum(int(row.get("answer_index", -1)) >= 0 for row in subset)
        label_correct = sum(bool(row.get("correct")) for row in subset)
        per_label[label] = {
            "n": len(subset), "accuracy": label_correct / len(subset),
            "coverage": label_answered / len(subset),
            "answered_precision": label_correct / label_answered if label_answered else 0.0,
        }
    return {
        "n": n, "accuracy": correct / n,
        "coverage": answered / n,
        "answered_precision": correct / answered if answered else 0.0,
        "per_gold_label": per_label,
    }


def build_execution_plan(*, baseline_path: Path, plan_path: Path, baseline: dict,
                         state, cfg: dict, task_ids: list[str], seeds: list[int]) -> dict:
    """Freeze the exact diagnostic replay scope and conservative request ceiling."""
    runtime = cfg.get("runtime") or {}
    task_model = (cfg.get("models") or {}).get("task") or {}
    router = cfg.get("skills") or {}
    max_model_calls = max(1, int(runtime.get("max_model_calls", 12)))
    pi_retries = max(0, min(4, int(runtime.get("pi_provider_max_retries", 4))))
    pi_bridge_attempts = max(1, min(3, int(runtime.get("pi_timeout_max_attempts", 2))))
    pi_attempts = (pi_retries + 1) * pi_bridge_attempts
    http_attempts = max(1, int(task_model.get("max_retries", 6)))
    per_operation_physical = max(pi_attempts, http_attempts)
    task_runs = len(task_ids) * len(seeds)
    logical_cap = task_runs * max_model_calls
    physical_cap = logical_cap * per_operation_physical
    unsigned = {
        "schema": "pubmedqa-label-compatible-diagnostic-execution-plan-v1",
        "role": "previously_seen_diagnostic_only",
        "source_report_sha256": sha256_file(baseline_path),
        "historical_cohort_plan_sha256": sha256_file(plan_path),
        "source_baseline_identity_sha256": baseline.get("baseline_identity_sha256"),
        "source_release_id": baseline.get("release_id"),
        "diagnostic_release_id": state.release_id,
        "registry_sha256": state.registry.fingerprint(),
        "config_sha256": config_fingerprint(cfg),
        "task_provider": str(runtime.get("provider") or ""),
        "task_model_id": str(task_model.get("model_id") or ""),
        "skill_router_provider": str(router.get("router_provider") or ""),
        "skill_router_model_id": str(router.get("router_model_id") or ""),
        "task_ids_already_seen": list(task_ids),
        "execution_labels": list(seeds),
        "task_runner_count_upper_bound": task_runs,
        "model_operations_per_task_runner_upper_bound": max_model_calls,
        "pi_provider_retries_after_first": pi_retries,
        "pi_timeout_subprocess_attempts": pi_bridge_attempts,
        "pi_http_attempts_per_operation_upper_bound": pi_attempts,
        "http_attempts_per_operation_upper_bound": http_attempts,
        "logical_request_cap": logical_cap,
        "physical_request_cap": physical_cap,
        "request_ledger_required": True,
        "pi_physical_reservations_required": True,
        "model_calls_during_preflight": 0,
        "selection_gate_read": False,
        "confirmation_ids_read": False,
        "final_holdout_read": False,
        "candidate_selection_allowed": False,
        "confirmation_eligible": False,
        "independent_confirmation_eligible": False,
        "live_authorized": False,
        "explicit_user_approval_required_before_execution": True,
    }
    return {**unsigned, "plan_sha256": sha256_obj(unsigned)}


def verify_execution_plan(plan: dict, expected_sha256: str) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    actual = sha256_obj(unsigned)
    if plan.get("plan_sha256") != actual:
        raise ValueError("frozen diagnostic plan hash is internally inconsistent")
    if expected_sha256 != actual:
        raise ValueError("expected plan hash does not match the current diagnostic plan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", required=True,
                        help="a previously recorded v2 baseline report")
    parser.add_argument("--config", required=True,
                        help="matched task-model and Skill-router config")
    parser.add_argument("--expected-task-model", default="Qwen/Qwen3-32B",
                        help="require this exact task-model ID for the paired replay")
    parser.add_argument("--release", default="task-input-baseline-v2-label-compatible")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true",
                      help="freeze and write the zero-call execution plan")
    mode.add_argument("--execute", action="store_true",
                      help="execute only the exact plan explicitly approved below")
    parser.add_argument("--expected-plan-sha256", default="",
                        help="required with --execute; must match the frozen plan")
    args = parser.parse_args()
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("diagnostic replay requires an isolated RSI_HOME")

    baseline_path = Path(args.baseline_report).resolve()
    baseline = read_json(baseline_path)
    if baseline.get("protocol") != OLD_PROTOCOL:
        raise ValueError("source is not the expected historical PubMedQA v2 baseline")
    if (baseline.get("confirmation_labels_accessed") is not False
            or baseline.get("final_holdout_accessed") is not False):
        raise ValueError("historical source does not attest isolation")
    if baseline.get("release_id") != "task-input-baseline-v2":
        raise ValueError("historical source must use the old v2 prompt baseline")

    plan_path = Path(str(baseline["cohort_plan"])).resolve()
    plan = read_json(plan_path)
    if sha256_file(plan_path) != baseline.get("cohort_plan_sha256"):
        raise ValueError("historical cohort plan hash changed")
    task_ids = [str(value) for value in baseline.get("discovery_ids", [])]
    expected_source_scope = (set(baseline.get("train_ids", []))
                             | set(baseline.get("discovery_ids", [])))
    if (not task_ids or set(baseline.get("evaluated_ids", [])) != expected_source_scope
            or len(baseline.get("evaluated_ids", [])) != len(expected_source_scope)):
        raise ValueError("historical source must contain only its frozen train+discovery scope")
    if any(task_id not in set(plan.get("discovery_ids", [])) for task_id in task_ids):
        raise ValueError("diagnostic scope escaped the historical discovery slice")

    tasks_all = {str(task["task_id"]): task for task in
                 load_tasks("pubmedqa_threeway_probe", splits=["dev_optimize"])}
    if not set(task_ids) <= tasks_all.keys():
        raise ValueError("historical discovery slice is absent from the frozen dev dataset")
    tasks = {task_id: tasks_all[task_id] for task_id in task_ids}
    old_rows = []
    baseline_runs = baseline.get("baseline_runs") or []
    seeds = [int(run["seed"]) for run in baseline_runs]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError("source needs at least three distinct execution labels")
    for run in baseline_runs:
        path = artifacts_path("runs", str(run["run_id"]), "per_task.jsonl")
        rows = read_jsonl(path)
        matched = [row for row in rows if str(row.get("task_id")) in set(task_ids)]
        if {str(row.get("task_id")) for row in matched} != set(task_ids):
            raise ValueError(f"old run {run['run_id']} does not cover historical discovery IDs")
        for row in matched:
            checked = validate_task_answer_row(
                {**row, "seed": int(run["seed"])}, tasks[str(row["task_id"])],
                context=f"old-diagnostic:{run['seed']}:{row['task_id']}",
                require_reported_gold_index=True)
            old_rows.append({**checked, "seed": int(run["seed"])})

    cfg = copy.deepcopy(load_config(args.config))
    cfg.setdefault("runtime", {}).update({
        "provider": "pi", "input_mode": "task_input", "task_input_max_chars": 30000,
        "pi_request_cooldown_seconds": 6,
    })
    cfg["runtime"].pop("pi_fake_response", None)
    cfg.setdefault("evaluation", {}).update({
        "insufficient_option": False, "main_metric": "accuracy",
        "primary_axis": "accuracy", "sealed_pairing": False,
    })
    cfg.setdefault("skills", {})["routing_mode"] = str(baseline["skill_routing_mode"])
    cfg["skills"].setdefault("router_model_id", cfg["models"]["task"].get("model_id"))
    cfg.setdefault("evolution", {})["allowed_artifacts"] = ["system_prompt"]
    cfg["evolution"].setdefault("fewshot", {}).update({"enabled": False, "k": 0})
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg["tools"].get("active", []) if name != "academic.search"]
    if str(cfg["models"]["task"].get("model_id", "")) != args.expected_task_model:
        raise ValueError("configured task model differs from --expected-task-model")
    if "8b" not in str(cfg.get("skills", {}).get("router_model_id", "")).casefold():
        raise ValueError("diagnostic must keep the historical 8B Skill router")

    state = load_state(args.release)
    if state.release_id not in NEW_RELEASES or len(state.registry.skills) != 10:
        raise ValueError("diagnostic requires an approved label-compatible 10-Skill release")
    key_env = str(cfg["models"]["task"].get("api_key_env") or "").strip()
    api_key_configured = bool(key_env and os.environ.get(key_env))
    execution_plan = build_execution_plan(
        baseline_path=baseline_path, plan_path=plan_path, baseline=baseline,
        state=state, cfg=cfg, task_ids=task_ids, seeds=seeds)
    plan_output_path = artifacts_path(
        "probe", "pubmedqa_label_compatibility",
        "diagnostic-plan-" + execution_plan["plan_sha256"] + ".json")
    if plan_output_path.exists():
        if read_json(plan_output_path) != execution_plan:
            raise ValueError("saved diagnostic plan differs from the recomputed plan")
    else:
        write_json(plan_output_path, execution_plan)
    identity = execution_plan["plan_sha256"]
    run_dir = artifacts_path("probe", "pubmedqa_label_compatibility",
                             "diagnostic-" + identity[:12])
    ensure_dir(run_dir)
    checkpoint_path = run_dir / "checkpoint.jsonl"

    if args.validate_only:
        print(json.dumps({
            "status": "validated_not_run", "diagnostic_only": True,
            "plan": str(plan_output_path), "plan_sha256": identity,
            "task_ids_already_seen": len(task_ids), "execution_labels": seeds,
            "release_id": state.release_id,
            "task_model": cfg["models"]["task"]["model_id"],
            "skill_router": cfg["skills"]["router_model_id"],
            "config_sha256": config_fingerprint(cfg), "model_calls": 0,
            "api_key_configured": api_key_configured,
            "candidate_selection_allowed": False,
            "confirmation_eligible": False,
            "logical_request_cap": execution_plan["logical_request_cap"],
            "physical_request_cap": execution_plan["physical_request_cap"],
            "selection_gate_read": False, "final_holdout_read": False,
        }, ensure_ascii=False))
        return 0
    verify_execution_plan(execution_plan, args.expected_plan_sha256)
    if not api_key_configured:
        raise RuntimeError("diagnostic replay is missing the configured task-model API key")

    ledger_path = run_dir / "call-ledger.jsonl"
    if ledger_path.exists():
        raise RuntimeError("diagnostic request ledger already exists; refusing an unbounded resume")
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(execution_plan["plan_sha256"])
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path)
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(execution_plan["logical_request_cap"])
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(execution_plan["physical_request_cap"])
    os.environ["RSI_CALL_LEDGER_RESERVE_PI_PHYSICAL"] = "1"

    saved = {}
    if checkpoint_path.exists():
        for row in read_jsonl(checkpoint_path):
            key = (int(row["seed"]), str(row["task_id"]))
            if key[0] not in seeds or key[1] not in tasks:
                raise ValueError("diagnostic checkpoint contains an out-of-scope row")
            if row.get("status") in {"ok", "abstain"}:
                row = validate_task_answer_row(
                    row, tasks[key[1]], context=f"diagnostic-checkpoint:{key}",
                    require_reported_gold_index=True)
            saved[key] = row

    new_rows = []
    for seed in seeds:
        provider = build_provider(cfg, seed=seed)
        for task_id in task_ids:
            key = (seed, task_id)
            row = saved.get(key)
            if row is None or row.get("status") not in {"ok", "abstain"}:
                # Match the project baseline runner: TaskRunner call budgets
                # are per task, while the provider may be reused per seed.
                outcome = TaskRunner(cfg, state, provider=provider, seed=seed).run(
                    tasks[task_id])
                row = outcome.to_dict()
                row.update({
                    "seed": seed,
                    "gold_label": str(tasks[task_id]["options"][
                        tasks[task_id]["correct_index"]]).strip().casefold(),
                    "config_sha256": config_fingerprint(cfg),
                })
                with checkpoint_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                saved[key] = row
            if row.get("status") not in {"ok", "abstain"}:
                raise RuntimeError(
                    f"task execution failed for {seed}/{task_id}: {row.get('error', '')}")
            row = validate_task_answer_row(
                row, tasks[task_id], context=f"new-diagnostic:{seed}:{task_id}",
                require_reported_gold_index=True)
            new_rows.append({**row, "seed": seed})

    paired = compare_rows(old_rows, new_rows)
    old_metrics = {str(seed): _metrics([r for r in old_rows if int(r["seed"]) == seed], task_ids)
                   for seed in seeds}
    new_metrics = {str(seed): _metrics([r for r in new_rows if int(r["seed"]) == seed], task_ids)
                   for seed in seeds}
    report = {
        "protocol": "pubmedqa-label-compatible-prompt-replay-diagnostic-v1",
        "status": "completed", "diagnostic_only": True,
        "previously_seen_task_ids": task_ids,
        "source_baseline_report": str(baseline_path),
        "source_baseline_report_sha256": sha256_file(baseline_path),
        "source_baseline_identity_sha256": baseline.get("baseline_identity_sha256"),
        "old_release_id": baseline.get("release_id"),
        "new_release_id": state.release_id,
        "only_intended_variable": "label-compatible system prompt",
        "task_model": cfg["models"]["task"]["model_id"],
        "skill_router": cfg["skills"]["router_model_id"],
        "api_key_configured": True,
        "config_sha256": config_fingerprint(cfg),
        "execution_labels": seeds,
        "execution_labels_are_independent_model_rng_seeds": False,
        "retrieval_used": False,
        "candidate_selection_allowed": False,
        "confirmation_eligible": False,
        "final_holdout_accessed": False,
        "old_metrics_by_execution_label": old_metrics,
        "new_metrics_by_execution_label": new_metrics,
        "paired_summary": paired,
        "new_per_task": str(run_dir / "per_task.jsonl"),
        "checkpoint": str(checkpoint_path),
        "execution_plan_sha256": execution_plan["plan_sha256"],
        "execution_plan_path": str(plan_output_path),
        "request_ledger": str(ledger_path),
        "request_ledger_summary": summarize_ledger(ledger_path),
    }
    write_jsonl(run_dir / "per_task.jsonl", new_rows)
    write_json(run_dir / "report.json", report)
    print(json.dumps({"report": str(run_dir / "report.json"),
                      "paired_summary": paired,
                      "old_metrics_by_execution_label": old_metrics,
                      "new_metrics_by_execution_label": new_metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
