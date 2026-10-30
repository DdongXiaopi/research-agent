"""Paired three-arm, fixed-slice comparison of frozen task-demo bundles.

This exploratory probe compares the baseline, a frozen search winner, and a
predeclared answer-label-mixed bundle. It never selects a bundle on the probe
slice and cannot access selection_gate or final_holdout.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_fingerprint, load_config  # noqa: E402
from src.contracts.types import Candidate  # noqa: E402
from src.data.split import load_tasks  # noqa: E402
from src.eval.options import inject_all  # noqa: E402
from src.evolution.fewshot import parse_demos_bundle, render_demos_block  # noqa: E402
from src.evolution.task_input_gates import validate_task_demo_rendering  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.eval.task_rows import validate_task_answer_row  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.runtime.runner import TaskRunner  # noqa: E402
from src.util import (artifacts_path, read_json, sha256_file, sha256_obj,
                      sha256_text, write_json)  # noqa: E402


def _row(task: Dict[str, Any], outcome: Any) -> Dict[str, Any]:
    if (str(getattr(outcome, "task_id", "")) != str(task["task_id"])
            or str(getattr(outcome, "dataset", "")) != str(task.get("dataset", ""))
            or str(getattr(outcome, "split", "")) != str(task.get("split", "dev_optimize"))):
        raise ValueError("outcome identity differs from frozen task")
    row = {"task_id": str(outcome.task_id),
            "dataset": str(outcome.dataset),
            "split": str(outcome.split),
            "answer_index": outcome.answer_index,
            "gold_index": outcome.gold_index,
            "correct": outcome.correct,
            "status": outcome.status}
    row = validate_task_answer_row(row, task, context="task-demo bundle comparison",
                                   require_reported_gold_index=True)
    row.update({
        "latency_ms": int(getattr(outcome, "latency_ms", 0) or 0),
        "cost": dict(getattr(outcome, "cost", {}) or {}),
        "pi_session_path": getattr(outcome, "pi_session_path", None),
        "bridge_attempts": getattr(outcome, "pi_bridge_attempts", None),
        "bridge_timeouts": getattr(outcome, "pi_bridge_timeouts", None),
    })
    return row


def _plan_hash(plan: Dict[str, Any]) -> str:
    return sha256_obj({key: value for key, value in plan.items()
                       if key != "plan_sha256"})


def _task_demo_plan(cfg: Dict[str, Any], state: Any, tasks: List[Dict[str, Any]],
                    candidates: Dict[str, Dict[str, Any]], args: Any) -> Dict[str, Any]:
    runtime = cfg.get("runtime") or {}
    task_calls = len(tasks) * len(args.seeds) * 3
    pi_retries = max(0, min(4, int(runtime.get("pi_provider_max_retries", 4))))
    bridge_attempts = max(1, min(3, int(runtime.get("pi_timeout_max_attempts", 2))))
    logical_by_component = {"task_answer": task_calls}
    physical_by_component = {"task_answer": task_calls * (pi_retries + 1) * bridge_attempts}
    skills = cfg.get("skills") or {}
    model_router = (str(skills.get("routing_mode", "lexical")) == "model"
                    and bool(getattr(state.registry, "skills", []))
                    and not str(skills.get("pinned_skill") or ""))
    router_attempts = max(1, int((cfg.get("models", {}).get("task") or {})
                                 .get("max_retries", 6)))
    if model_router:
        logical_by_component["skill_router"] = task_calls
        physical_by_component["skill_router"] = task_calls * router_attempts
    unsigned = {
        "schema": "task-demo-bundle-comparison-plan-v1",
        "role": "exploratory_dev_optimize_paired_comparison_not_selection",
        "dataset": "pubmedqa", "split": "dev_optimize",
        "start_index": args.start_index, "limit": args.limit,
        "eval_ids": [str(task["task_id"]) for task in tasks],
        "task_material_sha256": sha256_obj(tasks),
        "seeds": list(args.seeds),
        "candidate_files": {key: value["artifact_sha256"]
                             for key, value in candidates.items()},
        "candidate_ids": list(candidates),
        "candidate_demo_ids": {key: value["ids"] for key, value in candidates.items()},
        "candidate_demo_gold_labels": {key: value["labels"] for key, value in candidates.items()},
        "candidate_blocks": {key: sha256_text(value["block"])
                             for key, value in candidates.items()},
        "candidate_static_gates": {key: value["static_gate"]
                                   for key, value in candidates.items()},
        "candidate_render_gates": {key: value["render_gate"]
                                   for key, value in candidates.items()},
        "config_sha256": config_fingerprint(cfg),
        "release_id": state.release_id,
        "system_prompt_sha256": sha256_obj(state.system_prompt),
        "registry_sha256": state.registry.fingerprint(),
        "task_demos_sha256": sha256_obj(state.task_demos),
        "task_model_id": str((cfg.get("models", {}).get("task") or {}).get("model_id") or ""),
        "logical_by_component": logical_by_component,
        "physical_by_component": physical_by_component,
        "logical_request_cap": sum(logical_by_component.values()),
        "physical_request_cap": sum(physical_by_component.values()),
        "pi_physical_reservations_required": True,
        "model_calls_during_preflight": 0,
        "selection_gate_read": False, "final_holdout_read": False,
        "candidate_selection": False, "independent_confirmation": False,
    }
    return {**unsigned, "plan_sha256": _plan_hash(unsigned)}


def _metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    answered = sum(int(row["answer_index"]) >= 0 for row in rows)
    return {"n": n, "correct": correct, "answered": answered,
            "accuracy": correct / n if n else 0.0,
            "coverage": answered / n if n else 0.0,
            "answered_precision": correct / answered if answered else 0.0}


def _transition(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> str:
    b_answered = int(baseline["answer_index"]) >= 0
    c_answered = int(candidate["answer_index"]) >= 0
    if not b_answered and not c_answered:
        return "abstain_to_abstain"
    if not b_answered:
        return "abstain_to_correct" if candidate["correct"] else "abstain_to_wrong"
    if not c_answered:
        return "correct_to_abstain" if baseline["correct"] else "wrong_to_abstain"
    if not baseline["correct"] and candidate["correct"]:
        return "wrong_to_correct"
    if baseline["correct"] and not candidate["correct"]:
        return "correct_to_wrong"
    return "unchanged"


def _cluster_ci(pairs: List[Dict[str, Any]], arm: str, seed: int,
                samples: int = 10000) -> Dict[str, List[float]]:
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for pair in pairs:
        by_id.setdefault(str(pair["task_id"]), []).append(pair)
    ids = sorted(by_id)
    rng = random.Random(seed)
    deltas = {key: [] for key in ("accuracy", "coverage", "answered_precision")}
    for _ in range(samples):
        sampled_ids = [ids[rng.randrange(len(ids))] for _ in ids]
        b = [row["baseline"] for task_id in sampled_ids for row in by_id[task_id]]
        c = [row["candidates"][arm] for task_id in sampled_ids
             for row in by_id[task_id]]
        mb, mc = _metrics(b), _metrics(c)
        for key in deltas:
            deltas[key].append(mc[key] - mb[key])
    result = {}
    for key, values in deltas.items():
        values.sort()
        result[key] = [values[int(0.025 * samples)],
                       values[min(samples - 1, int(0.975 * samples))]]
    return result


def _make_config() -> Dict[str, Any]:
    cfg = copy.deepcopy(load_config("configs/experiment.http.pubmedqa-pilot.yaml"))
    cfg["runtime"].update({"provider": "pi", "input_mode": "task_input",
                           "task_input_max_chars": 30000})
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg.get("tools", {}).get("active", [])
        if name != "academic.search"]
    return cfg


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True,
                        help="candidate JSON; provide exactly two bundles")
    parser.add_argument("--start-index", type=int, default=250)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[20261023, 20261024, 20261025])
    parser.add_argument("--logical-request-budget", type=int, default=0)
    parser.add_argument("--physical-request-budget", type=int, default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-plan-sha256", default="")
    args = parser.parse_args(argv)
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("probe requires an explicit isolated sandbox RSI_HOME")
    if len(args.candidate) != 2:
        raise ValueError("provide the frozen winner and one predeclared comparison bundle")
    if args.start_index < 0 or args.limit <= 0:
        raise ValueError("start-index must be nonnegative and limit must be positive")
    if (args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256 or "")):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")

    cfg = _make_config()
    state = load_state()
    all_tasks = sorted(load_tasks("pubmedqa", splits=["dev_optimize"]),
                       key=lambda task: str(task["task_id"]))
    if cfg.get("evaluation", {}).get("insufficient_option", True):
        all_tasks = inject_all(all_tasks)
    tasks = all_tasks[args.start_index:args.start_index + args.limit]
    if len(tasks) != args.limit:
        raise ValueError(f"requested {args.limit} tasks but only found {len(tasks)}")
    eval_ids = {str(task["task_id"]) for task in tasks}

    candidates = {}
    source_ids = set()
    for candidate_path in args.candidate:
        artifact = read_json(Path(candidate_path))
        if artifact.get("artifact_type") != "task_demos":
            raise ValueError("all comparison artifacts must be task_demos")
        demos = parse_demos_bundle(str(artifact.get("new_content") or ""))
        render_gate = validate_task_demo_rendering(demos, cfg)
        if not render_gate.get("passed"):
            raise ValueError(f"demo bundle fails runtime render gate: {render_gate}")
        demo_ids = [str(item["task_id"]) for item in demos]
        if not demos or len(demo_ids) != len(set(demo_ids)):
            raise ValueError("demo bundle must be nonempty with unique IDs")
        if eval_ids & set(demo_ids):
            raise ValueError("demo/evaluation ID overlap")
        generator = artifact.get("generator") or {}
        candidate_source_ids = set(map(str, generator.get("train_ids", [])))
        candidate_source_ids.update(map(str, generator.get("val_ids", [])))
        source_ids.update(candidate_source_ids)
        if eval_ids & source_ids:
            raise ValueError("candidate search train/val overlaps evaluation slice")
        candidate = Candidate(**{
            key: artifact[key] for key in Candidate.__annotations__ if key in artifact})
        gate = static_gate(candidate, cfg, state)
        if not gate.get("passed"):
            raise ValueError(f"candidate failed static/source-integrity gate: {gate}")
        block = render_demos_block(demos)
        if not block:
            raise ValueError("demo bundle rendered to an empty block")
        key = str(artifact.get("candidate_id"))
        if not key or key in candidates:
            raise ValueError("candidate IDs must be present and unique")
        labels = [item["task"]["options"][int(item["correct_index"])]
                  for item in demos]
        candidates[key] = {"artifact": artifact, "ids": demo_ids,
                           "labels": labels, "block": block,
                           "static_gate": gate, "render_gate": render_gate,
                           "artifact_sha256": sha256_file(Path(candidate_path))}

    plan = _task_demo_plan(cfg, state, tasks, candidates, args)
    if args.execute:
        if args.expected_plan_sha256 != plan["plan_sha256"]:
            raise ValueError("execution plan hash mismatch")
        if (args.logical_request_budget != plan["logical_request_cap"]
                or args.physical_request_budget != plan["physical_request_cap"]):
            raise ValueError("request budgets must exactly match the frozen plan")
    plan_path = artifacts_path("probe", "task_demo_bundle_comparison",
                               "preflight-" + plan["plan_sha256"] + ".json")
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("saved plan differs from recomputed plan")
    if not plan_path.exists():
        write_json(plan_path, plan)
    if args.validate_only:
        print(json.dumps({"status": "validated_not_run", "plan": str(plan_path),
                          "plan_sha256": plan["plan_sha256"],
                          "logical_request_cap": plan["logical_request_cap"],
                          "physical_request_cap": plan["physical_request_cap"],
                          "model_calls": 0}, ensure_ascii=False))
        return 0

    task_model = (cfg.get("models") or {}).get("task") or {}
    key_env = str(task_model.get("api_key_env") or "").strip()
    if not key_env or not os.environ.get(key_env):
        raise RuntimeError("comparison execution is missing the configured task-model credential")

    arm_cfgs = {}
    for key, value in candidates.items():
        variant = copy.deepcopy(cfg)
        variant["runtime"].update({"pi_task_demos_block": value["block"],
                                   "pi_task_demo_ids": value["ids"],
                                   "pi_task_demo_split": "dev_optimize"})
        arm_cfgs[key] = variant

    pairs: List[Dict[str, Any]] = []
    output = artifacts_path("probe", "task_demo_bundle_comparison",
                            "comparison-" + uuid.uuid4().hex[:12] + ".json")
    report: Dict[str, Any] = {
        "protocol": "fixed-task-demo-two-bundle-paired-comparison-v1-exploratory",
        **plan,
        "status": "running",
        "candidate_sources": {key: value["artifact"].get("content_sha256")
                               for key, value in candidates.items()},
        "baseline_config_sha256": config_fingerprint(cfg),
        "candidate_config_sha256": {key: config_fingerprint(value)
                                     for key, value in arm_cfgs.items()},
        "release_id": state.release_id,
        "pairs": pairs,
        "execution_plan": plan,
        "request_ledger_path": str((artifacts_path("probe", "task_demo_bundle_comparison")
                                     / ("ledger-" + uuid.uuid4().hex + ".jsonl")).resolve()),
    }
    ledger_path = Path(report["request_ledger_path"])
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(plan["plan_sha256"])
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path)
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(args.logical_request_budget)
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(args.physical_request_budget)
    os.environ["RSI_CALL_LEDGER_RESERVE_PI_PHYSICAL"] = "1"
    for component, count in plan["logical_by_component"].items():
        os.environ["RSI_CALL_LEDGER_MAX_COMPONENT_" + component.upper()] = str(count)
    write_json(output, report)

    try:
        for seed in args.seeds:
            for index, task in enumerate(tasks, start=1):
                try:
                    base = TaskRunner(cfg, state, seed=seed).run(task)
                    baseline = _row(task, base)
                    arms = {}
                    for key, variant_cfg in arm_cfgs.items():
                        outcome = TaskRunner(variant_cfg, state, seed=seed).run(task)
                        arms[key] = _row(task, outcome)
                except Exception as exc:
                    report.update({"status": "invalid_evaluation_no_score",
                                   "invalid_task_id": str(task["task_id"]),
                                   "failure_type": type(exc).__name__,
                                   "request_ledger": summarize_ledger(ledger_path),
                                   "complete_paired_score_available": False})
                    write_json(output, report)
                    print(json.dumps({"output": str(output), "status": report["status"]},
                                     ensure_ascii=False))
                    return 2
                pairs.append({"seed": seed, "task_id": str(task["task_id"]),
                              "baseline": baseline, "candidates": arms,
                              "transitions": {key: _transition(baseline, outcome)
                                              for key, outcome in arms.items()}})
                report["completed_pairs"] = len(pairs)
                write_json(output, report)
                print(json.dumps({"seed": seed, "completed_within_seed": index,
                                  "total_per_seed": len(tasks),
                                  "task_id": str(task["task_id"])}, ensure_ascii=False),
                      flush=True)
    except Exception as exc:
        report.update({"status": "invalid_evaluation_no_score",
                       "failure_type": type(exc).__name__,
                       "request_ledger": summarize_ledger(ledger_path),
                       "complete_paired_score_available": False})
        write_json(output, report)
        return 2

    aggregate = {"baseline": _metrics([pair["baseline"] for pair in pairs])}
    transitions = {}
    for key in candidates:
        rows = [pair["candidates"][key] for pair in pairs]
        aggregate[key] = _metrics(rows)
        aggregate[key]["delta_vs_baseline"] = {
            metric: aggregate[key][metric] - aggregate["baseline"][metric]
            for metric in ("accuracy", "coverage", "answered_precision")}
        aggregate[key]["task_cluster_bootstrap_95ci"] = _cluster_ci(
            pairs, key, seed=20261004 + len(transitions))
        transitions[key] = dict(Counter(pair["transitions"][key] for pair in pairs))
    all_outcomes = [pair["baseline"] for pair in pairs] + [
        row for pair in pairs for row in pair["candidates"].values()]
    attempts_known = all(row["bridge_attempts"] is not None for row in all_outcomes)
    timeouts_known = all(row["bridge_timeouts"] is not None for row in all_outcomes)
    report.update({"status": "completed", "aggregate": aggregate,
                   "transition_counts": transitions,
                   "usage_observability": {
                       "total_logical_task_calls": len(all_outcomes),
                       "total_bridge_attempts": (sum(int(row["bridge_attempts"])
                                                      for row in all_outcomes)
                                                  if attempts_known else None),
                       "total_bridge_timeouts": (sum(int(row["bridge_timeouts"])
                                                      for row in all_outcomes)
                                                  if timeouts_known else None),
                       "usd_cost_auditable": False},
                   "usd_cost": None,
                   "request_ledger": summarize_ledger(ledger_path),
                   "effective_independent_task_count": len(tasks),
                   "interpretation_limits": [
                       "Three repeat labels share the same task IDs; cluster by task and do not count repeated requests as independent tasks.",
                       "Development-split exploration only; not a sealed confirmation set.",
                       "No candidate is selected or approved by this probe.",
                       "USD price/cache tariffs are unregistered; no cost gate is passed.",
                   ]})
    write_json(output, report)
    print(json.dumps({"output": str(output), "aggregate": aggregate,
                      "transition_counts": transitions}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
