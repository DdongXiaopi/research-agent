"""Paired multi-arm probe for visible task-input decision rules.

This is exploratory only: it reads dev_optimize, never reads selection_gate or
final_holdout, and does not publish a candidate.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_fingerprint, load_config  # noqa: E402
from src.contracts.types import Candidate, content_id  # noqa: E402
from src.data.split import load_tasks  # noqa: E402
from src.eval.options import inject_all  # noqa: E402
from src.eval.task_rows import validate_task_answer_row  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.runner import ArtifactState, TaskRunner  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.util import (artifacts_path, ensure_dir, read_json, sha256_obj,
                      sha256_text, write_json)  # noqa: E402


def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    answered = sum(int(row["answer_index"]) >= 0 for row in rows)
    return {
        "n": n,
        "correct": correct,
        "answered": answered,
        "accuracy": correct / n if n else 0.0,
        "coverage": answered / n if n else 0.0,
        "answered_precision": correct / answered if answered else 0.0,
    }


def _usage(rows: list[dict]) -> dict:
    costs = [row.get("cost") or {} for row in rows]
    usd_complete = bool(costs) and all(
        cost.get("usd") is not None and not cost.get("cost_estimated", False)
        for cost in costs)
    nominal_complete = bool(costs) and all(
        cost.get("usd_nominal") is not None for cost in costs)
    return {
        "calls": len(rows),
        "prompt_tokens": sum(int(cost.get("prompt_tokens") or 0) for cost in costs),
        "completion_tokens": sum(int(cost.get("completion_tokens") or 0)
                                  for cost in costs),
        "cache_read_tokens": sum(int(cost.get("cache_read_tokens") or 0)
                                  for cost in costs),
        "cache_write_tokens": sum(int(cost.get("cache_write_tokens") or 0)
                                   for cost in costs),
        "latency_ms": sum(int(row.get("latency_ms") or 0) for row in rows),
        "cost_usd": (round(sum(float(cost["usd"]) for cost in costs), 6)
                     if usd_complete else None),
        "cost_usd_nominal": (round(sum(float(cost["usd_nominal"])
                                        for cost in costs), 6)
                             if nominal_complete else None),
        "cost_estimated": not usd_complete,
        "usage_complete": all(cost.get("usage_available", True) for cost in costs),
        "cost_basis": "; ".join(sorted({str(cost["cost_basis"]) for cost in costs
                                         if cost.get("cost_basis")})),
    }


def _cluster_delta_ci(pairs: list[dict], arm: str, reference: str = "baseline",
                      *, draws: int = 4000, seed: int = 20261004) -> dict:
    """Bootstrap paired metric deltas by task ID."""
    grouped: dict[str, list[dict]] = {}
    for row in pairs:
        grouped.setdefault(str(row["task_id"]), []).append(row)
    task_ids = sorted(grouped)
    if not task_ids:
        return {name: [0.0, 0.0] for name in
                ("accuracy", "coverage", "answered_precision")}

    def delta(selected_ids: list[str], metric: str) -> float:
        base_rows, arm_rows = [], []
        for task_id in selected_ids:
            for row in grouped[task_id]:
                base_rows.append(row[reference])
                arm_rows.append(row[arm])
        return _metrics(arm_rows)[metric] - _metrics(base_rows)[metric]

    rng = random.Random(seed)
    result = {}
    for metric in ("accuracy", "coverage", "answered_precision"):
        samples = sorted(delta([rng.choice(task_ids) for _ in task_ids], metric)
                         for _ in range(draws))
        result[metric] = [samples[int(0.025 * draws)],
                          samples[min(draws - 1, int(0.975 * draws))]]
    return result


def _row(outcome, task: dict, *, arm: str) -> dict:
    checked = validate_task_answer_row(
        outcome.to_dict(), task,
        context=f"Pi decision-rule probe {arm}:{task['task_id']}",
        require_reported_gold_index=True)
    cost = getattr(outcome, "cost", {}) or {}
    return {
        "task_id": checked["task_id"],
        "answer_index": checked["answer_index"],
        "correct": checked["correct"],
        "status": checked["status"],
        "gold_index": checked["gold_index"],
        "pi_session_path": getattr(outcome, "pi_session_path", None),
        "prompt_tokens": cost.get("prompt_tokens"),
        "completion_tokens": cost.get("completion_tokens"),
        "cache_read_tokens": cost.get("cache_read_tokens"),
        "latency_ms": getattr(outcome, "latency_ms", 0),
        "cost": dict(cost),
    }


def build_rule_arm(state, cfg: dict, rule_path: Path, rule: str,
                   name: str) -> dict:
    """Build a prompt-only arm while preserving every other release artifact."""
    prompt = state.system_prompt.rstrip() + "\n\n" + rule
    candidate = Candidate(
        candidate_id=content_id("cand", state.release_id, "system_prompt", prompt),
        parent_release_id=state.release_id,
        artifact_type="system_prompt", artifact_id="system_prompt",
        diff=f"Append frozen decision rule from {rule_path.name}",
        new_content=prompt)
    gate = static_gate(candidate, cfg, state)
    if not gate["passed"]:
        raise RuntimeError(f"static gate failed for {name}: "
                           + json.dumps(gate["results"], ensure_ascii=False))
    return {
        "rule_path": str(rule_path),
        "rule_sha256": sha256_text(rule),
        "candidate_id": candidate.candidate_id,
        "candidate_prompt_sha256": sha256_text(prompt),
        "static_gate": gate,
        "artifact_state": ArtifactState(
            state.release_id, prompt, state.registry,
            task_demos=copy.deepcopy(state.task_demos)),
    }


def build_execution_plan(*, cfg: dict, state, tasks: list[dict],
                         arms: dict[str, dict], offset: int, repeats: int,
                         seed: int) -> dict:
    runtime = cfg.get("runtime") or {}
    task_model = (cfg.get("models") or {}).get("task") or {}
    skills = cfg.get("skills") or {}
    pi_retries = max(0, min(4, int(runtime.get("pi_provider_max_retries", 4))))
    bridge_attempts = max(1, min(3, int(runtime.get("pi_timeout_max_attempts", 2))))
    pi_attempts = (pi_retries + 1) * bridge_attempts
    http_attempts = max(1, int(task_model.get("max_retries", 6)))
    task_runs = len(tasks) * repeats * (1 + len(arms))
    router_enabled = (str(skills.get("routing_mode", "lexical")) == "model"
                      and bool(state.registry.skills)
                      and not str(skills.get("pinned_skill") or ""))
    router_provider = str(skills.get("router_provider", "http") or "http").lower()
    if router_enabled and router_provider not in {"http", "pi", "fake"}:
        raise ValueError("unsupported model skill-router provider for bounded plan")
    router_runs = task_runs if router_enabled and router_provider != "fake" else 0
    router_attempts = (pi_attempts if router_provider == "pi" else http_attempts
                       if router_provider == "http" else 0)
    arm_identity = {
        name: {
            "rule_path": arm["rule_path"],
            "rule_sha256": arm["rule_sha256"],
            "candidate_id": arm["candidate_id"],
            "candidate_prompt_sha256": arm["candidate_prompt_sha256"],
            "static_gate_passed": arm["static_gate"].get("passed") is True,
            "static_gate": arm["static_gate"],
        }
        for name, arm in arms.items()
    }
    unsigned = {
        "schema": "pi-decision-rule-arms-visible-optimize-plan-v1",
        "role": "exploratory_fixed_prompt_intervention",
        "dataset": "pubmedqa", "split": "dev_optimize",
        "task_offset": offset, "task_limit": len(tasks),
        "task_ids": [str(task["task_id"]) for task in tasks],
        "task_material_sha256": sha256_obj(tasks),
        "parent_release_id": state.release_id,
        "parent_system_prompt_sha256": sha256_text(state.system_prompt),
        "registry_sha256": state.registry.fingerprint(),
        "task_demos_sha256": sha256_obj(state.task_demos),
        "config_sha256": config_fingerprint(cfg),
        "seed": seed, "repeats": repeats,
        "arm_order": ["baseline", *arms.keys()],
        "arms": arm_identity,
        "taskrunner_runs_upper_bound": task_runs,
        "skill_router_enabled": router_enabled,
        "skill_router_provider": router_provider,
        "skill_router_runs_upper_bound": router_runs,
        "pi_provider_retries_after_first": pi_retries,
        "pi_timeout_subprocess_attempts": bridge_attempts,
        "pi_http_attempts_per_operation_upper_bound": pi_attempts,
        "router_physical_attempts_per_operation_upper_bound": router_attempts,
        "logical_by_component": {
            "task_answer": task_runs, "skill_router": router_runs},
        "physical_by_component": {
            "task_answer": task_runs * pi_attempts,
            "skill_router": router_runs * router_attempts},
        "logical_request_cap": task_runs + router_runs,
        "physical_request_cap": task_runs * pi_attempts + router_runs * router_attempts,
        "request_ledger_required": True,
        "pi_physical_reservations_required": True,
        "model_calls_during_preflight": 0,
        "retrieval_used": False,
        "selection_gate_read": False,
        "final_holdout_read": False,
        "candidate_selection_or_confirmation": False,
        "causal_attribution_supported": False,
        "live_authorized": False,
        "explicit_user_approval_required_before_execution": True,
    }
    return {**unsigned, "plan_sha256": sha256_obj(unsigned)}


def verify_execution_plan(plan: dict, expected_sha256: str) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    actual = sha256_obj(unsigned)
    if plan.get("plan_sha256") != actual:
        raise ValueError("frozen Pi decision-rule plan hash is internally inconsistent")
    if expected_sha256 != actual:
        raise ValueError("expected plan hash does not match current Pi decision-rule plan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule-file", action="append", required=True,
                        help="Repeat for each frozen challenger rule.")
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--logical-request-budget", type=int, default=0)
    parser.add_argument("--physical-request-budget", type=int, default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-plan-sha256", default="")
    args = parser.parse_args(argv)
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("decision-rule probe requires an isolated sandbox RSI_HOME")
    if args.offset < 0 or not 1 <= args.limit <= 200:
        raise ValueError("offset and limit must select visible optimize tasks")
    if not 1 <= args.repeats <= 10:
        raise ValueError("repeats must be between 1 and 10")
    if not 1 <= len(args.rule_file) <= 8:
        raise ValueError("one to eight frozen rule files are supported")
    if args.logical_request_budget < 0 or args.physical_request_budget < 0:
        raise ValueError("request budgets must be non-negative")
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    if args.validate_only and (args.logical_request_budget or args.physical_request_budget):
        parser.error("request budgets are frozen from the plan during --validate-only")

    cfg = copy.deepcopy(load_config(args.config))
    cfg["runtime"].update({"provider": "pi", "input_mode": "task_input",
                           "task_input_max_chars": 30000})
    cfg.setdefault("evaluation", {}).update({"main_metric": "accuracy",
                                               "primary_axis": "accuracy",
                                               "sealed_pairing": False})
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg["tools"].get("active", [])
        if name != "academic.search"]
    cfg.setdefault("skills", {}).setdefault("routing_mode", "lexical")
    cfg.setdefault("evaluation", {}).setdefault("semantic_citation", {})[
        "enabled"] = False

    state = load_state()
    task_input_dir = (ROOT / "configs" / "task_input").resolve()
    arms: dict[str, dict] = {}
    for raw_path in args.rule_file:
        rule_path = (ROOT / raw_path).resolve()
        if not rule_path.is_relative_to(task_input_dir):
            raise RuntimeError("rule files must be under configs/task_input")
        name = re.sub(r"[^a-zA-Z0-9_-]+", "_", rule_path.stem)
        if not name or name in {"baseline", *arms.keys()}:
            raise ValueError(f"duplicate or reserved rule arm name: {name}")
        rule = rule_path.read_text(encoding="utf-8").strip()
        if not rule:
            raise ValueError(f"rule file is empty: {rule_path}")
        arms[name] = build_rule_arm(state, cfg, rule_path, rule, name)

    tasks = sorted(load_tasks("pubmedqa", splits=["dev_optimize"]),
                   key=lambda task: task["task_id"])[args.offset:args.offset + args.limit]
    if len(tasks) != args.limit:
        raise ValueError(f"requested {args.limit} tasks but selected {len(tasks)}")
    if len(tasks) != args.limit:
        raise ValueError(f"requested {args.limit} tasks but selected {len(tasks)}")
    if cfg["evaluation"].get("insufficient_option", True):
        tasks = inject_all(tasks)
    plan = build_execution_plan(cfg=cfg, state=state, tasks=tasks, arms=arms,
                                offset=args.offset, repeats=args.repeats,
                                seed=args.seed)
    if args.execute:
        verify_execution_plan(plan, args.expected_plan_sha256)
        if (args.logical_request_budget != plan["logical_request_cap"]
                or args.physical_request_budget != plan["physical_request_cap"]):
            raise ValueError("execution request budgets must exactly match the frozen plan")

    plan_path = artifacts_path("probe", "pi_decision_rule_arms",
                               "preflight-" + plan["plan_sha256"] + ".json")
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise RuntimeError("saved Pi decision-rule plan differs from recomputed plan")
    else:
        write_json(plan_path, plan)
    if args.validate_only:
        print(json.dumps({
            "status": "validated_not_run", "plan": str(plan_path),
            "plan_sha256": plan["plan_sha256"], "task_count": len(tasks),
            "repeats": args.repeats,
            "logical_request_cap": plan["logical_request_cap"],
            "physical_request_cap": plan["physical_request_cap"],
            "model_calls": 0, "selection_gate_read": False,
            "final_holdout_read": False,
        }, ensure_ascii=False))
        return 0

    task_model = (cfg.get("models") or {}).get("task") or {}
    key_env = str(task_model.get("api_key_env") or "").strip()
    if not key_env or not os.environ.get(key_env):
        raise RuntimeError("decision-rule execution is missing the configured task credential")
    run_dir = artifacts_path("probe", "pi_decision_rule_arms",
                             "run-" + plan["plan_sha256"][:16] + "-" + uuid.uuid4().hex[:12])
    ensure_dir(run_dir)
    ledger_path = run_dir / "call-ledger.jsonl"
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(plan["plan_sha256"])
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path)
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(plan["logical_request_cap"])
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(plan["physical_request_cap"])
    os.environ["RSI_CALL_LEDGER_RESERVE_PI_PHYSICAL"] = "1"
    for component, count in plan["logical_by_component"].items():
        os.environ["RSI_CALL_LEDGER_MAX_COMPONENT_" + component.upper()] = str(count)
    os.environ["RSI_HTTP_NO_CACHE"] = "1"

    pairs = []
    for repetition in range(args.repeats):
        for task in tasks:
            baseline = _row(
                TaskRunner(cfg, state, seed=args.seed + repetition).run(task),
                task, arm="baseline")
            row = {"task_id": task["task_id"], "repeat": repetition + 1,
                   "baseline": baseline}
            for name, arm in arms.items():
                row[name] = _row(TaskRunner(
                    cfg, arm["artifact_state"], seed=args.seed + repetition).run(task),
                    task, arm=name)
            pairs.append(row)

    flat = {"baseline": [row["baseline"] for row in pairs]}
    for name in arms:
        flat[name] = [row[name] for row in pairs]
    contrasts = {}
    for name in arms:
        flips = Counter()
        changed_ids = set()
        for row in pairs:
            before, after = row["baseline"], row[name]
            if before["answer_index"] != after["answer_index"]:
                changed_ids.add(row["task_id"])
                b = "abstain" if before["answer_index"] < 0 else (
                    "correct" if before["correct"] else "wrong")
                a = "abstain" if after["answer_index"] < 0 else (
                    "correct" if after["correct"] else "wrong")
                flips[f"{b}_to_{a}"] += 1
        contrasts[name] = {
            "delta_vs_baseline": {
                metric: _metrics(flat[name])[metric] - _metrics(flat["baseline"])[metric]
                for metric in ("accuracy", "coverage", "answered_precision")},
            "task_cluster_bootstrap_95ci": _cluster_delta_ci(pairs, name),
            "changed_task_ids": sorted(changed_ids),
            "transition_counts": dict(flips),
        }

    direct_contrasts = {}
    arm_names = list(arms)
    for index, left in enumerate(arm_names):
        for right in arm_names[index + 1:]:
            flips = Counter()
            changed_ids = set()
            for row in pairs:
                before, after = row[left], row[right]
                if before["answer_index"] == after["answer_index"]:
                    continue
                changed_ids.add(row["task_id"])
                b = "abstain" if before["answer_index"] < 0 else (
                    "correct" if before["correct"] else "wrong")
                a = "abstain" if after["answer_index"] < 0 else (
                    "correct" if after["correct"] else "wrong")
                flips[f"{b}_to_{a}"] += 1
            direct_contrasts[f"{right}_minus_{left}"] = {
                "metric_delta": {
                    metric: _metrics(flat[right])[metric] - _metrics(flat[left])[metric]
                    for metric in ("accuracy", "coverage", "answered_precision")},
                "task_cluster_bootstrap_95ci": _cluster_delta_ci(
                    pairs, right, left),
                "changed_task_ids": sorted(changed_ids),
                "transition_counts": dict(flips),
            }

    report = {
        "protocol": "pi-target-outcome-rule-multiarm-visible-optimize-v1-exploratory",
        "execution_plan_sha256": plan["plan_sha256"],
        "execution_plan_path": str(plan_path),
        "request_ledger": str(ledger_path),
        "request_ledger_summary": summarize_ledger(ledger_path),
        "dataset": "pubmedqa", "split": "dev_optimize",
        "selection_gate_read": False, "final_holdout_read": False,
        "candidate_selection": False, "independent_confirmation": False,
        "causal_attribution_supported": False, "exploratory_only": True,
        "retrieval_used_as_evolution_input": False,
        "release_id": state.release_id,
        "config_fingerprint": config_fingerprint(cfg),
        "task_ids": [task["task_id"] for task in tasks],
        "repeats": args.repeats,
        "seed": args.seed,
        "arms": {name: {k: v for k, v in arm.items() if k != "artifact_state"}
                 for name, arm in arms.items()},
        "metrics": {name: _metrics(rows) for name, rows in flat.items()},
        "usage": {name: _usage(rows) for name, rows in flat.items()},
        "contrasts_vs_baseline": contrasts,
        "direct_candidate_contrasts": direct_contrasts,
        "pairs": pairs,
        "interpretation_limits": [
            "Repeated requests share task IDs; bootstrap resamples task clusters.",
            "All observations are exploratory dev_optimize evidence.",
            "The fixed hand-written rules test a mechanism, not optimizer efficacy.",
            "USD cost is not asserted unless the provider billing record is auditable.",
        ],
    }
    path = run_dir / ("report-" + plan["plan_sha256"] + ".json")
    write_json(path, report)
    print(json.dumps({"path": str(path), "task_count": len(tasks),
                      "repeats": args.repeats, "metrics": report["metrics"],
                      "contrasts_vs_baseline": contrasts,
                      "direct_candidate_contrasts": direct_contrasts,
                      "request_ledger": report["request_ledger_summary"],
                      "candidate_ids": {name: arm["candidate_id"]
                                        for name, arm in arms.items()}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
