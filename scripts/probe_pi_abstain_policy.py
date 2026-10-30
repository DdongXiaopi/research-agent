"""Paired exploratory probe of fixed Pi abstention policies on visible tasks.

This diagnostic reads only PubMedQA ``dev_optimize``. Live execution is
separated from a deterministic zero-call preflight and requires an exact plan
hash challenge plus bounded request-ledger budgets.
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
from src.data.split import load_tasks  # noqa: E402
from src.eval.options import inject_all  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.runtime.runner import TaskRunner  # noqa: E402
from src.util import (  # noqa: E402
    artifacts_path, ensure_dir, sha256_obj, sha256_text, write_json,
)

POLICIES = ("literal", "supported_inference", "evidence_qualified")
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20261010


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


def _cluster_delta_ci(pairs: list[dict], arm: str, reference: str = "literal",
                      *, draws: int = BOOTSTRAP_DRAWS,
                      seed: int = BOOTSTRAP_SEED) -> dict:
    """Bootstrap paired metric deltas by task ID, keeping repeats clustered."""
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


def _row(outcome) -> dict:
    cost = getattr(outcome, "cost", {}) or {}
    return {
        "answer_index": outcome.answer_index,
        "correct": bool(outcome.correct),
        "status": outcome.status,
        "pi_session_path": getattr(outcome, "pi_session_path", None),
        "latency_ms": getattr(outcome, "latency_ms", 0),
        "cost": dict(cost),
    }


def _prepare_config(config_path: str) -> dict:
    cfg = copy.deepcopy(load_config(config_path))
    cfg.setdefault("runtime", {}).update({
        "provider": "pi", "input_mode": "task_input",
        "task_input_max_chars": 30000,
    })
    cfg.setdefault("evaluation", {}).update({
        "main_metric": "accuracy", "primary_axis": "accuracy",
        "sealed_pairing": False,
    })
    cfg["evaluation"].setdefault("semantic_citation", {})["enabled"] = False
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg["tools"].get("active", [])
        if name != "academic.search"]
    cfg.setdefault("skills", {}).setdefault("routing_mode", "lexical")
    return cfg


def build_execution_plan(*, cfg: dict, state, tasks: list[dict],
                         offset: int, repeats: int,
                         policy_cfgs: dict[str, dict]) -> dict:
    runtime = cfg.get("runtime") or {}
    task_model = (cfg.get("models") or {}).get("task") or {}
    skills = cfg.get("skills") or {}
    pi_retries = max(0, min(4, int(runtime.get("pi_provider_max_retries", 4))))
    bridge_attempts = max(1, min(3, int(runtime.get("pi_timeout_max_attempts", 2))))
    pi_attempts = (pi_retries + 1) * bridge_attempts
    http_attempts = max(1, int(task_model.get("max_retries", 6)))
    runs = len(tasks) * repeats * len(POLICIES)
    router_enabled = (str(skills.get("routing_mode", "lexical")) == "model"
                      and bool(state.registry.skills)
                      and not str(skills.get("pinned_skill") or ""))
    router_logical = runs if router_enabled else 0
    logical_by_component = {"task_answer": runs,
                            "skill_router": router_logical}
    physical_by_component = {
        "task_answer": runs * pi_attempts,
        "skill_router": router_logical * http_attempts,
    }
    unsigned = {
        "schema": "pi-abstain-policy-visible-optimize-plan-v1",
        "role": "one_shot_exploratory_fixed_policy_probe",
        "dataset": "pubmedqa", "split": "dev_optimize",
        "task_offset": offset, "task_limit": len(tasks),
        "task_ids": [str(task["task_id"]) for task in tasks],
        "task_material_sha256": sha256_obj(tasks),
        "repeats": repeats, "policy_order": list(POLICIES),
        "policy_config_sha256": {
            name: config_fingerprint(policy_cfgs[name]) for name in POLICIES},
        "config_sha256": config_fingerprint(cfg),
        "release_id": state.release_id,
        "system_prompt_sha256": sha256_text(state.system_prompt),
        "registry_sha256": state.registry.fingerprint(),
        "task_demos_sha256": sha256_obj(getattr(state, "task_demos", [])),
        "task_model_id": str(task_model.get("model_id") or ""),
        "task_provider": "pi",
        "skill_routing_mode": str(skills.get("routing_mode", "lexical")),
        "semantic_citation_judge_enabled": False,
        "router_http_attempts_per_operation_upper_bound": http_attempts,
        "pi_provider_retries_after_first": pi_retries,
        "pi_timeout_subprocess_attempts": bridge_attempts,
        "pi_http_attempts_per_operation_upper_bound": pi_attempts,
        "logical_request_cap": sum(logical_by_component.values()),
        "physical_request_cap": sum(physical_by_component.values()),
        "logical_by_component": logical_by_component,
        "physical_by_component": physical_by_component,
        "request_ledger_required": True,
        "pi_physical_reservations_required": True,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "model_calls_during_preflight": 0,
        "retrieval_used": False,
        "selection_gate_read": False,
        "final_holdout_read": False,
        "candidate_selection_or_confirmation": False,
        "live_authorized": False,
        "explicit_user_approval_required_before_execution": True,
    }
    return {**unsigned, "plan_sha256": sha256_obj(unsigned)}


def verify_execution_plan(plan: dict, expected_sha256: str) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    actual = sha256_obj(unsigned)
    if plan.get("plan_sha256") != actual:
        raise ValueError("frozen Pi abstention plan hash is internally inconsistent")
    if expected_sha256 != actual:
        raise ValueError("expected plan hash does not match current Pi abstention plan")


def _configure_policy(cfg: dict, policy: str) -> dict:
    out = copy.deepcopy(cfg)
    out.setdefault("runtime", {})["pi_abstain_policy"] = policy
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--task-id", action="append", default=[],
                        help="Optional exact visible task ID; repeat to select multiple")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--logical-request-budget", type=int, default=0)
    parser.add_argument("--physical-request-budget", type=int, default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-plan-sha256", default="")
    args = parser.parse_args(argv)
    if args.offset < 0 or not 1 <= args.limit <= 200:
        parser.error("--offset must be non-negative and --limit between 1 and 200")
    if not 1 <= args.repeats <= 10:
        parser.error("--repeats must be between 1 and 10")
    if args.logical_request_budget < 0 or args.physical_request_budget < 0:
        parser.error("request budgets must be non-negative")
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("Pi abstention probe requires an explicit sandbox RSI_HOME")
    if len(args.task_id) != len(set(args.task_id)):
        raise ValueError("duplicate --task-id selection")

    cfg = _prepare_config(args.config)
    state = load_state()
    visible = sorted(load_tasks("pubmedqa", splits=["dev_optimize"]),
                     key=lambda task: task["task_id"])
    window = visible[args.offset:args.offset + args.limit]
    if len(window) != args.limit:
        raise ValueError(f"requested {args.limit} tasks but selected {len(window)}")
    if args.task_id:
        selected = set(args.task_id)
        absent = selected - {task["task_id"] for task in window}
        if absent:
            raise ValueError("requested task IDs are outside the selected visible window")
        tasks = [task for task in window if task["task_id"] in selected]
    else:
        tasks = window
    if cfg["evaluation"].get("insufficient_option", True):
        tasks = inject_all(tasks)
    if not tasks:
        raise ValueError("Pi abstention probe selected no visible tasks")

    policy_cfgs = {name: _configure_policy(cfg, name) for name in POLICIES}
    plan = build_execution_plan(cfg=cfg, state=state, tasks=tasks,
                                offset=args.offset, repeats=args.repeats,
                                policy_cfgs=policy_cfgs)
    if args.execute:
        verify_execution_plan(plan, args.expected_plan_sha256)
        if not args.logical_request_budget or not args.physical_request_budget:
            raise ValueError("execution requires positive logical and physical request budgets")
        if (args.logical_request_budget < plan["logical_request_cap"]
                or args.physical_request_budget < plan["physical_request_cap"]):
            raise ValueError("provided budgets are below the frozen plan's upper bounds")

    plan_path = artifacts_path("probe", "pi_abstain_policy",
                               "preflight-" + plan["plan_sha256"] + ".json")
    if plan_path.exists():
        from src.util import read_json
        if read_json(plan_path) != plan:
            raise RuntimeError("saved Pi abstention preflight differs from recomputed plan")
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
        raise RuntimeError("Pi abstention execution is missing the configured task-model credential")
    run_id = uuid.uuid4().hex[:12]
    run_dir = artifacts_path("probe", "pi_abstain_policy",
                             "run-" + plan["plan_sha256"][:16] + "-" + run_id)
    ensure_dir(run_dir)
    ledger_path = run_dir / "call-ledger.jsonl"
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(plan["plan_sha256"])
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path)
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(args.logical_request_budget)
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(args.physical_request_budget)
    os.environ["RSI_CALL_LEDGER_RESERVE_PI_PHYSICAL"] = "1"
    for component, count in plan["logical_by_component"].items():
        os.environ["RSI_CALL_LEDGER_MAX_COMPONENT_" + component.upper()] = str(count)
    if plan["logical_by_component"].get("skill_router"):
        os.environ["RSI_HTTP_NO_CACHE"] = "1"

    pairs = []
    for repetition in range(args.repeats):
        for task in tasks:
            row = {"task_id": task["task_id"], "repeat": repetition + 1}
            for name in POLICIES:
                row[name] = _row(TaskRunner(
                    policy_cfgs[name], state,
                    seed=BOOTSTRAP_SEED + repetition).run(task))
            pairs.append(row)

    flat = {name: [row[name] for row in pairs] for name in POLICIES}
    contrasts = {}
    for name in POLICIES[1:]:
        flips = Counter()
        changed_ids = set()
        for row in pairs:
            before, after = row["literal"], row[name]
            if before["answer_index"] != after["answer_index"]:
                changed_ids.add(row["task_id"])
                b = "abstain" if before["answer_index"] < 0 else (
                    "correct" if before["correct"] else "wrong")
                a = "abstain" if after["answer_index"] < 0 else (
                    "correct" if after["correct"] else "wrong")
                flips[f"{b}_to_{a}"] += 1
        contrasts[name] = {
            "delta_vs_literal": {
                metric: _metrics(flat[name])[metric] - _metrics(flat["literal"])[metric]
                for metric in ("accuracy", "coverage", "answered_precision")},
            "task_cluster_bootstrap_95ci": _cluster_delta_ci(pairs, name),
            "changed_task_ids": sorted(changed_ids),
            "transition_counts": dict(flips),
        }
    report = {
        "protocol": "pi-abstain-policy-visible-optimize-v1-exploratory",
        "execution_plan_sha256": plan["plan_sha256"],
        "execution_plan_path": str(plan_path),
        "dataset": "pubmedqa", "split": "dev_optimize",
        "selection_gate_read": False, "final_holdout_read": False,
        "retrieval_used_as_evolution_input": False,
        "release_id": state.release_id,
        "config_fingerprint": config_fingerprint(cfg),
        "task_ids": [task["task_id"] for task in tasks],
        "repeats": args.repeats, "request_ledger": str(ledger_path),
        "request_ledger_summary": summarize_ledger(ledger_path),
        "metrics": {name: _metrics(flat[name]) for name in POLICIES},
        "usage": {name: _usage(flat[name]) for name in POLICIES},
        "contrasts_vs_literal": contrasts,
        "pairs": pairs,
        "interpretation_limits": [
            "Repeated requests share task IDs; bootstrap resamples task clusters.",
            "All observations are exploratory dev_optimize evidence.",
            "The fixed abstention policies test a mechanism, not optimizer efficacy.",
            "USD cost is not asserted unless the provider billing record is auditable.",
        ],
    }
    report_path = run_dir / ("report-" + plan["plan_sha256"] + ".json")
    write_json(report_path, report)
    print(json.dumps({"path": str(report_path), "task_count": len(tasks),
                      "repeats": args.repeats, "metrics": report["metrics"],
                      "contrasts_vs_literal": contrasts,
                      "request_ledger": report["request_ledger_summary"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
