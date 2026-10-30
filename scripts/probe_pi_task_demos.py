"""Exploratory task-model few-shot mechanism test on disjoint visible dev tasks.

This is a fixed demonstration arm, not vendor MIPROv2 and not a selection gate.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_fingerprint, load_config  # noqa: E402
from src.contracts.types import Candidate, content_id, content_sha256  # noqa: E402
from src.data.split import load_tasks  # noqa: E402
from src.eval.options import inject_all  # noqa: E402
from src.eval.task_rows import validate_task_answer_row  # noqa: E402
from src.evolution.fewshot import render_demos_block  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.runtime.runner import TaskRunner  # noqa: E402
from src.util import (  # noqa: E402
    artifacts_path, ensure_dir, sha256_obj, sha256_text, write_json,
)


def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(bool(r["correct"]) for r in rows)
    answered = sum(r["answer_index"] >= 0 for r in rows)
    return {"n": n, "accuracy": correct / n if n else 0.0,
            "coverage": answered / n if n else 0.0,
            "answered_precision": correct / answered if answered else 0.0}


def _row(task: dict, outcome) -> dict:
    if (str(getattr(outcome, "task_id", "")) != str(task.get("task_id", ""))
            or str(getattr(outcome, "dataset", ""))
            != str(task.get("dataset", "pubmedqa"))
            or str(getattr(outcome, "split", ""))
            != str(task.get("split", "dev_optimize"))):
        raise ValueError("Pi task-demo outcome identity differs from frozen task")
    row = {"task_id": outcome.task_id, "dataset": outcome.dataset,
            "split": outcome.split,
            "answer_index": outcome.answer_index, "gold_index": outcome.gold_index,
            "correct": outcome.correct, "status": outcome.status,
            "skill": outcome.skill_recalled,
            "pi_session_path": outcome.pi_session_path,
            "searched": any(e.get("tool_name") == "academic.search"
                            for e in outcome.events)}
    return validate_task_answer_row(
        row, task, context="Pi task-demo probe outcome",
        require_reported_gold_index=True)


def build_execution_plan(*, cfg: dict, state, tasks: list[dict],
                         config_path: str, demo_selection: str) -> dict:
    runtime = cfg.get("runtime") or {}
    task_model = (cfg.get("models") or {}).get("task") or {}
    skills = cfg.get("skills") or {}
    pool_count = 10
    eval_count = 20
    task_runs = pool_count + eval_count * 2
    pi_retries = max(0, min(4, int(runtime.get("pi_provider_max_retries", 4))))
    bridge_attempts = max(1, min(3, int(runtime.get("pi_timeout_max_attempts", 2))))
    pi_attempts = (pi_retries + 1) * bridge_attempts
    router_attempts = max(1, int(task_model.get("max_retries", 6)))
    router_enabled = (str(skills.get("routing_mode", "lexical")) == "model"
                      and bool(state.registry.skills)
                      and not str(skills.get("pinned_skill") or ""))
    router_runs = task_runs if router_enabled else 0
    logical_by_component = {"task_answer": task_runs,
                            "skill_router": router_runs}
    physical_by_component = {
        "task_answer": task_runs * pi_attempts,
        "skill_router": router_runs * router_attempts,
    }
    unsigned = {
        "schema": "pi-task-demos-visible-optimize-plan-v1",
        "role": "exploratory_demo_generation_and_disjoint_comparison",
        "config_path": str(Path(config_path).as_posix()),
        "config_sha256": config_fingerprint(cfg),
        "dataset": "pubmedqa", "split": "dev_optimize",
        "task_limit": len(tasks),
        "task_ids": [str(task["task_id"]) for task in tasks],
        "task_material_sha256": sha256_obj(tasks),
        "demo_pool_task_ids": [str(task["task_id"]) for task in tasks[:pool_count]],
        "evaluation_task_ids": [str(task["task_id"]) for task in tasks[pool_count:]],
        "demo_selection": demo_selection,
        "demo_selection_policy_version": "first-correct-or-balanced-no-v1",
        "max_demo_count": 3,
        "parent_release_id": state.release_id,
        "system_prompt_sha256": sha256_text(state.system_prompt),
        "registry_sha256": state.registry.fingerprint(),
        "task_demos_sha256": sha256_obj(getattr(state, "task_demos", [])),
        "task_model_id": str(task_model.get("model_id") or ""),
        "task_provider": "pi",
        "skill_routing_mode": str(skills.get("routing_mode", "lexical")),
        "task_answer_runs_upper_bound": task_runs,
        "skill_router_runs_upper_bound": router_runs,
        "pi_provider_retries_after_first": pi_retries,
        "pi_timeout_subprocess_attempts": bridge_attempts,
        "pi_http_attempts_per_operation_upper_bound": pi_attempts,
        "router_http_attempts_per_operation_upper_bound": router_attempts,
        "logical_by_component": logical_by_component,
        "physical_by_component": physical_by_component,
        "logical_request_cap": sum(logical_by_component.values()),
        "physical_request_cap": sum(physical_by_component.values()),
        "request_ledger_required": True,
        "pi_physical_reservations_required": True,
        "model_calls_during_preflight": 0,
        "retrieval_used": False,
        "selection_gate_read": False,
        "final_holdout_read": False,
        "candidate_selection_scope": "visible_dev_optimize_demo_pool_only",
        "independent_confirmation": False,
        "causal_attribution_supported": False,
        "live_authorized": False,
        "explicit_user_approval_required_before_execution": True,
    }
    return {**unsigned, "plan_sha256": sha256_obj(unsigned)}


def verify_execution_plan(plan: dict, expected_sha256: str) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    actual = sha256_obj(unsigned)
    if plan.get("plan_sha256") != actual:
        raise ValueError("frozen Pi task-demo plan hash is internally inconsistent")
    if expected_sha256 != actual:
        raise ValueError("expected plan hash does not match current Pi task-demo plan")


def _gate_fewshot_candidate(chosen: list[dict], block: str, cfg: dict,
                            state) -> dict:
    """Authenticate a constructed demo arm before any paired evaluation."""
    from src.evolution.fewshot import serialize_demos_bundle
    from src.evolution.task_input_gates import validate_task_demo_rendering

    bundle = serialize_demos_bundle(chosen)
    candidate = Candidate(
        candidate_id=content_id("cand", state.release_id, "task_demos", bundle),
        parent_release_id=state.release_id, artifact_type="task_demos",
        artifact_id="task_demos", diff="evaluate exploratory Pi task demos",
        new_content=bundle, content_sha256=content_sha256(bundle))
    result = static_gate(candidate, cfg, state)
    if not result.get("passed"):
        raise ValueError("few-shot candidate failed static gate")
    rendering = validate_task_demo_rendering(chosen, cfg)
    if not rendering.get("passed"):
        raise ValueError("few-shot candidate failed rendering gate")
    if render_demos_block(chosen) != block:
        raise ValueError("few-shot candidate block differs from gated demos")
    return {"static_gate": result, "rendering_gate": rendering,
            "candidate_id": candidate.candidate_id,
            "content_sha256": candidate.content_sha256}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--demo-selection", choices=["first-correct", "balanced-no"],
                        default="first-correct")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--logical-request-budget", type=int, default=0)
    parser.add_argument("--physical-request-budget", type=int, default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-plan-sha256", default="")
    args = parser.parse_args(argv)
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("few-shot probe requires an explicit sandbox RSI_HOME")
    if args.logical_request_budget < 0 or args.physical_request_budget < 0:
        parser.error("request budgets must be non-negative")
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    cfg = copy.deepcopy(load_config(args.config))
    cfg["runtime"].update({"provider": "pi", "input_mode": "task_input",
                           "task_input_max_chars": 30000})
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg["tools"].get("active", []) if name != "academic.search"]
    tasks = sorted(load_tasks("pubmedqa", splits=["dev_optimize"]),
                   key=lambda x: x["task_id"])[:30]
    if cfg["evaluation"].get("insufficient_option", True):
        tasks = inject_all(tasks)
    if len(tasks) != 30:
        raise RuntimeError("expected 30 visible optimize tasks")
    state = load_state()
    plan = build_execution_plan(cfg=cfg, state=state, tasks=tasks,
                                config_path=args.config,
                                demo_selection=args.demo_selection)
    if args.execute:
        verify_execution_plan(plan, args.expected_plan_sha256)
        if not args.logical_request_budget or not args.physical_request_budget:
            raise ValueError("execution requires positive logical and physical request budgets")
        if (args.logical_request_budget < plan["logical_request_cap"]
                or args.physical_request_budget < plan["physical_request_cap"]):
            raise ValueError("provided budgets are below the frozen plan's upper bounds")
    plan_path = artifacts_path("probe", "pi_task_demos",
                               "preflight-" + plan["plan_sha256"] + ".json")
    if plan_path.exists():
        from src.util import read_json
        if read_json(plan_path) != plan:
            raise RuntimeError("saved Pi task-demo preflight differs from recomputed plan")
    else:
        write_json(plan_path, plan)
    if args.validate_only:
        print(json.dumps({
            "status": "validated_not_run", "plan": str(plan_path),
            "plan_sha256": plan["plan_sha256"],
            "task_count": len(tasks), "demo_pool_count": 10,
            "evaluation_count": 20,
            "logical_request_cap": plan["logical_request_cap"],
            "physical_request_cap": plan["physical_request_cap"],
            "model_calls": 0, "selection_gate_read": False,
            "final_holdout_read": False,
        }, ensure_ascii=False))
        return 0

    task_model = (cfg.get("models") or {}).get("task") or {}
    key_env = str(task_model.get("api_key_env") or "").strip()
    if not key_env or not os.environ.get(key_env):
        raise RuntimeError("Pi task-demo execution is missing the configured task-model credential")
    run_id = uuid.uuid4().hex[:12]
    run_dir = artifacts_path("probe", "pi_task_demos",
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

    demo_pool, eval_tasks = tasks[:10], tasks[10:]
    pool_rows = []
    eligible = []
    for task in demo_pool:
        outcome = TaskRunner(cfg, state, seed=args.seed).run(task)
        row = _row(task, outcome)
        pool_rows.append(row)
        if row["correct"]:
            eligible.append({"task_id": task["task_id"], "task": task,
                             "correct_index": task["correct_index"]})
    if args.demo_selection == "balanced-no":
        yes = [d for d in eligible if str(d["task"]["options"][
            d["correct_index"]]).strip().casefold() == "yes"]
        no = [d for d in eligible if str(d["task"]["options"][
            d["correct_index"]]).strip().casefold() == "no"]
        if not yes or len(no) < 2:
            raise RuntimeError("balanced-no requires one correct yes and two correct no demos")
        ordered = [yes[0], no[0], no[1]]
    else:
        ordered = eligible
    chosen = []
    block = ""
    for demo in ordered:
        proposed = render_demos_block(chosen + [demo])
        if not proposed or proposed == block:
            break
        chosen.append(demo)
        block = proposed
        if len(chosen) == 3:
            break
    if not chosen:
        raise RuntimeError("no correct demonstration fits the prompt budget")
    candidate_gate = _gate_fewshot_candidate(chosen, block, cfg, state)
    ids = [d["task_id"] for d in chosen]
    if set(ids) & {t["task_id"] for t in eval_tasks}:
        raise RuntimeError("demonstration/evaluation overlap")
    fewshot_cfg = copy.deepcopy(cfg)
    fewshot_cfg["runtime"].update({"pi_task_demos_block": block,
                                   "pi_task_demo_ids": ids,
                                   "pi_task_demo_split": "dev_optimize"})
    pairs = []
    for task in eval_tasks:
        baseline = TaskRunner(cfg, state, seed=args.seed).run(task)
        fewshot = TaskRunner(fewshot_cfg, state, seed=args.seed).run(task)
        pairs.append({"task_id": task["task_id"],
                      "baseline": _row(task, baseline),
                      "fewshot": _row(task, fewshot)})
    baseline_rows = [p["baseline"] for p in pairs]
    fewshot_rows = [p["fewshot"] for p in pairs]
    report = {
        "protocol": "pi-task-model-fewshot-mechanism-v2-exploratory",
        "execution_plan_sha256": plan["plan_sha256"],
        "execution_plan_path": str(plan_path),
        "dataset": "pubmedqa", "split": "dev_optimize",
        "release_id": state.release_id,
        "seed": args.seed,
        "demo_selection": args.demo_selection,
        "pool_ids": [t["task_id"] for t in demo_pool],
        "eval_ids": [t["task_id"] for t in eval_tasks],
        "demo_ids": ids, "demo_block_sha256": sha256_text(block),
        "demo_block_chars": len(block),
        "candidate_gate": candidate_gate,
        "baseline_config_sha256": config_fingerprint(cfg),
        "fewshot_config_sha256": config_fingerprint(fewshot_cfg),
        "request_ledger": str(ledger_path),
        "request_ledger_summary": summarize_ledger(ledger_path),
        "selection_gate_read": False, "final_holdout_read": False,
        "candidate_selection_scope": "visible_dev_optimize_demo_pool_only",
        "independent_confirmation": False,
        "causal_attribution_supported": False,
        "pool_rows": pool_rows,
        "baseline_metrics": _metrics(baseline_rows),
        "fewshot_metrics": _metrics(fewshot_rows),
        "changed_answer_ids": [p["task_id"] for p in pairs
                               if p["baseline"]["answer_index"] != p["fewshot"]["answer_index"]],
        "fixed_ids": [p["task_id"] for p in pairs
                      if not p["baseline"]["correct"] and p["fewshot"]["correct"]],
        "broken_ids": [p["task_id"] for p in pairs
                       if p["baseline"]["correct"] and not p["fewshot"]["correct"]],
        "pairs": pairs,
    }
    path = run_dir / ("report-" + plan["plan_sha256"] + ".json")
    write_json(path, report)
    print(json.dumps({k: report[k] for k in (
        "demo_ids", "demo_block_chars", "baseline_metrics", "fewshot_metrics",
        "changed_answer_ids", "fixed_ids", "broken_ids")}
        | {"path": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
