"""Paired, exploratory replay of a frozen GEPA Prompt on unused dev IDs.

This probe reads only dev_optimize. It never selects or reads a gate/final split.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_fingerprint, load_config  # noqa: E402
from src.contracts.types import Candidate, content_id  # noqa: E402
from src.data.split import load_tasks  # noqa: E402
from src.eval.options import inject_all  # noqa: E402
from src.eval.task_rows import validate_task_answer_row  # noqa: E402
from src.evolution.gepa_search import training_question_overlap  # noqa: E402
from src.evolution.prompt_rewrite import validate_rewrite  # noqa: E402
from src.evolution.reflection import unified_diff  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.runtime.runner import ArtifactState, TaskRunner  # noqa: E402
from src.util import (  # noqa: E402
    artifacts_path, ensure_dir, read_json, sha256_file, sha256_obj, write_json,
)


def _config(args) -> dict:
    cfg = copy.deepcopy(load_config(args.config))
    cfg.setdefault("runtime", {}).update({"provider": "pi", "input_mode": "task_input"})
    cfg["runtime"].setdefault("task_input_max_chars", 30000)
    cfg.setdefault("evaluation", {}).update({"sealed_pairing": False,
                                              "main_metric": "accuracy",
                                              "primary_axis": "accuracy"})
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        item for item in cfg["tools"].get("active", []) if item != "academic.search"]
    gp = cfg.setdefault("evolution", {}).setdefault("gepa", {})
    gp.update({"enabled": True, "max_metric_calls": args.metric_calls,
               "max_candidate_proposals": args.max_candidate_proposals,
               "reflection_minibatch_size": args.minibatch_size,
               "acceptance_criterion": args.acceptance_criterion,
               "multi_objective": args.multi_objective,
               "feedback_evidence_signals": args.feedback_signals,
               "use_merge": args.use_merge})
    cfg.setdefault("models", {}).setdefault("reflection", {})["model_id"] = (
        args.reflection_model)
    return cfg


def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    answered = sum(int(row["answer_index"]) >= 0 for row in rows)
    wrong = sum(int(row["answer_index"]) >= 0 and not row["correct"] for row in rows)
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "coverage": answered / n if n else 0.0,
        "answered_precision": correct / answered if answered else 0.0,
        "answer_quality": (correct - wrong) / n if n else 0.0,
        "correct": correct, "answered": answered, "wrong_answered": wrong,
    }


def _paired_ci(base: list[dict], challenger: list[dict], seed: int,
               samples: int = 5000) -> dict:
    import random

    rng = random.Random(seed)
    n = len(base)
    metrics = ("accuracy", "coverage", "answered_precision", "answer_quality")
    draws = {name: [] for name in metrics}
    point_base, point_candidate = _metrics(base), _metrics(challenger)
    for _ in range(samples):
        ids = [rng.randrange(n) for _ in range(n)]
        b = _metrics([base[i] for i in ids])
        c = _metrics([challenger[i] for i in ids])
        for name in metrics:
            draws[name].append(c[name] - b[name])
    out = {}
    for name, values in draws.items():
        values.sort()
        out[name] = {
            "delta": point_candidate[name] - point_base[name],
            "paired_task_bootstrap_95_ci": [values[int(0.025 * samples)],
                                             values[min(samples - 1,
                                                        int(0.975 * samples))]],
        }
    return out


def _row(task: dict, outcome) -> dict:
    if (str(getattr(outcome, "task_id", "")) != str(task["task_id"])
            or str(getattr(outcome, "dataset", "")) != str(task.get("dataset", ""))
            or str(getattr(outcome, "split", "")) != str(task.get("split", "dev_optimize"))):
        raise ValueError("frozen-prompt outcome identity differs from frozen task")
    row = {
        "task_id": str(outcome.task_id), "dataset": str(outcome.dataset),
        "split": str(outcome.split), "answer_index": outcome.answer_index,
        "gold_index": outcome.gold_index, "correct": outcome.correct,
        "status": outcome.status,
    }
    row = validate_task_answer_row(
        row, task, context="frozen-prompt diagnostic",
        require_reported_gold_index=True)
    row.update({
        "bridge_attempts": getattr(outcome, "pi_bridge_attempts", None),
        "bridge_timeouts": getattr(outcome, "pi_bridge_timeouts", None),
        "pi_session_path": getattr(outcome, "pi_session_path", None),
    })
    return row


def build_execution_plan(*, cfg: dict, state, search: dict, search_path: Path,
                         candidate: Candidate, gate: dict, tasks: list[dict],
                         args) -> dict:
    runtime = cfg.get("runtime") or {}
    task_model = (cfg.get("models") or {}).get("task") or {}
    pi_retries = max(0, min(4, int(runtime.get("pi_provider_max_retries", 4))))
    bridge_attempts = max(1, min(3, int(runtime.get("pi_timeout_max_attempts", 2))))
    physical_per_operation = (pi_retries + 1) * bridge_attempts
    task_operations = len(tasks) * 2  # paired baseline and frozen prompt
    logical_by_component = {"task_answer": task_operations}
    physical_by_component = {
        "task_answer": task_operations * physical_per_operation}
    skills_cfg = cfg.get("skills") or {}
    routing_mode = str(skills_cfg.get("routing_mode", "lexical"))
    model_router = (routing_mode == "model"
                    and bool(getattr(state.registry, "skills", []))
                    and not str(skills_cfg.get("pinned_skill") or ""))
    router_max_attempts = max(1, int(task_model.get("max_retries", 6)))
    if model_router:
        logical_by_component["skill_router"] = task_operations
        physical_by_component["skill_router"] = task_operations * router_max_attempts
    logical = sum(logical_by_component.values())
    physical = sum(physical_by_component.values())
    unsigned = {
        "schema": "frozen-gepa-prompt-unseen-dev-plan-v1",
        "role": "exploratory_unseen_dev_replay_not_confirmation",
        "source_search_report": str(search_path.resolve()),
        "source_search_report_sha256": sha256_file(search_path),
        "source_search_protocol_sha256": search.get("protocol_sha256"),
        "candidate_id": candidate.candidate_id,
        "candidate_sha256": sha256_obj(str(candidate.new_content).strip()),
        "candidate_static_gate": gate,
        "config_path": str(Path(args.config).as_posix()),
        "config_sha256": config_fingerprint(cfg),
        "dataset": str(tasks[0].get("dataset") or "") if tasks else "",
        "split": "dev_optimize",
        "start_index": int(args.start_index),
        "task_ids": [str(task["task_id"]) for task in tasks],
        "task_material_sha256": sha256_obj(tasks),
        "release_id": state.release_id,
        "system_prompt_sha256": sha256_obj(state.system_prompt),
        "registry_sha256": state.registry.fingerprint(),
        "task_demos_sha256": sha256_obj(state.task_demos),
        "task_model_id": str(task_model.get("model_id") or ""),
        "task_provider": "pi",
        "skill_routing_mode": routing_mode,
        "skill_router_enabled": bool(model_router),
        "skill_router_provider": str(skills_cfg.get("router_provider") or "http"),
        "skill_router_model_id": str(skills_cfg.get("router_model_id") or ""),
        "skill_router_http_attempts_per_operation_upper_bound": (
            router_max_attempts if model_router else 0),
        "seed": int(args.seed),
        "bootstrap_seed": int(args.bootstrap_seed),
        "metric_calls": int(args.metric_calls),
        "max_candidate_proposals": int(args.max_candidate_proposals),
        "minibatch_size": int(args.minibatch_size),
        "acceptance_criterion": args.acceptance_criterion,
        "multi_objective": bool(args.multi_objective),
        "feedback_signals": bool(args.feedback_signals),
        "use_merge": bool(args.use_merge),
        "retrieval_used": False,
        "pi_provider_retries_after_first": pi_retries,
        "pi_timeout_subprocess_attempts": bridge_attempts,
        "logical_request_cap": logical,
        "physical_request_cap": physical,
        "logical_by_component": logical_by_component,
        "physical_by_component": physical_by_component,
        "request_ledger_required": True,
        "pi_physical_reservations_required": True,
        "model_calls_during_preflight": 0,
        "selection_gate_read": False,
        "final_holdout_read": False,
        "independent_confirmation": False,
        "candidate_selection": False,
        "causal_attribution_supported": False,
        "live_authorized": False,
        "explicit_user_approval_required_before_execution": True,
    }
    return {**unsigned, "plan_sha256": sha256_obj(unsigned)}


def verify_execution_plan(plan: dict, expected_sha256: str) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    actual = sha256_obj(unsigned)
    if plan.get("plan_sha256") != actual:
        raise ValueError("frozen-prompt plan hash is internally inconsistent")
    if expected_sha256 != actual:
        raise ValueError("expected plan hash does not match current frozen-prompt plan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-report", type=Path, required=True)
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--dataset", default="pubmedqa")
    parser.add_argument("--start-index", type=int, default=350)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261035)
    parser.add_argument("--bootstrap-seed", type=int, default=20261036)
    parser.add_argument("--metric-calls", type=int, default=200)
    parser.add_argument("--max-candidate-proposals", type=int, default=8)
    parser.add_argument("--minibatch-size", type=int, default=5)
    parser.add_argument("--acceptance-criterion", default="any_objective_improved")
    parser.add_argument("--multi-objective", action="store_true")
    parser.add_argument("--feedback-signals", action="store_true")
    parser.add_argument("--use-merge", action="store_true")
    parser.add_argument("--reflection-model", default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--logical-request-budget", type=int, default=0)
    parser.add_argument("--physical-request-budget", type=int, default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-plan-sha256", default="")
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 200:
        parser.error("--limit must be 1..200")
    if args.logical_request_budget < 0 or args.physical_request_budget < 0:
        parser.error("request budgets must be non-negative")
    if args.execute and not re.fullmatch(
            r"[0-9a-f]{64}", args.expected_plan_sha256 or ""):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("probe requires the isolated task-input sandbox RSI_HOME")
    if args.start_index < 0 or args.limit < 1:
        raise ValueError("start-index must be nonnegative and limit positive")

    cfg = _config(args)
    search = read_json(args.search_report)
    state = load_state(search.get("release_id"))
    if search.get("release_id") != state.release_id:
        raise ValueError("search report release is not the active isolated release")
    if (search.get("input_mode") != "task_input" or
            search.get("task_model_id") != cfg["models"]["task"].get("model_id") or
            search.get("reflection_model_id") !=
            cfg["models"]["reflection"].get("model_id")):
        raise ValueError("probe task/reflection models do not match candidate search")
    if (bool(search.get("multi_objective", False)) != args.multi_objective or
            search.get("acceptance_criterion") != args.acceptance_criterion or
            bool(search.get("use_merge", False)) != args.use_merge or
            int(search.get("max_metric_calls", -1)) != args.metric_calls or
            int(search.get("max_candidate_proposals", -1)) != args.max_candidate_proposals or
            int(search.get("reflection_minibatch_size", -1)) != args.minibatch_size):
        raise ValueError("probe optimizer settings do not match candidate search")
    index = int(search.get("selected_idx", -1))
    texts = search.get("candidate_texts") or []
    if index <= 0 or index >= len(texts):
        raise ValueError("search report has no selected non-seed challenger")
    prompt = str(texts[index]).strip()
    if sha256_obj(prompt) != search.get("selected_sha256"):
        raise ValueError("selected prompt hash does not match the search report")

    tasks = sorted(load_tasks(args.dataset, splits=["dev_optimize"]),
                   key=lambda item: str(item["task_id"]))
    if cfg.get("evaluation", {}).get("insufficient_option", True):
        tasks = inject_all(tasks)
    eval_tasks = tasks[args.start_index:args.start_index + args.limit]
    if len(eval_tasks) != args.limit:
        raise ValueError("requested dev slice is outside available dev_optimize IDs")
    eval_ids = [str(item["task_id"]) for item in eval_tasks]
    source_ids = set(search.get("train_ids", [])) | set(search.get("val_ids", []))
    if source_ids.intersection(eval_ids):
        raise ValueError("unseen dev slice overlaps the GEPA train or val IDs")
    if training_question_overlap(prompt, eval_tasks):
        raise ValueError("candidate contains a phrase copied from the diagnostic slice")

    candidate = Candidate(
        candidate_id=content_id("cand", state.release_id, "system_prompt", prompt),
        parent_release_id=state.release_id, artifact_type="system_prompt",
        artifact_id="system_prompt", diff=unified_diff(state.system_prompt, prompt),
        new_content=prompt)
    if validate_rewrite(prompt, state.system_prompt, cfg):
        raise ValueError("candidate fails Prompt rewrite validation")
    candidate_gate = static_gate(candidate, cfg, state)
    if not candidate_gate["passed"]:
        raise ValueError("candidate fails the static gate")

    plan = build_execution_plan(
        cfg=cfg, state=state, search=search, search_path=args.search_report,
        candidate=candidate, gate=candidate_gate, tasks=eval_tasks, args=args)
    if args.execute:
        verify_execution_plan(plan, args.expected_plan_sha256)
        if (args.logical_request_budget != plan["logical_request_cap"]
                or args.physical_request_budget != plan["physical_request_cap"]):
            raise ValueError("execution request budgets must exactly match the frozen plan")

    plan_path = artifacts_path(
        "probe", "gepa_multiobjective_unseen_dev",
        "preflight-" + plan["plan_sha256"] + ".json")
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise RuntimeError("saved frozen-prompt plan differs from recomputed plan")
    else:
        write_json(plan_path, plan)
    if args.validate_only:
        print(json.dumps({
            "status": "validated_not_run", "plan": str(plan_path),
            "plan_sha256": plan["plan_sha256"], "task_count": len(eval_tasks),
            "logical_request_cap": plan["logical_request_cap"],
            "physical_request_cap": plan["physical_request_cap"],
            "model_calls": 0, "selection_gate_read": False,
            "final_holdout_read": False,
        }, ensure_ascii=False))
        return 0

    task_model = (cfg.get("models") or {}).get("task") or {}
    key_env = str(task_model.get("api_key_env") or "").strip()
    if not key_env or not os.environ.get(key_env):
        raise RuntimeError("frozen-prompt execution is missing the configured task-model credential")

    trial = ArtifactState(state.release_id, prompt, state.registry,
                          task_demos=state.task_demos)
    run_id = uuid.uuid4().hex[:12]
    run_dir = artifacts_path(
        "probe", "gepa_multiobjective_unseen_dev",
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

    report = {"execution_plan": plan, "execution_plan_sha256": plan["plan_sha256"],
              "execution_plan_path": str(plan_path),
              "source_search_report_sha256": plan["source_search_report_sha256"],
              "candidate_id": candidate.candidate_id,
              "status": "running", "pairs": [],
              "generation_protocol_sha256": search.get("protocol_sha256"),
              "evaluation_protocol_sha256": config_fingerprint(cfg),
              "request_ledger_path": str(ledger_path),
              "evaluation_is_paired_within_current_runtime": True,
              "search_source_drift_note": (
                  "The GEPA adapter/search sources changed after generation; this frozen "
                  "candidate replay uses the current task runtime identically for both arms.")}
    output = run_dir / "report.json"
    write_json(output, report)
    baseline_rows, candidate_rows = [], []
    for task in eval_tasks:
        try:
            b = TaskRunner(cfg, state, seed=args.seed).run(task)
            br = _row(task, b)
            c = TaskRunner(cfg, trial, seed=args.seed).run(task)
            cr = _row(task, c)
        except Exception as exc:
            report.update({
                "status": "invalid_evaluation_no_score",
                "invalid_task_id": str(task["task_id"]),
                "failure_type": type(exc).__name__,
                "request_ledger": summarize_ledger(ledger_path),
                "complete_paired_score_available": False,
            })
            write_json(output, report)
            print(json.dumps({"report": str(output),
                              "status": report["status"],
                              "invalid_task_id": report["invalid_task_id"]},
                             ensure_ascii=False))
            return 2
        baseline_rows.append(br)
        candidate_rows.append(cr)
        report["pairs"].append({"task_id": br["task_id"],
                                "baseline": br, "candidate": cr})
        write_json(output, report)
    report.update({"status": "complete", "baseline_metrics": _metrics(baseline_rows),
                   "candidate_metrics": _metrics(candidate_rows),
                   "paired_deltas": _paired_ci(baseline_rows, candidate_rows,
                                                args.bootstrap_seed),
                   "baseline_bridge_attempts": sum(x["bridge_attempts"] or 0
                                                    for x in baseline_rows),
                   "candidate_bridge_attempts": sum(x["bridge_attempts"] or 0
                                                     for x in candidate_rows),
                   "baseline_bridge_timeouts": sum(x["bridge_timeouts"] or 0
                                                   for x in baseline_rows),
                   "candidate_bridge_timeouts": sum(x["bridge_timeouts"] or 0
                                                     for x in candidate_rows),
                   "request_ledger": summarize_ledger(ledger_path),
                   "complete_paired_score_available": True,
                   "cost_usd": None, "cost_reason": "model price/cache tariff unregistered"})
    write_json(output, report)
    print(json.dumps({"report": str(output), "status": report["status"],
                      "baseline": report["baseline_metrics"],
                      "candidate": report["candidate_metrics"],
                      "deltas": report["paired_deltas"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
