"""Visible optimize-only counterfactual for a prewritten outcome interpretation rule.

Live execution is separated from a deterministic zero-call preflight and
requires an exact plan hash challenge plus bounded request-ledger budgets.
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
from src.data.split import load_tasks  # noqa: E402
from src.eval.options import inject_all  # noqa: E402
from src.eval.task_rows import validate_task_answer_row  # noqa: E402
from scripts.probe_pi_decision_rule_arms import build_rule_arm  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.runtime.runner import TaskRunner  # noqa: E402
from src.util import (  # noqa: E402
    artifacts_path, ensure_dir, sha256_obj, sha256_text, write_json,
)


def _prepare_config(config_path: str) -> dict:
    cfg = copy.deepcopy(load_config(config_path))
    cfg["runtime"].update({"provider": "pi", "input_mode": "task_input",
                           "task_input_max_chars": 30000})
    cfg.setdefault("evaluation", {}).update({"main_metric": "accuracy",
                                               "primary_axis": "accuracy",
                                               "sealed_pairing": False})
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg["tools"].get("active", []) if name != "academic.search"]
    return cfg


def build_execution_plan(*, cfg: dict, state, tasks: list[dict],
                         config_path: str, rule_path: Path, rule: str,
                         candidate_id: str, candidate_prompt_sha256: str,
                         static_gate: dict, seed: int) -> dict:
    runtime = cfg.get("runtime") or {}
    task_model = (cfg.get("models") or {}).get("task") or {}
    pi_retries = max(0, min(4, int(runtime.get("pi_provider_max_retries", 4))))
    bridge_attempts = max(1, min(3, int(runtime.get("pi_timeout_max_attempts", 2))))
    pi_attempts = (pi_retries + 1) * bridge_attempts
    logical = len(tasks) * 2  # baseline and frozen rule candidate
    unsigned = {
        "schema": "pi-decision-rule-visible-optimize-plan-v1",
        "role": "one_shot_exploratory_counterfactual",
        "config_path": str(Path(config_path).as_posix()),
        "config_sha256": config_fingerprint(cfg),
        "dataset": str(tasks[0].get("dataset") or "") if tasks else "",
        "split": "dev_optimize",
        "task_ids": [str(task["task_id"]) for task in tasks],
        "task_material_sha256": sha256_obj(tasks),
        "release_id": state.release_id,
        "system_prompt_sha256": sha256_text(state.system_prompt),
        "registry_sha256": state.registry.fingerprint(),
        "task_demos_sha256": sha256_obj(getattr(state, "task_demos", [])),
        "candidate_id": candidate_id,
        "candidate_prompt_sha256": candidate_prompt_sha256,
        "static_gate": static_gate,
        "rule_path": str(rule_path),
        "rule_sha256": sha256_text(rule),
        "task_model_id": str(task_model.get("model_id") or ""),
        "task_provider": "pi",
        "seed": int(seed),
        "semantic_citation_judge_enabled": False,
        "retrieval_used": False,
        "pi_provider_retries_after_first": pi_retries,
        "pi_timeout_subprocess_attempts": bridge_attempts,
        "pi_http_attempts_per_operation_upper_bound": pi_attempts,
        "logical_request_cap": logical,
        "physical_request_cap": logical * pi_attempts,
        "logical_by_component": {"task_answer": logical},
        "physical_by_component": {"task_answer": logical * pi_attempts},
        "request_ledger_required": True,
        "pi_physical_reservations_required": True,
        "model_calls_during_preflight": 0,
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


def _outcome_row(task: dict, outcome) -> dict:
    if (str(getattr(outcome, "task_id", "")) != str(task.get("task_id", ""))
            or str(getattr(outcome, "dataset", "")) != str(task.get("dataset", ""))
            or str(getattr(outcome, "split", "")) != str(task.get("split", "dev_optimize"))):
        raise ValueError("Pi decision-rule outcome identity differs from frozen task")
    row = {
        "task_id": str(outcome.task_id), "dataset": str(outcome.dataset),
        "split": str(outcome.split), "answer_index": outcome.answer_index,
        "gold_index": outcome.gold_index, "correct": outcome.correct,
        "status": outcome.status, "pi_session_path": outcome.pi_session_path,
        "cost": outcome.cost, "error": outcome.error,
    }
    return validate_task_answer_row(
        row, task, context="Pi decision-rule counterfactual outcome",
        require_reported_gold_index=True)


def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    answered = sum(row["answer_index"] >= 0 for row in rows)
    return {"n": n, "accuracy": correct / n if n else 0.0,
            "coverage": answered / n if n else 0.0,
            "answered_precision": correct / answered if answered else 0.0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--rule-file", default="configs/task_input/strategy_target_outcome_v1.txt")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261010)
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
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("decision-rule probe requires an explicit sandbox RSI_HOME")

    cfg = _prepare_config(args.config)
    rule_path = (ROOT / args.rule_file).resolve()
    if not rule_path.is_relative_to((ROOT / "configs" / "task_input").resolve()):
        raise RuntimeError("rule file must be under configs/task_input")
    rule = rule_path.read_text(encoding="utf-8").strip()
    if not rule:
        raise ValueError("decision rule must be non-empty text")
    state = load_state()
    arm = build_rule_arm(state, cfg, rule_path, rule, "rule")
    tasks = sorted(load_tasks("pubmedqa", splits=["dev_optimize"]),
                   key=lambda task: task["task_id"])[:args.limit]
    if len(tasks) != args.limit:
        raise ValueError(f"requested {args.limit} tasks but selected {len(tasks)}")
    if cfg["evaluation"].get("insufficient_option", True):
        tasks = inject_all(tasks)
    plan = build_execution_plan(
        cfg=cfg, state=state, tasks=tasks, config_path=args.config,
        rule_path=rule_path, rule=rule, candidate_id=arm["candidate_id"],
        candidate_prompt_sha256=sha256_text(arm["artifact_state"].system_prompt),
        static_gate=arm["static_gate"], seed=args.seed)
    if args.execute:
        verify_execution_plan(plan, args.expected_plan_sha256)
        if (args.logical_request_budget != plan["logical_request_cap"]
                or args.physical_request_budget != plan["physical_request_cap"]):
            raise ValueError("execution request budgets must exactly match the frozen plan")

    plan_path = artifacts_path("probe", "pi_decision_rule",
                               "preflight-" + plan["plan_sha256"] + ".json")
    if plan_path.exists():
        from src.util import read_json
        if read_json(plan_path) != plan:
            raise RuntimeError("saved Pi decision-rule preflight differs from recomputed plan")
    else:
        write_json(plan_path, plan)
    if args.validate_only:
        print(json.dumps({
            "status": "validated_not_run", "plan": str(plan_path),
            "plan_sha256": plan["plan_sha256"], "task_count": len(tasks),
            "logical_request_cap": plan["logical_request_cap"],
            "physical_request_cap": plan["physical_request_cap"],
            "model_calls": 0, "selection_gate_read": False,
            "final_holdout_read": False,
        }, ensure_ascii=False))
        return 0

    task_model = (cfg.get("models") or {}).get("task") or {}
    key_env = str(task_model.get("api_key_env") or "").strip()
    if not key_env or not os.environ.get(key_env):
        raise RuntimeError("Pi decision-rule execution is missing the configured task-model credential")
    run_id = uuid.uuid4().hex[:12]
    run_dir = artifacts_path("probe", "pi_decision_rule",
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

    pairs = []
    for task in tasks:
        base = _outcome_row(task, TaskRunner(cfg, state, seed=args.seed).run(task))
        changed = _outcome_row(task, TaskRunner(
            cfg, arm["artifact_state"], seed=args.seed).run(task))
        pairs.append({"task_id": task["task_id"], "baseline": base,
                      "candidate": changed})
    baseline_rows = [pair["baseline"] for pair in pairs]
    candidate_rows = [pair["candidate"] for pair in pairs]
    report = {
        "protocol": "pi-target-outcome-rule-visible-optimize-v2",
        "execution_plan_sha256": plan["plan_sha256"],
        "execution_plan_path": str(plan_path),
        "dataset": "pubmedqa", "split": "dev_optimize",
        "config_fingerprint": config_fingerprint(cfg),
        "release_id": state.release_id,
        "candidate_id": arm["candidate_id"],
        "candidate_prompt_sha256": sha256_text(arm["artifact_state"].system_prompt),
        "rule_path": str(rule_path), "rule_sha256": sha256_text(rule),
        "static_gate": arm["static_gate"],
        "baseline_metrics": _metrics(baseline_rows),
        "candidate_metrics": _metrics(candidate_rows),
        "changed_answer_ids": [pair["task_id"] for pair in pairs
                               if pair["baseline"]["answer_index"] != pair["candidate"]["answer_index"]],
        "fixed_ids": [pair["task_id"] for pair in pairs
                      if not pair["baseline"]["correct"] and pair["candidate"]["correct"]],
        "broken_ids": [pair["task_id"] for pair in pairs
                       if pair["baseline"]["correct"] and not pair["candidate"]["correct"]],
        "pairs": pairs,
        "request_ledger": str(ledger_path),
        "request_ledger_summary": summarize_ledger(ledger_path),
        "selection_gate_read": False, "final_holdout_read": False,
        "candidate_selection_or_confirmation": False,
        "causal_attribution_supported": False,
    }
    report_path = run_dir / ("report-" + plan["plan_sha256"] + ".json")
    write_json(report_path, report)
    print(json.dumps({"path": str(report_path),
                      "baseline_metrics": report["baseline_metrics"],
                      "candidate_metrics": report["candidate_metrics"],
                      "changed_answer_ids": report["changed_answer_ids"],
                      "fixed_ids": report["fixed_ids"],
                      "broken_ids": report["broken_ids"],
                      "request_ledger": report["request_ledger_summary"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
