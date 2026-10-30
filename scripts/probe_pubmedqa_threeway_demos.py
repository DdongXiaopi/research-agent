"""Paired exploratory native-label task-demo probe on PubMedQA dev data.

The probe asks whether a small, fixed set of native yes/no/maybe examples
changes answers when task evidence is held constant. It is not confirmation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_fingerprint, load_config  # noqa: E402
from src.contracts.types import Candidate, content_id, content_sha256  # noqa: E402
from src.data.pubmedqa_threeway_probe import DATASET, OPTIONS  # noqa: E402
from src.data.split import load_tasks  # noqa: E402
from src.evolution.fewshot import render_demos_block, serialize_demos_bundle  # noqa: E402
from src.eval.task_rows import validate_task_answer_row  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.runtime.runner import TaskRunner  # noqa: E402
from src.util import (  # noqa: E402
    artifacts_path, ensure_dir, sha256_obj, sha256_text, write_json,
)


def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    answered = sum(row["answer_index"] >= 0 for row in rows)
    result = {
        "n": n,
        "accuracy": sum(bool(row["correct"]) for row in rows) / n if n else 0.0,
        "coverage": answered / n if n else 0.0,
        "answered_precision": (sum(bool(row["correct"]) for row in rows) / answered
                               if answered else 0.0),
        "abstentions": n - answered,
        "per_label": {},
    }
    for label in OPTIONS:
        subset = [row for row in rows if row["gold_label"] == label]
        result["per_label"][label] = {
            "n": len(subset),
            "recall": (sum(bool(row["correct"]) for row in subset) / len(subset)
                       if subset else None),
            "abstention_rate": (sum(row["answer_index"] < 0 for row in subset) / len(subset)
                                if subset else None),
        }
    return result


def _run_row(task: dict, outcome) -> dict:
    if str(getattr(outcome, "status", "")) not in {"ok", "abstain"}:
        raise ValueError("PubMedQA demo probe received an unscorable task outcome")
    if (str(getattr(outcome, "task_id", "")) != str(task.get("task_id", ""))
            or str(getattr(outcome, "dataset", "")) != str(task.get("dataset", DATASET))
            or str(getattr(outcome, "split", "")) != str(task.get("split", "dev_optimize"))):
        raise ValueError("PubMedQA demo outcome identity differs from frozen task")
    label = str(task["options"][task["correct_index"]]).strip().casefold()
    row = {
        "task_id": task["task_id"], "gold_label": label,
        "dataset": str(outcome.dataset), "split": str(outcome.split),
        "answer_index": outcome.answer_index,
        "gold_index": outcome.gold_index,
        "answer_label": (task["options"][outcome.answer_index]
                         if 0 <= outcome.answer_index < len(task["options"]) else "abstain"),
        "correct": bool(outcome.correct), "status": outcome.status,
        "skill": outcome.skill_recalled, "pi_session_path": outcome.pi_session_path,
        "cost": outcome.cost,
        "input_doc_ids": outcome.input_doc_ids,
        "input_passage_hashes": outcome.input_passage_hashes,
        "searched": any(event.get("tool_name") == "academic.search"
                         for event in outcome.events),
    }
    return validate_task_answer_row(
        row, task, context="PubMedQA three-way demo probe",
        require_reported_gold_index=True)


def gate_demo_candidate(state, cfg: dict, bundle: str) -> tuple[Candidate, dict]:
    """Bind the exploratory demo arm to its parent and run the shared gate."""
    candidate = Candidate(
        candidate_id=content_id("cand", state.release_id, "task_demos", bundle),
        parent_release_id=state.release_id, artifact_type="task_demos",
        artifact_id="task_demos", diff="add three native-label examples",
        new_content=bundle, content_sha256=content_sha256(bundle))
    static_result = static_gate(candidate, cfg, state=state)
    if not static_result["passed"]:
        raise RuntimeError("demo candidate failed static/grounding checks: "
                           + json.dumps(static_result, ensure_ascii=False))
    return candidate, static_result


def _make_config() -> dict:
    cfg = copy.deepcopy(load_config("configs/experiment.http.pubmedqa-pilot.yaml"))
    cfg["runtime"].update({"provider": "pi", "input_mode": "task_input",
                           "task_input_max_chars": 30000})
    cfg["runtime"].pop("pi_fake_response", None)
    cfg["evaluation"]["insufficient_option"] = False
    cfg["evolution"]["allowed_artifacts"] = ["task_demos"]
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg["tools"].get("active", []) if name != "academic.search"]
    return cfg


def build_execution_plan(*, cfg: dict, state, candidate: Candidate,
                         static_result: dict, demos: list[dict], eval_tasks: list[dict],
                         per_label: int, evaluation_offset: int) -> dict:
    runtime = cfg.get("runtime") or {}
    task_model = (cfg.get("models") or {}).get("task") or {}
    skills = cfg.get("skills") or {}
    task_runs = len(eval_tasks) * 2
    pi_retries = max(0, min(4, int(runtime.get("pi_provider_max_retries", 4))))
    bridge_attempts = max(1, min(3, int(runtime.get("pi_timeout_max_attempts", 2))))
    pi_attempts = (pi_retries + 1) * bridge_attempts
    model_router = (str(skills.get("routing_mode", "lexical")) == "model"
                    and bool(getattr(state.registry, "skills", []))
                    and not str(skills.get("pinned_skill") or ""))
    router_runs = task_runs if model_router else 0
    router_attempts = max(1, int(task_model.get("max_retries", 6)))
    logical_by_component = {"task_answer": task_runs}
    physical_by_component = {"task_answer": task_runs * pi_attempts}
    if router_runs:
        logical_by_component["skill_router"] = router_runs
        physical_by_component["skill_router"] = router_runs * router_attempts
    candidate_cfg = copy.deepcopy(cfg)
    block = render_demos_block(demos)
    demo_ids = [str(item["task_id"]) for item in demos]
    candidate_cfg["runtime"].update({
        "pi_task_demos_block": block,
        "pi_task_demo_ids": demo_ids,
        "pi_task_demo_split": "dev_optimize",
    })
    unsigned = {
        "schema": "pubmedqa-native-threeway-demo-plan-v1",
        "protocol": "pubmedqa-native-threeway-fixed-demo-pair-v2-plan-bound",
        "role": "exploratory_fixed_demo_pair_not_candidate_selection_or_confirmation",
        "dataset": DATASET,
        "split": "dev_optimize",
        "per_label_evaluation_n": per_label,
        "evaluation_offset_after_demo": evaluation_offset,
        "cohort_policy": "first lexical task ID per class is demo; next per-label rows are paired evaluation",
        "demo_ids": demo_ids,
        "evaluation_ids": [str(task["task_id"]) for task in eval_tasks],
        "task_material_sha256": sha256_obj({
            "demos": [item["task"] for item in demos],
            "evaluation": eval_tasks,
        }),
        "demo_bundle_sha256": content_sha256(serialize_demos_bundle(demos)),
        "demo_block_sha256": sha256_text(block),
        "candidate_id": candidate.candidate_id,
        "candidate_content_sha256": candidate.content_sha256,
        "parent_release_id": state.release_id,
        "parent_system_prompt_sha256": sha256_text(state.system_prompt),
        "parent_registry_sha256": state.registry.fingerprint(),
        "parent_task_demos_sha256": sha256_obj(getattr(state, "task_demos", [])),
        "candidate_static_gate": static_result,
        "baseline_config_sha256": config_fingerprint(cfg),
        "candidate_config_sha256": config_fingerprint(candidate_cfg),
        "task_model_id": str(task_model.get("model_id") or ""),
        "task_provider": "pi",
        "skill_routing_mode": str(skills.get("routing_mode", "lexical")),
        "task_answer_operations_upper_bound": task_runs,
        "skill_router_operations_upper_bound": router_runs,
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
        "selection_gate_read": False,
        "official_validation_read": False,
        "official_test_read": False,
        "final_holdout_read": False,
        "retrieval_used": False,
        "eligible_for_confirmation": False,
        "causal_attribution_supported": False,
        "live_authorized": False,
        "explicit_user_approval_required_before_execution": True,
    }
    return {**unsigned, "plan_sha256": sha256_obj(unsigned)}


def verify_execution_plan(plan: dict, expected_sha256: str) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    actual = sha256_obj(unsigned)
    if plan.get("plan_sha256") != actual:
        raise ValueError("PubMedQA demo execution plan is internally inconsistent")
    if expected_sha256 != actual:
        raise ValueError("expected plan hash does not match current PubMedQA demo plan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-label", type=int, default=4,
                        help="evaluation tasks per native label; one demo per label")
    parser.add_argument("--evaluation-offset", type=int, default=0,
                        help="skip this many eligible rows per label after the demo")
    parser.add_argument("--logical-request-budget", type=int, default=0)
    parser.add_argument("--physical-request-budget", type=int, default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-plan-sha256", default="")
    args = parser.parse_args(argv)
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("three-way demo probe requires an explicit sandbox RSI_HOME")
    if args.per_label < 1 or args.evaluation_offset < 0:
        raise ValueError("--per-label must be positive and --evaluation-offset nonnegative")
    if args.logical_request_budget < 0 or args.physical_request_budget < 0:
        parser.error("request budgets must be non-negative")
    if args.execute and not re.fullmatch(
            r"[0-9a-f]{64}", args.expected_plan_sha256 or ""):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")

    cfg = _make_config()

    tasks = load_tasks(DATASET, splits=["dev_optimize"])
    grouped = defaultdict(list)
    for task in tasks:
        label = str(task["options"][task["correct_index"]]).strip().casefold()
        if label not in OPTIONS:
            raise RuntimeError(f"unexpected native label in {task['task_id']}: {label}")
        grouped[label].append(task)
    for label in OPTIONS:
        grouped[label].sort(key=lambda task: task["task_id"])
        if len(grouped[label]) < args.per_label + args.evaluation_offset + 1:
            raise RuntimeError(f"not enough {label} rows for one demo plus evaluation")

    demos = []
    eval_tasks = []
    for label in OPTIONS:
        demo_task = grouped[label][0]
        start = 1 + args.evaluation_offset
        remaining = grouped[label][start:start + args.per_label]
        demos.append({"task_id": demo_task["task_id"], "task": demo_task,
                      "correct_index": demo_task["correct_index"]})
        eval_tasks.extend(remaining)
    eval_tasks.sort(key=lambda task: task["task_id"])
    demo_ids = [item["task_id"] for item in demos]
    eval_ids = [task["task_id"] for task in eval_tasks]
    if set(demo_ids) & set(eval_ids):
        raise RuntimeError("demo/evaluation overlap")

    state = load_state()
    bundle = serialize_demos_bundle(demos)
    candidate, static_result = gate_demo_candidate(state, cfg, bundle)
    block = render_demos_block(demos)
    candidate_cfg = copy.deepcopy(cfg)
    candidate_cfg["runtime"].update({
        "pi_task_demos_block": block,
        "pi_task_demo_ids": demo_ids,
        "pi_task_demo_split": "dev_optimize",
    })

    plan = build_execution_plan(
        cfg=cfg, state=state, candidate=candidate, static_result=static_result,
        demos=demos, eval_tasks=eval_tasks, per_label=args.per_label,
        evaluation_offset=args.evaluation_offset)
    if args.execute:
        verify_execution_plan(plan, args.expected_plan_sha256)
        if (args.logical_request_budget != plan["logical_request_cap"]
                or args.physical_request_budget != plan["physical_request_cap"]):
            raise ValueError("request budgets must exactly match the frozen plan")
    plan_path = artifacts_path(
        "probe", "pubmedqa_threeway",
        "preflight-" + plan["plan_sha256"] + ".json")
    if plan_path.exists():
        from src.util import read_json
        if read_json(plan_path) != plan:
            raise ValueError("saved PubMedQA three-way plan differs from recomputed plan")
    else:
        write_json(plan_path, plan)
    if args.validate_only:
        print(json.dumps({
            "status": "validated_not_run",
            "plan": str(plan_path),
            "plan_sha256": plan["plan_sha256"],
            "evaluation_n": len(eval_tasks),
            "demo_ids": demo_ids,
            "logical_request_cap": plan["logical_request_cap"],
            "physical_request_cap": plan["physical_request_cap"],
            "model_calls": 0,
            "selection_gate_read": False,
            "final_holdout_read": False,
        }, ensure_ascii=False))
        return 0

    task_model = (cfg.get("models") or {}).get("task") or {}
    key_env = str(task_model.get("api_key_env") or "").strip()
    if not key_env or not os.environ.get(key_env):
        raise RuntimeError("PubMedQA demo execution is missing the configured task-model credential")

    run_id = uuid.uuid4().hex[:12]
    run_dir = artifacts_path("probe", "pubmedqa_threeway",
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
    output = run_dir / ("report-" + plan["plan_sha256"] + ".json")
    try:
        for task in eval_tasks:
            baseline = TaskRunner(cfg, state).run(task)
            baseline_row = _run_row(task, baseline)
            candidate_outcome = TaskRunner(candidate_cfg, state).run(task)
            candidate_row = _run_row(task, candidate_outcome)
            pairs.append({"task_id": task["task_id"],
                          "baseline": baseline_row,
                          "candidate": candidate_row})
    except Exception as exc:
        write_json(output, {
            "protocol": plan["protocol"],
            "execution_plan_sha256": plan["plan_sha256"],
            "status": "invalid_no_score",
            "failure_type": type(exc).__name__,
            "completed_pairs": len(pairs),
            "request_ledger_path": str(ledger_path),
            "request_ledger_summary": summarize_ledger(ledger_path),
            "eligible_for_confirmation": False,
            "causal_attribution_supported": False,
        })
        raise RuntimeError("PubMedQA demo execution failed; no paired score was produced") from None
    if len(pairs) != len(eval_tasks):
        raise RuntimeError("incomplete paired task coverage; no score was produced")

    baseline_rows = [pair["baseline"] for pair in pairs]
    candidate_rows = [pair["candidate"] for pair in pairs]
    transitions = defaultdict(int)
    for pair in pairs:
        before, after = pair["baseline"], pair["candidate"]
        transitions[f"{before['answer_label']} -> {after['answer_label']}"] += 1
    report = {
        **plan,
        "protocol": plan["protocol"],
        "execution_plan_path": str(plan_path),
        "request_ledger_path": str(ledger_path),
        "request_ledger_summary": summarize_ledger(ledger_path),
        "status": "complete",
        "dataset": DATASET, "split": "dev_optimize",
        "eligible_for_confirmation": False,
        "release_id": state.release_id,
        "selection_gate_access": False, "final_holdout_access": False,
        "retrieval_used": False,
        "label_policy": "native yes/no/maybe; synthetic Insufficient option disabled",
        "cohort_policy": "first lexical task ID per class is demo; next per-label rows are paired evaluation",
        "evaluation_offset_after_demo": args.evaluation_offset,
        "per_label_evaluation_n": args.per_label,
        "demo_ids": demo_ids, "evaluation_ids": eval_ids,
        "demo_bundle_sha256": content_sha256(bundle),
        "candidate_id": candidate.candidate_id,
        "demo_block_sha256": sha256_text(block),
        "static_source_gate": static_result,
        "baseline_config_sha256": config_fingerprint(cfg),
        "candidate_config_sha256": config_fingerprint(candidate_cfg),
        "baseline_metrics": _metrics(baseline_rows),
        "candidate_metrics": _metrics(candidate_rows),
        "answer_transitions": dict(sorted(transitions.items())),
        "fixed_ids": [pair["task_id"] for pair in pairs
                      if not pair["baseline"]["correct"] and pair["candidate"]["correct"]],
        "broken_ids": [pair["task_id"] for pair in pairs
                       if pair["baseline"]["correct"] and not pair["candidate"]["correct"]],
        "pairs": pairs,
    }
    write_json(output, report)
    print(json.dumps({"path": str(output), "demo_ids": demo_ids,
                      "evaluation_n": len(eval_tasks),
                      "baseline_metrics": report["baseline_metrics"],
                      "candidate_metrics": report["candidate_metrics"],
                      "answer_transitions": report["answer_transitions"],
                      "static_source_gate_passed": static_result["passed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
