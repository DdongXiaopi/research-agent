"""Paired diagnostic of a frozen task-demo bundle on an unused dev slice.

This is exploratory mechanism evidence only. It never selects demos from the
evaluation slice and it cannot access selection_gate or final_holdout.
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
from src.evolution.fewshot import parse_demos_bundle, render_demos_block  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.eval.task_rows import validate_task_answer_row  # noqa: E402
from src.evolution.task_input_gates import validate_task_demo_rendering  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.runtime.runner import TaskRunner  # noqa: E402
from src.util import (  # noqa: E402
    artifacts_path, ensure_dir, read_json, sha256_file, sha256_obj, sha256_text,
    write_json,
)


def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    answered = sum(int(row["answer_index"]) >= 0 for row in rows)
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "coverage": answered / n if n else 0.0,
        "answered_precision": correct / answered if answered else 0.0,
        "correct": correct,
        "answered": answered,
    }


def _row(task: dict, outcome) -> dict:
    if (str(getattr(outcome, "task_id", "")) != str(task["task_id"])
            or str(getattr(outcome, "dataset", "")) != str(task.get("dataset", ""))
            or str(getattr(outcome, "split", "")) != str(task.get("split", "dev_optimize"))):
        raise ValueError("fixed-demo outcome identity differs from frozen task")
    row = {
        "task_id": str(outcome.task_id), "dataset": str(outcome.dataset),
        "split": str(outcome.split), "answer_index": outcome.answer_index,
        "gold_index": outcome.gold_index, "correct": outcome.correct,
        "status": outcome.status,
    }
    row = validate_task_answer_row(
        row, task, context="fixed-demo diagnostic",
        require_reported_gold_index=True)
    row.update({
        "pi_session_path": getattr(outcome, "pi_session_path", None),
        "bridge_attempts": getattr(outcome, "pi_bridge_attempts", None),
        "bridge_timeouts": getattr(outcome, "pi_bridge_timeouts", None),
    })
    return row


def build_execution_plan(*, cfg: dict, candidate_cfg: dict, state, candidate,
                         candidate_record_path: Path, static_gate: dict,
                         render_gate: dict, eval_tasks: list[dict], demo_ids: list[str],
                         block: str, args, sorted_dev_count: int,
                         selection_ids: list[str]) -> dict:
    runtime = cfg.get("runtime") or {}
    task_model = (cfg.get("models") or {}).get("task") or {}
    pi_retries = max(0, min(4, int(runtime.get("pi_provider_max_retries", 4))))
    bridge_attempts = max(1, min(3, int(runtime.get("pi_timeout_max_attempts", 2))))
    physical_per_operation = (pi_retries + 1) * bridge_attempts
    task_operations = len(eval_tasks) * 2  # paired baseline and demo candidate
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
        "schema": "fixed-demos-unseen-dev-plan-v1",
        "role": "exploratory_unseen_dev_replay_not_confirmation",
        "candidate_artifact_path": str(candidate_record_path.resolve()),
        "candidate_artifact_sha256": sha256_file(candidate_record_path),
        "candidate_id": candidate.candidate_id,
        "candidate_content_sha256": candidate.content_sha256,
        "candidate_static_gate": static_gate,
        "candidate_render_gate": render_gate,
        "demo_ids": list(demo_ids),
        "demo_block_sha256": sha256_text(block),
        "config_sha256": config_fingerprint(cfg),
        "candidate_config_sha256": config_fingerprint(candidate_cfg),
        "dataset": str(eval_tasks[0].get("dataset") or "") if eval_tasks else "",
        "split": "dev_optimize",
        "start_index": int(args.start_index),
        "sorted_dev_count": int(sorted_dev_count),
        "task_ids": [str(task["task_id"]) for task in eval_tasks],
        "task_material_sha256": sha256_obj(eval_tasks),
        "selection_ids": list(selection_ids),
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
        "candidate_selection": False,
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
        raise ValueError("fixed-demo plan hash is internally inconsistent")
    if expected_sha256 != actual:
        raise ValueError("expected plan hash does not match current fixed-demo plan")


def _transition(baseline: dict, candidate: dict) -> str:
    b_answered = baseline["answer_index"] >= 0
    c_answered = candidate["answer_index"] >= 0
    if not b_answered and not c_answered:
        return "abstain_to_abstain"
    if not b_answered and candidate["correct"]:
        return "abstain_to_correct"
    if not b_answered:
        return "abstain_to_wrong"
    if not c_answered and baseline["correct"]:
        return "correct_to_abstain"
    if not c_answered:
        return "wrong_to_abstain"
    if not baseline["correct"] and candidate["correct"]:
        return "wrong_to_correct"
    if baseline["correct"] and not candidate["correct"]:
        return "correct_to_wrong"
    if baseline["answer_index"] != candidate["answer_index"]:
        return "wrong_answer_changed"
    return "unchanged"


def _gated_task_demos_candidate(record: dict, cfg: dict, state):
    """Authenticate and statically gate an external task-demo artifact."""
    if not isinstance(record, dict) or record.get("artifact_type") != "task_demos":
        raise ValueError("candidate artifact must have artifact_type=task_demos")
    content = record.get("new_content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("candidate task-demo content must be non-empty text")
    if record.get("parent_release_id") != state.release_id:
        raise ValueError("candidate parent release differs from the active release")
    if record.get("artifact_id") != "task_demos":
        raise ValueError("candidate artifact_id must be task_demos")
    if record.get("content_sha256") != sha256_text(content):
        raise ValueError("candidate content hash does not match its task-demo bundle")
    expected_id = content_id("cand", state.release_id, "task_demos", content)
    if record.get("candidate_id") != expected_id:
        raise ValueError("candidate ID does not match its parent and task-demo bundle")
    candidate = Candidate(
        candidate_id=expected_id,
        parent_release_id=state.release_id,
        artifact_type="task_demos",
        artifact_id="task_demos",
        diff=str(record.get("diff") or ""),
        rationale=str(record.get("rationale") or ""),
        new_content=content,
        attributed_class=str(record.get("attributed_class") or ""),
        source_failure_ids=list(record.get("source_failure_ids") or []),
        generator=dict(record.get("generator") or {}),
        content_sha256=sha256_text(content),
        extra=dict(record.get("extra") or {}),
    )
    gate = static_gate(candidate, cfg, state)
    if not gate.get("passed"):
        failures = [f"{row['check']}:{row['detail']}" for row in gate["results"]
                    if not row["passed"]]
        raise ValueError("candidate failed static gate: " + "; ".join(failures))
    return candidate, gate


def _mcnemar_exact(baseline_rows: list[dict], candidate_rows: list[dict]) -> dict:
    b_to_c = sum(not b["correct"] and c["correct"]
                 for b, c in zip(baseline_rows, candidate_rows))
    c_to_b = sum(b["correct"] and not c["correct"]
                 for b, c in zip(baseline_rows, candidate_rows))
    n = b_to_c + c_to_b
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(b_to_c, c_to_b) + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)
    return {"baseline_wrong_to_candidate_correct": b_to_c,
            "baseline_correct_to_candidate_wrong": c_to_b,
            "discordant_pairs": n, "exact_two_sided_p": p}


def _paired_bootstrap(baseline_rows: list[dict], candidate_rows: list[dict],
                      seed: int, samples: int = 10000) -> dict:
    # Use Python's local RNG for stable replay without adding a dependency.
    import random

    rng = random.Random(seed)
    n = len(baseline_rows)
    out = {}
    for metric in ("accuracy", "coverage", "answered_precision"):
        deltas = []
        for _ in range(samples):
            indices = [rng.randrange(n) for _ in range(n)]
            b = _metrics([baseline_rows[i] for i in indices])[metric]
            c = _metrics([candidate_rows[i] for i in indices])[metric]
            deltas.append(c - b)
        deltas.sort()
        out[metric] = {
            "delta": _metrics(candidate_rows)[metric] - _metrics(baseline_rows)[metric],
            "paired_percentile_95_ci": [deltas[int(0.025 * samples)],
                                         deltas[min(samples - 1, int(0.975 * samples))]],
        }
    return out


def _make_config():
    cfg = copy.deepcopy(load_config("configs/experiment.http.pubmedqa-pilot.yaml"))
    cfg["runtime"].update({"provider": "pi", "input_mode": "task_input",
                           "task_input_max_chars": 30000})
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg.get("tools", {}).get("active", [])
        if name != "academic.search"]
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True,
                        help="frozen candidate artifact JSON")
    parser.add_argument("--dataset", default="pubmedqa",
                        help="visible dev dataset whose task evidence will be supplied")
    parser.add_argument("--start-index", type=int, default=150,
                        help="zero-based offset in sorted dev_optimize IDs")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--bootstrap-seed", type=int, default=20261003)
    parser.add_argument("--seed", type=int, default=20261020,
                        help="paired model-repeat label")
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
        raise RuntimeError("probe requires an explicit isolated sandbox RSI_HOME")
    if args.start_index < 0 or args.limit <= 0:
        raise ValueError("start-index must be nonnegative and limit must be positive")

    candidate_record = read_json(Path(args.candidate))
    if candidate_record.get("artifact_type") != "task_demos":
        raise ValueError("candidate artifact must have artifact_type=task_demos")
    demos = parse_demos_bundle(str(candidate_record.get("new_content") or ""))
    demo_ids = [str(item["task_id"]) for item in demos]
    cfg = _make_config()
    render_gate = validate_task_demo_rendering(demos, cfg)
    if not render_gate.get("passed"):
        raise ValueError("candidate task demos fail runtime render gate: "
                         + str(render_gate.get("reason")))
    block = render_demos_block(demos)
    if not block:
        raise ValueError("candidate demo bundle renders to an empty prompt block")

    state = load_state()
    candidate, candidate_static_gate = _gated_task_demos_candidate(
        candidate_record, cfg, state)
    all_tasks = sorted(load_tasks(args.dataset, splits=["dev_optimize"]),
                       key=lambda task: str(task["task_id"]))
    if cfg.get("evaluation", {}).get("insufficient_option", True):
        all_tasks = inject_all(all_tasks)
    eval_tasks = all_tasks[args.start_index:args.start_index + args.limit]
    if len(eval_tasks) != args.limit:
        raise ValueError(f"requested {args.limit} tasks but only found {len(eval_tasks)}")

    source_ids = set((candidate_record.get("generator") or {}).get("train_ids", []))
    source_ids.update((candidate_record.get("generator") or {}).get("val_ids", []))
    overlap = sorted(set(demo_ids) & {str(t["task_id"]) for t in eval_tasks})
    source_overlap = sorted(source_ids & {str(t["task_id"]) for t in eval_tasks})
    if overlap or source_overlap:
        raise ValueError(f"candidate/eval leakage: demo={overlap}, selection={source_overlap[:5]}")

    candidate_cfg = copy.deepcopy(cfg)
    candidate_cfg["runtime"].update({"pi_task_demos_block": block,
                                     "pi_task_demo_ids": demo_ids,
                                     "pi_task_demo_split": "dev_optimize"})
    plan = build_execution_plan(
        cfg=cfg, candidate_cfg=candidate_cfg, state=state, candidate=candidate,
        candidate_record_path=Path(args.candidate),
        static_gate=candidate_static_gate, render_gate=render_gate,
        eval_tasks=eval_tasks, demo_ids=demo_ids, block=block, args=args,
        sorted_dev_count=len(all_tasks), selection_ids=sorted(source_ids))
    if args.execute:
        verify_execution_plan(plan, args.expected_plan_sha256)
        if (args.logical_request_budget != plan["logical_request_cap"]
                or args.physical_request_budget != plan["physical_request_cap"]):
            raise ValueError("execution request budgets must exactly match the frozen plan")

    plan_path = artifacts_path(
        "probe", "fixed_demos_unseen_dev",
        "preflight-" + plan["plan_sha256"] + ".json")
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise RuntimeError("saved fixed-demo plan differs from recomputed plan")
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
        raise RuntimeError("fixed-demo execution is missing the configured task-model credential")

    pairs = []
    # Sequential paired calls keep provider load bounded and preserve a clean
    # per-task baseline-then-candidate trace.
    run_id = uuid.uuid4().hex[:12]
    run_dir = artifacts_path(
        "probe", "fixed_demos_unseen_dev",
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

    report = {
        "protocol": "pi-fixed-task-demos-unseen-dev-v1-exploratory",
        "execution_plan": plan,
        "execution_plan_sha256": plan["plan_sha256"],
        "execution_plan_path": str(plan_path),
        "request_ledger_path": str(ledger_path),
        "candidate_id": candidate.candidate_id,
        "candidate_sha256": candidate.content_sha256,
        "candidate_source": str(Path(args.candidate).resolve()),
        "release_id": state.release_id,
        "baseline_config_sha256": config_fingerprint(cfg),
        "candidate_config_sha256": config_fingerprint(candidate_cfg),
        "candidate_generation_protocol_sha256": (candidate.get("generator") or {}).get(
            "search_protocol_sha256"),
        "status": "running", "pairs": pairs,
    }
    output = run_dir / "report.json"
    write_json(output, report)
    for idx, task in enumerate(eval_tasks, start=1):
        try:
            baseline_outcome = TaskRunner(cfg, state, seed=args.seed).run(task)
            b = _row(task, baseline_outcome)
            candidate_outcome = TaskRunner(candidate_cfg, state, seed=args.seed).run(task)
            c = _row(task, candidate_outcome)
        except Exception as exc:
            report.update({
                "status": "invalid_evaluation_no_score",
                "invalid_task_id": str(task["task_id"]),
                "failure_type": type(exc).__name__,
                "request_ledger": summarize_ledger(ledger_path),
                "complete_paired_score_available": False,
            })
            write_json(output, report)
            print(json.dumps({"path": str(output), "status": report["status"],
                              "invalid_task_id": report["invalid_task_id"]},
                             ensure_ascii=False))
            return 2
        pairs.append({"task_id": str(task["task_id"]), "baseline": b,
                      "candidate": c, "transition": _transition(b, c)})
        report["completed_pairs"] = idx
        write_json(output, report)
        print(json.dumps({"completed": idx, "total": len(eval_tasks),
                          "task_id": str(task["task_id"]),
                          "transition": pairs[-1]["transition"]}, ensure_ascii=False),
              flush=True)

    baseline_rows = [pair["baseline"] for pair in pairs]
    candidate_rows = [pair["candidate"] for pair in pairs]
    transitions = {}
    for pair in pairs:
        key = pair["transition"]
        transitions[key] = transitions.get(key, 0) + 1
    all_rows = baseline_rows + candidate_rows
    attempts_known = all(row["bridge_attempts"] is not None for row in all_rows)
    timeouts_known = all(row["bridge_timeouts"] is not None for row in all_rows)
    report.update({
        "status": "completed",
        "baseline_metrics": _metrics(baseline_rows),
        "candidate_metrics": _metrics(candidate_rows),
        "transition_counts": transitions,
        "accuracy_mcnemar": _mcnemar_exact(baseline_rows, candidate_rows),
        "paired_bootstrap": _paired_bootstrap(
            baseline_rows, candidate_rows, args.bootstrap_seed),
        "total_bridge_attempts": (sum(int(row["bridge_attempts"]) for row in all_rows)
                                   if attempts_known else None),
        "total_bridge_timeouts": (sum(int(row["bridge_timeouts"]) for row in all_rows)
                                   if timeouts_known else None),
        "request_ledger": summarize_ledger(ledger_path),
        "complete_paired_score_available": True,
        "pairs": pairs,
    })
    write_json(output, report)
    print(json.dumps({"path": str(output), "status": report["status"],
                      "baseline_metrics": report["baseline_metrics"],
                      "candidate_metrics": report["candidate_metrics"],
                      "transition_counts": transitions,
                      "accuracy_mcnemar": report["accuracy_mcnemar"],
                      "paired_bootstrap": report["paired_bootstrap"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
