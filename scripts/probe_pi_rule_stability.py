"""Repeat the two observed optimize-set flips under fixed Pi task inputs."""

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
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.runtime.runner import ArtifactState, TaskRunner  # noqa: E402
from src.util import (  # noqa: E402
    artifacts_path, ensure_dir, sha256_obj, sha256_text, write_json,
)


SELECTED_TASK_IDS = ("11340218", "11566686")
REPEATS = 5
ARMS = ("baseline", "rule")


def build_stability_rule_candidate(cfg: dict, state, rule_path: Path,
                                   rule: str) -> tuple[Candidate, ArtifactState, dict]:
    """Gate a prompt-only stability arm and preserve every parent component."""
    if not isinstance(rule, str) or not rule.strip():
        raise ValueError("stability rule must be non-empty text")
    prompt = state.system_prompt.rstrip() + "\n\n" + rule.strip()
    candidate = Candidate(
        candidate_id=content_id("cand", state.release_id, "system_prompt", prompt),
        parent_release_id=state.release_id, artifact_type="system_prompt",
        artifact_id="system_prompt", diff=f"append stability rule from {rule_path.name}",
        new_content=prompt, content_sha256=content_sha256(prompt))
    gate = static_gate(candidate, cfg, state)
    if not gate.get("passed"):
        raise ValueError("stability Prompt failed static gate: "
                         + str(gate.get("reason") or gate.get("results")))
    trial = ArtifactState(
        state.release_id, prompt, state.registry,
        task_demos=copy.deepcopy(state.task_demos))
    return candidate, trial, gate


def build_execution_plan(*, cfg: dict, state, tasks: list[dict],
                         config_path: str, rule_path: Path, rule: str,
                         candidate, static_gate_result: dict,
                         repeats: int = REPEATS, seed: int = 0) -> dict:
    """Freeze the exact optimize-only scope and worst-case Pi request budget."""
    if not tasks or repeats < 1:
        raise ValueError("stability plan requires tasks and at least one repeat")
    if any(task.get("split") != "dev_optimize" for task in tasks):
        raise ValueError("stability probe may read dev_optimize tasks only")
    if static_gate_result.get("passed") is not True:
        raise ValueError("stability candidate must pass its static gate before planning")
    runtime = cfg.get("runtime") or {}
    task_model = (cfg.get("models") or {}).get("task") or {}
    skills = cfg.get("skills") or {}
    if str(runtime.get("provider", "")).lower() != "pi":
        raise ValueError("stability execution plan requires the Pi provider")
    router_enabled = (str(skills.get("routing_mode", "lexical")) == "model"
                      and bool(state.registry.skills)
                      and not str(skills.get("pinned_skill") or ""))
    router_provider = str(skills.get("router_provider", "http")).lower()
    if router_enabled and router_provider != "http":
        raise ValueError("model Skill routing requires the measured HTTP router")

    runs = len(tasks) * repeats * len(ARMS)
    router_logical = runs if router_enabled else 0
    pi_retries = max(0, min(4, int(runtime.get("pi_provider_max_retries", 4))))
    bridge_attempts = max(1, min(3, int(runtime.get("pi_timeout_max_attempts", 2))))
    pi_attempts = (pi_retries + 1) * bridge_attempts
    router_attempts = max(1, int(task_model.get("max_retries", 6)))
    logical_by_component = {"task_answer": runs, "skill_router": router_logical}
    physical_by_component = {
        "task_answer": runs * pi_attempts,
        "skill_router": router_logical * router_attempts,
    }
    unsigned = {
        "schema": "pi-rule-stability-visible-optimize-plan-v1",
        "role": "one_shot_exploratory_repeated_seed_diagnostic",
        "config_path": str(Path(config_path).as_posix()),
        "config_sha256": config_fingerprint(cfg),
        "dataset": str(tasks[0].get("dataset") or ""),
        "split": "dev_optimize",
        "task_ids": [str(task["task_id"]) for task in tasks],
        "task_material_sha256": sha256_obj(tasks),
        "release_id": str(state.release_id),
        "system_prompt_sha256": sha256_text(state.system_prompt),
        "registry_sha256": state.registry.fingerprint(),
        "task_demos_sha256": sha256_obj(getattr(state, "task_demos", [])),
        "candidate_id": candidate.candidate_id,
        "candidate_prompt_sha256": candidate.content_sha256,
        "static_gate": static_gate_result,
        "rule_path": str(rule_path),
        "rule_sha256": sha256_text(rule),
        "task_model_id": str(task_model.get("model_id") or ""),
        "task_provider": "pi",
        "skill_routing_mode": str(skills.get("routing_mode", "lexical")),
        "seed": int(seed), "repeats": int(repeats), "arm_order": list(ARMS),
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
        "candidate_selection_or_confirmation": False,
        "causal_attribution_supported": False,
        "live_authorized": False,
        "explicit_user_approval_required_before_execution": True,
    }
    return {**unsigned, "plan_sha256": sha256_obj(unsigned)}


def verify_execution_plan(plan: dict, expected_sha256: str, *,
                          logical_budget: int, physical_budget: int) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    actual = sha256_obj(unsigned)
    if plan.get("plan_sha256") != actual or expected_sha256 != actual:
        raise ValueError("expected plan hash does not match the current stability plan")
    if logical_budget != plan["logical_request_cap"]:
        raise ValueError("logical budget must exactly match the frozen plan")
    if physical_budget != plan["physical_request_cap"]:
        raise ValueError("physical budget must exactly match the frozen plan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--rule-file", default="configs/task_input/strategy_target_outcome_v1.txt")
    parser.add_argument("--logical-request-budget", type=int, default=0)
    parser.add_argument("--physical-request-budget", type=int, default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-plan-sha256", default="")
    args = parser.parse_args()
    if args.logical_request_budget < 0 or args.physical_request_budget < 0:
        parser.error("request budgets must be nonnegative")
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    cfg = copy.deepcopy(load_config(args.config))
    cfg["runtime"].update({"provider": "pi", "input_mode": "task_input",
                           "task_input_max_chars": 30000})
    cfg.setdefault("evaluation", {}).update({"main_metric": "accuracy",
                                               "primary_axis": "accuracy",
                                               "sealed_pairing": False})
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg["tools"].get("active", []) if name != "academic.search"]
    state = load_state()
    rule_path = (ROOT / args.rule_file).resolve()
    if not rule_path.is_relative_to((ROOT / "configs" / "task_input").resolve()):
        raise RuntimeError("rule file must be under configs/task_input")
    rule = rule_path.read_text(encoding="utf-8").strip()
    candidate, trial, gate = build_stability_rule_candidate(
        cfg, state, rule_path, rule)
    selected = set(SELECTED_TASK_IDS)
    tasks = {str(t["task_id"]): t for t in inject_all(load_tasks(
        "pubmedqa", splits=["dev_optimize"])) if str(t["task_id"]) in selected}
    if set(tasks) != selected:
        raise RuntimeError("selected optimize tasks are missing")
    ordered_tasks = [tasks[task_id] for task_id in SELECTED_TASK_IDS]
    plan = build_execution_plan(
        cfg=cfg, state=state, tasks=ordered_tasks, config_path=args.config,
        rule_path=rule_path, rule=rule, candidate=candidate,
        static_gate_result=gate)
    if args.execute:
        verify_execution_plan(
            plan, args.expected_plan_sha256,
            logical_budget=args.logical_request_budget,
            physical_budget=args.physical_request_budget)
    else:
        if (args.logical_request_budget
                and args.logical_request_budget != plan["logical_request_cap"]):
            raise ValueError("logical budget must exactly match the frozen plan")
        if (args.physical_request_budget
                and args.physical_request_budget != plan["physical_request_cap"]):
            raise ValueError("physical budget must exactly match the frozen plan")
    plan_path = artifacts_path(
        "probe", "pi_rule_stability", "preflight-" + plan["plan_sha256"] + ".json")
    if plan_path.exists():
        from src.util import read_json
        if read_json(plan_path) != plan:
            raise RuntimeError("saved stability preflight differs from recomputed plan")
    else:
        write_json(plan_path, plan)
    if args.validate_only:
        print(json.dumps({
            "status": "validated_not_run", "plan": str(plan_path),
            "plan_sha256": plan["plan_sha256"],
            "task_count": len(ordered_tasks), "repeats": REPEATS,
            "logical_request_cap": plan["logical_request_cap"],
            "physical_request_cap": plan["physical_request_cap"],
            "model_calls": 0, "selection_gate_read": False,
            "final_holdout_read": False,
        }, ensure_ascii=False))
        return 0
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("stability execution requires sandbox RSI_HOME")
    task_model = (cfg.get("models") or {}).get("task") or {}
    key_env = str(task_model.get("api_key_env") or "").strip()
    if not key_env or not os.environ.get(key_env):
        raise RuntimeError("stability execution is missing configured task credentials")
    run_dir = artifacts_path(
        "probe", "pi_rule_stability",
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
    rows = []
    for repetition in range(REPEATS):
        for task_id in sorted(tasks):
            task = tasks[task_id]
            for arm, artifact in (("baseline", state), ("rule", trial)):
                out = TaskRunner(cfg, artifact).run(task)
                rows.append({"repeat": repetition + 1, "task_id": task_id,
                             "arm": arm, "answer_index": out.answer_index,
                             "correct": out.correct, "status": out.status,
                             "pi_session_path": out.pi_session_path})
    counts = {}
    for task_id in sorted(tasks):
        counts[task_id] = {}
        for arm in ("baseline", "rule"):
            observed = [r["answer_index"] for r in rows
                        if r["task_id"] == task_id and r["arm"] == arm]
            counts[task_id][arm] = {str(x): observed.count(x) for x in sorted(set(observed))}
    report = {"protocol": "pi-target-outcome-visible-optimize-stability-v3",
              "execution_plan": plan, "execution_plan_path": str(plan_path),
              "request_ledger": str(ledger_path),
              "request_accounting": summarize_ledger(ledger_path),
              "config_fingerprint": config_fingerprint(cfg),
              "release_id": state.release_id,
              "rule_path": str(rule_path),
              "rule_sha256": content_sha256(rule),
              "candidate_id": candidate.candidate_id,
              "candidate_prompt_sha256": candidate.content_sha256,
              "candidate_static_gate": gate,
              "parent_task_demos_preserved": (
                  trial.task_demos == state.task_demos),
              "task_ids": sorted(tasks), "repeats": 5,
              "answer_counts": counts, "rows": rows,
              "selection_gate_read": False}
    path = artifacts_path("probe", "pi_rule_stability",
                          "probe-" + uuid.uuid4().hex[:12] + ".json")
    write_json(path, report)
    print(json.dumps({"path": str(path), "answer_counts": counts},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
