"""Plan-bound visible-dev counterfactual for domain Skills and lexical routing."""

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
from src.contracts.types import content_sha256  # noqa: E402
from src.data.split import load_tasks  # noqa: E402
from src.eval.options import inject_all  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.runtime.provider import build_provider  # noqa: E402
from src.runtime.runner import TaskRunner  # noqa: E402
from src.runtime.skills import recall  # noqa: E402
from src.util import artifacts_path, sha256_obj, write_json  # noqa: E402


DOMAIN_SKILL = {"litqa2": "litqa-literature-synthesis",
                "protocolqa": "protocol-troubleshooting"}
DOMAIN_TAKE = {"litqa2": 7, "protocolqa": 10}
SCREEN_LIMIT = 60
PROTOCOL_VERSION = "task-input-domain-skill-pin-v3-plan-bound"
PLAN_SCHEMA = "task-input-domain-skill-pin-execution-v1"


def prepare_config(config_path: str) -> dict:
    cfg = copy.deepcopy(load_config(config_path))
    cfg.setdefault("runtime", {}).update({
        "provider": "http", "input_mode": "task_input",
        "task_input_max_chars": 30000,
    })
    cfg["runtime"].pop("pi_fake_response", None)
    cfg.setdefault("evaluation", {})["sealed_pairing"] = False
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg.get("tools", {}).get("active", [])
        if name != "academic.search"]
    skills = cfg.setdefault("skills", {})
    skills["routing_mode"] = "lexical"
    skills.pop("pinned_skill", None)
    return cfg


def build_protocol_identity(base_cfg: dict, release_id: str,
                            registry_fingerprint: str,
                            task_ids_by_dataset: dict[str, list[str]], *,
                            seed: int = 20260924) -> dict:
    pinned_config_sha256 = {}
    for skill_name in sorted(set(DOMAIN_SKILL.values())):
        pinned_cfg = copy.deepcopy(base_cfg)
        pinned_cfg.setdefault("skills", {})["pinned_skill"] = skill_name
        pinned_config_sha256[skill_name] = config_fingerprint(pinned_cfg)
    identity = {
        "schema": PROTOCOL_VERSION,
        "base_config_sha256": config_fingerprint(base_cfg),
        "pinned_config_sha256": pinned_config_sha256,
        "release_id": str(release_id),
        "registry_fingerprint": str(registry_fingerprint),
        "split": "dev_optimize",
        "cohort_rule": "first 60 sorted visible tasks; first 7/10 lexical misroutes",
        "domain_skills": dict(sorted(DOMAIN_SKILL.items())),
        "task_ids_by_dataset": {
            str(dataset): [str(task_id) for task_id in task_ids]
            for dataset, task_ids in sorted(task_ids_by_dataset.items())},
        "seed": int(seed),
        "seed_semantics": "provider construction label; stochastic control is not assumed",
        "arms": ["baseline", "same_state_repeat", "pinned_target_skill"],
        "arm_order": ["baseline", "repeat", "pinned"],
    }
    return {**identity, "protocol_sha256": sha256_obj(identity)}


def select_probe_tasks(state) -> tuple[dict[str, list[dict]], dict[str, list[str]], dict[str, list[str]]]:
    """Select fixed visible optimize tasks using the exact lexical router."""
    screening: dict[str, list[dict]] = {}
    selected: dict[str, list[dict]] = {}
    screening_ids: dict[str, list[str]] = {}
    for dataset, target in DOMAIN_SKILL.items():
        tasks = inject_all(sorted(
            load_tasks(dataset, splits=["dev_optimize"]),
            key=lambda task: str(task["task_id"]))[:SCREEN_LIMIT])
        if not tasks:
            raise ValueError(f"no visible dev_optimize rows for {dataset}")
        misrouted = []
        for task in tasks:
            query = str(task["question"]) + " " + " ".join(task["options"])
            hit = recall(query, state.registry, min_score=2)
            if hit is None or hit["name"] != target:
                misrouted.append(task)
        screening[dataset] = tasks
        selected[dataset] = misrouted[:DOMAIN_TAKE[dataset]]
        screening_ids[dataset] = [str(task["task_id"]) for task in tasks]
    if not any(selected.values()):
        raise ValueError("the frozen lexical screen selected no misrouted tasks")
    return screening, selected, screening_ids


def build_execution_plan(*, cfg: dict, state, selected: dict[str, list[dict]],
                         screening: dict[str, list[dict]],
                         screening_ids: dict[str, list[str]], seed: int,
                         logical_request_budget: int = 0,
                         physical_request_budget: int = 0) -> dict:
    """Freeze exact task scope, intervention arms, and conservative HTTP bounds."""
    provider = str((cfg.get("runtime") or {}).get("provider") or "").lower()
    if provider != "http":
        raise ValueError("domain Skill routing plan requires the HTTP provider")
    selected_tasks = [task for dataset in DOMAIN_SKILL for task in selected[dataset]]
    task_ids_by_dataset = {
        dataset: [str(task["task_id"]) for task in selected[dataset]]
        for dataset in DOMAIN_SKILL}
    for dataset in DOMAIN_SKILL:
        if (not set(task_ids_by_dataset[dataset]) <= set(screening_ids[dataset])
                or len(task_ids_by_dataset[dataset]) > DOMAIN_TAKE[dataset]):
            raise ValueError("selected task IDs exceed the frozen lexical screen")
    if len({str(task["task_id"]) for task in selected_tasks}) != len(selected_tasks):
        raise ValueError("selected domain routing task IDs must be unique")
    task_model = (cfg.get("models") or {}).get("task") or {}
    http_attempts = int(task_model.get("max_retries", 6))
    if http_attempts < 1:
        raise ValueError("HTTP attempt limit must be positive")
    runtime = cfg.get("runtime") or {}
    max_model_calls = int(runtime.get("max_model_calls", 12))
    if max_model_calls < 1:
        raise ValueError("runtime.max_model_calls must be positive")
    if str((cfg.get("skills") or {}).get("routing_mode") or "lexical") != "lexical":
        raise ValueError("the baseline arm must use lexical Skill routing")

    # Baseline and repeat each have one answer and may route locally; the pinned
    # arm has one answer and skips the router. Retrieval and academic search are off.
    answer_operations = len(selected_tasks) * 3
    logical_by_component = {
        "task_answer": answer_operations,
        "skill_router": 0,  # lexical Skill selection is local, not a model call
    }
    physical_by_component = {
        component: count * http_attempts
        for component, count in logical_by_component.items()}
    logical_upper = sum(logical_by_component.values())
    physical_upper = sum(physical_by_component.values())
    logical_cap = int(logical_request_budget or logical_upper)
    physical_cap = int(physical_request_budget or physical_upper)
    if logical_cap < logical_upper:
        raise ValueError("logical request cap is below the frozen operation ceiling")
    if physical_cap < physical_upper:
        raise ValueError("physical request cap is below the frozen HTTP attempt ceiling")

    material_rows = []
    for dataset in DOMAIN_SKILL:
        if [str(task["task_id"]) for task in screening[dataset]] != screening_ids[dataset]:
            raise ValueError("screening task IDs changed while building the plan")
        for task in screening[dataset]:
            task_id = str(task["task_id"])
            material_rows.append({
                "dataset": dataset, "task_id": task_id,
                "question": task.get("question"),
                "evidence": task.get("evidence") or task.get("context") or "",
                "options": task.get("options"),
                "correct_index": task.get("correct_index"),
            })
    identity = build_protocol_identity(
        cfg, state.release_id, state.registry.fingerprint(), task_ids_by_dataset,
        seed=seed)
    unsigned = {
        "schema": PLAN_SCHEMA,
        "protocol": PROTOCOL_VERSION,
        "protocol_identity": identity,
        "source_config_fingerprint": config_fingerprint(cfg),
        "parent_prompt_sha256": content_sha256(str(state.system_prompt)),
        "parent_task_demos_sha256": sha256_obj(
            getattr(state, "task_demos", []) or []),
        "screening_task_ids_by_dataset": screening_ids,
        "task_material_sha256": sha256_obj(material_rows),
        "selected_task_ids_by_dataset": task_ids_by_dataset,
        "selected_task_count": len(selected_tasks),
        "task_model_id": str(task_model.get("model_id") or ""),
        "provider": provider,
        "http_attempts_per_operation": http_attempts,
        "max_model_calls_per_task_runner": max_model_calls,
        "arms": ["baseline", "same_state_repeat", "pinned_target_skill"],
        "logical_operations_by_component": logical_by_component,
        "physical_attempts_by_component": physical_by_component,
        "logical_request_upper_bound": logical_upper,
        "logical_request_cap": logical_cap,
        "physical_request_upper_bound": physical_upper,
        "physical_request_cap": physical_cap,
        "request_ledger_required": True,
        "retrieval_used": False,
        "selection_gate_read": False,
        "official_validation_read": False,
        "official_test_read": False,
        "independent_confirmation_eligible": False,
        "candidate_selection": False,
        "candidate_promotion": False,
        "model_calls_during_preflight": 0,
        "live_authorized": False,
        "explicit_user_approval_required_before_execution": True,
    }
    return {**unsigned, "plan_sha256": sha256_obj(unsigned)}


def verify_execution_plan(plan: dict, expected_plan_sha256: str | None, *,
                          validate_only: bool) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if sha256_obj(unsigned) != plan.get("plan_sha256"):
        raise ValueError("routing execution plan self-hash is invalid")
    if validate_only:
        return
    if not re.fullmatch(r"[0-9a-f]{64}", expected_plan_sha256 or ""):
        raise ValueError("execution requires --expected-plan-sha256")
    if expected_plan_sha256 != plan["plan_sha256"]:
        raise ValueError("expected routing plan SHA-256 does not match the recomputed plan")


def require_task_credentials(cfg: dict) -> None:
    model_cfg = (cfg.get("models") or {}).get("task") or {}
    if str(model_cfg.get("provider") or "").lower() != "http":
        raise ValueError("domain routing probe requires an HTTP task model")
    key_env = str(model_cfg.get("api_key_env") or "").strip()
    if not key_env or not os.environ.get(key_env):
        raise RuntimeError("missing configured task-model credential: "
                           + (key_env or "models.task.api_key_env"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--logical-request-budget", type=int, default=0)
    parser.add_argument("--physical-request-budget", type=int, default=0)
    parser.add_argument("--expected-plan-sha256", default="")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.logical_request_budget < 0 or args.physical_request_budget < 0:
        raise ValueError("request budgets must be nonnegative")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}",
                                         args.expected_plan_sha256 or ""):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("domain routing probe requires an isolated RSI_HOME")

    cfg = prepare_config(args.config)
    state = load_state()
    screening, selected, screening_ids = select_probe_tasks(state)
    plan = build_execution_plan(
        cfg=cfg, state=state, selected=selected, screening=screening,
        screening_ids=screening_ids,
        seed=args.seed, logical_request_budget=args.logical_request_budget,
        physical_request_budget=args.physical_request_budget)
    plan_path = artifacts_path(
        "probe", "skill_routing", "preflight-" + plan["plan_sha256"] + ".json")
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("saved domain routing plan differs from recomputed plan")
    else:
        write_json(plan_path, plan)
    verify_execution_plan(
        plan, args.expected_plan_sha256, validate_only=args.validate_only)
    if args.validate_only:
        print(json.dumps({
            "status": "validated_not_run", "plan": str(plan_path),
            "plan_sha256": plan["plan_sha256"],
            "selected_task_count": plan["selected_task_count"],
            "selected_task_ids_by_dataset": plan["selected_task_ids_by_dataset"],
            "logical_request_cap": plan["logical_request_cap"],
            "physical_request_cap": plan["physical_request_cap"],
            "model_calls": 0, "selection_gate_read": False,
            "official_validation_read": False, "official_test_read": False,
        }, ensure_ascii=False))
        return 0

    require_task_credentials(cfg)
    ledger_path = artifacts_path(
        "probe", "skill_routing", plan["plan_sha256"], "call-ledger.jsonl")
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(plan["plan_sha256"])
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path)
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(plan["logical_request_cap"])
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(plan["physical_request_cap"])
    for component, count in plan["logical_operations_by_component"].items():
        os.environ[f"RSI_CALL_LEDGER_MAX_COMPONENT_{component.upper()}"] = str(count)

    provider = build_provider(cfg, seed=args.seed)
    if hasattr(provider, "cache"):
        provider.cache = False
    rows = []
    for dataset, target in DOMAIN_SKILL.items():
        selected_ids = set(plan["selected_task_ids_by_dataset"][dataset])
        tasks = [task for task in screening[dataset]
                 if str(task["task_id"]) in selected_ids]
        for task in tasks:
            pinned_cfg = copy.deepcopy(cfg)
            pinned_cfg.setdefault("skills", {})["pinned_skill"] = target
            baseline = TaskRunner(cfg, state, provider=provider).run(task)
            repeated = TaskRunner(cfg, state, provider=provider).run(task)
            pinned = TaskRunner(pinned_cfg, state, provider=provider).run(task)

            def observation(outcome):
                return {"answer_index": outcome.answer_index,
                        "status": outcome.status, "correct": outcome.correct,
                        "skill": outcome.skill_recalled, "error": outcome.error,
                        "searched": any(e.get("tool_name") == "academic.search"
                                         for e in outcome.events)}

            rows.append({
                "dataset": dataset, "task_id": str(task["task_id"]),
                "target_skill": target,
                "material_sha256": sha256_obj(
                    task.get("evidence") or task.get("context") or ""),
                "baseline": observation(baseline),
                "repeat": observation(repeated),
                "pinned": observation(pinned),
                "aa_stable": ((baseline.answer_index, baseline.status)
                              == (repeated.answer_index, repeated.status)),
                "route_effect": ((baseline.answer_index, baseline.status)
                                 != (pinned.answer_index, pinned.status)),
            })
    report = {
        "protocol": PROTOCOL_VERSION,
        "execution_plan": plan,
        "execution_plan_path": str(plan_path),
        "execution_plan_sha256": plan["plan_sha256"],
        "request_ledger_path": str(ledger_path),
        "request_ledger_summary": summarize_ledger(ledger_path),
        "release_id": state.release_id,
        "n": len(rows),
        "aa_stable": sum(row["aa_stable"] for row in rows),
        "route_effect": sum(row["route_effect"] for row in rows),
        "selection_gate_read": False,
        "official_validation_read": False,
        "official_test_read": False,
        "retrieval_used": False,
        "candidate_selection": False,
        "independent_confirmation_eligible": False,
        "attribution_is_causal": False,
        "rows": rows,
    }
    out = artifacts_path(
        "probe", "skill_routing", "report-" + plan["plan_sha256"] + ".json")
    if out.exists():
        raise FileExistsError("a report already exists for this frozen routing plan")
    write_json(out, report)
    print(json.dumps({
        "path": str(out), "plan_sha256": plan["plan_sha256"],
        "n": report["n"], "aa_stable": report["aa_stable"],
        "route_effect": report["route_effect"],
        "request_ledger_summary": report["request_ledger_summary"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
