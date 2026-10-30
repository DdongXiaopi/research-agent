"""Run bounded upstream MIPROv2 on a frozen PubMedQA train cohort.

This is an exploratory candidate-generation stage. It never evaluates the
external discovery partition; its report can be consumed by
``evaluate_pubmedqa_prompt_discovery.py`` only after internal confirmation.
"""

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

from scripts.evaluate_pubmedqa_prompt_discovery import (  # noqa: E402
    run_prompt_grounding_review,
)
from scripts.run_pubmedqa_v2_joint_search import (  # noqa: E402
    validate_baseline_task_rows,
)
from scripts.pubmedqa_threeway_protocol import verify_dataset_snapshot  # noqa: E402
from src.config import config_fingerprint, load_config  # noqa: E402
from src.contracts.types import Candidate, content_id, content_sha256  # noqa: E402
from src.data.split import load_tasks  # noqa: E402
from src.eval.task_rows import validate_task_answer_row  # noqa: E402
from src.evolution.gepa_search import visible_examples  # noqa: E402
from src.evolution.gepa_search import task_label_contract_conflict  # noqa: E402
from src.evolution.mipro_task_input import (  # noqa: E402
    MIPRO_SELECTION_POLICY_VERSION, compile_mipro_task_input, extract_task_program,
)
from src.evolution.task_contracts import task_contract_for_examples  # noqa: E402
from src.evolution.task_demo_search import split_demo_validation  # noqa: E402
from src.evolution.task_program import (changed_task_program_components,
                                       serialize_task_program)  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.runtime.provider import build_provider  # noqa: E402
from src.runtime.runner import ArtifactState, TaskRunner  # noqa: E402
from src.util import (  # noqa: E402
    artifacts_path, read_json, read_jsonl, sha256_file, sha256_obj, write_json,
)

MIPRO_EXECUTION_PLAN_SCHEMA = "pubmedqa-v2-mipro-task-input-execution-v1"


def validate_source(source_path: Path, source: dict) -> tuple[list[str], list[str]]:
    verify_dataset_snapshot(source)
    if source.get("protocol") != "pubmedqa-threeway-v2-skill-routing-baseline-v2":
        raise ValueError("MIPRO requires a v2 train-plus-discovery baseline report")
    if source.get("skill_routing_mode") not in {"lexical", "model"}:
        raise ValueError("source baseline has an invalid Skill routing mode")
    train_ids = [str(value) for value in source.get("train_ids") or []]
    discovery_ids = [str(value) for value in source.get("discovery_ids") or []]
    confirmation_ids = [str(value) for value in source.get("confirmation_ids") or []]
    partitions = [train_ids, discovery_ids, confirmation_ids]
    if any(not ids or len(ids) != len(set(ids)) for ids in partitions):
        raise ValueError("frozen train/discovery/confirmation IDs must be non-empty and unique")
    if (set(train_ids) & set(discovery_ids)
            or set(train_ids) & set(confirmation_ids)
            or set(discovery_ids) & set(confirmation_ids)):
        raise ValueError("frozen train/discovery/confirmation partitions overlap")
    if source.get("confirmation_labels_accessed") is not False:
        raise ValueError("baseline report does not attest external-confirmation isolation")
    if source.get("train_answers_accessed") is not True:
        raise ValueError("MIPRO requires train-plus-discovery baseline rows")
    expected = set(train_ids) | set(discovery_ids)
    evaluated = [str(value) for value in source.get("evaluated_ids") or []]
    if len(evaluated) != len(set(evaluated)) or set(evaluated) != expected:
        raise ValueError("baseline evaluated IDs differ from frozen train+discovery scope")
    runs = source.get("baseline_runs") or []
    seeds = [int(run["seed"]) for run in runs]
    if (len(runs) < 3 or len(seeds) != len(set(seeds))
            or seeds != [int(value) for value in source.get("execution_labels") or []]):
        raise ValueError("MIPRO requires three or more unique paired baseline seeds")
    for item in runs:
        if not item.get("per_task_includes_train"):
            raise ValueError("paired baseline is missing per-task train rows")
        if int(item.get("run_task_count", -1)) != len(expected):
            raise ValueError("baseline run task count differs from train+discovery scope")
    return train_ids, discovery_ids


def configure(source: dict, config_path: str, train_ids: list[str],
              proposer_calls: int) -> dict:
    cfg = copy.deepcopy(load_config(config_path))
    cfg.setdefault("runtime", {}).update({
        "provider": "pi", "input_mode": "task_input",
        "task_input_max_chars": 30000, "pi_request_cooldown_seconds": 6,
    })
    cfg["runtime"].pop("pi_fake_response", None)
    cfg.setdefault("evaluation", {}).update({
        "insufficient_option": False, "main_metric": "accuracy",
        "primary_axis": "accuracy", "sealed_pairing": False,
    })
    cfg.setdefault("skills", {})["routing_mode"] = str(source["skill_routing_mode"])
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg.get("tools", {}).get("active", [])
        if name != "academic.search"]
    evolution = cfg.setdefault("evolution", {})
    evolution["allowed_artifacts"] = ["task_program"]
    evolution.setdefault("gepa", {}).update({
        "visible_task_ids": train_ids, "stratify_by_gold_label": True,
    })
    evolution.setdefault("fewshot", {}).update({"enabled": True, "k": 3})
    evolution.setdefault("mipro", {}).update({"max_proposer_calls": proposer_calls})
    reflection = cfg.setdefault("models", {}).setdefault("reflection", {})
    # The shared MIPRO adapter deliberately caps SDK retries at two so its
    # physical-attempt reservation remains finite and reviewable.
    reflection["max_retries"] = max(1, min(2, int(reflection.get("max_retries", 2))))
    return cfg


def build_proposal_task_contract(examples: list[dict]) -> tuple[str | None, dict]:
    """Build the frozen contract context used only by MIPRO's data-aware proposer."""
    task_contract = task_contract_for_examples(examples)
    if task_contract is None:
        return None, {
            "task_contract_id": None,
            "task_contract_sha256": None,
            "context_sha256": None,
            "injected_into_data_aware_proposer": False,
            "context_format": "frozen-task-contract-plus-train-summary-v1",
            "train_summary_retained": True,
            "student_task_prompt_modified": False,
        }
    context = (
        "Frozen task decision contract "
        f"(id={task_contract.contract_id}; sha256={task_contract.sha256}):\n"
        f"{task_contract.text}")
    return context, {
        "task_contract_id": task_contract.contract_id,
        "task_contract_sha256": task_contract.sha256,
        "context_sha256": content_sha256(context.strip()),
        "injected_into_data_aware_proposer": True,
        "context_format": "frozen-task-contract-plus-train-summary-v1",
        "train_summary_retained": True,
        "student_task_prompt_modified": False,
    }


def build_execution_plan(*, source_path: Path, source: dict, cfg: dict, state,
                         dataset_hashes: tuple[str, str], train_ids: list[str],
                         discovery_ids: list[str], train_task_ids: set[str],
                         search_ids: set[str], internal_ids: set[str],
                         task_by_id: dict[str, dict], baseline_rows: dict,
                         proposal_task_contract: dict, args) -> dict:
    """Freeze the exact nested MIPRO search, internal scope, and request bounds."""
    runtime = cfg.get("runtime") or {}
    models = cfg.get("models") or {}
    task_model = models.get("task") or {}
    reflection_model = models.get("reflection") or {}
    task_evaluations = int(args.task_evaluation_budget)
    proposer_calls = int(args.proposer_call_budget)
    grounding_calls = int(args.grounding_call_budget)
    if min(task_evaluations, proposer_calls, grounding_calls) < 1:
        raise ValueError("MIPRO execution-plan budgets must be positive")

    model_routing = (str((cfg.get("skills") or {}).get("routing_mode") or "lexical")
                     == "model" and bool(state.registry.skills))
    router_operations = task_evaluations if model_routing else 0
    proposer_retries = int(reflection_model.get("max_retries", 2))
    if not 0 <= proposer_retries <= 2:
        raise ValueError("MIPRO proposal retries exceed the locked adapter limit")
    task_http_attempts = int(task_model.get("max_retries", 6))
    grounding_http_attempts = int(reflection_model.get("max_retries", 2))
    if task_http_attempts < 1 or grounding_http_attempts < 1:
        raise ValueError("MIPRO HTTP attempt limits must be positive")
    pi_provider_retries = max(0, min(4, int(runtime.get("pi_provider_max_retries", 4))))
    pi_bridge_attempts = max(1, min(3, int(runtime.get("pi_timeout_max_attempts", 2))))
    pi_attempts_per_answer = (pi_provider_retries + 1) * pi_bridge_attempts

    logical_by_component = {
        "task_answer": task_evaluations,
        "skill_router": router_operations,
        "mipro_proposer": proposer_calls,
        "grounding": grounding_calls,
    }
    physical_by_component = {
        "task_answer": task_evaluations * pi_attempts_per_answer,
        "skill_router": router_operations * task_http_attempts,
        "mipro_proposer": proposer_calls * (proposer_retries + 1),
        "grounding": grounding_calls * grounding_http_attempts,
    }
    logical_upper = sum(logical_by_component.values())
    physical_upper = sum(physical_by_component.values())
    logical_cap = int(args.logical_request_budget or logical_upper)
    physical_cap = int(args.physical_request_budget or physical_upper)
    if logical_cap < logical_upper:
        raise ValueError("logical request cap is below the frozen MIPRO operation ceiling")
    if physical_cap < physical_upper:
        raise ValueError("physical request cap is below the frozen MIPRO attempt ceiling")

    baseline_scoring_rows = {
        str(seed): [{key: row.get(key) for key in (
            "task_id", "status", "answer_index", "gold_index", "correct", "gold_label")}
                    for _, row in sorted(rows.items())]
        for seed, rows in sorted(baseline_rows.items())
    }
    task_material = [{
        "task_id": task_id,
        "question": row.get("question"),
        "evidence": row.get("evidence") or row.get("context") or "",
        "options": row.get("options"),
        "correct_index": row.get("correct_index"),
    } for task_id, row in sorted(task_by_id.items())]
    reflection_max_retries = int(reflection_model.get("max_retries", 2))
    unsigned = {
        "schema": MIPRO_EXECUTION_PLAN_SCHEMA,
        "role": "exploratory_mipro_search_with_nested_internal_candidate_check",
        "source_report_sha256": sha256_file(source_path),
        "config_sha256": config_fingerprint(cfg),
        "dataset": str(source.get("dataset") or ""),
        "dataset_manifest_sha256": dataset_hashes[0],
        "dataset_normalized_sha256": dataset_hashes[1],
        "source_train_ids": sorted(str(value) for value in train_ids),
        "source_discovery_ids": sorted(str(value) for value in discovery_ids),
        "mipro_train_ids": sorted(train_task_ids),
        "mipro_search_validation_ids": sorted(search_ids),
        "internal_candidate_ids": sorted(internal_ids),
        "task_material_sha256": sha256_obj(task_material),
        "baseline_train_outcomes_sha256": sha256_obj(baseline_scoring_rows),
        "parent_release_id": str(state.release_id),
        "parent_prompt_sha256": content_sha256(str(state.system_prompt)),
        "parent_registry_sha256": state.registry.fingerprint(),
        "parent_task_demos_sha256": sha256_obj(getattr(state, "task_demos", []) or []),
        "proposal_task_contract": proposal_task_contract,
        "task_model_id": str(task_model.get("model_id") or ""),
        "proposer_model_id": str(reflection_model.get("model_id") or ""),
        "skill_routing_mode": str((cfg.get("skills") or {}).get("routing_mode") or "lexical"),
        "execution_labels": [int(value) for value in source["execution_labels"]],
        "mipro": {
            "seed": int(args.seed),
            "num_candidates": int(args.num_candidates),
            "num_trials": int(args.num_trials),
            "max_labeled_demos": int(args.max_labeled_demos),
            "task_evaluation_budget": task_evaluations,
            "minimum_full_validation_evaluations": (
                (int(args.num_trials) + 1) * len(search_ids)),
            "internal_candidate_evaluation_upper_bound": (
                len(internal_ids) * len(source["execution_labels"])),
            "proposer_call_budget": proposer_calls,
            "grounding_call_budget": grounding_calls,
            "proposer_max_retries_after_first": proposer_retries,
        },
        "request_budget_version": "bounded-physical-attempt-budget-v1",
        "logical_operations_by_component": logical_by_component,
        "physical_attempts_by_component": physical_by_component,
        "logical_request_upper_bound": logical_upper,
        "logical_request_cap": logical_cap,
        "physical_request_upper_bound": physical_upper,
        "physical_request_cap": physical_cap,
        "pi_provider_retries_after_first": pi_provider_retries,
        "pi_timeout_subprocess_attempts": pi_bridge_attempts,
        "task_router_http_attempts_per_operation": task_http_attempts,
        "grounding_http_attempts_per_operation": grounding_http_attempts,
        "request_ledger_required": True,
        "pi_physical_reservations_required": True,
        "retrieval_used": False,
        "new_external_discovery_evaluations": False,
        "source_confirmation_accessed": False,
        "selection_gate_read": False,
        "official_validation_read": False,
        "official_test_read": False,
        "independent_confirmation_eligible": False,
        "model_calls_during_preflight": 0,
        "live_authorized": False,
        "explicit_user_approval_required_before_execution": True,
    }
    return {**unsigned, "plan_sha256": sha256_obj(unsigned)}


def verify_execution_plan(plan: dict, expected_sha256: str | None, *,
                          validate_only: bool) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != sha256_obj(unsigned):
        raise ValueError("MIPRO execution plan self-hash is invalid")
    if validate_only:
        return
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or ""):
        raise ValueError("MIPRO execution requires --expected-plan-sha256")
    if expected_sha256 != plan["plan_sha256"]:
        raise ValueError("expected MIPRO plan hash does not match the recomputed plan")


def require_configured_model_credentials(cfg: dict) -> None:
    """Fail before any measured verifier/task call when configured keys are absent."""
    import os
    missing = []
    for section in ("task", "reflection"):
        model_cfg = (cfg.get("models") or {}).get(section) or {}
        if str(model_cfg.get("provider") or "").lower() != "http":
            continue
        key_env = str(model_cfg.get("api_key_env") or "").strip()
        if not key_env or not os.environ.get(key_env):
            missing.append(key_env or f"models.{section}.api_key_env")
    if missing:
        raise RuntimeError(
            "MIPRO search requires configured HTTP model credentials; missing: "
            + ", ".join(missing))


def task_label_grounding_gate(prompt: str, examples: list[dict]) -> dict:
    """Reject prompt instructions that emit labels absent from visible task options."""
    conflict = task_label_contract_conflict(prompt, examples)
    return {
        "passed": conflict is None,
        "reason": conflict or "all explicit output labels are in task options",
        "scope": "optimizer-train plus nested search/internal task options",
    }


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    answered = sum(int(row.get("answer_index", -1)) >= 0 for row in rows)
    correct = sum(bool(row.get("correct")) for row in rows)
    labels = sorted({str(row["gold_label"]).strip().casefold() for row in rows})
    by_label = {}
    for label in labels:
        subset = [row for row in rows if str(row["gold_label"]).strip().casefold() == label]
        label_answered = sum(int(row.get("answer_index", -1)) >= 0 for row in subset)
        label_correct = sum(bool(row.get("correct")) for row in subset)
        by_label[label] = {
            "n": len(subset), "accuracy": label_correct / len(subset),
            "coverage": label_answered / len(subset),
            "answered_precision": label_correct / label_answered if label_answered else 0.0,
        }
    return {
        "n": n, "accuracy": correct / n if n else 0.0,
        "coverage": answered / n if n else 0.0,
        "answered_precision": correct / answered if answered else 0.0,
        "per_gold_label": by_label,
    }


def nonregresses(candidate: dict, baseline: dict) -> bool:
    for metric_name in ("accuracy", "coverage", "answered_precision"):
        if candidate[metric_name] + 1e-12 < baseline[metric_name]:
            return False
    for label, base in baseline["per_gold_label"].items():
        current = candidate["per_gold_label"].get(label)
        if current is None:
            return False
        for metric_name in ("accuracy", "coverage", "answered_precision"):
            if current[metric_name] + 1e-12 < base[metric_name]:
                return False
    return True


def _candidate(parent: str, prompt: str, content: str) -> Candidate:
    return Candidate(
        candidate_id=content_id("cand", parent, "task_program", content),
        parent_release_id=parent, artifact_type="task_program",
        artifact_id="task_program", diff="MIPROv2 selected prompt × labeled train demos",
        rationale="Bounded upstream DSPy MIPROv2 search over nested train validation.",
        status="draft", new_content=content, content_sha256=content_sha256(content),
        generator={"backend": "dspy-mipro-v2", "is_mipro": True},
    )


def check_fr201_mipro_program(content: str, state) -> dict:
    """Only a Prompt-only task-program may become a mainline candidate."""
    changed = changed_task_program_components(content, state)
    return {
        "passed": changed == ["system_prompt"],
        "changed_components": changed,
        "reason": (None if changed == ["system_prompt"] else
                   "fr201_candidate_category_not_system_prompt"),
    }


def _baseline_train_rows(source: dict, train_ids: set[str]) -> dict[int, dict[str, dict]]:
    result = {}
    for item in source["baseline_runs"]:
        rows = read_jsonl(artifacts_path("runs", str(item["run_id"]), "per_task.jsonl"))
        selected = {}
        for row in rows:
            task_id = str(row.get("task_id") or "")
            if task_id in train_ids:
                if task_id in selected:
                    raise ValueError(f"duplicate baseline train row: {task_id}")
                selected[task_id] = row
        if not train_ids <= set(selected):
            raise ValueError(f"baseline run {item['run_id']} omits train rows")
        result[int(item["seed"])] = selected
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True,
                        help="frozen v2 train-plus-discovery baseline report")
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--seed", type=int, default=202610051)
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--num-trials", type=int, default=4)
    parser.add_argument("--max-labeled-demos", type=int, default=3)
    parser.add_argument("--task-evaluation-budget", type=int, default=120)
    parser.add_argument("--proposer-call-budget", type=int, default=24)
    parser.add_argument("--grounding-call-budget", type=int, default=8)
    parser.add_argument("--logical-request-budget", type=int, default=0,
                        help="defaults to the frozen component-operation ceiling")
    parser.add_argument("--physical-request-budget", type=int, default=0,
                        help="defaults to the conservative HTTP-attempt ceiling")
    parser.add_argument("--expected-plan-sha256", default="",
                        help="required with --execute; must match a prior zero-call plan")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true",
                      help="write the complete zero-call execution plan")
    mode.add_argument("--execute", action="store_true",
                      help="execute only the plan identified by --expected-plan-sha256")
    args = parser.parse_args()
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256 or ""):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("MIPRO search requires an explicit isolated RSI_HOME")
    if min(args.num_candidates, args.num_trials, args.task_evaluation_budget,
           args.proposer_call_budget, args.grounding_call_budget) <= 0:
        raise ValueError("all MIPRO candidate, trial, and call budgets must be positive")
    if args.logical_request_budget < 0 or args.physical_request_budget < 0:
        raise ValueError("request budgets must be nonnegative")
    if args.num_candidates < 2:
        raise ValueError("MIPRO requires at least two candidate instructions/demo sets")
    if not 1 <= args.max_labeled_demos <= 3:
        raise ValueError("MIPRO labeled-demo count must be between one and three")
    source_path = Path(args.source_report).resolve()
    source = read_json(source_path)
    if source.get("synthetic_preflight_only") and not args.validate_only:
        raise ValueError("synthetic preflight fixture cannot be used for MIPRO search")
    train_ids, discovery_ids = validate_source(source_path, source)
    validate_baseline_task_rows(source, train_ids)
    cfg = configure(source, args.config, train_ids, args.proposer_call_budget)
    state = load_state(str(source["release_id"]))
    if state.release_id != str(source["release_id"]):
        raise ValueError("loaded release differs from frozen source report")
    outer_train, outer_validation = visible_examples(
        cfg, [str(source["dataset"])], len(train_ids))
    search_validation, internal_confirmation = split_demo_validation(outer_validation)
    train_task_ids = {str(row["task_id"]) for row in outer_train}
    search_ids = {str(row["task_id"]) for row in search_validation}
    internal_ids = {str(row["task_id"]) for row in internal_confirmation}
    if (train_task_ids | search_ids | internal_ids) != set(train_ids):
        raise ValueError("nested MIPRO partitions do not cover frozen train IDs")
    if train_task_ids & (search_ids | internal_ids) or search_ids & internal_ids:
        raise ValueError("nested MIPRO train/search/confirmation IDs overlap")
    task_rows = [row for row in load_tasks(
        str(source["dataset"]), splits=["dev_optimize"])
        if str(row["task_id"]) in train_task_ids | search_ids | internal_ids]
    task_by_id = {str(row["task_id"]): row for row in task_rows}
    if len(task_by_id) != len(task_rows):
        raise ValueError("frozen MIPRO task IDs are duplicated in dev_optimize")
    if set(task_by_id) != train_task_ids | search_ids | internal_ids:
        raise ValueError("frozen train cohort is missing rows in dev_optimize")
    baseline_rows = _baseline_train_rows(source, train_task_ids)
    task_contract_context, proposal_task_contract = build_proposal_task_contract(outer_train)
    confirmation_task_evaluations = len(internal_ids) * len(source["execution_labels"])
    planned_mipro_full_eval = (args.num_trials + 1) * len(search_ids)
    if planned_mipro_full_eval + confirmation_task_evaluations > args.task_evaluation_budget:
        raise ValueError(
            "total task-evaluation budget cannot cover MIPRO full trials plus "
            "three-seed internal confirmation")
    dataset_hashes = verify_dataset_snapshot(source)
    plan = build_execution_plan(
        source_path=source_path, source=source, cfg=cfg, state=state,
        dataset_hashes=dataset_hashes, train_ids=train_ids,
        discovery_ids=discovery_ids, train_task_ids=train_task_ids,
        search_ids=search_ids, internal_ids=internal_ids,
        task_by_id=task_by_id, baseline_rows=baseline_rows,
        proposal_task_contract=proposal_task_contract, args=args)
    plan_path = artifacts_path(
        "probe", "pubmedqa_v2_skill_routing",
        "mipro-preflight-" + plan["plan_sha256"] + ".json")
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError("saved MIPRO plan differs from the recomputed plan")
    else:
        write_json(plan_path, plan)
    verify_execution_plan(
        plan, args.expected_plan_sha256, validate_only=args.validate_only)
    if args.validate_only:
        print(json.dumps({
            "status": "validated_not_run", "plan": str(plan_path),
            "plan_sha256": plan["plan_sha256"],
            "release_id": state.release_id, "dataset": source["dataset"],
            "skill_routing_mode": source["skill_routing_mode"],
            "train_ids": len(train_ids), "mipro_train_ids": len(train_task_ids),
            "search_validation_ids": len(search_ids),
            "internal_confirmation_ids": len(internal_ids),
            "external_discovery_ids": len(discovery_ids),
            "task_evaluation_budget": args.task_evaluation_budget,
            "planned_mipro_full_eval_minimum": planned_mipro_full_eval,
            "planned_internal_confirmation_evaluations": confirmation_task_evaluations,
            "planned_total_task_evaluations_minimum": (
                planned_mipro_full_eval + confirmation_task_evaluations),
            "proposer_call_budget": args.proposer_call_budget,
            "grounding_call_budget": args.grounding_call_budget,
            "logical_request_cap": plan["logical_request_cap"],
            "physical_request_cap": plan["physical_request_cap"],
            "api_key_configured": bool(os.environ.get(
                str((cfg.get("models", {}).get("reflection") or {}).get("api_key_env") or ""))),
            "external_discovery_accessed": False,
            "external_confirmation_accessed": False,
            "model_calls": 0, "config_sha256": config_fingerprint(cfg),
        }, ensure_ascii=False))
        return 0
    require_configured_model_credentials(cfg)

    study_spec = {
        "protocol": "pubmedqa-threeway-v2-mipro-task-input-v5-fr201-prompt-only-plan-bound-selection",
        "plan_sha256": plan["plan_sha256"],
        "candidate_selection_policy": MIPRO_SELECTION_POLICY_VERSION,
        "source_report_sha256": plan["source_report_sha256"],
        "config_sha256": plan["config_sha256"],
        "seed": plan["mipro"]["seed"],
        "num_candidates": args.num_candidates,
        "num_trials": args.num_trials,
        "max_labeled_demos": args.max_labeled_demos,
        "task_evaluation_budget": args.task_evaluation_budget,
        "proposer_call_budget": args.proposer_call_budget,
        "grounding_call_budget": args.grounding_call_budget,
    }
    study_id = plan["plan_sha256"][:16]
    study_dir = artifacts_path("probe", "pubmedqa_v2_skill_routing",
                               "mipro-" + study_id)
    study_dir.mkdir(parents=True, exist_ok=True)
    compiled_path = study_dir / "compiled-candidate.json"
    confirmation_checkpoint_path = study_dir / "internal-confirmation.jsonl"
    ledger_path = study_dir / "call-ledger.jsonl"
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(plan["plan_sha256"])
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path)
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(plan["logical_request_cap"])
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(plan["physical_request_cap"])
    os.environ["RSI_CALL_LEDGER_MAX_COMPONENT_TASK_ANSWER"] = str(
        plan["logical_operations_by_component"]["task_answer"])
    os.environ["RSI_CALL_LEDGER_MAX_COMPONENT_SKILL_ROUTER"] = str(
        plan["logical_operations_by_component"]["skill_router"])
    os.environ["RSI_CALL_LEDGER_MAX_COMPONENT_MIPRO_PROPOSER"] = str(
        plan["logical_operations_by_component"]["mipro_proposer"])
    os.environ["RSI_CALL_LEDGER_MAX_COMPONENT_GROUNDING"] = str(
        plan["logical_operations_by_component"]["grounding"])
    os.environ["RSI_CALL_LEDGER_RESERVE_PI_PHYSICAL"] = "1"

    mipro_task_budget = args.task_evaluation_budget - confirmation_task_evaluations
    candidate_label_examples = list(task_by_id.values())
    gate_records = []
    candidate_calls = 0
    evolution_cfg = copy.deepcopy(cfg)
    evolution_cfg["evolution"]["allowed_artifacts"] = ["system_prompt"]

    def validate_instruction(prompt: str) -> dict:
        nonlocal candidate_calls
        prompt = str(prompt or "").strip()
        if not prompt:
            return {"passed": False, "reason": "empty instruction"}
        identity = content_id("cand", state.release_id, "system_prompt", prompt)
        label_gate = task_label_grounding_gate(prompt, candidate_label_examples)
        if not label_gate["passed"]:
            record = {
                "candidate_id": identity,
                "content_sha256": content_sha256(prompt),
                "task_label_grounding_gate": label_gate,
                "static_gate": None,
                "grounding_review": None,
                "passed": False,
                "reason": "task_label_contract_rejected",
            }
            gate_records.append(record)
            return record
        candidate = Candidate(
            candidate_id=identity, parent_release_id=state.release_id,
            artifact_type="system_prompt", artifact_id="system_prompt",
            diff="MIPRO proposal compared with incumbent prompt",
            new_content=prompt, content_sha256=content_sha256(prompt),
        )
        static = static_gate(candidate, evolution_cfg, state)
        record = {"candidate_id": identity, "content_sha256": candidate.content_sha256,
                  "task_label_grounding_gate": label_gate,
                  "static_gate": static, "grounding_review": None}
        if not static.get("passed"):
            record["passed"] = False
            record["reason"] = "static_gate_rejected"
            gate_records.append(record)
            return record
        if candidate_calls >= args.grounding_call_budget:
            raise RuntimeError("MIPRO exhausted candidate-grounding call budget")
        candidate_calls += 1
        review = run_prompt_grounding_review(prompt, cfg, "system_prompt")
        record["grounding_review"] = review
        record["passed"] = bool(review.get("passed"))
        record["reason"] = review.get("reason")
        gate_records.append(record)
        return record

    # MIPRO evaluates its incumbent once before proposing alternatives; run the
    # same gates on that instruction first so the default evaluation is covered.
    incumbent_gate = validate_instruction(state.system_prompt)
    if not incumbent_gate.get("passed"):
        raise RuntimeError("incumbent prompt failed MIPRO pre-evaluation gates")

    program_gate_records = []
    program_cfg = copy.deepcopy(cfg)
    program_cfg["evolution"]["allowed_artifacts"] = ["task_program"]

    def validate_program(program) -> dict:
        artifact = extract_task_program(program, train_task_ids)
        content = serialize_task_program(artifact["system_prompt"], artifact["task_demos"])
        candidate = _candidate(state.release_id, artifact["system_prompt"], content)
        component_gate = check_fr201_mipro_program(content, state)
        if not component_gate["passed"]:
            record = {
                "candidate_id": candidate.candidate_id,
                "content_sha256": candidate.content_sha256,
                "passed": False,
                "reason": component_gate["reason"],
                "changed_components": component_gate["changed_components"],
                "task_label_grounding_gate": None,
                "static_gate": None,
            }
            program_gate_records.append(record)
            return record
        label_gate = task_label_grounding_gate(
            artifact["system_prompt"], candidate_label_examples)
        if not label_gate["passed"]:
            program_gate_records.append({
                "candidate_id": candidate.candidate_id,
                "content_sha256": candidate.content_sha256,
                "passed": False,
                "reason": "task_label_contract_rejected",
                "task_label_grounding_gate": label_gate,
                "static_gate": None,
            })
            return {"passed": False, "reason": "task_label_contract_rejected",
                    "task_label_grounding_gate": label_gate}
        gate = static_gate(candidate, program_cfg, state)
        program_gate_records.append({"candidate_id": candidate.candidate_id,
                                     "content_sha256": candidate.content_sha256,
                                     "passed": gate.get("passed"),
                                     "task_label_grounding_gate": label_gate,
                                     "static_gate": gate})
        return gate

    mipro_error = None
    if compiled_path.exists():
        cached = read_json(compiled_path)
        for key, value in study_spec.items():
            if cached.get("study_spec", {}).get(key) != value:
                raise ValueError("MIPRO compile checkpoint protocol does not match this run")
        mipro = cached.get("mipro") or {}
        gate_records = cached.get("candidate_instruction_gates") or []
        program_gate_records = cached.get("candidate_program_static_gates") or []
        print(json.dumps({"status": "reused_compiled_candidate",
                          "study_id": study_id,
                          "task_evaluations": mipro.get("task_evaluations", 0),
                          "proposer_model_calls": mipro.get("prompt_model_calls", 0)},
                         ensure_ascii=False))
    else:
        try:
            mipro = compile_mipro_task_input(
                cfg=cfg, state=state,
                train_tasks=[task_by_id[str(row["task_id"])] for row in outer_train],
                validation_tasks=[task_by_id[str(row["task_id"])] for row in search_validation],
                execution_tasks=list(task_by_id.values()), train_ids=train_task_ids,
                validation_ids=search_ids, candidate_validator=validate_instruction,
                program_validator=validate_program, execution_seed=args.seed,
                task_evaluation_budget=mipro_task_budget,
                num_candidates=args.num_candidates, num_trials=args.num_trials,
                max_labeled_demos=args.max_labeled_demos,
                task_contract_context=task_contract_context,
            )
            write_json(compiled_path, {
                "study_spec": study_spec,
                "candidate_instruction_gates": gate_records,
                "candidate_program_static_gates": program_gate_records,
                "mipro": mipro,
            })
        except Exception as exc:
            mipro = {}
            mipro_error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    candidate = None
    candidate_task_label_grounding_gate = None
    internal_report = None
    selected_artifact = mipro.get("artifact")
    if selected_artifact and mipro_error is None:
        selected_content = serialize_task_program(
            selected_artifact["system_prompt"], selected_artifact["task_demos"])
        component_gate = check_fr201_mipro_program(selected_content, state)
        mipro["selected_component_gate"] = component_gate
        if not component_gate["passed"]:
            # Protect against old or inconsistent compile checkpoints before
            # any internal candidate-confirmation TaskRunner calls.
            mipro["artifact"] = None
            mipro["candidate_status"] = "fr201_nonpromotable_component"
            mipro["candidate_status_reason"] = component_gate["reason"]
    if mipro.get("artifact") and mipro_error is None:
        artifact = mipro["artifact"]
        content = serialize_task_program(artifact["system_prompt"], artifact["task_demos"])
        candidate = _candidate(state.release_id, artifact["system_prompt"], content)
        final_label_gate = task_label_grounding_gate(
            artifact["system_prompt"], candidate_label_examples)
        if not final_label_gate["passed"]:
            raise RuntimeError(
                "selected MIPRO task program failed final task-label grounding gate: "
                + str(final_label_gate["reason"]))
        candidate_task_label_grounding_gate = final_label_gate
        final_gate = static_gate(candidate, program_cfg, state)
        if not final_gate.get("passed"):
            raise RuntimeError("selected MIPRO task program failed final static gate")
        demo_ids = {str(item["task_id"]) for item in artifact["task_demos"]}
        if not demo_ids or not demo_ids <= train_task_ids:
            raise RuntimeError("selected MIPRO demos are not restricted to optimizer train IDs")
        registry = state.registry
        candidate_state = ArtifactState(
            release_id=state.release_id, system_prompt=artifact["system_prompt"],
            registry=registry, task_demos=artifact["task_demos"])
        baseline_rows = _baseline_train_rows(source, set(train_ids))
        by_seed = {}
        saved_confirm = {}
        if confirmation_checkpoint_path.exists():
            for row in read_jsonl(confirmation_checkpoint_path):
                key = (int(row.get("seed", -1)), str(row.get("task_id") or ""))
                if key in saved_confirm:
                    raise ValueError(f"duplicate MIPRO internal-confirmation checkpoint row: {key}")
                if key[0] not in baseline_rows or key[1] not in internal_ids:
                    raise ValueError(f"out-of-scope MIPRO confirmation checkpoint row: {key}")
                if row.get("candidate_id") != candidate.candidate_id:
                    raise ValueError("MIPRO confirmation checkpoint candidate ID mismatch")
                saved_confirm[key] = validate_task_answer_row(
                    row, task_by_id[key[1]], context=f"mipro-internal-checkpoint:{key}",
                    require_reported_gold_index=True)
        for seed, rows in sorted(baseline_rows.items()):
            base_rows = [rows[task_id] for task_id in sorted(internal_ids)]
            for row in base_rows:
                validate_task_answer_row(row, task_by_id[str(row["task_id"])],
                                         context=f"mipro-internal-baseline:{seed}",
                                         require_reported_gold_index=True)
            provider = build_provider(cfg, seed=seed)
            candidate_rows = []
            for task_id in sorted(internal_ids):
                key = (seed, task_id)
                if key in saved_confirm:
                    candidate_rows.append(saved_confirm[key])
                    continue
                task = task_by_id[task_id]
                outcome = TaskRunner(cfg, candidate_state, provider=provider,
                                     seed=seed).run(task)
                result = outcome.to_dict()
                result["seed"] = seed
                result["gold_label"] = str(task["options"][task["correct_index"]]).strip().casefold()
                result["candidate_id"] = candidate.candidate_id
                checked = validate_task_answer_row(
                    result, task, context=f"mipro-internal-candidate:{seed}:{task_id}",
                    require_reported_gold_index=True)
                with confirmation_checkpoint_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(checked, ensure_ascii=False) + "\n")
                    handle.flush()
                saved_confirm[key] = checked
                candidate_rows.append(checked)
            base_metrics, candidate_metrics = metrics(base_rows), metrics(candidate_rows)
            by_seed[str(seed)] = {
                "baseline": base_metrics, "candidate": candidate_metrics,
                "nonregresses_all_aggregate_and_class_metrics": nonregresses(
                    candidate_metrics, base_metrics),
                "rows": candidate_rows,
            }
        baseline_mean = sum(value["baseline"]["accuracy"] for value in by_seed.values()) / len(by_seed)
        candidate_mean = sum(value["candidate"]["accuracy"] for value in by_seed.values()) / len(by_seed)
        passed = (all(value["nonregresses_all_aggregate_and_class_metrics"]
                      for value in by_seed.values())
                  and candidate_mean > baseline_mean + 1e-12)
        internal_report = {
            "status": "passed" if passed else "rejected",
            "ids": sorted(internal_ids), "execution_labels": sorted(baseline_rows),
            "results_by_seed": by_seed, "mean_baseline_accuracy": baseline_mean,
            "mean_candidate_accuracy": candidate_mean,
            "requires_positive_accuracy_gain": True,
            "class_aware_all_metric_nonregression": True,
        }
        if not passed:
            candidate = None

    output = study_dir / "report.json"
    report = {
        "protocol": "pubmedqa-threeway-v2-mipro-task-input-v5-fr201-prompt-only-plan-bound-selection-exploratory",
        "execution_plan": plan,
        "execution_plan_path": str(plan_path),
        "execution_plan_sha256": plan["plan_sha256"],
        "request_ledger_path": str(ledger_path),
        "request_ledger_summary": summarize_ledger(ledger_path),
        "source_report": str(source_path),
        "source_report_sha256": sha256_file(source_path),
        "dataset": source["dataset"], "train_ids": train_ids,
        "candidate_selection_policy": MIPRO_SELECTION_POLICY_VERSION,
        "study_spec": study_spec,
        "mipro_train_ids": sorted(train_task_ids),
        "mipro_search_validation_ids": sorted(search_ids),
        "internal_confirmation_ids": sorted(internal_ids),
        "internal_confirmation_ids_accessed": bool(internal_report),
        "discovery_ids_accessed": False,
        "confirmation_ids_accessed": False,
        "selection_gate_accessed": False, "final_holdout_accessed": False,
        "retrieval_used": False, "release_id": state.release_id,
        "skill_routing_mode": source["skill_routing_mode"],
        "target_artifact": "task_program", "candidate": candidate.to_dict() if candidate else None,
        "candidate_id": candidate.candidate_id if candidate else None,
        "candidate_sha256": candidate.content_sha256 if candidate else None,
        "candidate_status": ("candidate" if candidate else
                             internal_report["status"] if internal_report else
                             "mipro_error" if mipro_error else
                             mipro.get("candidate_status", "no_candidate")),
        "eligible_for_discovery": bool(candidate),
        "candidate_instruction_gates": gate_records,
        "candidate_task_label_grounding_gate": candidate_task_label_grounding_gate,
        "candidate_program_static_gates": program_gate_records,
        "internal_confirmation": internal_report,
        "mipro": {key: value for key, value in mipro.items() if key != "artifact"},
        "mipro_error": mipro_error,
        "task_evaluations": {
            "mipro": mipro.get("task_evaluations", 0),
            "internal_confirmation": (confirmation_task_evaluations
                                       if internal_report else 0),
            "total": (mipro.get("task_evaluations", 0)
                      + (confirmation_task_evaluations if internal_report else 0)),
            "budget": args.task_evaluation_budget,
        },
        "config_sha256": config_fingerprint(cfg),
        "study_id": study_id,
        "compile_checkpoint": str(compiled_path),
        "internal_confirmation_checkpoint": str(confirmation_checkpoint_path),
    }
    write_json(output, report)
    print(json.dumps({"report": str(output), "candidate_id": report["candidate_id"],
                      "candidate_status": report["candidate_status"],
                      "eligible_for_discovery": report["eligible_for_discovery"],
                      "discovery_confirmation_accessed": False}, ensure_ascii=False))
    return 2 if mipro_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
