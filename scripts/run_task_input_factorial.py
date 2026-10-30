"""Run the real task-input GEPA ablation arms from one frozen release.

Each arm gets the same datasets, seed cohort and parent release. The script
never approves or publishes candidates. It records aggregate selection-gate
results and leaves per-task outputs in dev_optimize only.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

# Capture launcher-provided variables before importing runtime modules. The
# provider imports `.env` values into os.environ for local development; a live
# research run must not silently treat an old dotenv key as evidence that the
# exposed credential was rotated.
_PROCESS_ENV_AT_START = frozenset(os.environ)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_fingerprint, load_config  # noqa: E402
from src.contracts.types import content_sha256  # noqa: E402
from src.data.split import load_tasks  # noqa: E402
from src.evolution.task_demo_search import (  # noqa: E402
    DEFAULT_MAX_DEMO_CANDIDATE_SETS,
    split_demo_validation,
)
from src.evolution.gepa_search import visible_examples  # noqa: E402
from src.evolution.round import run_round  # noqa: E402
from src.governance import release  # noqa: E402
from src.util import (artifacts_path, artifacts_root, read_json, repo_root,
                      sha256_obj, write_json)  # noqa: E402

ARMS = ("prompt_only", "skill_description_only", "skill_body_only", "joint")
GEPA_ACCEPTANCE_CRITERIA = (
    "strict_improvement", "improvement_or_equal", "any_objective_improved")
DEFAULT_GEPA_MAX_METRIC_CALLS = 120
DEFAULT_GEPA_MAX_CANDIDATE_PROPOSALS = 8
DEFAULT_GEPA_REFLECTION_MINIBATCH_SIZE = 5
PI_MAX_PHYSICAL_ATTEMPTS_PER_OPERATION = 6
HTTP_MAX_PHYSICAL_ATTEMPTS_PER_OPERATION = 6


def estimate_factorial_request_budget(dataset_scopes: Dict[str, Dict[str, Any]],
                                      arms: Sequence[str], seeds: Sequence[int],
                                      *, max_metric_calls: int,
                                      max_candidate_proposals: int,
                                      fewshot_metric_calls: int = 0,
                                      model_routing: bool = False,
                                      grounding_calls: int = 0) -> Dict[str, Any]:
    """Conservative logical/physical request ceilings for one factorial.

    Counts one baseline, one exact-repeat control, and one candidate evaluation
    per arm. GEPA and demo metric ceilings are added independently because they
    are search-time task executions. Model routing adds one router operation to
    every task execution. One reflection operation is reserved per proposal.
    """
    seed_count = len(seeds)
    task_runs_per_dataset: dict[str, int] = {}
    for dataset, scope in dataset_scopes.items():
        optimize = int(scope["dev_optimize_count"])
        gate = int(scope["selection_gate_count"])
        # Baseline + one candidate, each on every seed, plus one exact repeat.
        task_runs_per_dataset[dataset] = (
            seed_count * 2 * (optimize + gate) + optimize)
    round_task_operations_per_arm = sum(task_runs_per_dataset.values())
    gepa_task_operations = len(arms) * int(max_metric_calls)
    task_operations = (len(arms) * round_task_operations_per_arm
                       + gepa_task_operations + int(fewshot_metric_calls))
    router_operations = task_operations if model_routing else 0
    reflection_operations = len(arms) * int(max_candidate_proposals)
    grounding_operations = int(grounding_calls)
    has_three_factors = {"prompt_only", "skill_description_only",
                         "skill_body_only"}.issubset(set(arms))
    factorial_operations = (8 * seed_count
                            * sum(int(scope["dev_optimize_count"])
                                  + int(scope["selection_gate_count"])
                                  for scope in dataset_scopes.values())
                            if has_three_factors else 0)
    logical = (task_operations + router_operations + reflection_operations
               + grounding_operations + factorial_operations
               + (factorial_operations if model_routing else 0))
    # Pi task calls allow two bridge launches × three provider attempts. HTTP
    # router/reflection clients currently cap their retry loop at six attempts.
    task_operations_total = task_operations + factorial_operations
    router_operations_total = router_operations + (
        factorial_operations if model_routing else 0)
    physical = (task_operations_total * PI_MAX_PHYSICAL_ATTEMPTS_PER_OPERATION
                + (router_operations_total + reflection_operations)
                * HTTP_MAX_PHYSICAL_ATTEMPTS_PER_OPERATION
                + grounding_operations * HTTP_MAX_PHYSICAL_ATTEMPTS_PER_OPERATION)
    return {
        "logical_request_ceiling": logical,
        "physical_http_attempt_ceiling": physical,
        "task_answer_operations": task_operations_total,
        "skill_router_operations": router_operations_total,
        "combined_factorial_task_operations": factorial_operations,
        "gepa_reflection_operations": reflection_operations,
        "candidate_grounding_operations": grounding_operations,
        "round_task_operations_per_arm": round_task_operations_per_arm,
        "gepa_task_operations": gepa_task_operations,
        "fewshot_task_operations": int(fewshot_metric_calls),
        "round_task_operations_by_dataset_per_arm": task_runs_per_dataset,
        "assumptions": {
            "candidate_evaluations_per_arm": 1,
            "candidate_grounding_calls_per_arm": 1,
            "full_2x2x2_factorial_conditions": 8 if has_three_factors else 0,
            "baseline_exact_repeat_per_dataset": 1,
            "gepa_metric_calls_are_additive_to_round_evaluations": True,
            "fewshot_metric_calls_are_additive_to_gepa_metric_calls": True,
            "pi_physical_attempts_per_task_operation": PI_MAX_PHYSICAL_ATTEMPTS_PER_OPERATION,
            "http_physical_attempts_per_router_or_reflection_operation": HTTP_MAX_PHYSICAL_ATTEMPTS_PER_OPERATION,
            "http_physical_attempts_per_grounding_operation": HTTP_MAX_PHYSICAL_ATTEMPTS_PER_OPERATION,
        },
    }


def build_combined_factorial_conditions(base_state, prompt_candidate,
                                        description_candidate,
                                        body_candidate):
    """Create the real 2^3 P×description×body conditions from gated winners."""
    from src.evolution.apply import combine
    from src.runtime.runner import ArtifactState
    from src.runtime.skills import with_body, with_description
    from src.eval.ablation import build_factorial_conditions

    expected = ((prompt_candidate, "system_prompt"),
                (description_candidate, "skill_description"),
                (body_candidate, "skill_body"))
    for candidate, kind in expected:
        if candidate.artifact_type != kind:
            raise ValueError(f"combined ablation expected {kind}, got {candidate.artifact_type}")
        if candidate.parent_release_id != base_state.release_id:
            raise ValueError("combined ablation candidate parent differs from frozen release")
        grounding = (candidate.generator or {}).get("pre_evaluation_gate") or {}
        if grounding.get("passed") is not True:
            raise ValueError(f"combined ablation candidate lacks a passed grounding gate: {kind}")

    description_registry = with_description(
        base_state.registry, description_candidate.artifact_id,
        description_candidate.new_content,
        list(description_candidate.extra.get("new_triggers", [])) or None)
    body_registry = with_body(base_state.registry, body_candidate.artifact_id,
                              body_candidate.new_content)
    both_registry = combine(base_state,
                            [description_candidate, body_candidate]).registry
    conditions = build_factorial_conditions(
        base_state, prompt_candidate.new_content, description_registry,
        body_registry, both_registry)
    # Assert factor isolation at construction time; a registry replacement bug
    # here would contaminate every reported interaction and Shapley effect.
    for code, state in conditions.items():
        p, d, b = code[1] == "1", code[2] == "1", code[3] == "1"
        if p != (state.system_prompt == prompt_candidate.new_content):
            raise RuntimeError(f"{code} Prompt factor does not match its assignment")
        skill_base = base_state.registry.get(description_candidate.artifact_id)
        skill_state = state.registry.get(description_candidate.artifact_id)
        if d != (skill_state.description == description_candidate.new_content):
            raise RuntimeError(f"{code} description factor does not match its assignment")
        body_base = base_state.registry.get(body_candidate.artifact_id)
        body_state = state.registry.get(body_candidate.artifact_id)
        if b != (body_state.body == body_candidate.new_content):
            raise RuntimeError(f"{code} body factor does not match its assignment")
        if skill_base is None or body_base is None:
            raise ValueError("combined ablation target Skill is absent from the baseline release")
    if any(state.task_demos != base_state.task_demos for state in conditions.values()):
        raise RuntimeError("Prompt/Skill factorial unexpectedly changed few-shot demos")
    return conditions


def _accepted_arm_candidate(arm_result: Dict[str, Any], expected_artifact: str):
    candidate_rows = [item for item in (arm_result.get("candidates") or [])
                      if item.get("target_artifact") == expected_artifact
                      and item.get("status") == "proposed_accept"
                      and item.get("static_gate_passed") is True
                      and (item.get("pre_evaluation_gate") or {}).get("passed") is True]
    if len(candidate_rows) != 1:
        return None
    candidate_id = str(candidate_rows[0].get("candidate_id") or "")
    candidate_path = artifacts_path("candidates", candidate_id + ".json")
    if not candidate_id or not candidate_path.is_file():
        return None
    from src.contracts.types import Candidate, content_sha256
    candidate = Candidate(**read_json(candidate_path))
    if candidate.artifact_type != expected_artifact:
        raise RuntimeError("saved candidate artifact differs from its factorial arm")
    return candidate


def _run_combined_factorial(base_cfg: Dict[str, Any], arm_results: Dict[str, Any],
                            datasets: Sequence[str], seeds: Sequence[int],
                            limit: int | None, skill_name: str,
                            *, task_offset: int, selection_gate_offset: int,
                            max_metric_calls: int, max_candidate_proposals: int,
                            reflection_minibatch_size: int,
                            acceptance_criterion: str,
                            fewshot_max_candidate_sets: int,
                            preflight_plan_sha256: str | None,
                            current_release: Callable[[], str],
                            parent_release_id: str) -> Dict[str, Any]:
    """Evaluate the full P×description×body matrix from three gated winners."""
    singleton = {
        "prompt_only": ("system_prompt", "prompt_candidate"),
        "skill_description_only": ("skill_description", "description_candidate"),
        "skill_body_only": ("skill_body", "body_candidate"),
    }
    selected = {
        key: _accepted_arm_candidate(arm_results.get(key, {}), artifact)
        for key, (artifact, _) in singleton.items()
    }
    missing = [key for key, candidate in selected.items() if candidate is None]
    if missing:
        return {"status": "unavailable_missing_gated_factor",
                "missing_arms": missing,
                "reason": "A4 requires individually accepted, static-gated, grounded factors"}
    if current_release() != parent_release_id:
        raise RuntimeError("ACTIVE release changed before combined factorial evaluation")
    if preflight_plan_sha256:
        current_plan = build_factorial_preflight(
            base_cfg, datasets, seeds, limit, skill_name,
            task_offset=task_offset,
            selection_gate_offset=selection_gate_offset,
            max_metric_calls=max_metric_calls,
            max_candidate_proposals=max_candidate_proposals,
            reflection_minibatch_size=reflection_minibatch_size,
            acceptance_criterion=acceptance_criterion,
            fewshot_max_candidate_sets=fewshot_max_candidate_sets,
            arms=list(arm_results), parent_release_id=parent_release_id)
        if current_plan["plan_sha256"] != preflight_plan_sha256:
            raise RuntimeError("preflight plan hash changed before combined factorial dispatch")

    from src.eval import ablation
    from src.governance import release

    base_state = release.load_state()
    if base_state.release_id != parent_release_id:
        raise RuntimeError("combined factorial did not load the frozen parent release")
    conditions = build_combined_factorial_conditions(
        base_state, selected["prompt_only"],
        selected["skill_description_only"], selected["skill_body_only"])
    ablation_cfg = configure_arm(
        base_cfg, "prompt_only", skill_name, task_offset=task_offset,
        selection_gate_offset=selection_gate_offset,
        max_metric_calls=max_metric_calls,
        max_candidate_proposals=max_candidate_proposals,
        reflection_minibatch_size=reflection_minibatch_size,
        acceptance_criterion=acceptance_criterion,
        fewshot_max_candidate_sets=fewshot_max_candidate_sets)
    result = ablation.run_ablation(
        ablation_cfg, conditions, list(datasets),
        splits=("dev_optimize", "selection_gate"), seeds=list(seeds),
        limit=limit, metric="accuracy",
        B=int((ablation_cfg.get("evaluation") or {}).get(
            "bootstrap", {}).get("B", 10000)),
        rng_seed=int((ablation_cfg.get("evaluation") or {}).get(
            "bootstrap", {}).get("rng_seed", 20260924)),
        execution_plan_sha256=preflight_plan_sha256)
    result.update({
        "scope": "exploratory_task_input_full_factorial",
        "parent_release_id": parent_release_id,
        "candidate_sources": {
            arm: {"candidate_id": cand.candidate_id,
                  "artifact_type": cand.artifact_type,
                  "artifact_id": cand.artifact_id,
                  "content_sha256": cand.content_sha256}
            for arm, cand in (("prompt_only", selected["prompt_only"]),
                              ("skill_description_only", selected["skill_description_only"]),
                              ("skill_body_only", selected["skill_body_only"]))},
        "final_holdout_read": False,
        "published": False,
    })
    report_path = ablation.save(result)
    return {"status": "complete", "report_path": report_path,
            "design": result.get("design"),
            "factorial_codes": sorted((result.get("factorial") or {}).get(
                datasets[0], {}).get("dev_optimize", {}).get("conditions", {})),
            "candidate_sources": result["candidate_sources"],
            "published": False, "final_holdout_read": False}


def _factorial_candidate_pre_evaluation_gate(candidate, cfg, state):
    """Ground each statically valid candidate before round-level scoring."""
    from src.evolution.task_program import parse_task_program
    from src.runtime.call_ledger import call_component
    from scripts.evaluate_pubmedqa_prompt_discovery import run_prompt_grounding_review

    artifact_type = str(candidate.artifact_type)
    content = str(candidate.new_content)
    scan_type = artifact_type
    if artifact_type == "task_program":
        program = parse_task_program(content)
        content = str(program["system_prompt"])
        scan_type = "system_prompt"
    if scan_type not in {"system_prompt", "skill_description", "skill_body"}:
        return {"passed": False, "reason": "unsupported_artifact_grounding_scope",
                "artifact_type": artifact_type}
    with call_component("grounding"):
        review = run_prompt_grounding_review(content, cfg, scan_type)
    passed = bool(review.get("passed") is True
                  and review.get("verdict") == "claim_free"
                  and review.get("candidate_sha256") == content_sha256(content.strip())
                  and review.get("artifact_type") == scan_type
                  and review.get("assertion_scan_complete") is True
                  and str(review.get("model_id") or "").strip())
    return {"passed": passed,
            "reason": "claim_free_grounding_passed" if passed else
                      "grounding_review_failed_or_unresolved",
            "artifact_type": artifact_type,
            "scanned_artifact_type": scan_type,
            "grounding_review": review}


def configure_arm(base: Dict[str, Any], arm: str, skill_name: str, *,
                  task_offset: int = 0,
                  selection_gate_offset: int = 0,
                  max_metric_calls: int = DEFAULT_GEPA_MAX_METRIC_CALLS,
                  max_candidate_proposals: int = DEFAULT_GEPA_MAX_CANDIDATE_PROPOSALS,
                  reflection_minibatch_size: int = DEFAULT_GEPA_REFLECTION_MINIBATCH_SIZE,
                  acceptance_criterion: str = "improvement_or_equal",
                  fewshot_max_candidate_sets: int = DEFAULT_MAX_DEMO_CANDIDATE_SETS,
                  ) -> Dict[str, Any]:
    """Create an isolated, retrieval-free protocol config for one ablation arm."""
    if arm not in ARMS:
        raise ValueError(f"unknown factorial arm: {arm}")
    if task_offset < 0 or selection_gate_offset < 0:
        raise ValueError("task offsets must be nonnegative")
    if min(max_metric_calls, max_candidate_proposals, reflection_minibatch_size,
           fewshot_max_candidate_sets) < 1:
        raise ValueError("GEPA budget, proposal count, minibatch size and few-shot candidate count must be positive")
    if acceptance_criterion not in GEPA_ACCEPTANCE_CRITERIA:
        raise ValueError("unsupported GEPA acceptance criterion")
    cfg = copy.deepcopy(base)
    runtime = cfg.setdefault("runtime", {})
    runtime["provider"] = "pi"
    runtime["input_mode"] = "task_input"
    runtime.setdefault("task_input_max_chars", 30000)
    tools = cfg.setdefault("tools", {})
    tools["active"] = [name for name in tools.get("active", [])
                       if name != "academic.search"]
    retrieval = cfg.setdefault("retrieval", {})
    retrieval.pop("query_control", None)
    retrieval.pop("pubmed_source_queries", None)

    evaluation = cfg.setdefault("evaluation", {})
    evaluation["sealed_pairing"] = False
    evaluation["main_metric"] = "accuracy"
    evaluation["primary_axis"] = "accuracy"
    evaluation["task_offset"] = 0
    evaluation["task_offsets"] = {
        "dev_optimize": int(task_offset),
        "selection_gate": int(selection_gate_offset),
    }

    evolution = cfg.setdefault("evolution", {})
    gepa = evolution.setdefault("gepa", {})
    gepa["enabled"] = True
    gepa["max_metric_calls"] = int(max_metric_calls)
    gepa["max_candidate_proposals"] = int(max_candidate_proposals)
    gepa["reflection_minibatch_size"] = int(reflection_minibatch_size)
    # Keep accuracy primary and pass evidence-result hints through ASI in every
    # arm. Acceptance is an explicit protocol parameter: its default preserves
    # legacy tied-minibatch behavior, while strict mode filters ties early.
    gepa["multi_objective"] = True
    gepa["acceptance_criterion"] = acceptance_criterion
    gepa["feedback_evidence_signals"] = True
    gepa["task_offset"] = int(task_offset)
    skill_arm_artifact = {
        "skill_description_only": "skill_description",
        "skill_body_only": "skill_body",
    }.get(arm)
    gepa["target_artifact"] = skill_arm_artifact or "system_prompt"
    if skill_arm_artifact:
        gepa["target_skill"] = skill_name
    else:
        gepa.pop("target_skill", None)
    fewshot = evolution.setdefault("fewshot", {})
    fewshot["enabled"] = arm == "joint"
    fewshot["max_candidate_sets"] = int(fewshot_max_candidate_sets)
    if arm == "joint":
        # The demo search evaluates every candidate on every paired optimize
        # seed; reserve room for six ordered demo programs × two instructions
        # × three seeds × internal validation (computed below).
        fewshot["max_metric_calls"] = max(
            int(fewshot.get("max_metric_calls", 0)), 720)
    evolution["allowed_artifacts"] = (
        ["skill_description"] if arm == "skill_description_only" else
        ["skill_body"] if arm == "skill_body_only" else
        ["task_program", "system_prompt"] if arm == "joint" else
        ["system_prompt"])
    return cfg


def build_factorial_preflight(base_cfg: Dict[str, Any], datasets: Sequence[str],
                              seeds: Sequence[int], limit: int | None = None,
                              skill_name: str = "litqa-literature-synthesis",
                              task_offset: int = 0,
                              selection_gate_offset: int = 0,
                              max_metric_calls: int = DEFAULT_GEPA_MAX_METRIC_CALLS,
                              max_candidate_proposals: int = DEFAULT_GEPA_MAX_CANDIDATE_PROPOSALS,
                              reflection_minibatch_size: int = DEFAULT_GEPA_REFLECTION_MINIBATCH_SIZE,
                              acceptance_criterion: str = "improvement_or_equal",
                              fewshot_max_candidate_sets: int = DEFAULT_MAX_DEMO_CANDIDATE_SETS,
                              arms: Sequence[str] = ARMS,
                              parent_release_id: str | None = None) -> Dict[str, Any]:
    """Freeze and validate the data/protocol plan without calling any model."""
    seed_list = [int(seed) for seed in seeds]
    dataset_list = [str(dataset) for dataset in datasets]
    arm_list = list(arms)
    if len(seed_list) < 3 or len(set(seed_list)) < 3:
        raise ValueError("real task-input factorial requires at least three distinct seeds")
    if not dataset_list or any(not name for name in dataset_list) or len(set(dataset_list)) != len(dataset_list):
        raise ValueError("factorial requires distinct, non-empty datasets")
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer when specified")
    if task_offset < 0 or selection_gate_offset < 0:
        raise ValueError("task offsets must be nonnegative")
    if acceptance_criterion not in GEPA_ACCEPTANCE_CRITERIA:
        raise ValueError("unsupported GEPA acceptance criterion")
    if not arm_list or len(set(arm_list)) != len(arm_list) or any(
            arm not in ARMS for arm in arm_list):
        raise ValueError("arms must be distinct members of ARMS")

    from src.data.freeze import require_verified
    require_verified(dataset_list)
    dataset_scopes: dict[str, dict[str, Any]] = {}
    selected_optimize_ids: list[str] = []
    for dataset in dataset_list:
        optimize_rows = sorted(load_tasks(dataset, splits=["dev_optimize"]),
                               key=lambda row: str(row["task_id"]))
        gate_rows = sorted(load_tasks(dataset, splits=["selection_gate"]),
                           key=lambda row: str(row["task_id"]))
        optimize_rows = optimize_rows[task_offset:task_offset + limit if limit is not None else None]
        gate_rows = gate_rows[selection_gate_offset:selection_gate_offset + limit
                              if limit is not None else None]
        if not optimize_rows:
            raise ValueError(f"no dev_optimize tasks in planned cohort for {dataset}")
        if not gate_rows:
            raise ValueError(f"no selection_gate tasks in planned cohort for {dataset}")
        optimize_ids = [str(row["task_id"]) for row in optimize_rows]
        gate_ids = [str(row["task_id"]) for row in gate_rows]
        selected_optimize_ids.extend(optimize_ids)
        if len(set(optimize_ids)) != len(optimize_ids) or len(set(gate_ids)) != len(gate_ids):
            raise ValueError(f"duplicate task IDs in planned cohort for {dataset}")
        if set(optimize_ids) & set(gate_ids):
            raise ValueError(f"dev_optimize and selection_gate overlap for {dataset}")
        dataset_scopes[dataset] = {
            "dev_optimize_count": len(optimize_rows),
            "dev_optimize_ids_sha256": sha256_obj(optimize_ids),
            "dev_optimize_material_sha256": sha256_obj(optimize_rows),
            "selection_gate_count": len(gate_rows),
            "selection_gate_ids_sha256": sha256_obj(gate_ids),
            "selection_gate_material_sha256": sha256_obj(gate_rows),
            "final_holdout_read": False,
        }
    if len(set(selected_optimize_ids)) != len(selected_optimize_ids):
        raise ValueError("dev_optimize task IDs must be unique across factorial datasets")

    arm_plans: dict[str, dict[str, Any]] = {}
    expected_search_ids: tuple[str, ...] | None = None
    for arm in arm_list:
        cfg = configure_arm(
            base_cfg, arm, skill_name, task_offset=task_offset,
            selection_gate_offset=selection_gate_offset,
            max_metric_calls=max_metric_calls,
            max_candidate_proposals=max_candidate_proposals,
            reflection_minibatch_size=reflection_minibatch_size,
            acceptance_criterion=acceptance_criterion,
            fewshot_max_candidate_sets=fewshot_max_candidate_sets)
        train, validation = visible_examples(cfg, dataset_list, limit)
        train_ids = [str(row["task_id"]) for row in train]
        validation_ids = [str(row["task_id"]) for row in validation]
        if set(train_ids) & set(validation_ids):
            raise ValueError("GEPA train and inner-validation task IDs overlap")
        search_ids = tuple(sorted(train_ids + validation_ids))
        if search_ids != tuple(sorted(selected_optimize_ids)):
            raise ValueError(
                "GEPA train/validation IDs do not exactly match the planned dev_optimize cohort")
        if expected_search_ids is None:
            expected_search_ids = search_ids
        elif search_ids != expected_search_ids:
            raise ValueError("factorial arms do not share one GEPA task cohort")
        arm_plan: dict[str, Any] = {
            "config_fingerprint": config_fingerprint(cfg),
            "gepa_train_count": len(train_ids),
            "gepa_train_ids_sha256": sha256_obj(sorted(train_ids)),
            "gepa_inner_validation_count": len(validation_ids),
            "gepa_inner_validation_ids_sha256": sha256_obj(sorted(validation_ids)),
            "gepa_metric_call_ceiling_per_round": max_metric_calls,
            "candidate_proposal_ceiling_per_round": max_candidate_proposals,
            "reflection_minibatch_size": reflection_minibatch_size,
            "acceptance_criterion": acceptance_criterion,
            "fewshot_candidate_set_ceiling": fewshot_max_candidate_sets,
            "selected_artifacts": cfg["evolution"]["allowed_artifacts"],
            "retrieval_input": False,
            "final_holdout_read": False,
        }
        if arm == "joint":
            discovery, confirmation = split_demo_validation(
                validation, require_class_coverage=True)
            fewshot_calls = (
                len(discovery) * len(seed_list) * fewshot_max_candidate_sets * 2
                + len(confirmation) * len(seed_list))
            arm_plan["fewshot_search"] = {
                "discovery_count": len(discovery),
                "confirmation_count": len(confirmation),
                "candidate_sets": fewshot_max_candidate_sets,
                "instruction_variants_upper_bound": 2,
                "metric_call_ceiling": fewshot_calls,
                "discovery_ids_sha256": sha256_obj(sorted(
                    str(row["task_id"]) for row in discovery)),
                "confirmation_ids_sha256": sha256_obj(sorted(
                    str(row["task_id"]) for row in confirmation)),
            }
        arm_plans[arm] = arm_plan

    fewshot_total = sum(int((item.get("fewshot_search") or {}).get(
        "metric_call_ceiling", 0)) for item in arm_plans.values())
    request_budget = estimate_factorial_request_budget(
        dataset_scopes, arm_list, seed_list,
        max_metric_calls=max_metric_calls,
        max_candidate_proposals=max_candidate_proposals,
        fewshot_metric_calls=fewshot_total,
        model_routing=(str((base_cfg.get("skills") or {}).get(
            "routing_mode", "lexical")) == "model"),
        grounding_calls=len(arm_list))
    plan: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": "task-input-factorial-preflight-v1",
        "status": "valid_zero_call_preflight",
        "datasets": dataset_scopes,
        "gepa_visible_cohort_ids_sha256": sha256_obj(list(expected_search_ids or ())),
        "seeds": seed_list,
        "seed_semantics": "repeat calls on the same frozen cohort; no model RNG seed is forwarded",
        "limit_per_dataset": limit,
        "task_offset": task_offset,
        "selection_gate_offset": selection_gate_offset,
        "selected_arms": arm_list,
        "arms": arm_plans,
        "request_budget": request_budget,
        "parent_release_id": parent_release_id,
        "selection_gate_access": "aggregate metrics only during execution",
        "final_holdout_read": False,
        "model_calls": 0,
        "plan_sha256": "",
    }
    plan["plan_sha256"] = sha256_obj({key: value for key, value in plan.items()
                                      if key != "plan_sha256"})
    return plan


def run_factorial(base_cfg: Dict[str, Any], datasets: Sequence[str],
                  seeds: Sequence[int], limit: int | None = None,
                  skill_name: str = "litqa-literature-synthesis",
                  task_offset: int = 0,
                  selection_gate_offset: int = 0,
                  round_runner: Callable[..., Dict[str, Any]] | None = None,
                  current_release: Callable[[], str] | None = None,
                  gepa_preflight: Callable[[], Any] | None = None,
                  max_metric_calls: int = DEFAULT_GEPA_MAX_METRIC_CALLS,
                  max_candidate_proposals: int = DEFAULT_GEPA_MAX_CANDIDATE_PROPOSALS,
                  reflection_minibatch_size: int = DEFAULT_GEPA_REFLECTION_MINIBATCH_SIZE,
                  acceptance_criterion: str = "improvement_or_equal",
                  fewshot_max_candidate_sets: int = DEFAULT_MAX_DEMO_CANDIDATE_SETS,
                  arms: Sequence[str] = ARMS,
                  preflight_plan_sha256: str | None = None,
                  ) -> Dict[str, Any]:
    """Run selected nonpublishing arms on one release and repeated cohort."""
    seed_list = [int(seed) for seed in seeds]
    if len(seed_list) < 3 or len(set(seed_list)) < 3:
        raise ValueError("real task-input factorial requires at least three distinct seeds")
    if not datasets:
        raise ValueError("factorial requires at least one dataset")
    if task_offset < 0 or selection_gate_offset < 0:
        raise ValueError("task offsets must be nonnegative")
    if acceptance_criterion not in GEPA_ACCEPTANCE_CRITERIA:
        raise ValueError("unsupported GEPA acceptance criterion")
    arm_list = list(arms)
    if not arm_list or len(set(arm_list)) != len(arm_list) or any(
            arm not in ARMS for arm in arm_list):
        raise ValueError("arms must be distinct members of ARMS")
    injected_round_runner = round_runner is not None
    round_runner = round_runner or run_round
    current_release = current_release or release.current_release_id
    parent = current_release()
    if preflight_plan_sha256 is not None:
        current_plan = build_factorial_preflight(
            base_cfg, datasets, seed_list, limit, skill_name,
            task_offset=task_offset,
            selection_gate_offset=selection_gate_offset,
            max_metric_calls=max_metric_calls,
            max_candidate_proposals=max_candidate_proposals,
            reflection_minibatch_size=reflection_minibatch_size,
            acceptance_criterion=acceptance_criterion,
            fewshot_max_candidate_sets=fewshot_max_candidate_sets,
            arms=arm_list, parent_release_id=parent)
        if current_plan["plan_sha256"] != preflight_plan_sha256:
            raise ValueError("preflight plan hash mismatch before factorial dispatch")
    run_id = "task-input-factorial-" + uuid.uuid4().hex[:12]
    report: Dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "cohort_status": "exploratory_dev_optimize_and_selection_gate",
        "selected_arms": arm_list,
        "experiment_scope": ("full_factorial" if tuple(arm_list) == ARMS
                             else "targeted_arm_followup"),
        "optimizer": "pinned GEPA; joint arm crosses instruction with task-visible demos",
        "gepa_search_budget": {
            "max_metric_calls": int(max_metric_calls),
            "max_candidate_proposals": int(max_candidate_proposals),
            "reflection_minibatch_size": int(reflection_minibatch_size),
            "multi_objective": True,
            "acceptance_criterion": acceptance_criterion,
            "feedback_evidence_signals": True,
        },
        "fewshot_search_budget": {
            "max_candidate_sets": int(fewshot_max_candidate_sets),
            "max_metric_calls": None,
            "budget_formula": "discovery_tasks × repeated_seeds × candidate_sets × max_instruction_variants(2) + confirmation_tasks × repeated_seeds",
        },
        "datasets": list(datasets),
        "seeds": seed_list,
        "seed_semantics": {
            "distinct_execution_labels": len(set(seed_list)),
            "task_cohort": "same frozen split IDs for every seed",
            "model_seed_forwarded": False,
            "interpretation": (
                "repeated calls on the same cohort; not independent cohorts or "
                "controlled model RNG seeds"),
        },
        "limit": limit,
        "task_offset": int(task_offset),
        "selection_gate_offset": int(selection_gate_offset),
        "preflight_plan_sha256": preflight_plan_sha256,
        "parent_release_id": parent,
        "selection_gate_access": "aggregate metrics only",
        "final_holdout_access": False,
        "published": False,
        "arms": {},
    }
    path = artifacts_path("probe", "task_input_factorial", run_id + ".json")
    write_json(path, report)

    ledger_path = artifacts_path("probe", "task_input_factorial",
                                 run_id + ".requests.jsonl")
    execution_env_names = (
        "RSI_CALL_LEDGER_PATH", "RSI_CALL_LEDGER_MAX_LOGICAL",
        "RSI_CALL_LEDGER_MAX_PHYSICAL", "RSI_ACTIVE_EXECUTION_PLAN_SHA256",
        "RSI_EXECUTION_GUARD_PATH")
    previous_execution_env = {
        name: os.environ.get(name) for name in execution_env_names}
    execution_guard_path = None
    if not injected_round_runner:
        budget = (build_factorial_preflight(
            base_cfg, datasets, seed_list, limit, skill_name,
            task_offset=task_offset,
            selection_gate_offset=selection_gate_offset,
            max_metric_calls=max_metric_calls,
            max_candidate_proposals=max_candidate_proposals,
            reflection_minibatch_size=reflection_minibatch_size,
            acceptance_criterion=acceptance_criterion,
            fewshot_max_candidate_sets=fewshot_max_candidate_sets,
            arms=arm_list, parent_release_id=parent)["request_budget"])
        os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path)
        os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(
            budget["logical_request_ceiling"])
        os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(
            budget["physical_http_attempt_ceiling"])
        report["request_budget"] = {**budget,
                                    "ledger_path": ledger_path.as_posix(),
                                    "enforced_before_dispatch": True}

    if not injected_round_runner:
        try:
            if gepa_preflight is None:
                from src.evolution.gepa_adapter import require_gepa
                gepa_preflight = require_gepa
            gepa_preflight()
        except Exception as exc:
            for name, value in previous_execution_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            _record_failure(report, path, "preflight", exc, base_cfg)
            raise

    if not injected_round_runner and preflight_plan_sha256 is not None:
        execution_guard_path = artifacts_path(
            "probe", "task_input_factorial",
            run_id + ".execution-started.json")
        with execution_guard_path.open("x", encoding="utf-8") as handle:
            json.dump({"plan_sha256": preflight_plan_sha256,
                       "run_id": run_id, "status": "started"},
                      handle, sort_keys=True)
            handle.write("\n")
        os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = preflight_plan_sha256
        os.environ["RSI_EXECUTION_GUARD_PATH"] = str(execution_guard_path)

    def execute_arm(arm: str) -> Dict[str, Any]:
        if current_release() != parent:
            raise RuntimeError("ACTIVE release changed during factorial; aborting remaining arms")
        if preflight_plan_sha256 is not None:
            current_plan = build_factorial_preflight(
                base_cfg, datasets, seed_list, limit, skill_name,
                task_offset=task_offset,
                selection_gate_offset=selection_gate_offset,
                max_metric_calls=max_metric_calls,
                max_candidate_proposals=max_candidate_proposals,
                reflection_minibatch_size=reflection_minibatch_size,
                acceptance_criterion=acceptance_criterion,
                fewshot_max_candidate_sets=fewshot_max_candidate_sets,
                arms=arm_list, parent_release_id=parent)
            if current_plan["plan_sha256"] != preflight_plan_sha256:
                raise RuntimeError("preflight plan hash changed before arm dispatch")
        cfg = configure_arm(
            base_cfg, arm, skill_name,
            task_offset=task_offset,
            selection_gate_offset=selection_gate_offset,
            max_metric_calls=max_metric_calls,
            max_candidate_proposals=max_candidate_proposals,
            reflection_minibatch_size=reflection_minibatch_size,
            acceptance_criterion=acceptance_criterion,
            fewshot_max_candidate_sets=fewshot_max_candidate_sets)
        if arm == "joint":
            from src.evolution.gepa_search import visible_examples
            from src.evolution.task_demo_search import split_demo_validation
            _, internal_validation = visible_examples(cfg, datasets, limit)
            # This split only estimates the planned metric-call budget. The
            # real candidate search uses strict class coverage and fails closed
            # before evaluation if any gold label cannot appear in both cohorts.
            discovery, confirmation = split_demo_validation(
                internal_validation, require_class_coverage=False)
            fewshot = cfg["evolution"]["fewshot"]
            fewshot["max_metric_calls"] = (
                len(discovery) * len(seed_list)
                * int(fewshot.get("max_candidate_sets",
                                 DEFAULT_MAX_DEMO_CANDIDATE_SETS)) * 2
                + len(confirmation) * len(seed_list))
            report["fewshot_search_budget"].update({
                "max_metric_calls": fewshot["max_metric_calls"],
                "discovery_tasks": len(discovery),
                "confirmation_tasks": len(confirmation),
                "instruction_variants_upper_bound": 2,
            })
        result = round_runner(
            cfg, datasets=list(datasets), seeds=seed_list, limit=limit,
            max_candidates=1, fewshot_search=(arm == "joint"),
            candidate_pre_evaluation_gate=(
                None if injected_round_runner else
                _factorial_candidate_pre_evaluation_gate))
        if current_release() != parent:
            raise RuntimeError("round changed ACTIVE release; factorial must not publish")
        if result.get("release_id") != parent:
            raise RuntimeError(f"{arm} round used a different parent release")
        if result.get("seeds") != seed_list:
            raise RuntimeError(f"{arm} round did not use the requested seed cohort")
        if not (result.get("seed_requirement") or {}).get("passed", False):
            raise RuntimeError(f"{arm} round failed the three-seed requirement")
        candidates = result.get("candidates") or []
        if any(c.get("status") == "proposed_accept" and
               not c.get("static_gate_passed", False) for c in candidates):
            raise RuntimeError(f"{arm} contains an accepted candidate that failed static gate")
        if any(c.get("status") == "proposed_accept" and not
               ((c.get("pre_evaluation_gate") or {}).get("passed") is True)
               for c in candidates):
            raise RuntimeError(
                f"{arm} contains an accepted candidate without a passed grounding gate")
        return {
            "round_id": result.get("round_id"),
            "round_report_path": result.get("report_path"),
            "config_fingerprint": result.get("config_fingerprint"),
            "candidate_count": len(candidates),
            "candidates": [{key: candidate.get(key) for key in (
                "candidate_id", "attributed_class", "target_artifact", "status",
                "static_gate_passed", "pre_evaluation_gate", "delta_pp", "task_support", "verdict_reason",
                "review_request") if key in candidate} for candidate in candidates],
        }

    try:
        for arm in arm_list:
            try:
                report["arms"][arm] = execute_arm(arm)
            except Exception as exc:
                _record_failure(report, path, arm, exc, base_cfg)
                raise
            if not injected_round_runner:
                from src.runtime.call_ledger import summarize_ledger
                report["request_budget"]["ledger_snapshot"] = summarize_ledger(
                    ledger_path)
            write_json(path, report)
        has_three_singleton_arms = {
            "prompt_only", "skill_description_only", "skill_body_only"
        }.issubset(set(arm_list))
        if has_three_singleton_arms and not injected_round_runner:
            try:
                report["combined_factorial"] = _run_combined_factorial(
                    base_cfg, report["arms"], datasets, seed_list, limit, skill_name,
                    task_offset=task_offset,
                    selection_gate_offset=selection_gate_offset,
                    max_metric_calls=max_metric_calls,
                    max_candidate_proposals=max_candidate_proposals,
                    reflection_minibatch_size=reflection_minibatch_size,
                    acceptance_criterion=acceptance_criterion,
                    fewshot_max_candidate_sets=fewshot_max_candidate_sets,
                    preflight_plan_sha256=preflight_plan_sha256,
                    current_release=current_release,
                    parent_release_id=parent)
            except Exception as exc:
                _record_failure(report, path, "combined_factorial", exc, base_cfg)
                raise
            write_json(path, report)
    finally:
        if not injected_round_runner:
            from src.runtime.call_ledger import summarize_ledger
            report["request_budget"]["ledger_snapshot"] = summarize_ledger(
                ledger_path)
            write_json(path, report)
        for name, value in previous_execution_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    report["status"] = "complete"
    report["report_path"] = path.as_posix()
    write_json(path, report)
    if execution_guard_path is not None:
        write_json(execution_guard_path, {
            "plan_sha256": preflight_plan_sha256,
            "run_id": run_id,
            "status": "completed",
            "report_path": path.as_posix(),
        })
    return report


def _record_failure(report: Dict[str, Any], path: Path, phase: str,
                    error: Exception, cfg: Dict[str, Any]) -> None:
    """Persist partial progress and a redacted failure reason before propagating."""
    message = str(error)[:1000]
    for model_name in ("task", "reflection"):
        model = (cfg.get("models") or {}).get(model_name) or {}
        secret = os.environ.get(str(model.get("api_key_env") or ""))
        if secret:
            message = message.replace(secret, "[REDACTED]")
    report.update({
        "status": "failed",
        "failed_phase": phase,
        "error_type": type(error).__name__,
        "error": message,
        "report_path": path.as_posix(),
    })
    write_json(path, report)


def _require_isolated_home() -> None:
    if (not os.environ.get("RSI_HOME") or
            artifacts_root().resolve() == (repo_root() / "artifacts").resolve()):
        raise ValueError("factorial runs require RSI_HOME pointing to an isolated artifact directory")


def _required_api_envs(cfg: Dict[str, Any]) -> list[str]:
    return [name for name in _configured_api_envs(cfg)
            if not os.environ.get(name)]


def _configured_api_envs(cfg: Dict[str, Any]) -> list[str]:
    names = []
    for name in ("task", "reflection"):
        model = (cfg.get("models") or {}).get(name) or {}
        if str(model.get("provider") or "http").lower() == "http":
            key_env = str(model.get("api_key_env") or "").strip()
            if key_env:
                names.append(key_env)
    return sorted(set(names))


def _dotenv_only_api_envs(cfg: Dict[str, Any]) -> list[str]:
    """Configured live credentials absent at launch but later loaded from `.env`."""
    return [name for name in _configured_api_envs(cfg)
            if name not in _PROCESS_ENV_AT_START and os.environ.get(name)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--datasets", nargs="+", default=["pubmedqa"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[20261003, 20261004, 20261005])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0,
                        help="start index in the dev_optimize split used for search and optimize evaluation")
    parser.add_argument("--selection-gate-offset", type=int, default=0,
                        help="start index in the train-derived selection_gate split")
    parser.add_argument("--skill", default="litqa-literature-synthesis")
    parser.add_argument("--gepa-max-metric-calls", type=int,
                        default=DEFAULT_GEPA_MAX_METRIC_CALLS)
    parser.add_argument("--gepa-max-candidate-proposals", type=int,
                        default=DEFAULT_GEPA_MAX_CANDIDATE_PROPOSALS)
    parser.add_argument("--gepa-reflection-minibatch-size", type=int,
                        default=DEFAULT_GEPA_REFLECTION_MINIBATCH_SIZE)
    parser.add_argument("--gepa-acceptance-criterion",
                        choices=GEPA_ACCEPTANCE_CRITERIA,
                        default="improvement_or_equal",
                        help="GEPA proposal acceptance; recorded in the round protocol fingerprint")
    parser.add_argument("--fewshot-max-candidate-sets", type=int,
                        default=DEFAULT_MAX_DEMO_CANDIDATE_SETS,
                        help="number of ordered demo programs to score in the joint arm")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS),
                        help="arms to run; a subset is reported as a targeted follow-up")
    parser.add_argument("--validate-only", action="store_true",
                        help="freeze and validate cohort, split, arm fingerprints and budgets; makes zero model calls")
    parser.add_argument("--preflight-plan-sha256",
                        help="required for execution; must match the current zero-call preflight plan")
    args = parser.parse_args()
    _require_isolated_home()
    base = load_config(args.config)
    plan = build_factorial_preflight(
        base, args.datasets, args.seeds, args.limit,
        task_offset=args.offset,
        selection_gate_offset=args.selection_gate_offset,
        max_metric_calls=args.gepa_max_metric_calls,
        max_candidate_proposals=args.gepa_max_candidate_proposals,
        reflection_minibatch_size=args.gepa_reflection_minibatch_size,
        acceptance_criterion=args.gepa_acceptance_criterion,
        fewshot_max_candidate_sets=args.fewshot_max_candidate_sets,
        arms=args.arms,
        parent_release_id=release.current_release_id())
    if args.validate_only:
        from src.evolution.gepa_adapter import require_gepa
        require_gepa()
        plan_path = artifacts_path(
            "probe", "task_input_factorial",
            "preflight-" + plan["plan_sha256"][:16] + ".json")
        write_json(plan_path, plan)
        print(json.dumps({"status": plan["status"],
                          "plan_sha256": plan["plan_sha256"],
                          "plan_path": str(plan_path),
                          "model_calls": 0,
                          "datasets": plan["datasets"],
                          "arms": plan["arms"],
                          "request_budget": plan["request_budget"]},
                         ensure_ascii=False, indent=2))
        return 0
    if not args.preflight_plan_sha256:
        raise ValueError("real factorial execution requires --preflight-plan-sha256 from a reviewed zero-call plan")
    if args.preflight_plan_sha256 != plan["plan_sha256"]:
        raise ValueError("preflight plan hash mismatch; cohort, config, release, or budget changed")
    missing = _required_api_envs(base)
    if missing:
        raise ValueError("missing configured model credentials: " + ", ".join(missing))
    dotenv_only = _dotenv_only_api_envs(base)
    if dotenv_only:
        raise ValueError(
            "live factorial requires rotated model credentials explicitly set in "
            "the process environment before launch; refusing `.env`-only keys: "
            + ", ".join(dotenv_only))
    report = run_factorial(
        base, args.datasets, args.seeds, args.limit, args.skill,
        task_offset=args.offset,
        selection_gate_offset=args.selection_gate_offset,
        max_metric_calls=args.gepa_max_metric_calls,
        max_candidate_proposals=args.gepa_max_candidate_proposals,
        reflection_minibatch_size=args.gepa_reflection_minibatch_size,
        acceptance_criterion=args.gepa_acceptance_criterion,
        fewshot_max_candidate_sets=args.fewshot_max_candidate_sets,
        arms=args.arms,
        preflight_plan_sha256=plan["plan_sha256"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
