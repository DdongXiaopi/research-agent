"""Visible-dev Skill-body mechanism probe with fixed task input and no retrieval."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_fingerprint, load_config  # noqa: E402
from src.data.split import load_tasks  # noqa: E402
from src.eval.options import inject_all  # noqa: E402
from src.eval.stats import (holm_bonferroni, mcnemar_exact,
                            paired_bootstrap_binary)  # noqa: E402
from src.eval.task_rows import validate_task_answer_row  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.evolution.counterfactual import describe_intervention  # noqa: E402
from src.runtime.provider import build_provider  # noqa: E402
from src.runtime.runner import ArtifactState, TaskRunner  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.util import artifacts_path, read_json, sha256_obj, write_json  # noqa: E402


PROTOCOL_VERSION = "task-input-skill-route-and-body-ablation-v17-paired-negative-control"
SAME_STATE_REPEAT_COUNT = 2
INTERVENTION_REPEAT_COUNT = 2
STATISTICAL_MIN_PAIRED_TASKS = 8
FAMILYWISE_ALPHA = 0.05
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEEDS = {"body": 2026101501, "route": 2026101502}
FOLLOWUP_ARMS = (
    "same_state_repeat_1",
    "same_state_repeat_2",
    "bodyless_repeat_1",
    "bodyless_repeat_2",
    "rerouted_repeat_1",
    "rerouted_repeat_2",
)
MEASURED_ARMS = ("baseline_measurement", *FOLLOWUP_ARMS)


def prepare_config(config_path: str) -> dict:
    cfg = copy.deepcopy(load_config(config_path))
    cfg.setdefault("runtime", {})["input_mode"] = "task_input"
    cfg["runtime"].setdefault("task_input_max_chars", 30000)
    cfg["runtime"]["http_no_cache_required"] = True
    cfg.setdefault("evaluation", {})["sealed_pairing"] = False
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg["tools"].get("active", []) if name != "academic.search"]
    return cfg


def build_plan(cfg: dict, state, tasks: list[dict], *, dataset: str, seed: int,
               offset: int, limit: int, logical_budget: int = 0,
               physical_budget: int = 0) -> dict:
    """Freeze the visible cohort, runtime identity, and conservative request ceilings."""
    provider_kind = str(cfg.get("runtime", {}).get("provider", "fake")).lower()
    if provider_kind != "http":
        raise ValueError("counterfactual request ceilings currently require the HTTP provider")
    if cfg.get("runtime", {}).get("http_no_cache_required") is not True:
        raise ValueError("counterfactual A/A and intervention arms require HTTP cache disabled")
    retries = int(cfg.get("models", {}).get("task", {}).get("max_retries", 6))
    if retries < 1:
        raise ValueError("HTTP max_retries must be positive")
    max_model_calls = int(cfg.get("runtime", {}).get("max_model_calls", 12))
    if max_model_calls < 1:
        raise ValueError("runtime.max_model_calls must be positive")
    routing_mode = str(cfg.get("skills", {}).get("routing_mode", "lexical"))
    if routing_mode not in {"lexical", "model"}:
        raise ValueError("skills.routing_mode must be lexical or model")
    pinned_skill = str(cfg.get("skills", {}).get("pinned_skill", "") or "")
    if pinned_skill and state.registry.get(pinned_skill) is None:
        raise ValueError("configured pinned Skill is absent from the frozen registry")
    if not tasks:
        raise ValueError("counterfactual cohort is empty")

    # A setup call discovers the recalled Skill but is excluded from scoring.
    # Seven measured arms (baseline, A/A repeats, and repeated component
    # interventions) then run in a frozen randomized order. Each TaskRunner has
    # one answer call and, at most, one model-routing call:
    # retrieval/query planners are disabled by this protocol, and citation.verify
    # is a non-model local event. The HTTP router shares the task provider.
    arm_count = 1 + len(MEASURED_ARMS)
    model_route_possible = (
        routing_mode == "model" and not pinned_skill
        and bool(state.registry.skills))
    operations_per_task_runner = 1 + int(model_route_possible)
    operations_per_task_runner = min(operations_per_task_runner, max_model_calls)
    logical_upper = len(tasks) * arm_count * operations_per_task_runner
    physical_upper = logical_upper * retries
    logical_budget = logical_budget or logical_upper
    physical_budget = physical_budget or physical_upper
    if logical_budget < logical_upper:
        raise ValueError("logical request budget is below the frozen conservative ceiling")
    if physical_budget < physical_upper:
        raise ValueError("physical request budget is below the configured retry ceiling")

    identity = build_protocol_identity(
        cfg, dataset=dataset, seed=seed, offset=offset, limit=limit,
        task_ids=[str(task["task_id"]) for task in tasks])
    plan = {
        "schema_version": 1,
        "protocol": PROTOCOL_VERSION,
        **identity,
        "release_id": str(state.release_id),
        "parent_system_prompt_sha256": sha256_obj(state.system_prompt),
        "registry_sha256": state.registry.fingerprint(),
        "parent_task_demos_sha256": sha256_obj(state.task_demos),
        # Question/options/gold are part of the evaluated protocol too. Hash
        # each complete frozen row instead of only IDs and evidence so a data
        # refresh cannot silently reuse an otherwise identical plan.
        "task_material_sha256": sha256_obj(tasks),
        "provider": provider_kind,
        "http_response_cache_disabled_for_run": True,
        "provider_model_id": str(cfg.get("models", {}).get("task", {}).get("model_id", "")),
        "task_input_mode": str(cfg.get("runtime", {}).get("input_mode", "")),
        "retrieval_used": False,
        "academic_search_enabled": "academic.search" in cfg.get("tools", {}).get("active", []),
        "active_tools": list(cfg.get("tools", {}).get("active", [])),
        "routing_mode": routing_mode,
        "pinned_skill": pinned_skill or None,
        "registry_skill_count": len(state.registry.skills),
        "max_model_calls_per_task_runner": max_model_calls,
        "maximum_model_operations_per_task_runner": operations_per_task_runner,
        "http_max_attempts_per_operation": retries,
        "arm_count_upper_bound": arm_count,
        "unscored_route_setup_count_per_task": 1,
        "measured_arm_count_per_task": len(MEASURED_ARMS),
        "intervention_repeat_count": INTERVENTION_REPEAT_COUNT,
        "maximum_logical_operations": logical_upper,
        "logical_budget": logical_budget,
        "maximum_physical_attempts": physical_upper,
        "physical_budget": physical_budget,
        "selection_gate_read": False,
        "sealed_or_final_data_read": False,
        "experiment_role": "exploratory_mechanism_diagnostic",
        "candidate_selection": False,
        "candidate_promotion": False,
        "independent_confirmation_eligible": False,
        "model_calls_during_preflight": 0,
    }
    plan["plan_sha256"] = sha256_obj(plan)
    return plan


def verify_frozen_plan(plan: dict, *, expected_sha256: str | None,
                       validate_only: bool, logical_budget: int,
                       physical_budget: int) -> None:
    if logical_budget and logical_budget != plan["logical_budget"]:
        raise ValueError("logical budget differs from the frozen plan")
    if physical_budget and physical_budget != plan["physical_budget"]:
        raise ValueError("physical budget differs from the frozen plan")
    if validate_only:
        return
    if not expected_sha256:
        raise ValueError("execution requires --expected-plan-sha256 from a prior preflight")
    if expected_sha256 != plan["plan_sha256"]:
        raise ValueError("expected plan SHA-256 does not match the recomputed frozen plan")
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("counterfactual execution requires an isolated RSI_HOME")


def build_protocol_identity(cfg: dict, *, dataset: str, seed: int,
                            offset: int, limit: int,
                            task_ids: list[str]) -> dict:
    """Bind the report to code, runtime config, task scope, and arm semantics."""
    identity = {
        "schema": PROTOCOL_VERSION,
        "config_sha256": config_fingerprint(cfg),
        "dataset": str(dataset),
        "split": "dev_optimize",
        "task_ids": [str(task_id) for task_id in task_ids],
        "seed": int(seed),
        "seed_semantics": (
            "execution label and deterministic per-task arm-order input; "
            "provider randomness is not assumed controlled"),
        "offset": int(offset),
        "limit": int(limit),
        "arms": [
            "unscored_route_setup", *MEASURED_ARMS,
        ],
        "arm_order_policy": (
            "one unscored route-setup call precedes measurement to freeze ablation target; "
            "all seven planned measured-arm slots are seed-keyed and position-balanced "
            "across tasks; eligibility skips can unbalance the executed subset, "
            "whose positions are reported separately; task execution order is "
            "separately seed-keyed; intervention contrasts count only when the "
            "scored baseline selected the same Skill as route setup; A/A stability "
            "also requires the same selected Skill route; exact same-state repeats are the "
            "negative-control placebo because paraphrasing can itself change LLM behavior"),
        "task_execution_order": randomized_task_order(task_ids, seed),
        "task_arm_orders": [
            {"task_id": str(task_id),
             "setup": "unscored_route_setup", "arms": measured_order}
            for task_id, measured_order in balanced_measured_arm_orders(
                task_ids, seed).items()
        ],
            "intervention_policy": (
                "skip_noop_or_multicomponent_arms_before_task_evaluation; "
            "skip_description_ablation_for_pinned_skill; "
            "two_same_state_repeats; two_repetitions_per_eligible_single_component_intervention; "
            "exclude_intervention_effects_if_route_setup_target_differs_from_scored_baseline; "
            "route_setup_answer_is_not_scored; randomized_all_measured_arm_order; "
                 "report_all_three_pairwise_AA_answer_and_skill_route_comparisons; "
                 "same_state_stability_requires_answer_status_and_selected_skill_agreement; "
                 "body_contrast_requires_stable_unchanged_skill_route; "
                 "route_contrast_requires_stable_changed_skill_route; "
                 "two_co_primary_component_hypotheses_use_exact_McNemar_with_Holm_FWER_0.05; "
                 "minimum_eight_stable_paired_tasks_and_positive_bootstrap_CI; "
                 "repeatable_contrasts_are_not_general_causal_attribution"),
    }
    return {**identity, "protocol_sha256": sha256_obj(identity)}


def _order_digest(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(
        f"{int(seed)}\0{namespace}\0{value}".encode("utf-8")).hexdigest()


def randomized_task_order(task_ids: list[str], seed: int) -> list[str]:
    """Freeze a reproducible task order independent of normalized-file order."""
    normalized = [str(task_id) for task_id in task_ids]
    if not normalized or any(not task_id for task_id in normalized) \
            or len(set(normalized)) != len(normalized):
        raise ValueError("task-order IDs must be non-empty and unique")
    return sorted(normalized,
                  key=lambda task_id: _order_digest(
                      seed, "task_execution_order", task_id))


def balanced_measured_arm_orders(task_ids: list[str], seed: int) -> dict[str, list[str]]:
    """Seed-keyed block allocation balances all measured arm positions."""
    normalized = [str(task_id) for task_id in task_ids]
    if not normalized or any(not task_id for task_id in normalized) \
            or len(set(normalized)) != len(normalized):
        raise ValueError("arm-order task IDs must be non-empty and unique")
    base_order = sorted(
        MEASURED_ARMS,
        key=lambda arm: _order_digest(seed, "arm", arm),
    )
    randomized_tasks = sorted(
        normalized,
        key=lambda task_id: _order_digest(seed, "task", task_id),
    )
    orders = {}
    for position, task_id in enumerate(randomized_tasks):
        shift = position % len(base_order)
        orders[task_id] = base_order[shift:] + base_order[:shift]
    return {task_id: orders[task_id] for task_id in normalized}


def intervention_arm_status(intervention: dict) -> str:
    """Only evaluate a changed, single-component arm in this diagnostic."""
    if not intervention.get("state_changed"):
        return "noop_skipped"
    if not intervention.get("single_component_intervention"):
        return "multicomponent_skipped"
    return "evaluated"


def align_intervention_target_to_baseline(
        status: str, route_setup_skill: str | None,
        baseline_selected_skill: str | None) -> str:
    """Exclude an ablation when route setup chose a different baseline Skill."""
    if status != "evaluated":
        return status
    if (not route_setup_skill or not baseline_selected_skill
            or route_setup_skill != baseline_selected_skill):
        return "route_target_mismatch"
    return "evaluated"


def route_ablation_applicable(cfg: dict, skill_name: str) -> bool:
    """A description ablation cannot change routing for a pinned Skill."""
    pinned = str((cfg.get("skills") or {}).get("pinned_skill") or "")
    return not pinned or pinned != str(skill_name)


def arm_position_counts(rows: list[dict], order_field: str) -> dict[str, list[int]]:
    """Count each arm's positions in scheduled or actually executed sequences."""
    counts = {arm: [0] * len(MEASURED_ARMS) for arm in MEASURED_ARMS}
    for row in rows:
        order = row.get(order_field)
        if not isinstance(order, list):
            raise ValueError(f"counterfactual row lacks {order_field}")
        if len(order) > len(MEASURED_ARMS) or len(set(order)) != len(order) or any(
                arm not in counts for arm in order):
            raise ValueError(f"counterfactual row has invalid {order_field}")
        for position, arm in enumerate(order):
            counts[arm][position] += 1
    return counts


def summarize_intervention_effects(rows: list[dict], arm: str) -> dict:
    """Summarize effects and frozen paired statistics for one component arm."""
    effect_key = f"{arm}_effect"
    status_key = f"{arm}_status"
    evaluated = [row for row in rows if row.get(status_key) == "evaluated"]
    observed = [row for row in evaluated if row.get(effect_key) is True]
    rescues = [row for row in evaluated
               if row.get(f"{arm}_contrast", {}).get(
                   "repeatable_valid_wrong_rescue") is True]
    harms = [row for row in evaluated
             if row.get(f"{arm}_contrast", {}).get(
                 "repeatable_valid_correct_harm") is True]
    repetitions_key = "bodyless_repeats" if arm == "body" else "rerouted_repeats"
    paired_rows = []
    for row in evaluated:
        contrast = row.get(f"{arm}_contrast") or {}
        baseline = row.get("base") or {}
        repetitions = row.get(repetitions_key) or []
        if (row.get("aa_stable") is True
                and contrast.get("baseline_stable") is True
                and contrast.get("intervention_stable") is True
                and contrast.get("route_compliant") is True
                and type(baseline.get("correct")) is bool
                and repetitions
                and type(repetitions[0].get("correct")) is bool):
            paired_rows.append((int(baseline["correct"]),
                                int(repetitions[0]["correct"])))
    before = [row[0] for row in paired_rows]
    after = [row[1] for row in paired_rows]
    exact = mcnemar_exact(before, after)
    bootstrap = paired_bootstrap_binary(
        before, after, B=BOOTSTRAP_REPLICATES,
        rng_seed=BOOTSTRAP_SEEDS[arm], alpha=FAMILYWISE_ALPHA)
    return {
        "n_evaluated": len(evaluated),
        "n_noop_skipped": sum(row.get(status_key) == "noop_skipped" for row in rows),
        "n_multicomponent_skipped": sum(
            row.get(status_key) == "multicomponent_skipped" for row in rows),
        "n_route_target_mismatches": sum(
            row.get(status_key) == "route_target_mismatch" for row in rows),
        "n_not_applicable": sum(row.get(status_key) not in {
            "evaluated", "noop_skipped", "multicomponent_skipped"} for row in rows),
        "n_effect_observed": len(observed),
        "n_effect_observed_with_aa_stable": sum(
            row.get("aa_stable") is True for row in observed),
        "n_effect_observed_with_aa_discordant": sum(
            row.get("aa_stable") is False for row in observed),
        "n_repeatable_valid_wrong_rescues": len(rescues),
        "n_repeatable_valid_correct_harms": len(harms),
        "n_stable_paired_correctness": len(paired_rows),
        "n_excluded_from_paired_statistics": len(evaluated) - len(paired_rows),
        "paired_correctness": {
            "before_correct": sum(before), "after_correct": sum(after),
            "delta_correct": sum(after) - sum(before),
            "exact_mcnemar": exact,
            "paired_bootstrap_effect": bootstrap,
        },
        "scoped_component_effect_supported": False,
        "causal_attribution_supported": False,
    }


def apply_familywise_component_decision(summaries: dict[str, dict]) -> dict:
    """Apply the frozen two-hypothesis decision without broad causal claims."""
    arms = ["body", "route"]
    adjusted = holm_bonferroni([
        summaries[arm]["paired_correctness"]["exact_mcnemar"]["p_value"]
        for arm in arms
    ], alpha=FAMILYWISE_ALPHA)
    decisions = {}
    for index, arm in enumerate(arms):
        summary = summaries[arm]
        paired = summary["paired_correctness"]
        exact = paired["exact_mcnemar"]
        bootstrap = paired["paired_bootstrap_effect"]
        enough = summary["n_stable_paired_correctness"] >= STATISTICAL_MIN_PAIRED_TASKS
        all_stable = (summary["n_stable_paired_correctness"]
                      == summary["n_evaluated"])
        positive = (paired["delta_correct"] > 0
                    and exact["n_improved"] > exact["n_regressed"])
        interval_positive = bootstrap.get("ci_low_pp", 0.0) > 0.0
        supported = bool(enough and all_stable and positive and interval_positive
                         and adjusted["reject"][index])
        if not enough:
            status = "insufficient_stable_paired_tasks"
        elif not all_stable:
            status = "negative_control_or_intervention_instability"
        elif not positive:
            status = "no_positive_component_effect"
        elif not interval_positive or not adjusted["reject"][index]:
            status = "statistically_inconclusive"
        else:
            status = "supported_on_frozen_visible_cohort"
        summary["familywise_adjusted_p_value"] = adjusted["adjusted"][index]
        summary["familywise_reject_null"] = adjusted["reject"][index]
        summary["scoped_component_effect_supported"] = supported
        summary["evidence_status"] = status
        decisions[arm] = {
            "status": status,
            "scoped_component_effect_supported": supported,
            "general_causal_attribution_supported": False,
        }
    return {
        "method": adjusted["method"], "alpha": FAMILYWISE_ALPHA,
        "hypotheses": arms, "adjusted_p_values": adjusted["adjusted"],
        "reject": adjusted["reject"],
        "minimum_stable_paired_tasks_per_component": STATISTICAL_MIN_PAIRED_TASKS,
        "decisions": decisions,
        "scope": (
            "prospective component effects on this frozen visible cohort only; "
            "no general causal attribution or candidate-selection claim"),
    }


def summarize_aa_pairwise(base, repeats: list) -> dict:
    """Compare both answers and selected Skill routes across same-state runs."""
    outcomes = [base, *repeats]
    missing_route = object()
    selected_skills = [getattr(outcome, "skill_recalled", missing_route)
                       for outcome in outcomes]
    answer_agreements = []
    route_agreements = []
    for left_index in range(len(outcomes)):
        for right_index in range(left_index + 1, len(outcomes)):
            left, right = outcomes[left_index], outcomes[right_index]
            answer_agreements.append(
                (left.answer_index, left.status) ==
                (right.answer_index, right.status))
            left_skill = selected_skills[left_index]
            right_skill = selected_skills[right_index]
            route_agreements.append(
                left_skill is not missing_route and right_skill is not missing_route
                and left_skill == right_skill)
    answer_pairwise_agreements = sum(answer_agreements)
    route_pairwise_agreements = sum(route_agreements)
    answer_baseline_agreements = sum(
        (base.answer_index, base.status) ==
        (repeat.answer_index, repeat.status)
        for repeat in repeats)
    route_baseline_agreements = sum(
        selected_skills[0] is not missing_route
        and selected_skills[index + 1] is not missing_route
        and selected_skills[0] == selected_skills[index + 1]
        for index, _repeat in enumerate(repeats))
    return {
        "baseline_repeat_comparisons": len(repeats),
        "baseline_repeat_agreements": answer_baseline_agreements,
        "baseline_repeat_route_agreements": route_baseline_agreements,
        "pairwise_comparisons": len(answer_agreements),
        "pairwise_agreements": answer_pairwise_agreements,
        "pairwise_disagreements": len(answer_agreements) - answer_pairwise_agreements,
        "route_pairwise_comparisons": len(route_agreements),
        "route_pairwise_agreements": route_pairwise_agreements,
        "route_pairwise_disagreements": len(route_agreements) - route_pairwise_agreements,
        "all_answer_pairwise_agree": all(answer_agreements),
        "all_route_pairwise_agree": all(route_agreements),
        "all_pairwise_agree": (
            all(answer_agreements) and all(route_agreements)),
    }


def summarize_repeated_intervention_contrast(
        baseline: list[dict], intervention: list[dict], *,
        single_component_intervention: bool,
        intervention_kind: str | None = None) -> dict:
    """Report repeatable valid-answer contrasts without asserting causality."""
    def contract_valid(rows: list[dict]) -> bool:
        return bool(rows) and all(
            isinstance(row, dict) and row.get("answer_contract_valid") is True
            and row.get("status") in {"ok", "abstain"}
            and type(row.get("answer_index")) is int
            and type(row.get("correct")) is bool
            for row in rows)

    def signature(row: dict) -> tuple:
        return row.get("status"), row.get("answer_index"), row.get("correct")

    baseline_valid = contract_valid(baseline)
    intervention_valid = contract_valid(intervention)
    baseline_stable = (
        baseline_valid and len({signature(row) for row in baseline}) == 1)
    intervention_stable = (
        intervention_valid and len({signature(row) for row in intervention}) == 1)
    route_values_valid = (
        intervention_kind in {"body", "route"}
        and all(isinstance(row.get("skill_recalled"), str)
                for row in [*baseline, *intervention]))
    baseline_routes = [row.get("skill_recalled") for row in baseline]
    intervention_routes = [row.get("skill_recalled") for row in intervention]
    baseline_route_stable = (
        bool(baseline_routes) and route_values_valid
        and len(set(baseline_routes)) == 1)
    intervention_route_stable = (
        bool(intervention_routes) and route_values_valid
        and len(set(intervention_routes)) == 1)
    route_relation_observed = None
    route_compliant = True
    if intervention_kind in {"body", "route"}:
        route_relation_observed = bool(
            baseline_route_stable and intervention_route_stable
            and ((baseline_routes[0] == intervention_routes[0])
                 if intervention_kind == "body"
                 else (baseline_routes[0] != intervention_routes[0])))
        route_compliant = route_relation_observed
    stable_wrong_baseline = (
        baseline_stable and all(row["status"] == "ok" and not row["correct"]
                                for row in baseline))
    stable_correct_baseline = (
        baseline_stable and all(row["status"] == "ok" and row["correct"]
                                for row in baseline))
    stable_correct_intervention = (
        intervention_stable and all(row["status"] == "ok" and row["correct"]
                                    for row in intervention))
    stable_wrong_intervention = (
        intervention_stable and all(row["status"] == "ok" and not row["correct"]
                                    for row in intervention))
    repeatable_rescue = bool(
        single_component_intervention and route_compliant and stable_wrong_baseline
        and stable_correct_intervention)
    repeatable_harm = bool(
        single_component_intervention and route_compliant and stable_correct_baseline
        and stable_wrong_intervention)
    response_changed = bool(
        baseline_stable and intervention_stable and route_compliant
        and signature(baseline[0]) != signature(intervention[0]))
    return {
        "baseline_repetitions": len(baseline),
        "intervention_repetitions": len(intervention),
        "baseline_contract_valid": baseline_valid,
        "intervention_contract_valid": intervention_valid,
        "baseline_stable": bool(baseline_stable),
        "intervention_stable": bool(intervention_stable),
        "baseline_route_stable": bool(baseline_route_stable),
        "intervention_route_stable": bool(intervention_route_stable),
        "intervention_kind": intervention_kind,
        "route_relation_observed": route_relation_observed,
        "route_compliant": bool(route_compliant),
        "repeatable_response_change": response_changed,
        "repeatable_valid_wrong_rescue": repeatable_rescue,
        "repeatable_valid_correct_harm": repeatable_harm,
        "single_component_intervention": bool(single_component_intervention),
        "causal_attribution_supported": False,
        "interpretation": (
            "repeatable_single_component_contrast_not_causal"
            if response_changed and single_component_intervention
            else "descriptive_only"),
    }


def task_outcome_observation(outcome, task: dict, *, arm: str) -> dict | None:
    """Validate the scored answer against the frozen task before comparing arms."""
    if outcome is None:
        return None
    raw = outcome.to_dict()
    try:
        expected_identity = (
            str(task.get("task_id") or ""),
            str(task.get("dataset") or ""),
            str(task.get("split") or "dev_optimize"),
        )
        observed_identity = (
            str(raw.get("task_id") or ""),
            str(raw.get("dataset") or ""),
            str(raw.get("split") or ""),
        )
        if observed_identity != expected_identity:
            raise ValueError("outcome dataset/split/task identity differs from frozen task")
        checked = validate_task_answer_row(
            raw, task, context=f"skill counterfactual {arm}:{task['task_id']}",
            require_reported_gold_index=True)
        valid = True
    except (TypeError, ValueError, AttributeError):
        checked = raw
        valid = False
    return {
        "answer_index": checked.get("answer_index"),
        "dataset": checked.get("dataset"),
        "split": checked.get("split"),
        "status": checked.get("status"),
        "correct": checked.get("correct"),
        "answer_contract_valid": valid,
        "skill_recalled": outcome.skill_recalled,
        "skill_behaviors": outcome.skill_behaviors,
        "error": outcome.error,
        "searched": any(e.get("tool_name") == "academic.search"
                         for e in outcome.events),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--dataset", default="pubmedqa")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--logical-call-budget", type=int, default=0,
                        help="defaults to the conservative frozen logical ceiling")
    parser.add_argument("--physical-call-budget", type=int, default=0,
                        help="defaults to the configured HTTP retry ceiling")
    parser.add_argument("--validate-only", action="store_true",
                        help="freeze the plan and request ceilings without provider calls")
    parser.add_argument("--expected-plan-sha256", default="",
                        help="required to execute; copy from a prior --validate-only run")
    args = parser.parse_args()
    if args.limit < 1 or args.offset < 0:
        raise ValueError("limit must be positive and offset nonnegative")
    if args.logical_call_budget < 0 or args.physical_call_budget < 0:
        raise ValueError("request budgets must be nonnegative")

    cfg = prepare_config(args.config)
    tasks = sorted(load_tasks(args.dataset, splits=["dev_optimize"]),
                   key=lambda task: task["task_id"])[args.offset:args.offset + args.limit]
    if cfg["evaluation"].get("insufficient_option", True):
        tasks = inject_all(tasks)
    state = load_state()

    plan = build_plan(
        cfg, state, tasks, dataset=args.dataset, seed=args.seed,
        offset=args.offset, limit=args.limit,
        logical_budget=args.logical_call_budget,
        physical_budget=args.physical_call_budget)
    verify_frozen_plan(
        plan, expected_sha256=args.expected_plan_sha256,
        validate_only=args.validate_only,
        logical_budget=args.logical_call_budget,
        physical_budget=args.physical_call_budget)
    plan_path = artifacts_path(
        "probe", "skill_counterfactual", f"plan-{plan['plan_sha256']}.json")
    if plan_path.exists():
        saved_plan = read_json(plan_path)
        if saved_plan != plan:
            raise ValueError("saved skill counterfactual plan differs from recomputed plan")
    else:
        write_json(plan_path, plan)

    if args.validate_only:
        print(json.dumps({
            "status": "validated_not_run", "plan": str(plan_path),
            "plan_sha256": plan["plan_sha256"], "tasks": len(tasks),
            "logical_call_upper_bound": plan["maximum_logical_operations"],
            "logical_call_budget": plan["logical_budget"],
            "physical_attempt_upper_bound": plan["maximum_physical_attempts"],
            "physical_call_budget": plan["physical_budget"],
            "model_calls": 0, "selection_gate_read": False,
            "sealed_or_final_data_read": False}, ensure_ascii=False))
        return 0

    ledger_path = artifacts_path(
        "probe", "skill_counterfactual", f"ledger-{plan['plan_sha256']}.jsonl")
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(plan["plan_sha256"])
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path)
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(plan["logical_budget"])
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(plan["physical_budget"])
    provider = build_provider(cfg, seed=args.seed)
    if hasattr(provider, "cache"):
        provider.cache = False

    rows = []
    measured_orders = {
        str(row["task_id"]): list(row["arms"])
        for row in plan["task_arm_orders"]
    }
    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    for task_id in plan["task_execution_order"]:
        task = tasks_by_id[str(task_id)]
        route_setup = TaskRunner(cfg, state, provider=provider, seed=args.seed).run(task)
        measured_order = measured_orders[str(task["task_id"])]
        skill_name = route_setup.skill_recalled
        if skill_name:
            registry = copy.deepcopy(state.registry)
            skill = registry.get(skill_name)
            if skill is None:
                raise RuntimeError(f"recalled Skill missing from registry: {skill_name}")
            skill.body = ""
            skill.body_path = ""
            bodyless_state = ArtifactState(
                state.release_id, state.system_prompt, registry,
                task_demos=copy.deepcopy(state.task_demos))
            body_intervention = describe_intervention(state, bodyless_state)
            body_status = intervention_arm_status(body_intervention)

            if route_ablation_applicable(cfg, skill_name):
                route_registry = copy.deepcopy(state.registry)
                route_skill = route_registry.get(skill_name)
                route_skill.description = ""
                route_skill.triggers = []
                rerouted_state = ArtifactState(
                    state.release_id, state.system_prompt, route_registry,
                    task_demos=copy.deepcopy(state.task_demos))
                route_intervention = describe_intervention(state, rerouted_state)
                route_status = intervention_arm_status(route_intervention)
            else:
                route_intervention = None
                route_status = "pinned_skill_not_ablated"
        else:
            body_intervention = None
            route_intervention = None
            body_status = "skill_not_selected"
            route_status = "skill_not_selected"

        arm_actions = {
            "baseline_measurement": lambda: TaskRunner(
                cfg, state, provider=provider, seed=args.seed).run(task),
            "same_state_repeat_1": lambda: TaskRunner(
                cfg, state, provider=provider, seed=args.seed).run(task),
            "same_state_repeat_2": lambda: TaskRunner(
                cfg, state, provider=provider, seed=args.seed).run(task),
            "bodyless_repeat_1": (lambda: TaskRunner(
                cfg, bodyless_state, provider=provider, seed=args.seed).run(task)
                         if body_status == "evaluated" else None),
            "bodyless_repeat_2": (lambda: TaskRunner(
                cfg, bodyless_state, provider=provider, seed=args.seed).run(task)
                         if body_status == "evaluated" else None),
            "rerouted_repeat_1": (lambda: TaskRunner(
                cfg, rerouted_state, provider=provider, seed=args.seed).run(task)
                         if route_status == "evaluated" else None),
            "rerouted_repeat_2": (lambda: TaskRunner(
                cfg, rerouted_state, provider=provider, seed=args.seed).run(task)
                         if route_status == "evaluated" else None),
        }
        arm_outcomes = {}
        executed_arm_order = []
        for arm_name in measured_order:
            arm_outcomes[arm_name] = arm_actions[arm_name]()
            if arm_outcomes[arm_name] is not None:
                executed_arm_order.append(arm_name)
        base = arm_outcomes["baseline_measurement"]
        body_status = align_intervention_target_to_baseline(
            body_status, skill_name, base.skill_recalled)
        route_status = align_intervention_target_to_baseline(
            route_status, skill_name, base.skill_recalled)
        repeat_outcomes = [arm_outcomes["same_state_repeat_1"],
                           arm_outcomes["same_state_repeat_2"]]

        aa_summary = summarize_aa_pairwise(base, repeat_outcomes)
        base_observation = task_outcome_observation(base, task, arm="baseline")
        repeat_observations = [task_outcome_observation(
            row, task, arm=f"same_state_repeat_{index + 1}")
            for index, row in enumerate(repeat_outcomes)]
        baseline_observations = [base_observation, *repeat_observations]
        body_observations = [task_outcome_observation(
            arm_outcomes[f"bodyless_repeat_{index + 1}"], task,
            arm=f"bodyless_repeat_{index + 1}")
            for index in range(INTERVENTION_REPEAT_COUNT)]
        route_observations = [task_outcome_observation(
            arm_outcomes[f"rerouted_repeat_{index + 1}"], task,
            arm=f"rerouted_repeat_{index + 1}")
            for index in range(INTERVENTION_REPEAT_COUNT)]
        body_contrast = summarize_repeated_intervention_contrast(
            baseline_observations, body_observations,
            single_component_intervention=bool(body_status == "evaluated"
                and body_intervention and body_intervention.get(
                    "single_component_intervention")),
            intervention_kind="body")
        route_contrast = summarize_repeated_intervention_contrast(
            baseline_observations, route_observations,
            single_component_intervention=bool(route_status == "evaluated"
                and route_intervention and route_intervention.get(
                    "single_component_intervention")),
            intervention_kind="route")

        rows.append({"task_id": task["task_id"],
                     "material_sha256": sha256_obj(task.get("evidence") or task.get("context") or ""),
                     "skill": skill_name, "base": base_observation,
                     "same_state_repeats": repeat_observations,
                     "bodyless": body_observations[0],
                     "bodyless_repeats": body_observations,
                     "rerouted": route_observations[0],
                     "rerouted_repeats": route_observations,
                     "body_contrast": body_contrast,
                     "route_contrast": route_contrast,
                     "route_setup_skill": skill_name,
                     "baseline_selected_skill": base.skill_recalled,
                     "intervention_target_matches_baseline": (
                         None if not skill_name else base.skill_recalled == skill_name),
                     "route_setup_used_for_scoring": False,
                     "planned_arm_order": measured_order,
                     "executed_arm_order": executed_arm_order,
                     "body_intervention": body_intervention,
                     "route_intervention": route_intervention,
                     "body_status": body_status,
                     "route_status": route_status,
                     **{f"aa_{key}": value for key, value in aa_summary.items()},
                     "aa_stable": aa_summary["all_pairwise_agree"],
                     "aa_answer_stable": aa_summary["all_answer_pairwise_agree"],
                     "aa_route_stable": aa_summary["all_route_pairwise_agree"],
                     "body_effect": (None if body_status != "evaluated" else
                                     body_contrast["repeatable_response_change"]),
                     "route_effect": (None if route_status != "evaluated" else
                                      route_contrast["repeatable_response_change"])})

    body_summary = summarize_intervention_effects(rows, "body")
    route_summary = summarize_intervention_effects(rows, "route")
    component_summaries = {"body": body_summary, "route": route_summary}
    familywise_decision = apply_familywise_component_decision(
        component_summaries)

    report = {"protocol": PROTOCOL_VERSION,
              "plan": plan, "plan_path": str(plan_path),
              "request_ledger": str(ledger_path),
              "request_accounting": summarize_ledger(ledger_path),
              "experiment_role": plan["experiment_role"],
              "attribution_is_causal": False,
              "negative_control_placebo": {
                  "type": "exact_same_state_repeat",
                  "repeat_count_per_task": SAME_STATE_REPEAT_COUNT,
                  "reason": (
                      "a semantic paraphrase is an active language intervention; "
                      "the exact frozen state is the valid no-change control"),
              },
              "candidate_selection": False,
              "candidate_promotion": False,
              "independent_confirmation_eligible": False,
              "retrieval_used": False,
              "release_id": state.release_id,
              "dataset": args.dataset, "split": "dev_optimize", "seed": args.seed,
              "offset": args.offset,
              "n": len(rows), "aa_stable": sum(r["aa_stable"] for r in rows),
              "aa_control_summary": {
                  "repeat_count_per_task": SAME_STATE_REPEAT_COUNT,
                  "n_baseline_repeat_comparisons": sum(
                      row["aa_baseline_repeat_comparisons"] for row in rows),
                  "n_baseline_repeat_agreements": sum(
                      row["aa_baseline_repeat_agreements"] for row in rows),
                  "n_baseline_repeat_route_comparisons": sum(
                      row["aa_baseline_repeat_comparisons"] for row in rows),
                  "n_baseline_repeat_route_agreements": sum(
                      row["aa_baseline_repeat_route_agreements"] for row in rows),
                  "n_pairwise_comparisons": sum(
                      row["aa_pairwise_comparisons"] for row in rows),
                  "n_pairwise_agreements": sum(
                      row["aa_pairwise_agreements"] for row in rows),
                  "n_pairwise_disagreements": sum(
                      row["aa_pairwise_disagreements"] for row in rows),
                  "n_tasks_with_all_pairwise_agree": sum(
                      row["aa_stable"] for row in rows),
                  "n_tasks_with_any_pairwise_disagreement": sum(
                      not row["aa_stable"] for row in rows),
                  "n_tasks_with_all_answer_pairwise_agree": sum(
                      row["aa_answer_stable"] for row in rows),
                  "n_tasks_with_all_route_pairwise_agree": sum(
                      row["aa_route_stable"] for row in rows),
                  "n_route_pairwise_comparisons": sum(
                      row["aa_route_pairwise_comparisons"] for row in rows),
                  "n_route_pairwise_agreements": sum(
                      row["aa_route_pairwise_agreements"] for row in rows),
                  "n_route_pairwise_disagreements": sum(
                      row["aa_route_pairwise_disagreements"] for row in rows),
              },
              "arm_position_counts": {
                  "planned_slots": arm_position_counts(rows, "planned_arm_order"),
                  "executed_sequences": arm_position_counts(rows, "executed_arm_order"),
                  "interpretation": (
                      "planned slots are balanced by protocol; executed-sequence "
                      "positions can differ when ineligible interventions are skipped"),
              },
              "body_effect": body_summary["n_effect_observed"],
              "route_effect": route_summary["n_effect_observed"],
              "intervention_effect_summaries": component_summaries,
              "prospective_component_evidence": familywise_decision,
              "interpretation": (
                  "pre_registered_paired_component_test; compare effects with "
                  "answer/status and selected-Skill-route disagreements across all "
                  "three pairwise same_state checks; a supported result is scoped "
                  "to the frozen visible cohort and is never candidate selection or "
                  "general causal attribution"),
              "rows": rows}
    out = artifacts_path("probe", "skill_counterfactual",
                         "probe-" + uuid.uuid4().hex[:12] + ".json")
    write_json(out, report)
    print(json.dumps({"path": str(out), "n": report["n"],
                      "aa_stable": report["aa_stable"],
                      "body_effect": report["body_effect"],
                      "route_effect": report["route_effect"],
                      "body_effects_with_aa_discordance": (
                          body_summary["n_effect_observed_with_aa_discordant"]),
                      "route_effects_with_aa_discordance": (
                          route_summary["n_effect_observed_with_aa_discordant"]),
                      "aa_pairwise_comparisons": (
                          report["aa_control_summary"]["n_pairwise_comparisons"])},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
