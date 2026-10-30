#!/usr/bin/env python3
"""Run an EI four-arm Prompt × Skill GEPA factorial on frozen article scopes.

Prompt and Skill are optimized independently on the same train/search scopes,
then both candidates are frozen and passed through static/grounding gates
before the untouched internal scope is used for the paired four-arm replay.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_pubmedqa_prompt_discovery import run_prompt_grounding_review  # noqa: E402
from scripts.run_evidence_inference_gepa_skill_search import (  # noqa: E402
    LABEL_SEMANTICS, TARGET_SKILL,
)
from scripts.run_evidence_inference_mipro_search import (  # noqa: E402
    DATASET, RELEASE_ID, _score_repeated_scope, class_aware_gain,
    load_frozen_cohort, metrics, paired_internal_analysis, prepare_config,
)
from src.config import config_fingerprint  # noqa: E402
from src.contracts.types import Candidate, content_sha256  # noqa: E402
from src.data.evidence_inference_v2 import load_tasks  # noqa: E402
from src.evolution.gepa_search import search_prompt_candidate, training_question_overlap  # noqa: E402
from src.evolution.gepa_skill_search import search_skill_body_candidate  # noqa: E402
from src.evolution.method_rule_gate import validate_statistical_method_rules  # noqa: E402
from src.evolution.task_input_gates import validate_task_input_prompt  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.call_ledger import call_component, summarize_ledger  # noqa: E402
from src.runtime.runner import ArtifactState  # noqa: E402
from src.runtime.skills import with_body  # noqa: E402
from src.util import (artifacts_path, sha256_file, sha256_obj, write_json)  # noqa: E402


def _prepare_config(config_path: str, cohort: dict[str, Any], train_ids: list[str],
                    search_ids: list[str], metric_calls: int, proposals: int,
                    minibatch: int, task_model: str,
                    reflection_model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    common = prepare_config(config_path, proposer_calls=proposals,
                            routing_mode="lexical", task_model_id=task_model)
    # Keep the project-configured six HTTP attempts for reflection transport;
    # the previous EI-specific override of two attempts exhausted on transient
    # proxy/SSL failures during cycle14 Skill search.
    common["models"]["reflection"].update({
        "model_id": reflection_model, "provider": "http", "max_retries": 6})
    common["models"]["task"]["max_retries"] = 2
    common.setdefault("runtime", {}).update({
        "provider": "pi", "input_mode": "task_input",
        "task_input_max_chars": 30000, "pi_timeout_max_attempts": 2,
        "http_no_cache_required": True,
    })
    common.setdefault("skills", {}).update({
        "routing_mode": "lexical", "router_provider": "http",
        "router_model_id": "Qwen/Qwen3-8B",
    })
    common.setdefault("evaluation", {}).update({
        "datasets": [DATASET], "insufficient_option": False,
        "main_metric": "accuracy", "primary_axis": "accuracy",
        "sealed_pairing": False, "require_split_freeze": False,
    })
    common.setdefault("retrieval", {}).pop("query_control", None)
    common["retrieval"].pop("pubmed_source_queries", None)
    common.setdefault("tools", {})["active"] = [
        tool for tool in common.get("tools", {}).get("active", [])
        if tool != "academic.search"]

    prompt_cfg = json.loads(json.dumps(common))
    prompt_cfg["evolution"]["allowed_artifacts"] = ["system_prompt"]
    prompt_cfg["evolution"].pop("mipro", None)
    prompt_cfg["evolution"]["gepa"] = {
        "enabled": True, "max_metric_calls": int(metric_calls),
        "max_candidate_proposals": int(proposals),
        "reflection_minibatch_size": int(minibatch),
        "acceptance_criterion": "strict_improvement",
        "feedback_evidence_signals": True, "multi_objective": False,
        "visible_train_task_ids": list(train_ids),
        "visible_validation_task_ids": list(search_ids),
        "cohort_sha256": str(cohort["cohort_sha256"]),
        "reflection_label_semantics": LABEL_SEMANTICS, "task_offset": 0,
    }

    skill_cfg = json.loads(json.dumps(common))
    skill_cfg["evolution"]["allowed_artifacts"] = ["skill_body"]
    skill_cfg["evolution"].pop("mipro", None)
    skill_cfg["evolution"]["gepa"] = {
        "enabled": True, "max_metric_calls": int(metric_calls),
        "max_candidate_proposals": int(proposals),
        "reflection_minibatch_size": int(minibatch),
        "acceptance_criterion": "strict_improvement",
        "feedback_evidence_signals": True, "multi_objective": False,
        "visible_train_task_ids": list(train_ids),
        "visible_validation_task_ids": list(search_ids),
        "cohort_sha256": str(cohort["cohort_sha256"]),
        "reflection_label_semantics": LABEL_SEMANTICS, "task_offset": 0,
        "target_artifact": "skill_body", "target_skill": TARGET_SKILL,
    }
    return prompt_cfg, skill_cfg


def _plan(cohort_path: Path, cohort: dict[str, Any], scopes: dict[str, list[dict]],
          state: Any, prompt_cfg: dict[str, Any], skill_cfg: dict[str, Any],
          runner_path: Path, labels: list[int], metric_calls: int,
          proposals: int, grounding_calls: int, retry_reserve: int = 12) -> dict[str, Any]:
    internal_evaluations = (len(scopes["internal_confirmation"])
                            * len(labels) * 4)
    task_total = metric_calls * 2 + internal_evaluations
    task_physical_attempts = ((int(prompt_cfg["runtime"].get(
        "pi_provider_max_retries", 2)) + 1) * int(prompt_cfg["runtime"].get(
            "pi_timeout_max_attempts", 1)))
    reflection_calls = proposals * 2
    reflection_attempts = int(prompt_cfg["models"]["reflection"].get(
        "max_retries", 2)) + 1
    grounding_attempts = int(prompt_cfg["models"]["task"].get("max_retries", 2)) + 1
    configured_logical = task_total + reflection_calls + grounding_calls + retry_reserve
    plan = {
        "protocol": "evidence-inference-v2-gepa-prompt-skill-factorial-v1",
        "dataset": DATASET, "cohort_sha256": cohort["cohort_sha256"],
        "cohort_file_sha256": sha256_file(cohort_path),
        "release_id": state.release_id,
        "target_artifacts": ["system_prompt", "skill_body"],
        "target_skill": TARGET_SKILL,
        "configuration_sha256": sha256_obj({
            "prompt": config_fingerprint(prompt_cfg),
            "skill": config_fingerprint(skill_cfg),
            "runner_sha256": sha256_file(runner_path),
        }),
        "optimizer": "pinned vendor GEPA optimize_anything; independent Prompt and Skill searches",
        "task_model": prompt_cfg["models"]["task"]["model_id"],
        "reflection_model": prompt_cfg["models"]["reflection"]["model_id"],
        "routing_mode": "lexical", "retrieval_used": False,
        "scopes": {name: len(rows) for name, rows in scopes.items()},
        "class_counts": {name: cohort["scopes"][name]["class_counts"]
                          for name in scopes},
        "execution_labels": list(labels),
        "execution_labels_are_independent_rng_seeds": False,
        "gepa_metric_call_budget_per_artifact": int(metric_calls),
        "gepa_proposal_call_budget_per_artifact": int(proposals),
        "candidate_grounding_call_budget": int(grounding_calls),
        "internal_baseline_prompt_skill_and_joint_evaluations": internal_evaluations,
        "logical_retry_reserve": int(retry_reserve),
        "pi_bridge_process_attempts": int(prompt_cfg["runtime"].get(
            "pi_timeout_max_attempts", 1)),
        "maximum_task_model_calls": task_total,
        "maximum_expected_model_calls": task_total + reflection_calls + grounding_calls,
        "maximum_configured_model_calls": configured_logical,
        "maximum_physical_http_requests": (
            (task_total + retry_reserve) * task_physical_attempts
            + reflection_calls * reflection_attempts
            + grounding_calls * grounding_attempts),
        "logical_budget_counts_provider_operations": True,
        "selection_gate_read": False, "official_validation_read": False,
        "official_test_read": False, "confirmation_eligible": False,
        "api_key_configured": {
            key: bool(os.environ.get(str(
                prompt_cfg["models"].get(key, {}).get("api_key_env") or "")))
            for key in ("task", "reflection")
        },
    }
    plan["plan_sha256"] = sha256_obj(plan)
    return plan


def _gate_skill(candidate: Candidate, cfg: dict[str, Any], state: Any,
                gate_examples: list[dict[str, Any]], grounding: dict[str, Any]) -> dict[str, Any]:
    static = static_gate(candidate, cfg, state)
    overlap = training_question_overlap(candidate.new_content, gate_examples)
    grounding_passed = bool(
        grounding.get("passed") is True and grounding.get("verdict") == "claim_free"
        and grounding.get("candidate_sha256") == content_sha256(
            candidate.new_content.strip())
        and grounding.get("artifact_type") == "skill_body"
        and grounding.get("assertion_scan_complete") is True
        and grounding.get("model_id"))
    method_rules = validate_statistical_method_rules(candidate.new_content)
    passed = bool(static.get("passed") and not overlap and grounding_passed
                  and method_rules.get("passed"))
    return {"passed": passed,
            "reason": "all_pre_evaluation_gates_passed" if passed else
            "skill_static_grounding_or_overlap_gate_rejected",
            "static_gate": static, "training_question_overlap": overlap,
            "method_rule_gate": method_rules,
            "grounding_review": grounding}


def _build_eligible_arms(state: ArtifactState, eligible: dict[str, Candidate],
                         skill_name: str) -> dict[str, ArtifactState]:
    """Build only distinct ablation arms backed by individually gated candidates."""
    arms = {"baseline": state}
    prompt = eligible.get("prompt_only")
    skill_candidate = eligible.get("skill_only")
    if prompt:
        arms["prompt_only"] = ArtifactState(
            state.release_id, prompt.new_content, state.registry, state.task_demos)
    if skill_candidate:
        skill_registry = with_body(state.registry, skill_name,
                                   skill_candidate.new_content)
        arms["skill_only"] = ArtifactState(
            state.release_id, state.system_prompt, skill_registry, state.task_demos)
    if prompt and skill_candidate:
        arms["joint"] = ArtifactState(
            state.release_id, prompt.new_content, skill_registry, state.task_demos)
    return arms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--task-model-id", choices=["Qwen/Qwen3-8B", "Qwen/Qwen3-32B"],
                        default="Qwen/Qwen3-8B")
    parser.add_argument("--reflection-model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--metric-call-budget-per-artifact", type=int, default=96)
    parser.add_argument("--proposals-per-artifact", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=5)
    parser.add_argument("--execution-labels", nargs="+", type=int,
                        default=[202610101, 202610102])
    parser.add_argument("--seed", type=int, default=202610101)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true",
                      help="write the zero-call plan for review")
    mode.add_argument("--execute", action="store_true",
                      help="execute only the reviewed plan with its exact hash")
    parser.add_argument("--expected-plan-sha256", default="",
                        help="required with --execute; must match the recomputed plan")
    args = parser.parse_args(argv)
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("EI joint GEPA requires an isolated RSI_HOME")
    if args.metric_call_budget_per_artifact <= 48 or args.proposals_per_artifact < 1:
        raise ValueError("each artifact budget must cover baseline and candidate validation")
    if not args.execution_labels or len(set(args.execution_labels)) != len(args.execution_labels):
        raise ValueError("execution labels must be non-empty and unique")

    cohort_path = args.cohort.resolve()
    train_rows = load_tasks(splits=["dev_optimize"])
    cohort, scopes, _ = load_frozen_cohort(cohort_path, train_rows)
    train_ids = [str(row["task_id"]) for row in scopes["mipro_train"]]
    search_ids = [str(row["task_id"]) for row in scopes["search_validation"]]
    cfgs = _prepare_config(args.config, cohort, train_ids, search_ids,
                           args.metric_call_budget_per_artifact,
                           args.proposals_per_artifact, args.minibatch,
                           args.task_model_id, args.reflection_model_id)
    prompt_cfg, skill_cfg = cfgs
    state = load_state(RELEASE_ID)
    active = artifacts_path("releases", "ACTIVE")
    if (state.release_id != RELEASE_ID or not active.exists()
            or active.read_text(encoding="utf-8").strip() != RELEASE_ID):
        raise ValueError("isolated RSI_HOME must have the EI release active")
    skill = state.registry.get(TARGET_SKILL)
    if skill is None or skill.frozen:
        raise ValueError(f"target Skill is absent or frozen: {TARGET_SKILL}")
    plan = _plan(cohort_path, cohort, scopes, state, prompt_cfg, skill_cfg,
                 Path(__file__).resolve(), args.execution_labels,
                 args.metric_call_budget_per_artifact,
                 args.proposals_per_artifact, grounding_calls=2)
    if args.validate_only:
        print(json.dumps({"status": "valid", **plan}, ensure_ascii=False))
        return 0
    if args.expected_plan_sha256 != plan["plan_sha256"]:
        raise ValueError("expected plan hash does not match the recomputed EI joint GEPA plan")
    missing = [name for name, present in plan["api_key_configured"].items() if not present]
    if missing:
        raise RuntimeError("missing configured credentials for " + ", ".join(missing))

    os.environ["RSI_HTTP_NO_CACHE"] = "1"
    study_id = plan["plan_sha256"][:16]
    study = artifacts_path("probe", "evidence_inference_gepa_joint", study_id)
    study.mkdir(parents=True, exist_ok=True)
    report_path, plan_path = study / "report.json", study / "plan.json"
    ledger_path = study / "request-ledger.jsonl"
    if report_path.exists() or plan_path.exists() or ledger_path.exists():
        raise RuntimeError("joint GEPA study already has state; use a fresh frozen plan")
    write_json(plan_path, plan)
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(plan["plan_sha256"])
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path.resolve())
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(plan["maximum_configured_model_calls"])
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(plan["maximum_physical_http_requests"])
    os.environ["RSI_CALL_LEDGER_MAX_COMPONENT_TASK_ANSWER"] = str(
        plan["maximum_task_model_calls"] + plan["logical_retry_reserve"])
    os.environ["RSI_CALL_LEDGER_MAX_COMPONENT_GEPA_REFLECTION"] = str(
        plan["gepa_proposal_call_budget_per_artifact"] * 2)
    os.environ["RSI_CALL_LEDGER_MAX_COMPONENT_GROUNDING"] = str(
        plan["candidate_grounding_call_budget"])

    gate_examples = scopes["mipro_train"] + scopes["search_validation"]
    candidates: dict[str, Candidate] = {}
    gates: dict[str, dict[str, Any]] = {}
    # Search artifacts independently. No challenger or a failed candidate gate
    # for one artifact must not suppress the other's search and valid ablation.
    for arm in ("prompt_only", "skill_only"):
        try:
            if arm == "prompt_only":
                candidate = search_prompt_candidate(
                    prompt_cfg, state, [DATASET], len(train_ids) + len(search_ids),
                    args.seed, failure_ids=[])
            else:
                candidate = search_skill_body_candidate(
                    skill_cfg, state, [DATASET], len(train_ids) + len(search_ids),
                    args.seed, failure_ids=[], skill_name=TARGET_SKILL)
            if candidate is None:
                gates[arm] = {"passed": False,
                              "reason": "no_strict_search_challenger"}
                continue
            candidates[arm] = candidate
            artifact_type = "system_prompt" if arm == "prompt_only" else "skill_body"
            cfg = prompt_cfg if arm == "prompt_only" else skill_cfg
            if arm == "prompt_only":
                preliminary = validate_task_input_prompt(
                    candidate.new_content, gate_examples, prompt_cfg, state,
                    candidate_id=candidate.candidate_id)
                if preliminary.get("reason") != "grounding_review_missing_or_failed":
                    gates[arm] = preliminary
                    continue
            else:
                static = static_gate(candidate, skill_cfg, state)
                overlap = training_question_overlap(candidate.new_content, gate_examples)
                method_rules = validate_statistical_method_rules(candidate.new_content)
                if (not static.get("passed") or overlap
                        or not method_rules.get("passed")):
                    gates[arm] = {
                        "passed": False,
                        "reason": "skill_pre_grounding_gate_rejected",
                        "static_gate": static,
                        "training_question_overlap": overlap,
                        "method_rule_gate": method_rules,
                    }
                    continue
            with call_component("grounding"):
                grounding = run_prompt_grounding_review(
                    candidate.new_content, cfg, artifact_type)
            if arm == "prompt_only":
                gate = validate_task_input_prompt(
                    candidate.new_content, gate_examples, prompt_cfg, state,
                    grounding_review=grounding, candidate_id=candidate.candidate_id)
            else:
                gate = _gate_skill(candidate, skill_cfg, state,
                                   gate_examples, grounding)
            gates[arm] = gate
        except Exception as exc:
            gates[arm] = {"passed": False, "reason": "search_or_gate_error",
                          "error_type": type(exc).__name__,
                          "error_detail_omitted": True}

    eligible = {arm: candidates[arm] for arm in ("prompt_only", "skill_only")
                if arm in candidates and gates.get(arm, {}).get("passed") is True}
    if not eligible:
        write_json(report_path, {
            "status": "candidate_search_or_gate_failed",
            "error": "neither artifact produced a candidate that passed pre-internal gates",
            "plan": plan,
            "candidates": {k: v.to_dict() for k, v in candidates.items()},
            "candidate_gates": gates, "request_accounting": summarize_ledger(ledger_path),
            "selection_gate_read": False, "official_validation_read": False,
            "official_test_read": False, "retrieval_used": False,
        })
        print(json.dumps({"report": str(report_path),
                          "status": "candidate_search_or_gate_failed"}))
        return 2

    arms = _build_eligible_arms(state, eligible, TARGET_SKILL)
    rows_by_arm: dict[str, dict[str, list[dict[str, Any]]]] = {}
    try:
        for arm, arm_state in arms.items():
            rows_by_arm[arm] = _score_repeated_scope(
                prompt_cfg, arm_state, scopes["internal_confirmation"],
                list(args.execution_labels), study / f"internal-{arm}.jsonl",
                candidate_id=(None if arm == "baseline" else
                              sha256_obj({k: v.candidate_id for k, v in eligible.items()})
                              if arm == "joint" else eligible[arm].candidate_id))
    except Exception as exc:
        write_json(report_path, {
            "status": "internal_confirmation_failed", "error_type": type(exc).__name__,
            "error_detail_omitted": True, "plan": plan,
            "candidates": {k: v.to_dict() for k, v in candidates.items()},
            "candidate_gates": gates,
            "checkpoints": {arm: str(study / f"internal-{arm}.jsonl") for arm in arms},
            "request_accounting": summarize_ledger(ledger_path),
            "selection_gate_read": False, "official_validation_read": False,
            "official_test_read": False, "retrieval_used": False,
        })
        print(json.dumps({"report": str(report_path),
                          "status": "internal_confirmation_failed"}))
        return 2

    metrics_by_arm = {
        arm: {str(seed): metrics(by_seed[seed]) for seed in args.execution_labels}
        for arm, by_seed in rows_by_arm.items()
    }
    paired = {arm: paired_internal_analysis(
        rows_by_arm["baseline"], rows_by_arm[arm], list(args.execution_labels))
        for arm in rows_by_arm if arm != "baseline"}
    internal_gates = {
        arm: all(class_aware_gain(metrics_by_arm[arm][str(seed)],
                                 metrics_by_arm["baseline"][str(seed)])
                 for seed in args.execution_labels)
        for arm in rows_by_arm if arm != "baseline"
    }
    report = {
        "status": ("completed" if {"prompt_only", "skill_only"}.issubset(eligible)
                   else "completed_partial_factorial"),
        "factorial_complete": {"prompt_only", "skill_only"}.issubset(eligible),
        "missing_candidate_arms": sorted({"prompt_only", "skill_only"} - set(eligible)),
        "study_id": study_id, "plan": plan,
        "candidates": {k: v.to_dict() for k, v in candidates.items()},
        "candidate_gates": gates, "internal_metrics_by_arm_and_label": metrics_by_arm,
        "paired_analysis_vs_baseline": paired, "internal_gate_by_arm": internal_gates,
        "joint_arm_eligible_for_official_selection_gate": internal_gates.get("joint", False),
        "official_selection_gate_run": False,
        "execution_labels_are_independent_rng_seeds": False,
        "retrieval_used": False, "confirmation_eligible": False,
        "selection_gate_read": False, "official_validation_read": False,
        "official_test_read": False,
        "request_accounting": summarize_ledger(ledger_path),
        "checkpoints": {arm: str(study / f"internal-{arm}.jsonl") for arm in arms},
        "search_reports": {
            arm: candidate.generator.get("search_report")
            for arm, candidate in candidates.items()
        },
    }
    write_json(report_path, report)
    print(json.dumps({"report": str(report_path), "status": report["status"],
                      "internal_gate_by_arm": internal_gates,
                      "joint_eligible_for_official_selection_gate": internal_gates.get("joint", False)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
