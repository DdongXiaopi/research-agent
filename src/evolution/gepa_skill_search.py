"""Pinned GEPA search for one Skill description or body on fixed task input."""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Optional, Sequence

from ..config import config_fingerprint
from ..contracts.types import Candidate, content_id, content_sha256
from ..eval.metrics import total_cost
from ..eval.task_rows import validate_task_answer_row
from ..governance.policy import static_gate
from ..runtime.provider import HttpProvider, build_provider
from ..runtime.runner import ArtifactState, TaskRunner
from ..runtime.skills import load_body, with_body, with_description
from ..util import artifacts_path, sha256_obj, write_json
from .gepa_adapter import (TaskExecutionFailure, build_metric, build_optimizer, run_reflection,
                           summarize_chat_usage, summarize_vendor_proposals)
from .gepa_search import (feedback_material, ranked_challenger_indices,
                          MetricBudgetExceeded, task_trace_side_info, training_question_overlap,
                          visible_examples, task_input_fingerprint,
                          build_gepa_proposal_task_contract,
                          TASK_INPUT_FINGERPRINT_VERSION,
                          TRACE_FEEDBACK_RULE_VERSION,
                          FEEDBACK_RULE_VERSION,
                          feedback_claim_relation_hint,
                          feedback_evidence_signals,
                          feedback_target_result_sentences,
                          OBJECTIVE_SELECTION_POLICY_VERSION)
from .reflection import unified_diff
from .method_rule_gate import (METHOD_RULE_GATE_VERSION,
                               validate_statistical_method_rules)
from .task_contracts import (TASK_CONTRACT_RULE_VERSION,
                            validate_task_contract_candidate)


def _reflection_system(label_semantics: str = "",
                      label_semantics_source: str = "", *,
                      artifact_type: str = "skill_body",
                      task_contract_context: str = "") -> str:
    if artifact_type == "skill_description":
        system = (
            "Rewrite only the target Skill's short routing description using "
            "the task feedback. Return one concise single-line description, "
            "under 400 characters. Describe the kinds of tasks the Skill can "
            "help with; do not include a task answer, named benchmark example, "
            "task ID, or new procedure steps. This runtime receives fixed "
            "task-provided material and has no search tool. Trace IDs are "
            "audit metadata. AttributionHint is a heuristic routing label, "
            "and AlternativeFailureMechanisms are plausible alternatives, "
            "not confirmed causes. Compare them with task traces and answer "
            "feedback; never copy IDs or memorize examples. Return only the "
            "replacement description in one fenced block.")
    else:
        system = (
            "Rewrite the complete Skill procedure using the task feedback. "
            "Preserve its numbered Procedure with at least three executable steps. "
            "This runtime has task-provided material and no search tool. "
            "Do not request retrieval, cite invented sources, or memorize examples. "
            "Trace IDs are audit metadata. AttributionHint is a heuristic routing "
            "label, and AlternativeFailureMechanisms are plausible alternatives, "
            "not confirmed causes. Compare them with the task trace and answer "
            "feedback before changing the Skill; never copy IDs into the Skill. "
            "Return only the replacement Skill body in one fenced block.")
    semantics = str(label_semantics or "").strip()
    source = str(label_semantics_source or "").strip()
    if semantics:
        system += ("\n\nUse this source-backed task label contract exactly; "
                   "it defines dataset labels and does not authorize adding "
                   "facts absent from task material:\n" + semantics)
        if source:
            system += "\nSource: " + source
    contract_context = str(task_contract_context or "").strip()
    if len(contract_context) > 4000:
        raise ValueError("GEPA Skill task contract context must not exceed 4000 characters")
    if contract_context:
        system += (
            "\n\nApply this frozen task decision contract as a hard boundary when "
            "rewriting the Skill artifact. Preserve its evidence scope, decision "
            "criterion, and exact label distinctions; do not broaden it into a "
            "generic heuristic or change the runtime task prompt:\n" + contract_context)
    return system


def search_skill_body_candidate(cfg: Dict[str, Any], state: ArtifactState,
                                datasets: Sequence[str], limit: Optional[int],
                                seed: int, failure_ids: Sequence[str],
                                skill_name: str) -> Optional[Candidate]:
    """Backward-compatible entry point for the Skill-body GEPA arm."""
    return search_skill_candidate(cfg, state, datasets, limit, seed, failure_ids,
                                  skill_name, artifact_type="skill_body")


def search_skill_description_candidate(cfg: Dict[str, Any], state: ArtifactState,
                                       datasets: Sequence[str], limit: Optional[int],
                                       seed: int, failure_ids: Sequence[str],
                                       skill_name: str) -> Optional[Candidate]:
    """Optimize only a Skill routing description; preserve triggers and body."""
    return search_skill_candidate(cfg, state, datasets, limit, seed, failure_ids,
                                  skill_name, artifact_type="skill_description")


def search_skill_candidate(cfg: Dict[str, Any], state: ArtifactState,
                           datasets: Sequence[str], limit: Optional[int],
                           seed: int, failure_ids: Sequence[str],
                           skill_name: str, *, artifact_type: str) -> Optional[Candidate]:
    """Optimize one Skill artifact on visible dev only; outer round owns gate."""
    if artifact_type not in {"skill_body", "skill_description"}:
        raise ValueError("Skill GEPA supports skill_body or skill_description")
    if (cfg.get("runtime") or {}).get("input_mode") != "task_input":
        raise ValueError("Skill GEPA search requires task_input mode")
    if (cfg.get("evaluation") or {}).get("main_metric") != "accuracy":
        raise ValueError("Skill GEPA task score requires accuracy outer gate")
    skill = state.registry.get(skill_name)
    if skill is None or skill.frozen:
        raise ValueError("target Skill is absent or frozen")
    is_body = artifact_type == "skill_body"
    seed_content = (load_body(skill) if is_body else str(skill.description)).strip()
    if not seed_content:
        raise ValueError(f"target Skill has no {artifact_type.removeprefix('skill_')}")
    artifact_label = "body" if is_body else "description"
    attributed_class = "experiment.skill_body" if is_body else "experiment.skill_description"
    train, val = visible_examples(cfg, datasets, limit)
    proposal_task_contract_context, proposal_task_contract = (
        build_gepa_proposal_task_contract(train + val))
    validation_keys = {(str(x.get("dataset", "")), str(x.get("task_id", "")))
                       for x in val}
    gp = (cfg.get("evolution") or {}).get("gepa") or {}
    max_calls = int(gp.get("max_metric_calls", 120))
    max_candidate_proposals = int(gp.get("max_candidate_proposals", 8))
    reflection_minibatch_size = int(gp.get("reflection_minibatch_size", 5))
    acceptance_criterion = str(gp.get("acceptance_criterion", "strict_improvement"))
    multi_objective = bool(gp.get("multi_objective", False))
    use_evidence_signals = bool(gp.get("feedback_evidence_signals", False))
    feedback_version = (TRACE_FEEDBACK_RULE_VERSION +
                        (("+" + FEEDBACK_RULE_VERSION) if use_evidence_signals else ""))
    use_merge = bool(gp.get("use_merge", False))
    merge_max_invocations = int(gp.get("merge_max_invocations", 5))
    merge_val_overlap_floor = int(gp.get("merge_val_overlap_floor", 5))
    label_semantics = str(gp.get("reflection_label_semantics") or "").strip()
    label_semantics_source = str(
        gp.get("reflection_label_semantics_source") or "").strip()
    reflection_system_message = _reflection_system(
        label_semantics, label_semantics_source, artifact_type=artifact_type,
        task_contract_context=proposal_task_contract_context or "")
    reflection_system_message_sha256 = content_sha256(reflection_system_message)
    validation_reserve = len(val)
    optimizer_max_metric_calls = max_calls - validation_reserve
    if optimizer_max_metric_calls <= len(val):
        raise ValueError(
            "Skill GEPA budget must cover seed validation, at least one proposal "
            "evaluation, and one full candidate validation "
            f"(need > {2 * len(val)} calls; got {max_calls})")
    run_id = "gepa-skill-" + uuid.uuid4().hex[:12]
    out_dir = artifacts_path("search", "gepa_skill", run_id)
    out_dir.mkdir(parents=True, exist_ok=False)
    report_path = out_dir / "report.json"
    task_provider = build_provider(cfg, seed=seed)
    reflection_cfg = dict((cfg.get("models") or {}).get("reflection") or {})
    if reflection_cfg.pop("provider", None) != "http":
        raise ValueError("actual Skill GEPA requires an HTTP reflection model")
    reflection_cfg["max_tokens"] = max(1800, int(reflection_cfg.get("max_tokens") or 0))
    reflection_provider = HttpProvider(cache=False, **reflection_cfg)
    metric = build_metric(multi_objective=multi_objective)
    attempts = 0
    task_model_evaluations = 0
    task_outcomes: List[Any] = []
    task_evaluation_rows: List[Dict[str, Any]] = []
    reflection_usage: List[Dict[str, Any]] = []

    def reflection_lm(prompt: Any) -> str:
        user = ("\n\n".join(str(x.get("content", "")) for x in prompt)
                if isinstance(prompt, list) else str(prompt))
        measured = reflection_provider.chat_measured(
            reflection_system_message, user, max_tokens=1800)
        reflection_usage.append({
            "prompt_tokens": measured.prompt_tokens,
            "completion_tokens": measured.completion_tokens,
            "latency_ms": measured.latency_ms,
            "cache_hit": measured.cache_hit,
            "usage_available": measured.usage_available,
            "cost": measured.cost,
        })
        report["reflection_usage"] = summarize_chat_usage(reflection_usage)
        report["reflection_provider_stats"] = (
            dict(reflection_provider.stats)
            if isinstance(getattr(reflection_provider, "stats", None), dict) else {})
        write_json(report_path, report)
        return measured.content

    def candidate_for(text: str) -> Candidate:
        return Candidate(
            candidate_id=content_id("cand", state.release_id, artifact_type,
                                    skill_name, text),
            parent_release_id=state.release_id, artifact_type=artifact_type,
            artifact_id=skill_name,
            diff=unified_diff(seed_content, text,
                              f"a/{skill_name}.{artifact_label}",
                              f"b/{skill_name}.{artifact_label}"),
            new_content=text, attributed_class=attributed_class,
            content_sha256=content_sha256(text))

    def evaluator(candidate: str, example: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        nonlocal attempts, task_model_evaluations
        attempts += 1
        if attempts > max_calls:
            raise MetricBudgetExceeded(
                f"Skill GEPA requested task evaluation {attempts} above hard "
                f"budget {max_calls}")
        proposed = str(candidate).strip()
        if proposed != seed_content:
            if is_body:
                task_contract_gate = validate_task_contract_candidate(
                    proposed, [example])
                if not task_contract_gate["passed"]:
                    return 0.0, {
                        "Feedback": "Candidate rejected before evaluation: "
                                    + task_contract_gate["reason"],
                        "Status": "task_contract_reject",
                        "TaskContractGate": task_contract_gate,
                    }
            overlap = training_question_overlap(proposed, train + val)
            method_rules = validate_statistical_method_rules(proposed)
            gate = static_gate(candidate_for(proposed), cfg, state)
            if overlap or not method_rules["passed"] or not gate["passed"]:
                failed = [x["check"] for x in gate["results"] if not x["passed"]]
                return 0.0, {"Feedback": "Candidate rejected before evaluation: "
                             + str(overlap or
                                   (None if method_rules["passed"] else
                                    method_rules["reason"]) or ", ".join(failed)),
                             "Status": ("method_rule_reject" if not method_rules["passed"]
                                        else "static_reject"),
                             "MethodRuleGate": method_rules}
        trial_registry = (with_body(state.registry, skill_name, proposed)
                          if is_body else
                          with_description(state.registry, skill_name, proposed))
        trial = ArtifactState(
            state.release_id, state.system_prompt, trial_registry,
            task_demos=copy.deepcopy(state.task_demos))
        task_model_evaluations += 1
        outcome = TaskRunner(cfg, trial, provider=task_provider, seed=seed).run(example)
        task_outcomes.append(outcome)
        report["task_evaluation_cost"] = total_cost(task_outcomes)
        write_json(report_path, report)
        if str(getattr(outcome, "status", "")) in {"ok", "abstain"}:
            try:
                outcome_row = (outcome.to_dict() if callable(
                    getattr(outcome, "to_dict", None)) else vars(outcome))
                validate_task_answer_row(
                    outcome_row, example, context="GEPA Skill task evaluation",
                    require_reported_gold_index=True)
            except (AttributeError, TypeError, ValueError) as exc:
                raise TaskExecutionFailure(
                    str(getattr(outcome, "task_id", example.get("task_id", "unknown"))),
                    "invalid_outcome") from exc
        score, side = metric(outcome)
        task_evaluation_rows.append({
            "dataset": str(example.get("dataset", "")),
            "task_id": str(example.get("task_id", "")),
            "partition": ("validation" if (str(example.get("dataset", "")),
                                               str(example.get("task_id", "")))
                          in validation_keys else "train"),
            "candidate_sha256": sha256_obj(proposed),
            "task_input_sha256": task_input_fingerprint(example),
            "task_input_fingerprint_version": TASK_INPUT_FINGERPRINT_VERSION,
            "gold_index": int(example["correct_index"]),
            "score": float(score),
            "status": str(outcome.status),
            "correct": bool(outcome.correct),
            "answer_index": int(outcome.answer_index),
        })
        report["task_evaluation_rows"] = list(task_evaluation_rows)
        write_json(report_path, report)
        options = list(example["options"])
        expected = options[int(example["correct_index"])]
        selected = (options[outcome.answer_index]
                    if 0 <= outcome.answer_index < len(options) else "ABSTAIN")
        side.update({"Question": str(example["question"])[:400],
                     "Options": options, "TaskMaterial": feedback_material(example),
                     "Expected": expected, "Selected": selected,
                     "SkillRecalled": outcome.skill_recalled,
                     "SkillBehaviors": list(outcome.skill_behaviors),
                     "Feedback": ("Correct" if outcome.correct else
                                  f"Selected {selected}; expected {expected}; "
                                  f"status={outcome.status}")})
        side.update(task_trace_side_info(outcome))
        if use_evidence_signals:
            relation_hint = feedback_claim_relation_hint(example)
            if relation_hint:
                side["TargetRelationHint"] = relation_hint
            side["PossibleNegativeResultSentences"] = feedback_evidence_signals(example)
            side["PossibleTargetResultSentences"] = (
                feedback_target_result_sentences(example))
        return score, side

    report: Dict[str, Any] = {
        "run_id": run_id, "status": "searching",
        "optimizer": "pinned vendor GEPA optimize_anything",
        "artifact_type": artifact_type, "artifact_id": skill_name,
        "seed_content_sha256": content_sha256(seed_content),
        "seed_body_sha256": content_sha256(seed_content) if is_body else None,
        "task_model_id": (cfg.get("models") or {}).get("task", {}).get("model_id"),
        "reflection_model_id": reflection_cfg.get("model_id"),
        "max_metric_calls": max_calls,
        "optimizer_max_metric_calls": optimizer_max_metric_calls,
        "validation_reserve": validation_reserve,
        "max_candidate_proposals": max_candidate_proposals,
        "reflection_minibatch_size": reflection_minibatch_size,
        "acceptance_criterion": acceptance_criterion,
        "multi_objective": multi_objective,
        "objective_names": (["accuracy", "coverage", "answer_quality"]
                            if multi_objective else ["accuracy"]),
        "objective_selection_policy": (
            OBJECTIVE_SELECTION_POLICY_VERSION if multi_objective else None),
        "use_merge": use_merge,
        "merge_max_invocations": merge_max_invocations if use_merge else None,
        "merge_val_overlap_floor": merge_val_overlap_floor if use_merge else None,
        "protocol_sha256": config_fingerprint(cfg), "release_id": state.release_id,
        "feedback_rule_version": feedback_version,
        "method_rule_gate_version": METHOD_RULE_GATE_VERSION,
        "task_contract_rule_version": TASK_CONTRACT_RULE_VERSION,
        "proposal_task_contract": proposal_task_contract,
        "reflection_system_message_sha256": reflection_system_message_sha256,
        "reflection_label_semantics": label_semantics or None,
        "reflection_label_semantics_source": label_semantics_source or None,
        "task_offset": int(gp.get("task_offset", 0)),
        "optimizer_protocol_sha256": sha256_obj({
            "config": cfg, "feedback_rule_version": feedback_version,
            "method_rule_gate_version": METHOD_RULE_GATE_VERSION,
            "task_contract_rule_version": TASK_CONTRACT_RULE_VERSION,
            "proposal_task_contract": proposal_task_contract,
            "reflection_system_message_sha256": reflection_system_message_sha256,
            "objective_selection_policy": (
                OBJECTIVE_SELECTION_POLICY_VERSION if multi_objective else None)}),
        "train_ids": [x["task_id"] for x in train],
        "val_ids": [x["task_id"] for x in val],
        "task_input_fingerprint_version": TASK_INPUT_FINGERPRINT_VERSION,
        "task_evaluation_rows": task_evaluation_rows,
        "selection_gate_read": False,
        "reflection_usage": summarize_chat_usage(reflection_usage),
        "reflection_provider_stats": (
            dict(reflection_provider.stats)
            if isinstance(getattr(reflection_provider, "stats", None), dict) else {}),
        "task_evaluation_cost": total_cost(task_outcomes),
    }
    write_json(report_path, report)
    optimizer = build_optimizer(
        {"seed": seed,
         "max_candidate_proposals": max_candidate_proposals,
         "reflection_minibatch_size": reflection_minibatch_size,
         "acceptance_criterion": acceptance_criterion,
         "multi_objective": multi_objective,
         "use_merge": use_merge,
         "merge_max_invocations": merge_max_invocations,
         "merge_val_overlap_floor": merge_val_overlap_floor},
        reflection_lm, str(out_dir / "state"),
        max_metric_calls=optimizer_max_metric_calls)
    try:
        result = run_reflection(
            seed_candidate=seed_content, evaluator=evaluator,
            trainset=train, valset=val, config=optimizer,
            objective=("Improve fixed-task routing and scientific task performance "
                       f"by changing only this Skill {artifact_label}."),
            background=(f"Only the Skill {artifact_label} changes. Task material "
                        "is supplied with each question. The task runtime has no "
                        "retrieval tool or hidden answer key."))
    except MetricBudgetExceeded as exc:
        report.update({
            "status": "budget_exhausted_no_selection",
            "budget_abort_reason": str(exc),
            "optimizer_run_aborted": True,
            "total_metric_calls": None,
            "requested_max_metric_calls": max_calls,
            "evaluator_attempts": attempts,
            "budgeted_metric_calls": max_calls,
            "budgeted_task_evaluations": task_model_evaluations,
            "budget_guard_rejections": 1,
            "val_aggregate_scores": None,
            "val_scores_valid": False,
            "reflection_usage": summarize_chat_usage(reflection_usage),
            "reflection_provider_stats": (
                dict(reflection_provider.stats)
                if isinstance(getattr(reflection_provider, "stats", None), dict) else {}),
            "task_evaluation_cost": total_cost(task_outcomes),
        })
        write_json(report_path, report)
        return None
    except Exception as exc:
        report.update({
            "status": "optimizer_error_no_selection",
            "optimizer_run_aborted": True,
            "optimizer_error_type": type(exc).__name__,
            # Do not persist provider exception text, which can echo request
            # contents or credentials. The exception class is retained above.
            "optimizer_error_detail_omitted": True,
            "total_metric_calls": None,
            "requested_max_metric_calls": max_calls,
            "evaluator_attempts": attempts,
            "budgeted_metric_calls": min(attempts, max_calls),
            "budgeted_task_evaluations": task_model_evaluations,
            "val_scores_valid": False,
            "val_aggregate_scores": None,
            "val_aggregate_subscores": None,
            "selected_idx": None,
            "selected_sha256": None,
            "task_evaluation_rows": [],
            "reflection_usage": summarize_chat_usage(reflection_usage),
            "reflection_provider_stats": (
                dict(reflection_provider.stats)
                if isinstance(getattr(reflection_provider, "stats", None), dict)
                else {}),
            "task_evaluation_cost": total_cost(task_outcomes),
        })
        if isinstance(exc, TaskExecutionFailure):
            report["task_execution_failure"] = dict(exc.failure)
        write_json(report_path, report)
        raise
    texts = [str(next(iter(x.values()))) if isinstance(x, dict) else str(x)
             for x in result.candidates]
    scores = list(result.val_aggregate_scores)
    report.update({"candidate_texts": texts, "val_aggregate_scores": scores,
                   "val_aggregate_subscores": (
                       [dict(x) for x in result.val_aggregate_subscores]
                       if getattr(result, "val_aggregate_subscores", None) is not None
                       else None),
                   "n_candidates": len(texts), "total_metric_calls": result.total_metric_calls,
                   "evaluator_attempts": attempts,
                   "requested_max_metric_calls": max_calls,
                   "budgeted_metric_calls": min(attempts, max_calls),
                   "budgeted_task_evaluations": task_model_evaluations,
                   "budget_guard_rejections": max(0, attempts - max_calls),
                   "val_scores_valid": attempts <= max_calls,
                   "reflection_usage": summarize_chat_usage(reflection_usage),
                   "reflection_provider_stats": (
                       dict(reflection_provider.stats)
                       if isinstance(getattr(reflection_provider, "stats", None), dict)
                       else {}),
                   "task_evaluation_cost": total_cost(task_outcomes),
                   "vendor_run_log": str(out_dir / "state" / "run_log.json")})
    report.update(summarize_vendor_proposals(out_dir / "state" / "run_log.json"))
    if attempts > max_calls:
        report["status"] = "budget_exhausted_no_selection"
        report["val_scores_valid"] = False
        report["val_aggregate_scores"] = None
        write_json(report_path, report)
        return None
    objective_scores = (getattr(result, "val_aggregate_subscores", None)
                        if multi_objective else None)
    if multi_objective and not objective_scores:
        report.update({"status": "invalid_objective_validation",
                       "val_scores_valid": False,
                       "objective_selection_error":
                           "multiobjective search returned no validation objective scores"})
        write_json(report_path, report)
        return None
    try:
        selection_order = ranked_challenger_indices(scores, objective_scores)
    except ValueError as exc:
        report.update({"status": "invalid_objective_validation",
                       "val_scores_valid": False,
                       "objective_selection_error": str(exc)})
        write_json(report_path, report)
        return None
    rejected: List[Dict[str, Any]] = []
    for index in selection_order:
        proposed = texts[index].strip()
        if not proposed or proposed == seed_content:
            continue
        task_contract_gate = (validate_task_contract_candidate(proposed, train + val)
                              if is_body else {"passed": True})
        overlap = training_question_overlap(proposed, train + val)
        trial = candidate_for(proposed)
        method_rules = validate_statistical_method_rules(proposed)
        gate = static_gate(trial, cfg, state)
        if (not task_contract_gate["passed"] or overlap
                or not method_rules["passed"] or not gate["passed"]):
            rejected.append({"index": index,
                             "reason": (None if task_contract_gate["passed"] else
                                        task_contract_gate["reason"]) or overlap or
                             (None if method_rules["passed"] else
                              method_rules["reason"]) or
                             ", ".join(x["check"] for x in gate["results"]
                                       if not x["passed"])})
            continue
        tied = float(scores[index]) <= float(scores[0]) + 1e-9
        trial.source_failure_ids = list(failure_ids)
        trial.rationale = (f"Pinned GEPA Skill-{artifact_label} visible-dev challenger")
        trial.generator = {"backend": "pinned-gepa", "is_gepa": True,
                           "search_report": str(report_path), "selected_idx": index,
                           "validation_tie": tied,
                           "total_metric_calls": result.total_metric_calls}
        report.update({"status": "selected_tied_challenger" if tied else "selected",
                       "selected_idx": index, "selected_sha256": sha256_obj(proposed),
                       "selected_val_score": scores[index],
                       "selected_val_subscores": (
                           dict(result.val_aggregate_subscores[index])
                           if getattr(result, "val_aggregate_subscores", None) is not None
                           else None),
                       "final_static_rejections": rejected})
        write_json(report_path, report)
        return trial
    report.update({"status": "no_change", "final_static_rejections": rejected})
    write_json(report_path, report)
    return None
