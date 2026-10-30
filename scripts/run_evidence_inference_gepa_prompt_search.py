#!/usr/bin/env python3
"""Run bounded EI Prompt-only GEPA on an article-disjoint nested cohort.

The optimizer sees only frozen train/search IDs. Internal tasks are evaluated
only after the Prompt candidate is frozen and passes static/claim-free gates.
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_pubmedqa_prompt_discovery import run_prompt_grounding_review  # noqa: E402
from scripts.run_evidence_inference_mipro_search import (  # noqa: E402
    DATASET, RELEASE_ID, _score_repeated_scope, class_aware_gain,
    load_frozen_cohort, metrics, paired_internal_analysis, prepare_config,
)
from scripts.run_gepa_prompt_search import build_search_config  # noqa: E402
from src.config import config_fingerprint  # noqa: E402
from src.contracts.types import Candidate  # noqa: E402
from src.data.evidence_inference_v2 import load_tasks  # noqa: E402
from src.evolution.gepa_search import search_prompt_candidate  # noqa: E402
from src.evolution.task_input_gates import validate_task_input_prompt  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.call_ledger import call_component, summarize_ledger  # noqa: E402
from src.runtime.runner import ArtifactState  # noqa: E402
from src.util import artifacts_path, read_json, sha256_file, sha256_obj, write_json  # noqa: E402

LABEL_SEMANTICS = (
    "Choose exactly one of the task-provided labels: significantly increased, "
    "significantly decreased, or no significant difference. Match both the "
    "direction and statistical significance of the intervention versus comparator "
    "for the outcome asked. Do not infer significance from direction or magnitude."
)


def build_ei_gepa_config(config_path: str, *, cohort: dict[str, Any],
                         metric_calls: int, proposals: int, minibatch: int,
                         task_model_id: str, reflection_model_id: str,
                         train_ids: list[str], search_ids: list[str]) -> dict[str, Any]:
    cfg = prepare_config(config_path, proposer_calls=proposals,
                         routing_mode="lexical", task_model_id=task_model_id)
    cfg["models"]["reflection"]["model_id"] = reflection_model_id
    cfg["models"]["reflection"]["provider"] = "http"
    cfg["models"]["reflection"]["max_retries"] = 2
    cfg["models"]["task"]["max_retries"] = 2
    cfg["evolution"]["allowed_artifacts"] = ["system_prompt"]
    cfg["evolution"].pop("mipro", None)
    cfg["evolution"]["gepa"] = {
        "enabled": True,
        "max_metric_calls": int(metric_calls),
        "max_candidate_proposals": int(proposals),
        "reflection_minibatch_size": int(minibatch),
        "acceptance_criterion": "strict_improvement",
        "feedback_evidence_signals": True,
        "multi_objective": False,
        "visible_train_task_ids": list(train_ids),
        "visible_validation_task_ids": list(search_ids),
        "cohort_sha256": str(cohort["cohort_sha256"]),
        "reflection_label_semantics": LABEL_SEMANTICS,
        "task_offset": 0,
    }
    cfg.setdefault("runtime", {}).update({
        "provider": "pi", "input_mode": "task_input",
        "task_input_max_chars": 30000,
        "pi_provider_max_retries": 2,
        "pi_timeout_max_attempts": 1,
        "http_no_cache_required": True,
    })
    cfg.setdefault("skills", {}).update({
        "routing_mode": "lexical", "router_provider": "http",
        "router_model_id": "Qwen/Qwen3-8B",
    })
    cfg.setdefault("evaluation", {}).update({
        "datasets": [DATASET], "insufficient_option": False,
        "main_metric": "accuracy", "primary_axis": "accuracy",
        "sealed_pairing": False, "require_split_freeze": False,
    })
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        tool for tool in cfg.get("tools", {}).get("active", [])
        if tool != "academic.search"]
    return cfg


def build_gepa_plan(*, cohort_path: Path, cohort: dict[str, Any], cfg: dict[str, Any],
                    state: Any, execution_labels: list[int], metric_calls: int,
                    proposals: int, grounding_calls: int,
                    logical_retry_reserve: int = 12) -> dict[str, Any]:
    train_rows = load_tasks(splits=["dev_optimize"])
    _, scopes, _ = load_frozen_cohort(cohort_path, train_rows)
    task_evaluations = int(metric_calls)
    internal_evaluations = (len(scopes["internal_confirmation"])
                            * len(execution_labels) * 2)
    task_total = task_evaluations + internal_evaluations
    task_attempts = int(cfg["runtime"].get("pi_provider_max_retries", 2)) + 1
    reflection_attempts = int(cfg["models"]["reflection"].get("max_retries", 2)) + 1
    grounding_attempts = int(cfg["models"]["task"].get("max_retries", 2)) + 1
    physical_total = (task_total * task_attempts + proposals * reflection_attempts
                      + grounding_calls * grounding_attempts
                      + logical_retry_reserve * task_attempts)
    plan = {
        "protocol": "evidence-inference-v2-gepa-prompt-task-input-v1",
        "dataset": DATASET, "cohort_sha256": cohort["cohort_sha256"],
        "cohort_file_sha256": sha256_file(cohort_path),
        "release_id": state.release_id,
        "configuration_sha256": config_fingerprint(cfg),
        "optimizer": "pinned vendor GEPA optimize_anything",
        "task_model": cfg["models"]["task"]["model_id"],
        "reflection_model": cfg["models"]["reflection"]["model_id"],
        "routing_mode": "lexical", "retrieval_used": False,
        "scopes": {name: len(rows) for name, rows in scopes.items()},
        "class_counts": {name: cohort["scopes"][name]["class_counts"]
                          for name in scopes},
        "execution_labels": list(execution_labels),
        "execution_labels_are_independent_rng_seeds": False,
        "gepa_metric_call_budget": int(metric_calls),
        "gepa_proposal_call_budget": int(proposals),
        "candidate_grounding_call_budget": int(grounding_calls),
        "logical_retry_reserve": int(logical_retry_reserve),
        "internal_baseline_and_candidate_evaluations": internal_evaluations,
        "maximum_task_model_calls": task_total,
        "maximum_expected_model_calls": task_total + proposals + grounding_calls,
        "maximum_configured_model_calls": (
            task_total + proposals + grounding_calls + logical_retry_reserve),
        "maximum_physical_http_requests": physical_total,
        "logical_budget_counts_provider_operations": True,
        "selection_gate_read": False, "official_validation_read": False,
        "official_test_read": False, "confirmation_eligible": False,
        "api_key_configured": {
            section: bool(os.environ.get(str(
                cfg["models"].get(section, {}).get("api_key_env") or "")))
            for section in ("task", "reflection")
        },
    }
    plan["plan_sha256"] = sha256_obj(plan)
    return plan


def _failure_report(plan: dict[str, Any], study_id: str, status: str,
                    ledger_path: Path, **fields: Any) -> dict[str, Any]:
    return {
        "protocol": plan["protocol"], "plan": plan, "study_id": study_id,
        "status": status, "request_accounting": summarize_ledger(ledger_path),
        "selection_gate_read": False, "official_validation_read": False,
        "official_test_read": False, "retrieval_used": False,
        "confirmation_eligible": False, **fields,
    }


def verify_internal_resume(plan: dict[str, Any], frozen_plan: dict[str, Any],
                          previous_report: dict[str, Any],
                          frozen_candidate: dict[str, Any]) -> tuple[Candidate, dict[str, Any]]:
    """Load only a candidate whose completed search and passing gate are frozen."""
    if frozen_plan != plan or previous_report.get("plan", {}).get("plan_sha256") != plan.get("plan_sha256"):
        raise ValueError("saved GEPA plan/report differs from current protocol")
    if previous_report.get("status") != "internal_confirmation_failed":
        raise ValueError("GEPA resume is limited to interrupted internal confirmation")
    if (previous_report.get("candidate_gates", {}).get("passed") is not True
            or frozen_candidate.get("plan_sha256") != plan.get("plan_sha256")
            or frozen_candidate.get("candidate_id") != previous_report.get("candidate_id")
            or frozen_candidate.get("candidate") != previous_report.get("candidate")
            or frozen_candidate.get("candidate_gates") != previous_report.get("candidate_gates")):
        raise ValueError("saved candidate or passing gate does not match failed report")
    candidate_data = frozen_candidate.get("candidate")
    if not isinstance(candidate_data, dict):
        raise ValueError("frozen GEPA candidate payload is malformed")
    candidate = Candidate(**{
        key: value for key, value in candidate_data.items()
        if key in Candidate.__annotations__
    })
    if (candidate.candidate_id != previous_report.get("candidate_id")
            or not candidate.new_content.strip()):
        raise ValueError("frozen GEPA candidate identity/content is invalid")
    return candidate, dict(frozen_candidate["candidate_gates"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--task-model-id", choices=["Qwen/Qwen3-8B", "Qwen/Qwen3-32B"],
                        default="Qwen/Qwen3-8B")
    parser.add_argument("--reflection-model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--metric-call-budget", type=int, default=96)
    parser.add_argument("--proposals", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=5)
    parser.add_argument("--grounding-call-budget", type=int, default=1)
    parser.add_argument("--execution-labels", nargs="+", type=int,
                        default=[202610061, 202610062])
    parser.add_argument("--seed", type=int, default=202610061)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true",
                      help="write the zero-call plan for review")
    mode.add_argument("--execute", action="store_true",
                      help="execute only the reviewed plan with its exact hash")
    parser.add_argument("--expected-plan-sha256", default="",
                        help="required with --execute; must match the recomputed plan")
    parser.add_argument("--resume-internal-study-id")
    args = parser.parse_args(argv)
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    if args.resume_internal_study_id and args.validate_only:
        parser.error("--resume-internal-study-id requires --execute")

    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("EI GEPA requires an isolated RSI_HOME")
    if args.metric_call_budget <= 48 or args.proposals < 1 or args.minibatch < 1:
        raise ValueError("GEPA budget must cover baseline validation, a proposal, and full validation")
    if args.grounding_call_budget != 1:
        raise ValueError("selected Prompt candidate requires exactly one measured grounding call")
    if not args.execution_labels or len(set(args.execution_labels)) != len(args.execution_labels):
        raise ValueError("execution labels must be non-empty and unique")

    cohort_path = args.cohort.resolve()
    train_rows = load_tasks(splits=["dev_optimize"])
    cohort, scopes, task_by_id = load_frozen_cohort(cohort_path, train_rows)
    train_ids = [str(row["task_id"]) for row in scopes["mipro_train"]]
    search_ids = [str(row["task_id"]) for row in scopes["search_validation"]]
    internal_ids = {str(row["task_id"]) for row in scopes["internal_confirmation"]}
    if (set(train_ids) & set(search_ids) or set(train_ids + search_ids) & internal_ids):
        raise ValueError("frozen GEPA train/search/internal scopes overlap")
    cfg = build_ei_gepa_config(
        args.config, cohort=cohort, metric_calls=args.metric_call_budget,
        proposals=args.proposals, minibatch=args.minibatch,
        task_model_id=args.task_model_id,
        reflection_model_id=args.reflection_model_id,
        train_ids=train_ids, search_ids=search_ids)
    state = load_state(RELEASE_ID)
    active_path = artifacts_path("releases", "ACTIVE")
    if state.release_id != RELEASE_ID or not active_path.exists() or (
            active_path.read_text(encoding="utf-8").strip() != RELEASE_ID):
        raise ValueError("isolated RSI_HOME must have the Evidence Inference release active")
    plan = build_gepa_plan(
        cohort_path=cohort_path, cohort=cohort, cfg=cfg, state=state,
        execution_labels=args.execution_labels,
        metric_calls=args.metric_call_budget, proposals=args.proposals,
        grounding_calls=args.grounding_call_budget)
    if args.validate_only:
        print(json.dumps({"status": "valid", **plan}, ensure_ascii=False))
        return 0
    if args.expected_plan_sha256 != plan["plan_sha256"]:
        raise ValueError("expected plan hash does not match the recomputed EI GEPA Prompt plan")
    missing = [key for key, configured in plan["api_key_configured"].items()
               if not configured]
    if missing:
        raise RuntimeError("missing configured credentials for " + ", ".join(missing))

    os.environ["RSI_HTTP_NO_CACHE"] = "1"
    study_id = plan["plan_sha256"][:16]
    if args.resume_internal_study_id and args.resume_internal_study_id != study_id:
        raise ValueError("resume study ID differs from the current frozen GEPA plan")
    study_dir = artifacts_path("probe", "evidence_inference_gepa_prompt", study_id)
    study_dir.mkdir(parents=True, exist_ok=True)
    report_path = study_dir / "report.json"
    plan_path = study_dir / "plan.json"
    ledger_path = study_dir / "request-ledger.jsonl"
    candidate_path = study_dir / "frozen-candidate.json"
    resume_internal = bool(args.resume_internal_study_id)
    if resume_internal:
        if not plan_path.exists() or not report_path.exists() or not candidate_path.exists():
            raise ValueError("internal resume requires the frozen plan, failed report, and candidate")
        frozen_plan = read_json(plan_path)
        previous_report = read_json(report_path)
        frozen_candidate = read_json(candidate_path)
        candidate, gates = verify_internal_resume(
            plan, frozen_plan, previous_report, frozen_candidate)
    else:
        if report_path.exists() or ledger_path.exists() or plan_path.exists():
            raise RuntimeError("GEPA study already has state; use --resume-internal-study-id after a failed gate")
        write_json(plan_path, plan)
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(plan["plan_sha256"])
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path.resolve())
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(plan["maximum_configured_model_calls"])
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(plan["maximum_physical_http_requests"])
    os.environ["RSI_CALL_LEDGER_MAX_COMPONENT_TASK_ANSWER"] = str(
        plan["maximum_task_model_calls"] + plan["logical_retry_reserve"])
    os.environ["RSI_CALL_LEDGER_MAX_COMPONENT_GEPA_REFLECTION"] = str(
        plan["gepa_proposal_call_budget"])
    os.environ["RSI_CALL_LEDGER_MAX_COMPONENT_GROUNDING"] = str(
        plan["candidate_grounding_call_budget"])
    baseline_checkpoint = study_dir / "internal-baseline.jsonl"
    candidate_checkpoint = study_dir / "internal-candidate.jsonl"

    if not resume_internal:
        search_report_paths: list[Path] = []

        def record_search_report_path(path: Path) -> None:
            search_report_paths.append(path)

        try:
            candidate = search_prompt_candidate(
                cfg, state, [DATASET], len(train_ids) + len(search_ids), args.seed,
                failure_ids=[], report_path_callback=record_search_report_path)
        except Exception as exc:
            write_json(report_path, _failure_report(
                plan, study_id, "gepa_search_failed", ledger_path,
                error_type=type(exc).__name__, error_detail_omitted=True))
            return 2
        if candidate is None:
            write_json(report_path, _failure_report(
                plan, study_id, "gepa_no_candidate", ledger_path,
                search_report=(str(search_report_paths[-1]) if search_report_paths else None)))
            return 0

        gate_examples = scopes["mipro_train"] + scopes["search_validation"]
        try:
            with call_component("grounding"):
                grounding = run_prompt_grounding_review(
                    candidate.new_content, cfg, "system_prompt")
            gates = validate_task_input_prompt(
                candidate.new_content, gate_examples, cfg, state,
                grounding_review={**grounding,
                                 "model_id": grounding.get("model_id"),
                                 "artifact_type": "system_prompt",
                                 "method": "separate measured verifier call; candidate artifact only, no benchmark evidence supplied",
                                 "assertion_scan_complete": bool(grounding.get("assertion_scan_complete"))},
                candidate_id=candidate.candidate_id)
        except Exception as exc:
            write_json(report_path, _failure_report(
                plan, study_id, "candidate_gate_failed", ledger_path,
                candidate=candidate.to_dict(),
                error_type=type(exc).__name__, error_detail_omitted=True))
            return 2
        if not gates.get("passed"):
            write_json(report_path, _failure_report(
                plan, study_id, "candidate_gate_rejected", ledger_path,
                candidate=candidate.to_dict(), candidate_gates=gates))
            return 0
        write_json(candidate_path, {"candidate_id": candidate.candidate_id,
                                    "candidate": candidate.to_dict(),
                                    "candidate_gates": gates,
                                    "plan_sha256": plan["plan_sha256"]})

    candidate_state = ArtifactState(
        release_id=state.release_id, system_prompt=candidate.new_content,
        registry=state.registry, task_demos=state.task_demos)
    seeds = list(args.execution_labels)
    internal = scopes["internal_confirmation"]
    task_by_internal = {str(row["task_id"]): row for row in internal}
    try:
        baseline_rows = _score_repeated_scope(
            cfg, state, internal, seeds, baseline_checkpoint)
        candidate_rows = _score_repeated_scope(
            cfg, candidate_state, internal, seeds, candidate_checkpoint,
            candidate_id=candidate.candidate_id)
    except Exception as exc:
        write_json(report_path, _failure_report(
            plan, study_id, "internal_confirmation_failed", ledger_path,
            candidate_id=candidate.candidate_id,
            candidate=candidate.to_dict(), candidate_gates=gates,
            internal_baseline_checkpoint=str(baseline_checkpoint),
            internal_candidate_checkpoint=str(candidate_checkpoint),
            error_type=type(exc).__name__, error_detail_omitted=True))
        return 2

    baseline_metrics = {str(seed): metrics(rows) for seed, rows in baseline_rows.items()}
    candidate_metrics = {str(seed): metrics(rows) for seed, rows in candidate_rows.items()}
    paired = paired_internal_analysis(baseline_rows, candidate_rows, seeds)
    accepted = all(class_aware_gain(candidate_metrics[str(seed)],
                                    baseline_metrics[str(seed)])
                   for seed in seeds)
    report = _failure_report(
        plan, study_id, "completed", ledger_path,
        candidate=candidate.to_dict(), candidate_id=candidate.candidate_id,
        candidate_gates=gates,
        baseline_internal_metrics_by_execution_label=baseline_metrics,
        candidate_internal_metrics_by_execution_label=candidate_metrics,
        internal_paired_analysis=paired,
        internal_baseline_checkpoint=str(baseline_checkpoint),
        internal_candidate_checkpoint=str(candidate_checkpoint),
        candidate_status="internal_gate_passed" if accepted else "internal_gate_rejected",
        eligible_for_official_selection_gate=bool(accepted),
        execution_labels_are_independent_rng_seeds=False,
        internal_scope_task_count=len(task_by_internal),
        search_report=candidate.generator.get("search_report"),
    )
    write_json(report_path, report)
    print(json.dumps({"report": str(report_path),
                      "candidate_status": report["candidate_status"],
                      "eligible_for_official_selection_gate": accepted}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
