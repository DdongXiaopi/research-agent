#!/usr/bin/env python3
"""Aggregate-only selection-gate evaluation for an internally accepted EI MIPRO candidate.

Default behavior is metadata-only. Official validation rows are loaded only with --execute.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_evidence_inference_mipro_search import prepare_config  # noqa: E402
from src.config import config_fingerprint  # noqa: E402
from src.data.evidence_inference_v2 import DATASET, load_tasks  # noqa: E402
from src.eval.gate_verdict import compare_on_split  # noqa: E402
from src.evolution.evidence_inference_promotion import (  # noqa: E402
    candidate_static_gate, selection_gate_preflight, validate_search_report,
)
from src.evolution.evidence_inference_confirmation import (  # noqa: E402
    load_confirmation_attestation,
)
from src.evolution.task_program import parse_task_program  # noqa: E402
from src.governance.budget import BudgetGateError, check_tolerances  # noqa: E402
from src.governance.promotion import promotion_decision  # noqa: E402
from src.governance import review as review_api  # noqa: E402
from src.governance.release import current_release_id, load_state  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.runtime.runner import ArtifactState  # noqa: E402
from src.util import (artifacts_path, read_json, sha256_file, sha256_obj,
                      write_json)  # noqa: E402


def build_execution_plan(*, source_path: Path, source: dict, candidate,
                         search_plan: dict, cfg: dict, parent, preflight: dict,
                         max_task_evaluations: int,
                         max_model_calls: int,
                         confirmation_attestation: dict | None = None) -> dict:
    """Bind one aggregate-only official validation pass without loading rows."""
    models = cfg.get("models") or {}
    task_model = models.get("task") or {}
    router_model = models.get("skill_router") or task_model
    manifest_sha = str(preflight.get("manifest_sha256") or "")
    split_sha = str(preflight.get("selection_file_sha256") or "")
    if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
           for value in (manifest_sha, split_sha)):
        raise ValueError("selection-gate manifest/split SHA-256 is missing or invalid")
    max_attempts = max(
        1, int(task_model.get("max_retries", 1)),
        int(router_model.get("max_retries", 1)))
    unsigned = {
        "schema": "evidence-inference-selection-gate-execution-plan-v1",
        "role": "one_shot_aggregate_only_candidate_gate",
        "source_report_sha256": sha256_file(source_path),
        "source_study_id": source.get("study_id"),
        "source_candidate_id": source.get("candidate_id"),
        "source_search_plan_sha256": search_plan.get("plan_sha256"),
        "independent_confirmation": confirmation_attestation,
        "candidate_id": candidate.candidate_id,
        "candidate_content_sha256": candidate.content_sha256,
        "parent_release_id": parent.release_id,
        "registry_sha256": parent.registry.fingerprint(),
        "config_sha256": config_fingerprint(cfg),
        "dataset": preflight["dataset"],
        "split": preflight["split"],
        "task_count": int(preflight["task_count"]),
        "selection_ids_sha256": preflight["manifest_selection_ids_sha256"],
        "selection_manifest_sha256": manifest_sha,
        "selection_rows_sha256": split_sha,
        "execution_labels": list(preflight["execution_labels"]),
        "execution_labels_are_independent_rng_seeds": False,
        "task_evaluation_estimate": int(
            preflight["baseline_candidate_task_evaluations"]),
        "model_call_estimate": int(preflight["estimated_total_model_calls"]),
        "candidate_static_gate_sha256": sha256_obj(
            preflight.get("static_gate") or {}),
        "candidate_static_gate_passed": (
            preflight.get("static_gate_passed") is True),
        "max_task_evaluations": int(max_task_evaluations),
        "task_evaluation_cap_passed": (
            preflight.get("task_evaluation_cap_passed") is True),
        "model_call_cap_passed": preflight.get("model_call_cap_passed") is True,
        "logical_request_cap": int(max_model_calls),
        "physical_http_attempt_cap": int(max_model_calls * max_attempts),
        "http_attempts_per_operation_upper_bound": max_attempts,
        "task_model_id": str(task_model.get("model_id") or ""),
        "router_model_id": str(router_model.get("model_id") or ""),
        "request_ledger_required": True,
        "selection_gate_read_during_preflight": False,
        "selection_gate_read_if_executed": True,
        "official_test_read": False,
        "per_task_output_written": False,
        "live_authorized": False,
        "explicit_user_approval_required_before_execution": True,
    }
    return {**unsigned, "plan_sha256": sha256_obj(unsigned)}


def verify_execution_plan(plan: dict, expected_sha256: str) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    actual = sha256_obj(unsigned)
    if plan.get("plan_sha256") != actual:
        raise ValueError("selection-gate plan hash is internally inconsistent")
    if actual != expected_sha256:
        raise ValueError("expected plan hash does not match current selection-gate plan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--confirmation-report", type=Path,
                        help="passing post-preselection source-disjoint confirmation report")
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--max-task-evaluations", type=int, default=2500)
    parser.add_argument(
        "--max-model-calls", type=int, default=492,
        help=("model-request ceiling for this invocation; the default is the approved "
              "492-call MIPRO budget, so a larger official selection gate requires an "
              "explicit higher limit"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true",
                      help="Write a zero-call plan without loading official validation rows")
    mode.add_argument("--execute", action="store_true",
                      help="Run the approved aggregate validation plan")
    parser.add_argument("--expected-plan-sha256", default="",
                        help="required with --execute; exact hash from reviewed preflight")
    args = parser.parse_args(argv)
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    if not args.source_report.is_file():
        raise FileNotFoundError(args.source_report)
    if args.max_task_evaluations <= 0 or args.max_model_calls <= 0:
        raise ValueError("evaluation/model-call caps must be positive")
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("EI selection gate requires an isolated RSI_HOME")

    source = read_json(args.source_report)
    release_id = str((source.get("plan") or {}).get("release_id") or "")
    parent = load_state(release_id)
    if current_release_id() != release_id:
        raise ValueError("isolated ACTIVE release no longer matches the MIPRO candidate parent")
    routing_mode = str((source.get("plan") or {}).get("routing_mode") or "model")
    plan = source.get("plan") or {}
    proposer_calls = int(plan.get("proposer_call_budget") or 0)
    if proposer_calls <= 0:
        raise ValueError("MIPRO source plan is missing its proposer-call budget")
    cfg = prepare_config(args.config, proposer_calls=proposer_calls,
                         routing_mode=routing_mode)
    cfg["evaluation"]["datasets"] = [DATASET]
    cfg["evaluation"]["require_split_freeze"] = False
    # The MIPRO artifact is only valid under the exact task/runtime protocol
    # that produced it. In particular, runtime demo rendering limits affect
    # which examples actually reached the task model. Check before loading even
    # the development rows, and before any optional official selection data.
    expected_config_sha = str(plan.get("config_sha256") or "")
    if not expected_config_sha or config_fingerprint(cfg) != expected_config_sha:
        raise ValueError(
            "MIPRO source config fingerprint differs from the current task protocol; "
            "rerun search under the current config before selection-gate evaluation")
    cfg["gates"]["accept"]["tolerance"] = {
        "cost_pct": 15, "required": ["cost"], "strict": True,
    }

    # This reads only the official training-derived optimize partition.
    candidate, plan = validate_search_report(
        source, load_tasks(splits=["dev_optimize"]), parent)
    confirmation_attestation = None
    if args.confirmation_report is not None:
        confirmation_attestation = load_confirmation_attestation(
            args.confirmation_report, candidate_id=candidate.candidate_id,
            candidate_content_sha256=candidate.content_sha256,
            source_plan_sha256=str(plan.get("plan_sha256") or ""),
            source_report_path=args.source_report,
            source_report_sha256=sha256_file(args.source_report))
    preflight = selection_gate_preflight(plan)
    preflight.update({
        "candidate_id": candidate.candidate_id,
        "independent_confirmation": confirmation_attestation,
        "independent_confirmation_gate_passed": confirmation_attestation is not None,
        "static_gate": candidate_static_gate(candidate, cfg, parent),
        "execute_requested": bool(args.execute),
    })
    preflight["static_gate_passed"] = bool(preflight["static_gate"].get("passed"))
    preflight["task_evaluation_cap_passed"] = (
        preflight["baseline_candidate_task_evaluations"] <= args.max_task_evaluations)
    preflight["model_call_cap_passed"] = (
        preflight["estimated_total_model_calls"] <= args.max_model_calls)
    execution_plan = build_execution_plan(
        source_path=args.source_report, source=source, candidate=candidate,
        search_plan=plan, cfg=cfg, parent=parent, preflight=preflight,
        confirmation_attestation=confirmation_attestation,
        max_task_evaluations=args.max_task_evaluations,
        max_model_calls=args.max_model_calls)
    plan_path = artifacts_path(
        "probe", "evidence_inference_selection_gate",
        f"plan-{execution_plan['plan_sha256']}.json")
    if plan_path.exists():
        if read_json(plan_path) != execution_plan:
            raise RuntimeError("saved selection-gate plan differs from recomputed plan")
    else:
        write_json(plan_path, execution_plan)
    preflight["execution_plan"] = execution_plan
    preflight["execution_plan_path"] = str(plan_path)
    preflight["execution_plan_sha256"] = execution_plan["plan_sha256"]
    if not preflight["static_gate_passed"]:
        print(json.dumps({"status": "static_gate_blocked", **preflight}, ensure_ascii=False))
        return 2
    if not preflight["task_evaluation_cap_passed"] or not preflight["model_call_cap_passed"]:
        print(json.dumps({"status": "budget_blocked", **preflight}, ensure_ascii=False))
        return 2
    if args.execute and confirmation_attestation is None:
        raise ValueError(
            "selection-gate execution requires a passing independent confirmation report")
    if args.validate_only:
        status = ("preflight_valid" if confirmation_attestation is not None
                  else "independent_confirmation_required")
        print(json.dumps({"status": status, **preflight}, ensure_ascii=False))
        return 0

    verify_execution_plan(execution_plan, args.expected_plan_sha256)
    task_model = (cfg.get("models") or {}).get("task") or {}
    key_env = str(task_model.get("api_key_env") or "").strip()
    if not key_env or not os.environ.get(key_env):
        raise RuntimeError("selection-gate execution is missing the configured task-model credential")

    run_dir = artifacts_path(
        "probe", "evidence_inference_selection_gate",
        "execute-" + uuid.uuid4().hex[:12])
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = run_dir / "call-ledger.jsonl"
    ledger_env = {
        "RSI_ACTIVE_EXECUTION_PLAN_SHA256": str(execution_plan["plan_sha256"]),
        "RSI_CALL_LEDGER_PATH": str(ledger_path),
        "RSI_CALL_LEDGER_MAX_LOGICAL": str(execution_plan["logical_request_cap"]),
        "RSI_CALL_LEDGER_MAX_PHYSICAL": str(
            execution_plan["physical_http_attempt_cap"]),
        "RSI_HTTP_NO_CACHE": "1",
    }
    previous_ledger_env = {key: os.environ.get(key) for key in ledger_env}
    os.environ.update(ledger_env)

    # The official validation rows are not materialized until the explicit --execute.
    try:
        gate_rows = load_tasks(splits=["selection_gate"])
        if len(gate_rows) != preflight["task_count"]:
            raise ValueError("selection_gate task count differs from frozen manifest")
        program = parse_task_program(candidate.new_content)
        challenger = ArtifactState(
            release_id=candidate.candidate_id,
            system_prompt=program["system_prompt"],
            registry=parent.registry,
            task_demos=program["task_demos"],
        )
        comparison = compare_on_split(
            cfg, parent, challenger, [DATASET], split="selection_gate",
            seeds=preflight["execution_labels"], limit=None,
        )
        call_ledger = summarize_ledger(ledger_path)
    finally:
        for key, value in previous_ledger_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    # A configured cost tolerance is not enough: every execution label must
    # have provider-priced USD before a review request can be generated.
    cost_reports = {}
    for label, row in comparison["per_dataset"][DATASET]["per_seed"].items():
        pair = row.get("cost") or {}
        try:
            cost_reports[label] = check_tolerances(
                pair.get("baseline") or {}, pair.get("candidate") or {},
                copy.deepcopy(cfg), kind="http")
        except BudgetGateError as exc:
            cost_reports[label] = {
                "passed": False, "provider_kind": "http",
                "not_measurable": ["cost"], "error": str(exc),
            }
    all_costs_passed = bool(cost_reports) and all(
        report.get("passed") is True and any(
            item.get("quantity") == "cost" and item.get("status") == "pass"
            for item in report.get("items", []))
        for report in cost_reports.values())
    promotion = promotion_decision(
        cfg, comparison, protocol_checked=False, assessment_scope="exploratory",
        tolerance_report={"authoritative_cost": {
            "passed": all_costs_passed, "per_execution_label": cost_reports,
        }},
    )
    comparison.update({
        "protocol": plan["protocol"], "candidate_id": candidate.candidate_id,
        "execution_plan": execution_plan,
        "independent_confirmation": confirmation_attestation,
        "execution_plan_sha256": execution_plan["plan_sha256"],
        "call_ledger": call_ledger,
        "source_study_id": source.get("study_id"),
        "selection_gate_read": True, "official_validation_read": True,
        "official_test_read": False,
        "validation_rows_loaded": True, "retrieval_used": False,
        "per_task_output_written": False,
        "promotion": promotion,
    })
    review_request_path = None
    cost_exception_reviewable = (
        promotion.get("cost_overrun_exception_reviewable") is True)
    if promotion.get("promotable") is True or cost_exception_reviewable:
        candidate_path = artifacts_path("candidates", f"{candidate.candidate_id}.json")
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(candidate_path, candidate.to_dict())
        primary_block = comparison["per_dataset"][DATASET]
        primary_row = primary_block["per_seed"][str(primary_block["primary_seed"])]
        request = review_api.build_request(
            candidate,
            optimize_metrics={
                "source_study_id": source.get("study_id"),
                "baseline_internal_metrics_by_execution_label":
                    source.get("baseline_internal_metrics_by_execution_label"),
                "candidate_internal_metrics_by_execution_label":
                    source.get("candidate_internal_metrics_by_execution_label"),
            },
            gate_metrics=comparison, rollback_point=parent.release_id,
            static_gate=preflight["static_gate"],
        )
        tolerance_report = promotion["checks"][-1]["detail"]
        request.update({
            "verdict": "proposed_accept",
            "evidence_type": "aggregate_selection_gate_v1",
            "cost_overrun_review": promotion.get("cost_overrun_exception"),
            "evaluation_summary": {
                "official_validation_read": True,
                "official_test_read": False,
                "per_task_output_written": False,
                "promotion_promotable": promotion.get("promotable") is True,
                "cost_exception_reviewable": cost_exception_reviewable,
                "protocol_checked": False,
                "execution_labels": preflight["execution_labels"],
                "execution_plan_sha256": execution_plan["plan_sha256"],
                "source_search_plan_sha256": plan.get("plan_sha256"),
                "source_report_sha256": sha256_file(args.source_report),
                "source_report_path": str(args.source_report.resolve()),
                "independent_confirmation": confirmation_attestation,
                "request_caps": {
                    "logical": execution_plan["logical_request_cap"],
                    "physical_http_attempts": execution_plan[
                        "physical_http_attempt_cap"],
                },
                "tolerance_report": tolerance_report,
                "baseline_metrics": primary_row.get("baseline"),
                "candidate_metrics": primary_row.get("candidate"),
            },
        })
        if not review_api.request_effect_gate_ok(request):
            candidate_path.unlink(missing_ok=True)
            raise RuntimeError("selection-gate result failed the release review effect guard")
        review_request_path = str(review_api.write_request(request))
    report_path = artifacts_path(
        "probe", "evidence_inference_selection_gate", f"{candidate.candidate_id}.json")
    write_json(report_path, comparison)
    print(json.dumps({
        "status": "completed", "report_path": str(report_path),
        "candidate_id": candidate.candidate_id,
        "execution_plan_sha256": execution_plan["plan_sha256"],
        "call_ledger": call_ledger,
        "promotable": promotion["promotable"],
        "authoritative_cost_passed": all_costs_passed,
        "official_validation_read": True, "official_test_read": False,
        "review_request_path": review_request_path,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
