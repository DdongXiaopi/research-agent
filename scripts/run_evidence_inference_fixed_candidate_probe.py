#!/usr/bin/env python3
"""Independently confirm a frozen, preselected EI MIPRO candidate.

The runner performs no optimization. Its one-shot cohort must have been frozen
for the exact candidate after its internal preselection and must exclude every
source-candidate cohort scope. It never reads official selection/test rows.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_evidence_inference_mipro_search as mipro  # noqa: E402
from src.config import config_fingerprint  # noqa: E402
from src.contracts.types import Candidate, content_id, content_sha256  # noqa: E402
from src.data.evidence_inference_v2 import DATASET, load_tasks  # noqa: E402
from src.evolution.evidence_inference_confirmation import (  # noqa: E402
    validate_fresh_scope_independence,
)
from src.evolution.task_program import serialize_task_program  # noqa: E402
from src.eval.task_rows import validate_task_answer_row  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.runtime.runner import ArtifactState, TaskRunner  # noqa: E402
from src.util import (artifacts_path, read_json, sha256_file, sha256_obj,
                      write_json)  # noqa: E402

DEFAULT_COHORT = ROOT / (
    "artifacts/probe/evidence_inference_nested_cohort_"
    "fixed_candidate_replication_20261007.json"
)
DEFAULT_CANDIDATE_REPORT = ROOT / (
    "artifacts/sandboxes/ei-mipro-live-v2-20261005/probe/"
    "evidence_inference_mipro/718e2886ef592166/report.json"
)
DEFAULT_LOGICAL_BUDGET = 1100
DEFAULT_PHYSICAL_BUDGET = 3000
ARM_ORDER_POLICY = "per-seed-task-balanced-block-v1"


def request_upper_bound(task_count: int, execution_label_count: int) -> int:
    """Two arms, one task answer and one model-router call per row."""
    if task_count <= 0 or execution_label_count <= 0:
        raise ValueError("fixed-candidate replication requires tasks and execution labels")
    return int(task_count) * int(execution_label_count) * 2 * 2


def build_arm_order_schedule(task_ids: list[str], execution_labels: list[int],
                             schedule_seed: int) -> dict[str, dict[str, list[str]]]:
    """Randomize first arm within task blocks with exact balance per label."""
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("arm schedule requires unique task IDs")
    if not execution_labels or len(execution_labels) != len(set(execution_labels)):
        raise ValueError("arm schedule requires unique execution labels")
    if len(task_ids) % 2:
        raise ValueError("balanced arm schedule requires an even number of tasks")
    result: dict[str, dict[str, list[str]]] = {}
    for label in execution_labels:
        shuffled = sorted(str(value) for value in task_ids)
        random.Random(int(schedule_seed) ^ int(label)).shuffle(shuffled)
        baseline_first = set(shuffled[:len(shuffled) // 2])
        result[str(label)] = {
            task_id: (["baseline", "candidate"] if task_id in baseline_first
                      else ["candidate", "baseline"])
            for task_id in sorted(shuffled)
        }
    return result


def score_balanced_arms(cfg: dict[str, Any], baseline_state: ArtifactState,
                        candidate_state: ArtifactState, scope: list[dict[str, Any]],
                        execution_labels: list[int], baseline_path: Path,
                        candidate_path: Path, schedule: dict[str, dict[str, list[str]]],
                        candidate_id: str) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    """Score paired arms task-by-task in a frozen balanced randomized order."""
    task_by_id = {str(row["task_id"]): row for row in scope}
    task_ids = set(task_by_id)
    if not task_ids or len(task_by_id) != len(scope):
        raise ValueError("balanced confirmation scope must have unique task IDs")
    if set(schedule) != {str(value) for value in execution_labels}:
        raise ValueError("arm-order schedule labels differ from execution labels")
    for label in execution_labels:
        per_label = schedule[str(label)]
        if set(per_label) != task_ids or any(
                arms not in (["baseline", "candidate"], ["candidate", "baseline"])
                for arms in per_label.values()):
            raise ValueError("arm-order schedule does not cover the exact confirmation scope")
        if sum(arms[0] == "baseline" for arms in per_label.values()) != len(task_ids) // 2:
            raise ValueError("arm-order schedule is not balanced")

    baseline_saved = mipro._load_rows(baseline_path, execution_labels, task_by_id)
    candidate_saved = mipro._load_rows(
        candidate_path, execution_labels, task_by_id, candidate_id=candidate_id)
    output_baseline: dict[int, list[dict]] = {}
    output_candidate: dict[int, list[dict]] = {}
    for label in execution_labels:
        providers: dict[str, Any] = {}
        for task_id in sorted(task_ids):
            task = task_by_id[task_id]
            for arm in schedule[str(label)][task_id]:
                saved = baseline_saved if arm == "baseline" else candidate_saved
                key = (int(label), task_id)
                if key in saved:
                    continue
                if arm not in providers:
                    providers[arm] = mipro.build_provider(cfg, seed=int(label))
                state = baseline_state if arm == "baseline" else candidate_state
                result = TaskRunner(
                    cfg, state, provider=providers[arm], seed=int(label),
                ).run(task)
                row = validate_task_answer_row(
                    result.to_dict(), task,
                    context=f"EI-independent-confirmation:{label}:{task_id}:{arm}",
                    require_reported_gold_index=True)
                row.update({
                    "seed": int(label),
                    "gold_label": str(task["options"][task["correct_index"]]),
                })
                if arm == "candidate":
                    row["candidate_id"] = candidate_id
                output_path = baseline_path if arm == "baseline" else candidate_path
                with output_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                saved[key] = row
        output_baseline[int(label)] = [
            baseline_saved[(int(label), task_id)] for task_id in sorted(task_ids)]
        output_candidate[int(label)] = [
            candidate_saved[(int(label), task_id)] for task_id in sorted(task_ids)]
    return output_baseline, output_candidate


def require_plan_challenge(plan: dict[str, Any], expected_plan_sha256: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_plan_sha256 or ""):
        raise ValueError("execution requires --expected-plan-sha256")
    if expected_plan_sha256 != plan.get("plan_sha256"):
        raise ValueError("expected plan hash does not match the frozen fixed-candidate plan")


def require_one_shot_unused(report_path: Path, ledger_path: Path,
                            baseline_path: Path, candidate_path: Path) -> None:
    """Never replay a completed or interrupted independent confirmation cohort."""
    if report_path.exists():
        raise ValueError(
            "independent confirmation is one-shot; this cohort already has a report")
    if any(path.exists() and path.stat().st_size > 0
           for path in (ledger_path, baseline_path, candidate_path)):
        raise ValueError(
            "independent confirmation has partial execution artifacts; freeze a new cohort")


def verify_candidate_report(report: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    """Return frozen prompt/demos only for a current internally preselected candidate."""
    if report.get("protocol") != mipro.EI_MIPRO_PROTOCOL:
        raise ValueError("candidate report is not from the current EI MIPRO protocol")
    if (report.get("status") != "completed"
            or report.get("candidate_status") != "internal_gate_passed"
            or report.get("eligible_for_official_selection_gate") is not True):
        raise ValueError("confirmation requires a completed internally preselected candidate")
    if (report.get("selection_gate_read") is not False
            or report.get("official_validation_read") is not False
            or report.get("official_test_read") is not False
            or report.get("confirmation_eligible") is not False):
        raise ValueError("candidate report does not attest that confirmation data stayed sealed")
    plan = report.get("plan") or {}
    if (plan.get("protocol") != mipro.EI_MIPRO_PROTOCOL
            or plan.get("plan_sha256") != mipro.compute_plan_sha256(plan)
            or plan.get("cohort_sha256") is None):
        raise ValueError("candidate report plan identity is invalid")
    artifact = report.get("candidate") or {}
    prompt = str(artifact.get("system_prompt") or "")
    demos = artifact.get("task_demos")
    if not prompt or not isinstance(demos, list) or not demos:
        raise ValueError("frozen MIPRO candidate lacks its prompt or labeled demos")
    prompt_hash = content_sha256(prompt.strip())
    instruction_gate = next((row for row in report.get("candidate_instruction_gates", [])
                             if row.get("content_sha256") == prompt_hash
                             and row.get("passed") is True), None)
    if not instruction_gate:
        raise ValueError("frozen candidate prompt lacks a passing static/grounding gate")
    grounding = instruction_gate.get("grounding_review") or {}
    if grounding.get("passed") is not True:
        raise ValueError("frozen candidate prompt grounding review did not pass")
    if grounding.get("candidate_sha256") != content_sha256(prompt.strip()):
        raise ValueError("frozen candidate grounding review belongs to different prompt content")
    if not report.get("candidate_program_gates") or not all(
            row.get("passed") is True for row in report["candidate_program_gates"]):
        raise ValueError("frozen task-program candidates do not all have passing pre-evaluation gates")
    return prompt, prompt_hash, [dict(demo) for demo in demos]


def _load_fresh_scope(cohort_path: Path):
    mipro.configure_dataset_adapter("evidence_inference")
    train_rows = load_tasks(splits=["dev_optimize"])
    cohort, scopes, task_by_id = mipro.load_frozen_cohort(cohort_path, train_rows)
    fresh = scopes["internal_confirmation"]
    if len(fresh) < 90:
        raise ValueError("fresh replication internal scope must contain at least 90 tasks")
    if len({str(row["task_id"]) for row in fresh}) != len(fresh):
        raise ValueError("fresh replication scope contains duplicate task IDs")
    return cohort, scopes, task_by_id, fresh


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("fixed-candidate EI replication requires an isolated RSI_HOME")
    if not args.execution_labels or len(set(args.execution_labels)) != len(args.execution_labels):
        raise ValueError("execution labels must be unique")
    if args.logical_call_budget <= 0 or args.physical_call_budget <= 0:
        raise ValueError("request budgets must be positive")
    if not args.validate_only and not os.environ.get("SILICONFLOW_API_KEY"):
        raise RuntimeError("live replication requires SILICONFLOW_API_KEY in the launcher environment")

    cohort, scopes, task_by_id, fresh = _load_fresh_scope(args.cohort.resolve())
    source_report = read_json(args.candidate_report.resolve())
    prompt, prompt_hash, demos = verify_candidate_report(source_report)
    source_cohort_path = Path(str(source_report.get("cohort_path") or ""))
    if not source_cohort_path.is_file():
        raise ValueError("frozen candidate's source cohort is missing")
    source_cohort = read_json(source_cohort_path)
    source_report_sha256 = sha256_file(args.candidate_report.resolve())
    validate_fresh_scope_independence(
        cohort, source_cohort, source_report.get("plan") or {},
        candidate_id=str(source_report.get("candidate_id") or ""),
        candidate_report_sha256=source_report_sha256)
    source_train_ids = set((source_cohort.get("scopes") or {}).get(
        "mipro_train", {}).get("task_ids") or [])
    if any(str(demo.get("task_id") or "") not in source_train_ids for demo in demos):
        raise ValueError("frozen candidate contains a demo outside its original MIPRO train scope")
    serialized = serialize_task_program(prompt, demos)
    candidate_id = content_id("cand", mipro.RELEASE_ID, "task_program", serialized)
    if candidate_id != source_report.get("candidate_id"):
        raise ValueError("reconstructed candidate ID differs from the frozen MIPRO report")

    cfg = mipro.prepare_config(
        args.config, proposer_calls=1, routing_mode=args.routing_mode,
        task_model_id=args.task_model_id,
    )
    cfg["runtime"]["pi_request_cooldown_seconds"] = args.cooldown_seconds
    cfg["runtime"]["http_no_cache_required"] = True
    state = load_state(mipro.RELEASE_ID)
    if state.release_id != mipro.RELEASE_ID:
        raise ValueError("isolated RSI_HOME has the wrong EI incumbent release")
    active_path = artifacts_path("releases", "ACTIVE")
    if not active_path.is_file() or active_path.read_text(encoding="utf-8").strip() != mipro.RELEASE_ID:
        raise ValueError("isolated RSI_HOME ACTIVE does not name the EI incumbent release")

    program_cfg = dict(cfg)
    program_cfg["evolution"] = {
        **cfg.get("evolution", {}), "allowed_artifacts": ["task_program"]}
    candidate = Candidate(
        candidate_id=candidate_id,
        parent_release_id=mipro.RELEASE_ID,
        artifact_type="task_program",
        artifact_id="task_program",
        diff="frozen MIPRO prompt × original train-only demos; no retuning",
        new_content=serialized,
        content_sha256=content_sha256(serialized),
    )
    gate = static_gate(candidate, program_cfg, state)
    if gate.get("passed") is not True:
        raise ValueError("frozen candidate no longer passes the static task-program gate")
    candidate_state = ArtifactState(
        release_id=state.release_id,
        system_prompt=prompt,
        registry=state.registry,
        task_demos=demos,
    )

    operation_upper = request_upper_bound(len(fresh), len(args.execution_labels))
    physical_by_component = mipro.physical_http_attempt_upper_bounds(
        cfg,
        task_evaluations=len(fresh) * len(args.execution_labels) * 2,
        router_calls=(len(fresh) * len(args.execution_labels) * 2
                      if args.routing_mode == "model" else 0),
        proposer_calls=0,
        grounding_calls=0,
    )
    physical_upper = sum(physical_by_component.values())
    if operation_upper > args.logical_call_budget:
        raise ValueError("logical request budget is below the frozen replication lower bound")
    if physical_upper > args.physical_call_budget:
        raise ValueError("physical request budget is below the configured retry ceiling")

    task_ids = sorted(str(row["task_id"]) for row in fresh)
    arm_order_schedule = build_arm_order_schedule(
        task_ids, args.execution_labels, args.arm_order_seed)
    arm_order_schedule_sha256 = sha256_obj(arm_order_schedule)

    sandbox_root = Path(os.environ["RSI_HOME"]).resolve()
    study_dir = (sandbox_root / "artifacts" / "probe"
                 / "evidence_inference_fixed_candidate"
                 / f"{cohort['cohort_sha256'][:16]}-balanced-v2")
    study_dir.mkdir(parents=True, exist_ok=True)
    plan_path = study_dir / "plan.json"
    ledger_path = study_dir / "request-ledger.jsonl"
    baseline_path = study_dir / "baseline.jsonl"
    candidate_path = study_dir / "candidate.jsonl"
    report_path = study_dir / "report.json"
    plan = {
        "schema_version": 1,
        "protocol": "evidence-inference-independent-confirmation-v2",
        "candidate_id": candidate_id,
        "candidate_source_report_sha256": sha256_obj(source_report),
        "candidate_source_report_file_sha256": source_report_sha256,
        "candidate_source_report_path": str(args.candidate_report.resolve()),
        "candidate_source_protocol": source_report["protocol"],
        "candidate_source_status": source_report["candidate_status"],
        "candidate_content_sha256": content_sha256(serialized),
        "candidate_prompt_sha256": prompt_hash,
        "cohort_sha256": cohort["cohort_sha256"],
        "cohort_path": str(args.cohort.resolve()),
        "cohort_file_sha256": sha256_file(args.cohort.resolve()),
        "source_cohort_sha256": source_cohort["cohort_sha256"],
        "source_cohort_file_sha256": sha256_file(source_cohort_path),
        "scopes": {name: len(rows) for name, rows in scopes.items()},
        "task_count": len(fresh),
        "task_ids": task_ids,
        "article_count": len({str(row["group_id"]) for row in fresh}),
        "class_counts": cohort["scopes"]["internal_confirmation"]["class_counts"],
        "minimum_annotation_votes": cohort["minimum_annotation_votes"],
        "execution_labels": list(args.execution_labels),
        "execution_labels_are_independent_rng_seeds": False,
        "arm_order_policy": ARM_ORDER_POLICY,
        "arm_order_seed": int(args.arm_order_seed),
        "arm_order_schedule_sha256": arm_order_schedule_sha256,
        "arm_order_schedule": arm_order_schedule,
        "task_model": cfg["models"]["task"]["model_id"],
        "router_model": cfg["skills"]["router_model_id"],
        "routing_mode": args.routing_mode,
        "retrieval_used": False,
        "candidate_tuning_or_optimization": False,
        "confirmation_policy": "class-aware-gain-v1",
        "official_validation_read": False,
        "official_test_read": False,
        "selection_gate_read": False,
        "eligible_for_confirmation": False,
        "maximum_logical_operations": operation_upper,
        "logical_budget": args.logical_call_budget,
        "maximum_physical_attempts_by_component": physical_by_component,
        "maximum_physical_attempts": physical_upper,
        "physical_budget": args.physical_call_budget,
        "config_sha256": config_fingerprint(cfg),
        "source_plan_sha256": (source_report.get("plan") or {}).get("plan_sha256"),
    }
    plan["plan_sha256"] = sha256_obj(plan)
    require_one_shot_unused(report_path, ledger_path, baseline_path, candidate_path)
    if plan_path.exists():
        frozen = read_json(plan_path)
        if frozen != plan:
            raise ValueError("saved fixed-candidate plan differs from current inputs")
    else:
        if ledger_path.exists() and ledger_path.stat().st_size:
            raise ValueError("request ledger exists without its frozen fixed-candidate plan")
        write_json(plan_path, plan)
    if args.validate_only:
        return {
            "status": "validated_not_run",
            "plan": str(plan_path),
            "plan_sha256": plan["plan_sha256"],
            "candidate_id": candidate_id,
            "task_count": len(fresh),
            "class_counts": plan["class_counts"],
            "logical_call_upper_bound": operation_upper,
            "physical_attempt_upper_bound": physical_upper,
            "model_calls": 0,
            "independent_confirmation_completed": False,
            "independent_confirmation_passed": False,
        }

    require_plan_challenge(plan, args.expected_plan_sha256)
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(plan["plan_sha256"])
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path)
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(args.logical_call_budget)
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(args.physical_call_budget)
    baseline_rows, candidate_rows = score_balanced_arms(
        cfg, state, candidate_state, fresh, args.execution_labels,
        baseline_path, candidate_path, arm_order_schedule, candidate_id)
    baseline_metrics = {
        str(label): mipro.metrics(rows) for label, rows in baseline_rows.items()}
    candidate_metrics = {
        str(label): mipro.metrics(rows) for label, rows in candidate_rows.items()}
    confirmation_passed = all(
        mipro.class_aware_gain(candidate_metrics[str(label)],
                               baseline_metrics[str(label)])
        for label in args.execution_labels)
    result = {
        "status": ("independent_confirmation_passed" if confirmation_passed
                   else "independent_confirmation_failed"),
        "plan": plan,
        "plan_path": str(plan_path),
        "candidate_id": candidate_id,
        "candidate_status_in_source_experiment": source_report["candidate_status"],
        "candidate_static_gate": gate,
        "baseline_metrics_by_execution_label": baseline_metrics,
        "candidate_metrics_by_execution_label": candidate_metrics,
        "paired_analysis": mipro.paired_internal_analysis(
            baseline_rows, candidate_rows, args.execution_labels),
        "baseline_rows_path": str(baseline_path),
        "candidate_rows_path": str(candidate_path),
        "retrieval_used": False,
        "selection_gate_read": False,
        "official_validation_read": False,
        "official_test_read": False,
        "independent_confirmation_completed": True,
        "independent_confirmation_passed": confirmation_passed,
        "confirmation_eligible": confirmation_passed,
        "request_accounting": summarize_ledger(ledger_path),
    }
    write_json(report_path, result)
    return {
        "report": str(report_path),
        "status": result["status"],
        "candidate_id": candidate_id,
        "logical_operations": result["request_accounting"]["logical_operations"],
        "physical_http_attempts": result["request_accounting"]["physical_http_attempts"],
        "independent_confirmation_completed": True,
        "independent_confirmation_passed": confirmation_passed,
        "eligible_for_confirmation": confirmation_passed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--candidate-report", type=Path, default=DEFAULT_CANDIDATE_REPORT)
    parser.add_argument("--config", default="artifacts/sandboxes/ei-mipro-live-v2-20261005/frozen-run-config.yaml")
    parser.add_argument("--execution-labels", type=int, nargs="+",
                        default=[202610081, 202610082, 202610083])
    parser.add_argument("--logical-call-budget", type=int, default=DEFAULT_LOGICAL_BUDGET)
    parser.add_argument("--physical-call-budget", type=int, default=DEFAULT_PHYSICAL_BUDGET)
    parser.add_argument("--task-model-id", choices=["Qwen/Qwen3-32B", "Qwen/Qwen3-8B"],
                        default="Qwen/Qwen3-32B")
    parser.add_argument("--routing-mode", choices=["lexical", "model"], default="model")
    parser.add_argument("--cooldown-seconds", type=int, default=6)
    parser.add_argument("--arm-order-seed", type=int, default=20261011)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-plan-sha256", default="",
                        help="required with --execute; copy from the zero-call preflight")
    args = parser.parse_args(argv)
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256 or ""):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    print(json.dumps(run(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
