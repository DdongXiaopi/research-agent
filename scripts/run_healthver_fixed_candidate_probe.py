"""Compare a frozen HealthVer candidate and incumbent on a fixed diagnostic cohort.

This runner performs no optimization or candidate selection. Public HealthVer
data is exploratory only; it cannot serve as blind confirmation.
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

from scripts import run_evidence_inference_mipro_search as mipro_runner  # noqa: E402
from src.config import config_fingerprint  # noqa: E402
from src.contracts.types import Candidate, content_id, content_sha256  # noqa: E402
from src.data.healthver_task_input import DATASET, LABELS, load_tasks  # noqa: E402
from src.evolution.task_program import serialize_task_program  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.runner import ArtifactState  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.util import read_json, sha256_obj, write_json  # noqa: E402

DEFAULT_COHORT = ROOT / "artifacts/probe/healthver_remaining_support_probe.json"
DEFAULT_CANDIDATE_REPORT = ROOT / (
    "artifacts/sandboxes/healthver-mipro-fresh-components-20261007/probe/"
    "healthver_mipro/05a1f50d2fc0437d/report.json"
)


def scope_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty diagnostic scope")
    correct = sum(bool(row.get("correct")) for row in rows)
    answered = sum(int(row.get("answer_index", -1)) >= 0 for row in rows)
    per_label = {}
    for label in sorted({str(row.get("gold_label") or "") for row in rows}):
        subset = [row for row in rows if str(row.get("gold_label") or "") == label]
        n_correct = sum(bool(row.get("correct")) for row in subset)
        n_answered = sum(int(row.get("answer_index", -1)) >= 0 for row in subset)
        per_label[label] = {
            "n": len(subset),
            "accuracy": n_correct / len(subset),
            "coverage": n_answered / len(subset),
            "answered_precision": n_correct / n_answered if n_answered else 0.0,
        }
    return {
        "n": len(rows), "accuracy": correct / len(rows),
        "coverage": answered / len(rows),
        "answered_precision": correct / answered if answered else 0.0,
        "per_gold_label": per_label,
    }


def verify_cohort(cohort: dict[str, Any]) -> None:
    saved_hash = str(cohort.get("cohort_sha256") or "")
    unhashed = dict(cohort)
    unhashed.pop("cohort_sha256", None)
    if not saved_hash or sha256_obj(unhashed) != saved_hash:
        raise ValueError("diagnostic cohort checksum mismatch")
    if cohort.get("dataset") != DATASET:
        raise ValueError("diagnostic cohort is not HealthVer")
    if cohort.get("eligible_for_confirmation") is not False:
        raise ValueError("HealthVer diagnostic cohort cannot be confirmation eligible")
    if cohort.get("source_test_is_blind_confirmation") is not False:
        raise ValueError("HealthVer source test must be disclosed as non-blind")
    if cohort.get("candidate_runtime_uses_live_retrieval") is not False:
        raise ValueError("diagnostic cohort must use fixed task evidence")
    if int(cohort.get("task_count", 0)) != len(cohort.get("task_ids") or []):
        raise ValueError("diagnostic task count does not match its task ID list")
    if len(set(cohort.get("task_ids") or [])) != int(cohort.get("task_count", 0)):
        raise ValueError("diagnostic cohort contains duplicate task IDs")


def validate_unstarted_resume(ledger_path: Path, result_paths: list[Path]) -> None:
    """Allow execution after zero-call preflight froze the plan, but reject orphan outputs."""
    has_ledger = ledger_path.exists() and ledger_path.stat().st_size > 0
    has_results = any(path.exists() and path.stat().st_size > 0 for path in result_paths)
    if not has_ledger and has_results:
        raise ValueError("saved diagnostic outputs exist without a request ledger")


def require_plan_challenge(plan: dict[str, Any], expected_plan_sha256: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_plan_sha256 or ""):
        raise ValueError("execution requires --expected-plan-sha256")
    if expected_plan_sha256 != plan.get("plan_sha256"):
        raise ValueError("expected plan hash does not match the frozen HealthVer probe plan")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("fixed-candidate probe requires an isolated RSI_HOME")
    if len(args.execution_labels) < 1 or len(set(args.execution_labels)) != len(args.execution_labels):
        raise ValueError("execution labels must be unique")
    if args.logical_call_budget <= 0 or args.physical_call_budget <= 0:
        raise ValueError("request budgets must be positive")

    cohort = read_json(args.cohort)
    verify_cohort(cohort)
    prior_report = read_json(args.candidate_report)
    if prior_report.get("candidate_status") != "internal_gate_rejected":
        raise ValueError("fixed candidate must come from the expected rejected MIPRO study")
    if prior_report.get("candidate_id") != cohort.get("candidate_id"):
        raise ValueError("diagnostic cohort candidate ID differs from frozen MIPRO candidate")
    if prior_report.get("confirmation_eligible") is not False:
        raise ValueError("candidate report must explicitly deny confirmation eligibility")

    mipro_runner.configure_dataset_adapter("healthver")
    cfg = mipro_runner.prepare_config(
        args.config, proposer_calls=1, routing_mode=args.routing_mode,
        task_model_id=args.task_model_id)
    cfg["runtime"]["pi_request_cooldown_seconds"] = args.cooldown_seconds
    cfg["runtime"]["http_no_cache_required"] = True
    state = load_state(mipro_runner.RELEASE_ID)
    if state.release_id != mipro_runner.RELEASE_ID:
        raise ValueError("isolated RSI_HOME has the wrong HealthVer incumbent release")
    # releases/ACTIVE is resolved through RSI_HOME, not the repository root.
    from src.util import artifacts_path  # local import avoids stale cwd assumptions
    active_path = artifacts_path("releases", "ACTIVE")
    if not active_path.is_file() or active_path.read_text(encoding="utf-8").strip() != mipro_runner.RELEASE_ID:
        raise ValueError("isolated RSI_HOME ACTIVE does not name the HealthVer release")

    task_by_id = {str(row["task_id"]): row for row in load_tasks(["dev_optimize"])}
    task_ids = [str(task_id) for task_id in cohort["task_ids"]]
    if any(task_id not in task_by_id for task_id in task_ids):
        raise ValueError("diagnostic cohort contains a missing HealthVer task")
    scope = [task_by_id[task_id] for task_id in task_ids]
    observed_components = sorted({str(row["group_id"]) for row in scope})
    if observed_components != sorted(cohort.get("component_ids") or []):
        raise ValueError("diagnostic cohort connected-component IDs changed")
    observed_labels = {label: 0 for label in LABELS}
    for row in scope:
        observed_labels[str(row["options"][int(row["correct_index"])])] += 1
    if observed_labels != cohort.get("label_counts"):
        raise ValueError("diagnostic cohort labels changed after freeze")

    artifact = prior_report.get("candidate") or {}
    prompt = str(artifact.get("system_prompt") or "")
    demos = list(artifact.get("task_demos") or [])
    if not prompt or not demos:
        raise ValueError("frozen MIPRO candidate lacks prompt or train-only demos")
    prior_cohort = read_json(prior_report["cohort_path"])
    train_ids = set((prior_cohort.get("scopes") or {}).get("mipro_train", {}).get("task_ids", []))
    if any(str(demo.get("task_id")) not in train_ids for demo in demos):
        raise ValueError("candidate demonstration escaped its frozen train component pool")
    serialized = serialize_task_program(prompt, demos)
    candidate_id = content_id("cand", mipro_runner.RELEASE_ID,
                              "task_program", serialized)
    if candidate_id != cohort["candidate_id"]:
        raise ValueError("reconstructed frozen candidate ID differs from diagnostic cohort")

    program_cfg = dict(cfg)
    program_cfg["evolution"] = {**cfg.get("evolution", {}), "allowed_artifacts": ["task_program"]}
    candidate = Candidate(
        candidate_id=candidate_id, parent_release_id=mipro_runner.RELEASE_ID,
        artifact_type="task_program", artifact_id="task_program",
        diff="frozen MIPRO prompt × train-only demos",
        new_content=serialized, content_sha256=content_sha256(serialized),
    )
    gate = static_gate(candidate, program_cfg, state)
    if gate.get("passed") is not True:
        raise ValueError("frozen candidate no longer passes the static task-program gate")
    candidate_state = ArtifactState(
        release_id=state.release_id, system_prompt=prompt,
        registry=state.registry, task_demos=demos,
    )

    sandbox_root = Path(os.environ["RSI_HOME"]).resolve()
    study_dir = sandbox_root / "artifacts" / "probe" / "healthver_fixed_candidate_probe" / cohort["cohort_sha256"][:16]
    study_dir.mkdir(parents=True, exist_ok=True)
    plan_path = study_dir / "plan.json"
    ledger_path = study_dir / "request-ledger.jsonl"
    result_paths = [study_dir / "baseline.jsonl", study_dir / "candidate.jsonl"]
    # One router and one task-answer operation per task and arm and execution
    # label; the wire ceiling allows three attempts per provider operation.
    operation_upper = len(scope) * len(args.execution_labels) * 2 * 2
    physical_upper = operation_upper * 3
    if operation_upper > args.logical_call_budget:
        raise ValueError("logical request budget is below the frozen probe lower bound")
    if physical_upper > args.physical_call_budget:
        raise ValueError("physical request budget is below the configured retry ceiling")
    plan = {
        "schema_version": 1,
        "protocol": "healthver-paired-fixed-candidate-diagnostic-v2",
        "cohort_sha256": cohort["cohort_sha256"],
        "candidate_id": candidate_id,
        "candidate_report_status": prior_report["candidate_status"],
        "labels": list(args.execution_labels),
        "execution_labels_are_independent_rng_seeds": False,
        "task_count": len(scope),
        "task_ids": sorted(task_ids),
        "component_ids": observed_components,
        "label_counts": observed_labels,
        "task_model": args.task_model_id,
        "routing_mode": args.routing_mode,
        "retrieval_used": False,
        "source_upstream_test_materialized": True,
        "source_test_is_blind_confirmation": False,
        "eligible_for_confirmation": False,
        "selection_gate_read": False,
        "candidate_tuning_or_optimization": False,
        "maximum_logical_operations": operation_upper,
        "logical_budget": args.logical_call_budget,
        "maximum_physical_attempts": physical_upper,
        "physical_budget": args.physical_call_budget,
        "config_sha256": config_fingerprint(cfg),
        "cohort_file_sha256": sha256_obj(cohort),
    }
    plan["plan_sha256"] = sha256_obj(plan)
    if plan_path.exists():
        frozen_plan = read_json(plan_path)
        for key in ("protocol", "cohort_sha256", "candidate_id", "labels", "task_ids",
                    "component_ids", "label_counts", "task_model", "routing_mode",
                    "maximum_logical_operations", "logical_budget",
                    "maximum_physical_attempts", "physical_budget",
                    "config_sha256", "cohort_file_sha256"):
            if frozen_plan.get(key) != plan.get(key):
                raise ValueError(f"saved diagnostic plan differs in {key}")
        validate_unstarted_resume(ledger_path, result_paths)
        plan = frozen_plan
    else:
        if ledger_path.exists() and ledger_path.stat().st_size:
            raise ValueError("request ledger exists without its frozen diagnostic plan")
        write_json(plan_path, plan)
    if args.validate_only:
        return {"status": "validated_not_run", "plan": str(plan_path),
                "plan_sha256": plan["plan_sha256"],
                "candidate_id": candidate_id,
                "tasks": len(scope), "components": len(observed_components),
                "labels": observed_labels,
                "logical_call_upper_bound": operation_upper,
                "physical_attempt_upper_bound": physical_upper,
                "model_calls": 0,
                "eligible_for_confirmation": False}
    require_plan_challenge(plan, args.expected_plan_sha256)
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(plan["plan_sha256"])
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path)
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(args.logical_call_budget)
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(args.physical_call_budget)

    baseline_rows = mipro_runner._score_repeated_scope(
        cfg, state, scope, args.execution_labels,
        study_dir / "baseline.jsonl")
    candidate_rows = mipro_runner._score_repeated_scope(
        cfg, candidate_state, scope, args.execution_labels,
        study_dir / "candidate.jsonl", candidate_id=candidate_id)
    report = {
        "status": "completed",
        "plan": plan,
        "plan_path": str(plan_path),
        "candidate_id": candidate_id,
        "candidate_static_gate": gate,
        "baseline_metrics_by_execution_label": {
            str(label): scope_metrics(rows) for label, rows in baseline_rows.items()},
        "candidate_metrics_by_execution_label": {
            str(label): scope_metrics(rows) for label, rows in candidate_rows.items()},
        "paired_analysis": mipro_runner.paired_internal_analysis(
            baseline_rows, candidate_rows, args.execution_labels),
        "baseline_rows_path": str(study_dir / "baseline.jsonl"),
        "candidate_rows_path": str(study_dir / "candidate.jsonl"),
        "retrieval_used": False,
        "selection_gate_read": False,
        "official_confirmation_read": False,
        "eligible_for_confirmation": False,
        "request_accounting": summarize_ledger(ledger_path),
    }
    report_path = study_dir / "report.json"
    write_json(report_path, report)
    return {"report": str(report_path), "status": report["status"],
            "candidate_id": candidate_id,
            "baseline_accuracy": next(iter(report["baseline_metrics_by_execution_label"].values()))["accuracy"],
            "candidate_accuracy": next(iter(report["candidate_metrics_by_execution_label"].values()))["accuracy"],
            "logical_operations": report["request_accounting"]["logical_operations"],
            "physical_http_attempts": report["request_accounting"]["physical_http_attempts"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--candidate-report", type=Path, default=DEFAULT_CANDIDATE_REPORT)
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--execution-labels", type=int, nargs="+", default=[202610091, 202610092])
    parser.add_argument("--logical-call-budget", type=int, default=120)
    parser.add_argument("--physical-call-budget", type=int, default=360)
    parser.add_argument("--task-model-id", choices=["Qwen/Qwen3-32B", "Qwen/Qwen3-8B"],
                        default="Qwen/Qwen3-32B")
    parser.add_argument("--routing-mode", choices=["lexical", "model"], default="model")
    parser.add_argument("--cooldown-seconds", type=int, default=6)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true",
                      help="validate cohort, artifact, gates and budget without model calls")
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
