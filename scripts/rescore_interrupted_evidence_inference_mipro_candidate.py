#!/usr/bin/env python3
"""Finish scoring one gated MIPRO program after an interrupted optimizer trial.

This recovery path never treats the optimizer's partial score as complete. It
re-evaluates the exact saved candidate on the full frozen search scope, then
runs the already-planned internal confirmation only when the full search score
is at least the observed incumbent score. Task rows are checkpointed and can
be resumed without repeating completed rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.recover_evidence_inference_mipro_candidate import (  # noqa: E402
    reconstruct_trial_candidate, require_frozen_plan_challenge,
)
from scripts.run_evidence_inference_mipro_search import (  # noqa: E402
    RELEASE_ID, _score_repeated_scope, class_aware_gain, load_frozen_cohort,
    metrics, paired_internal_analysis, prepare_config,
)
from src.contracts.types import Candidate, content_sha256  # noqa: E402
from src.data.evidence_inference_v2 import load_tasks  # noqa: E402
from src.evolution.task_input_gates import validate_task_demo_rendering  # noqa: E402
from src.evolution.task_program import serialize_task_program  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.runtime.runner import ArtifactState  # noqa: E402
from src.util import read_json, read_jsonl, sha256_file, write_json  # noqa: E402


def failure_report(*, plan: dict, study_id: str, candidate_id: str,
                   source_path: Path, candidate: dict, candidate_sha256: str,
                   saved_gate: dict, static: dict, rendering: dict,
                   checkpoint_path: Path, expected_rows: int, error: Exception,
                   ledger_path: Path) -> dict:
    """Describe an interrupted rescore without exposing partial credit."""
    rows = read_jsonl(checkpoint_path) if checkpoint_path.exists() else []
    return {
        "protocol": plan["protocol"],
        "status": "recovered_trial_search_failed",
        "study_id": study_id,
        "plan": plan,
        "plan_sha256": plan["plan_sha256"],
        "source_failed_report": str(source_path),
        "source_failed_report_sha256": sha256_file(source_path),
        "candidate_id": candidate_id,
        "candidate": candidate,
        "candidate_content_sha256": candidate_sha256,
        "saved_program_gate": saved_gate,
        "rechecked_static_gate": static,
        "rechecked_demo_rendering_gate": rendering,
        "search_validation": {
            "scope": "partial checkpoint only; no score assigned",
            "checkpoint_path": str(checkpoint_path),
            "completed_task_rows": len(rows),
            "expected_task_rows": expected_rows,
            "score": None,
            "eligible_for_internal_confirmation": False,
        },
        "failure": {"stage": "full_search_rescore",
                    "error_type": type(error).__name__,
                    "message": str(error)[:512]},
        "candidate_status": "internal_pending_search_failure",
        "eligible_for_official_selection_gate": False,
        "confirmation_eligible": False,
        "selection_gate_read": False,
        "official_validation_read": False,
        "official_test_read": False,
        "retrieval_used": False,
        "request_accounting": summarize_ledger(ledger_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--observed-incumbent-correct", type=int, required=True,
                        help="full incumbent correct count observed in the optimizer output")
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--backoff-base-seconds", type=int, default=60)
    args = parser.parse_args(argv)

    if args.observed_incumbent_correct < 0 or args.backoff_base_seconds < 0:
        raise ValueError("incumbent score and backoff must be non-negative")
    sandbox = os.environ.get("RSI_HOME", "").strip()
    if not sandbox:
        raise RuntimeError("candidate rescore requires the original isolated RSI_HOME")
    study_dir = Path(sandbox) / "probe" / "evidence_inference_mipro" / args.study_id
    failed_path = study_dir / "report.json"
    report = read_json(failed_path)
    if report.get("status") != "candidate_search_failed":
        raise ValueError("rescore requires the preserved candidate_search_failed report")
    plan = report.get("plan") or {}
    if plan.get("plan_sha256", "")[:16] != args.study_id:
        raise ValueError("failed report does not match the requested frozen plan")
    require_frozen_plan_challenge(plan, args.expected_plan_sha256)
    if report.get("selection_gate_read") is not False or report.get("official_test_read") is not False:
        raise ValueError("refusing a source report that accessed official evaluation data")
    if report.get("retrieval_used") is not False:
        raise ValueError("refusing a search that used retrieval as optimizer input")

    cfg = prepare_config(
        args.config, proposer_calls=int(plan["proposer_call_budget"]),
        routing_mode=str(plan["routing_mode"]), task_model_id=str(plan["task_model"]))
    from src.config import config_fingerprint
    if config_fingerprint(cfg) != plan.get("config_sha256"):
        raise ValueError("current task protocol differs from the frozen failed plan")
    state = load_state(str(plan["release_id"]))
    if state.release_id != RELEASE_ID:
        raise ValueError("frozen MIPRO parent release is not the expected isolated release")

    train_rows = load_tasks(splits=["dev_optimize"])
    cohort, scopes, task_by_id = load_frozen_cohort(args.cohort.resolve(), train_rows)
    if cohort.get("cohort_sha256") != plan.get("cohort_sha256"):
        raise ValueError("cohort does not match the frozen plan")
    candidate_artifact, content, saved_gate = reconstruct_trial_candidate(
        report, args.candidate_id, task_by_id, state)
    if saved_gate.get("passed") is not True:
        raise ValueError("candidate has no saved passing program gate")

    candidate = Candidate(
        candidate_id=args.candidate_id,
        parent_release_id=state.release_id,
        artifact_type="task_program", artifact_id="task_program",
        diff="Re-score the exact gated MIPRO trial after a later trial evaluation failed",
        rationale="The original trial score was partial and discarded; this report contains a fresh full-scope score.",
        new_content=content, content_sha256=content_sha256(content),
    )
    program_cfg = dict(cfg)
    program_cfg["evolution"] = dict(program_cfg.get("evolution") or {})
    program_cfg["evolution"]["allowed_artifacts"] = ["task_program"]
    static = static_gate(candidate, program_cfg, state)
    rendering = validate_task_demo_rendering(candidate_artifact["task_demos"], cfg)
    if not static.get("passed") or not rendering.get("passed"):
        raise ValueError("candidate failed the current static or demo-rendering gate")

    trial_path = study_dir / f"recovered-search-{args.candidate_id}.jsonl"
    search_seed = int(plan["mipro_execution_seed"])
    candidate_state = ArtifactState(
        release_id=state.release_id,
        system_prompt=candidate_artifact["system_prompt"],
        registry=state.registry,
        task_demos=candidate_artifact["task_demos"],
    )
    os.environ["RSI_HTTP_NO_CACHE"] = "1"
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(plan["plan_sha256"])
    os.environ["RSI_CALL_LEDGER_PATH"] = str((study_dir / "request-ledger.jsonl").resolve())
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(plan["maximum_configured_model_calls"])
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(plan["maximum_physical_http_requests"])
    os.environ["RSI_PROVIDER_429_BACKOFF_BASE_SECONDS"] = str(args.backoff_base_seconds)

    try:
        search_rows = _score_repeated_scope(
            cfg, candidate_state, scopes["search_validation"], [search_seed],
            trial_path, candidate_id=args.candidate_id)[search_seed]
    except Exception as exc:
        recovery = failure_report(
            plan=plan, study_id=args.study_id, candidate_id=args.candidate_id,
            source_path=failed_path, candidate=candidate_artifact,
            candidate_sha256=candidate.content_sha256, saved_gate=saved_gate,
            static=static, rendering=rendering, checkpoint_path=trial_path,
            expected_rows=len(scopes["search_validation"]), error=exc,
            ledger_path=study_dir / "request-ledger.jsonl")
        out = study_dir / f"recovered-trial-{args.candidate_id}.json"
        write_json(out, recovery)
        print(json.dumps({"status": recovery["status"], "report": str(out),
                          "completed_task_rows": recovery["search_validation"][
                              "completed_task_rows"],
                          "expected_task_rows": recovery["search_validation"][
                              "expected_task_rows"],
                          "score": None,
                          "request_accounting": recovery["request_accounting"]},
                         ensure_ascii=False))
        return 2
    search_metrics = metrics(search_rows)
    search_correct = int(round(search_metrics["accuracy"] * search_metrics["n"]))
    recovery = {
        "protocol": plan["protocol"],
        "status": "recovered_trial_search_scored",
        "study_id": args.study_id,
        "plan": plan,
        "plan_sha256": plan["plan_sha256"],
        "source_failed_report": str(failed_path),
        "source_failed_report_sha256": sha256_file(failed_path),
        "candidate_id": args.candidate_id,
        "candidate": candidate_artifact,
        "candidate_content_sha256": candidate.content_sha256,
        "saved_program_gate": saved_gate,
        "rechecked_static_gate": static,
        "rechecked_demo_rendering_gate": rendering,
        "search_validation": {
            "scope": "full frozen search_validation scope; partial optimizer score discarded",
            "search_seed_label": search_seed,
            "correct": search_correct,
            "n": int(search_metrics["n"]),
            "accuracy": search_metrics["accuracy"],
            "observed_incumbent_correct": args.observed_incumbent_correct,
            "eligible_for_internal_confirmation":
                search_correct >= args.observed_incumbent_correct,
            "checkpoint_path": str(trial_path),
        },
        "candidate_status": "search_scored_internal_pending",
        "eligible_for_official_selection_gate": False,
        "confirmation_eligible": False,
        "selection_gate_read": False,
        "official_validation_read": False,
        "official_test_read": False,
        "retrieval_used": False,
        "request_accounting": summarize_ledger(study_dir / "request-ledger.jsonl"),
    }
    if search_correct < args.observed_incumbent_correct:
        out = study_dir / f"recovered-trial-{args.candidate_id}.json"
        write_json(out, recovery)
        print(json.dumps({"status": recovery["status"], "report": str(out),
                          "search_correct": search_correct,
                          "observed_incumbent_correct": args.observed_incumbent_correct,
                          "internal_confirmation_run": False}, ensure_ascii=False))
        return 0

    baseline_path = Path(report["internal_baseline_checkpoint"])
    baseline_rows = read_jsonl(baseline_path)
    labels = list(map(int, plan["execution_labels"]))
    internal_scope = scopes["internal_confirmation"]
    expected_rows = len(internal_scope) * len(labels)
    if len(baseline_rows) != expected_rows:
        raise ValueError("frozen internal baseline checkpoint is incomplete")
    baseline_by_label = {
        seed: [row for row in baseline_rows if int(row.get("seed", -1)) == seed]
        for seed in labels
    }
    if any(len(rows) != len(internal_scope) for rows in baseline_by_label.values()):
        raise ValueError("frozen baseline rows do not match every execution label")
    candidate_path = study_dir / f"recovered-internal-{args.candidate_id}.jsonl"
    candidate_by_label = _score_repeated_scope(
        cfg, candidate_state, internal_scope, labels, candidate_path,
        candidate_id=args.candidate_id)
    baseline_metrics = {str(seed): metrics(rows)
                        for seed, rows in baseline_by_label.items()}
    candidate_metrics = {str(seed): metrics(rows)
                         for seed, rows in candidate_by_label.items()}
    accepted = all(class_aware_gain(candidate_metrics[str(seed)], baseline_metrics[str(seed)])
                   for seed in labels)
    recovery.update({
        "status": "recovered_trial_internal_gate_completed",
        "baseline_internal_metrics_by_execution_label": baseline_metrics,
        "candidate_internal_metrics_by_execution_label": candidate_metrics,
        "internal_paired_analysis": paired_internal_analysis(
            baseline_by_label, candidate_by_label, labels),
        "candidate_status": "internal_gate_passed" if accepted else "internal_gate_rejected",
        "eligible_for_official_selection_gate": bool(accepted),
        "internal_baseline_checkpoint": str(baseline_path),
        "internal_candidate_checkpoint": str(candidate_path),
        "request_accounting": summarize_ledger(study_dir / "request-ledger.jsonl"),
    })
    out = study_dir / f"recovered-trial-{args.candidate_id}.json"
    write_json(out, recovery)
    print(json.dumps({"status": recovery["status"], "report": str(out),
                      "candidate_status": recovery["candidate_status"],
                      "candidate_id": args.candidate_id,
                      "eligible_for_official_selection_gate": accepted,
                      "request_accounting": recovery["request_accounting"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
