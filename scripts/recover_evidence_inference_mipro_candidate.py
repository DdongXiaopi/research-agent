#!/usr/bin/env python3
"""Score a fully evaluated MIPRO trial after a later search trial failed.

This recovery path never turns a partial optimizer search into a completed
MIPRO report. It reconstructs only a candidate whose exact program and gates
were saved before the infrastructure failure, then runs the frozen internal
gate against the matching baseline checkpoint.
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

from scripts.run_evidence_inference_mipro_search import (  # noqa: E402
    LABELS, RELEASE_ID, _score_repeated_scope, class_aware_gain,
    compute_plan_sha256, load_frozen_cohort,
    metrics, paired_internal_analysis, prepare_config,
)
from src.config import config_fingerprint  # noqa: E402
from src.contracts.types import Candidate, content_id, content_sha256  # noqa: E402
from src.data.evidence_inference_v2 import load_tasks  # noqa: E402
from src.evolution.task_input_gates import validate_task_demo_rendering  # noqa: E402
from src.evolution.task_program import serialize_task_program  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.runtime.runner import ArtifactState  # noqa: E402
from src.util import read_json, read_jsonl, sha256_file, write_json  # noqa: E402


def require_frozen_plan_challenge(plan: dict, expected_sha256: str) -> None:
    saved_sha256 = str(plan.get("plan_sha256") or "")
    if not expected_sha256 or expected_sha256 != saved_sha256:
        raise ValueError("recovery requires the exact --expected-plan-sha256")
    if compute_plan_sha256(plan) != saved_sha256:
        raise ValueError("recovery plan self-hash does not match its frozen contents")


def reconstruct_trial_candidate(report: dict, candidate_id: str, train_by_id: dict,
                                state) -> tuple[dict, str, dict]:
    """Rebuild an exact previously gated task program from its saved trial rows."""
    if report.get("status") != "candidate_search_failed":
        raise ValueError("recovery requires a candidate_search_failed MIPRO report")
    instruction_by_hash = {
        row.get("content_sha256"): row for row in report.get("candidate_instruction_gates", [])
        if row.get("content_sha256") and row.get("passed") is True
    }
    for gate in report.get("candidate_program_gates", []):
        if gate.get("candidate_id") != candidate_id or gate.get("passed") is not True:
            continue
        render = gate.get("task_demo_rendering_gate") or {}
        demo_ids = [str(value) for value in render.get("candidate_demo_ids") or []]
        if (not demo_ids or render.get("rendered_demo_ids") != demo_ids
                or render.get("passed") is not True):
            raise ValueError("saved trial did not pass exact demo rendering")
        demos = []
        for task_id in demo_ids:
            task = train_by_id.get(task_id)
            if task is None or task.get("split") != "dev_optimize":
                raise ValueError("saved MIPRO demo is outside frozen train IDs")
            demos.append({"task_id": task_id,
                          "correct_index": int(task["correct_index"]),
                          "task": task})
        for prompt_hash, instruction in instruction_by_hash.items():
            prompt = str(instruction.get("candidate_text") or "")
            if (not prompt or content_sha256(prompt) != prompt_hash
                    or not (instruction.get("grounding_review") or {}).get("passed")):
                continue
            content = serialize_task_program(prompt, demos)
            if content_id("cand", state.release_id, "task_program", content) != candidate_id:
                continue
            if content_sha256(content) != gate.get("content_sha256"):
                raise ValueError("reconstructed candidate differs from the saved trial gate")
            return {"system_prompt": prompt, "task_demos": demos}, content, gate
    raise ValueError("candidate ID has no exact passing prompt × demo gate record")


def select_completed_trial_checkpoint(rows: list[dict], *, plan_sha256: str,
                                      expected_evaluation_count: int,
                                      candidate_id: str | None = None) -> dict:
    """Choose only a fully scored, plan-matched trial with a gated artifact."""
    eligible = []
    for row in rows:
        if (row.get("plan_sha256") != plan_sha256
                or row.get("evaluation_type") != "full_search_validation"
                or int(row.get("evaluation_count", -1)) != expected_evaluation_count
                or not isinstance(row.get("candidate"), dict)
                or not row.get("candidate_id")):
            continue
        if candidate_id is not None and row.get("candidate_id") != candidate_id:
            continue
        eligible.append(row)
    if not eligible:
        raise ValueError("no fully evaluated matching candidate in trial checkpoint")
    eligible.sort(key=lambda row: (
        -float(row.get("score_percent", float("-inf"))),
        -len(row.get("demo_label_counts") or {}),
        int((row.get("trial") or {}).get("console_trial_number", 0)),
        str(row.get("candidate_id")),
    ))
    return eligible[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--candidate-id")
    parser.add_argument("--observed-trial-number", type=int)
    parser.add_argument("--observed-search-score", type=float,
                        help="Score shown by the completed DSPy trial before a later trial failed")
    parser.add_argument("--observed-parameters",
                        help="Literal completed-trial parameter summary shown by DSPy")
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--cohort", type=Path, required=True)
    args = parser.parse_args(argv)

    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("recovery requires the original isolated RSI_HOME")
    study_dir = Path(os.environ["RSI_HOME"]) / "probe" / "evidence_inference_mipro" / args.study_id
    report_path = study_dir / "report.json"
    report = read_json(report_path)
    if report.get("study_id") != args.study_id:
        raise ValueError("study ID differs from the failed report")
    plan = report.get("plan") or {}
    if plan.get("plan_sha256", "")[:16] != args.study_id:
        raise ValueError("failed report is not bound to the requested study ID")
    require_frozen_plan_challenge(plan, args.expected_plan_sha256)

    cfg = prepare_config(
        args.config, proposer_calls=int(plan["proposer_call_budget"]),
        routing_mode=str(plan["routing_mode"]), task_model_id=str(plan["task_model"]))
    if config_fingerprint(cfg) != plan.get("config_sha256"):
        raise ValueError("current task protocol differs from the frozen recovery plan")
    state = load_state(RELEASE_ID)
    train_rows = load_tasks(splits=["dev_optimize"])
    cohort, scopes, task_by_id = load_frozen_cohort(args.cohort.resolve(), train_rows)
    if cohort.get("cohort_sha256") != plan.get("cohort_sha256"):
        raise ValueError("recovery cohort differs from the failed MIPRO plan")
    train_ids = {str(row["task_id"]) for row in scopes["mipro_train"]}
    search_ids = {str(row["task_id"]) for row in scopes["search_validation"]}
    internal = scopes["internal_confirmation"]
    expected_ids = [str(row["task_id"]) for row in train_rows]
    if len(task_by_id) != len(set(expected_ids)):
        raise ValueError("frozen train task mapping contains duplicate IDs")
    trial_checkpoint_path = study_dir / "completed-mipro-trials.jsonl"
    observation_source = "DSPy console output captured during the original run"
    observed_trial_number = args.observed_trial_number
    observed_search_score = args.observed_search_score
    observed_parameters = args.observed_parameters
    candidate_id = args.candidate_id
    checkpoint = None
    if trial_checkpoint_path.exists():
        checkpoint = select_completed_trial_checkpoint(
            read_jsonl(trial_checkpoint_path),
            plan_sha256=str(plan["plan_sha256"]),
            expected_evaluation_count=len(scopes["search_validation"]),
            candidate_id=candidate_id,
        )
        checkpoint_trial = checkpoint.get("trial") or {}
        checkpoint_number = int(checkpoint_trial.get("console_trial_number", 0))
        checkpoint_score = float(checkpoint["score_percent"])
        checkpoint_parameters = str(checkpoint_trial.get("chosen_parameters") or [])
        if (observed_trial_number is not None
                and observed_trial_number != checkpoint_number):
            raise ValueError("observed trial number differs from durable checkpoint")
        if (observed_search_score is not None
                and abs(observed_search_score - checkpoint_score) > 1e-9):
            raise ValueError("observed score differs from durable checkpoint")
        if (observed_parameters is not None
                and observed_parameters != checkpoint_parameters):
            raise ValueError("observed parameters differ from durable checkpoint")
        candidate_id = str(checkpoint["candidate_id"])
        observed_trial_number = checkpoint_number
        observed_search_score = checkpoint_score
        observed_parameters = checkpoint_parameters
        observation_source = "durable full-trial checkpoint written after task evaluation"
    if (not candidate_id or observed_trial_number is None
            or observed_search_score is None or observed_parameters is None):
        raise ValueError(
            "recovery needs a durable full-trial checkpoint or explicit completed-trial observations")
    candidate_artifact, content, saved_gate = reconstruct_trial_candidate(
        report, candidate_id, task_by_id, state)
    if checkpoint is not None:
        checkpoint_content = serialize_task_program(
            checkpoint["candidate"]["system_prompt"], checkpoint["candidate"]["task_demos"])
        if (content_sha256(checkpoint_content) != content_sha256(content)
                or checkpoint.get("candidate_content_sha256") != content_sha256(content)):
            raise ValueError("durable checkpoint program differs from saved passing gates")
    demo_ids = {str(item["task_id"]) for item in candidate_artifact["task_demos"]}
    if not demo_ids <= train_ids or demo_ids & search_ids:
        raise ValueError("recovered demos escaped train or overlap search validation")

    program_cfg = cfg.copy()
    program_cfg["evolution"] = dict(program_cfg.get("evolution") or {})
    program_cfg["evolution"]["allowed_artifacts"] = ["task_program"]
    candidate = Candidate(
        candidate_id=candidate_id,
        parent_release_id=state.release_id,
        artifact_type="task_program",
        artifact_id="task_program",
        diff="Recover one fully evaluated MIPRO trial after a later trial transport failure",
        new_content=content,
        content_sha256=content_sha256(content),
    )
    static = static_gate(candidate, program_cfg, state)
    rendering = validate_task_demo_rendering(candidate_artifact["task_demos"], cfg)
    if not static.get("passed") or not rendering.get("passed"):
        raise ValueError("reconstructed candidate failed a current static/render gate")

    baseline_path = Path(report["internal_baseline_checkpoint"])
    baseline_rows = read_jsonl(baseline_path)
    expected_baseline_rows = len(internal) * len(plan["execution_labels"])
    if len(baseline_rows) != expected_baseline_rows:
        raise ValueError("frozen baseline checkpoint is incomplete")
    baseline_by_seed = {
        int(seed): [row for row in baseline_rows if int(row.get("seed", -1)) == int(seed)]
        for seed in plan["execution_labels"]
    }
    if any(len(rows) != len(internal) for rows in baseline_by_seed.values()):
        raise ValueError("frozen baseline rows do not match the internal task scope")

    ledger_path = study_dir / "request-ledger.jsonl"
    os.environ["RSI_HTTP_NO_CACHE"] = "1"
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(plan["plan_sha256"])
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path.resolve())
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(
        (report.get("request_budget") or {}).get("logical_ceiling", 468))
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(
        (report.get("request_budget") or {}).get("physical_ceiling", 1404))
    os.environ["RSI_PROVIDER_429_BACKOFF_BASE_SECONDS"] = "15"

    recovered_state = ArtifactState(
        release_id=state.release_id,
        system_prompt=candidate_artifact["system_prompt"],
        registry=state.registry,
        task_demos=candidate_artifact["task_demos"],
    )
    candidate_path = study_dir / "internal-candidate-recovered.jsonl"
    candidate_by_seed = _score_repeated_scope(
        cfg, recovered_state, internal, list(map(int, plan["execution_labels"])),
        candidate_path, candidate_id=candidate_id)
    candidate_metrics = {str(seed): metrics(rows)
                         for seed, rows in candidate_by_seed.items()}
    baseline_metrics = {str(seed): metrics(rows)
                        for seed, rows in baseline_by_seed.items()}
    paired = paired_internal_analysis(baseline_by_seed, candidate_by_seed,
                                      list(map(int, plan["execution_labels"])))
    accepted = all(class_aware_gain(candidate_metrics[str(seed)], baseline_metrics[str(seed)])
                   for seed in plan["execution_labels"])

    recovery = {
        "protocol": plan["protocol"],
        "status": "recovered_trial_internal_gate_completed",
        "source_failed_report": str(report_path),
        "source_failed_report_sha256": sha256_file(report_path),
        "study_id": args.study_id,
        "plan": plan,
        "plan_sha256": plan["plan_sha256"],
        "candidate_id": candidate_id,
        "recovery_method": "exact reconstruction from a saved passing program gate; internal evaluation only",
        "completed_trial_observation": {
            "trial_number": observed_trial_number,
            "search_validation_score_percent": observed_search_score,
            "parameters": observed_parameters,
            "source": observation_source,
        },
        "candidate": candidate_artifact,
        "candidate_content_sha256": candidate.content_sha256,
        "saved_program_gate": saved_gate,
        "rechecked_static_gate": static,
        "rechecked_demo_rendering_gate": rendering,
        "baseline_internal_metrics_by_execution_label": baseline_metrics,
        "candidate_internal_metrics_by_execution_label": candidate_metrics,
        "internal_paired_analysis": paired,
        "candidate_status": "internal_gate_passed" if accepted else "internal_gate_rejected",
        "eligible_for_official_selection_gate": bool(accepted),
        "confirmation_eligible": False,
        "internal_baseline_checkpoint": str(baseline_path),
        "internal_candidate_checkpoint": str(candidate_path),
        "selection_gate_read": False,
        "official_validation_read": False,
        "official_test_read": False,
        "retrieval_used": False,
        "request_accounting": summarize_ledger(ledger_path),
    }
    recovery_path = study_dir / "recovered-trial-report.json"
    write_json(recovery_path, recovery)
    print(json.dumps({"report": str(recovery_path),
                      "candidate_status": recovery["candidate_status"],
                      "candidate_id": candidate_id,
                      "eligible_for_official_selection_gate": recovery[
                          "eligible_for_official_selection_gate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
