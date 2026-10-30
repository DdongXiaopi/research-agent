"""Aggregate-only exploratory transfer gate for a fixed Prompt hypothesis."""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_fingerprint, load_config  # noqa: E402
from src.contracts.types import Candidate, content_id  # noqa: E402
from src.eval.harness import evaluate, gate_summary  # noqa: E402
from src.eval.leakage import check_pi_gate_sessions  # noqa: E402
from src.evolution.round import gate_delta_threshold, task_input_secondary_gate  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.runtime.runner import ArtifactState  # noqa: E402
from src.util import (  # noqa: E402
    artifacts_path, artifacts_root, ensure_dir, read_json, sha256_file,
    sha256_obj, sha256_text, write_json,
)


def build_execution_plan(*, report_path: Path, optimize: dict, state,
                         cfg: dict, candidate: Candidate,
                         static_gate_result: dict, limit: int = 30) -> dict:
    """Freeze one aggregate-only fixed-candidate gate run without reading gate rows."""
    runtime = cfg.get("runtime") or {}
    task_model = (cfg.get("models") or {}).get("task") or {}
    max_model_calls = int(runtime.get("max_model_calls", 12))
    if limit <= 0 or max_model_calls <= 0:
        raise ValueError("gate plan limits must be positive")
    pi_retries = max(0, min(4, int(runtime.get("pi_provider_max_retries", 4))))
    bridge_attempts = max(1, min(3, int(runtime.get("pi_timeout_max_attempts", 2))))
    pi_http_attempts = (pi_retries + 1) * bridge_attempts
    http_attempts = max(1, int(task_model.get("max_retries", 6)))
    per_operation_http = max(pi_http_attempts, http_attempts)
    logical_cap = 2 * limit * max_model_calls
    unsigned = {
        "schema": "pi-rule-fixed-candidate-selection-gate-plan-v1",
        "role": "one_shot_aggregate_only_candidate_gate",
        "source_optimize_report_sha256": sha256_file(report_path),
        "source_optimize_protocol": optimize.get("protocol"),
        "source_candidate_id": optimize.get("candidate_id"),
        "source_candidate_prompt_sha256": optimize.get("candidate_prompt_sha256"),
        "candidate_id": candidate.candidate_id,
        "candidate_content_sha256": candidate.content_sha256,
        "parent_release_id": state.release_id,
        "registry_sha256": state.registry.fingerprint(),
        "config_sha256": config_fingerprint(cfg),
        "dataset": "pubmedqa",
        "split": "selection_gate",
        "task_limit_per_arm": limit,
        "evaluation_arms": ["baseline", "fixed_candidate"],
        "result_policy": "aggregate_only_no_per_task_persistence",
        "task_provider": str(runtime.get("provider") or ""),
        "task_model_id": str(task_model.get("model_id") or ""),
        "taskrunner_model_operations_per_task_upper_bound": max_model_calls,
        "pi_provider_retries_after_first": pi_retries,
        "pi_timeout_subprocess_attempts": bridge_attempts,
        "pi_http_attempts_per_operation_upper_bound": pi_http_attempts,
        "http_attempts_per_operation_upper_bound": http_attempts,
        "logical_request_cap": logical_cap,
        "physical_request_cap": logical_cap * per_operation_http,
        "request_ledger_required": True,
        "pi_physical_reservations_required": True,
        "model_calls_during_preflight": 0,
        "static_gate_passed_before_plan": static_gate_result.get("passed") is True,
        "selection_gate_read_during_preflight": False,
        "selection_gate_read_if_executed": True,
        "final_holdout_read": False,
        "candidate_search_or_retry_after_gate": False,
        "live_authorized": False,
        "explicit_user_approval_required_before_execution": True,
    }
    return {**unsigned, "plan_sha256": sha256_obj(unsigned)}


def verify_execution_plan(plan: dict, expected_sha256: str) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    actual = sha256_obj(unsigned)
    if plan.get("plan_sha256") != actual:
        raise ValueError("frozen gate plan hash is internally inconsistent")
    if expected_sha256 != actual:
        raise ValueError("expected plan hash does not match current gate plan")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimize-report", default=os.environ.get(
        "RSI_PI_RULE_OPTIMIZE_REPORT", ""),
        help="frozen optimize-only report that selected this fixed candidate")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true",
                      help="write a zero-call plan without opening selection_gate")
    mode.add_argument("--execute", action="store_true",
                      help="run exactly one gate pass for the fixed candidate")
    parser.add_argument("--expected-plan-sha256", default="",
                        help="required with --execute; must match the frozen plan")
    args = parser.parse_args(argv)
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("Pi gate probe requires an explicit sandbox RSI_HOME")
    # A specific optimize report is selected before any gate observation; do
    # not choose among candidate reports after seeing a gate result.
    selected = args.optimize_report
    if not selected:
        raise RuntimeError("RSI_PI_RULE_OPTIMIZE_REPORT must name the selected optimize report")
    report_path = Path(selected).resolve()
    if not report_path.is_relative_to(artifacts_root().resolve()):
        raise RuntimeError("optimize report must be inside the active sandbox")
    optimize = read_json(report_path)
    if optimize.get("protocol") != "pi-target-outcome-rule-visible-optimize-v1" \
            or optimize.get("selection_gate_read") is not False:
        raise RuntimeError("not a clean optimize-only rule report")
    cfg = copy.deepcopy(load_config("configs/experiment.http.pubmedqa-pilot.yaml"))
    cfg["runtime"].update({"provider": "pi", "input_mode": "task_input",
                           "task_input_max_chars": 30000})
    cfg.setdefault("evaluation", {}).update({"main_metric": "accuracy",
                                               "primary_axis": "accuracy",
                                               "sealed_pairing": False})
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg["tools"].get("active", []) if name != "academic.search"]
    if optimize.get("config_fingerprint") != config_fingerprint(cfg):
        raise RuntimeError("Pi protocol changed since optimize selection")
    state = load_state()
    if optimize.get("release_id") != state.release_id:
        raise RuntimeError("release changed since optimize selection")
    rule = (ROOT / "configs" / "task_input" /
            "strategy_target_outcome_v1.txt").read_text(encoding="utf-8").strip()
    new_prompt = state.system_prompt.rstrip() + "\n\n" + rule
    if optimize.get("candidate_prompt_sha256") != sha256_text(new_prompt):
        raise RuntimeError("candidate Prompt changed since optimize selection")
    candidate = Candidate(
        candidate_id=content_id("cand", state.release_id, "system_prompt", new_prompt),
        parent_release_id=state.release_id, artifact_type="system_prompt",
        artifact_id="system_prompt", diff="Append target-outcome interpretation rule",
        new_content=new_prompt, content_sha256=sha256_text(new_prompt))
    if optimize.get("candidate_id") != candidate.candidate_id:
        raise RuntimeError("candidate identity mismatch")
    gate_result = static_gate(candidate, cfg, state)
    if not gate_result.get("passed"):
        raise RuntimeError("candidate static gate failed")
    plan = build_execution_plan(
        report_path=report_path, optimize=optimize, state=state, cfg=cfg,
        candidate=candidate, static_gate_result=gate_result)
    plan_path = artifacts_path("probe", "pi_rule_gate",
                               "plan-" + plan["plan_sha256"] + ".json")
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise RuntimeError("saved PI rule gate plan differs from recomputed plan")
    else:
        write_json(plan_path, plan)
    if args.validate_only:
        print(json.dumps({
            "status": "validated_not_run", "plan": str(plan_path),
            "plan_sha256": plan["plan_sha256"],
            "dataset": plan["dataset"], "split": plan["split"],
            "task_limit_per_arm": plan["task_limit_per_arm"],
            "logical_request_cap": plan["logical_request_cap"],
            "physical_request_cap": plan["physical_request_cap"],
            "model_calls": 0, "selection_gate_read": False,
            "final_holdout_read": False,
        }, ensure_ascii=False))
        return 0

    verify_execution_plan(plan, args.expected_plan_sha256)
    task_model = (cfg.get("models") or {}).get("task") or {}
    key_env = str(task_model.get("api_key_env") or "").strip()
    if not key_env or not os.environ.get(key_env):
        raise RuntimeError("gate execution is missing the configured task-model credential")
    run_id = uuid.uuid4().hex[:12]
    run_dir = artifacts_path("probe", "pi_rule_gate", "gate-" + run_id)
    ensure_dir(run_dir)
    ledger_path = run_dir / "call-ledger.jsonl"
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(plan["plan_sha256"])
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path)
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(plan["logical_request_cap"])
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(plan["physical_request_cap"])
    os.environ["RSI_CALL_LEDGER_RESERVE_PI_PHYSICAL"] = "1"

    # These session checks scan historical durable Pi sessions, so they are
    # deliberately after exact-plan verification and never run in preflight.
    before = check_pi_gate_sessions(artifacts_path("pi_sessions"))
    if not before["ok"]:
        raise RuntimeError("Pi sandbox already has durable gate/unknown sessions")
    trial = ArtifactState(state.release_id, new_prompt, state.registry)
    seed = int(cfg.get("evaluation", {}).get("seed", 20260924))
    baseline = evaluate(cfg, state, "pubmedqa", "selection_gate",
                        seed=seed, limit=30, run_tag="baseline-rule-gate",
                        write_per_task=False, cfg_fp=config_fingerprint(cfg))
    changed = evaluate(cfg, trial, "pubmedqa", "selection_gate",
                       seed=seed, limit=30, run_tag=candidate.candidate_id,
                       write_per_task=False, cfg_fp=config_fingerprint(cfg))
    if baseline.per_task or changed.per_task:
        raise RuntimeError("selection gate returned per-task answers")
    after = check_pi_gate_sessions(artifacts_path("pi_sessions"))
    if not after["ok"] or after["files_scanned"] != before["files_scanned"]:
        raise RuntimeError("Pi selection gate persisted per-task sessions")
    base_metrics, changed_metrics = baseline.metric_values, changed.metric_values
    secondary = task_input_secondary_gate(
        {"pubmedqa": {"optimize": optimize["baseline_metrics"],
                       "gate": base_metrics}},
        {"pubmedqa": {"optimize": optimize["candidate_metrics"],
                       "gate": {"metrics": changed_metrics}}}, ["pubmedqa"])
    tolerance = float(cfg["evaluation"].get("primary_axis_tolerance_pp", 0.0))
    primary_no_regress = all(
        (optimize["candidate_metrics"]["accuracy"] >=
         optimize["baseline_metrics"]["accuracy"] - tolerance / 100.0,
         changed_metrics["accuracy"] >= base_metrics["accuracy"] - tolerance / 100.0))
    threshold = gate_delta_threshold(
        100.0 * (changed_metrics["accuracy"] - base_metrics["accuracy"]),
        float(cfg["gates"]["accept"]["gate_min_delta_pp"]),
        float(cfg["gates"]["accept"]["gate_min_prob_delta_gt_0"]))
    result = {
        "protocol": "pi-target-outcome-rule-aggregate-gate-v1-exploratory",
        "source_optimize_report": str(report_path),
        "candidate_id": candidate.candidate_id,
        "config_fingerprint": config_fingerprint(cfg),
        "execution_plan_sha256": plan["plan_sha256"],
        "execution_plan_path": str(plan_path),
        "request_ledger": str(ledger_path),
        "request_ledger_summary": summarize_ledger(ledger_path),
        "release_id": state.release_id,
        "baseline_gate": gate_summary(baseline),
        "candidate_gate": gate_summary(changed),
        "secondary_gate": secondary,
        "primary_no_regress": primary_no_regress,
        "threshold": threshold,
        "engineering_pass": bool(primary_no_regress and secondary["passed"]
                                 and threshold["passed"]),
        "pi_session_audit": after,
        "selection_gate_per_task_exposed": False,
        "final_holdout_read": False,
    }
    path = run_dir / "report.json"
    write_json(path, result)
    print(json.dumps({"path": str(path),
                      "baseline_accuracy": base_metrics["accuracy"],
                      "candidate_accuracy": changed_metrics["accuracy"],
                      "delta_pp": threshold["delta_pp"],
                      "secondary_passed": secondary["passed"],
                      "engineering_pass": result["engineering_pass"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
