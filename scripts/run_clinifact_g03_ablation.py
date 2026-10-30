#!/usr/bin/env python3
"""Execute the hash-bound G03 A0/A1/A2 multi-seed development study.

The plan builder is intentionally zero-call.  This runner is the only live
entry point: it validates every bound input before dispatch, uses the exact
same frozen task rows for all arms and seeds, and writes one ledger-backed
report.  It never reads selection-gate, validation, or official-test data.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_evidence_inference_mipro_search as mipro  # noqa: E402
from scripts.run_clinifact_candidate_bundle_confirmation import _candidate_states  # noqa: E402
from src.config import load_config  # noqa: E402
from src.contracts.types import Candidate, content_sha256  # noqa: E402
from src.data.clinifact_task_input import load_tasks  # noqa: E402
from src.eval.g03_ablation_plan import ARMS, validate_g03_plan  # noqa: E402
from src.eval.stats import mcnemar_exact, paired_bootstrap_binary  # noqa: E402
from src.eval.task_rows import validate_task_answer_row  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.runtime.runner import ArtifactState, TaskRunner  # noqa: E402
from src.runtime.skills import with_description  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.util import read_json, sha256_file, sha256_obj, write_json  # noqa: E402


def _task_map(cohort: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tasks = load_tasks(splits=["dev_optimize"])
    by_id = {str(task["task_id"]): task for task in tasks}
    rows = cohort.get("task_rows") or []
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        task = by_id.get(task_id)
        if task is None:
            raise ValueError(f"G03 cohort task is absent from the audited task source: {task_id}")
        if sha256_obj(task) != row["material_sha256"]:
            raise ValueError(f"G03 task material hash mismatch: {task_id}")
        selected[task_id] = task
    if len(selected) != len(rows):
        raise ValueError("G03 cohort contains duplicate task IDs")
    return selected


def _states(bundle: dict[str, Any], prompt_report: dict[str, Any],
            skill_result: dict[str, Any], cfg: dict[str, Any], parent: ArtifactState) -> dict[str, ArtifactState]:
    combined, _ = _candidate_states(
        bundle, prompt_report, Path("prompt-report.json"), skill_result,
        Path("skill-result.json"), cfg, parent)
    prompt = combined.system_prompt
    description = str((skill_result.get("candidate") or {}).get("new_content") or "").strip()
    skill_name = str(bundle["skill_name"])
    a0 = ArtifactState(parent.release_id, parent.system_prompt, parent.registry,
                       [dict(x) for x in parent.task_demos])
    a1 = ArtifactState(parent.release_id, prompt, parent.registry,
                       [dict(x) for x in combined.task_demos])
    a2 = ArtifactState(parent.release_id, parent.system_prompt,
                       with_description(parent.registry, skill_name, description),
                       [dict(x) for x in parent.task_demos])
    return {"A0": a0, "A1": a1, "A2": a2}


def _metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(int(bool(row.get("correct"))) for row in rows)
    return {"n": total, "correct": correct, "accuracy": (correct / total if total else 0.0)}


def _paired(base: list[int], after: list[int], *, seed: int, B: int) -> dict[str, Any]:
    if not base or not after or len(base) != len(after):
        raise ValueError("paired G03 inputs must be non-empty and have equal length")
    return {"mcnemar_exact": mcnemar_exact(base, after),
            "paired_bootstrap": paired_bootstrap_binary(base, after, B=B, rng_seed=seed),
            "baseline_accuracy": sum(base) / len(base) if base else 0.0,
            "candidate_accuracy": sum(after) / len(after) if after else 0.0}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("G03 execution requires an isolated RSI_HOME")
    plan_path = args.plan.resolve()
    plan = read_json(plan_path)
    validate_g03_plan(plan)
    if sha256_file(plan_path) != args.plan_file_sha256:
        raise ValueError("G03 plan file SHA does not match the approved execution input")
    bundle = read_json(args.bundle.resolve())
    confirmation = read_json(args.confirmation.resolve())
    cohort = read_json(args.cohort.resolve())
    if plan.get("candidate_bundle_id") != bundle.get("candidate_bundle_id"):
        raise ValueError("G03 plan and bundle identity differ")
    if plan.get("confirmation_report_sha256") != sha256_file(args.confirmation.resolve()):
        raise ValueError("G03 confirmation report file hash differs from the plan")
    if plan.get("cohort_file_sha256") != sha256_file(args.cohort.resolve()):
        raise ValueError("G03 cohort file hash differs from the plan")
    if confirmation.get("target_model_id") != plan.get("target_model_id"):
        raise ValueError("G03 target model differs from confirmation")
    task_map = _task_map(cohort)
    ordered_ids = [str(row["task_id"]) for row in cohort["task_rows"]]
    prompt_report = read_json(args.prompt_report.resolve())
    skill_result = read_json(args.skill_result.resolve())
    cfg = copy.deepcopy(load_config(str(args.config.resolve())))
    cfg.setdefault("models", {}).setdefault("task", {}).update({
        "provider": "http", "model_id": plan["target_model_id"],
        "max_retries": 5, "structured_output": "json_schema"})
    cfg.setdefault("runtime", {}).update({
        "provider": "pi", "input_mode": "task_input", "pi_provider_max_retries": 4,
        "pi_request_cooldown_seconds": max(0.0, float(os.environ.get(
            "RSI_CONFIRMATION_COOLDOWN_SECONDS", "1.0"))),
        "http_no_cache_required": True})
    cfg.setdefault("skills", {})["routing_mode"] = "lexical"
    mipro.configure_dataset_adapter("clinifact")
    parent = load_state(bundle["parent_release_id"])
    states = _states(bundle, prompt_report, skill_result, cfg, parent)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ledger = Path(plan["request_ledger"]["path"]).resolve()
    if ledger.exists() or any((out / name).exists() for name in ("report.json", "rows.jsonl")):
        raise FileExistsError("G03 output is one-shot and cannot reuse a ledger or report")
    expected_plan_sha256 = plan["plan_sha256"]
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = expected_plan_sha256
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger)
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(plan["logical_limit"])
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(plan["physical_limit"])
    rows: list[dict[str, Any]] = []
    providers: dict[tuple[str, int], Any] = {}
    for seed in plan["seeds"]:
        for arm in ARMS:
            providers[(arm, int(seed))] = mipro.build_provider(cfg, seed=int(seed))
            for task_id in ordered_ids:
                task = task_map[task_id]
                outcome = TaskRunner(cfg, states[arm], provider=providers[(arm, int(seed))],
                                     seed=int(seed)).run(task)
                row = validate_task_answer_row(
                    outcome.to_dict(), task,
                    context=f"CliniFact-G03:{arm}:{seed}:{task_id}",
                    require_reported_gold_index=True)
                row.update({"arm": arm, "seed": int(seed),
                            "candidate_bundle_id": plan["candidate_bundle_id"]})
                rows.append(row)
                with (out / "rows.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush(); os.fsync(handle.fileno())
    by_key = {(str(row["task_id"]), int(row["seed"]), row["arm"]): row for row in rows}
    expected = len(ordered_ids) * len(plan["seeds"])
    metrics = {arm: _metric([row for row in rows if row["arm"] == arm]) for arm in ARMS}
    contrasts: dict[str, Any] = {}
    for arm in ("A1", "A2"):
        per_seed = {}
        for seed in plan["seeds"]:
            base = [int(by_key[(task_id, int(seed), "A0")]["correct"]) for task_id in ordered_ids]
            after = [int(by_key[(task_id, int(seed), arm)]["correct"]) for task_id in ordered_ids]
            per_seed[str(seed)] = _paired(base, after, seed=int(plan["paired_statistics"]["bootstrap_seed"]) + int(seed),
                                           B=int(plan["paired_statistics"]["bootstrap_replicates"]))
        contrasts[f"{arm}_vs_A0"] = per_seed
    accounting = summarize_ledger(ledger)
    if accounting.get("logical_operations") != plan["minimum_logical_operations"] \
            or accounting.get("logical_incomplete") != 0 \
            or accounting.get("physical_incomplete") != 0 \
            or accounting.get("physical_attempts_exact") is not True \
            or accounting.get("physical_http_attempts", 0) > plan["physical_limit"]:
        raise ValueError("G03 ledger does not reconcile with the frozen plan")
    report: dict[str, Any] = {
        "schema": "g03-fixed-candidate-ablation-report-v1",
        "status": "completed",
        "candidate_bundle_id": plan["candidate_bundle_id"],
        "candidate_bundle_sha256": plan["candidate_bundle_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "target_model_id": plan["target_model_id"],
        "target_protocol": plan["target_protocol"],
        "arms": list(ARMS), "seeds": plan["seeds"],
        "task_count": len(ordered_ids), "expected_rows_per_arm": expected,
        "metrics": metrics, "paired_contrasts": contrasts,
        "rows_path": str((out / "rows.jsonl").resolve()),
        "request_ledger_path": str(ledger),
        "request_ledger_sha256": sha256_file(ledger), "request_accounting": accounting,
        "official_test_read": False, "official_validation_read": False,
        "selection_gate_read": False, "final_holdout_read": False,
    }
    write_json(out / "report.json", report)
    return {"status": report["status"], "report_path": str((out / "report.json").resolve()),
            "request_accounting": accounting, "metrics": metrics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-file-sha256", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--prompt-report", type=Path, required=True)
    parser.add_argument("--skill-result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
