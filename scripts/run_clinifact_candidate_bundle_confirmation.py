#!/usr/bin/env python3
"""Run a fresh paired confirmation for an exact prompt+Skill candidate bundle."""
from __future__ import annotations

import argparse
import copy
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
from scripts.run_clinifact_candidate_confirmation import (  # noqa: E402
    MAX_ATTEMPTS, _get_dotenv_key, _metrics, _tasks_from_reference,
)
from src.config import load_config  # noqa: E402
from src.contracts.types import Candidate, content_sha256  # noqa: E402
from src.data.clinifact_task_input import LABELS  # noqa: E402
from src.eval.clinifact_candidate_bundle import validate_candidate_bundle  # noqa: E402
from src.eval.g03_ablation_plan import ATTESTATION_SCHEMA  # noqa: E402
from src.eval.stats import mcnemar_exact, paired_bootstrap_binary  # noqa: E402
from src.eval.task_rows import validate_task_answer_row  # noqa: E402
from src.evolution.task_program import serialize_task_program  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.runtime.runner import ArtifactState, TaskRunner  # noqa: E402
from src.runtime.skills import with_description  # noqa: E402
from src.util import artifacts_path, read_json, sha256_file, sha256_obj, write_json  # noqa: E402

TASK_MODEL = "XingChenAGI/Xing4.0-29B"
TARGET_PROTOCOL = "clinifact-task-input-bundle-confirmation-v1"
EXECUTION_SEED = 2026101802
PROTOCOL = "clinifact-bundle-independent-confirmation-v1"


def confirmation_model_identity() -> tuple[str, str | None, str | None]:
    """Return the independently configured confirmation model identity.

    The historical default remains Xing for backward-compatible plans. New
    runs may select any approved OpenAI-compatible endpoint without editing
    this runner; the selected model/endpoint/credential variable are written
    into the immutable plan and report.
    """
    return (
        str(os.environ.get("RSI_CONFIRMATION_MODEL_ID") or TASK_MODEL),
        (str(os.environ["RSI_CONFIRMATION_BASE_URL"])
         if os.environ.get("RSI_CONFIRMATION_BASE_URL") else None),
        (str(os.environ["RSI_CONFIRMATION_API_KEY_ENV"])
         if os.environ.get("RSI_CONFIRMATION_API_KEY_ENV") else None),
    )


def _load_bundle_inputs(bundle_path: Path, cohort_path: Path,
                        reference_path: Path) -> tuple[dict, dict, list[dict]]:
    bundle = read_json(bundle_path)
    validate_candidate_bundle(bundle)
    cohort = read_json(cohort_path)
    unsigned = dict(cohort)
    cohort_sha = unsigned.pop("cohort_sha256", None)
    if (cohort.get("eligible_for_confirmation") is not True
            or cohort.get("official_test_read") is not False
            or cohort.get("selection_gate_read") is not False
            or not cohort_sha or sha256_obj(unsigned) != cohort_sha):
        raise ValueError("bundle confirmation cohort is invalid or ineligible")
    binding = cohort.get("candidate_binding") or {}
    if (binding.get("candidate_bundle_id") != bundle["candidate_bundle_id"]
            or binding.get("candidate_bundle_sha256") != bundle["candidate_bundle_sha256"]):
        raise ValueError("confirmation cohort is not bound to the exact candidate bundle")
    reference = read_json(reference_path)
    if (reference.get("schema") != "clinifact-blind-ai-reference-v1"
            or reference.get("candidate_binding") != binding
            or reference.get("capacity_audit", {}).get("eligible") is not True):
        raise ValueError("AI reference is not the exact eligible bundle reference")
    rows = reference.get("consensus") or []
    if len(rows) != cohort.get("task_count"):
        raise ValueError("reference and cohort task counts differ")
    counts = {label: sum(row.get("resolved_label") == label for row in rows) for label in LABELS}
    if counts != cohort.get("class_counts") or any(counts[label] < 30 for label in LABELS):
        raise ValueError("bundle confirmation reference does not meet 30-per-class capacity")
    if any(row.get("resolution") not in {"independent_agreement", "third_ai_majority"} for row in rows):
        raise ValueError("bundle confirmation reference contains unresolved rows")
    return bundle, cohort, rows


def _candidate_states(bundle: dict, prompt_report: dict, prompt_path: Path,
                      skill_result: dict, skill_path: Path, cfg: dict,
                      parent: ArtifactState) -> tuple[ArtifactState, str]:
    candidate = prompt_report.get("candidate") or {}
    prompt = str(candidate.get("system_prompt") or "")
    demos = candidate.get("task_demos")
    if not prompt or not isinstance(demos, list):
        raise ValueError("prompt report lacks a frozen task program")
    serialized = serialize_task_program(prompt, demos)
    if content_sha256(prompt) != bundle["members"][0]["content_sha256"]:
        raise ValueError("prompt report content differs from bundle member")
    skill_member = next(row for row in bundle["members"] if row["role"] == "skill_description")
    skill_result_candidate = skill_result.get("candidate") or {}
    description = str(skill_result_candidate.get("new_content") or "").strip()
    if content_sha256(description) != skill_member["content_sha256"]:
        raise ValueError("Skill result content differs from bundle member")
    prompt_artifact = Candidate(
        candidate_id=bundle["members"][0]["candidate_id"], parent_release_id=parent.release_id,
        artifact_type="system_prompt", artifact_id="system_prompt",
        new_content=prompt, content_sha256=content_sha256(prompt),
        diff="frozen prompt candidate from the exact bundle")
    if static_gate(prompt_artifact, cfg, parent).get("passed") is not True:
        raise ValueError("prompt bundle member fails the current static gate")
    skill_artifact = Candidate(
        candidate_id=skill_member["candidate_id"], parent_release_id=parent.release_id,
        artifact_type="skill_description", artifact_id=bundle["skill_name"],
        new_content=description, content_sha256=content_sha256(description),
        diff="frozen Skill-description candidate from the exact bundle")
    if static_gate(skill_artifact, cfg, parent).get("passed") is not True:
        raise ValueError("Skill-description bundle member fails the current static gate")
    state = ArtifactState(
        release_id=parent.release_id, system_prompt=prompt,
        registry=with_description(parent.registry, bundle["skill_name"], description),
        task_demos=[dict(x) for x in demos])
    return state, content_sha256(serialized)


def _score(cfg: dict, baseline: ArtifactState, candidate: ArtifactState,
           tasks: list[dict], schedule: dict[str, list[str]], out: Path,
           candidate_id: str) -> tuple[list[dict], list[dict]]:
    providers = {arm: mipro.build_provider(cfg, seed=EXECUTION_SEED)
                 for arm in ("baseline", "candidate")}
    states = {"baseline": baseline, "candidate": candidate}
    rows = {"baseline": [], "candidate": []}
    baseline_path, candidate_path = out / "baseline.jsonl", out / "candidate.jsonl"
    with baseline_path.open("w", encoding="utf-8") as base_handle, candidate_path.open("w", encoding="utf-8") as cand_handle:
        handles = {"baseline": base_handle, "candidate": cand_handle}
        for task in tasks:
            for arm in schedule[str(task["task_id"])]:
                outcome = TaskRunner(cfg, states[arm], provider=providers[arm], seed=EXECUTION_SEED).run(task)
                row = validate_task_answer_row(outcome.to_dict(), task,
                    context=f"CliniFact-bundle-confirmation:{arm}:{task['task_id']}",
                    require_reported_gold_index=True)
                row.update({"seed": EXECUTION_SEED, "gold_label": task["gold_label"],
                            "reference_case_id": task["metadata"]["reference_case_id"]})
                if arm == "candidate":
                    row["candidate_bundle_id"] = candidate_id
                handles[arm].write(json.dumps(row, ensure_ascii=False) + "\n")
                handles[arm].flush(); os.fsync(handles[arm].fileno())
                rows[arm].append(row)
    return rows["baseline"], rows["candidate"]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("bundle confirmation requires an isolated RSI_HOME")
    bundle_path, cohort_path, reference_path = (args.bundle.resolve(), args.cohort.resolve(), args.reference.resolve())
    bundle, cohort, reference_rows = _load_bundle_inputs(bundle_path, cohort_path, reference_path)
    prompt_report = read_json(args.prompt_report.resolve())
    skill_result = read_json(args.skill_result.resolve())
    cfg = load_config(str(args.config.resolve()))
    cfg = copy.deepcopy(cfg)
    target_model, target_base_url, target_api_key_env = confirmation_model_identity()
    cfg.setdefault("models", {}).setdefault("task", {}).update({
        "provider": "http", "model_id": target_model, "max_retries": MAX_ATTEMPTS,
        "structured_output": "json_schema"})
    if target_base_url:
        cfg["models"]["task"]["base_url"] = target_base_url
    if target_api_key_env:
        cfg["models"]["task"]["api_key_env"] = target_api_key_env
    cfg.setdefault("runtime", {}).update({"provider": "pi", "input_mode": "task_input",
                                           # Free-tier providers commonly enforce a rolling RPM/TPM
                                           # window.  Keep this bounded and explicit in the plan so a
                                           # transient 429 cannot consume the whole confirmation run.
                                           "pi_request_cooldown_seconds": max(
                                               0.0, float(os.environ.get(
                                                   "RSI_CONFIRMATION_COOLDOWN_SECONDS", "1.0"))),
                                           "pi_provider_max_retries": 4,
                                           "http_no_cache_required": True})
    cfg.setdefault("skills", {})["routing_mode"] = "lexical"
    mipro.configure_dataset_adapter("clinifact")
    parent = load_state(bundle["parent_release_id"])
    active = artifacts_path("releases", "ACTIVE")
    if parent.release_id != bundle["parent_release_id"] or not active.is_file():
        raise ValueError("isolated RSI_HOME does not contain the explicit parent release")
    candidate_state, candidate_content_sha = _candidate_states(
        bundle, prompt_report, args.prompt_report, skill_result, args.skill_result, cfg, parent)
    tasks = _tasks_from_reference(reference_rows)
    task_ids = sorted(str(row["task_id"]) for row in tasks)
    order = sorted(tasks, key=lambda row: str(row["task_id"]))
    random.Random(EXECUTION_SEED).shuffle(order)
    first_baseline = {row["task_id"] for row in order[:len(order) // 2]}
    schedule = {row["task_id"]: (["baseline", "candidate"] if row["task_id"] in first_baseline
                                  else ["candidate", "baseline"]) for row in tasks}
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    plan_path, report_path, ledger_path = out / "plan.json", out / "report.json", out / "request-ledger.jsonl"
    if any(path.exists() for path in (plan_path, report_path, ledger_path, out / "baseline.jsonl", out / "candidate.jsonl")):
        raise FileExistsError("bundle confirmation output is one-shot")
    logical_limit, physical_limit = len(tasks) * 2, len(tasks) * 2 * MAX_ATTEMPTS
    source_cohort_sha = str((read_json(args.source_cohort.resolve()) if args.source_cohort else {}).get("cohort_sha256") or bundle.get("source_cohort_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", source_cohort_sha):
        source_cohort_sha = sha256_obj({"bundle": bundle["candidate_bundle_sha256"], "purpose": "g02-source"})
    plan = {"schema": "g02-bundle-confirmation-plan-v1", "protocol": PROTOCOL,
            "candidate_bundle_id": bundle["candidate_bundle_id"],
            "candidate_bundle_sha256": bundle["candidate_bundle_sha256"],
            "candidate_members": [{k: row[k] for k in ("role", "candidate_id", "artifact_type", "artifact_id", "content_sha256")} for row in bundle["members"]],
            "parent_release_id": bundle["parent_release_id"],
            "target_protocol": TARGET_PROTOCOL, "source_cohort_sha256": source_cohort_sha,
            "confirmation_cohort_sha256": cohort["cohort_sha256"],
            "confirmation_cohort_file_sha256": sha256_file(cohort_path),
            "reference_file_sha256": sha256_file(reference_path), "task_ids": task_ids,
            "task_count": len(tasks), "execution_seed": EXECUTION_SEED,
            "target_model_id": target_model,
            "arm_order_schedule": schedule, "arm_order_schedule_sha256": sha256_obj(schedule),
            "logical_request_budget": logical_limit, "maximum_physical_attempts": physical_limit,
            "official_test_read": False, "official_validation_read": False,
            "selection_gate_read": False, "candidate_tuning": False,
            "source_code_sha256": {"runner": sha256_file(Path(__file__))}}
    plan["plan_sha256"] = sha256_obj(plan)
    write_json(plan_path, plan)
    _get_dotenv_key()
    expected_plan_sha256 = plan["plan_sha256"]
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = expected_plan_sha256
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path)
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(logical_limit)
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(physical_limit)
    base_rows, candidate_rows = _score(cfg, parent, candidate_state, order, schedule, out, bundle["candidate_bundle_id"])
    base_by, cand_by = {r["task_id"]: r for r in base_rows}, {r["task_id"]: r for r in candidate_rows}
    if set(base_by) != set(cand_by) or set(base_by) != set(task_ids):
        raise ValueError("bundle confirmation arms do not cover the same frozen tasks")
    ordered_base, ordered_cand = [base_by[k] for k in task_ids], [cand_by[k] for k in task_ids]
    base_metrics, cand_metrics = _metrics(ordered_base), _metrics(ordered_cand)
    per_class_nonregression = all(cand_metrics["per_gold_label"][label]["accuracy"] + 1e-12 >= base_metrics["per_gold_label"][label]["accuracy"] for label in LABELS)
    strict_gain = cand_metrics["accuracy"] > base_metrics["accuracy"] + 1e-12
    passed = bool(strict_gain and per_class_nonregression)
    accounting = summarize_ledger(ledger_path)
    if accounting.get("logical_operations") != logical_limit or accounting.get("logical_incomplete") != 0 or accounting.get("physical_incomplete") != 0 or accounting.get("physical_http_attempts") > physical_limit:
        raise ValueError("bundle confirmation ledger does not reconcile")
    report = {"schema": ATTESTATION_SCHEMA,
              "status": "independent_confirmation_passed" if passed else "independent_confirmation_failed",
              "independent_confirmation_completed": True,
              "independent_confirmation_passed": passed, "confirmation_eligible": passed,
              "candidate_bundle_id": bundle["candidate_bundle_id"], "candidate_bundle_sha256": bundle["candidate_bundle_sha256"],
              "candidate_members": plan["candidate_members"], "parent_release_id": bundle["parent_release_id"],
              "target_model_id": target_model, "source_search_model_id": target_model,
              "target_protocol": TARGET_PROTOCOL, "source_cohort_sha256": source_cohort_sha,
              "confirmation_cohort_sha256": cohort["cohort_sha256"], "plan": plan,
              "baseline_metrics": base_metrics, "candidate_metrics": cand_metrics,
              "strict_overall_gain": strict_gain, "per_class_nonregression": per_class_nonregression,
              "paired_statistics": {"mcnemar_exact": mcnemar_exact([int(r["correct"]) for r in ordered_base], [int(r["correct"]) for r in ordered_cand]),
                                    "paired_bootstrap": paired_bootstrap_binary([int(r["correct"]) for r in ordered_base], [int(r["correct"]) for r in ordered_cand], B=10000, rng_seed=EXECUTION_SEED + 10)},
              "baseline_rows_path": str((out / "baseline.jsonl").resolve()), "candidate_rows_path": str((out / "candidate.jsonl").resolve()),
              "request_ledger_sha256": sha256_file(ledger_path), "request_accounting": accounting,
              "official_test_read": False, "official_validation_read": False, "selection_gate_read": False}
    write_json(report_path, report)
    return {"status": report["status"], "report_path": str(report_path),
            "candidate_bundle_id": bundle["candidate_bundle_id"],
            "baseline_accuracy": base_metrics["accuracy"], "candidate_accuracy": cand_metrics["accuracy"],
            "request_accounting": accounting}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prompt-report", type=Path, required=True)
    parser.add_argument("--skill-result", type=Path, required=True)
    parser.add_argument("--source-cohort", type=Path, required=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
