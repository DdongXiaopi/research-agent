#!/usr/bin/env python3
"""Run one frozen, paired CliniFact confirmation for a preselected candidate.

This adapter reuses the project's fixed-evidence TaskRunner and MIPRO candidate
contract, but consumes only the separately frozen, AI-referenced confirmation
cohort. It never optimizes the prompt or reads official validation/test data.
"""
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
from src.config import config_fingerprint  # noqa: E402
from src.contracts.types import Candidate, content_id, content_sha256  # noqa: E402
from src.data.clinifact_task_input import LABELS  # noqa: E402
from src.eval.stats import mcnemar_exact, paired_bootstrap_binary  # noqa: E402
from src.eval.task_rows import validate_task_answer_row  # noqa: E402
from src.evolution.task_program import serialize_task_program  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.runtime.runner import ArtifactState, TaskRunner  # noqa: E402
from src.util import (artifacts_path, read_json, sha256_file, sha256_obj,
                      write_json)  # noqa: E402

DEFAULT_OUT = ROOT / ".workbuddy/clinifact-ai-confirmation-20261016/candidate-confirmation"
DEFAULT_COHORT = ROOT / ".workbuddy/clinifact-ai-confirmation-20261016/supplemental-source/confirmation_cohort_v2.json"
DEFAULT_REFERENCE = ROOT / ".workbuddy/clinifact-ai-confirmation-20261016/supplemental-source/resolved_reference.json"
DEFAULT_SOURCE_REPORT = ROOT / "artifacts/sandboxes/clinifact-mipro-v18-20261014/probe/clinifact_mipro/5b9b9cf61889430e/report.json"
DEFAULT_CONFIG = ROOT / "artifacts/sandboxes/ei-mipro-live-v2-20261005/frozen-run-config.yaml"
TASK_MODEL = "Qwen/Qwen3-8B"  # verified free-model allowlist; source search used paid 32B
EXECUTION_SEED = 2026101601
MAX_ATTEMPTS = 5
PROTOCOL = "clinifact-ai-reference-independent-candidate-confirmation-v1"


def _load_inputs(cohort_path: Path, reference_path: Path) -> tuple[dict, list[dict]]:
    cohort = read_json(cohort_path)
    unsigned = dict(cohort)
    digest = unsigned.pop("cohort_sha256", None)
    if (not digest or sha256_obj(unsigned) != digest
            or cohort.get("schema") != "clinifact-ai-confirmation-cohort-v2"
            or cohort.get("eligible_for_confirmation") is not True
            or cohort.get("official_test_read") is not False
            or cohort.get("selection_gate_read") is not False):
        raise ValueError("CliniFact confirmation cohort is invalid or ineligible")
    reference = read_json(reference_path)
    if (sha256_file(reference_path) != cohort.get("supplemental_reference_sha256")
            or reference.get("schema") != "clinifact-ai-confirmation-reference-v2"
            or reference.get("candidate_binding") != cohort.get("candidate_binding")
            or reference.get("capacity_audit", {}).get("eligible") is not True):
        raise ValueError("AI reference does not bind the frozen eligible cohort")
    rows = reference.get("consensus") or []
    if (len(rows) != cohort.get("task_count")
            or {row["task_id"] for row in rows} != set(cohort.get("task_ids") or [])
            or {row["component_id"] for row in rows} != set(cohort.get("component_ids") or [])):
        raise ValueError("AI reference rows differ from the frozen confirmation cohort")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("confirmation tasks are not unique")
    expected = {"EVIDENCE", "INCONCLUSIVE", "NEI"}
    counts = {label: sum(row.get("resolved_label") == label for row in rows)
              for label in LABELS}
    if set(counts) != expected or counts != cohort.get("class_counts") or any(n < 30 for n in counts.values()):
        raise ValueError("AI reference class capacity does not meet the frozen 30-per-class minimum")
    if not rows or any(row.get("resolution") not in {"independent_agreement", "third_ai_majority"}
                       for row in rows):
        raise ValueError("confirmation rows must have two-reviewer agreement or third-model majority")
    return cohort, rows


def _tasks_from_reference(rows: list[dict]) -> list[dict]:
    tasks = []
    for item in rows:
        source = item["input"]
        label = item.get("resolved_label")
        if label not in LABELS:
            raise ValueError("reference contains an unsupported CliniFact label")
        tasks.append({
            "task_id": str(item["task_id"]), "dataset": "clinifact_fixed_abstract",
            "split": "independent_confirmation", "task_type": "mcq",
            "domain": "clinical_research_claim_verification",
            "question": ("Using only the supplied publication abstract, classify the claim according "
                         "to whether the abstract reports supporting evidence, an inconclusive result, "
                         "or not enough information.\nClaim: " + str(source["claim"])),
            "options": list(LABELS), "correct_index": LABELS.index(label), "gold_label": label,
            "context": "", "evidence": str(source["title"]) + "\n" + str(source["abstract"]),
            "sources": ["independent:" + str(source["task_id"])],
            "group_id": str(item["component_id"]),
            "metadata": {"reference_case_id": item["case_id"],
                         "reference_resolution": item["resolution"],
                         "input_sha256": item["input_sha256"],
                         "fixed_evidence": True},
        })
    return tasks


def _candidate_state(report: dict, parent: ArtifactState, cfg: dict) -> tuple[ArtifactState, str, str]:
    candidate = report.get("candidate") or {}
    prompt = str(candidate.get("system_prompt") or "")
    demos = candidate.get("task_demos")
    if not prompt or not isinstance(demos, list):
        raise ValueError("candidate report lacks the frozen task program")
    serialized = serialize_task_program(prompt, demos)
    candidate_id = content_id("cand", mipro.RELEASE_ID, "task_program", serialized)
    if candidate_id != report.get("candidate_id"):
        raise ValueError("reconstructed task program differs from the preselected candidate")
    program_cfg = copy.deepcopy(cfg)
    program_cfg["evolution"]["allowed_artifacts"] = ["task_program"]
    artifact = Candidate(candidate_id=candidate_id, parent_release_id=mipro.RELEASE_ID,
                         artifact_type="task_program", artifact_id="task_program",
                         diff="frozen preselected CliniFact task program",
                         new_content=serialized, content_sha256=content_sha256(serialized))
    gate = static_gate(artifact, program_cfg, parent)
    if gate.get("passed") is not True:
        raise ValueError("preselected task program fails its current static gate")
    state = ArtifactState(release_id=parent.release_id, system_prompt=prompt,
                          registry=parent.registry, task_demos=[dict(x) for x in demos])
    return state, candidate_id, content_sha256(serialized)


def _metrics(rows: list[dict]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty candidate arm")
    classes = {}
    for label in LABELS:
        subset = [row for row in rows if row["gold_label"] == label]
        if not subset:
            raise ValueError("confirmation cohort is missing a label class")
        classes[label] = {"n": len(subset), "accuracy": sum(bool(x["correct"]) for x in subset) / len(subset),
                          "coverage": sum(int(x["answer_index"]) >= 0 for x in subset) / len(subset),
                          "answered_precision": (sum(bool(x["correct"]) for x in subset) /
                                                 max(1, sum(int(x["answer_index"]) >= 0 for x in subset)))}
    answered = sum(int(x["answer_index"]) >= 0 for x in rows)
    correct = sum(bool(x["correct"]) for x in rows)
    return {"n": len(rows), "accuracy": correct / len(rows), "coverage": answered / len(rows),
            "answered_precision": correct / max(1, answered), "per_gold_label": classes}


def _score_paired_arms(cfg: dict, baseline_state: ArtifactState,
                       candidate_state: ArtifactState, tasks: list[dict],
                       candidate_id: str, schedule: dict[str, list[str]],
                       baseline_path: Path, candidate_path: Path) -> tuple[list[dict], list[dict]]:
    providers = {arm: mipro.build_provider(cfg, seed=EXECUTION_SEED)
                 for arm in ("baseline", "candidate")}
    states = {"baseline": baseline_state, "candidate": candidate_state}
    paths = {"baseline": baseline_path, "candidate": candidate_path}
    rows = {"baseline": [], "candidate": []}
    with (baseline_path.open("w", encoding="utf-8") as baseline_handle,
          candidate_path.open("w", encoding="utf-8") as candidate_handle):
        handles = {"baseline": baseline_handle, "candidate": candidate_handle}
        for task in tasks:
            task_id = str(task["task_id"])
            for arm in schedule[task_id]:
                outcome = TaskRunner(cfg, states[arm], provider=providers[arm],
                                    seed=EXECUTION_SEED).run(task)
                row = validate_task_answer_row(
                    outcome.to_dict(), task,
                    context=f"CliniFact-confirmation:{arm}:{task_id}",
                    require_reported_gold_index=True)
                row.update({"seed": EXECUTION_SEED, "gold_label": task["gold_label"],
                            "reference_case_id": task["metadata"]["reference_case_id"]})
                if arm == "candidate":
                    row["candidate_id"] = candidate_id
                handle = handles[arm]
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                rows[arm].append(row)
    return rows["baseline"], rows["candidate"]


def _get_dotenv_key() -> None:
    if os.environ.get("SILICONFLOW_API_KEY"):
        return
    try:
        from dotenv import dotenv_values
        value = dotenv_values(ROOT / ".env").get("SILICONFLOW_API_KEY")
    except ImportError:
        value = None
    if not value:
        raise RuntimeError("SILICONFLOW_API_KEY is absent from process environment and project .env")
    os.environ["SILICONFLOW_API_KEY"] = str(value)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("confirmation requires an isolated RSI_HOME")
    if args.task_model_id != TASK_MODEL:
        raise ValueError("only the verified free confirmation model is allowed by this protocol")
    mipro.configure_dataset_adapter("clinifact")
    cohort, reference_rows = _load_inputs(args.cohort.resolve(), args.reference.resolve())
    source_report = read_json(args.candidate_report.resolve())
    source_sha = sha256_file(args.candidate_report.resolve())
    source_plan = source_report.get("plan") or {}
    if (source_report.get("candidate_id") != cohort["candidate_binding"]["candidate_id"]
            or source_sha != cohort["candidate_binding"]["candidate_report_sha256"]
            or source_plan.get("cohort_sha256") != cohort["candidate_binding"]["source_cohort_sha256"]
            or source_report.get("candidate_status") != "internal_gate_passed"
            or source_report.get("confirmation_eligible") is not False
            or source_report.get("official_test_read") is not False
            or source_report.get("selection_gate_read") is not False
            or source_report.get("official_validation_read") is not False):
        raise ValueError("source candidate report is not the exact sealed internal preselection")
    tasks = _tasks_from_reference(reference_rows)
    task_ids = sorted(row["task_id"] for row in tasks)
    class_counts = {label: sum(row["gold_label"] == label for row in tasks) for label in LABELS}
    cfg = mipro.prepare_config(str(args.config.resolve()), proposer_calls=1,
                               routing_mode="lexical", task_model_id=TASK_MODEL,
                               task_request_cooldown_seconds=0)
    cfg["runtime"]["pi_request_cooldown_seconds"] = 0.0
    cfg["runtime"]["http_no_cache_required"] = True
    cfg["models"]["task"]["max_retries"] = MAX_ATTEMPTS
    parent = load_state(mipro.RELEASE_ID)
    active = artifacts_path("releases", "ACTIVE")
    if (parent.release_id != mipro.RELEASE_ID or not active.is_file()
            or active.read_text(encoding="utf-8").strip() != mipro.RELEASE_ID):
        raise ValueError("isolated RSI_HOME does not contain the CliniFact baseline release")
    candidate_state, candidate_id, candidate_content_sha = _candidate_state(source_report, parent, cfg)
    out = args.output.resolve()
    plan_path, report_path = out / "plan.json", out / "report.json"
    baseline_path, candidate_path = out / "baseline.jsonl", out / "candidate.jsonl"
    ledger_path = out / "request-ledger.jsonl"
    if report_path.exists() or any(p.exists() and p.stat().st_size for p in (baseline_path, candidate_path, ledger_path)):
        raise ValueError("this frozen confirmation is one-shot and already has execution artifacts")
    order = sorted(tasks, key=lambda item: item["task_id"])
    random.Random(2026101601).shuffle(order)
    first_baseline = {row["task_id"] for row in order[:len(order) // 2]}
    arm_order_schedule = {
        task_id: (["baseline", "candidate"] if task_id in first_baseline
                  else ["candidate", "baseline"])
        for task_id in task_ids
    }
    if sum(arms[0] == "baseline" for arms in arm_order_schedule.values()) != len(tasks) // 2:
        raise ValueError("confirmation arm order did not balance")
    plan = {
        "schema": "clinifact-independent-candidate-confirmation-plan-v1",
        "protocol": PROTOCOL, "candidate_id": candidate_id,
        "candidate_content_sha256": candidate_content_sha,
        "candidate_report_path": str(args.candidate_report.resolve()),
        "candidate_report_sha256": source_sha,
        "source_plan_sha256": source_plan.get("plan_sha256"),
        "source_cohort_sha256": cohort["candidate_binding"]["source_cohort_sha256"],
        "confirmation_cohort_path": str(args.cohort.resolve()),
        "confirmation_cohort_file_sha256": sha256_file(args.cohort.resolve()),
        "confirmation_cohort_sha256": cohort["cohort_sha256"],
        "reference_file_sha256": sha256_file(args.reference.resolve()),
        "task_ids": task_ids, "task_count": len(tasks), "class_counts": class_counts,
        "reference_origin": "blind independent AI agreement / third-model majority",
        "capacity_supplement_count": 1,
        "capacity_sampling_caveat": cohort["supplemental_selection_caveat"],
        "task_model": TASK_MODEL, "source_search_task_model": source_plan.get("task_model"),
        "model_change_reason": "user restricted subsequent model calls to verified free models",
        "routing_mode": "lexical", "retrieval_used": False,
        "execution_seeds": [EXECUTION_SEED], "arm_order_policy": "per-task-seeded-randomized-balanced-v1",
        "arm_order_seed": 2026101601,
        "arm_order_schedule": arm_order_schedule,
        "arm_order_schedule_sha256": sha256_obj(arm_order_schedule),
        "logical_request_budget": len(tasks) * 2,
        "maximum_physical_attempts": len(tasks) * 2 * MAX_ATTEMPTS,
        "max_attempts_per_task": MAX_ATTEMPTS,
        "config_sha256": config_fingerprint(cfg), "source_code_sha256": sha256_file(Path(__file__)),
        "official_test_read": False, "official_validation_read": False,
        "selection_gate_read": False, "candidate_tuning": False,
    }
    plan["plan_sha256"] = sha256_obj(plan)
    out.mkdir(parents=True, exist_ok=True)
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("existing confirmation plan differs from current frozen inputs")
    if not plan_path.exists():
        write_json(plan_path, plan)
    if args.validate_only:
        return {"status": "validated_not_run", "plan_path": str(plan_path),
                "plan_sha256": plan["plan_sha256"], "candidate_id": candidate_id,
                "task_count": len(tasks), "class_counts": class_counts,
                "logical_request_budget": plan["logical_request_budget"],
                "maximum_physical_attempts": plan["maximum_physical_attempts"],
                "task_model": TASK_MODEL, "model_calls": 0}
    if args.expected_plan_sha256 != plan["plan_sha256"]:
        raise ValueError("execution challenge does not match the frozen plan SHA-256")
    _get_dotenv_key()
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = plan["plan_sha256"]
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path)
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(plan["logical_request_budget"])
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(plan["maximum_physical_attempts"])
    # Score the paired arms in a balanced randomized order within each case.
    baseline_rows, candidate_rows = _score_paired_arms(
        cfg, parent, candidate_state, order, candidate_id, arm_order_schedule,
        baseline_path, candidate_path)
    base_by_id = {row["task_id"]: row for row in baseline_rows}
    cand_by_id = {row["task_id"]: row for row in candidate_rows}
    if set(base_by_id) != set(cand_by_id) or set(base_by_id) != set(task_ids):
        raise ValueError("confirmation arms did not score the exact same frozen task IDs")
    ordered_base = [base_by_id[key] for key in task_ids]
    ordered_cand = [cand_by_id[key] for key in task_ids]
    baseline_metrics, candidate_metrics = _metrics(ordered_base), _metrics(ordered_cand)
    per_class_nonregression = all(
        candidate_metrics["per_gold_label"][label]["accuracy"] + 1e-12
        >= baseline_metrics["per_gold_label"][label]["accuracy"] for label in LABELS)
    strict_gain = candidate_metrics["accuracy"] > baseline_metrics["accuracy"] + 1e-12
    passed = bool(strict_gain and per_class_nonregression)
    base_correct = [int(row["correct"]) for row in ordered_base]
    cand_correct = [int(row["correct"]) for row in ordered_cand]
    result = {
        "schema": "clinifact-independent-candidate-confirmation-report-v1",
        "status": "independent_confirmation_passed" if passed else "independent_confirmation_failed",
        "plan": plan, "candidate_id": candidate_id,
        "baseline_metrics": baseline_metrics, "candidate_metrics": candidate_metrics,
        "paired_statistics": {"mcnemar_exact": mcnemar_exact(base_correct, cand_correct),
                              "paired_bootstrap": paired_bootstrap_binary(
                                  base_correct, cand_correct, B=10000, rng_seed=2026101610)},
        "strict_overall_gain": strict_gain,
        "per_class_nonregression": per_class_nonregression,
        "baseline_rows_path": str(baseline_path), "candidate_rows_path": str(candidate_path),
        "request_accounting": summarize_ledger(ledger_path),
        "baseline_cost": [row.get("cost") for row in ordered_base],
        "candidate_cost": [row.get("cost") for row in ordered_cand],
        "independent_confirmation_completed": True,
        "independent_confirmation_passed": passed,
        "confirmation_eligible": passed,
        "official_test_read": False, "official_validation_read": False,
        "selection_gate_read": False,
    }
    if (result["request_accounting"].get("logical_operations") != plan["logical_request_budget"]
            or result["request_accounting"].get("logical_incomplete") != 0
            or result["request_accounting"].get("physical_incomplete") != 0
            or result["request_accounting"].get("physical_http_attempts") > plan["maximum_physical_attempts"]):
        raise ValueError("confirmation request ledger does not reconcile to the frozen plan")
    write_json(report_path, result)
    return {"status": result["status"], "report_path": str(report_path),
            "candidate_id": candidate_id, "class_counts": class_counts,
            "baseline_accuracy": baseline_metrics["accuracy"],
            "candidate_accuracy": candidate_metrics["accuracy"],
            "request_accounting": result["request_accounting"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--candidate-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--task-model-id", default=TASK_MODEL)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-plan-sha256", default="")
    args = parser.parse_args(argv)
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256 or ""):
        parser.error("--execute requires --expected-plan-sha256 from the frozen plan")
    print(json.dumps(run(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
