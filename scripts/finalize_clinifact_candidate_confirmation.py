#!/usr/bin/env python3
"""Offline, fail-closed reconciliation for a completed CliniFact confirmation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_clinifact_candidate_confirmation import (  # noqa: E402
    _metrics, _tasks_from_reference,
)
from src.data.clinifact_task_input import LABELS  # noqa: E402
from src.eval.stats import mcnemar_exact, paired_bootstrap_binary  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.util import read_json, sha256_file, sha256_obj, write_json  # noqa: E402


def classify_confirmation(passed: bool, model_changed: bool) -> dict[str, Any]:
    if model_changed:
        return {"status": "cross_model_transfer_passed" if passed else "cross_model_transfer_failed",
                "independent_confirmation_completed": False,
                "independent_confirmation_passed": False,
                "cross_model_replication_completed": True,
                "cross_model_transfer_passed": bool(passed),
                "confirmation_eligible": False}
    return {"status": "independent_confirmation_passed" if passed else "independent_confirmation_failed",
            "independent_confirmation_completed": True,
            "independent_confirmation_passed": bool(passed),
            "cross_model_replication_completed": False,
            "cross_model_transfer_passed": None,
            "confirmation_eligible": bool(passed)}


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid persisted row at {path}:{line_no}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object persisted row at {path}:{line_no}")
            rows.append(value)
    return rows


def finalize(run_dir: Path, *, cohort_path: Path, reference_path: Path,
             source_report_path: Path) -> dict:
    plan_path = run_dir / "plan.json"
    report_path = run_dir / "report.json"
    baseline_path = run_dir / "baseline.jsonl"
    candidate_path = run_dir / "candidate.jsonl"
    ledger_path = run_dir / "request-ledger.jsonl"
    if report_path.exists():
        raise ValueError("confirmation report already exists; refusing to overwrite")
    plan = read_json(plan_path)
    plan_unsigned = dict(plan)
    plan_hash = plan_unsigned.pop("plan_sha256", None)
    if not plan_hash or sha256_obj(plan_unsigned) != plan_hash:
        raise ValueError("frozen confirmation plan self-hash is invalid")
    if (sha256_file(cohort_path) != plan.get("confirmation_cohort_file_sha256")
            or sha256_file(reference_path) != plan.get("reference_file_sha256")
            or sha256_file(source_report_path) != plan.get("candidate_report_sha256")):
        raise ValueError("confirmation source, cohort, or reference hash changed")
    cohort = read_json(cohort_path)
    reference = read_json(reference_path)
    source_report = read_json(source_report_path)
    if (cohort.get("cohort_sha256") != plan.get("confirmation_cohort_sha256")
            or source_report.get("candidate_id") != plan.get("candidate_id")
            or reference.get("candidate_binding") != cohort.get("candidate_binding")):
        raise ValueError("candidate/cohort/reference binding mismatch")
    tasks = _tasks_from_reference(reference["consensus"])
    task_by_id = {row["task_id"]: row for row in tasks}
    expected_ids = set(plan.get("task_ids") or [])
    if (len(task_by_id) != plan.get("task_count")
            or set(task_by_id) != expected_ids
            or {label: sum(t["gold_label"] == label for t in tasks) for label in LABELS}
            != plan.get("class_counts")):
        raise ValueError("frozen plan differs from exact referenced task IDs/class counts")
    baseline, candidate = _read_jsonl(baseline_path), _read_jsonl(candidate_path)
    for name, rows in (("baseline", baseline), ("candidate", candidate)):
        by_id = {str(row.get("task_id")): row for row in rows}
        if len(rows) != plan["task_count"] or len(by_id) != len(rows) or set(by_id) != expected_ids:
            raise ValueError(f"{name} output does not cover each frozen task exactly once")
        for task_id, row in by_id.items():
            task = task_by_id[task_id]
            if (row.get("gold_label") != task["gold_label"]
                    or row.get("gold_index") != task["correct_index"]
                    or type(row.get("answer_index")) is not int
                    or row.get("correct") is not (row["answer_index"] == row["gold_index"])
                    or row.get("seed") != plan["execution_seeds"][0]):
                raise ValueError(f"{name} row failed independent outcome reconciliation: {task_id}")
            if name == "candidate" and row.get("candidate_id") != plan["candidate_id"]:
                raise ValueError(f"candidate row is not bound to the frozen candidate: {task_id}")
            if row.get("status") not in {"ok", "abstain"}:
                raise ValueError(f"{name} row has an unrecognized terminal status: {task_id}")
    accounting = summarize_ledger(ledger_path)
    if (accounting.get("logical_operations") != plan.get("logical_request_budget")
            or accounting.get("logical_incomplete") != 0
            or accounting.get("physical_incomplete") != 0
            or accounting.get("physical_attempts_exact") is not True
            or accounting.get("physical_http_attempts") > plan.get("maximum_physical_attempts")
            or accounting.get("physical_status_counts") != {"200": plan["logical_request_budget"]}
            or set(accounting.get("execution_plan_sha256s") or []) != {plan_hash}):
        raise ValueError("request ledger is incomplete or does not reconcile with the frozen plan")
    baseline_by_id = {row["task_id"]: row for row in baseline}
    candidate_by_id = {row["task_id"]: row for row in candidate}
    base_rows = [baseline_by_id[key] for key in plan["task_ids"]]
    cand_rows = [candidate_by_id[key] for key in plan["task_ids"]]
    base_metrics, cand_metrics = _metrics(base_rows), _metrics(cand_rows)
    nonregression = all(cand_metrics["per_gold_label"][label]["accuracy"] + 1e-12
                        >= base_metrics["per_gold_label"][label]["accuracy"] for label in LABELS)
    strict_gain = cand_metrics["accuracy"] > base_metrics["accuracy"] + 1e-12
    passed = strict_gain and nonregression
    model_changed = plan["source_search_task_model"] != plan["task_model"]
    outcome_classification = classify_confirmation(passed, model_changed)
    report = {
        "schema": "clinifact-independent-candidate-confirmation-report-v1",
        **outcome_classification,
        "plan": plan, "candidate_id": plan["candidate_id"],
        "baseline_metrics": base_metrics, "candidate_metrics": cand_metrics,
        "paired_statistics": {
            "mcnemar_exact": mcnemar_exact([int(r["correct"]) for r in base_rows],
                                           [int(r["correct"]) for r in cand_rows]),
            "paired_bootstrap": paired_bootstrap_binary(
                [int(r["correct"]) for r in base_rows], [int(r["correct"]) for r in cand_rows],
                B=10000, rng_seed=2026101610),
        },
        "strict_overall_gain": strict_gain,
        "per_class_nonregression": nonregression,
        "baseline_rows_path": str(baseline_path), "candidate_rows_path": str(candidate_path),
        "baseline_rows_sha256": sha256_file(baseline_path),
        "candidate_rows_sha256": sha256_file(candidate_path),
        "request_ledger_sha256": sha256_file(ledger_path),
        "request_accounting": accounting,
        "cost_summary": {
            "baseline_usd": sum(float((r.get("cost") or {}).get("usd") or 0.0) for r in base_rows),
            "candidate_usd": sum(float((r.get("cost") or {}).get("usd") or 0.0) for r in cand_rows),
            "usage_complete": all((r.get("cost") or {}).get("usage_complete") is True
                                   for r in base_rows + cand_rows),
            "all_costs_zero": all(float((r.get("cost") or {}).get("usd") or 0.0) == 0.0
                                  for r in base_rows + cand_rows),
        },
        "sampling_caveat": plan["capacity_sampling_caveat"],
        "task_model_transfer": {"source_search_model": plan["source_search_task_model"],
                                "confirmation_model": plan["task_model"],
                                "model_changed": model_changed},
        "finalization_source_sha256": sha256_file(Path(__file__)),
        "official_test_read": False, "official_validation_read": False,
        "selection_gate_read": False,
    }
    write_json(report_path, report)
    return {"status": report["status"], "report_path": str(report_path),
            "candidate_id": report["candidate_id"], "baseline_accuracy": base_metrics["accuracy"],
            "candidate_accuracy": cand_metrics["accuracy"], "class_counts": plan["class_counts"],
            "request_accounting": accounting, "cost_summary": report["cost_summary"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.run_dir.resolve(), cohort_path=args.cohort.resolve(),
                              reference_path=args.reference.resolve(),
                              source_report_path=args.candidate_report.resolve()), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
