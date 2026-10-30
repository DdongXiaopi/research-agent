#!/usr/bin/env python3
"""Finalize a complete bundle confirmation after explicit Pi timeout closure.

This is a derived artifact operation.  It never changes the original
one-shot ledger; it copies it, closes only HTTP starts proven to have been
terminated by the outer Pi timeout, and records both hashes in the report.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.stats import mcnemar_exact, paired_bootstrap_binary  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.util import read_json, sha256_file, write_json  # noqa: E402


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = {}
    for label in ("EVIDENCE", "INCONCLUSIVE", "NEI"):
        subset = [row for row in rows if row.get("gold_label") == label]
        by_label[label] = {"n": len(subset), "correct": sum(int(bool(row.get("correct"))) for row in subset),
                           "accuracy": (sum(int(bool(row.get("correct"))) for row in subset) / len(subset)
                                        if subset else 0.0)}
    return {"n": len(rows), "correct": sum(int(bool(row.get("correct"))) for row in rows),
            "accuracy": (sum(int(bool(row.get("correct"))) for row in rows) / len(rows) if rows else 0.0),
            "per_gold_label": by_label}


def _close_timeout_starts(source: Path, target: Path) -> tuple[str, int]:
    events = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    starts = {(str(item.get("operation_id") or ""), int(item.get("attempt", 0)))
              for item in events if item.get("event") == "http_attempt_start"}
    ends = {(str(item.get("operation_id") or ""), int(item.get("attempt", 0)))
            for item in events if item.get("event") == "http_attempt_end"}
    already = {(str(item.get("operation_id") or ""), int(item.get("attempt", 0)))
               for item in events if item.get("event") == "http_attempt_timeout"}
    missing = sorted(starts - ends - already)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if missing:
        with target.open("a", encoding="utf-8", newline="\n") as handle:
            for operation_id, attempt in missing:
                handle.write(json.dumps({"event": "http_attempt_timeout",
                                         "operation_id": operation_id,
                                         "attempt": attempt,
                                         "error_type": "PiBridgeTimeout",
                                         "source_ledger_sha256": sha256_file(source)},
                                        sort_keys=True, separators=(",", ":")) + "\n")
    return sha256_file(source), len(missing)


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = read_json(args.plan.resolve())
    if plan.get("schema") != "g02-bundle-confirmation-plan-v1":
        raise ValueError("unsupported bundle confirmation plan")
    source_ledger = args.ledger.resolve()
    final_ledger = args.final_ledger.resolve()
    if final_ledger.exists():
        raise FileExistsError("final ledger already exists")
    original_sha, timeout_count = _close_timeout_starts(source_ledger, final_ledger)
    accounting = summarize_ledger(final_ledger)
    logical_limit = int(plan["logical_request_budget"])
    physical_limit = int(plan["maximum_physical_attempts"])
    if (accounting.get("logical_operations") != logical_limit
            or accounting.get("logical_incomplete") != 0
            or accounting.get("physical_incomplete") != 0
            or accounting.get("physical_attempts_exact") is not True
            or accounting.get("physical_http_attempts", 0) > physical_limit):
        raise ValueError("derived bundle confirmation ledger does not reconcile")
    base = _rows(args.baseline.resolve())
    cand = _rows(args.candidate.resolve())
    if len(base) != len(cand) or len(base) * 2 != logical_limit:
        raise ValueError("baseline/candidate rows do not cover the frozen plan")
    base_by = {str(row["task_id"]): row for row in base}
    cand_by = {str(row["task_id"]): row for row in cand}
    if set(base_by) != set(cand_by):
        raise ValueError("baseline and candidate task IDs differ")
    ordered_ids = sorted(base_by)
    ordered_base = [base_by[key] for key in ordered_ids]
    ordered_cand = [cand_by[key] for key in ordered_ids]
    base_metrics, cand_metrics = _metrics(ordered_base), _metrics(ordered_cand)
    per_class_nonregression = all(
        cand_metrics["per_gold_label"][label]["accuracy"] + 1e-12 >=
        base_metrics["per_gold_label"][label]["accuracy"]
        for label in ("EVIDENCE", "INCONCLUSIVE", "NEI"))
    strict_gain = cand_metrics["accuracy"] > base_metrics["accuracy"] + 1e-12
    passed = bool(strict_gain and per_class_nonregression)
    report = {
        "schema": "g03-candidate-bundle-confirmation-report-v1",
        "status": "independent_confirmation_passed" if passed else "independent_confirmation_failed",
        "independent_confirmation_completed": True,
        "independent_confirmation_passed": passed,
        "confirmation_eligible": passed,
        "candidate_bundle_id": plan["candidate_bundle_id"],
        "candidate_bundle_sha256": plan["candidate_bundle_sha256"],
        "candidate_members": plan["candidate_members"],
        "parent_release_id": plan["parent_release_id"],
        "target_model_id": plan["target_model_id"],
        "source_search_model_id": plan["target_model_id"],
        "target_protocol": plan["target_protocol"],
        "source_cohort_sha256": plan["source_cohort_sha256"],
        "confirmation_cohort_sha256": plan["confirmation_cohort_sha256"],
        "plan": plan,
        "baseline_metrics": base_metrics,
        "candidate_metrics": cand_metrics,
        "strict_overall_gain": strict_gain,
        "per_class_nonregression": per_class_nonregression,
        "paired_statistics": {
            "mcnemar_exact": mcnemar_exact([int(bool(row["correct"])) for row in ordered_base],
                                             [int(bool(row["correct"])) for row in ordered_cand]),
            "paired_bootstrap": paired_bootstrap_binary(
                [int(bool(row["correct"])) for row in ordered_base],
                [int(bool(row["correct"])) for row in ordered_cand], B=10000,
                rng_seed=int(plan["execution_seed"]) + 10),
        },
        "baseline_rows_path": str(args.baseline.resolve()),
        "candidate_rows_path": str(args.candidate.resolve()),
        "request_ledger_sha256": sha256_file(final_ledger),
        "request_ledger_path": str(final_ledger),
        "request_accounting": accounting,
        "ledger_correction": {"original_ledger_path": str(source_ledger),
                              "original_ledger_sha256": original_sha,
                              "derived_ledger_sha256": sha256_file(final_ledger),
                              "closed_timeout_slots": timeout_count,
                              "method": "terminal_http_attempt_timeout_event_v1"},
        "official_test_read": False,
        "official_validation_read": False,
        "selection_gate_read": False,
    }
    write_json(args.report.resolve(), report)
    return {"status": report["status"], "report_path": str(args.report.resolve()),
            "baseline_accuracy": base_metrics["accuracy"],
            "candidate_accuracy": cand_metrics["accuracy"],
            "request_accounting": accounting,
            "closed_timeout_slots": timeout_count}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--final-ledger", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
