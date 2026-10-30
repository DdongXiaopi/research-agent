"""Merge completed HealthVer rationales into an offline, non-relabeling review queue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.util import read_json, sha256_file, sha256_obj, write_json  # noqa: E402

AUDIT_PATHS = (
    ROOT / "artifacts/probe/healthver_mipro_fresh_error_attribution.json",
    ROOT / "artifacts/probe/healthver_support_probe_rationale_audit.json",
    ROOT / "artifacts/probe/healthver_remaining_search_rationale_audit.json",
)
DEFAULT_OUTPUT = ROOT / "artifacts/probe/healthver_rationale_review_queue.json"

OBSERVATIONS = {
    "healthver-claim-538-abstract-186": (
        "The cited rationale discusses plant-based foods, immunity and COVID-19; "
        "it does not directly discuss lemon juice tea or its efficacy."),
    "healthver-claim-775-abstract-41": (
        "The cited rationale discusses general mental/physical health benefits and "
        "park closures; it does not directly assert the claim about crisis-related "
        "mental-health harms."),
    "healthver-claim-88-abstract-21": (
        "The cited rationale discusses remdesivir, vaccines, masks and distancing; "
        "it does not mention apple cider vinegar."),
    "healthver-claim-1239-abstract-230": (
        "The cited rationale reports recovered COVID-19 patients later testing "
        "positive; the claim contrasts MERS reinfection with uncertainty about "
        "COVID-19 protection."),
    "healthver-claim-1456-abstract-325": (
        "The cited rationale reports in-vitro favipiravir activity against SARS-CoV-2; "
        "the broader claim also mentions trial count and unknown safety/benefit."),
    "healthver-claim-707-abstract-56": (
        "The cited rationale says no vaccine or therapy was then available; it does "
        "not directly address whether six vaccine candidates were considered likely "
        "to succeed."),
}


def build_queue(audit_paths: tuple[Path, ...] = AUDIT_PATHS) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    candidate_ids: set[str] = set()
    archive_hashes: set[str] = set()
    for path in audit_paths:
        audit = read_json(path)
        if audit.get("rationale_text_used_as_runtime_input") is not False:
            raise ValueError(f"audit says rationale text was used at runtime: {path}")
        if audit.get("rationale_text_used_for_candidate_selection") is not False:
            raise ValueError(f"audit says rationales affected candidate selection: {path}")
        hashes[path.name] = sha256_file(path)
        candidate_ids.add(str(audit.get("candidate_id") or ""))
        archive_hashes.add(str(audit.get("source_archive_sha256") or ""))
        for item in audit.get("tasks") or []:
            task_id = str(item["task_id"])
            if task_id in records:
                raise ValueError(f"duplicate HealthVer task in rationale audits: {task_id}")
            if task_id not in OBSERVATIONS:
                raise ValueError(f"no offline review observation recorded for {task_id}")
            records[task_id] = item
    if len(candidate_ids) != 1 or "" in candidate_ids:
        raise ValueError("rationale audits do not share one frozen candidate")
    if len(archive_hashes) != 1 or "" in archive_hashes:
        raise ValueError("rationale audits do not share one source archive identity")
    if set(records) != set(OBSERVATIONS):
        raise ValueError("rationale review queue is missing or has unexpected changed tasks")

    tasks = []
    for task_id in sorted(records):
        row = records[task_id]
        tasks.append({
            "task_id": task_id,
            "claim": row["claim"],
            "gold_label": row.get("gold_document_label", row.get("gold_label")),
            "source_title": row.get("source_title", ""),
            "prediction_changes": row["prediction_changes"],
            "source_gold_rationales": row["source_gold_rationales"],
            "offline_observation": OBSERVATIONS[task_id],
            "human_adjudication_status": "not_reviewed",
            "source_gold_label_changed": False,
            "use_as_optimizer_demo_before_review": False,
        })
    queue = {
        "schema_version": 1,
        "protocol": "healthver-rationale-review-queue-v1",
        "candidate_id": next(iter(candidate_ids)),
        "source_archive_sha256": next(iter(archive_hashes)),
        "input_audit_sha256": hashes,
        "task_count": len(tasks),
        "tasks": tasks,
        "rationale_text_used_as_runtime_input": False,
        "rationale_text_used_for_candidate_selection": False,
        "labels_rewritten": False,
        "confirmation_eligible": False,
    }
    queue["queue_sha256"] = sha256_obj(queue)
    return queue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    queue = build_queue()
    if args.write:
        write_json(args.output, queue)
    print(json.dumps({
        "status": "written" if args.write else "validated_not_written",
        "path": str(args.output),
        "candidate_id": queue["candidate_id"],
        "tasks": queue["task_count"],
        "labels_rewritten": False,
        "model_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
