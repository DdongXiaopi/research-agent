"""Expand the same unpublished PubMedQA discovery search to three demo families."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_fingerprint, load_config  # noqa: E402
from src.data.pubmedqa_threeway_probe import (  # noqa: E402
    DATASET, MANIFEST_PATH, NORMALIZED_PATH,
)
from src.evolution.gepa_search import visible_examples  # noqa: E402
from src.evolution.task_demo_search import (  # noqa: E402
    SEARCH_PROTOCOL_VERSION, search_task_demo_candidate, split_demo_validation,
)
from src.governance.release import load_state  # noqa: E402
from src.util import (  # noqa: E402
    artifacts_path, read_json, read_jsonl, sha256_file, write_json,
)
from scripts.run_pubmedqa_threeway_class_aware_search import _configure  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-report",
        default="artifacts/sandboxes/pubmedqa-threeway-class-aware-v1/probe/"
                "pubmedqa_threeway_v4/run-2277a7575237.json")
    args = parser.parse_args()
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("class-aware search expansion requires RSI_HOME sandbox")

    source_path = Path(args.source_report)
    source = read_json(source_path)
    if (source.get("dataset") != DATASET or source.get("eligible_for_confirmation")
            or source.get("selection_gate_access") or source.get("final_holdout_access")):
        raise RuntimeError("source report is not an unpublished dev-only exploration")
    exclusions = list((source.get("run_adjustment") or {}).get(
        "provider_timeout_excluded_task_ids") or [])
    if not exclusions:
        raise RuntimeError("source report must disclose timeout exclusions")

    cfg = _configure("configs/experiment.http.pubmedqa-pilot.yaml")
    cfg["evolution"]["fewshot"].update({
        "max_candidate_sets": 3,
        "excluded_task_ids": exclusions,
    })
    state = load_state()
    if state.release_id != source.get("release_id"):
        raise RuntimeError("release changed since the source baseline")

    limit = int((source.get("run_adjustment") or {}).get("effective_limit", 0))
    train, val = visible_examples(cfg, [DATASET], limit)
    excluded = set(exclusions)
    train = [row for row in train if str(row["task_id"]) not in excluded]
    val = [row for row in val if str(row["task_id"]) not in excluded]
    discovery, confirmation = split_demo_validation(val)
    seeds = [int(row["seed"]) for row in source.get("baseline_runs") or []]
    baseline_run_ids = {
        DATASET: {str(row["seed"]): str(row["run_id"])
                  for row in source.get("baseline_runs") or []}}
    expected_ids = {str(row["task_id"]) for row in [*train, *val]}
    failures = set()
    for run_id in baseline_run_ids[DATASET].values():
        rows = read_jsonl(artifacts_path("runs", run_id, "per_task.jsonl"))
        ids = {str(row.get("task_id")) for row in rows
               if row.get("status") == "ok"}
        if ids != expected_ids:
            raise RuntimeError(f"baseline run {run_id} does not cover the filtered cohort")
        failures.update(str(row["task_id"]) for row in rows
                        if row.get("status") == "ok" and not row.get("correct"))
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise RuntimeError("source must contain three distinct execution labels")

    planned_discovery = 3 * len(discovery) * len(seeds)
    planned_confirmation = len(confirmation) * len(seeds)
    cfg["evolution"]["fewshot"]["max_metric_calls"] = (
        planned_discovery + planned_confirmation)
    search_root = artifacts_path("search", "task_demos")
    previous_reports = set(search_root.rglob("report.json")) if search_root.exists() else set()
    candidate = search_task_demo_candidate(
        cfg, state, [DATASET], limit, seeds[0], sorted(failures), baseline_run_ids)
    new_reports = sorted(
        (set(search_root.rglob("report.json")) if search_root.exists() else set())
        - previous_reports,
        key=lambda path: path.stat().st_mtime_ns)
    if len(new_reports) != 1:
        raise RuntimeError(f"expected one expansion report, found {len(new_reports)}")
    search_path = new_reports[0]
    search = read_json(search_path)

    report = {
        "protocol": "pubmedqa-threeway-class-aware-search-expansion-v1-exploratory",
        "search_protocol_version": SEARCH_PROTOCOL_VERSION,
        "dataset": DATASET,
        "dataset_manifest_sha256": sha256_file(MANIFEST_PATH),
        "dataset_normalized_sha256": sha256_file(NORMALIZED_PATH),
        "source_run_report": str(source_path.resolve()),
        "source_run_report_sha256": sha256_file(source_path),
        "source_search_report": source.get("search_report_path"),
        "source_search_status": source.get("search_status"),
        "candidate_family_count": 3,
        "candidate_set_generation": "bounded deterministic families, not Bayesian MIPROv2",
        "eligible_for_confirmation": False,
        "selection_gate_access": False,
        "final_holdout_access": False,
        "published": False,
        "retrieval_used": False,
        "excluded_task_ids": exclusions,
        "exclusion_policy": "post hoc provider-timeout exclusions applied symmetrically; exploratory only",
        "release_id": state.release_id,
        "config_sha256": config_fingerprint(cfg),
        "execution_labels": seeds,
        "model_rng_seed_forwarded": False,
        "visible_task_count": len(expected_ids),
        "train_ids": sorted(str(row["task_id"]) for row in train),
        "internal_validation_ids": sorted(str(row["task_id"]) for row in val),
        "discovery_ids": sorted(str(row["task_id"]) for row in discovery),
        "confirmation_ids": sorted(str(row["task_id"]) for row in confirmation),
        "planned_discovery_calls": planned_discovery,
        "planned_confirmation_calls": planned_confirmation,
        "actual_metric_calls": search.get("actual_metric_calls"),
        "search_report_path": str(search_path),
        "search_status": search.get("status"),
        "candidate_results": search.get("candidate_results"),
        "candidate_confirmation": search.get("internal_confirmation"),
        "selected_candidate": candidate.to_dict() if candidate else None,
        "selected_static_gate": search.get("selected_static_gate"),
    }
    out_path = artifacts_path("probe", "pubmedqa_threeway_v4",
                              "expansion-" + uuid.uuid4().hex[:12] + ".json")
    write_json(out_path, report)
    print(json.dumps({
        "path": str(out_path), "search_report": str(search_path),
        "status": report["search_status"],
        "candidate_count": len(report["candidate_results"] or []),
        "actual_metric_calls": report["actual_metric_calls"],
        "candidate_confirmation": report["candidate_confirmation"],
        "selected_demo_ids": search.get("selected_demo_ids"),
        "static_gate_passed": bool((report["selected_static_gate"] or {}).get("passed")),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
