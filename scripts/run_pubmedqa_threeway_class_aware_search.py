"""Run an unpublished real-Pi class-aware task-demo search on PubMedQA dev.

This is an engineering/exploration run only: the threeway dataset has no
selection gate or final holdout and is never eligible for confirmation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_fingerprint, load_config  # noqa: E402
from src.data.pubmedqa_threeway_probe import (  # noqa: E402
    DATASET, MANIFEST_PATH, NORMALIZED_PATH,
)
from src.evolution.gepa_search import visible_examples  # noqa: E402
from src.evolution.task_demo_search import (  # noqa: E402
    SEARCH_PROTOCOL_VERSION, _metrics, search_task_demo_candidate,
    split_demo_validation,
)
from src.eval.task_rows import validate_task_answer_row  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.provider import build_provider  # noqa: E402
from src.runtime.runner import TaskRunner  # noqa: E402
from src.runtime.pi_trace import read_pi_session  # noqa: E402
from src.runtime.provider import HttpProvider  # noqa: E402
from src.util import (  # noqa: E402
    artifacts_path, ensure_dir, read_json, sha256_file, write_json, write_jsonl,
)
from src.eval.metrics import total_cost  # noqa: E402


def _configure(config_path: str) -> dict:
    cfg = copy.deepcopy(load_config(config_path))
    cfg["runtime"].update({"provider": "pi", "input_mode": "task_input",
                           "task_input_max_chars": 30000,
                           "pi_request_cooldown_seconds": 6})
    cfg["runtime"].pop("pi_fake_response", None)
    cfg["evaluation"].update({"insufficient_option": False,
                              "main_metric": "accuracy",
                              "primary_axis": "accuracy",
                              "sealed_pairing": False})
    cfg["evolution"]["allowed_artifacts"] = ["task_demos"]
    cfg["evolution"].setdefault("fewshot", {}).update({
        "enabled": True, "k": 3, "max_candidate_sets": 1,
    })
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg["tools"].get("active", []) if name != "academic.search"]
    return cfg


def _checkpoint_path() -> Path:
    return artifacts_path("probe", "pubmedqa_threeway_v4",
                          "class-aware-search-checkpoint.jsonl")


def _append_checkpoint(row: dict) -> None:
    path = _checkpoint_path()
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validated_baseline_row(row: dict, task: dict, *, seed: int,
                            config_sha256: str) -> dict:
    """Validate fresh and resumed baseline rows before they affect search."""
    checked = validate_task_answer_row(
        row, task, context="PubMedQA threeway class-aware baseline",
        require_reported_gold_index=True)
    return {**checked, "seed": int(seed), "config_sha256": config_sha256}


def _recover_completed_sessions(task_by_id: dict, state, expected_ids: list[str]) -> list[dict]:
    """Recover completed first-seed baseline answers after an interrupted run."""
    sessions_root = artifacts_path("pi_sessions")
    recovered = []
    for path in sorted(sessions_root.rglob("*.jsonl")) if sessions_root.exists() else []:
        try:
            session = read_pi_session(path)
        except Exception:
            continue
        manifest = session["task_manifest"].get("data") or {}
        task_id = str(manifest.get("taskId") or "")
        if (task_id not in task_by_id or manifest.get("taskSplit") != "dev_optimize"
                or manifest.get("releaseId") != state.release_id):
            continue
        result = session["results"][-1]
        if result.get("status") != "completed":
            continue
        assistant = next((message for message in reversed(session["messages"])
                          if message.get("role") == "assistant"), None)
        if assistant is None:
            continue
        task = task_by_id[task_id]
        answer_index, _cited, parse_failed = HttpProvider._parse(
            None, assistant.get("text", ""), task["options"], [])
        if parse_failed:
            continue
        usage = assistant.get("usage") or {}
        prompt_tokens = (int(usage.get("input") or 0)
                         + int(usage.get("cacheRead") or 0)
                         + int(usage.get("cacheWrite") or 0))
        completion_tokens = int(usage.get("output") or 0)
        recovered.append({
            "task_id": task_id,
            "gold_label": str(task["options"][task["correct_index"]]).casefold(),
            "answer_index": answer_index,
            "gold_index": int(task["correct_index"]),
            "correct": answer_index >= 0 and answer_index == int(task["correct_index"]),
            "status": "ok",
            "dataset": task["dataset"],
            "split": task["split"],
            "input_mode": "task_input",
            "cost": {"usd": None, "usd_nominal": None,
                     "cost_estimated": True,
                     "cost_basis": "recovered from completed Pi JSONL usage",
                     "usage_available": bool(assistant.get("usage")),
                     "prompt_tokens": prompt_tokens,
                     "completion_tokens": completion_tokens},
            "pi_session_path": str(path),
            "recovered_from_session": True,
        })
    recovered.sort(key=lambda row: expected_ids.index(row["task_id"]))
    if [row["task_id"] for row in recovered] != expected_ids[:len(recovered)]:
        raise RuntimeError(
            "completed Pi sessions are not a contiguous prefix of the first baseline seed; "
            "refusing to infer execution labels")
    return recovered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--limit", type=int, default=90)
    parser.add_argument("--timeout-excluded-task-id", nargs="+", default=[],
                        help="record dev task IDs excluded after repeated provider timeout")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[202610051, 202610052, 202610053])
    args = parser.parse_args()
    raise SystemExit(
        "This exploratory PubMedQA class-aware search is disabled until it has a "
        "frozen execution-plan hash and enforced logical/physical request ledger. "
        "Its candidate static gate is not an experiment-budget approval. No config, "
        "task rows, release state, provider, or TaskRunner will be loaded.")
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("class-aware PubMedQA search requires an explicit sandbox RSI_HOME")
    if args.limit < 9:
        raise ValueError("--limit must provide at least nine visible dev tasks")
    if len(args.seeds) < 3 or len(set(args.seeds)) < 3:
        raise ValueError("class-aware search requires at least three distinct execution labels")

    cfg = _configure(args.config)
    if args.timeout_excluded_task_id:
        cfg["evolution"]["fewshot"]["excluded_task_ids"] = list(
            args.timeout_excluded_task_id)
    state = load_state()
    train, val = visible_examples(cfg, [DATASET], args.limit)
    visible_ids = {str(row["task_id"]) for row in [*train, *val]}
    missing_exclusions = set(args.timeout_excluded_task_id) - visible_ids
    if missing_exclusions:
        raise RuntimeError("timeout-excluded IDs must belong to the prefiltered cohort: "
                           + ", ".join(sorted(missing_exclusions)))
    excluded_ids = set(args.timeout_excluded_task_id)
    train = [row for row in train if str(row["task_id"]) not in excluded_ids]
    val = [row for row in val if str(row["task_id"]) not in excluded_ids]
    discovery, confirmation = split_demo_validation(val)
    labels = sorted({
        str(row["options"][row["correct_index"]]).strip().casefold()
        for row in [*train, *val]
    })
    task_by_id = {str(row["task_id"]): row for row in [*train, *val]}
    all_tasks = [task_by_id[key] for key in sorted(task_by_id)]
    checkpoint_path = _checkpoint_path()
    checkpoint_rows = []
    if checkpoint_path.exists():
        checkpoint_rows = [json.loads(line) for line in
                           checkpoint_path.read_text(encoding="utf-8").splitlines()
                           if line.strip()]
    if not checkpoint_rows:
        recovered = _recover_completed_sessions(
            task_by_id, state, [str(task["task_id"]) for task in all_tasks])
        for offset, row in enumerate(recovered):
            row["seed"] = int(args.seeds[0])
            row["config_sha256"] = config_fingerprint(cfg)
            _append_checkpoint(row)
            checkpoint_rows.append(row)
    run_ids = {}
    all_costs = []
    baseline_reports = []
    failures = set()

    for seed in args.seeds:
        outcomes = []
        existing = {(int(row["seed"]), str(row["task_id"])): row
                    for row in checkpoint_rows}
        provider = build_provider(cfg, seed=seed)
        for task in all_tasks:
            key = (int(seed), str(task["task_id"]))
            saved = existing.get(key)
            if saved is not None and saved.get("status") == "ok":
                row = _validated_baseline_row(
                    saved, task, seed=seed,
                    config_sha256=config_fingerprint(cfg))
            else:
                # The runtime call budget is per TaskRunner. Keep the same
                # one-task scope as the production round instead of reusing
                # a runner across the cohort.
                runner = TaskRunner(cfg, state, provider=provider, seed=seed)
                outcome = runner.run(task)
                row = _validated_baseline_row(
                    outcome.to_dict(), task, seed=seed,
                    config_sha256=config_fingerprint(cfg))
                _append_checkpoint(row)
                checkpoint_rows.append(row)
                existing[key] = row
            outcomes.append(row)
            if not row.get("correct"):
                failures.add(str(task["task_id"]))
        run_id = f"pqa-threeway-v4-{seed}-{uuid.uuid4().hex[:8]}"
        run_dir = artifacts_path("runs", run_id)
        ensure_dir(run_dir)
        run_rows = [{key: value for key, value in row.items()
                     if key not in {"seed", "config_sha256", "recovered_from_session"}}
                    for row in outcomes]
        write_jsonl(run_dir / "per_task.jsonl", run_rows)
        run_ids[str(seed)] = run_id
        all_costs.extend(SimpleNamespace(cost=row.get("cost", {})) for row in outcomes)
        baseline_rows = [{"task_id": row["task_id"],
                          "gold_label": row["gold_label"],
                          "answer_index": row["answer_index"],
                          "correct": row["correct"]}
                         for row in outcomes]
        baseline_reports.append({"seed": seed, "run_id": run_id,
                                 "metrics": _metrics(baseline_rows),
                                 "task_count": len(outcomes)})

    discovery_calls = len(discovery) * len(args.seeds)
    confirmation_calls = len(confirmation) * len(args.seeds)
    cfg["evolution"]["fewshot"]["max_metric_calls"] = (
        discovery_calls + confirmation_calls)
    search_dir_before = artifacts_path("search", "task_demos")
    existing_reports = set(search_dir_before.rglob("report.json")) if search_dir_before.exists() else set()
    candidate = search_task_demo_candidate(
        cfg, state, [DATASET], args.limit, args.seeds[0], sorted(failures),
        {DATASET: run_ids})
    search_reports = sorted(
        (set(search_dir_before.rglob("report.json")) if search_dir_before.exists() else set())
        - existing_reports,
        key=lambda path: path.stat().st_mtime_ns)
    if len(search_reports) != 1:
        raise RuntimeError(f"expected one search report, found {len(search_reports)}")
    search_report_path = search_reports[0]
    search_report = read_json(search_report_path)

    report = {
        "protocol": "pubmedqa-threeway-class-aware-search-v1-exploratory",
        "search_protocol_version": SEARCH_PROTOCOL_VERSION,
        "dataset": DATASET,
        "dataset_manifest_sha256": sha256_file(MANIFEST_PATH),
        "dataset_normalized_sha256": sha256_file(NORMALIZED_PATH),
        "eligible_for_confirmation": False,
        "selection_gate_access": False,
        "final_holdout_access": False,
        "published": False,
        "retrieval_used": False,
        "synthetic_insufficient_option": False,
        "release_id": state.release_id,
        "config_sha256": config_fingerprint(cfg),
        "execution_labels": list(args.seeds),
        "model_rng_seed_forwarded": False,
        "pi_request_cooldown_seconds": cfg["runtime"]["pi_request_cooldown_seconds"],
        "checkpoint_entry_count": len(checkpoint_rows),
        "checkpoint_unique_seed_task_count": len({
            (int(row["seed"]), str(row["task_id"])) for row in checkpoint_rows}),
        "superseded_checkpoint_error_count": sum(
            row.get("status") != "ok" for row in checkpoint_rows),
        "visible_task_count": len(all_tasks),
        "run_adjustment": ({
            "planned_limit": 90,
            "effective_limit": args.limit,
            "included_task_count_after_exclusions": len(all_tasks),
            "reason": "repeated Pi bridge timeouts; post hoc infrastructure exclusions are applied symmetrically to baseline and candidate search",
            "provider_timeout_excluded_task_ids": list(args.timeout_excluded_task_id),
        } if args.timeout_excluded_task_id else None),
        "train_ids": sorted(str(row["task_id"]) for row in train),
        "internal_validation_ids": sorted(str(row["task_id"]) for row in val),
        "discovery_ids": sorted(str(row["task_id"]) for row in discovery),
        "confirmation_ids": sorted(str(row["task_id"]) for row in confirmation),
        "label_counts": {
            label: sum(str(row["options"][row["correct_index"]]).casefold() == label
                       for row in all_tasks)
            for label in labels
        },
        "baseline_runs": baseline_reports,
        "baseline_total_cost": total_cost(all_costs),
        "planned_candidate_calls": discovery_calls + confirmation_calls,
        "search_report_path": str(search_report_path),
        "search_status": search_report.get("status"),
        "selected_candidate": candidate.to_dict() if candidate else None,
        "static_gate": (candidate.generator.get("static_gate")
                        if candidate else None),
    }
    out_path = artifacts_path("probe", "pubmedqa_threeway_v4",
                              "run-" + uuid.uuid4().hex[:12] + ".json")
    write_json(out_path, report)
    print(json.dumps({
        "path": str(out_path), "search_report": str(search_report_path),
        "search_status": report["search_status"],
        "baseline_metrics": baseline_reports,
        "candidate_confirmation": search_report.get("internal_confirmation"),
        "candidate_demo_ids": search_report.get("selected_demo_ids"),
        "static_gate_passed": bool(search_report.get("selected_static_gate", {}).get("passed")),
        "baseline_tokens": {"prompt": report["baseline_total_cost"]["prompt_tokens"],
                            "completion": report["baseline_total_cost"]["completion_tokens"]},
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
