"""Paired BioASQ train replay of the same frozen ranking at two evidence depths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pilot_pubmed_window_grounding import (append_checkpoint, compact,
                                                    load_checkpoint)
from scripts.rsi import load_config
from src.data.bioasq_unitor import load_tasks
from src.eval.final_eval import protocol_fingerprint
from src.eval.metrics import aggregate
from src.governance.release import load_state
from src.retrieval.pubmed_snapshot import PubMedSnapshotGateway
from src.runtime.provider import build_provider
from src.runtime.runner import TaskRunner
from src.util import sha256_obj


def compact_depth(out):
    row = compact(out)
    row["retrieved_ids"] = row.pop("top3")
    row["retrieval_k"] = len(row["retrieved_ids"])
    return row


def main() -> None:
    raise SystemExit(
        "unplanned historical top-k scoring is retired: direct train replay "
        "has no frozen execution-plan hash, request caps, or shared call ledger"
    )

    # Retained as historical implementation only pending governed replay.
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--baseline-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.baseline_k < args.candidate_k <= 12:
        parser.error("evidence depths must satisfy 1 <= baseline < candidate <= 12")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    manifest = json.loads(args.query_manifest.read_text(encoding="utf-8"))
    if manifest.get("snapshot_content_sha256") != snapshot["content_sha256"]:
        raise ValueError("frozen query manifest uses another snapshot")
    queries = {row["task_id"]: row["query"]
               for row in manifest["per_task"]["baseline"]}
    ids = [row["task_id"] for row in snapshot["tasks"]]
    if set(ids) != set(queries) or len(ids) != len(queries):
        raise ValueError("snapshot and frozen query IDs differ")
    if args.task_id:
        if not set(args.task_id) <= set(ids):
            parser.error("selected task absent from snapshot")
        ids = [task_id for task_id in ids if task_id in set(args.task_id)]
    tasks = {task["task_id"]: task for task in load_tasks(
        "train", task_ids=set(ids))}
    if set(tasks) != set(ids):
        raise ValueError("non-train task in snapshot")
    state = load_state("baseline-v1-cited")
    configs = {}
    gateways = {}
    for arm, depth in (("baseline", args.baseline_k),
                       ("deeper", args.candidate_k)):
        cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
        cfg["retrieval"]["query_control"] = {
            "enabled": True, "mode": "frozen",
            "queries": {task_id: queries[task_id] for task_id in ids}}
        cfg["retrieval"]["top_k"] = depth
        cfg["retrieval"]["anchor_question_terms"] = True
        cfg["retrieval"]["pubmed_passage_mode"] = "query_window"
        cfg["retrieval"]["snapshot_content_sha256"] = snapshot["content_sha256"]
        cfg["skills"]["pinned_skill"] = "litqa-literature-synthesis"
        cfg["evaluation"]["semantic_citation"] = {"enabled": False}
        configs[arm] = cfg
        gateways[arm] = PubMedSnapshotGateway(
            snapshot, top_k=depth, passage_mode="query_window")
    provider = build_provider(configs["baseline"])
    provider.cache = False
    pilot_id = sha256_obj({"snapshot": snapshot["content_sha256"],
                           "manifest": manifest["manifest_id"],
                           "ids": ids, "configs": configs,
                           "protocols": {arm: protocol_fingerprint(cfg)
                                         for arm, cfg in configs.items()},
                           "code": "pubmed-depth-pair-v1"})
    journal = args.output.with_name(args.output.stem + ".partial.jsonl")
    saved = load_checkpoint(journal, pilot_id)
    outcomes = {arm: [] for arm in gateways}
    for task_id in ids:
        for arm, gateway in gateways.items():
            key = (task_id, arm)
            if key in saved:
                outcome = saved[key]
            else:
                outcome = TaskRunner(configs[arm], state, provider=provider,
                                     gateway=gateway).run(tasks[task_id])
                append_checkpoint(journal, pilot_id, task_id, arm, outcome)
            outcomes[arm].append(outcome)
    metrics = {arm: aggregate(rows) for arm, rows in outcomes.items()}
    report = {"status": "train_only_pubmed_evidence_depth_probe",
              "pilot_id": pilot_id,
              "snapshot_content_sha256": snapshot["content_sha256"],
              "query_manifest_id": manifest["manifest_id"],
              "task_ids": ids,
              "depths": {"baseline": args.baseline_k,
                         "deeper": args.candidate_k},
              "metrics": metrics,
              "ranking_prefix_preserved": all(
                  deeper.retrieved_doc_ids[:args.baseline_k] ==
                  base.retrieved_doc_ids for base, deeper in zip(
                      outcomes["baseline"], outcomes["deeper"])),
              "per_task": {arm: [compact_depth(row) for row in rows]
                           for arm, rows in outcomes.items()},
              "limit": "Same frozen PubMed snapshot and query; only evidence "
                       "depth changes. Train-only, no calibrated grounding or "
                       "confirmation claim."}
    if not report["ranking_prefix_preserved"]:
        raise ValueError("deeper evidence changed BM25 prefix unexpectedly")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({"metrics": {arm: {key: value[key] for key in (
        "accuracy", "coverage", "answered_precision")}
        for arm, value in metrics.items()},
        "ranking_prefix_preserved": report["ranking_prefix_preserved"]},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
