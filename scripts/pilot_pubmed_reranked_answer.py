"""Train-only paired answer test for a frozen reranker result.

The rerank probe used question and article text only. This replay uses the
same frozen query and corpus in both arms; it never reorders from gold labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pilot_pubmed_window_grounding import append_checkpoint, compact, load_checkpoint
from scripts.rsi import load_config
from src.data.bioasq_unitor import load_tasks
from src.eval.final_eval import protocol_fingerprint
from src.eval.metrics import aggregate
from src.governance.release import load_state
from src.retrieval.pubmed_snapshot import PubMedSnapshotGateway
from src.runtime.provider import build_provider
from src.runtime.runner import TaskRunner
from src.util import sha256_obj


class FrozenRerankGateway(PubMedSnapshotGateway):
    """Replay API ranks, then apply the ordinary question-dependent window."""

    def __init__(self, snapshot: dict, rows: dict[str, dict]):
        super().__init__(snapshot, top_k=3, passage_mode="query_window")
        self.rank_rows = rows

    def search(self, task, skill_recalled=False, rerank=False, query=None,
               corpus=None):
        source = corpus if corpus is not None else self.corpus(task)
        by_id = {doc["doc_id"]: doc for doc in source}
        row = self.rank_rows[task["task_id"]]
        if query != row["query"]:
            raise ValueError("rerank replay query differs from frozen query")
        ids = row["rerank_top3"]
        if len(ids) != len(set(ids)) or any(doc_id not in by_id for doc_id in ids):
            raise ValueError("rerank report contains invalid document IDs")
        # Parent applies exactly the normal passage selection to these texts.
        windowed = super().search(task, skill_recalled=skill_recalled,
                                  rerank=rerank, query=query,
                                  corpus=[by_id[doc_id] for doc_id in ids])
        selected = {doc["doc_id"]: doc for doc in windowed}
        return [dict(selected[doc_id], rank=index)
                for index, doc_id in enumerate(ids)]


def checked_inputs(snapshot: dict, manifest: dict, rerank: dict) -> tuple[list[str], dict, dict]:
    checksum = snapshot["content_sha256"]
    if manifest.get("snapshot_content_sha256") != checksum or rerank.get(
            "snapshot_content_sha256") != checksum:
        raise ValueError("snapshot, frozen queries and rerank report differ")
    if rerank.get("status") != "train_only_frozen_corpus_rerank_probe":
        raise ValueError("not a train-only rerank report")
    ids = [row["task_id"] for row in snapshot["tasks"]]
    queries = {row["task_id"]: row["query"] for row in manifest[
        "per_task"]["baseline"]}
    rows = {row["task_id"]: row for row in rerank["rows"]}
    if set(ids) != set(queries) or set(ids) != set(rows):
        raise ValueError("task IDs differ across inputs")
    if len(ids) != len(queries) or len(ids) != len(rows):
        raise ValueError("duplicate task IDs in inputs")
    tasks = {task["task_id"]: task for task in load_tasks(
        "train", task_ids=set(ids))}
    if set(tasks) != set(ids):
        raise ValueError("snapshot contains non-train tasks")
    gateway = PubMedSnapshotGateway(snapshot)
    for task_id in ids:
        actual_query = queries[task_id] + " " + gateway.anchor_query(
            tasks[task_id]["question"], max_terms=20)
        if rows[task_id]["query"] != actual_query:
            raise ValueError("rerank report used a different query")
        available = {doc["doc_id"] for doc in gateway.corpus(tasks[task_id])}
        if any(doc_id not in available for doc_id in rows[task_id]["rerank_top3"]):
            raise ValueError("rerank report references a missing article")
    return ids, queries, rows


def main() -> None:
    raise SystemExit(
        "unplanned historical reranker scoring is retired: direct answer "
        "replay has no frozen execution-plan hash, request caps, or shared call ledger"
    )

    # Retained as historical implementation only pending governed replay.
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--rerank-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    manifest = json.loads(args.query_manifest.read_text(encoding="utf-8"))
    rerank = json.loads(args.rerank_report.read_text(encoding="utf-8"))
    ids, queries, rows = checked_inputs(snapshot, manifest, rerank)
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("limit must be positive")
        ids = ids[:args.limit]
    tasks = {task["task_id"]: task for task in load_tasks(
        "train", task_ids=set(ids))}
    cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
    cfg["retrieval"]["query_control"] = {
        "enabled": True, "mode": "frozen", "queries": {i: queries[i] for i in ids}}
    cfg["retrieval"]["anchor_question_terms"] = True
    cfg["retrieval"]["pubmed_passage_mode"] = "query_window"
    cfg["retrieval"]["snapshot_content_sha256"] = snapshot["content_sha256"]
    cfg["skills"]["pinned_skill"] = "litqa-literature-synthesis"
    cfg["evaluation"]["semantic_citation"] = {"enabled": False}
    state = load_state("baseline-v1-cited")
    provider = build_provider(cfg)
    provider.cache = False
    gateways = {
        "bm25": PubMedSnapshotGateway(snapshot, top_k=3,
                                        passage_mode="query_window"),
        "rerank": FrozenRerankGateway(snapshot, rows),
    }
    pilot_id = sha256_obj({"snapshot": snapshot["content_sha256"],
                           "manifest": manifest["manifest_id"],
                           "rerank": rerank["probe_id"], "ids": ids,
                           "cfg": cfg, "protocol": protocol_fingerprint(cfg),
                           "replay_code": "frozen-rerank-answer-v1"})
    journal = args.output.with_name(args.output.stem + ".partial.jsonl")
    saved = load_checkpoint(journal, pilot_id)
    outcomes = {arm: [] for arm in gateways}
    for task_id in ids:
        for arm, gateway in gateways.items():
            key = (task_id, arm)
            if key in saved:
                outcome = saved[key]
            else:
                outcome = TaskRunner(cfg, state, provider=provider,
                                     gateway=gateway).run(tasks[task_id])
                append_checkpoint(journal, pilot_id, task_id, arm, outcome)
            outcomes[arm].append(outcome)
    metrics = {arm: aggregate(values) for arm, values in outcomes.items()}
    report = {
        "status": "train_only_frozen_rerank_answer_probe",
        "pilot_id": pilot_id,
        "snapshot_content_sha256": snapshot["content_sha256"],
        "query_manifest_id": manifest["manifest_id"],
        "rerank_probe_id": rerank["probe_id"],
        "task_ids": ids,
        "metrics": metrics,
        "top3_changed": sum(a.retrieved_doc_ids != b.retrieved_doc_ids
                            for a, b in zip(outcomes["bm25"], outcomes["rerank"])),
        "passage_changed": sum(a.retrieved_passage_hashes != b.retrieved_passage_hashes
                               for a, b in zip(outcomes["bm25"], outcomes["rerank"])),
        "per_task": {arm: [compact(item) for item in values]
                     for arm, values in outcomes.items()},
        "limit": "Train-only retrieval mechanism test, with no claim of "
                 "evolution benefit or calibrated grounding. Dev untouched.",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({"metrics": {arm: {key: val[key] for key in (
        "accuracy", "coverage", "answered_precision")}
        for arm, val in metrics.items()},
        "top3_changed": report["top3_changed"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
