"""Train-only answer comparison between two frozen PubMed retrieval protocols."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_pubmed_snapshot import shingles
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


def main() -> None:
    raise SystemExit(
        "unplanned historical protocol-pair scoring is retired: direct train "
        "replay has no frozen execution-plan hash, request caps, or shared call ledger"
    )

    # Retained as historical implementation only pending governed replay.
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-snapshot", type=Path, required=True)
    parser.add_argument("--candidate-snapshot", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--baseline-source-query", action="append", default=[])
    parser.add_argument("--candidate-source-query", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshots = {"baseline": json.loads(args.baseline_snapshot.read_text(encoding="utf-8")),
                 "candidate": json.loads(args.candidate_snapshot.read_text(encoding="utf-8"))}
    manifests = {"baseline": json.loads(args.baseline_manifest.read_text(encoding="utf-8")),
                 "candidate": json.loads(args.candidate_manifest.read_text(encoding="utf-8"))}
    ids = [row["task_id"] for row in snapshots["candidate"]["tasks"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("candidate snapshot has no unique tasks")
    task_rows = {arm: {row["task_id"]: row for row in snapshot["tasks"]}
                 for arm, snapshot in snapshots.items()}
    queries = {}
    for arm in snapshots:
        if manifests[arm]["snapshot_content_sha256"] != snapshots[arm][
                "content_sha256"]:
            raise ValueError("manifest and snapshot differ")
        queries[arm] = {row["task_id"]: row["query"]
                        for row in manifests[arm]["per_task"]["baseline"]}
    for task_id in ids:
        if (task_id not in task_rows["baseline"] or
                task_rows["baseline"][task_id]["question_sha256"] !=
                task_rows["candidate"][task_id]["question_sha256"] or
                queries["baseline"].get(task_id) != queries["candidate"].get(task_id)):
            raise ValueError("task question or frozen model query differs")
    tasks = {task["task_id"]: task for task in load_tasks(
        "train", task_ids=set(ids))}
    state = load_state("baseline-v1-cited")
    configs = {}
    gateways = {}
    for arm in snapshots:
        cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
        cfg["retrieval"]["query_control"] = {
            "enabled": True, "mode": "frozen",
            "queries": {i: queries[arm][i] for i in ids}}
        cfg["retrieval"]["anchor_question_terms"] = True
        cfg["retrieval"]["pubmed_passage_mode"] = "query_window"
        cfg["retrieval"]["snapshot_content_sha256"] = snapshots[arm][
            "content_sha256"]
        source_names = (args.baseline_source_query if arm == "baseline"
                        else args.candidate_source_query)
        if source_names:
            cfg["retrieval"]["pubmed_source_queries"] = source_names
        cfg["skills"]["pinned_skill"] = "litqa-literature-synthesis"
        cfg["evaluation"]["semantic_citation"] = {"enabled": False}
        configs[arm] = cfg
        gateways[arm] = PubMedSnapshotGateway(snapshots[arm], top_k=3,
                                              passage_mode="query_window")
    provider = build_provider(configs["baseline"])
    provider.cache = False
    pilot_id = sha256_obj({"snapshots": {arm: x["content_sha256"]
                                         for arm, x in snapshots.items()},
                           "manifests": {arm: x["manifest_id"]
                                         for arm, x in manifests.items()},
                           "fingerprints": {arm: protocol_fingerprint(x)
                                            for arm, x in configs.items()},
                           "ids": ids, "version": "pubmed-protocol-pair-v1"})
    journal = args.output.with_name(args.output.stem + ".partial.jsonl")
    saved = load_checkpoint(journal, pilot_id)
    outcomes = {arm: [] for arm in snapshots}
    for task_id in ids:
        for arm in snapshots:
            key = (task_id, arm)
            if key in saved:
                outcome = saved[key]
            else:
                outcome = TaskRunner(configs[arm], state, provider=provider,
                                     gateway=gateways[arm]).run(tasks[task_id])
                append_checkpoint(journal, pilot_id, task_id, arm, outcome)
            outcomes[arm].append(outcome)
    metrics = {arm: aggregate(rows) for arm, rows in outcomes.items()}
    proxy = {}
    for arm in snapshots:
        count_corpus = count_top3 = 0
        for task_id, outcome in zip(ids, outcomes[arm]):
            task = tasks[task_id]
            gold_text = set().union(*(shingles(x["text"])
                                      for x in task["candidate_docs"]
                                      if x["is_gold"]))
            source_names = configs[arm]["retrieval"].get("pubmed_source_queries")
            corpus = (gateways[arm].corpus_for_query_names(task, source_names)
                      if source_names else gateways[arm].corpus(task))
            matched = {d["doc_id"] for d in corpus
                       if gold_text & shingles(d["text"])}
            count_corpus += bool(matched)
            count_top3 += bool(matched & set(outcome.retrieved_doc_ids))
        proxy[arm] = {"source_8gram_in_corpus": count_corpus,
                      "source_8gram_in_top3": count_top3}
    report = {"status": "train_only_pubmed_protocol_pair_probe",
              "pilot_id": pilot_id, "task_ids": ids,
              "snapshot_content_sha256": {arm: x["content_sha256"]
                                          for arm, x in snapshots.items()},
              "query_manifest_id": {arm: x["manifest_id"]
                                    for arm, x in manifests.items()},
              "source_query_names": {arm: configs[arm]["retrieval"].get(
                  "pubmed_source_queries", []) for arm in snapshots},
              "metrics": metrics, "source_proxy": proxy,
              "query_changed": sum(a.retrieval_query != b.retrieval_query
                                   for a, b in zip(outcomes["baseline"],
                                                   outcomes["candidate"])),
              "top3_changed": sum(a.retrieved_doc_ids != b.retrieved_doc_ids
                                  for a, b in zip(outcomes["baseline"],
                                                  outcomes["candidate"])),
              "per_task": {arm: [compact(x) for x in rows]
                           for arm, rows in outcomes.items()},
              "limit": "Retrieval protocol comparison on train only; both arms "
                       "use the same frozen model query but protocol-specific "
                       "question anchor. This is not a Prompt/Skill candidate "
                       "or dev confirmation. Source 8-gram is only a proxy."}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({"metrics": {arm: {key: val[key] for key in (
        "accuracy", "coverage", "answered_precision")}
        for arm, val in metrics.items()},
        "source_proxy": proxy, "query_changed": report["query_changed"],
        "top3_changed": report["top3_changed"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
