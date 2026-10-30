"""Frozen-query BioASQ train retrieval probe: BM25 plus character TF-IDF RRF."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_pubmed_snapshot import shingles
from src.data.bioasq_unitor import load_tasks
from src.retrieval.pubmed_snapshot import PubMedSnapshotGateway
from src.util import sha256_obj


def hybrid_rank(gateway: PubMedSnapshotGateway, task: dict,
                query: str) -> tuple[list[str], list[str]]:
    """Ranks only frozen article text; no gold or supplied passages."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = gateway.corpus(task)
    all_bm25 = PubMedSnapshotGateway(
        {"protocol": gateway.snapshot_protocol,
         "tasks": [gateway.rows[task["task_id"]]],
         "content_sha256": sha256_obj({
             "protocol": gateway.snapshot_protocol,
             "tasks": [gateway.rows[task["task_id"]]]})},
        top_k=len(docs)).search(task, query=query, corpus=docs)
    bm25 = [doc["doc_id"] for doc in all_bm25]
    if len(docs) < 2:
        return bm25[:3], bm25[:3]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=2, sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform([doc["text"] for doc in docs])
        query_vector = vectorizer.transform([query])
    except ValueError:
        return bm25[:3], bm25[:3]
    scores = (query_vector @ matrix.T).toarray().ravel()
    char_rank = [doc["doc_id"] for _, doc in sorted(
        zip(scores, docs), key=lambda pair: (-pair[0], pair[1]["doc_id"]))]
    fused = defaultdict(float)
    for ranked in (bm25, char_rank):
        for rank, doc_id in enumerate(ranked):
            fused[doc_id] += 1 / (60 + rank + 1)
    hybrid = sorted(fused, key=lambda doc_id: (-fused[doc_id], doc_id))[:3]
    return bm25[:3], hybrid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--query-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    manifest = json.loads(args.query_report.read_text(encoding="utf-8"))
    if manifest.get("snapshot_content_sha256") != snapshot["content_sha256"]:
        raise ValueError("frozen query manifest uses another snapshot")
    queries = {row["task_id"]: row["query"] for row in
               manifest["per_task"]["baseline"]}
    ids = [row["task_id"] for row in snapshot["tasks"]]
    if set(ids) != set(queries):
        raise ValueError("frozen query task IDs differ from snapshot")
    tasks = {task["task_id"]: task for task in load_tasks(
        "train", task_ids=set(ids))}
    gateway = PubMedSnapshotGateway(snapshot, top_k=3)
    rows = []
    for task_id in ids:
        task = tasks[task_id]
        public_task = {"task_id": task_id, "question": task["question"]}
        query = queries[task_id] + " " + gateway.anchor_query(
            task["question"], max_terms=20)
        baseline, hybrid = hybrid_rank(gateway, public_task, query)
        source = set().union(*(shingles(doc["text"]) for doc in
                               task["candidate_docs"] if doc["is_gold"]))
        matched = {doc["doc_id"] for doc in gateway.corpus(task)
                   if source & shingles(doc["text"])}
        rows.append({"task_id": task_id, "query": query,
                     "bm25_top3": baseline, "rerank_top3": hybrid,
                     "source_bm25": bool(matched & set(baseline)),
                     "source_rerank": bool(matched & set(hybrid))})
    summary = {"n": len(rows),
               "top3_changed": sum(r["bm25_top3"] != r["rerank_top3"]
                                   for r in rows),
               "source_bm25_top3": sum(r["source_bm25"] for r in rows),
               "source_rerank_top3": sum(r["source_rerank"] for r in rows),
               "source_gained": sum(not r["source_bm25"] and r["source_rerank"]
                                    for r in rows),
               "source_lost": sum(r["source_bm25"] and not r["source_rerank"]
                                  for r in rows)}
    probe_id = sha256_obj({"snapshot": snapshot["content_sha256"],
                           "manifest": manifest["manifest_id"],
                           "ranker": "bm25-char-tfidf-rrf-v1", "ids": ids})
    report = {"status": "train_only_frozen_corpus_rerank_probe",
              "snapshot_content_sha256": snapshot["content_sha256"],
              "query_manifest_id": manifest["manifest_id"],
              "probe_id": probe_id,
              "model_id": "local-bm25-char-tfidf-rrf-v1",
              "summary": summary, "rows": rows,
              "limit": "Exact source 8-gram is a low-recall proxy, used only "
                       "after ranking. This is train-only, not answer gain."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
