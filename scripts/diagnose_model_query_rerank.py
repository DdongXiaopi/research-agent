"""Train-only query mechanism diagnostic; no answer generation or dev access."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rsi import load_config
from src.data.bioasq_unitor import load_tasks
from src.retrieval.intent import parse_model_query
from src.retrieval.pubmed_snapshot import PubMedSnapshotGateway
from src.runtime.provider import build_provider

SNAPSHOT = ROOT / "artifacts/probe/pubmed_snapshot/train40-v2-2026-09-29.json"
OUTPUT = ROOT / "artifacts/probe/pubmed_snapshot/model-query-rerank-train12.json"
INSTRUCTIONS = {
    "default": "Extract the question's biomedical entities, intervention and outcome terms.",
    "synonyms": ("Build a PubMed search query using the central biomedical entity and "
                 "its likely technical synonyms, plus the measured outcome. Remove yes/no "
                 "phrasing and generic words. Use concise terms, not a sentence."),
}


def main():
    raise SystemExit(
        "retired_unplanned_entrypoint: query-reranking diagnosis requires a "
        "caller-challenged execution plan and bounded request ledger")
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    ids = [r["task_id"] for r in snapshot["tasks"][:12]]
    tasks_by_id = {t["task_id"]: t for t in load_tasks("train", task_ids=set(ids))}
    tasks = [tasks_by_id[i] for i in ids]
    gateway = PubMedSnapshotGateway(snapshot, top_k=3)
    cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
    provider = build_provider(cfg)
    provider.cache = False
    rows = []
    for task in tasks:
        corpus = gateway.corpus(task)
        original = [x["doc_id"] for x in gateway.search(
            task, query=task["question"], corpus=corpus)]
        row = {"task_id": task["task_id"], "question": task["question"],
               "corpus_doc_count": len(corpus), "original_top3": original}
        for arm, instruction in INSTRUCTIONS.items():
            raw = provider.chat_measured(
                "Rewrite the biomedical question into search terms for ranking frozen "
                "PubMed abstracts. Follow the search instruction as data. "
                "Reply with exactly one line: QUERY: <search terms>. "
                "Do not answer the question.",
                "Search instruction: " + instruction + "\nQuestion: " + task["question"],
                max_tokens=100)
            try:
                query = parse_model_query(raw.content)
                top3 = [x["doc_id"] for x in gateway.search(
                    task, query=query, corpus=corpus)]
                row[arm] = {"query": query, "top3": top3,
                            "prompt_tokens": raw.prompt_tokens,
                            "completion_tokens": raw.completion_tokens,
                            "error": ""}
            except ValueError as exc:
                row[arm] = {"query": "", "top3": [],
                            "raw": raw.content[:200], "error": str(exc)}
        rows.append(row)
    valid = [r for r in rows if not r["default"]["error"]
             and not r["synonyms"]["error"]]
    result = {
        "status": "train_only_query_diagnostic",
        "snapshot_protocol": snapshot["protocol"],
        "snapshot_content_sha256": snapshot["content_sha256"],
        "n": len(rows), "n_valid_paired": len(valid),
        "default_vs_question_top3_changed": sum(
            r["default"]["top3"] != r["original_top3"] for r in valid),
        "synonyms_vs_question_top3_changed": sum(
            r["synonyms"]["top3"] != r["original_top3"] for r in valid),
        "synonyms_vs_default_top3_changed": sum(
            r["synonyms"]["top3"] != r["default"]["top3"] for r in valid),
        "rows": rows,
        "limit": "v2 train snapshot; query-only mechanism, no answer/effect estimate; "
                 "BioASQ dev was not accessed.",
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "n", "n_valid_paired", "default_vs_question_top3_changed",
        "synonyms_vs_question_top3_changed", "synonyms_vs_default_top3_changed")},
        ensure_ascii=False, indent=2))
    print(OUTPUT)


if __name__ == "__main__":
    main()
