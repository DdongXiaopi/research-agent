"""Train-only retrieval diagnostics; source text is used only after ranking."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.bioasq_unitor import load_tasks
from src.retrieval.pubmed_snapshot import PubMedSnapshotGateway

_WORD = re.compile(r"[a-z0-9]+", re.I)


def shingles(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = [x.lower() for x in _WORD.findall(text)]
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    gateway = PubMedSnapshotGateway(snapshot)
    wanted = {r["task_id"] for r in snapshot["tasks"]}
    train = {t["task_id"]: t for t in load_tasks("train", task_ids=wanted)}
    rows = []
    for entry in snapshot["tasks"]:
        task = train[entry["task_id"]]
        source = set().union(*(shingles(x["text"]) for x in task["candidate_docs"]
                               if x["is_gold"]))
        corpus = gateway.corpus(task)
        matched = {d["doc_id"] for d in corpus if source & shingles(d["text"])}
        baseline = gateway.search(task, query=task["question"], corpus=corpus)
        variant = gateway.search(task,
                                 query=task["question"] + " study results outcomes",
                                 corpus=corpus)
        a = [d["doc_id"] for d in baseline]
        b = [d["doc_id"] for d in variant]
        rows.append({"task_id": task["task_id"], "n_docs": len(corpus),
                     "matched_source_pmids": sorted(matched),
                     "baseline_top3": a, "variant_top3": b,
                     "source_in_baseline_top3": bool(matched & set(a)),
                     "source_in_variant_top3": bool(matched & set(b))})
    report = {
        "status": "train_only_structural_diagnostic",
        "snapshot_content_sha256": snapshot["content_sha256"],
        "n": len(rows), "n_no_docs": sum(x["n_docs"] == 0 for x in rows),
        "n_source_exact_8gram_in_corpus": sum(bool(x["matched_source_pmids"])
                                                for x in rows),
        "n_source_exact_8gram_baseline_top3": sum(x["source_in_baseline_top3"]
                                                   for x in rows),
        "n_source_exact_8gram_variant_top3": sum(x["source_in_variant_top3"]
                                                  for x in rows),
        "n_top3_order_changed": sum(x["baseline_top3"] != x["variant_top3"]
                                    for x in rows),
        "n_top3_membership_changed": sum(set(x["baseline_top3"]) !=
                                         set(x["variant_top3"]) for x in rows),
        "limitation": "Exact 8-token source matching is a high-precision, "
                      "low-recall proxy. Non-matches may still be relevant "
                      "or answer-supporting PubMed abstracts.",
        "per_task": rows,
    }
    output = args.snapshot.with_name(args.snapshot.stem + "-diagnostic.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_task"},
                     ensure_ascii=False, indent=2))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
