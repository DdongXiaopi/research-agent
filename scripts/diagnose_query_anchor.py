"""Label-free question-term anchor diagnostic on explored BioASQ train."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_pubmed_snapshot import shingles
from src.data.bioasq_unitor import load_tasks
from src.retrieval.pubmed_snapshot import PubMedSnapshotGateway


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    gateway = PubMedSnapshotGateway(snapshot, passage_mode="query_window")
    rows = {}
    for path in args.report:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["snapshot_content_sha256"] != snapshot["content_sha256"]:
            raise ValueError("report and snapshot differ")
        for arm in ("baseline", "instruction"):
            for row in report["per_task"][arm]:
                rows.setdefault(row["task_id"], {})[arm] = row
    ids = set(rows)
    tasks = {t["task_id"]: t for t in load_tasks("train", task_ids=ids)}
    per_task = []
    for task_id, pair in rows.items():
        task = tasks[task_id]
        source = set().union(*(shingles(x["text"]) for x in task["candidate_docs"]
                               if x["is_gold"]))
        entry = {"task_id": task_id, "anchor": gateway.anchor_query(
            task["question"], max_terms=20)}
        for arm in ("baseline", "instruction"):
            raw = pair[arm]
            anchored_query = raw["query"] + " " + entry["anchor"]
            anchored = gateway.search(task, query=anchored_query)
            matched = {d["doc_id"] for d in gateway.corpus(task)
                       if source & shingles(d["text"])}
            ids_anchored = [d["doc_id"] for d in anchored]
            entry[arm] = {"raw_query": raw["query"],
                          "raw_top3": raw["top3"],
                          "anchored_top3": ids_anchored,
                          "changed": raw["top3"] != ids_anchored,
                          "source_raw": bool(matched & set(raw["top3"])),
                          "source_anchored": bool(matched & set(ids_anchored))}
        per_task.append(entry)
    report = {"status": "train_only_offline_anchor_diagnostic",
              "snapshot_content_sha256": snapshot["content_sha256"],
              "n": len(per_task),
              "summary": {arm: {
                  "top3_changed": sum(x[arm]["changed"] for x in per_task),
                  "source_raw_top3": sum(x[arm]["source_raw"] for x in per_task),
                  "source_anchored_top3": sum(x[arm]["source_anchored"]
                                              for x in per_task)}
                  for arm in ("baseline", "instruction")},
              "limit": "Exact source 8-gram is high precision and low recall. "
                       "No model answers or dev tasks are used.",
              "per_task": per_task}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
