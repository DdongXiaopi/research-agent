"""Replay question-only model queries across PubMed snapshot protocols.

This changes the frozen article pool but holds query text fixed for a paired
retrieval diagnostic. Question hashes must agree, and no labels are loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.util import sha256_obj


def transfer(source_snapshot: dict, source_manifest: dict,
             target_snapshot: dict) -> dict:
    if source_manifest["snapshot_content_sha256"] != source_snapshot[
            "content_sha256"]:
        raise ValueError("source manifest and snapshot differ")
    source_rows = {r["task_id"]: r for r in source_snapshot["tasks"]}
    queries = {r["task_id"]: r["query"]
               for r in source_manifest["per_task"]["baseline"]}
    if len(queries) != len(source_manifest["per_task"]["baseline"]):
        raise ValueError("duplicate source queries")
    ids = [r["task_id"] for r in target_snapshot["tasks"]]
    if len(ids) != len(set(ids)) or not set(ids) <= set(source_rows) & set(queries):
        raise ValueError("target tasks are absent from source queries")
    for row in target_snapshot["tasks"]:
        if row["question_sha256"] != source_rows[row["task_id"]]["question_sha256"]:
            raise ValueError("question text changed between snapshots")
    manifest_id = sha256_obj({"source_manifest": source_manifest["manifest_id"],
                              "target_snapshot": target_snapshot["content_sha256"],
                              "queries": [(i, queries[i]) for i in ids]})
    return {"status": "train_only_transferred_frozen_query_manifest",
            "snapshot_content_sha256": target_snapshot["content_sha256"],
            "source_snapshot_content_sha256": source_snapshot["content_sha256"],
            "source_manifest_id": source_manifest["manifest_id"],
            "manifest_id": manifest_id,
            "model_id": source_manifest["model_id"],
            "instruction_source": source_manifest["instruction_source"],
            "task_ids": ids,
            "per_task": {"baseline": [{"manifest_id": manifest_id,
                                       "task_id": i, "query": queries[i]}
                                      for i in ids]},
            "limit": "Question hashes match; query text is copied exactly. "
                     "No model call, answer or supplied passage was used."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_snapshot.read_text(encoding="utf-8"))
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    target = json.loads(args.target_snapshot.read_text(encoding="utf-8"))
    result = transfer(source, manifest, target)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({"n": len(result["task_ids"]),
                      "manifest_id": result["manifest_id"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
