"""Freeze question-only model queries for a paired BioASQ train diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rsi import load_config
from src.data.bioasq_unitor import load_train_question_manifest
from src.governance.release import load_state
from src.retrieval.intent import (model_query_messages, parse_model_query,
                                  query_instruction)
from src.runtime.provider import build_provider
from src.runtime.skills import load_body
from src.util import sha256_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(
        "retired_unplanned_entrypoint: query freezing requires a caller-challenged "
        "execution plan and bounded request ledger")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    ids = [r["task_id"] for r in snapshot["tasks"]]
    question_rows = {r["task_id"]: r for r in
                     load_train_question_manifest(task_ids=set(ids))}
    base = load_state("baseline-v1-cited")
    body = load_body(base.registry.get("litqa-literature-synthesis"))
    instruction, source = query_instruction(base.system_prompt, body)
    cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
    model_id = cfg["models"]["task"]["model_id"]
    manifest_id = sha256_obj({"snapshot": snapshot["content_sha256"],
                              "instruction": instruction,
                              "model_id": model_id,
                              "query_prompt": "model_query_messages-v1",
                              "task_ids": ids})
    journal = args.output.with_name(args.output.stem + ".partial.jsonl")
    saved = {}
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["manifest_id"] != manifest_id:
                raise ValueError("query checkpoint uses a different protocol")
            if row["task_id"] in saved:
                raise ValueError("duplicate query checkpoint")
            saved[row["task_id"]] = row
    provider = build_provider(cfg)
    provider.cache = False
    rows = []
    for task_id in ids:
        if task_id in saved:
            row = saved[task_id]
        else:
            system, user = model_query_messages(
                instruction, question_rows[task_id]["question"])
            result = provider.chat_measured(system, user, max_tokens=100)
            query = parse_model_query(result.content)
            row = {"manifest_id": manifest_id, "task_id": task_id,
                   "query": query,
                   "prompt_tokens": result.prompt_tokens,
                   "completion_tokens": result.completion_tokens}
            with journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)
    report = {"status": "train_only_frozen_query_manifest",
              "snapshot_content_sha256": snapshot["content_sha256"],
              "manifest_id": manifest_id, "model_id": model_id,
              "instruction_source": source,
              "task_ids": ids,
              "per_task": {"baseline": rows},
              "limit": "Only public train questions were sent to the model. "
                       "No answers or supplied evidence were used."}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({"n": len(rows), "manifest_id": manifest_id,
                      "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
                      "completion_tokens": sum(r["completion_tokens"] for r in rows)},
                     ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
