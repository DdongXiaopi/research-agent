"""Compare cited-only and complete top-3 evidence on explored train cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rsi import load_config
from src.data.bioasq_unitor import load_tasks
from src.eval.grounding import (parse_proposition_verdict,
                                proposition_judge_messages)
from src.retrieval.pubmed_snapshot import PubMedSnapshotGateway
from src.runtime.provider import build_provider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--arm", default="instruction",
                        help="Name of an arm in report.per_task")
    parser.add_argument("--model-id", default="Qwen/Qwen3-32B")
    parser.add_argument("--reparse", type=Path,
                        help="Regrade saved raw outputs without API calls")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.reparse:
        raise SystemExit(
            "retired_unplanned_entrypoint: live retrieved-proposition calibration "
            "requires a caller-challenged execution plan and bounded request ledger")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    prior = json.loads(args.report.read_text(encoding="utf-8"))
    if prior["snapshot_content_sha256"] != snapshot["content_sha256"]:
        raise ValueError("report and snapshot differ")
    if args.arm not in prior.get("per_task", {}):
        raise ValueError("report lacks requested arm")
    ids = set(args.task_id)
    task_map = {t["task_id"]: t for t in load_tasks("train", task_ids=ids)}
    rows = {r["task_id"]: r for r in prior["per_task"][args.arm]}
    if not ids <= set(rows):
        raise ValueError("report lacks a requested train task")
    gateway = PubMedSnapshotGateway(snapshot)
    prior_results = {}
    if args.reparse:
        old = json.loads(args.reparse.read_text(encoding="utf-8"))
        if old["snapshot_content_sha256"] != snapshot["content_sha256"]:
            raise ValueError("reparse report and snapshot differ")
        prior_results = {(r["task_id"], r["scope"]): r for r in old["rows"]}
    provider = None
    if not args.reparse:
        cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
        cfg["models"]["task"]["model_id"] = args.model_id
        provider = build_provider(cfg)
        provider.cache = False
    results = []
    for task_id in args.task_id:
        task, row = task_map[task_id], rows[task_id]
        if row["answer_index"] < 0:
            continue
        answer = task["options"][row["answer_index"]]
        corpus = {d["doc_id"]: d for d in gateway.corpus(task)}
        retrieved = row.get("retrieved_ids", row.get("top3", []))
        for scope, doc_ids in (("cited", row["cited"]),
                               (f"retrieved_top{len(retrieved)}", retrieved)):
            docs = [corpus[i] for i in doc_ids if i in corpus]
            if args.reparse:
                saved = prior_results[(task_id, scope)]
                raw = saved["raw_verdict"]
                prompt_tokens = saved["prompt_tokens"]
                completion_tokens = saved["completion_tokens"]
            else:
                system, user = proposition_judge_messages(
                    task["question"], docs,
                    per_document=scope.startswith("retrieved_top"))
                result = provider.chat_measured(
                    system, user,
                    max_tokens=220 + 75 * len(docs)
                    if scope.startswith("retrieved_top") else 220)
                raw = result.content
                prompt_tokens = result.prompt_tokens
                completion_tokens = result.completion_tokens
            supported, reason = parse_proposition_verdict(
                raw, docs, answer,
                require_per_document=scope.startswith("retrieved_top"))
            results.append({"task_id": task_id, "scope": scope,
                            "gold_answer": task["options"][task["correct_index"]],
                            "selected_answer": answer,
                            "selected_correct": row["correct"],
                            "doc_ids": doc_ids, "supported": supported,
                            "reason": reason, "raw_verdict": raw,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens})
    report = {"status": "train_only_retrieved_evidence_calibration",
              "snapshot_content_sha256": snapshot["content_sha256"],
              "model_id": args.model_id, "arm": args.arm,
              "grading": "per-document completeness, exact quotes and conflict detection",
              "rows": results,
              "limit": "Citation attribution and answer sufficiency are distinct; "
                       "both are model proxies and train-only here."}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps([{"task_id": x["task_id"], "scope": x["scope"],
                       "supported": x["supported"], "reason": x["reason"]}
                      for x in results], ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
