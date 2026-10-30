"""Train-only causal probe: visible-abstract evidence ledger before answering."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pilot_pubmed_window_grounding import append_checkpoint, compact, load_checkpoint
from scripts.rsi import load_config
from src.data.bioasq_unitor import load_tasks
from src.eval.final_eval import protocol_fingerprint
from src.eval.grounding import (extract_per_document_evidence,
                                proposition_judge_messages)
from src.eval.metrics import aggregate
from src.governance.release import load_state
from src.retrieval.pubmed_snapshot import PubMedSnapshotGateway
from src.runtime.provider import build_provider
from src.runtime.runner import TaskRunner
from src.util import sha256_obj


def render_ledger(rows: list[dict]) -> str:
    relation = {"true": "supports the question's proposition",
                "false": "refutes the question's proposition",
                "uncertain": "does not decide the question's proposition"}
    lines = ["Fallible per-article evidence notes. Verify them against the "
             "retrieved excerpts before answering; do not treat them as an answer key."]
    for row in rows:
        line = f"[{row['doc_id']}] {relation[row['relation']]}"
        if row["quote"]:
            line += f"; exact excerpt: {row['quote']}"
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    raise SystemExit(
        "unplanned historical evidence-ledger scoring is retired: direct "
        "judge and answer calls have no frozen execution-plan hash, request "
        "caps, or shared call ledger"
    )

    # Retained as historical implementation only pending governed replay.
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--judge-model-id", default="Qwen/Qwen3-32B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.skip < 0 or args.limit <= 0:
        parser.error("skip must be nonnegative and limit positive")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    manifest = json.loads(args.query_manifest.read_text(encoding="utf-8"))
    if manifest.get("snapshot_content_sha256") != snapshot["content_sha256"]:
        raise ValueError("frozen queries and PubMed snapshot differ")
    ids = [row["task_id"] for row in snapshot["tasks"]][args.skip:args.skip + args.limit]
    queries = {row["task_id"]: row["query"]
               for row in manifest["per_task"]["baseline"]}
    if not ids or not set(ids) <= set(queries):
        raise ValueError("manifest omits selected tasks")
    tasks = {task["task_id"]: task for task in load_tasks(
        "train", task_ids=set(ids))}
    gateway = PubMedSnapshotGateway(snapshot, top_k=3,
                                    passage_mode="query_window")
    cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
    cfg["retrieval"]["query_control"] = {
        "enabled": True, "mode": "frozen", "queries": {i: queries[i] for i in ids}}
    cfg["retrieval"]["anchor_question_terms"] = True
    cfg["retrieval"]["pubmed_passage_mode"] = "query_window"
    cfg["retrieval"]["snapshot_content_sha256"] = snapshot["content_sha256"]
    cfg["skills"]["pinned_skill"] = "litqa-literature-synthesis"
    cfg["evaluation"]["semantic_citation"] = {"enabled": False}
    provider = build_provider(cfg)
    provider.cache = False
    judge_cfg = copy.deepcopy(cfg)
    judge_cfg["models"]["task"]["model_id"] = args.judge_model_id
    judge = build_provider(judge_cfg)
    judge.cache = False
    state = load_state("baseline-v1-cited")
    pilot_id = sha256_obj({"snapshot": snapshot["content_sha256"],
                           "manifest": manifest["manifest_id"],
                           "cfg": cfg, "protocol": protocol_fingerprint(cfg),
                           "judge_model": args.judge_model_id,
                           "ids": ids, "version": "visible-evidence-ledger-v1"})
    outcome_journal = args.output.with_name(args.output.stem + ".partial.jsonl")
    ledger_journal = args.output.with_name(args.output.stem + ".ledger.jsonl")
    saved = load_checkpoint(outcome_journal, pilot_id)
    ledgers = {}
    if ledger_journal.exists():
        for line in ledger_journal.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["pilot_id"] != pilot_id or row["task_id"] in ledgers:
                raise ValueError("ledger checkpoint protocol changed or duplicated")
            ledgers[row["task_id"]] = row
    outcomes = {"baseline": [], "ledger": []}
    for task_id in ids:
        task = tasks[task_id]
        query = queries[task_id] + " " + gateway.anchor_query(
            task["question"], max_terms=20)
        docs = gateway.search(task, query=query)
        if task_id in ledgers:
            note = ledgers[task_id]
        else:
            system, user = proposition_judge_messages(
                task["question"], docs, per_document=True)
            answer = judge.chat_measured(system, user, max_tokens=450)
            extracted, reason = extract_per_document_evidence(
                answer.content, docs)
            note = {"pilot_id": pilot_id, "task_id": task_id,
                    "query": query, "doc_ids": [d["doc_id"] for d in docs],
                    "valid": extracted is not None, "reason": reason,
                    "evidence": extracted, "raw": answer.content,
                    "prompt_tokens": answer.prompt_tokens,
                    "completion_tokens": answer.completion_tokens}
            with ledger_journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(note, ensure_ascii=False) + "\n")
            ledgers[task_id] = note
        if note["query"] != query or note["doc_ids"] != [d["doc_id"] for d in docs]:
            raise ValueError("ledger checkpoint query or document order differs")
        for arm in outcomes:
            key = (task_id, arm)
            if key in saved:
                outcome = saved[key]
            else:
                arm_task = dict(task)
                if arm == "ledger" and note["valid"]:
                    arm_task["context"] = render_ledger(note["evidence"])
                outcome = TaskRunner(cfg, state, provider=provider,
                                     gateway=gateway).run(arm_task)
                if outcome.retrieval_query != query or outcome.retrieved_doc_ids != note[
                        "doc_ids"]:
                    raise ValueError("answer arm did not use ledger's query and documents")
                append_checkpoint(outcome_journal, pilot_id, task_id, arm, outcome)
            outcomes[arm].append(outcome)
    metrics = {arm: aggregate(values) for arm, values in outcomes.items()}
    report = {"status": "train_only_visible_evidence_ledger_probe",
              "pilot_id": pilot_id,
              "snapshot_content_sha256": snapshot["content_sha256"],
              "query_manifest_id": manifest["manifest_id"],
              "judge_model_id": args.judge_model_id,
              "task_ids": ids,
              "ledger_valid": sum(ledgers[i]["valid"] for i in ids),
              "ledger_prompt_tokens": sum(ledgers[i]["prompt_tokens"] for i in ids),
              "ledger_completion_tokens": sum(ledgers[i]["completion_tokens"] for i in ids),
              "metrics": metrics,
              "per_task": {arm: [compact(item) for item in values]
                           for arm, values in outcomes.items()},
              "limit": "Train-only fixed preprocessing probe. Ledger uses model "
                       "judgments that are not gold entailment; extra model cost "
                       "is separate. No advancement or dev access."}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({"ledger_valid": report["ledger_valid"],
                      "metrics": {arm: {key: val[key] for key in (
                          "accuracy", "coverage", "answered_precision")}
                          for arm, val in metrics.items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
