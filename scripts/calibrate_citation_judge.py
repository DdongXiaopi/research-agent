"""Calibrate citation judge on four already explored BioASQ train tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rsi import load_config
from src.data.bioasq_unitor import load_tasks
from src.eval.grounding import (judge_messages, parse_proposition_verdict,
                                parse_verdict, proposition_judge_messages)
from src.retrieval.pubmed_snapshot import PubMedSnapshotGateway
from src.runtime.provider import build_provider


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contrastive", action="store_true")
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--mode", choices=("answer", "proposition"),
                        default="answer")
    args = parser.parse_args()
    raise SystemExit(
        "retired_unplanned_entrypoint: citation judge calibration requires a "
        "caller-challenged execution plan and bounded request ledger")
    prior = json.loads((ROOT / "artifacts/probe/pubmed_snapshot"
                        / "model-runtime-control-v2-grounding.json").read_text(encoding="utf-8"))
    snapshot = json.loads((ROOT / "artifacts/probe/pubmed_snapshot"
                           / "no-doc-four-v3-2026-09-29.json").read_text(encoding="utf-8"))
    gateway = PubMedSnapshotGateway(snapshot, top_k=3)
    ids = set(prior["task_ids"])
    tasks = {t["task_id"]: t for t in load_tasks("train", task_ids=ids)}
    cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
    cfg["models"]["task"]["model_id"] = args.model_id
    provider = build_provider(cfg)
    provider.cache = False
    rows = []
    for old in prior["paired_query"]["baseline"]:
        task = tasks[old["task_id"]]
        cited = [d for d in gateway.corpus(task) if d["doc_id"] in old["cited"]]
        if not cited:
            continue
        answer = task["options"][old["answer_index"]]
        if args.mode == "proposition":
            system, user = proposition_judge_messages(task["question"], cited)
        else:
            system, user = judge_messages(task["question"], answer, cited)
        result = provider.chat_measured(system, user, max_tokens=220)
        if args.mode == "proposition":
            supported, reason = parse_proposition_verdict(
                result.content, cited, answer)
        else:
            supported, reason = parse_verdict(result.content, cited)
        opposite_raw = ""
        if supported and args.contrastive and args.mode == "answer" and answer.casefold() in {"yes", "no"}:
            opposite = "no" if answer.casefold() == "yes" else "yes"
            opposite_system, opposite_user = judge_messages(
                task["question"], opposite, cited)
            opposite_result = provider.chat_measured(
                opposite_system, opposite_user, max_tokens=220)
            opposite_raw = opposite_result.content
            opposite_supported, _ = parse_verdict(opposite_raw, cited)
            if opposite_supported:
                supported, reason = False, "both_answers_supported"
        rows.append({"task_id": old["task_id"], "answer_index": old["answer_index"],
                     "correct": old["correct"], "cited": old["cited"],
                     "judge_supported": supported, "reason": reason,
                     "raw_verdict": result.content[:1000],
                     "opposite_verdict": opposite_raw[:1000]})
    suffix = args.model_id.rsplit("/", 1)[-1].lower()
    output = ROOT / "artifacts/probe/pubmed_snapshot" / (
        "citation-judge-v3-" + suffix + "-" + args.mode +
        ("-contrastive" if args.contrastive else "") + ".json")
    output.write_text(json.dumps({"status": "train_only_judge_calibration",
                                  "model_id": args.model_id,
                                  "rows": rows}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({"n": len(rows),
                      "false_support_on_wrong_answer": sum(
                          x["judge_supported"] and not x["correct"] for x in rows),
                      "supported_correct": sum(
                          x["judge_supported"] and x["correct"] for x in rows)},
                     ensure_ascii=False))
    print(output)


if __name__ == "__main__":
    main()
