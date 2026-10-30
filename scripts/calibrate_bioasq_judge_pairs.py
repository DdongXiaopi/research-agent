"""Train-only contrastive calibration of citation support; no manual labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rsi import load_config
from src.data.bioasq_unitor import SOURCE
from src.eval.grounding import (judge_messages, parse_proposition_verdict,
                                parse_verdict, proposition_judge_messages)
from src.runtime.provider import build_provider

SPAN = re.compile(r"\[BS\](.*?)\[ES\]", re.S)


def _cases(per_label: int, task_ids: set[str] | None = None) -> list[dict]:
    rows = json.loads((SOURCE / "train_set_answGEN.json").read_text(encoding="utf-8"))
    cases = {"yes": [], "no": []}
    for row in rows:
        label = str(row.get("output", "")).strip().casefold()
        if row.get("type") != "yesno" or label not in cases:
            continue
        spans = [" ".join(span.split()) for key in ("input_2", "input_3", "input_4")
                 for span in SPAN.findall(str(row.get(key, ""))) if len(span.strip()) >= 60]
        if not spans or sum(map(len, spans)) > 4000:
            continue
        task_id = "bioasq-" + str(row["id"])
        if task_ids is not None and task_id not in task_ids:
            continue
        cases[label].append({"task_id": task_id,
                             "question": str(row["input_1"]).strip(),
                             "label": label,
                             "docs": [{"doc_id": f"source:{i}", "text": span}
                                      for i, span in enumerate(spans)]})
    if task_ids is not None:
        selected = [case for label in ("yes", "no") for case in cases[label]]
        if {case["task_id"] for case in selected} != task_ids:
            raise RuntimeError("some requested train tasks lack marked evidence")
        return selected
    selected = []
    for label in ("yes", "no"):
        ranked = sorted(cases[label], key=lambda x: hashlib.sha256(
            x["task_id"].encode()).hexdigest())
        selected.extend(ranked[:per_label])
    if len(selected) != per_label * 2:
        raise RuntimeError("insufficient balanced train examples")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-label", type=int, default=6)
    parser.add_argument("--task-id", action="append",
                        help="Calibrate named BioASQ train tasks without manual labels")
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--mode", choices=("answer", "proposition"),
                        default="answer")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/probe/pubmed_snapshot"
                        / "bioasq-judge-contrastive-train12.json")
    parser.add_argument("--reparse", type=Path,
                        help="Regrade saved raw model outputs without new API calls")
    args = parser.parse_args()
    if not args.reparse:
        raise SystemExit(
            "retired_unplanned_entrypoint: live BioASQ judge calibration requires "
            "a caller-challenged execution plan and bounded request ledger")
    if not 1 <= args.per_label <= 30:
        parser.error("per-label must be 1..30")
    prior = json.loads(args.reparse.read_text(encoding="utf-8")) if args.reparse else None
    prior_rows = {(r["task_id"], r["role"]): r for r in prior["rows"]} if prior else {}
    cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
    cfg["models"]["task"]["model_id"] = args.model_id
    provider = None if prior else build_provider(cfg)
    if provider:
        provider.cache = False
    cases = _cases(args.per_label, set(args.task_id) if args.task_id else None)
    calibration_id = hashlib.sha256(json.dumps({
        "source": str(SOURCE / "train_set_answGEN.json"),
        "task_ids": [case["task_id"] for case in cases],
        "model_id": args.model_id, "mode": args.mode,
        "judge_prompt": ("proposition-v3" if args.mode == "proposition"
                         else "answer-v1"),
    }, sort_keys=True).encode("utf-8")).hexdigest()
    checkpoint = args.output.with_suffix(args.output.suffix + ".partial.jsonl")
    checkpoint_rows = {}
    if not prior and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            saved = json.loads(line)
            if saved.get("calibration_id") != calibration_id:
                raise ValueError("BioASQ judge checkpoint protocol changed")
            key = (saved["task_id"], saved["role"])
            if key in checkpoint_rows:
                raise ValueError("duplicate BioASQ judge checkpoint row")
            checkpoint_rows[key] = saved
    rows = []
    for case in cases:
        shared = None
        saved_any = next((checkpoint_rows[(case["task_id"], role)]
                          for role in ("gold", "opposite")
                          if (case["task_id"], role) in checkpoint_rows), None)
        if args.mode == "proposition" and saved_any is not None:
            shared = (saved_any["raw_verdict"], saved_any["prompt_tokens"],
                      saved_any["completion_tokens"])
        for role, answer in (("gold", case["label"]),
                             ("opposite", "no" if case["label"] == "yes" else "yes")):
            key = (case["task_id"], role)
            if key in prior_rows:
                rows.append(prior_rows[key])
                continue
            if key in checkpoint_rows:
                rows.append(checkpoint_rows[key])
                continue
            if args.mode == "proposition":
                system, user = proposition_judge_messages(
                    case["question"], case["docs"])
            else:
                system, user = judge_messages(
                    case["question"], answer, case["docs"])
            if prior:
                saved = prior_rows[(case["task_id"], role)]
                raw = saved["raw_verdict"]
                prompt_tokens = saved["prompt_tokens"]
                completion_tokens = saved["completion_tokens"]
            elif args.mode == "proposition" and shared is not None:
                raw, prompt_tokens, completion_tokens = shared
            else:
                result = provider.chat_measured(system, user, max_tokens=220)
                raw = result.content
                prompt_tokens = result.prompt_tokens
                completion_tokens = result.completion_tokens
                if args.mode == "proposition":
                    shared = (raw, prompt_tokens, completion_tokens)
            if args.mode == "proposition":
                supported, reason = parse_proposition_verdict(
                    raw, case["docs"], answer)
            else:
                supported, reason = parse_verdict(raw, case["docs"])
            rows.append({"task_id": case["task_id"], "question": case["question"],
                         "gold_label": case["label"], "tested_answer": answer,
                         "role": role, "supported": supported, "reason": reason,
                         "n_evidence_spans": len(case["docs"]),
                         "raw_verdict": raw,
                         "prompt_tokens": prompt_tokens,
                         "completion_tokens": completion_tokens,
                         "calibration_id": calibration_id})
            if not prior:
                with checkpoint.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
    gold = [r for r in rows if r["role"] == "gold"]
    opposite = [r for r in rows if r["role"] == "opposite"]
    report = {"status": "train_only_contrastive_judge_calibration",
              "calibration_id": calibration_id,
              "source": str(SOURCE / "train_set_answGEN.json"),
              "model_id": prior.get("model_id", args.model_id) if prior else args.model_id,
              "judge_mode": args.mode,
              "judge_prompt_version": ("proposition-v3" if args.mode == "proposition"
                                       else "answer-v1"),
              "selection": "balanced yes/no by task-id hash; marked evidence spans only",
              "grading": "strict verdict + exact quote; unique numeric source/pmid alias",
              "n_cases": len(gold),
              "n_calls": len(gold) if args.mode == "proposition" else len(rows),
              "gold_support_rate": sum(r["supported"] for r in gold) / len(gold),
              "opposite_false_support_rate": sum(r["supported"] for r in opposite)
              / len(opposite),
              "rows": rows,
              "limit": "BioASQ marked spans are weak supervision for attribution; this "
                       "is calibration, not a definitive citation benchmark. Dev untouched."}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "n_cases", "n_calls", "gold_support_rate", "opposite_false_support_rate")},
        ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
