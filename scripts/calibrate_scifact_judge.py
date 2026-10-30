"""Calibrate the citation proposition verifier on official SciFact dev labels.

Only claim and the paper's annotated rationale sentence(s) go to the model.
Gold SUPPORT/CONTRADICT is used afterwards for scoring. Passing a whole abstract
would let unrelated sentences determine the verdict and hide rationale/label
quality problems, so this calibration intentionally uses rationale-only text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rsi import load_config
from src.eval.grounding import (_json_value, parse_proposition_verdict,
                                proposition_judge_messages)
from src.evolution.gepa_adapter import summarize_chat_usage
from src.runtime.provider import build_provider
from src.util import sha256_obj


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def select_pairs(data_dir: Path, per_label: int,
                 offset_per_label: int = 0,
                 split: str = "dev",
                 evidence_scope: str = "rationale") -> tuple[list[dict], str]:
    if split not in {"train", "dev"}:
        raise ValueError("SciFact calibration split must be train or dev")
    if evidence_scope not in {"rationale", "abstract"}:
        raise ValueError("evidence_scope must be rationale or abstract")
    claims_path = data_dir / f"claims_{split}.jsonl"
    corpus_path = data_dir / "corpus.jsonl"
    corpus = {str(row["doc_id"]): row for row in read_jsonl(corpus_path)}
    by_label = {"SUPPORT": [], "CONTRADICT": []}
    for claim in read_jsonl(claims_path):
        for doc_id, rationales in claim["evidence"].items():
            labels = {x["label"] for x in rationales}
            if len(labels) != 1 or doc_id not in corpus:
                continue
            label = next(iter(labels))
            if label not in by_label:
                continue
            article = corpus[doc_id]
            abstract = article.get("abstract") or []
            sentence_ids = sorted({int(i) for rationale in rationales
                                   for i in rationale.get("sentences", [])})
            if not sentence_ids or any(i < 0 or i >= len(abstract)
                                       for i in sentence_ids):
                continue
            rationale_text = " ".join(str(abstract[i]) for i in sentence_ids)
            text = (rationale_text if evidence_scope == "rationale" else
                    article.get("title", "") + "\n" + " ".join(abstract))
            by_label[label].append({"claim_id": claim["id"],
                                    "claim": claim["claim"],
                                    "doc_id": doc_id, "label": label,
                                    "rationale_sentence_ids": sentence_ids,
                                    "rationale_text": rationale_text,
                                    "abstract_text": article.get("title", "")
                                    + "\n" + " ".join(abstract),
                                    "text": text})
    selected = []
    for label, rows in by_label.items():
        if len(rows) < offset_per_label + per_label:
            raise ValueError("SciFact dev lacks requested label balance")
        rows.sort(key=lambda row: hashlib.sha256(
            f"{row['claim_id']}:{row['doc_id']}".encode()).hexdigest())
        selected.extend(rows[offset_per_label:offset_per_label + per_label])
    selected.sort(key=lambda row: hashlib.sha256(
        f"mixed:{row['claim_id']}:{row['doc_id']}".encode()).hexdigest())
    archive = data_dir.parent / "data.tar.gz"
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    return selected, archive_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "artifacts/probe/scifact/data")
    parser.add_argument("--per-label", type=int, default=12)
    parser.add_argument("--offset-per-label", type=int, default=0)
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--evidence-scope", choices=("rationale", "abstract"),
                        default="rationale")
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parser.error(
        "live SciFact calibration is disabled until a plan-bound, ledger-limited "
        "replacement is available; keep prior v6 results exploratory")
    if args.per_label <= 0 or args.offset_per_label < 0:
        parser.error("per-label must be positive and offset nonnegative")
    pairs, archive_hash = select_pairs(args.data_dir, args.per_label,
                                       args.offset_per_label, args.split,
                                       args.evidence_scope)
    models = args.model_id or ["Qwen/Qwen3-8B", "Qwen/Qwen3-32B"]
    protocol = sha256_obj({"dataset": archive_hash,
                           "selection": [(x["claim_id"], x["doc_id"], x["label"])
                                         for x in pairs],
                           "models": models,
                           "prompt": "proposition_judge_messages-v3",
                           "evidence_scope": f"scifact-{args.evidence_scope}-v1",
                           "parser": "parse_proposition_verdict-v2",
                           "doc_id_schema": "source:<SciFact doc id>"})
    journal = args.output.with_name(args.output.stem + ".partial.jsonl")
    saved = {}
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["protocol"] != protocol:
                raise ValueError("calibration checkpoint protocol changed")
            key = (row["model_id"], row["claim_id"], row["doc_id"])
            if key in saved:
                raise ValueError("duplicate calibration checkpoint")
            saved[key] = row
    rows = []
    for model_id in models:
        cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
        cfg["models"]["task"]["model_id"] = model_id
        provider = build_provider(cfg)
        provider.cache = False
        for pair in pairs:
            key = (model_id, pair["claim_id"], pair["doc_id"])
            if key in saved:
                rows.append(saved[key])
                continue
            # The production parser accepts a model's numeric shorthand for
            # source:<id> but not an arbitrary external prefix.
            doc = {"doc_id": "source:" + pair["doc_id"],
                   "text": pair["text"]}
            question = "Is the following scientific claim true? " + pair["claim"]
            system, user = proposition_judge_messages(question, [doc])
            response = provider.chat_measured(system, user, max_tokens=220)
            answer = "yes" if pair["label"] == "SUPPORT" else "no"
            supported, reason = parse_proposition_verdict(
                response.content, [doc], answer)
            raw = _json_value(response.content)
            if not isinstance(raw, dict):
                raw = {}
            row = {"protocol": protocol, "model_id": model_id,
                   "claim_id": pair["claim_id"], "doc_id": pair["doc_id"],
                   "rationale_sentence_ids": pair["rationale_sentence_ids"],
                   "gold": pair["label"], "relation": raw.get("relation"),
                   "label_and_quote_agree": supported, "reason": reason,
                   "raw_verdict": response.content,
                   "prompt_tokens": response.prompt_tokens,
                   "completion_tokens": response.completion_tokens,
                   "latency_ms": response.latency_ms,
                   "cache_hit": response.cache_hit,
                   "usage_available": response.usage_available,
                   "cost": response.cost}
            with journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
    summary = {}
    for model_id in models:
        selected = [r for r in rows if r["model_id"] == model_id]
        summary[model_id] = {
            "n": len(selected),
            "gold_support": sum(r["gold"] == "SUPPORT" for r in selected),
            "gold_contradict": sum(r["gold"] == "CONTRADICT" for r in selected),
            "label_and_quote_agree": sum(r["label_and_quote_agree"]
                                         for r in selected),
            "support_as_false": sum(r["gold"] == "SUPPORT" and
                                    r["relation"] == "false" for r in selected),
            "contradict_as_true": sum(r["gold"] == "CONTRADICT" and
                                      r["relation"] == "true" for r in selected),
            "uncertain": sum(r["relation"] == "uncertain" for r in selected),
            "invalid_quote_or_schema": sum(r["reason"] in {
                "quote_not_in_citation", "quote_too_short", "unknown_citation",
                "invalid_json", "invalid_schema"} for r in selected),
            "usage": summarize_chat_usage(selected),
        }
    report = {"status": f"public_scifact_{args.split}_judge_calibration",
              "source": "https://github.com/allenai/scifact",
              "judge_prompt_version": "proposition-v3",
              "evidence_scope": f"scifact-{args.evidence_scope}-v1",
              "split": args.split,
              "archive_sha256": archive_hash,
              "protocol": protocol, "summary": summary, "rows": rows,
              "limit": (f"Gold claim evidence labels are from SciFact {args.split}. "
                        "Transfer to PubMedQA/BioASQ questions is unproven; "
                        "no confirmation data was accessed.")}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
