#!/usr/bin/env python3
"""Freeze an article/question-disjoint BioASQ yes/no task-input cohort.

The upstream BioASQ package contains several answer-generation tasks for the
same question.  This adapter deliberately uses only its native ``yesno``
rows, keeps the supplied PubMed snippets verbatim, and records the upstream
answer as the source-native label.  It writes a 60/60/60 cohort with a sealed
final holdout so the normal task-input runner can enforce the same protocol as
the other self-evolution experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.util import sha256_file, sha256_obj
SOURCE = ROOT / "vendor/BioASQ2025-UNITOR-main/BioASQ2025-UNITOR-main/data/dataset/answer_generation/train_set_answGEN.json"
DATASET = "bioasq_yesno_fresh_20261019"
OPTIONS = ["yes", "no"]
SPLITS = ("dev_optimize", "selection_gate", "final_holdout")


def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\u00a0", " ").split()).strip()


def _source_text(row: dict[str, Any]) -> str:
    pieces = []
    for key in ("input_2", "input_3", "input_4"):
        value = _norm_text(row.get(key))
        if value:
            pieces.append(value)
    return "\n\n".join(pieces)


def _select(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    by_label = {label: [] for label in OPTIONS}
    seen: set[str] = set()
    source_sha = sha256_file(SOURCE)
    for row in rows:
        if row.get("type") != "yesno":
            continue
        qid = _norm_text(row.get("id"))
        label = _norm_text(row.get("output")).lower()
        question = _norm_text(row.get("input_1"))
        evidence = _source_text(row)
        if not qid or qid in seen or label not in by_label or not question or not evidence:
            continue
        seen.add(qid)
        by_label[label].append(row)
    # Keep the class balance explicit and deterministic.  The source has many
    # more yes than no records; a balanced cohort prevents accuracy from hiding
    # a weak negative-class recall.
    per_label = n // len(OPTIONS)
    chosen = []
    for label in OPTIONS:
        chosen.extend(sorted(by_label[label], key=lambda r: _norm_text(r["id"]))[:per_label])
    if len(chosen) != n:
        raise RuntimeError(f"source has insufficient balanced yes/no rows: {len(chosen)}")
    chosen.sort(key=lambda r: _norm_text(r["id"]))
    rows_out = []
    for row in chosen:
        qid = _norm_text(row["id"])
        label = _norm_text(row["output"]).lower()
        evidence = _source_text(row)
        rows_out.append({
            "task_id": f"bioasq-yesno-{qid}",
            "dataset": DATASET,
            "task_type": "mcq",
            "domain": "biomedical_literature",
            "question": (
                f"{_norm_text(row['input_1'])}\n"
                "Answer using only the supplied PubMed material."),
            "context": "",
            "evidence": evidence,
            "options": OPTIONS,
            "correct_index": OPTIONS.index(label),
            "gold_label": label,
            "sources": [f"bioasq-question:{qid}"],
            "group_id": f"bioasq-question:{qid}",
            "metadata": {
                "upstream_id": qid,
                "upstream_type": row["type"],
                "upstream_answer": label,
                "annotation_origin": "source_native",
                "source_file_sha256": source_sha,
                "source_snippet_fields": ["input_2", "input_3", "input_4"],
            },
        })
    return rows_out


def freeze(*, output_root: Path, n: int = 180) -> dict[str, Any]:
    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = _select(raw, n)
    # Stratify by the source-native answer before assigning the three equal
    # blocks.  A plain contiguous slice of the globally sorted IDs would put
    # nearly all ``no`` answers in the last block because the upstream class
    # distribution is strongly skewed.
    per_split = n // len(SPLITS)
    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in OPTIONS}
    for row in rows:
        by_label[row["gold_label"]].append(row)
    for label in OPTIONS:
        for index, row in enumerate(sorted(by_label[label], key=lambda r: r["task_id"])):
            row["split"] = SPLITS[index % len(SPLITS)]
    rows.sort(key=lambda r: r["task_id"])
    output_root.mkdir(parents=True, exist_ok=True)
    normalized = output_root / "data" / "normalized" / f"{DATASET}.jsonl"
    sealed = output_root / "data" / "sealed" / f"{DATASET}.final_holdout.jsonl"
    manifest_path = output_root / "data" / "manifests" / f"{DATASET}.json"
    normalized.parent.mkdir(parents=True, exist_ok=True)
    sealed.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    visible = [r for r in rows if r["split"] != "final_holdout"]
    final = [r for r in rows if r["split"] == "final_holdout"]
    normalized.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in visible), encoding="utf-8")
    sealed.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in final), encoding="utf-8")
    split_info = {}
    for split in SPLITS:
        part = [r for r in rows if r["split"] == split]
        split_info[split] = {
            "n": len(part),
            "ids": [r["task_id"] for r in part],
            "ids_sha256": sha256_obj([r["task_id"] for r in part]),
            "task_ids_sha256": sha256_obj([r["task_id"] for r in part]),
            "group_ids_sha256": sha256_obj(sorted(r["group_id"] for r in part)),
            "class_counts": dict(sorted(Counter(r["gold_label"] for r in part).items())),
        }
    manifest = {
        "schema": "task-input-dataset-manifest-v3",
        "dataset": DATASET,
        "protocol_version": "bioasq-yesno-fresh-task-input-v1",
        "source_file": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256_file(SOURCE),
        "n_questions": len(rows),
        "ratios": {"dev_optimize": 33, "selection_gate": 33, "final_holdout": 34},
        "salt": "bioasq-source-native-stratified-20261019-v1",
        "split_algorithm": "source-native-sorted-id-stratified-round-robin-v1",
        "source": {
            "path": str(SOURCE.resolve()),
            "sha256": sha256_file(SOURCE),
            "upstream_task_type": "yesno",
            "annotation_origin": "source_native",
        },
        "split_policy": "stable sorted source-native question IDs; 60 dev, 60 selection, 60 final",
        "final_holdout_sealed": True,
        "row_count": len(rows),
        "splits": split_info,
        "outputs": {
            "normalized": {"path": str(normalized.resolve()), "sha256": sha256_file(normalized)},
            "final_holdout": {"path": str(sealed.resolve()), "sha256": sha256_file(sealed)},
        },
    }
    manifest["normalized_path"] = str(normalized.relative_to(ROOT)).replace("\\", "/")
    manifest["normalized_sha256"] = sha256_file(normalized)
    manifest["sealed_path"] = str(sealed.relative_to(ROOT)).replace("\\", "/")
    manifest["sealed_sha256"] = sha256_file(sealed)
    manifest["manifest_sha256"] = sha256_obj(manifest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--n", type=int, default=180)
    args = parser.parse_args()
    report = freeze(output_root=args.output_root, n=args.n)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
