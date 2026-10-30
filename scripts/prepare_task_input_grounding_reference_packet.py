"""Prepare a blinded human reference packet for task-input grounding.

This script reads only PubMedQA's visible ``dev_optimize`` cases and never
constructs a provider. It creates two independently ordered reviewer copies,
two empty response templates, and a separate researcher key. Benchmark answer
labels and historical model outputs are excluded from the reviewer copies.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_task_input_grounding import select_cases
from src.runtime.input_evidence import task_input_docs
from src.util import sha256_obj


SCHEMA = "task-input-grounding-human-reference-v2"
MAX_DOC_CHARS = 6000
ANNOTATION_FIELDS = (
    "verdict", "supporting_quote", "quote_doc_id", "reason",
    "confidence_1_to_3",
)


def _task_input_record(task: dict[str, Any]) -> dict[str, Any]:
    docs = task_input_docs(task)
    return {
        "question": str(task["question"]),
        "evidence": [
            {"doc_id": str(doc["doc_id"]),
             "text": str(doc["text"])[:MAX_DOC_CHARS]}
            for doc in docs
        ],
    }


def _case_id(protocol_sha256: str, task_id: str) -> str:
    digest = hashlib.sha256(
        f"{protocol_sha256}\0{task_id}".encode("utf-8")).hexdigest()
    return "PUBMEDQA-GND-" + digest[:14]


def build_reference_packets(per_label: int = 30,
                            offset_per_label: int = 0) -> dict[str, Any]:
    """Build blind packets and a withheld researcher key without model calls."""
    if not 1 <= per_label <= 100 or offset_per_label < 0:
        raise ValueError("per_label must be 1..100 and offset_per_label nonnegative")
    tasks = select_cases(per_label, offset_per_label)
    if len({str(task["task_id"]) for task in tasks}) != len(tasks):
        raise ValueError("grounding reference cohort has duplicate task IDs")

    records = []
    key_rows = []
    for task in tasks:
        task_id = str(task["task_id"])
        payload = _task_input_record(task)
        records.append({
            "task_id": task_id,
            "answer_label": str(
                task["options"][int(task["correct_index"])]).strip().casefold(),
            "input_sha256": sha256_obj(payload),
            **payload,
        })
    # Keep benchmark labels and task IDs out of the reviewer-facing protocol
    # payload. The public protocol identity still binds every exact question /
    # evidence payload and the code/config used to construct the packet.
    # v2 packets are a legacy frozen protocol. Keep their source identity
    # stable while the scorer adds v3 dataset-general packet support.
    from scripts.score_task_input_grounding_human_reference import (
        PUBMEDQA_REFERENCE_SOURCE_HASHES_V2,
    )
    source_hashes = dict(PUBMEDQA_REFERENCE_SOURCE_HASHES_V2)
    protocol_context = {
        "schema": "task-input-grounding-reference-protocol-v2",
        "dataset": "pubmedqa",
        "split": "dev_optimize",
        "sampling": "sha256-ranked-within-benchmark-answer-label",
        "per_label": per_label,
        "offset_per_label": offset_per_label,
        "judge_evidence_character_limit_per_document": MAX_DOC_CHARS,
        "rubric_version": "human-grounding-proposition-v1",
        "case_input_sha256s": sorted(row["input_sha256"] for row in records),
        "source_hashes": source_hashes,
    }
    protocol = sha256_obj(protocol_context)

    reviewer_cases = []
    for row in records:
        reviewer_cases.append({
            "case_id": _case_id(protocol, row["task_id"]),
            "question": row["question"],
            "evidence": row["evidence"],
            "input_sha256": row["input_sha256"],
        })
    if len({case["case_id"] for case in reviewer_cases}) != len(reviewer_cases):
        raise ValueError("blinded case ID collision")

    instructions = (
        "Judge whether the supplied abstract evidence establishes the proposition "
        "in the yes/no research question. Use only the question and supplied text; "
        "do not search externally or infer from a benchmark answer. Verdict values: "
        "supported (the evidence establishes the proposition), contradicted (the "
        "evidence establishes its opposite), insufficient (the evidence does not "
        "establish either side, is conflicting, or misses a material qualifier), "
        "unable_to_judge (you cannot make a reliable assessment). Pay attention to "
        "population, intervention/exposure, comparator, outcome, direction, and "
        "negation. For supported or contradicted, quote one exact contiguous passage "
        "and give its doc_id. Do every case independently, do not discuss labels "
        "with the other reviewer, and leave no case blank. Use confidence 1 (low), "
        "2 (moderate), or 3 (high). Disagreement is preserved; do not force agreement."
    )

    packets: dict[str, dict[str, Any]] = {}
    responses: dict[str, dict[str, Any]] = {}
    for reviewer_slot, shuffle_seed in (("reviewer_a", 0), ("reviewer_b", 1)):
        ordered = list(reviewer_cases)
        random.Random(int(protocol[:16], 16) + shuffle_seed).shuffle(ordered)
        unsigned = {
            "schema": SCHEMA,
            "protocol_sha256": protocol,
            "protocol_context": protocol_context,
            "reviewer_slot": reviewer_slot,
            "scope": {
                "dataset": "pubmedqa",
                "split": "dev_optimize",
                "case_count": len(ordered),
                "use": "human semantic reference annotations for a task-input grounding judge",
                "limitations": [
                    "dev_optimize only; not a confirmation or selection-gate sample",
                    "benchmark answer labels are not entailment gold",
                    "historical model outputs are intentionally withheld",
                ],
            },
            "instructions": instructions,
            "cases": ordered,
        }
        packets[reviewer_slot] = {
            **unsigned,
            "packet_sha256": sha256_obj(unsigned),
        }
        responses[reviewer_slot] = {
            "schema": "task-input-grounding-human-reference-responses-v2",
            "protocol_sha256": protocol,
            "packet_sha256": packets[reviewer_slot]["packet_sha256"],
            "reviewer_id": "",
            "blind_attestation": False,
            "annotations": [
                {"case_id": case["case_id"],
                 **{field: "" for field in ANNOTATION_FIELDS},
                 "confidence_1_to_3": None}
                for case in ordered
            ],
        }

    id_by_task = {_case_id(protocol, row["task_id"]): row["task_id"]
                  for row in records}
    for row in records:
        case_id = _case_id(protocol, row["task_id"])
        key_rows.append({
            "case_id": case_id,
            "task_id": row["task_id"],
            "benchmark_answer_label_for_post_review_only": row["answer_label"],
            "input_sha256": row["input_sha256"],
        })
    key = {
        "schema": "task-input-grounding-human-reference-researcher-key-v1",
        "protocol_sha256": protocol,
        "reviewer_packet_sha256": {
            slot: packet["packet_sha256"] for slot, packet in packets.items()},
        "case_count": len(key_rows),
        "sampling": {
            "per_benchmark_answer_label": per_label,
            "offset_per_label": offset_per_label,
            "selection": "SHA-256 ranked task IDs within each label in dev_optimize",
        },
        "warning": (
            "Do not share this key with reviewers. PubMedQA benchmark answers are "
            "not semantic entailment gold and must not replace the two human labels."),
        "cases": sorted(key_rows, key=lambda row: row["case_id"]),
    }
    # Ensure blind copies contain no join key or benchmark-answer metadata.
    forbidden = {"task_id", "correct_index", "options", "answer_label",
                 "gold_answer", "raw_verdict", "model_output"}
    for packet in packets.values():
        for case in packet["cases"]:
            if forbidden & set(case):
                raise ValueError("reviewer packet contains answer or model metadata")
    if set(id_by_task) != {row["case_id"] for row in key_rows}:
        raise ValueError("researcher key case mapping is inconsistent")
    return {"protocol_sha256": protocol, "packets": packets,
            "responses": responses, "researcher_key": key}


def write_reference_packets(output_dir: Path, per_label: int = 30,
                            offset_per_label: int = 0) -> dict[str, Path]:
    bundle = build_reference_packets(per_label, offset_per_label)
    paths: dict[str, Path] = {}
    for slot in ("reviewer_a", "reviewer_b"):
        suffix = "a" if slot.endswith("a") else "b"
        paths[f"packet_{suffix}"] = output_dir / f"reviewer_{suffix}_packet.json"
        paths[f"response_{suffix}"] = output_dir / f"reviewer_{suffix}_responses.json"
    paths["researcher_key"] = output_dir / "researcher_key_do_not_share.json"
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite grounding reference artifacts: "
            + ", ".join(str(path) for path in existing))
    output_dir.mkdir(parents=True, exist_ok=True)
    for slot in ("reviewer_a", "reviewer_b"):
        suffix = "a" if slot.endswith("a") else "b"
        paths[f"packet_{suffix}"].write_text(
            json.dumps(bundle["packets"][slot], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        paths[f"response_{suffix}"].write_text(
            json.dumps(bundle["responses"][slot], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    paths["researcher_key"].write_text(
        json.dumps(bundle["researcher_key"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=30)
    parser.add_argument("--offset-per-label", type=int, default=0)
    args = parser.parse_args()
    paths = write_reference_packets(args.output_dir, args.per_label,
                                    args.offset_per_label)
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
