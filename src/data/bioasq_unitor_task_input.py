"""Train-only BioASQ yes/no tasks using fixed, task-supplied evidence.

This adapter is separate from ``bioasq_unitor``'s retrieval fixture: no hard
negatives, retrieval gateway, or confirmation split is constructed here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from ..util import commit_new_files, jsonl_bytes, sha256_file, sha256_obj

ROOT = Path(__file__).resolve().parents[2]
SOURCE_TRAIN = (ROOT / "vendor/BioASQ2025-UNITOR-main/BioASQ2025-UNITOR-main"
                / "data/dataset/answer_generation/train_set_answGEN.json")
DATASET = "bioasq_unitor_task_input"
PROTOCOL_VERSION = "bioasq-unitor-task-input-v1"
SPLIT_SALT = "rsi-bioasq-unitor-task-input-2026-10-04-v1"
SPLIT_RATIOS = {"dev_optimize": 80, "selection_gate": 20}
MIN_EVIDENCE_CHARS = 40
_TRUNCATION_PREFIX = re.compile(r"^\s*Abstract truncated here\s*\.\.\.\s*", re.I)
_TRUNCATION_SUFFIX = re.compile(r"\s*\.\.\.\s*Abstract truncated here\s*$", re.I)


def _clean(text: Any) -> str:
    value = str(text or "")
    value = re.sub(r"\[/?(?:BS|ES)\]", "", value, flags=re.I)
    value = _TRUNCATION_PREFIX.sub("", value)
    value = _TRUNCATION_SUFFIX.sub("", value)
    return re.sub(r"\s+", " ", value).strip()


def _split(task_id: str) -> str:
    digest = hashlib.sha256(
        f"{SPLIT_SALT}|{DATASET}|{task_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "dev_optimize" if bucket < 80 else "selection_gate"


def _source_rows() -> List[Dict[str, Any]]:
    return copy.deepcopy(_source_rows_cached(sha256_file(SOURCE_TRAIN)))


@lru_cache(maxsize=1)
def _source_rows_cached(source_sha256: str) -> List[Dict[str, Any]]:
    # Deliberately read only the public training file. Do not add a dev path or
    # an allow_confirmation escape hatch to this evolution-data adapter.
    rows = json.loads(SOURCE_TRAIN.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("BioASQ UniTor train source must be a JSON array")
    return rows


def _all_tasks() -> List[Dict[str, Any]]:
    return copy.deepcopy(_all_tasks_cached(sha256_file(SOURCE_TRAIN)))


@lru_cache(maxsize=1)
def _all_tasks_cached(source_sha256: str) -> List[Dict[str, Any]]:
    out = []
    seen_ids = set()
    for row in _source_rows_cached(source_sha256):
        if str(row.get("type", "")).casefold() != "yesno":
            continue
        answer = str(row.get("output", "")).strip().casefold()
        if answer not in {"yes", "no"}:
            raise ValueError("BioASQ yesno row has a non-binary reference answer")
        raw_id = str(row.get("id", "")).strip()
        question = str(row.get("input_1", "")).strip()
        if not raw_id or not question:
            raise ValueError("BioASQ yesno row is missing ID or question")
        task_id = "bioasq-unitor-" + raw_id
        if task_id in seen_ids:
            raise ValueError("duplicate BioASQ yesno task ID")
        seen_ids.add(task_id)
        evidence = []
        seen_material = set()
        for key in ("input_2", "input_3", "input_4"):
            text = _clean(row.get(key, ""))
            if len(text) < MIN_EVIDENCE_CHARS or text in seen_material:
                continue
            evidence.append(text)
            seen_material.add(text)
        if not evidence:
            continue
        out.append({
            "task_id": task_id,
            "dataset": DATASET,
            "split": _split(task_id),
            "task_type": "mcq",
            "domain": "biomedical",
            "question": question,
            "options": ["yes", "no"],
            "correct_index": 0 if answer == "yes" else 1,
            "context": "",
            "evidence": "\n\n".join(evidence),
            "sources": [],
        })
    return sorted(out, key=lambda task: task["task_id"])


def load_tasks(splits: List[str] | None = None,
               limit: int | None = None) -> List[Dict[str, Any]]:
    """Return task-input train examples; reject every non-development split."""
    allowed = {"dev_optimize", "selection_gate"}
    requested = set(splits) if splits is not None else allowed
    if not requested or not requested.issubset(allowed):
        raise PermissionError(
            "BioASQ task-input adapter exposes train-derived dev_optimize and "
            "selection_gate only; the opened UNITOR dev is not a blind holdout")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    rows = [task for task in _all_tasks() if task["split"] in requested]
    return rows if limit is None else rows[:limit]


def write_dataset() -> Dict[str, Any]:
    """Write deterministic normalized task inputs plus a source/split manifest."""
    from .freeze import require_unfrozen
    require_unfrozen(DATASET)
    destination = ROOT / "data/normalized" / f"{DATASET}.jsonl"
    manifest_path = ROOT / "data/manifests" / f"{DATASET}.json"
    if destination.exists() or manifest_path.exists():
        raise FileExistsError("BioASQ prepared outputs already exist")
    rows = _all_tasks()
    normalized_bytes = jsonl_bytes(rows)
    by_split = {
        name: [task["task_id"] for task in rows if task["split"] == name]
        for name in ("dev_optimize", "selection_gate")
    }
    manifest = {
        "dataset": DATASET,
        "source_dataset": "BioASQ 2025 UniTor answer_generation train_set_answGEN",
        "source_file": str(SOURCE_TRAIN.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256_file(SOURCE_TRAIN),
        "normalized_file": str(destination.relative_to(ROOT)).replace("\\", "/"),
        "normalized_sha256": hashlib.sha256(normalized_bytes).hexdigest(),
        "protocol_version": PROTOCOL_VERSION,
        "input_policy": "question plus unique input_2..input_4 text; strip [BS]/[ES] and outer truncation markers; no retrieval or synthetic negatives",
        "label_policy": "exact yes/no output maps to options [yes, no]",
        "split_algorithm": "sha256(salt|dataset|task_id)[:8] % 100",
        "salt": SPLIT_SALT,
        "ratios": SPLIT_RATIOS,
        "n_questions": len(rows),
        "task_ids_sha256": sha256_obj([task["task_id"] for task in rows]),
        "splits": {
            name: {"n": len(ids), "ids": ids,
                   "ids_sha256": sha256_obj(ids)}
            for name, ids in by_split.items()
        },
        "confirmation_status": "no_final_holdout; train-derived selection_gate is development only",
        "opened_source_dev_status": "disqualified_from_blind_confirmation",
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    commit_new_files([(destination, normalized_bytes), (manifest_path, manifest_bytes)])
    return manifest
