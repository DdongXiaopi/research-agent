"""Exploratory native three-way PubMedQA tasks.

The legacy PubMedQA adapter drops ``maybe`` and exposes an injected abstention
option. This isolated adapter keeps the upstream yes/no/maybe labels, reuses
yes/no questions only when they belong to the legacy development split, and
exposes no selection gate or final holdout. It is for development diagnostics,
not confirmation or release decisions.
"""

from __future__ import annotations

import math
import numbers
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from ..util import (ensure_dir, read_json, read_jsonl, sha256_file, sha256_obj,
                    write_json, write_jsonl)

ROOT = Path(__file__).resolve().parents[2]
DATASET = "pubmedqa_threeway_probe"
PROTOCOL_VERSION = "pubmedqa-native-threeway-dev-only-v1"
SOURCE_PARQUET = ROOT / "vendor/PubMedQA/train-00000-of-00001.parquet"
LEGACY_MANIFEST = ROOT / "data/manifests/pubmedqa.json"
NORMALIZED_PATH = ROOT / "data/normalized" / f"{DATASET}.jsonl"
MANIFEST_PATH = ROOT / "data/manifests" / f"{DATASET}.json"
OPTIONS = ("yes", "no", "maybe")
SPLIT = "dev_optimize"


def _qid(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        try:
            if math.isfinite(float(value)) and float(value).is_integer():
                return str(int(value))
        except (TypeError, ValueError, OverflowError):
            pass
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _evidence_texts(context: Any) -> List[str]:
    raw: Any = context
    if isinstance(context, Mapping):
        raw = context.get("contexts", [])
    if raw is None:
        return []
    if isinstance(raw, str):
        items: Iterable[Any] = [raw]
    else:
        try:
            items = list(raw)
        except TypeError:
            items = [raw]
    out = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if text and text.casefold() not in {"nan", "none"}:
            out.append(text)
    return out


def normalize_rows(rows: Iterable[Mapping[str, Any]],
                   legacy_dev_ids: Sequence[str]) -> List[Dict[str, Any]]:
    """Normalize labeled PQA rows without touching excluded yes/no task fields."""
    allowed_yes_no = {str(task_id) for task_id in legacy_dev_ids}
    out: List[Dict[str, Any]] = []
    seen_ids = set()
    for row in rows:
        qid = _qid(row.get("pubid"))
        label = str(row.get("final_decision", "")).strip().casefold()
        if label not in OPTIONS:
            raise ValueError(f"unexpected PubMedQA label for {qid or '<missing-id>'}: {label!r}")
        # The legacy manifest contains yes/no dev, gate, and final IDs. Only
        # allow its dev IDs here; maybe rows were excluded before that split.
        if label in {"yes", "no"} and qid not in allowed_yes_no:
            continue
        if not qid:
            raise ValueError("PubMedQA row is missing a valid pubid")
        task_id = f"{DATASET}-{qid}"
        if task_id in seen_ids:
            raise ValueError(f"duplicate PubMedQA three-way task ID: {task_id}")
        seen_ids.add(task_id)
        question = str(row.get("question", "") or "").strip()
        evidence_parts = _evidence_texts(row.get("context"))
        if not question or not evidence_parts:
            raise ValueError(f"PubMedQA task {qid} is missing question or context")
        out.append({
            "task_id": task_id,
            "dataset": DATASET,
            "split": SPLIT,
            "task_type": "mcq",
            "domain": "biomedical",
            "question": question,
            "options": list(OPTIONS),
            "correct_index": OPTIONS.index(label),
            "context": "",
            "evidence": "\n\n".join(evidence_parts),
            "sources": [f"https://pubmed.ncbi.nlm.nih.gov/{qid}/"],
        })
    out.sort(key=lambda task: task["task_id"])
    return out


def load_tasks(splits: Sequence[str] | None = None,
               limit: int | None = None) -> List[Dict[str, Any]]:
    """Load only visible development tasks; fail closed for gate/final requests."""
    requested = {SPLIT} if splits is None else set(splits)
    if requested != {SPLIT}:
        raise PermissionError(
            f"{DATASET} is development-only and exposes no selection_gate/final_holdout")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    rows = [row for row in read_jsonl(NORMALIZED_PATH)
            if row.get("split") == SPLIT]
    return rows if limit is None else rows[:limit]


def write_dataset(source_rows: Iterable[Mapping[str, Any]],
                  legacy_manifest: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Write a dev-only artifact and manifest; never create a gate/final split."""
    legacy = (dict(legacy_manifest) if legacy_manifest is not None
              else read_json(LEGACY_MANIFEST))
    legacy_dev_ids = (legacy.get("splits", {}).get("dev_optimize", {})
                      .get("ids", []))
    rows = normalize_rows(source_rows, legacy_dev_ids)
    if not rows:
        raise ValueError("no eligible PubMedQA development tasks were normalized")
    if any(row.get("split") != SPLIT or row.get("dataset") != DATASET
           for row in rows):
        raise RuntimeError("three-way probe rows escaped their development-only contract")

    from .freeze import require_unfrozen
    from ..util import commit_new_files, jsonl_bytes
    import hashlib
    import json
    require_unfrozen(DATASET)
    if NORMALIZED_PATH.exists() or MANIFEST_PATH.exists():
        raise FileExistsError("PubMedQA probe outputs already exist")
    normalized_bytes = jsonl_bytes(rows)
    labels = {label: sum(row["options"][row["correct_index"]] == label
                         for row in rows) for label in OPTIONS}
    manifest = {
        "dataset": DATASET,
        "protocol_version": PROTOCOL_VERSION,
        "source_dataset": "PubMedQA PQA-L labeled training parquet",
        "source_file": SOURCE_PARQUET.relative_to(ROOT).as_posix(),
        "source_sha256": sha256_file(SOURCE_PARQUET),
        "normalized_file": NORMALIZED_PATH.relative_to(ROOT).as_posix(),
        "normalized_sha256": hashlib.sha256(normalized_bytes).hexdigest(),
        "legacy_manifest_sha256": sha256_file(LEGACY_MANIFEST),
        "label_policy": "retain native yes/no/maybe labels with fixed option order; do not inject Insufficient",
        "yes_no_scope": "only question IDs already assigned to legacy pubmedqa/dev_optimize",
        "maybe_scope": "native maybe-labeled rows excluded by the legacy binary adapter",
        "split_policy": "all rows are dev_optimize; no selection_gate or final_holdout is defined",
        "eligible_for_confirmation": False,
        "confirmation_status": "exploratory development only; source/domain overlaps legacy PubMedQA",
        "n_questions": len(rows),
        "label_counts": labels,
        "task_ids_sha256": sha256_obj([row["task_id"] for row in rows]),
        "splits": {SPLIT: {
            "n": len(rows), "ids": [row["task_id"] for row in rows],
            "ids_sha256": sha256_obj([row["task_id"] for row in rows])}},
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    commit_new_files([(NORMALIZED_PATH, normalized_bytes), (MANIFEST_PATH, manifest_bytes)])
    return manifest
