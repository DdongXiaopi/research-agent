"""Independent UniTor BioASQ yes/no protocol candidate (no dev scoring here).

The downloaded source provides truncated, gold-marked contexts. This adapter
removes those marks, creates a fixed pool of own evidence plus public train
distractors, and keeps gold labels separate from model-visible text.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from ..retrieval.gateway import RetrievalGateway
from ..util import sha256_file

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "vendor/BioASQ2025-UNITOR-main/BioASQ2025-UNITOR-main"
          / "data/dataset/answer_generation")
PROTOCOL_VERSION = "bioasq-unitor-gold-plus-train-hardneg-v1"
MIN_OWN_EVIDENCE_CHARS = 40
_WORD = re.compile(r"[a-z0-9][a-z0-9-]{2,}", re.I)


def _clean(text: str) -> str:
    text = str(text or "")
    text = text.replace("[BS]", "").replace("[ES]", "")
    text = re.sub(r"^\s*Abstract truncated here\s*\.\.\.\s*", "", text,
                  flags=re.I)
    text = re.sub(r"\s*\.\.\.\s*Abstract truncated here\s*$", "", text,
                  flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _doc_id(text: str) -> str:
    return "doc::" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _rows(name: str) -> List[Dict[str, Any]]:
    if name not in {"train", "dev"}:
        raise ValueError(name)
    path = SOURCE / f"{name}_set_answGEN.json"
    return copy.deepcopy(_rows_cached(name, sha256_file(path)))


@lru_cache(maxsize=2)
def _rows_cached(name: str, source_sha256: str) -> List[Dict[str, Any]]:
    path = SOURCE / f"{name}_set_answGEN.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _train_passages() -> List[str]:
    return list(_train_passages_cached(sha256_file(SOURCE / "train_set_answGEN.json")))


@lru_cache(maxsize=1)
def _train_passages_cached(source_sha256: str) -> List[str]:
    passages = set()
    for row in _rows_cached("train", source_sha256):
        if row.get("type") != "yesno":
            continue
        for key in ("input_2", "input_3", "input_4"):
            text = _clean(row.get(key, ""))
            if len(text) >= 80:
                passages.add(text)
    return sorted(passages)


def _hard_negatives(question: str, exclude: set[str], k: int = 5,
                    passages: List[str] | None = None) -> List[str]:
    q = set(_WORD.findall(question.casefold()))
    ranked = []
    for text in _train_passages() if passages is None else passages:
        if text in exclude:
            continue
        overlap = len(q & set(_WORD.findall(text.casefold())))
        ranked.append((-overlap, _doc_id(text), text))
    ranked.sort()
    return [x[2] for x in ranked[:k]]


def load_train_question_manifest(limit: int | None = None,
                                 task_ids: set[str] | None = None
                                 ) -> List[Dict[str, Any]]:
    """Question-only inputs for PubMed crawling; no label or supplied evidence."""
    if limit is not None and task_ids is not None:
        raise ValueError("limit and task_ids cannot be combined")
    rows = [r for r in _rows("train") if r.get("type") == "yesno" and
            any(len(_clean(r.get(k, ""))) >= MIN_OWN_EVIDENCE_CHARS
                for k in ("input_2", "input_3", "input_4"))]
    if task_ids is not None:
        rows = [r for r in rows if "bioasq-" + str(r["id"]) in task_ids]
        if len(rows) != len(task_ids):
            raise ValueError("some task_ids are absent or ineligible")
    return [{"task_id": "bioasq-" + str(r["id"]),
             "dataset": "bioasq_unitor", "split": "dev_optimize",
             "question": str(r.get("input_1", "")).strip()}
            for r in rows[:limit]]


def load_tasks(split: str = "train", *, allow_confirmation: bool = False,
               limit: int | None = None,
               task_ids: set[str] | None = None) -> List[Dict[str, Any]]:
    """Load train fixtures; the opened UniTor dev is no longer a holdout."""
    if split == "dev" or allow_confirmation:
        raise PermissionError(
            "BioASQ UniTor dev was exposed during data compatibility audit and "
            "is permanently disqualified from blind confirmation")
    if split not in {"train", "dev"}:
        raise ValueError(split)
    if limit is not None and task_ids is not None:
        raise ValueError("limit and task_ids cannot be combined")
    out = []
    rows = [r for r in _rows(split) if r.get("type") == "yesno" and
            any(len(_clean(r.get(k, ""))) >= MIN_OWN_EVIDENCE_CHARS
                for k in ("input_2", "input_3", "input_4"))]
    if task_ids is not None:
        rows = [r for r in rows if "bioasq-" + str(r["id"]) in task_ids]
        if len(rows) != len(task_ids):
            raise ValueError("some task_ids are absent or ineligible")
    passages = _train_passages()
    for row in rows[:limit]:
        answer = str(row.get("output", "")).strip().casefold()
        if answer not in {"yes", "no"}:
            raise ValueError("invalid yes/no label")
        question = str(row.get("input_1", "")).strip()
        own = []
        seen = set()
        for key in ("input_2", "input_3", "input_4"):
            text = _clean(row.get(key, ""))
            if len(text) < MIN_OWN_EVIDENCE_CHARS or text in seen:
                continue
            seen.add(text)
            own.append(text)
        if not own:
            raise ValueError("yes/no question has no usable evidence text")
        negatives = _hard_negatives(question, seen, passages=passages)
        docs = [{"doc_id": _doc_id(text), "text": text, "is_gold": True}
                for text in own]
        docs.extend({"doc_id": _doc_id(text), "text": text, "is_gold": False}
                    for text in negatives)
        out.append({
            "task_id": "bioasq-" + str(row["id"]),
            "dataset": "bioasq_unitor", "split": "final_holdout" if split == "dev"
            else "dev_optimize",
            "question": question,
            "options": ["yes", "no", "Insufficient information"],
            "correct_index": 0 if answer == "yes" else 1,
            "context": "", "candidate_docs": docs,
        })
    if len({x["task_id"] for x in out}) != len(out):
        raise ValueError("duplicate BioASQ task IDs")
    return out


class BioASQGateway(RetrievalGateway):
    def __init__(self, top_k: int = 3):
        super().__init__(backend="offline-fixture", top_k=top_k)

    def corpus(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [dict(x) for x in task["candidate_docs"]]
