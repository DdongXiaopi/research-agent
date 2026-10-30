"""Load labeled SciFact train/dev claims as fixed-evidence task inputs.

This adapter is for exploratory self-evolution experiments only. It exposes no
test or final split, and it supplies only the abstracts already cited by each
claim; it never performs retrieval.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..util import sha256_obj


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ARCHIVE = ROOT / "artifacts/probe/scifact/data.tar.gz"
SOURCE_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
DATASET = "scifact_claim_verification"
PROTOCOL_VERSION = "scifact-fixed-cited-abstract-task-input-v1"
LABELS = ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")
ALLOWED_SOURCE_SPLITS = ("train", "dev")


class SealedSciFactTestError(PermissionError):
    """Raised when a caller tries to expose SciFact's test split."""


def _jsonl_member(archive: tarfile.TarFile, member_name: str) -> list[dict[str, Any]]:
    try:
        member = archive.extractfile(member_name)
    except KeyError as exc:
        raise FileNotFoundError(f"SciFact archive member missing: {member_name}") from exc
    if member is None:
        raise FileNotFoundError(f"SciFact archive member missing: {member_name}")
    try:
        rows = [json.loads(line) for line in member.read().decode("utf-8").splitlines()
                if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid SciFact JSONL member: {member_name}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"SciFact JSONL member contains a non-object row: {member_name}")
    return rows


def _label_for_claim(claim: dict[str, Any]) -> str:
    evidence = claim.get("evidence") or {}
    if not isinstance(evidence, dict):
        raise ValueError(f"SciFact claim {claim.get('id')} has invalid evidence annotations")
    observed = {
        str(item.get("label") or "").strip().upper()
        for annotations in evidence.values()
        for item in annotations
        if isinstance(item, dict) and item.get("label")
    }
    if not observed:
        return "NOT_ENOUGH_INFO"
    if observed == {"SUPPORT"}:
        return "SUPPORT"
    if observed == {"CONTRADICT"}:
        return "CONTRADICT"
    raise ValueError(
        f"SciFact claim {claim.get('id')} has conflicting or unknown labels: "
        f"{sorted(observed)}")


def _component_ids(claims: list[dict[str, Any]]) -> dict[tuple[str, int], str]:
    """Link claims sharing any cited/evidence abstract into one split unit."""
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(node: tuple[str, int]) -> tuple[str, int]:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: tuple[str, int], right: tuple[str, int]) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            if left_root <= right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

    owner: dict[str, tuple[str, int]] = {}
    for claim in claims:
        source_split = str(claim["_source_split"])
        claim_node = ("claim", int(claim["id"]))
        find(claim_node)
        docs = set(map(str, claim.get("cited_doc_ids") or []))
        docs.update(map(str, (claim.get("evidence") or {}).keys()))
        for doc_id in sorted(docs):
            doc_node = ("doc", int(doc_id))
            if doc_id in owner:
                union(claim_node, owner[doc_id])
            else:
                owner[doc_id] = claim_node
            find(doc_node)
            union(claim_node, doc_node)

    buckets: dict[tuple[str, int], dict[str, set[int]]] = defaultdict(
        lambda: {"claim": set(), "doc": set()})
    for node in parent:
        buckets[find(node)][node[0]].add(node[1])
    component_by_node: dict[tuple[str, int], str] = {}
    for nodes in buckets.values():
        component_id = sha256_obj({
            "claims": sorted(nodes["claim"]), "documents": sorted(nodes["doc"]),
        })
        for node_type in ("claim", "doc"):
            for value in nodes[node_type]:
                component_by_node[(node_type, value)] = component_id
    return component_by_node


def load_tasks_from_archive(
    archive_path: Path | str,
    *,
    splits: list[str] | tuple[str, ...] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load labeled claims from train/dev; the test member is never opened."""
    requested = {"dev_optimize"} if splits is None else set(splits)
    if "test" in requested or "final_holdout" in requested:
        raise SealedSciFactTestError(
            "SciFact test split is sealed and unavailable through the evolution adapter")
    if not requested or not requested.issubset({"dev_optimize"}):
        raise ValueError(f"unsupported SciFact task-input split request: {sorted(requested)}")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    source_path = Path(archive_path)
    archive_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if archive_sha256 != SOURCE_SHA256 and source_path == SOURCE_ARCHIVE:
        raise ValueError("SciFact archive checksum differs from the audited official source")
    with tarfile.open(source_path, "r:gz") as archive:
        claims = []
        for source_split in ALLOWED_SOURCE_SPLITS:
            member = f"data/claims_{source_split}.jsonl"
            claims.extend({**row, "_source_split": source_split}
                          for row in _jsonl_member(archive, member))
        corpus_rows = _jsonl_member(archive, "data/corpus.jsonl")

    corpus: dict[str, dict[str, Any]] = {}
    for row in corpus_rows:
        doc_id = str(row.get("doc_id") or "")
        abstract = row.get("abstract")
        if (not doc_id or doc_id in corpus or not isinstance(abstract, list)
                or not all(isinstance(sentence, str) for sentence in abstract)):
            raise ValueError("SciFact corpus contains a duplicate ID or malformed abstract")
        corpus[doc_id] = row

    seen: set[tuple[str, int]] = set()
    for claim in claims:
        claim_id = int(claim.get("id", -1))
        source_split = str(claim["_source_split"])
        key = (source_split, claim_id)
        if claim_id < 0 or key in seen or not str(claim.get("claim") or "").strip():
            raise ValueError(f"SciFact claim has invalid identity or text: {claim_id}")
        seen.add(key)
        cited_ids = claim.get("cited_doc_ids") or []
        if not isinstance(cited_ids, list):
            raise ValueError(f"SciFact claim {claim_id} has invalid cited document IDs")
        docs = sorted({str(value) for value in cited_ids}
                      | {str(value) for value in (claim.get("evidence") or {}).keys()})
        if not docs or any(doc_id not in corpus for doc_id in docs):
            raise ValueError(f"SciFact claim {claim_id} references missing cited abstracts")

    components = _component_ids(claims)
    rows = []
    for claim in claims:
        source_split = str(claim["_source_split"])
        claim_id = int(claim["id"])
        label = _label_for_claim(claim)
        doc_ids = sorted({str(value) for value in (claim.get("cited_doc_ids") or [])}
                         | {str(value) for value in (claim.get("evidence") or {}).keys()})
        evidence_blocks = []
        for doc_id in doc_ids:
            document = corpus[doc_id]
            title = str(document.get("title") or "").strip()
            abstract = " ".join(sentence.strip() for sentence in document["abstract"]
                                 if sentence.strip())
            if not abstract:
                raise ValueError(f"SciFact cited abstract {doc_id} is empty")
            evidence_blocks.append(
                f"Document {doc_id}" + (f" — {title}" if title else "")
                + f"\nAbstract: {abstract}")
        evidence = "\n\n".join(evidence_blocks)
        task_id = f"scifact-{source_split}-{claim_id}"
        rows.append({
            "task_id": task_id,
            "dataset": DATASET,
            "split": "dev_optimize",
            "task_type": "mcq",
            "domain": "scientific_claim_verification",
            "question": (
                "Based only on the supplied cited abstracts, determine whether the "
                "scientific claim is supported, contradicted, or not established. "
                "Claim: " + str(claim["claim"]).strip()),
            "options": list(LABELS),
            "correct_index": LABELS.index(label),
            "gold_label": label,
            "context": "",
            "evidence": evidence,
            "sources": [f"scifact:{doc_id}" for doc_id in doc_ids],
            "group_id": components[("claim", claim_id)],
            "metadata": {
                "claim_id": claim_id,
                "upstream_source_split": source_split,
                "cited_document_ids": doc_ids,
                "evidence_annotation_label": label,
                "source_archive_sha256": archive_sha256,
                "abstract_sha256": sha256_obj([
                    [doc_id, corpus[doc_id].get("title", ""), corpus[doc_id]["abstract"]]
                    for doc_id in doc_ids]),
            },
        })
    rows.sort(key=lambda row: (
        ALLOWED_SOURCE_SPLITS.index(row["metadata"]["upstream_source_split"]),
        int(row["metadata"]["claim_id"]),
    ))
    return rows if limit is None else rows[:limit]


def load_tasks(
    splits: list[str] | tuple[str, ...] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Project loader entry point; only the optimize pool is exposed."""
    return load_tasks_from_archive(SOURCE_ARCHIVE, splits=splits, limit=limit)
