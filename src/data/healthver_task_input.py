"""HealthVer claim/abstract task inputs for local self-evolution experiments.

The source is public HealthVer data, re-partitioned by claim/abstract connected
component. Each task carries its paired abstract as fixed evidence; this module
does not retrieve documents or expose an independent confirmation split.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tarfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..util import sha256_file, sha256_obj

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ARCHIVE = ROOT / "data/raw/healthver/data.tar.gz"
SOURCE_SHA256 = "0ce25066f18572af3307438a08d0d318a734d97d1d8c27ca43b51e81ae3d683c"
SPLIT_CANDIDATE = ROOT / "artifacts/probe/healthver_component_split_candidate_20261007.json"
MANIFEST_PATH = ROOT / "data/manifests/healthver_task_input.json"
DATASET = "healthver_entailment_task_input"
PROTOCOL_VERSION = "healthver-component-disjoint-task-input-v1"
LABELS = ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")
SOURCE_SPLITS = ("train", "dev", "test")
SPLIT_NAMES = ("dev_optimize",)


def verify_source_archive() -> None:
    if not SOURCE_ARCHIVE.is_file():
        raise FileNotFoundError(f"HealthVer source archive missing: {SOURCE_ARCHIVE}")
    if sha256_file(SOURCE_ARCHIVE) != SOURCE_SHA256:
        raise ValueError("HealthVer source archive hash differs from the audited source")


def component_id(claim_ids: list[int], abstract_ids: list[int]) -> str:
    """Return the stable component identity used by the audited split candidate."""
    canonical = json.dumps(
        {"claim_ids": sorted(int(value) for value in claim_ids),
         "abstract_ids": sorted(int(value) for value in abstract_ids)},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(self, node: tuple[str, int]) -> tuple[str, int]:
        self.parent.setdefault(node, node)
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, left: tuple[str, int], right: tuple[str, int]) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            if left_root <= right_root:
                self.parent[right_root] = left_root
            else:
                self.parent[left_root] = right_root


def _jsonl_member(archive: tarfile.TarFile, name: str) -> list[dict[str, Any]]:
    member = archive.extractfile(name)
    if member is None:
        raise FileNotFoundError(f"HealthVer archive member missing: {name}")
    rows = [json.loads(line) for line in member.read().decode("utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"HealthVer source member is not a JSONL object list: {name}")
    return rows


def _load_all_tasks() -> list[dict[str, Any]]:
    """Load a content-bound copy of the component-assigned public pairs."""
    identity = (sha256_file(SOURCE_ARCHIVE), sha256_file(SPLIT_CANDIDATE))
    return copy.deepcopy(_load_all_tasks_cached(identity))


@lru_cache(maxsize=1)
def _load_all_tasks_cached(identity: tuple[str, str]) -> list[dict[str, Any]]:
    verify_source_archive()
    if not SPLIT_CANDIDATE.is_file():
        raise FileNotFoundError(f"HealthVer component split candidate missing: {SPLIT_CANDIDATE}")
    split_doc = json.loads(SPLIT_CANDIDATE.read_text(encoding="utf-8"))
    split_hash = str(split_doc.get("manifest_sha256") or "")
    unhashed_split = dict(split_doc)
    unhashed_split.pop("manifest_sha256", None)
    if not split_hash or sha256_obj(unhashed_split) != split_hash:
        raise ValueError("HealthVer component split candidate checksum mismatch")
    if split_doc.get("source_sha256") != SOURCE_SHA256:
        raise ValueError("HealthVer split candidate is bound to a different source archive")

    with tarfile.open(SOURCE_ARCHIVE, mode="r:gz") as archive:
        corpus_rows = _jsonl_member(archive, "data/healthver/corpus.jsonl")
        claims_by_source = {
            source_split: _jsonl_member(
                archive, f"data/healthver/claims_{source_split}.jsonl")
            for source_split in SOURCE_SPLITS
        }

    corpus: dict[int, dict[str, Any]] = {}
    for row in corpus_rows:
        doc_id = int(row["doc_id"])
        if doc_id in corpus:
            raise ValueError("duplicate HealthVer abstract ID")
        abstract = row.get("abstract")
        if not isinstance(abstract, list) or not all(isinstance(s, str) for s in abstract):
            raise ValueError(f"invalid abstract sentence list for document {doc_id}")
        corpus[doc_id] = row

    raw_pairs: list[dict[str, Any]] = []
    union_find = _UnionFind()
    seen_claim_ids: set[int] = set()
    for source_split, claim_rows in claims_by_source.items():
        for claim_row in claim_rows:
            claim_id = int(claim_row["id"])
            if claim_id in seen_claim_ids:
                raise ValueError(f"duplicate HealthVer claim ID: {claim_id}")
            seen_claim_ids.add(claim_id)
            claim = str(claim_row.get("claim") or "").strip()
            if not claim:
                raise ValueError(f"empty HealthVer claim: {claim_id}")
            evidence_map = claim_row.get("evidence") or {}
            doc_ids = claim_row.get("doc_ids") or []
            if not isinstance(doc_ids, list):
                raise ValueError(f"invalid doc_ids for HealthVer claim {claim_id}")
            for raw_doc_id in doc_ids:
                doc_id = int(raw_doc_id)
                document = corpus.get(doc_id)
                if document is None:
                    raise ValueError(f"claim {claim_id} references missing abstract {doc_id}")
                annotation_rows = evidence_map.get(str(doc_id)) or []
                labels = {str(item.get("label") or "").strip().upper()
                          for item in annotation_rows if item.get("label")}
                labels.discard("")
                if not labels:
                    label = "NOT_ENOUGH_INFO"
                elif labels == {"SUPPORT"}:
                    label = "SUPPORT"
                elif labels in ({"CONTRADICT"}, {"REFUTE"}, {"REFUTED"}):
                    label = "CONTRADICT"
                else:
                    raise ValueError(
                        f"conflicting/unknown HealthVer labels for claim {claim_id}, abstract {doc_id}: {sorted(labels)}")
                union_find.union(("claim", claim_id), ("abstract", doc_id))
                abstract_text = " ".join(str(sentence).strip()
                                         for sentence in document["abstract"]
                                         if str(sentence).strip())
                if not abstract_text:
                    raise ValueError(f"empty HealthVer abstract: {doc_id}")
                raw_pairs.append({
                    "claim_id": claim_id, "abstract_id": doc_id,
                    "source_split": source_split, "claim": claim,
                    "title": str(document.get("title") or "").strip(),
                    "abstract": abstract_text, "label": label,
                })

    nodes_by_root: dict[tuple[str, int], dict[str, set[int]]] = {}
    for node in list(union_find.parent):
        bucket = nodes_by_root.setdefault(union_find.find(node),
                                          {"claim": set(), "abstract": set()})
        bucket[node[0]].add(node[1])
    component_by_claim: dict[int, str] = {}
    component_by_abstract: dict[int, str] = {}
    for bucket in nodes_by_root.values():
        cid = component_id(sorted(bucket["claim"]), sorted(bucket["abstract"]))
        for value in bucket["claim"]:
            component_by_claim[value] = cid
        for value in bucket["abstract"]:
            component_by_abstract[value] = cid

    expected_components = {
        component
        for scope in (split_doc.get("scopes") or {}).values()
        for component in scope.get("component_ids", [])
    }
    if len(nodes_by_root) != int(split_doc.get("source_counts", {}).get("components", -1)):
        raise ValueError("HealthVer connected-component count differs from audited split")
    observed_components = set(component_by_claim.values())
    if observed_components != expected_components:
        raise ValueError("HealthVer component identities differ from audited split candidate")

    task_rows: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    for pair in raw_pairs:
        task_id = f"healthver-claim-{pair['claim_id']}-abstract-{pair['abstract_id']}"
        if task_id in seen_task_ids:
            raise ValueError(f"duplicate HealthVer claim-abstract pair: {task_id}")
        seen_task_ids.add(task_id)
        title = pair["title"]
        evidence = (f"Title: {title}\nAbstract: {pair['abstract']}" if title
                    else pair["abstract"])
        task_rows.append({
            "task_id": task_id,
            "dataset": DATASET,
            "split": "dev_optimize",
            "task_type": "mcq",
            "domain": "biomedical",
            "question": pair["claim"],
            "options": list(LABELS),
            "correct_index": LABELS.index(pair["label"]),
            "context": "",
            "evidence": evidence,
            "sources": [],
            "group_id": component_by_claim[pair["claim_id"]],
            "metadata": {
                "claim_id": pair["claim_id"],
                "abstract_id": pair["abstract_id"],
                "upstream_source_split": pair["source_split"],
                "abstract_sha256": hashlib.sha256(pair["abstract"].encode("utf-8")).hexdigest(),
            },
        })
    expected_pairs = int(split_doc.get("source_counts", {}).get("pairs", -1))
    if len(task_rows) != expected_pairs:
        raise ValueError("HealthVer pair count differs from audited source split")

    candidate_pairs: dict[str, set[tuple[int, int]]] = {}
    candidate_components: dict[str, set[str]] = {}
    for scope_name, scope in (split_doc.get("scopes") or {}).items():
        candidate_pairs[scope_name] = {
            (int(pair[0]), int(pair[1])) for pair in scope.get("pair_ids", [])
        }
        candidate_components[scope_name] = {
            str(component) for component in scope.get("component_ids", [])
        }
        if len(candidate_pairs[scope_name]) != int(scope.get("n_pairs", -1)):
            raise ValueError(f"invalid pair count in HealthVer split scope {scope_name}")
    assigned_by_pair: dict[tuple[int, int], str] = {}
    component_scope: dict[str, str] = {}
    for scope_name, pairs in candidate_pairs.items():
        for pair in pairs:
            if pair in assigned_by_pair:
                raise ValueError("HealthVer pair appears in multiple component scopes")
            assigned_by_pair[pair] = scope_name
        for cid in candidate_components[scope_name]:
            if cid in component_scope:
                raise ValueError("HealthVer component appears in multiple candidate scopes")
            component_scope[cid] = scope_name
    if len(assigned_by_pair) != len(raw_pairs):
        raise ValueError("HealthVer split candidate does not cover every claim-abstract pair")
    if set(assigned_by_pair) != {(p["claim_id"], p["abstract_id"]) for p in raw_pairs}:
        raise ValueError("HealthVer split pair IDs differ from source pairs")
    for pair, scope_name in assigned_by_pair.items():
        cid = component_by_claim[pair[0]]
        if cid not in candidate_components[scope_name] or component_by_abstract[pair[1]] != cid:
            raise ValueError("HealthVer component split crosses claim/abstract scope")
    for row, pair in zip(task_rows, raw_pairs):
        row["cohort_candidate_scope"] = assigned_by_pair[
            (pair["claim_id"], pair["abstract_id"])]
    return task_rows


def load_tasks(splits: list[str] | tuple[str, ...] | None = None,
               limit: int | None = None) -> list[dict[str, Any]]:
    """Load the public pair pool; no blind confirmation split is exposed."""
    requested = set(splits) if splits is not None else {"dev_optimize"}
    if requested != {"dev_optimize"}:
        raise PermissionError(
            "HealthVer exposes the component-assigned development pool only; "
            "it has no blind confirmation split")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    rows = _load_all_tasks()
    return rows if limit is None else rows[:limit]


def load_manifest() -> dict[str, Any]:
    """Return a small provenance manifest for the non-confirmatory data pool."""
    verify_source_archive()
    tasks = load_tasks(["dev_optimize"])
    ids = [row["task_id"] for row in tasks]
    split_doc = json.loads(SPLIT_CANDIDATE.read_text(encoding="utf-8"))
    manifest_sha = str(split_doc["manifest_sha256"])
    counts = {label: 0 for label in LABELS}
    for row in tasks:
        counts[str(row["options"][row["correct_index"]])] += 1
    return {
        "dataset": DATASET,
        "protocol_version": PROTOCOL_VERSION,
        "source_archive": "data/raw/healthver/data.tar.gz",
        "source_archive_sha256": SOURCE_SHA256,
        "source_dataset_card": "https://huggingface.co/datasets/dwadden/healthver_entailment",
        "source_repository": "https://github.com/sarrouti/HealthVer",
        "source_license_status": {
            "huggingface_card_declared_license": "cc-by-nc-2.0",
            "upstream_repository_data_license": "not_stated_in_repository_readme",
            "use_scope": "local_noncommercial_exploration_only_pending_upstream_rights_review",
        },
        "source_split_policy": (
            "Public upstream train/dev/test pairs are combined and reassigned by "
            "claim-abstract connected component; original split is provenance only."),
        "source_component_split_candidate_sha256": manifest_sha,
        "source_components": int(split_doc["source_counts"]["components"]),
        "source_pair_count": int(split_doc["source_counts"]["pairs"]),
        "label_policy": "use per-pair evidence label; missing evidence label maps to NOT_ENOUGH_INFO",
        "input_policy": "claim plus exactly its paired source abstract/title as fixed task evidence; no retrieval",
        "labels": list(LABELS),
        "candidate_runtime_uses_live_retrieval": False,
        "eligible_for_confirmation": False,
        "selection_or_confirmation_evidence": False,
        "source_upstream_test_materialized": True,
        "source_test_is_blind_confirmation": False,
        "n_tasks": len(tasks),
        "task_ids_sha256": sha256_obj(ids),
        "label_counts": counts,
        "splits": {
            "dev_optimize": {"n": len(ids), "ids": ids,
                             "ids_sha256": sha256_obj(ids)},
        },
    }
