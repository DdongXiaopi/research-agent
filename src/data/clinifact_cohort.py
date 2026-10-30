"""Build disjoint exploratory CliniFact scopes from audited train/validation."""

from __future__ import annotations

import csv
import hashlib
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from .clinifact_source_audit import LABELS
from ..util import sha256_file, sha256_obj


SCOPE_TARGETS = {"mipro_train": 10, "search_validation": 6, "internal_candidate": 6}
LABELS_ORDERED = ("EVIDENCE", "INCONCLUSIVE", "NEI")
SPLITS = ("train", "validation")


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def read_source_rows(paths: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        with paths[split].open("r", encoding="utf-8-sig", newline="") as stream:
            for index, raw in enumerate(csv.DictReader(stream)):
                label = LABELS.get(str(raw.get("label") or "").strip())
                claim = str(raw.get("claim") or "").strip()
                title = str(raw.get("article_title") or "").strip()
                abstract = str(raw.get("article_abstract") or "").strip()
                trial_id = str(raw.get("nctId") or "").strip().upper()
                pmid = str(raw.get("PMID") or "").strip()
                if (label not in LABELS_ORDERED or not claim or not title or not abstract
                        or not trial_id or not pmid):
                    raise ValueError(f"CliniFact {split} row {index} is malformed")
                rows.append({
                    "task_id": f"clinifact:{split}:r{index:04d}",
                    "source_split": split,
                    "source_row_index": index,
                    "source_index": str(raw.get("index") or ""),
                    "trial_id": trial_id,
                    "pmid": pmid,
                    "claim": claim,
                    "title": title,
                    "abstract": abstract,
                    "label": label,
                })
    return rows


def component_ids(rows: list[dict[str, Any]], source_sha256: str) -> list[str]:
    union = _UnionFind(len(rows))
    first: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        claim = " ".join(unicodedata.normalize("NFKC", row["claim"]).casefold().split())
        keys = (("claim", claim), ("trial", row["trial_id"]), ("pmid", row["pmid"]))
        for kind, value in keys:
            if not value:
                raise ValueError(f"CliniFact row {row['task_id']} has empty grouping key")
            previous = first.setdefault((kind, value), index)
            union.union(index, previous)
    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        members[union.find(index)].append(index)
    result = [""] * len(rows)
    for indexes in members.values():
        component = sha256_obj(["clinifact-component-v1", source_sha256, sorted(indexes)])
        for index in indexes:
            result[index] = component
    return result


def build_clinifact_cohort(
    train_path: Path,
    validation_path: Path,
    audit: dict[str, Any],
    history: dict[str, Any],
    *,
    salt: str,
    scope_targets_per_class: dict[str, int] | None = None,
    audit_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic, task/article/claim-disjoint exploratory cohort."""
    paths = {"train": Path(train_path), "validation": Path(validation_path)}
    current_hashes = {split: sha256_file(path) for split, path in paths.items()}
    audited_hashes = {split: audit["splits"][split]["sha256"] for split in SPLITS}
    if current_hashes != audited_hashes:
        raise ValueError("CliniFact source files differ from the audit manifest")
    expected_history_hashes = history.get("source_sha256")
    if expected_history_hashes != current_hashes:
        raise ValueError("CliniFact history scan does not match audited source files")
    if history.get("complete") is not True:
        raise ValueError("CliniFact history scan is incomplete")
    if not salt.strip():
        raise ValueError("CliniFact cohort salt must be non-empty")
    targets = dict(SCOPE_TARGETS if scope_targets_per_class is None
                   else scope_targets_per_class)
    if (set(targets) != set(SCOPE_TARGETS)
            or any(type(value) is not int or value <= 0
                   for value in targets.values())):
        raise ValueError("CliniFact scope targets must be positive integers for all scopes")

    rows = read_source_rows(paths)
    source_sha256 = sha256_obj(current_hashes)
    components = component_ids(rows, source_sha256)
    row_by_ref = {(row["source_split"], row["source_row_index"]): index
                  for index, row in enumerate(rows)}
    history_refs = history.get("matched_row_indexes") or {}
    excluded_indexes: set[int] = set()
    for split in SPLITS:
        for row_index in history_refs.get(split, []):
            ref = (split, int(row_index))
            if ref not in row_by_ref:
                raise ValueError("CliniFact history report contains an unknown row index")
            excluded_indexes.add(row_by_ref[ref])
    excluded_components = {components[index] for index in excluded_indexes}

    rows_by_component_label: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list))
    for index, row in enumerate(rows):
        rows_by_component_label[components[index]][row["label"]].append(index)
    eligible = set(rows_by_component_label) - excluded_components
    chosen: dict[str, Any] | None = None
    for attempt in range(1000):
        current_salt = salt if attempt == 0 else f"{salt}:{attempt}"
        scope_by_component = {}
        for component in eligible:
            rank = int(sha256_obj([current_salt, "scope", component])[:8], 16) % 100
            scope_by_component[component] = (
                "mipro_train" if rank < 65 else
                "search_validation" if rank < 82 else "internal_candidate")
        used: set[str] = set()
        scopes: dict[str, Any] = {}
        failed = False
        for scope, target in targets.items():
            entry = {"task_ids": [], "component_ids": [],
                     "class_counts": {label: 0 for label in LABELS_ORDERED}}
            selected: set[str] = set()
            for label in LABELS_ORDERED:
                candidates = [component for component in eligible
                              if scope_by_component.get(component) == scope
                              and component not in used and component not in selected
                              and label in rows_by_component_label[component]]
                candidates.sort(key=lambda component: sha256_obj([
                    current_salt, scope, label, component]))
                if len(candidates) < target:
                    failed = True
                    break
                for component in candidates[:target]:
                    choices = rows_by_component_label[component][label]
                    row_index = min(choices, key=lambda index: sha256_obj([
                        current_salt, label, rows[index]["task_id"]]))
                    entry["task_ids"].append(rows[row_index]["task_id"])
                    entry["component_ids"].append(component)
                    entry["class_counts"][label] += 1
                    selected.add(component)
            if failed:
                break
            used.update(selected)
            entry["task_ids"].sort()
            entry["component_ids"] = sorted(set(entry["component_ids"]))
            entry["task_count"] = len(entry["task_ids"])
            entry["component_count"] = len(entry["component_ids"])
            scopes[scope] = entry
        if not failed:
            chosen = {
                "salt": current_salt,
                "salt_attempts": attempt + 1,
                "scopes": scopes,
            }
            break
    if chosen is None:
        raise ValueError("could not stratify CliniFact components after 1,000 salts")

    cohort = {
        "schema_version": 1,
        "status": "EXPLORATORY_NOT_PREREGISTERED",
        "dataset": "clinifact_fixed_abstract",
        "dataset_doi": audit["dataset_doi"],
        "dataset_license": audit["dataset_license"],
        "dataset_license_evidence": audit["dataset_license_evidence"],
        "protocol_version": "clinifact-fixed-abstract-grouped-v1",
        "source_hashes": current_hashes,
        "source_manifest_sha256": audit_manifest_sha256 or sha256_obj(audit),
        "history_scan_sha256": sha256_obj(history),
        "source_rows": len(rows),
        "source_task_ids_sha256": sha256_obj(sorted(row["task_id"] for row in rows)),
        "source_component_count": len(set(components)),
        "history_rows_excluded": len(excluded_indexes),
        "history_components_excluded": len(excluded_components),
        "excluded_task_ids_sha256": sha256_obj(sorted(
            rows[index]["task_id"] for index in excluded_indexes)),
        "excluded_component_ids_sha256": sha256_obj(sorted(excluded_components)),
        "grouping_policy": "connected components over normalized claim, NCT ID, and PMID",
        "fixed_evidence": True,
        "retrieval_used_for_evolution": False,
        "candidate_runtime_uses_live_retrieval": False,
        "official_test_exists": True,
        "official_test_downloaded_or_read": False,
        "eligible_for_confirmation": False,
        "project_selection_gate_read": False,
        "selection_or_confirmation_evidence": False,
        "labels": list(LABELS_ORDERED),
        "scope_targets_per_class": targets,
        **chosen,
    }
    cohort["cohort_sha256"] = sha256_obj(cohort)
    return cohort
