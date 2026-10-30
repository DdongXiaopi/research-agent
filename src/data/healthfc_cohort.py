"""Build group-disjoint exploratory HealthFC scopes from fixed evidence."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from .healthfc_source_audit import LABELS, _normalize
from ..util import sha256_obj


SCOPE_TARGETS = {"mipro_train": 10, "search_validation": 6, "internal_candidate": 6}
LABEL_NAMES = ("SUPPORTED", "NEI", "REFUTED")
SOURCE_AUDIT_PATH = (Path(__file__).resolve().parents[2]
                     / "artifacts/probe/healthfc_source_audit_20261008.json")


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"en_claim", "en_top_sentences", "label", "url"}
    if not rows or not required.issubset(set(rows[0])):
        raise ValueError("invalid HealthFC input schema or empty source")
    return rows


def _components(rows: list[dict[str, str]], source_sha256: str) -> list[str]:
    union_find = _UnionFind(len(rows))
    first_for_claim: dict[str, int] = {}
    first_for_article: dict[str, int] = {}
    for index, row in enumerate(rows):
        claim = _normalize(str(row.get("en_claim") or ""))
        article = _normalize(str(row.get("url") or "").rstrip("/"))
        if not claim or not article:
            raise ValueError(f"HealthFC row {index} lacks claim or article URL")
        for key, index_map in ((claim, first_for_claim), (article, first_for_article)):
            previous = index_map.setdefault(key, index)
            union_find.union(index, previous)
    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        members[union_find.find(index)].append(index)
    component_by_row: dict[int, str] = {}
    for indexes in members.values():
        component = sha256_obj(["healthfc-component-v1", source_sha256, sorted(indexes)])
        for index in indexes:
            component_by_row[index] = component
    return [component_by_row[index] for index in range(len(rows))]


def build_healthfc_cohort(
    csv_path: Path,
    history_report: dict[str, Any],
    *,
    salt: str = "healthfc-mipro-exploration-20261008-v1",
) -> dict[str, Any]:
    """Create a provisional cohort only after a complete matched-source scan."""
    csv_path = Path(csv_path)
    if not salt.strip():
        raise ValueError("HealthFC cohort salt must be non-empty")
    source_bytes = csv_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if history_report.get("source_sha256") != source_sha256:
        raise ValueError("HealthFC history report is for a different source file")
    if history_report.get("complete") is not True:
        raise ValueError("HealthFC history scan is incomplete")

    rows = _read_rows(csv_path)
    components = _components(rows, source_sha256)
    rows_by_component_label: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list))
    for index, row in enumerate(rows):
        label = LABELS.get(str(row.get("label") or "").strip())
        if label not in LABEL_NAMES:
            raise ValueError(f"HealthFC row {index} has unsupported label")
        rows_by_component_label[components[index]][label].append(index)

    hit_indexes = history_report.get("matched_row_indexes") or []
    excluded_indexes = {int(index) for index in hit_indexes}
    if any(index < 0 or index >= len(rows) for index in excluded_indexes):
        raise ValueError("HealthFC history report contains an out-of-range row index")
    excluded_components = {components[index] for index in excluded_indexes}
    eligible_components = set(rows_by_component_label) - excluded_components

    cohort = None
    attempts = 0
    for attempt in range(1000):
        current_salt = salt if attempt == 0 else f"{salt}:{attempt}"
        component_scope = {}
        for component in eligible_components:
            rank = int(sha256_obj([current_salt, "component-scope", component])[:8], 16) % 100
            component_scope[component] = (
                "mipro_train" if rank < 70 else
                "search_validation" if rank < 85 else
                "internal_candidate"
            )

        scopes: dict[str, Any] = {}
        selected_groups_by_scope: dict[str, set[str]] = defaultdict(set)
        failed = False
        for scope, target in SCOPE_TARGETS.items():
            entry = {
                "task_ids": [], "component_ids": [],
                "class_counts": {label: 0 for label in LABEL_NAMES},
            }
            selected_components: set[str] = set()
            for label in LABEL_NAMES:
                candidates = [
                    component for component, by_label in rows_by_component_label.items()
                    if component_scope.get(component) == scope
                    and component not in selected_groups_by_scope[scope]
                    and component not in selected_components
                    and label in by_label
                ]
                candidates.sort(key=lambda component: sha256_obj([
                    current_salt, scope, label, component]))
                if len(candidates) < target:
                    failed = True
                    break
                for component in candidates[:target]:
                    row_options = rows_by_component_label[component][label]
                    row_index = min(row_options, key=lambda index: sha256_obj([
                        current_salt, label, component, index]))
                    task_id = f"healthfc:{source_sha256[:12]}:r{row_index:04d}"
                    entry["task_ids"].append(task_id)
                    entry["component_ids"].append(component)
                    entry["class_counts"][label] += 1
                    selected_components.add(component)
            if failed:
                break
            entry["task_ids"].sort()
            entry["component_ids"] = sorted(set(entry["component_ids"]))
            entry["task_count"] = len(entry["task_ids"])
            entry["component_count"] = len(entry["component_ids"])
            entry["article_count"] = entry["component_count"]
            scopes[scope] = entry
            selected_groups_by_scope[scope].update(selected_components)
        if failed:
            continue
        cohort = {
            "schema_version": 1,
            "status": "EXPLORATORY_CANDIDATE_NOT_PREREGISTERED",
            "dataset": "healthfc_fixed_evidence",
            "protocol_version": "fixed-evidence-grouped-v1",
            "source_archive_sha256": source_sha256,
            "source_manifest_sha256": hashlib.sha256(
                SOURCE_AUDIT_PATH.read_bytes()).hexdigest(),
            "source_row_count": len(rows),
            "source_component_count": len(set(components)),
            "source_task_ids_sha256": sha256_obj([
                f"healthfc:{source_sha256[:12]}:r{index:04d}"
                for index in range(len(rows))]),
            "source_split_policy": "all public rows are exploratory; re-split by claim/article connected component",
            "grouping_policy": "normalized English claim and normalized source URL connected components",
            "fixed_task_evidence": True,
            "candidate_runtime_uses_live_retrieval": False,
            "official_test_read": False,
            "official_test_exists": False,
            "source_cv_reused": False,
            "history_scan_complete": True,
            "history_scan_sha256": sha256_obj(history_report),
            "history_rows_excluded": len(excluded_indexes),
            "history_components_excluded": len(excluded_components),
            "eligible_for_confirmation": False,
            "selection_or_confirmation_evidence": False,
            "project_selection_gate_or_confirmation_read": False,
            "license": "CC-BY-NC-ND-4.0",
            "api_transfer_approved": False,
            "salt": current_salt,
            "salt_attempts": attempt + 1,
            "excluded_task_count": len(excluded_indexes),
            "excluded_task_ids_sha256": sha256_obj(sorted(
                f"healthfc:{source_sha256[:12]}:r{index:04d}"
                for index in excluded_indexes)),
            "excluded_component_count": len(excluded_components),
            "excluded_component_ids_sha256": sha256_obj(sorted(excluded_components)),
            "labels": list(LABEL_NAMES),
            "scope_targets_per_class": SCOPE_TARGETS,
            "scopes": scopes,
        }
        break
    if cohort is None:
        raise ValueError("could not form disjoint HealthFC scopes after 1000 salts")

    # Recompute after loop because the chosen salt is included in the manifest.
    cohort["cohort_sha256"] = sha256_obj(cohort)
    return cohort
