"""Fixed-abstract task adapter for the CliniFact exploratory task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .clinifact_cohort import component_ids, read_source_rows
from ..util import sha256_file, sha256_obj


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data/raw/clinifact/train_set.csv"
VALIDATION_PATH = ROOT / "data/raw/clinifact/validation_set.csv"
AUDIT_PATH = ROOT / "artifacts/probe/clinifact_source_audit_20261008.json"
DATASET = "clinifact_fixed_abstract"
PROTOCOL_VERSION = "clinifact-fixed-abstract-grouped-v1"
LABELS = ("EVIDENCE", "INCONCLUSIVE", "NEI")


def load_tasks(splits: list[str] | tuple[str, ...] | None = None,
               limit: int | None = None) -> list[dict[str, Any]]:
    """Expose only paper train/validation rows as optimize-only tasks."""
    requested = {"dev_optimize"} if splits is None else set(splits)
    if requested != {"dev_optimize"}:
        raise ValueError("CliniFact adapter exposes only its exploratory optimize pool")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    paths = {"train": TRAIN_PATH, "validation": VALIDATION_PATH}
    for split, path in paths.items():
        if not path.is_file() or sha256_file(path) != audit["splits"][split]["sha256"]:
            raise ValueError(f"CliniFact {split} file differs from audited source")
    rows = read_source_rows(paths)
    components = component_ids(rows, sha256_obj({
        split: audit["splits"][split]["sha256"] for split in ("train", "validation")}))
    tasks = []
    for row, component in zip(rows, components):
        tasks.append({
            "task_id": row["task_id"],
            "dataset": DATASET,
            "split": "dev_optimize",
            "task_type": "mcq",
            "domain": "clinical_research_claim_verification",
            "question": (
                "Using only the supplied publication abstract, classify the claim according "
                "to whether the abstract reports supporting evidence, an inconclusive result, "
                "or not enough information.\nClaim: " + row["claim"]),
            "options": list(LABELS),
            "correct_index": LABELS.index(row["label"]),
            "gold_label": row["label"],
            "context": "",
            "evidence": f"{row['title']}\n{row['abstract']}",
            "sources": [f"PubMed:{row['pmid']}"],
            "group_id": component,
            "metadata": {
                "source_split": row["source_split"],
                "source_row_index": row["source_row_index"],
                "clinical_trial_id": row["trial_id"],
                "pmid": row["pmid"],
                "source_manifest_sha256": sha256_obj(audit),
                "fixed_evidence": True,
            },
        })
    return tasks if limit is None else tasks[:limit]
