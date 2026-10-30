"""Fixed-evidence HealthFC task adapter for exploratory self-evolution."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from .healthfc_cohort import _components
from .healthfc_source_audit import LABELS
from ..util import sha256_file

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "data/raw/healthfc/Datensatz.csv"
SOURCE_SHA256 = "6b832bd2a73af47f7a8d9dcd3c307defa962cff3746c26b33866170a406721c8"
AUDIT_PATH = ROOT / "artifacts/probe/healthfc_source_audit_20261008.json"
DATASET = "healthfc_fixed_evidence"
PROTOCOL_VERSION = "healthfc-fixed-supplied-sentences-v1"
LABELS_ORDERED = ("SUPPORTED", "NEI", "REFUTED")


def load_tasks(splits: list[str] | tuple[str, ...] | None = None,
               limit: int | None = None) -> list[dict[str, Any]]:
    """Expose all HealthFC rows as optimize-only tasks with fixed evidence."""
    requested = {"dev_optimize"} if splits is None else set(splits)
    if requested != {"dev_optimize"}:
        raise ValueError("HealthFC exposes only its exploratory dev_optimize pool")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if not SOURCE_PATH.is_file() or sha256_file(SOURCE_PATH) != SOURCE_SHA256:
        raise ValueError("HealthFC source CSV is missing or differs from audited bytes")
    with SOURCE_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    component_ids = _components(raw_rows, SOURCE_SHA256)
    rows = []
    for index, (raw, component) in enumerate(zip(raw_rows, component_ids)):
        label = LABELS.get(str(raw.get("label") or "").strip())
        claim = str(raw.get("en_claim") or "").strip()
        evidence = str(raw.get("en_top_sentences") or "").strip()
        if label not in LABELS_ORDERED or not claim or not evidence:
            raise ValueError(f"HealthFC source row {index} is malformed")
        article_hash = hashlib.sha256(
            str(raw.get("url") or "").strip().encode("utf-8")).hexdigest()[:16]
        task_id = f"healthfc:{SOURCE_SHA256[:12]}:r{index:04d}"
        rows.append({
            "task_id": task_id,
            "dataset": DATASET,
            "split": "dev_optimize",
            "task_type": "mcq",
            "domain": "clinical_claim_verification",
            "question": (
                "Using only the supplied evidence sentences, classify the claim.\n"
                f"Claim: {claim}"),
            "options": list(LABELS_ORDERED),
            "correct_index": LABELS_ORDERED.index(label),
            "gold_label": label,
            "context": "",
            "evidence": evidence,
            "sources": [f"healthfc-article:{article_hash}"],
            "group_id": component,
            "metadata": {
                "source_row_index": index,
                "source_archive_sha256": SOURCE_SHA256,
                "article_hash": article_hash,
                "fixed_evidence": True,
                "upstream_label": label,
            },
        })
    return rows if limit is None else rows[:limit]
