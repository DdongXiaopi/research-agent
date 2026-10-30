"""Shared integrity checks for the native three-way PubMedQA experiments."""

from __future__ import annotations

from pathlib import Path
from collections import Counter

from src.data.pubmedqa_threeway_probe import DATASET, MANIFEST_PATH, NORMALIZED_PATH
from src.util import sha256_file


def gold_label_counts(rows: list[dict]) -> dict[str, int]:
    labels = []
    for row in rows:
        options = row.get("options") or []
        try:
            label = str(options[int(row["correct_index"])]).strip().casefold()
        except (IndexError, KeyError, TypeError, ValueError):
            raise ValueError("cannot count gold labels for a malformed task")
        if not label:
            raise ValueError("cannot count an empty gold label")
        labels.append(label)
    return dict(sorted(Counter(labels).items()))


def verify_dataset_snapshot(report: dict, *, manifest_path: Path = MANIFEST_PATH,
                            normalized_path: Path = NORMALIZED_PATH) -> tuple[str, str]:
    """Verify the report's frozen source hashes against the current local data."""
    if str(report.get("dataset") or "") != DATASET:
        raise ValueError(f"expected dataset {DATASET}")
    expected_manifest = str(report.get("dataset_manifest_sha256") or "")
    expected_normalized = str(report.get("dataset_normalized_sha256") or "")
    if not expected_manifest or not expected_normalized:
        raise ValueError("source report is missing frozen dataset hashes")
    actual_manifest = sha256_file(manifest_path)
    actual_normalized = sha256_file(normalized_path)
    if expected_manifest != actual_manifest:
        raise ValueError("PubMedQA manifest changed since the cohort was frozen")
    if expected_normalized != actual_normalized:
        raise ValueError("PubMedQA normalized tasks changed since the cohort was frozen")
    return actual_manifest, actual_normalized
