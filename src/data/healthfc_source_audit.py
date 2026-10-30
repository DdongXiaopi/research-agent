"""Metadata-only intake audit for the public HealthFC CSV release."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {"en_claim", "en_top_sentences", "label", "url"}
LABELS = {"0": "SUPPORTED", "1": "NEI", "2": "REFUTED"}


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def audit_healthfc_csv(path: Path) -> dict[str, Any]:
    """Return counts and fingerprints without returning source text or URLs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    digest_builder = hashlib.sha256()
    source_bytes = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            source_bytes += len(chunk)
            digest_builder.update(chunk)

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing_columns = sorted(REQUIRED_COLUMNS - columns)
        if missing_columns:
            raise ValueError(f"missing required columns: {', '.join(missing_columns)}")

        label_counts: Counter[str] = Counter()
        missing_fields: Counter[str] = Counter()
        claim_hash_counts: Counter[str] = Counter()
        article_hash_counts: Counter[str] = Counter()
        rows = 0
        for row in reader:
            rows += 1
            for key in REQUIRED_COLUMNS:
                if not str(row.get(key) or "").strip():
                    missing_fields[key] += 1

            label = str(row.get("label") or "").strip()
            if label not in LABELS:
                label_counts[f"unknown:{label or 'missing'}"] += 1
            else:
                label_counts[LABELS[label]] += 1

            claim = _normalize(str(row.get("en_claim") or ""))
            if claim:
                claim_hash_counts[hashlib.sha256(claim.encode("utf-8")).hexdigest()] += 1

            article_url = _normalize(str(row.get("url") or "").rstrip("/"))
            if article_url:
                article_hash_counts[
                    hashlib.sha256(article_url.encode("utf-8")).hexdigest()
                ] += 1

    duplicate_claim_groups = sum(count > 1 for count in claim_hash_counts.values())
    duplicate_article_groups = sum(count > 1 for count in article_hash_counts.values())
    return {
        "source_file": path.name,
        "source_bytes": source_bytes,
        "source_sha256": digest_builder.hexdigest(),
        "rows": rows,
        "columns": sorted(columns),
        "label_counts": dict(sorted(label_counts.items())),
        "label_mapping": LABELS,
        "missing_required_field_counts": dict(sorted(missing_fields.items())),
        "duplicate_normalized_claim_groups": duplicate_claim_groups,
        "duplicate_normalized_claim_extra_rows": sum(
            count - 1 for count in claim_hash_counts.values() if count > 1),
        "unique_article_groups": len(article_hash_counts),
        "duplicate_article_groups": duplicate_article_groups,
        "duplicate_article_extra_rows": sum(
            count - 1 for count in article_hash_counts.values() if count > 1),
        "claim_or_url_hashes_emitted": False,
        "source_text_emitted": False,
        "official_held_out_test": False,
        "candidate_cohort_created": False,
    }
