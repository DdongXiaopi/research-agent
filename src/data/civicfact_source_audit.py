"""Metadata-only CIViC-Fact intake audit; never normalizes or scores examples.

The upstream build records inherit the source article's license. This module
reports train/dev labels, partition and license metadata, generated/flagged
status, and article partition overlap without returning claim or evidence text.
Gold labels in test and unassigned partitions are never accessed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


PARTITIONS = {"train", "dev", "test", "unassigned"}
LABEL_VISIBLE_PARTITIONS = {"train", "dev"}


def _field(row: dict[str, Any], dotted_name: str) -> Any:
    """Read a flattened upstream key or its equivalent nested object path."""
    if dotted_name in row:
        return row[dotted_name]
    current: Any = row
    for part in dotted_name.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().casefold() in {"true", "1", "yes"}


def _article_key(row: dict[str, Any]) -> str | None:
    for key in ("document.pmcid", "document.pmid", "document.doi"):
        value = str(_field(row, key) or "").strip().casefold()
        if value:
            return f"{key.removeprefix('document.')}:{value}"
    return None


def _overlap_pair_counts(groups: dict[str, set[str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for partitions in groups.values():
        for left, right in combinations(sorted(partitions), 2):
            counts[f"{left}|{right}"] += 1
    return dict(sorted(counts.items()))


def _read_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix.casefold() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at line {line_number}")
            yield row


def audit_civicfact_build(path: Path) -> dict[str, Any]:
    """Report source metadata only; leave label mapping and license policy open."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest_builder = hashlib.sha256()
    source_bytes = 0
    with path.open("rb") as source_stream:
        for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
            source_bytes += len(chunk)
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()

    partition_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    license_counts_by_partition: dict[str, Counter[str]] = defaultdict(Counter)
    generated_counts: Counter[str] = Counter()
    generated_counts_by_partition: dict[str, Counter[str]] = defaultdict(Counter)
    flagged_counts: Counter[str] = Counter()
    flagged_counts_by_partition: dict[str, Counter[str]] = defaultdict(Counter)
    lineage_presence_by_partition: dict[str, Counter[str]] = defaultdict(Counter)
    generated_missing_lineage_by_partition: Counter[str] = Counter()
    article_partitions: dict[str, set[str]] = defaultdict(set)
    claim_hash_partitions: dict[str, set[str]] = defaultdict(set)
    claim_hash_counts: Counter[str] = Counter()
    lineage_partitions: dict[str, set[str]] = defaultdict(set)
    ids: set[str] = set()
    duplicate_ids = 0
    missing_article = 0
    missing_license = 0
    missing_license_by_partition: Counter[str] = Counter()
    total = 0

    for row in _read_rows(path):
        total += 1
        row_id = str(_field(row, "id") or _field(row, "display_id") or "").strip()
        if row_id:
            if row_id in ids:
                duplicate_ids += 1
            ids.add(row_id)

        partition = str(_field(row, "partition") or "").strip().casefold()
        if not partition:
            raise ValueError(f"row {total} is missing partition")
        partition_counts[partition] += 1

        # Track exact normalized claim-text duplicates without returning text.
        # This catches duplicate claims even when article IDs differ, while
        # article and original-example checks below catch broader family leaks.
        claim_text = _field(row, "claim.flat")
        if isinstance(claim_text, str):
            normalized_claim = " ".join(
                unicodedata.normalize("NFKC", claim_text).casefold().split())
            if normalized_claim:
                claim_hash = hashlib.sha256(
                    normalized_claim.encode("utf-8")).hexdigest()
                claim_hash_counts[claim_hash] += 1
                claim_hash_partitions[claim_hash].add(partition)

        original_example = str(_field(row, "original_example") or "").strip()
        lineage_presence_by_partition[partition][
            "present" if original_example else "missing"] += 1
        if not original_example and any(_truth(_field(row, key)) for key in (
            "generated", "claim.generated", "evidence.generated")):
            generated_missing_lineage_by_partition[partition] += 1
        lineage_id = original_example or row_id
        if lineage_id:
            lineage_partitions[lineage_id].add(partition)

        if partition in LABEL_VISIBLE_PARTITIONS:
            label = str(_field(row, "gold_label_name") or "").strip().upper()
            if not label:
                raise ValueError(f"row {total} is missing gold_label_name")
            label_counts[label] += 1

        license_name = str(_field(row, "document.license") or "").strip()
        if license_name:
            license_counts[license_name] += 1
            license_counts_by_partition[partition][license_name] += 1
        else:
            missing_license += 1
            missing_license_by_partition[partition] += 1

        generated = any(_truth(_field(row, key)) for key in (
            "generated", "claim.generated", "evidence.generated"))
        flagged = _truth(_field(row, "flagged")) or _truth(_field(row, "civic.flagged"))
        generated_status = "generated" if generated else "not_generated"
        flagged_status = "flagged" if flagged else "not_flagged"
        generated_counts[generated_status] += 1
        generated_counts_by_partition[partition][generated_status] += 1
        flagged_counts[flagged_status] += 1
        flagged_counts_by_partition[partition][flagged_status] += 1

        article = _article_key(row)
        if article:
            article_partitions[article].add(partition)
        else:
            missing_article += 1

    overlap_articles = {
        article: sorted(partitions)
        for article, partitions in article_partitions.items()
        if len(partitions) > 1
    }
    duplicate_claim_hashes = {
        claim_hash for claim_hash, count in claim_hash_counts.items() if count > 1
    }
    claim_partition_overlaps = sum(
        1 for partitions in claim_hash_partitions.values() if len(partitions) > 1)
    lineage_partition_overlaps = sum(
        1 for partitions in lineage_partitions.values() if len(partitions) > 1)
    return {
        "source_file": path.name,
        "source_bytes": source_bytes,
        "source_sha256": digest,
        "rows": total,
        "partition_counts": dict(sorted(partition_counts.items())),
        "unknown_partitions": sorted(set(partition_counts) - PARTITIONS),
        "label_counts": dict(sorted(label_counts.items())),
        "label_counts_scope": sorted(LABEL_VISIBLE_PARTITIONS),
        "test_label_values_read": False,
        "label_mapping_status": "unresolved_no_automatic_mapping",
        "license_counts": dict(sorted(license_counts.items())),
        "license_counts_by_partition": {
            partition: dict(sorted(counts.items()))
            for partition, counts in sorted(license_counts_by_partition.items())
        },
        "rows_missing_license": missing_license,
        "rows_missing_license_by_partition": dict(sorted(missing_license_by_partition.items())),
        "license_eligibility_status": "unreviewed_per_record_source_license",
        "generated_counts": dict(sorted(generated_counts.items())),
        "generated_counts_by_partition": {
            partition: dict(sorted(counts.items()))
            for partition, counts in sorted(generated_counts_by_partition.items())
        },
        "flagged_counts": dict(sorted(flagged_counts.items())),
        "flagged_counts_by_partition": {
            partition: dict(sorted(counts.items()))
            for partition, counts in sorted(flagged_counts_by_partition.items())
        },
        "original_example_presence_by_partition": {
            partition: dict(sorted(counts.items()))
            for partition, counts in sorted(lineage_presence_by_partition.items())
        },
        "lineage_audit_status": (
            "available" if any(
                counts.get("present", 0) for counts in lineage_presence_by_partition.values())
            else "unavailable_no_original_example_metadata"
        ),
        "generated_rows_missing_original_example_by_partition": dict(
            sorted(generated_missing_lineage_by_partition.items())),
        "unique_article_groups": len(article_partitions),
        "rows_missing_article_identifier": missing_article,
        "article_partition_overlap_count": len(overlap_articles),
        "article_partition_overlaps": overlap_articles,
        "article_overlap_partition_pairs": _overlap_pair_counts(article_partitions),
        "duplicate_claim_text_group_count": len(duplicate_claim_hashes),
        "duplicate_claim_text_row_count": sum(
            claim_hash_counts[claim_hash] - 1 for claim_hash in duplicate_claim_hashes),
        "claim_groups_spanning_partitions": claim_partition_overlaps,
        "claim_overlap_partition_pairs": _overlap_pair_counts(claim_hash_partitions),
        "lineage_groups_spanning_partitions": lineage_partition_overlaps,
        "lineage_overlap_partition_pairs": _overlap_pair_counts(lineage_partitions),
        "duplicate_nonempty_ids": duplicate_ids,
        "source_text_emitted": False,
        "candidate_cohort_created": False,
    }
