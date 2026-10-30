"""Metadata-only audit for the CliniFact Figshare train/validation files.

The public test file is intentionally not part of this API. The source's numeric
label mapping is anchored to the paper's published train/validation counts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DATASET_DOI = "10.6084/m9.figshare.27188109.v1"
ROOT = Path(__file__).resolve().parents[2]
LICENSE_METADATA_PATH = ROOT / "data/manifests/clinifact_figshare_license.json"
FIGSHARE_RESPONSE_RELATIVE_PATH = "data/manifests/clinifact_figshare_article_27188109.json"
FIGSHARE_RESPONSE_PATH = ROOT / FIGSHARE_RESPONSE_RELATIVE_PATH
LABELS = {"0": "INCONCLUSIVE", "1": "EVIDENCE", "2": "NEI"}
PUBLISHED_SPLIT_LABEL_COUNTS = {
    "train": {"EVIDENCE": 366, "INCONCLUSIVE": 262, "NEI": 632},
    "validation": {"EVIDENCE": 83, "INCONCLUSIVE": 67, "NEI": 166},
}
REQUIRED_COLUMNS = {"index", "nctId", "claim", "PMID", "article_abstract", "label"}
EXPECTED_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"


def _read_license_evidence() -> dict[str, Any]:
    """Load a pinned official Figshare metadata snapshot; never infer a license."""
    try:
        source = json.loads(LICENSE_METADATA_PATH.read_text(encoding="utf-8"))
        response = json.loads(FIGSHARE_RESPONSE_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("CliniFact Figshare license source record is missing or invalid") from exc
    files = set(source.get("files") or [])
    response_hash = str(source.get("metadata_response_sha256") or "")
    observed_hash = hashlib.sha256(FIGSHARE_RESPONSE_PATH.read_bytes()).hexdigest()
    response_files = {str(item.get("name") or "") for item in response.get("files") or []}
    if (source.get("schema") != "figshare-license-source-record-v1"
            or source.get("doi") != DATASET_DOI
            or source.get("license_name") != "CC BY"
            or source.get("license_url") != EXPECTED_LICENSE_URL
            or source.get("metadata_api_url") !=
            "https://api.figshare.com/v2/articles/27188109"
            or source.get("metadata_snapshot_file") != FIGSHARE_RESPONSE_RELATIVE_PATH
            or source.get("record_url") !=
            "https://springernature.figshare.com/articles/dataset/"
            "A_dataset_for_evaluating_clinical_research_claims_in_large_language_models/27188109"
            or not {"train_set.csv", "validation_set.csv", "test_set.csv"}.issubset(files)
            or response.get("doi") != DATASET_DOI
            or response.get("url_public_html") != source.get("record_url")
            or (response.get("license") or {}).get("name") != "CC BY"
            or (response.get("license") or {}).get("url") != EXPECTED_LICENSE_URL
            or not {"train_set.csv", "validation_set.csv", "test_set.csv"}.issubset(response_files)
            or len(response_hash) != 64
            or any(char not in "0123456789abcdef" for char in response_hash)):
        raise ValueError("CliniFact Figshare license source record failed validation")
    if observed_hash != response_hash:
        raise ValueError("CliniFact Figshare metadata snapshot hash mismatch")
    return {
        "metadata_api_url": source["metadata_api_url"],
        "record_url": source["record_url"],
        "metadata_snapshot_file": source["metadata_snapshot_file"],
        "metadata_response_sha256": response_hash,
        "retrieved_at": source["retrieved_at"],
        "license_name": source["license_name"],
        "license_url": source["license_url"],
    }


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _read_split(path: Path, split: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size, digest = _sha256(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing_columns = sorted(REQUIRED_COLUMNS - columns)
        if missing_columns:
            raise ValueError(f"{split} is missing required columns: {missing_columns}")
        rows = list(reader)

    labels: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    for index, row in enumerate(rows):
        for field in REQUIRED_COLUMNS:
            if not str(row.get(field) or "").strip():
                missing[field] += 1
        raw_label = str(row.get("label") or "").strip()
        label = LABELS.get(raw_label)
        if label is None:
            raise ValueError(f"{split} row {index} has unsupported label value")
        labels[label] += 1
    if missing:
        raise ValueError(f"{split} has missing required field values: {dict(missing)}")
    expected = PUBLISHED_SPLIT_LABEL_COUNTS[split]
    if dict(labels) != expected:
        raise ValueError(
            f"{split} label mapping/counts differ from published table: {dict(labels)}")
    return rows, {
        "file": path.name,
        "bytes": size,
        "sha256": digest,
        "rows": len(rows),
        "columns": len(columns),
        "label_counts": dict(sorted(labels.items())),
        "missing_required_field_counts": dict(missing),
    }


def audit_clinifact_source(train_path: Path, validation_path: Path) -> dict[str, Any]:
    """Audit only train/validation; this function has no test-file argument."""
    license_evidence = _read_license_evidence()
    train_rows, train_summary = _read_split(Path(train_path), "train")
    validation_rows, validation_summary = _read_split(Path(validation_path), "validation")
    partition_by_key: dict[str, set[str]] = defaultdict(set)
    claim_counts: Counter[str] = Counter()
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for row in rows:
            claim = _normalize(str(row.get("claim") or ""))
            trial = str(row.get("nctId") or "").strip().upper()
            pmid = str(row.get("PMID") or "").strip()
            if claim:
                claim_key = hashlib.sha256(claim.encode("utf-8")).hexdigest()
                claim_counts[claim_key] += 1
                partition_by_key["claim:" + claim_key].add(split)
            if trial:
                partition_by_key["trial:" + trial].add(split)
            if pmid:
                partition_by_key["pmid:" + pmid].add(split)

    return {
        "schema": "clinifact-source-audit-v1",
        "dataset": "CliniFact",
        "dataset_doi": DATASET_DOI,
        "dataset_license": "CC BY 4.0 (Figshare article record)",
        "dataset_license_evidence": license_evidence,
        "label_mapping": dict(LABELS),
        "label_mapping_verified_by_published_counts": True,
        "official_test_downloaded_or_read": False,
        "splits": {"train": train_summary, "validation": validation_summary},
        "unique_claims": len(claim_counts),
        "duplicate_normalized_claim_groups": sum(n > 1 for n in claim_counts.values()),
        "duplicate_normalized_claim_extra_rows": sum(
            n - 1 for n in claim_counts.values() if n > 1),
        "train_validation_group_overlap": {
            key.split(":", 1)[0]: sum(len(splits) > 1 for name, splits in partition_by_key.items()
                                     if name.startswith(key + ":"))
            for key in ("claim", "trial", "pmid")
        },
        "unique_trials": len({str(row["nctId"]).strip().upper()
                              for row in train_rows + validation_rows}),
        "unique_pmids": len({str(row["PMID"]).strip()
                             for row in train_rows + validation_rows}),
        "source_text_or_claim_hashes_emitted": False,
        "candidate_cohort_created": False,
    }
