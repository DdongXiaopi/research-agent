#!/usr/bin/env python3
"""Normalize Evidence Inference article abstracts from NCBI's official BioC API.

This is a one-time dataset preparation tool. Candidate evaluation reads the
resulting frozen local JSONL; it never calls this API or the project's search
gateway. To avoid external processing of text under unclear article terms, the
default allowlist is limited to CC0 and CC BY. Other license classes are
recorded in the manifest and their abstract text is discarded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BIOC_BASE = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json"
ALLOWLIST = {"CC BY", "CC0"}


def _article_ids_from_archive(archive_path: Path) -> list[str]:
    import tarfile

    with tarfile.open(archive_path, "r:gz") as archive:
        prompts = list(
            csv.DictReader(
                io.StringIO(archive.extractfile("prompts_merged.csv").read().decode("utf-8-sig"))
            )
        )
        abstract_prompt_ids = {
            row["PromptID"]
            for row in csv.DictReader(
                io.StringIO(archive.extractfile("annotations_merged.csv").read().decode("utf-8-sig"))
            )
            if row.get("In Abstract", "").strip().casefold() == "true"
        }
        prompt_by_id = {row["PromptID"]: row for row in prompts}
        ids = {prompt_by_id[prompt_id]["PMCID"] for prompt_id in abstract_prompt_ids}
    return sorted(ids, key=int)


def _fetch_batch(article_ids: list[str], timeout: int = 90) -> list[dict[str, Any]]:
    pmc_ids = ",".join(f"PMC{article_id}" for article_id in article_ids)
    url = f"{BIOC_BASE}/{pmc_ids}/unicode"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "RSI-evolution-dataset-audit/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, list):
                raise ValueError("BioC response must be a JSON list")
            return parsed
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"BioC API request failed after retries: {last_error}")


def _flatten_batch(article_ids: list[str], payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    if len(payload) == len(article_ids):
        collection_pairs = zip(article_ids, payload)
    else:
        collection_pairs = (("", collection) for collection in payload)
    for requested_id, collection in collection_pairs:
        returned_documents = collection.get("documents", [])
        for index, document in enumerate(returned_documents):
            doc_id = str(document.get("id") or "")
            if doc_id.startswith("PMC"):
                doc_id = doc_id[3:]
            if not doc_id or doc_id == "unknown":
                # The API emits one collection per requested ID, in request order;
                # some older records have no stable document id in BioC.
                doc_id = requested_id if index == 0 else ""
            if doc_id:
                documents[doc_id] = document
    output: list[dict[str, Any]] = []
    retrieved_at = datetime.now(timezone.utc).isoformat()
    for article_id in article_ids:
        document = documents.get(article_id)
        if document is None:
            output.append(
                {
                    "pmcid": article_id,
                    "status": "not_returned_by_bioc",
                    "license": None,
                    "abstract": None,
                    "abstract_sha256": None,
                    "retrieved_at": retrieved_at,
                }
            )
            continue
        license_name = str(document.get("infons", {}).get("license", "")).strip()
        abstract_parts = [
            str(passage.get("text", "")).strip()
            for passage in document.get("passages", [])
            if str(passage.get("infons", {}).get("section_type", "")).upper() == "ABSTRACT"
            and str(passage.get("text", "")).strip()
        ]
        abstract = "\n".join(abstract_parts)
        title = next(
            (
                str(passage.get("text", "")).strip()
                for passage in document.get("passages", [])
                if str(passage.get("infons", {}).get("section_type", "")).upper() == "TITLE"
                and str(passage.get("text", "")).strip()
            ),
            None,
        )
        allowed = license_name in ALLOWLIST
        output.append(
            {
                "pmcid": article_id,
                "source_url": f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{article_id}/",
                "title": title,
                "status": "ready" if allowed and abstract else "license_excluded" if not allowed else "no_abstract",
                "license": license_name or None,
                "abstract": abstract if allowed else None,
                "abstract_sha256": hashlib.sha256(abstract.encode("utf-8")).hexdigest()
                if allowed and abstract
                else None,
                "retrieved_at": retrieved_at,
            }
        )
    return output


def normalize_abstracts(
    archive_path: Path,
    output_dir: Path,
    *,
    batch_size: int = 25,
    limit_articles: int | None = None,
) -> dict[str, Any]:
    if batch_size < 1 or batch_size > 100:
        raise ValueError("batch_size must be in [1, 100]")
    article_ids = _article_ids_from_archive(archive_path)
    if limit_articles is not None:
        article_ids = article_ids[:limit_articles]
    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "abstracts.jsonl"
    manifest_path = output_dir / "manifest.json"
    batch_cache_dir = output_dir / "batch_cache"
    batch_cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(article_ids), batch_size):
        batch = article_ids[start : start + batch_size]
        batch_key = hashlib.sha256(
            (archive_hash + "\n" + "\n".join(batch)).encode("utf-8")
        ).hexdigest()[:16]
        batch_path = batch_cache_dir / f"{batch_key}.jsonl"
        if batch_path.is_file():
            batch_rows = [json.loads(line) for line in batch_path.read_text(encoding="utf-8").splitlines() if line]
            if [row.get("pmcid") for row in batch_rows] != batch:
                raise ValueError(f"Mismatched batch cache: {batch_path}")
        else:
            payload = _fetch_batch(batch)
            batch_rows = _flatten_batch(batch, payload)
            temporary_batch = batch_path.with_suffix(".jsonl.tmp")
            with temporary_batch.open("w", encoding="utf-8", newline="\n") as stream:
                for row in batch_rows:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            temporary_batch.replace(batch_path)
        rows.extend(batch_rows)
        print(f"BioC abstracts checked: {len(rows)}/{len(article_ids)}")

    counts = Counter(row["status"] for row in rows)
    licenses = Counter(row["license"] or "missing" for row in rows)
    temporary_data = data_path.with_suffix(".jsonl.tmp")
    with temporary_data.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary_data.replace(data_path)
    manifest = {
        "dataset": "Evidence Inference 2.0",
        "input_archive": str(archive_path.resolve()),
        "input_archive_sha256": archive_hash,
        "source_api": BIOC_BASE,
        "abstract_section": "BioC passage infons.section_type == ABSTRACT",
        "external_text_processing_allowlist": sorted(ALLOWLIST),
        "article_count_requested": len(article_ids),
        "article_count_returned": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "license_counts": dict(sorted(licenses.items())),
        "abstracts_jsonl": str(data_path.resolve()),
        "abstracts_jsonl_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "candidate_runtime_uses_live_retrieval": False,
        "note": "Only CC0 and CC BY abstract text is retained; all other article text is discarded after license inspection.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/evidence-inference-2.0.tar.gz"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/normalized/evidence-inference-abstracts"),
    )
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--limit-articles", type=int, help="Smoke-test limit; omit for the full frozen cohort")
    args = parser.parse_args()
    manifest = normalize_abstracts(
        args.archive,
        args.output_dir,
        batch_size=args.batch_size,
        limit_articles=args.limit_articles,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
