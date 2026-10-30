"""Freeze a genuinely fresh, article-disjoint Evidence Inference cohort.

The repository has many historical exploratory EI cohorts.  This script
derives an explicit exclusion list from those immutable receipts, then picks
one task per article with balanced labels across three physically separated
splits.  It never reads the official test split and never changes the old
dataset files.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.util import sha256_file, sha256_obj, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    ROOT / "data/normalized/evidence-inference-task-input/train.jsonl",
    ROOT / "data/normalized/evidence-inference-task-input/validation.jsonl",
)
DATASET = "evidence_inference_fresh_20261018"
SALT = "SALT-37"
LABELS = (
    "significantly increased",
    "significantly decreased",
    "no significant difference",
)
TARGETS = {"dev_optimize": 20, "selection_gate": 20, "final_holdout": 20}


def _historical_exclusions() -> tuple[set[str], set[str], list[str]]:
    articles: set[str] = set()
    tasks: set[str] = set()
    receipts: list[str] = []
    for raw in glob.glob(str(ROOT / "artifacts/**/*.json"), recursive=True):
        path = Path(raw)
        lowered = raw.lower()
        if "evidence" not in lowered or ("cohort" not in lowered and "nested" not in lowered):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot inspect historical cohort receipt: {path.name}") from exc
        found = False

        def walk(value: Any) -> None:
            nonlocal found
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "article_ids" and isinstance(child, list):
                        articles.update(str(x) for x in child)
                        found = True
                    elif key == "task_ids" and isinstance(child, list):
                        tasks.update(str(x) for x in child)
                        found = True
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(payload)
        if found:
            receipts.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    return articles, tasks, sorted(receipts)


def _rank(*parts: str) -> str:
    import hashlib

    return hashlib.sha256("|".join((SALT, *parts)).encode("utf-8")).hexdigest()


def freeze(output_root: Path, *, write: bool = True) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    if write:
        try:
            output_root.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError("normalized output directory must be inside the project") from exc
        destinations = (output_root / f"{DATASET}.jsonl",
                        ROOT / "data/sealed" / f"{DATASET}.final_holdout.jsonl",
                        ROOT / "data/manifests" / f"{DATASET}.json",
                        ROOT / "data/manifests" / f"{DATASET}.excluded.json")
        freeze_file = ROOT / "data/manifests/freeze.json"
        if freeze_file.exists():
            frozen = json.loads(freeze_file.read_text(encoding="utf-8"))
            if DATASET in (frozen.get("datasets") or {}):
                raise FileExistsError("cannot rewrite a frozen dataset")
        if any(destination.exists() for destination in destinations):
            raise FileExistsError("dataset output already exists; choose a new dataset rather than overwrite it")
    excluded_articles, excluded_tasks, receipts = _historical_exclusions()
    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for source in SOURCE_FILES:
        source_hashes[str(source.relative_to(ROOT)).replace("\\", "/")] = sha256_file(source)
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                article = str(row.get("group_id") or "")
                task_id = str(row.get("task_id") or "")
                if article in excluded_articles or task_id in excluded_tasks:
                    continue
                if str(row.get("gold_label")) not in LABELS or not article or not task_id:
                    continue
                rows.append(row)

    by_label_article: dict[str, dict[str, list[dict[str, Any]]]] = {
        label: defaultdict(list) for label in LABELS
    }
    for row in rows:
        by_label_article[str(row["gold_label"])][str(row["group_id"])].append(row)

    # Assign every article to exactly one split first.  The chosen salt was
    # searched once against the frozen source inventory so each split has at
    # least 20 independent articles of every class; the resulting assignment
    # is then fully deterministic and recorded in the manifest.
    import hashlib
    article_split: dict[str, str] = {}
    for article in sorted({a for per_label in by_label_article.values() for a in per_label}):
        bucket = int(hashlib.sha256((SALT + "|" + article).encode("utf-8")).hexdigest()[:8], 16) % 100
        article_split[article] = (
            "dev_optimize" if bucket < 50 else
            "selection_gate" if bucket < 75 else
            "final_holdout"
        )
    selected: dict[str, list[dict[str, Any]]] = {split: [] for split in TARGETS}
    for split, target in TARGETS.items():
        for label in LABELS:
            articles = sorted(
                [a for a in by_label_article[label] if article_split[a] == split],
                key=lambda article: _rank("article", label, article),
            )
            if len(articles) < target:
                raise RuntimeError(f"{label}/{split}: only {len(articles)} fresh articles, need {target}")
            for article in articles[:target]:
                candidates = sorted(
                    by_label_article[label][article],
                    key=lambda row: _rank("task", split, str(row["task_id"])),
                )
                row = dict(candidates[0])
                row["dataset"] = DATASET
                row["split"] = split
                row["fresh_source_dataset"] = "evidence_inference_v2_abstract"
                row["fresh_protocol_version"] = "evidence-inference-fresh-article-disjoint-v1"
                selected[split].append(row)

    split_counts: dict[str, dict[str, int]] = {}
    split_articles: dict[str, list[str]] = {}
    all_ids: list[str] = []
    for split, split_rows in selected.items():
        split_rows.sort(key=lambda row: str(row["task_id"]))
        split_counts[split] = {label: sum(1 for row in split_rows if row["gold_label"] == label) for label in LABELS}
        split_articles[split] = sorted({str(row["group_id"]) for row in split_rows})
        all_ids.extend(str(row["task_id"]) for row in split_rows)
        per_label_articles = {
            label: {str(row["group_id"]) for row in split_rows if row["gold_label"] == label}
            for label in LABELS
        }
        if any(len(per_label_articles[label]) != TARGETS[split] for label in LABELS):
            raise RuntimeError(f"{split} does not have one independent article per selected label row")
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("fresh cohort task IDs overlap")
    if set().union(*(set(v) for v in split_articles.values())) and any(
        set(split_articles[a]) & set(split_articles[b]) for i, a in enumerate(TARGETS) for b in list(TARGETS)[i + 1:]
    ):
        raise RuntimeError("fresh cohort article groups cross splits")

    manifest: dict[str, Any] = {
        "dataset": DATASET,
        "protocol_version": "evidence-inference-fresh-article-disjoint-v1",
        "source_dataset": "evidence_inference_v2_abstract",
        "source_file": "data/normalized/evidence-inference-task-input/{train,validation}.jsonl",
        "source_sha256": sha256_obj(source_hashes),
        "ratios": {"dev_optimize": 33, "selection_gate": 33, "final_holdout": 34},
        "split_algorithm": "article-hash-assignment-v1; one row per label/article",
        "n_questions": sum(len(v) for v in selected.values()),
        "source_files": source_hashes,
        "historical_exclusion_receipts": receipts,
        "historical_exclusion_scope": "artifacts JSON paths containing evidence and cohort or nested",
        "historical_exclusion_inventory_complete": False,
        "excluded_article_count": len(excluded_articles),
        "excluded_article_ids_sha256": sha256_obj(sorted(excluded_articles)),
        "excluded_task_ids_sha256": sha256_obj(sorted(excluded_tasks)),
        "salt": SALT,
        "official_test_read": False,
        "candidate_runtime_uses_live_retrieval": False,
        "eligible_for_confirmation": False,
        "annotation_origin": "source_native",
        "split_targets_per_label": TARGETS,
        "split_counts": split_counts,
        "split_article_counts": {k: len(v) for k, v in split_articles.items()},
        "split_article_ids_sha256": {k: sha256_obj(v) for k, v in split_articles.items()},
        "split_task_ids_sha256": {
            k: sha256_obj(sorted(str(row["task_id"]) for row in rows))
            for k, rows in selected.items()
        },
        "rows": {k: len(v) for k, v in selected.items()},
        "splits": {
            split: {
                "ids": sorted(str(row["task_id"]) for row in rows),
                "ids_sha256": sha256_obj(sorted(str(row["task_id"]) for row in rows)),
                "n": len(rows),
                "source_split": split,
            }
            for split, rows in selected.items()
        },
    }
    if write:
        normalized, sealed, manifest_path, exclusions_path = destinations
        manifest["normalized_path"] = str(normalized.relative_to(ROOT)).replace("\\", "/")
        manifest["sealed_path"] = str(sealed.relative_to(ROOT)).replace("\\", "/")
        exclusions = {
            "dataset": DATASET,
            "excluded_article_ids": sorted(excluded_articles),
            "excluded_task_ids": sorted(excluded_tasks),
            "receipts": receipts,
            "sha256": sha256_obj({"articles": sorted(excluded_articles), "tasks": sorted(excluded_tasks)}),
        }
        data_root = ROOT / "data"
        data_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".fresh-ei-stage-", dir=data_root) as tmp:
            stage = Path(tmp)
            pending = [stage / f"file-{index}" for index in range(4)]
            write_jsonl(pending[0], selected["dev_optimize"] + selected["selection_gate"])
            write_jsonl(pending[1], selected["final_holdout"])
            manifest["normalized_sha256"] = sha256_file(pending[0])
            manifest["sealed_sha256"] = sha256_file(pending[1])
            write_json(pending[2], manifest)
            write_json(pending[3], exclusions)
            installed = []
            try:
                for source, destination in zip(pending, destinations, strict=True):
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source, destination)  # Atomic create; never replace an existing file.
                    installed.append((source, destination))
            except BaseException:
                for source, destination in reversed(installed):
                    if destination.exists() and os.path.samefile(source, destination):
                        destination.unlink()
                raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(ROOT / "data/normalized"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(freeze(Path(args.output_root), write=not args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
