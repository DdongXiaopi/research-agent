#!/usr/bin/env python3
"""Audit the official Evidence Inference 2.0 archive without calling a model.

The audit treats an abstract-only example as a PromptID with at least one
accepted annotation whose evidence was marked In Abstract. Gold labels are a
strict majority over accepted abstract-evidence annotations; ties and prompts
without an accepted three-way label are excluded. The official README's
known-bad prompt list is reported and excluded from the usable-count summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


SPLITS = ("train", "validation", "test")
LABEL_NAMES = {"-1": "decreased", "0": "no_difference", "1": "increased"}
NOISE_SECTIONS = (
    ("### Incorrect:", "### Questionable:"),
    ("### Questionable:", "### Somewhat malformed:"),
    ("### Somewhat malformed:", None),
)


def _read_member(archive: tarfile.TarFile, name: str) -> bytes:
    member = archive.getmember(name)
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"Archive member is not a regular file: {name}")
    return stream.read()


def _check_archive_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive path: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"Unexpected archive member type: {member.name}")
    return members


def _csv_rows(raw: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig", errors="strict"))))


def _truth(value: str | None) -> bool:
    return (value or "").strip().casefold() in {"true", "1", "yes"}


def _parse_noise_ids(readme: str) -> dict[str, set[int]]:
    sections: dict[str, set[int]] = {}
    for heading, next_heading in NOISE_SECTIONS:
        tail = readme.split(heading, 1)[1] if heading in readme else ""
        section = tail.split(next_heading, 1)[0] if next_heading and next_heading in tail else tail
        sections[heading.removeprefix("### ").removesuffix(":")] = {
            int(value) for value in re.findall(r"\b\d+\b", section)
        }
    return sections


def audit_archive(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with tarfile.open(path, mode="r:gz") as archive:
        members = _check_archive_members(archive)
        names = {member.name for member in members}
        readme = _read_member(archive, "README.md").decode("utf-8-sig", errors="replace")
        prompts = _csv_rows(_read_member(archive, "prompts_merged.csv"))
        annotations = _csv_rows(_read_member(archive, "annotations_merged.csv"))
        split_articles = {
            split: set(_read_member(archive, f"splits/{split}_article_ids.txt").decode("utf-8").splitlines())
            for split in SPLITS
        }

        prompt_by_id: dict[str, dict[str, str]] = {}
        for row in prompts:
            prompt_id = row["PromptID"].strip()
            if not prompt_id or prompt_id in prompt_by_id:
                raise ValueError(f"Missing or duplicate PromptID in prompt table: {prompt_id!r}")
            prompt_by_id[prompt_id] = row

        abstract_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
        abstract_prompt_ids: set[str] = set()
        for row in annotations:
            prompt_id = row["PromptID"].strip()
            if _truth(row.get("In Abstract")):
                abstract_prompt_ids.add(prompt_id)
                if _truth(row.get("Valid Label")) and _truth(row.get("Valid Reasoning")):
                    if row.get("Label Code") in LABEL_NAMES:
                        abstract_rows[prompt_id].append(row)

        noise_by_type = _parse_noise_ids(readme)
        noise_ids = set().union(*noise_by_type.values()) if noise_by_type else set()
        article_to_split: dict[str, str] = {}
        split_overlap: dict[str, int] = {}
        for split, article_ids in split_articles.items():
            for article_id in article_ids:
                if article_id in article_to_split:
                    split_overlap[f"{article_to_split[article_id]}:{split}"] = (
                        split_overlap.get(f"{article_to_split[article_id]}:{split}", 0) + 1
                    )
                else:
                    article_to_split[article_id] = split

        result_by_split: dict[str, Any] = {}
        usable_article_ids: set[str] = set()
        for split, article_ids in split_articles.items():
            selected = [
                prompt_id
                for prompt_id in abstract_prompt_ids
                if prompt_id in prompt_by_id and prompt_by_id[prompt_id]["PMCID"] in article_ids
            ]
            excluded_noise = [prompt_id for prompt_id in selected if int(prompt_id) in noise_ids]
            labels: Counter[str] = Counter()
            no_accepted_label = 0
            ties = 0
            clean_ids: list[str] = []
            for prompt_id in selected:
                if int(prompt_id) in noise_ids:
                    continue
                votes = Counter(row["Label Code"] for row in abstract_rows.get(prompt_id, []))
                if not votes:
                    no_accepted_label += 1
                    continue
                ranked = votes.most_common()
                if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                    ties += 1
                    continue
                labels[LABEL_NAMES[ranked[0][0]]] += 1
                clean_ids.append(prompt_id)
            article_ids_with_usable_examples = {
                prompt_by_id[prompt_id]["PMCID"] for prompt_id in clean_ids
            }
            missing_text = sorted(
                article_id
                for article_id in article_ids_with_usable_examples
                if f"txt_files/PMC{article_id}.txt" not in names
            )
            usable_article_ids.update(article_ids_with_usable_examples)
            result_by_split[split] = {
                "official_article_count": len(article_ids),
                "abstract_only_prompt_count": len(selected),
                "known_noise_excluded": len(excluded_noise),
                "no_accepted_threeway_label": no_accepted_label,
                "ambiguous_majority_tie": ties,
                "usable_prompt_count": len(clean_ids),
                "usable_class_counts": dict(sorted(labels.items())),
                "usable_article_count": len(article_ids_with_usable_examples),
                "missing_article_text_files": missing_text,
            }

        return {
            "dataset": "Evidence Inference 2.0",
            "source_url": "https://evidence-inference.ebm-nlp.com/v2.0.tar.gz",
            "source_archive": str(path.resolve()),
            "archive_sha256": digest,
            "archive_bytes": path.stat().st_size,
            "archive_members": len(members),
            "annotation_rows": len(annotations),
            "prompt_rows": len(prompts),
            "unique_abstract_only_prompt_ids": len(abstract_prompt_ids),
            "abstract_only_subset_matches_paper_count": len(abstract_prompt_ids) == 6375,
            "noise_prompt_ids_by_readme_category": {
                name: len(values) for name, values in noise_by_type.items()
            },
            "official_split_article_overlap": split_overlap,
            "splits": result_by_split,
            "all_usable_articles_have_text_files": all(
                not detail["missing_article_text_files"] for detail in result_by_split.values()
            ),
            "label_policy": (
                "Use Valid Label and Valid Reasoning annotations marked In Abstract; "
                "strict majority among -1/0/1 labels; drop ties and known-noise PromptIDs."
            ),
            "limitations": [
                "This checks annotation and article split integrity, not abstract-text extraction fidelity.",
                "The article archive has per-article copyright terms; confirm each article's reuse terms before external processing.",
                "The audit does not call a model and does not access labels beyond this public dataset package.",
            ],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/evidence-inference-2.0.tar.gz"),
        help="Official Evidence Inference 2.0 archive",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report output path")
    args = parser.parse_args()
    report = audit_archive(args.archive)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
