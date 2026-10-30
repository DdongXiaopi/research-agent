#!/usr/bin/env python3
"""Build frozen Evidence Inference MCQ rows from licensed abstract text.

This preparation path only materializes optimizer-train and development rows.
It deliberately refuses the official test split. TaskRunner consumes the
fixed ``evidence`` field; candidate evaluation never calls BioC or search.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.audit_evidence_inference_v2 import _parse_noise_ids, _truth
    from scripts.fetch_evidence_inference_abstracts import ALLOWLIST
    from src.util import sha256_obj
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from audit_evidence_inference_v2 import _parse_noise_ids, _truth
    from fetch_evidence_inference_abstracts import ALLOWLIST
    from src.util import sha256_obj


OPTIONS = ["significantly increased", "significantly decreased", "no significant difference"]
CODE_TO_INDEX = {"1": 0, "-1": 1, "0": 2}


def _read_archive_tables(archive_path: Path) -> tuple[dict[str, dict[str, str]], list[dict[str, str]], set[str], dict[str, set[int]]]:
    with tarfile.open(archive_path, "r:gz") as archive:
        prompts = list(csv.DictReader(io.StringIO(archive.extractfile("prompts_merged.csv").read().decode("utf-8-sig"))))
        annotations = list(csv.DictReader(io.StringIO(archive.extractfile("annotations_merged.csv").read().decode("utf-8-sig"))))
        readme = archive.extractfile("README.md").read().decode("utf-8-sig", errors="replace")
        article_splits = {
            split: set(archive.extractfile(f"splits/{split}_article_ids.txt").read().decode("utf-8").splitlines())
            for split in ("train", "validation", "test")
        }
    prompt_by_id = {row["PromptID"].strip(): row for row in prompts}
    noise = set().union(*_parse_noise_ids(readme).values())
    return prompt_by_id, annotations, noise, article_splits


def _load_abstracts(path: Path, article_ids: set[str]) -> dict[str, dict[str, Any]]:
    abstracts = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            article_id = str(row.get("pmcid") or "")
            if article_id not in article_ids:
                continue
            if not article_id or article_id in abstracts:
                raise ValueError(f"Missing or duplicate abstract PMCID at line {line_number}")
            if row.get("status") == "ready":
                text = str(row.get("abstract") or "").strip()
                if row.get("license") not in ALLOWLIST or not text:
                    raise ValueError(f"Invalid ready abstract record for PMCID {article_id}")
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if row.get("abstract_sha256") != digest:
                    raise ValueError(f"Abstract hash mismatch for PMCID {article_id}")
            abstracts[article_id] = row
    return abstracts


def prepare_rows(
    archive_path: Path,
    abstracts_path: Path,
    *,
    splits: tuple[str, ...] = ("train",),
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not splits or any(split not in {"train", "validation"} for split in splits):
        raise ValueError("Only train and validation may be materialized; test is sealed")
    prompt_by_id, annotations, noise_ids, article_splits = _read_archive_tables(archive_path)
    article_to_split: dict[str, str] = {}
    for split, article_ids in article_splits.items():
        for article_id in article_ids:
            if article_id in article_to_split:
                raise ValueError(f"Article {article_id} appears in multiple official splits")
            article_to_split[article_id] = split

    requested_articles = {
        row["PMCID"].strip()
        for row in prompt_by_id.values()
        if article_to_split.get(row["PMCID"].strip()) in splits
    }
    abstract_by_article = _load_abstracts(abstracts_path, requested_articles)

    votes: dict[str, Counter[str]] = defaultdict(Counter)
    abstract_prompt_ids: set[str] = set()
    for row in annotations:
        prompt_id = row["PromptID"].strip()
        prompt = prompt_by_id.get(prompt_id)
        if prompt is None or article_to_split.get(prompt["PMCID"].strip()) not in splits:
            continue
        if _truth(row.get("In Abstract")):
            abstract_prompt_ids.add(prompt_id)
            code = row.get("Label Code")
            if _truth(row.get("Valid Label")) and _truth(row.get("Valid Reasoning")) and code in CODE_TO_INDEX:
                votes[prompt_id][code] += 1

    out: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}
    excluded = Counter()
    for prompt_id in sorted(abstract_prompt_ids, key=int):
        prompt = prompt_by_id.get(prompt_id)
        if prompt is None:
            excluded["missing_prompt"] += 1
            continue
        article_id = prompt["PMCID"].strip()
        split = article_to_split.get(article_id)
        if split not in splits:
            continue
        if int(prompt_id) in noise_ids:
            excluded[f"{split}:known_noise"] += 1
            continue
        abstract_row = abstract_by_article.get(article_id)
        if not abstract_row or abstract_row.get("status") != "ready":
            excluded[f"{split}:license_or_abstract_unavailable"] += 1
            continue
        class_votes = votes[prompt_id]
        if not class_votes:
            excluded[f"{split}:no_valid_abstract_label"] += 1
            continue
        ordered_votes = class_votes.most_common()
        if len(ordered_votes) > 1 and ordered_votes[0][1] == ordered_votes[1][1]:
            excluded[f"{split}:label_tie"] += 1
            continue
        gold_code, winning_votes = ordered_votes[0]
        valid_vote_count = sum(class_votes.values())
        context = str(abstract_row["abstract"]).strip()
        question = (
            "Based only on the supplied trial abstract, characterize the reported effect of "
            f"the intervention ({prompt['Intervention'].strip()}) versus the comparator "
            f"({prompt['Comparator'].strip()}) with respect to the outcome "
            f"({prompt['Outcome'].strip()}). Choose the label that matches the reported "
            "statistical result; do not interpret which treatment is clinically preferable."
        )
        out[split].append(
            {
                "task_id": f"evidence-inference-v2-{prompt_id}",
                "dataset": "evidence_inference_v2_abstract",
                "task_type": "mcq",
                "domain": "biomedical",
                "split": split,
                "group_id": f"PMC:{article_id}",
                "question": question,
                "context": "",
                "evidence": context,
                "options": OPTIONS,
                "correct_index": CODE_TO_INDEX[gold_code],
                "gold_label": OPTIONS[CODE_TO_INDEX[gold_code]],
                "sources": [str(abstract_row["source_url"])],
                "metadata": {
                    "prompt_id": prompt_id,
                    "pmcid": article_id,
                    "article_license": abstract_row["license"],
                    "abstract_sha256": abstract_row["abstract_sha256"],
                    # Annotation-support metadata is for cohort quality audits;
                    # the candidate runtime does not place it in the prompt.
                    "valid_abstract_label_vote_count": valid_vote_count,
                    "winning_label_vote_count": winning_votes,
                    "label_agreement_fraction": winning_votes / valid_vote_count,
                },
            }
        )

    for split_rows in out.values():
        split_rows.sort(key=lambda row: row["task_id"])
    manifest = {
        "dataset": "Evidence Inference 2.0 abstract subset",
        "source_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "abstracts_jsonl_sha256": hashlib.sha256(abstracts_path.read_bytes()).hexdigest(),
        "splits_materialized": list(splits),
        "test_materialized": False,
        "candidate_runtime_uses_live_retrieval": False,
        "license_allowlist": sorted(ALLOWLIST),
        "label_policy": "Strict majority among Valid Label and Valid Reasoning annotations marked In Abstract; drop ties.",
        "known_noise_prompt_ids_excluded": len(noise_ids),
        "row_counts": {split: len(rows) for split, rows in out.items()},
        "class_counts": {
            split: dict(sorted(Counter(row["gold_label"] for row in rows).items()))
            for split, rows in out.items()
        },
        "annotation_support": {
            split: {
                "valid_vote_count_histogram": {
                    str(vote_count): count
                    for vote_count, count in sorted(
                        Counter(row["metadata"]["valid_abstract_label_vote_count"] for row in rows).items()
                    )
                },
                "agreement_fraction_histogram": {
                    f"{agreement:.3f}": count
                    for agreement, count in sorted(
                        Counter(row["metadata"]["label_agreement_fraction"] for row in rows).items()
                    )
                },
            }
            for split, rows in out.items()
        },
        "article_counts": {
            split: len({row["metadata"]["pmcid"] for row in rows})
            for split, rows in out.items()
        },
        "exclusion_counts": dict(sorted(excluded.items())),
    }
    return out, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/evidence-inference-2.0.tar.gz"))
    parser.add_argument(
        "--abstracts",
        type=Path,
        default=Path("data/normalized/evidence-inference-abstracts/abstracts.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/normalized/evidence-inference-task-input"))
    parser.add_argument("--splits", nargs="+", choices=("train", "validation"), default=("train",))
    args = parser.parse_args()
    from src.data.freeze import require_unfrozen
    from src.util import commit_new_files, jsonl_bytes
    require_unfrozen("evidence_inference_v2_abstract")
    if len(set(args.splits)) != len(args.splits):
        raise ValueError("requested source splits must be unique")
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("prepared split output must stay inside the project root") from exc
    manifest_path = PROJECT_ROOT / "data/manifests/evidence_inference_v2_abstract.json"
    destinations = [output_dir / f"{split}.jsonl" for split in args.splits]
    destinations.extend([output_dir / "manifest.json", manifest_path])
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError("Evidence Inference prepared outputs already exist")
    rows, manifest = prepare_rows(args.archive, args.abstracts, splits=tuple(args.splits))
    staged = []
    for split, split_rows in rows.items():
        path = output_dir / f"{split}.jsonl"
        content = jsonl_bytes(split_rows)
        staged.append((path, content))
        manifest.setdefault("outputs", {})[split] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    capture_manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    project_root = PROJECT_ROOT
    project_splits = {"train": "dev_optimize", "validation": "selection_gate"}
    dataset_manifest = {
        "dataset": "evidence_inference_v2_abstract",
        "protocol_version": "evidence-inference-v2-abstract-task-input-v2-annotation-support",
        "source_dataset": "Evidence Inference 2.0 abstract-only subset",
        "source_archive_sha256": manifest["source_archive_sha256"],
        "abstract_source_api": "NCBI PMC BioC API, section_type=ABSTRACT",
        "abstracts_jsonl_sha256": manifest["abstracts_jsonl_sha256"],
        "license_allowlist": manifest["license_allowlist"],
        "label_policy": manifest["label_policy"],
        "split_policy": {
            "official_train": "dev_optimize",
            "official_validation": "selection_gate",
            "official_test": "sealed; never materialized or exposed by this loader",
        },
        "candidate_runtime_uses_live_retrieval": False,
        "eligible_for_confirmation": False,
        "confirmation_status": "development source integration only; official test withheld",
        "known_noise_prompt_ids_excluded": manifest["known_noise_prompt_ids_excluded"],
        "splits": {},
    }
    for source_split, project_split in project_splits.items():
        output = manifest["outputs"].get(source_split)
        if output is None:
            continue
        output_path = Path(output["path"])
        try:
            relative_path = output_path.resolve().relative_to(project_root.resolve())
        except ValueError as exc:
            raise ValueError("prepared split output must stay inside the project root") from exc
        ids = [row["task_id"] for row in rows[source_split]]
        dataset_manifest["splits"][project_split] = {
            "source_split": source_split,
            "path": relative_path.as_posix(),
            "n": len(ids),
            "ids": ids,
            "ids_sha256": sha256_obj(ids),
            "file_sha256": output["sha256"],
            "class_counts": manifest["class_counts"][source_split],
            "article_count": manifest["article_counts"][source_split],
            "annotation_support": manifest["annotation_support"][source_split],
        }
    staged.append((output_dir / "manifest.json", capture_manifest_bytes))
    staged.append((manifest_path, (json.dumps(dataset_manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")))
    commit_new_files(staged)
    manifest["project_manifest"] = str(manifest_path.resolve())
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
