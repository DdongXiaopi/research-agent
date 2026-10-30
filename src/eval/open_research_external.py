"""Deterministic adapters for vendored open research benchmarks.

This module keeps ScholarQABench/QASA useful for G09 without treating its
reference answers as assistant-visible input.  QASA is a single-paper,
long-form benchmark with paper-level groups and gold context indices.  The
adapter emits the same task-only shape used by ``open_research_cohort``;
rubrics still have to be authored in a separate, task-first AI job before the
dataset can be frozen by ``open_benchmark``.

The adapter is deliberately offline.  It reads only a vendored JSONL file,
does not call a model or a network, and writes reference answers to a separate
sidecar for evaluator audit.  The sidecar must never be passed to rubric
authoring, answer execution, or final-holdout readers.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..util import sha256_file, sha256_obj, sha256_text


QASA_DEFAULT_SOURCE = Path(
    "vendor/ScholarQABench-main/ScholarQABench-main/data/single_paper_tasks/qasa_test.jsonl"
)
QASA_PROTOCOL = "scholarqabench-qasa-supplied-context-v1"
_ARXIV = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$")


def _group_id(context_id: str) -> str:
    """Return the paper identity encoded by QASA's ``<paper>_all_<n>`` IDs."""
    value = str(context_id or "").strip()
    if "_all_" not in value:
        raise ValueError("QASA context ID must contain the paper _all_ separator")
    paper, _, suffix = value.rpartition("_all_")
    if not paper or not suffix.isdigit():
        raise ValueError("QASA context ID has an invalid paper/chunk identity")
    return paper


def _source_id(group: str, index: int) -> str:
    # TaskRequest IDs intentionally allow only [A-Za-z0-9_-]; arXiv IDs contain
    # dots, so bind the readable group to a short deterministic digest.
    return f"qasa-{sha256_text(group)[:16]}-{index:04d}"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("QASA source is not valid UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid QASA JSON at line {line_no}") from exc
        if not isinstance(row, dict):
            raise ValueError("QASA rows must be JSON objects")
        rows.append(row)
    if not rows:
        raise ValueError("QASA source contains no rows")
    return rows


def _normalize_row(row: dict[str, Any], row_index: int, *, max_sources: int) -> tuple[dict, dict]:
    question = row.get("input")
    answer = row.get("answer")
    contexts = row.get("ctxs")
    gold = row.get("gold_ctxs")
    if (not isinstance(question, str) or not question.strip() or
            not isinstance(answer, str) or not answer.strip() or
            not isinstance(contexts, list) or not contexts or
            not isinstance(gold, list) or not gold):
        raise ValueError(f"QASA row {row_index} lacks question, answer, contexts or gold indices")
    if isinstance(max_sources, bool) or not isinstance(max_sources, int) or max_sources < 1:
        raise ValueError("max_sources must be a positive integer")
    if any(isinstance(value, bool) or not isinstance(value, int) or
           value < 0 or value >= len(contexts) for value in gold):
        raise ValueError(f"QASA row {row_index} has an out-of-range gold context index")
    gold_indices = sorted(set(gold))
    if len(gold_indices) > max_sources:
        raise ValueError("max_sources is smaller than the row's gold context set")
    group = None
    normalized_contexts: list[dict[str, str]] = []
    seen_context_ids: set[str] = set()
    for index, context in enumerate(contexts):
        if not isinstance(context, dict):
            raise ValueError(f"QASA row {row_index} contains a non-object context")
        context_id = context.get("id")
        text = context.get("text")
        if (not isinstance(context_id, str) or not context_id.strip() or
                not isinstance(text, str) or context_id in seen_context_ids):
            raise ValueError(f"QASA row {row_index} contains an invalid or duplicate context")
        current_group = _group_id(context_id)
        if group is None:
            group = current_group
        elif current_group != group:
            raise ValueError(f"QASA row {row_index} mixes paper groups")
        seen_context_ids.add(context_id)
        if not text.strip():
            # QASA contains a small number of whitespace-only chunks.  They
            # are harmless retrieval negatives, but must never be selected as
            # a gold context or emitted as an empty evidence source.
            if index in gold_indices:
                raise ValueError(f"QASA row {row_index} has an empty gold context")
            continue
        normalized_contexts.append({
            "id": context_id,
            "text": text,
            "title": str(context.get("title") or "").strip(),
            "index": index,
        })
    assert group is not None
    available = {context["index"]: context for context in normalized_contexts}
    chosen_indices = list(gold_indices)
    chosen_indices.extend(index for index in sorted(available)
                          if index not in gold_indices and len(chosen_indices) < max_sources)
    evidence = []
    for index in chosen_indices:
        context = available[index]
        locator = f"qasa:{context['id']}"
        url = f"https://arxiv.org/abs/{group}" if _ARXIV.fullmatch(group) else ""
        evidence.append({
            "source_id": _source_id(group, index),
            "text": context["text"],
            "url": url,
            "locator": locator,
            "retrieved_at": "vendored-scholarqabench",
        })
    task_id = f"qasa-{sha256_text(f'{group}\0{row_index}\0{question}')[:24]}"
    request = {
        "task_id": task_id,
        "query": question.strip(),
        "task_type": "open_research",
        "domain_profile": "scientific_literature",
        "tool_allowlist": ["workspace.read", "citation.verify"],
        "budget": {"max_evidence_chars": 60000, "max_answer_chars": 8000,
                   "max_completion_tokens": 4096, "max_physical_requests": 1},
        "project_id": "g09-qasa",
        "session_id": "qasa-group-" + sha256_text(group)[:16],
        "evidence": evidence,
    }
    reference = {
        "task_id": task_id,
        "upstream_row_index": row_index,
        "group_id": "qasa-paper-" + sha256_text(group)[:16],
        "upstream_paper_id": group,
        "answer": answer,
        "gold_source_ids": [_source_id(group, index) for index in gold_indices],
        "gold_context_indices": gold_indices,
        "source_sha256": sha256_text(json.dumps(row, ensure_ascii=False, sort_keys=True)),
    }
    return ({"request": request, "group_id": reference["group_id"],
             "evaluation_role": "primary"}, reference)


def load_qasa_cases(path: Path | str = QASA_DEFAULT_SOURCE, *, max_sources: int = 30,
                    limit: int | None = None) -> tuple[list[dict], list[dict], dict]:
    """Load QASA rows as task-only cases and evaluator-only references.

    ``limit`` truncates by source row after validation.  It is intended for
    deterministic development probes, never for silently constructing a
    final holdout.
    """
    source = Path(path)
    rows = _read_rows(source)
    cases, references = [], []
    seen_task_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        case, reference = _normalize_row(row, row_index, max_sources=max_sources)
        if case["request"]["task_id"] in seen_task_ids:
            raise ValueError("QASA source contains duplicate derived task IDs")
        seen_task_ids.add(case["request"]["task_id"])
        cases.append(case)
        references.append(reference)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer when supplied")
        cases, references = cases[:limit], references[:limit]
    groups = sorted({case["group_id"] for case in cases})
    report = {
        "schema": "g09-external-qasa-source-report-v1",
        "protocol_version": QASA_PROTOCOL,
        "source_path": source.as_posix(),
        "source_sha256": sha256_file(source),
        "rows_read": len(rows),
        "rows_emitted": len(cases),
        "paper_groups": len(groups),
        "contexts_per_case_max": max_sources,
        "gold_contexts_retained": all(
            set(reference["gold_source_ids"]).issubset(
                {source_row["source_id"] for source_row in case["request"]["evidence"]})
            for case, reference in zip(cases, references)),
        "task_ids_sha256": sha256_obj(sorted(case["request"]["task_id"] for case in cases)),
        "reference_sidecar_is_assistant_invisible": True,
        "final_holdout_read": False,
    }
    return cases, references, report


def build_qasa_unannotated_dataset(path: Path | str = QASA_DEFAULT_SOURCE, *,
                                   split_seed: str = "g09-qasa-v1",
                                   max_sources: int = 30,
                                   min_groups_per_split: int = 1) -> dict:
    """Create a group-disjoint task-only cohort plus a sealed-reference sidecar.

    The returned dataset intentionally has no rubrics.  Callers must pass its
    ``dataset`` through ``open_research_ai_workflow`` before ``open_benchmark``
    freezing.  All source groups are visible in the vendored file, so the
    output is suitable for development/selection diagnostics; it must not be
    described as a blind final benchmark.
    """
    if not isinstance(split_seed, str) or not split_seed.strip():
        raise ValueError("split_seed must be non-empty")
    cases, references, report = load_qasa_cases(path, max_sources=max_sources)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for case, reference in zip(cases, references):
        by_group[case["group_id"]].append((case, reference))
    groups = sorted(by_group, key=lambda group: sha256_text(split_seed + "\0" + group))
    if len(groups) < 3 * min_groups_per_split:
        raise ValueError("QASA has too few paper groups for requested split minima")
    n_dev = max(min_groups_per_split, int(len(groups) * 0.5))
    n_selection = max(min_groups_per_split, int(len(groups) * 0.25))
    if n_dev + n_selection >= len(groups):
        n_dev, n_selection = min_groups_per_split, min_groups_per_split
    split_groups = {
        "dev_optimize": set(groups[:n_dev]),
        "selection_gate": set(groups[n_dev:n_dev + n_selection]),
        "final_holdout": set(groups[n_dev + n_selection:]),
    }
    partitions = {split: [] for split in split_groups}
    references_by_task = {}
    for case, reference in zip(cases, references):
        split = next(name for name, owned in split_groups.items() if case["group_id"] in owned)
        partitions[split].append(case)
        references_by_task[reference["task_id"]] = {**reference, "split": split}
    if any(not partitions[split] for split in partitions):
        raise ValueError("QASA split construction produced an empty partition")
    report = {**report,
        "split_seed": split_seed,
        "groups_by_split": {split: len(owned) for split, owned in split_groups.items()},
        "tasks_by_split": {split: len(rows) for split, rows in partitions.items()},
        "group_disjoint": all(not (split_groups[left] & split_groups[right])
                               for index, left in enumerate(split_groups)
                               for right in list(split_groups)[index + 1:]),
        "rubrics_authored": False,
        "assistant_answers_generated": False,
        "dataset_sha256": sha256_obj(partitions),
        "reference_sidecar_sha256": sha256_obj(references_by_task),
    }
    provenance = {
        "source": "vendored ScholarQABench QASA single-paper task data",
        "revision": QASA_PROTOCOL,
        "license": "ODC-BY aggregate; constituent dataset terms retained in vendor README",
        "reference_origin": "upstream QASA answer and gold context indices (evaluator-only sidecar)",
        "license_policy": "personal_project_provenance_only",
    }
    # Match the existing task-first authoring contract exactly; this lets the
    # standard independent rubric workflow consume the adapter output without
    # a special-case path.
    dataset = {"schema": "open-research-unannotated-dataset-v1",
        "provenance": provenance, "partitions": partitions}
    return {"dataset": dataset, "references": references_by_task, "report": report}

