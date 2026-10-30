"""Stable, article-disjoint nested cohorts for Evidence Inference exploration.

This module only selects IDs from the official training-derived optimize split.
It never reads official validation/test rows or invokes retrieval/model code.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from ..util import sha256_obj


DATASET = "evidence_inference_v2_abstract"
COHORT_VERSION = "evidence-inference-article-nested-v2"
DEFAULT_SALT = "rsi-evidence-inference-article-nested-2026-10-05-v1"
LABELS = (
    "significantly increased",
    "significantly decreased",
    "no significant difference",
)
EVALUATION_SCOPE_NAMES = ("search_validation", "internal_confirmation")
ARTICLE_RATIOS = {
    "mipro_train": 80,
    "search_validation": 10,
    "internal_confirmation": 10,
}


def prior_cohort_article_ids(cohort: dict[str, Any], *,
                             scopes: tuple[str, ...] | None = None) -> set[str]:
    """Return article IDs to exclude, optionally only from prior evaluation scopes.

    Prior train articles remain reusable as training data. Previously evaluated
    articles must stay out of new search/confirmation cohorts to limit reuse.
    """
    entries = cohort.get("scopes") or {}
    selected_scopes = tuple(entries) if scopes is None else tuple(scopes)
    missing = set(selected_scopes) - set(entries)
    if missing:
        raise ValueError(f"prior cohort lacks exclusion scopes: {sorted(missing)}")
    return {
        str(article_id)
        for scope in selected_scopes
        for article_id in (entries[scope].get("article_ids") or [])
    }


def _rank(salt: str, *parts: str) -> str:
    value = "|".join((salt, *parts)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _article_scope(article_id: str, salt: str,
                   ratios: dict[str, int] | None = None) -> str:
    ratios = ratios or ARTICLE_RATIOS
    bucket = int(_rank(salt, "article-scope", article_id)[:8], 16) % 100
    if bucket < ratios["mipro_train"]:
        return "mipro_train"
    if bucket < ratios["mipro_train"] + ratios["search_validation"]:
        return "search_validation"
    return "internal_confirmation"


def build_nested_cohort(
    rows: list[dict[str, Any]],
    *,
    train_per_class: int = 10,
    search_validation_per_class: int = 6,
    confirmation_per_class: int = 6,
    minimum_annotation_votes: int = 1,
    excluded_article_ids: set[str] | None = None,
    article_scope_ratios: dict[str, int] | None = None,
    salt: str = DEFAULT_SALT,
) -> dict[str, Any]:
    """Select balanced task IDs while keeping every article in one scope.

    Rows not selected for the bounded cohort remain unused. The cohort is
    deterministic for a fixed input, salt, and requested class counts.
    """
    if min(train_per_class, search_validation_per_class, confirmation_per_class) <= 0:
        raise ValueError("each nested scope must request at least one task per class")
    if minimum_annotation_votes <= 0:
        raise ValueError("minimum_annotation_votes must be positive")
    ratios = dict(article_scope_ratios or ARTICLE_RATIOS)
    if (set(ratios) != set(ARTICLE_RATIOS)
            or any(not isinstance(value, int) or value <= 0 for value in ratios.values())
            or sum(ratios.values()) != 100):
        raise ValueError("article scope ratios must be positive integers summing to 100")

    targets = {
        "mipro_train": train_per_class,
        "search_validation": search_validation_per_class,
        "internal_confirmation": confirmation_per_class,
    }
    by_scope_label_article: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        scope: {label: defaultdict(list) for label in LABELS}
        for scope in ARTICLE_RATIOS
    }
    seen_ids: set[str] = set()
    article_scopes: dict[str, str] = {}
    excluded_articles = set(excluded_article_ids or ())
    vote_count_histogram: dict[str, int] = defaultdict(int)
    eligible_vote_count_histogram: dict[str, int] = defaultdict(int)

    for row in rows:
        task_id = str(row.get("task_id") or "")
        article_id = str(row.get("group_id") or "")
        options = row.get("options")
        try:
            correct_index = int(row.get("correct_index", -1))
            label = str(options[correct_index]) if isinstance(options, list) and 0 <= correct_index < len(options) else ""
        except (TypeError, ValueError):
            label = ""
        if row.get("dataset") != DATASET or row.get("split") != "dev_optimize":
            raise ValueError("nested cohort input must contain only Evidence Inference dev_optimize rows")
        if not task_id or not article_id or label not in LABELS:
            raise ValueError("nested cohort row is missing task ID, article group, or valid label")
        metadata = row.get("metadata")
        if minimum_annotation_votes > 1 and not isinstance(metadata, dict):
            raise ValueError("annotation vote metadata is required when minimum_annotation_votes > 1")
        raw_vote_count = metadata.get("valid_abstract_label_vote_count", 1) if isinstance(metadata, dict) else 1
        try:
            vote_count = int(raw_vote_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid annotation vote count for {task_id}") from exc
        if vote_count < 1:
            raise ValueError(f"invalid annotation vote count for {task_id}")
        vote_count_histogram[str(vote_count)] += 1
        if article_id in excluded_articles:
            continue
        if vote_count < minimum_annotation_votes:
            continue
        eligible_vote_count_histogram[str(vote_count)] += 1
        if task_id in seen_ids:
            raise ValueError(f"duplicate Evidence Inference task ID: {task_id}")
        seen_ids.add(task_id)
        scope = _article_scope(article_id, salt, ratios)
        article_scopes[article_id] = scope
        by_scope_label_article[scope][label][article_id].append(row)

    selected: dict[str, list[dict[str, Any]]] = {scope: [] for scope in ARTICLE_RATIOS}
    for scope, per_label in by_scope_label_article.items():
        target = targets[scope]
        for label in LABELS:
            articles = per_label[label]
            article_order = sorted(articles, key=lambda article: _rank(salt, "article-row", scope, label, article))
            picked = 0
            # Round-robin over articles, so repeated prompts from one paper do
            # not crowd out independent papers while enough articles exist.
            for round_index in range(max((len(v) for v in articles.values()), default=0)):
                for article_id in article_order:
                    candidates = sorted(
                        articles[article_id],
                        key=lambda row: _rank(salt, "task-row", scope, label, str(row["task_id"])),
                    )
                    if round_index >= len(candidates):
                        continue
                    selected[scope].append(candidates[round_index])
                    picked += 1
                    if picked == target:
                        break
                if picked == target:
                    break
            if picked != target:
                raise ValueError(
                    f"insufficient article-disjoint {label} rows in {scope}: {picked}/{target}"
                )

    scopes_by_article: dict[str, set[str]] = defaultdict(set)
    ids_by_scope: dict[str, list[str]] = {}
    class_counts: dict[str, dict[str, int]] = {}
    article_ids_by_scope: dict[str, list[str]] = {}
    for scope, scope_rows in selected.items():
        ids = [str(row["task_id"]) for row in scope_rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate IDs selected within {scope}")
        ids_by_scope[scope] = sorted(ids)
        class_counts[scope] = {label: 0 for label in LABELS}
        articles: set[str] = set()
        for row in scope_rows:
            options = row["options"]
            label = str(options[int(row["correct_index"])])
            class_counts[scope][label] += 1
            article_id = str(row["group_id"])
            articles.add(article_id)
            scopes_by_article[article_id].add(scope)
        article_ids_by_scope[scope] = sorted(articles)

    if any(len(scope_set) != 1 for scope_set in scopes_by_article.values()):
        raise ValueError("article group crossed nested cohort scopes")
    all_selected = [task_id for task_ids in ids_by_scope.values() for task_id in task_ids]
    if len(all_selected) != len(set(all_selected)):
        raise ValueError("nested cohort task IDs overlap")

    return {
        "dataset": DATASET,
        "cohort_version": COHORT_VERSION,
        "minimum_annotation_votes": minimum_annotation_votes,
        "excluded_article_count": len(excluded_articles),
        "excluded_article_ids_sha256": sha256_obj(sorted(excluded_articles)),
        "annotation_vote_count_histogram": dict(sorted(vote_count_histogram.items(), key=lambda item: int(item[0]))),
        "eligible_annotation_vote_count_histogram": dict(
            sorted(eligible_vote_count_histogram.items(), key=lambda item: int(item[0]))
        ),
        "salt": salt,
        "eligible_for_confirmation": False,
        "status": "exploratory_nested_cohort_not_preregistered_confirmation",
        "official_validation_or_test_read": False,
        "article_scope_ratios": ratios,
        "targets_per_class": targets,
        "scopes": {
            scope: {
                "task_ids": ids_by_scope[scope],
                "task_count": len(ids_by_scope[scope]),
                "article_ids": article_ids_by_scope[scope],
                "article_count": len(article_ids_by_scope[scope]),
                "class_counts": class_counts[scope],
            }
            for scope in ARTICLE_RATIOS
        },
        "unused_input_task_count": len(rows) - len(all_selected),
    }
