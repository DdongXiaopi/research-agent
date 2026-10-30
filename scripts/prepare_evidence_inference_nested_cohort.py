#!/usr/bin/env python3
"""Freeze a bounded, article-disjoint EI cohort for optimizer diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.evidence_inference_cohort import (
    EVALUATION_SCOPE_NAMES, build_nested_cohort, prior_cohort_article_ids,
)
from src.data.evidence_inference_v2 import DATASET, load_tasks
from src.evolution.evidence_inference_promotion import (
    PROTOCOL as EI_MIPRO_PROTOCOL,
    compute_mipro_plan_sha256,
)
from src.util import artifacts_path, path, read_json, sha256_file, sha256_obj, write_json


def candidate_confirmation_binding(candidate_report_path: Path) -> tuple[dict, Path]:
    """Authenticate a preselected MIPRO candidate before freezing its cohort."""
    report_path = candidate_report_path.resolve()
    report = read_json(report_path)
    if (report.get("status") != "completed"
            or report.get("protocol") != EI_MIPRO_PROTOCOL
            or report.get("candidate_status") != "internal_gate_passed"
            or report.get("eligible_for_official_selection_gate") is not True
            or report.get("confirmation_eligible") is not False):
        raise ValueError("independent confirmation requires a preselected v6 MIPRO candidate")
    if any(report.get(flag) is not False for flag in (
            "selection_gate_read", "official_validation_read", "official_test_read")):
        raise ValueError("candidate report has already read a protected selection split")
    plan = report.get("plan")
    if (not isinstance(plan, dict) or plan.get("protocol") != EI_MIPRO_PROTOCOL
            or plan.get("plan_sha256") != compute_mipro_plan_sha256(plan)):
        raise ValueError("candidate report has no valid current MIPRO plan")
    candidate_id = str(report.get("candidate_id") or "")
    source_cohort_sha = str(plan.get("cohort_sha256") or "")
    source_cohort_path = Path(str(report.get("cohort_path") or ""))
    if not candidate_id or not source_cohort_sha or not source_cohort_path.is_file():
        raise ValueError("candidate report lacks a candidate identity or source cohort")
    source_cohort = read_json(source_cohort_path)
    if source_cohort.get("cohort_sha256") != source_cohort_sha:
        raise ValueError("candidate report source cohort differs from its frozen plan")
    unsigned_source_cohort = {
        key: value for key, value in source_cohort.items()
        if key != "cohort_sha256"
    }
    if sha256_obj(unsigned_source_cohort) != source_cohort_sha:
        raise ValueError("candidate report source cohort self-hash is invalid")
    if int(plan.get("minimum_annotation_votes") or 0) < 2:
        raise ValueError("candidate source cohort lacks two-vote annotation support")
    report_sha = sha256_file(report_path)
    return ({
        "candidate_id": candidate_id,
        "candidate_report_sha256": report_sha,
        "source_plan_sha256": str(plan["plan_sha256"]),
        "source_cohort_sha256": source_cohort_sha,
        "created_after_candidate_preselection": True,
    }, source_cohort_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-class", type=int, default=10)
    parser.add_argument("--search-validation-per-class", type=int, default=6)
    parser.add_argument("--confirmation-per-class", type=int, default=6)
    parser.add_argument(
        "--article-scope-ratios", type=int, nargs=3, metavar=("TRAIN", "SEARCH", "INTERNAL"),
        default=tuple((80, 10, 10)),
        help="Article allocation percentages; must sum to 100 (default: 80 10 10).",
    )
    parser.add_argument(
        "--minimum-annotation-votes",
        type=int,
        default=1,
        help="Require this many valid abstract label votes for every selected task.",
    )
    parser.add_argument("--salt", default=None)
    parser.add_argument(
        "--for-candidate-report", type=Path, default=None,
        help=("freeze a new independent-confirmation cohort bound to an internally "
              "preselected current-protocol MIPRO candidate"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--exclude-cohort",
        type=Path,
        action="append",
        default=None,
        help=("Exclude every article used by prior cohorts; pass this option once per "
              "earlier cohort."),
    )
    parser.add_argument(
        "--exclude-evaluation-cohort",
        type=Path,
        action="append",
        default=None,
        help=("Exclude only search_validation/internal_confirmation articles from a "
              "prior cohort; its train articles remain reusable as training data."),
    )
    parser.add_argument(
        "--exclude-search-only-cohort",
        type=Path,
        action="append",
        default=None,
        help=("Exclude only search_validation articles from a prior cohort whose "
              "internal_confirmation scope was never accessed."),
    )
    args = parser.parse_args(argv)

    confirmation_binding = None
    confirmation_source_cohort = None
    if args.for_candidate_report:
        confirmation_binding, confirmation_source_cohort = candidate_confirmation_binding(
            args.for_candidate_report)
        if args.minimum_annotation_votes < 2:
            parser.error("independent-confirmation cohorts require at least two annotation votes")
        if args.confirmation_per_class < 30:
            parser.error(
                "independent-confirmation scope must include at least 90 tasks "
                "(30 per class)")

    excluded_articles: set[str] = set()
    excluded_cohort_sha256s = []
    exclusion_policies = []
    full_exclusion_paths = list(args.exclude_cohort or [])
    if confirmation_source_cohort is not None:
        resolved = {item.resolve() for item in full_exclusion_paths}
        if confirmation_source_cohort.resolve() not in resolved:
            full_exclusion_paths.append(confirmation_source_cohort)
    for prior_path in full_exclusion_paths:
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        excluded_articles.update(prior_cohort_article_ids(prior))
        cohort_sha256 = prior.get("cohort_sha256")
        if cohort_sha256:
            excluded_cohort_sha256s.append(str(cohort_sha256))
            exclusion_policies.append({
                "cohort_sha256": str(cohort_sha256), "scopes": "all"})
    for prior_path in args.exclude_evaluation_cohort or []:
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        excluded_articles.update(prior_cohort_article_ids(
            prior, scopes=EVALUATION_SCOPE_NAMES))
        cohort_sha256 = prior.get("cohort_sha256")
        if cohort_sha256:
            excluded_cohort_sha256s.append(str(cohort_sha256))
            exclusion_policies.append({
                "cohort_sha256": str(cohort_sha256),
                "scopes": list(EVALUATION_SCOPE_NAMES)})

    for prior_path in args.exclude_search_only_cohort or []:
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        excluded_articles.update(prior_cohort_article_ids(
            prior, scopes=("search_validation",)))
        cohort_sha256 = prior.get("cohort_sha256")
        if cohort_sha256:
            excluded_cohort_sha256s.append(str(cohort_sha256))
            exclusion_policies.append({
                "cohort_sha256": str(cohort_sha256),
                "scopes": ["search_validation"]})

    salt = args.salt
    if confirmation_binding:
        if salt is not None:
            parser.error("candidate-bound confirmation cohort salt is derived from its frozen source")
        salt = sha256_obj({
            "protocol": "evidence-inference-independent-confirmation-cohort-v1",
            **confirmation_binding,
        })
    cohort = build_nested_cohort(
        load_tasks(splits=["dev_optimize"]),
        train_per_class=args.train_per_class,
        search_validation_per_class=args.search_validation_per_class,
        confirmation_per_class=args.confirmation_per_class,
        minimum_annotation_votes=args.minimum_annotation_votes,
        excluded_article_ids=excluded_articles,
        article_scope_ratios=dict(zip(
            ("mipro_train", "search_validation", "internal_confirmation"),
            args.article_scope_ratios)),
        **({"salt": salt} if salt else {}),
    )
    manifest_path = path("data", "manifests", f"{DATASET}.json")
    manifest = read_json(manifest_path)
    cohort["source_manifest_sha256"] = sha256_file(manifest_path)
    cohort["source_train_ids_sha256"] = manifest["splits"]["dev_optimize"]["ids_sha256"]
    cohort["candidate_runtime_uses_live_retrieval"] = False
    cohort["selection_or_confirmation_evidence"] = False
    if confirmation_binding:
        cohort["independent_confirmation_for"] = confirmation_binding
    if excluded_cohort_sha256s:
        cohort["excluded_cohort_sha256s"] = sorted(set(excluded_cohort_sha256s))
        cohort["excluded_cohort_scope_policies"] = sorted(
            exclusion_policies,
            key=lambda item: (item["cohort_sha256"], str(item["scopes"])),
        )
    cohort["cohort_sha256"] = sha256_obj(cohort)

    output = args.output or artifacts_path("probe", "evidence_inference_nested_cohort.json")
    write_json(output, cohort)
    print(f"wrote {output}")
    print(f"cohort_sha256={cohort['cohort_sha256']}")
    counts = {name: entry["task_count"] for name, entry in cohort["scopes"].items()}
    print(f"counts={counts}")
    print(f"minimum_annotation_votes={cohort['minimum_annotation_votes']}")
    for scope, entry in cohort["scopes"].items():
        print(f"{scope}: articles={entry['article_count']} classes={entry['class_counts']}")


if __name__ == "__main__":
    main()
