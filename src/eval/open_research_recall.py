"""High-recall source candidate staging for G09.

The source adapter deliberately keeps a strict four-dimension literal PICO
gate for deterministic source acceptance.  That gate is unsuitable as the
first retrieval stage: a search result can be relevant while omitting one
PICO phrase from its title/abstract.  This module stages such results for a
later, separately audited semantic review.  It never promotes a row into a
source group and never creates labels, answers, scores, or a cohort.

The design follows the two-stage pattern used by the vendored PaperQA search
path: broad/multiple retrieval first, evidence validation and ranking later.
All operations here are offline and deterministic.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from typing import Any

from ..util import sha256_obj
from .open_research_cohort import HISTORICAL_DOI_PREFIX_EXCLUSIONS
from .open_research_sources import PICO_DIMENSIONS, assess_topic_pico_relevance

RECALL_LEDGER_SCHEMA = "g09-high-recall-candidate-ledger-v1"
SEMANTIC_REVIEW_PLAN_SCHEMA = "g09-recall-semantic-review-plan-v1"
DEFAULT_MIN_MATCHED_DIMENSIONS = 2
MAX_MATCHED_DIMENSIONS = len(PICO_DIMENSIONS)
SEMANTIC_REVIEW_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "g09_recall_semantic_review",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "semantic_verdict": {"type": "string", "enum": [
                    "accept_candidate", "reject_candidate", "uncertain"]},
                "matched_dimensions": {"type": "array", "items": {"type": "string",
                    "enum": list(PICO_DIMENSIONS)}, "maxItems": 4},
                "missing_dimensions": {"type": "array", "items": {"type": "string",
                    "enum": list(PICO_DIMENSIONS)}, "maxItems": 4},
                "evidence_terms": {"type": "array", "items": {"type": "string",
                    "minLength": 1, "maxLength": 160}, "maxItems": 8},
                "rationale": {"type": "string", "minLength": 1, "maxLength": 1200},
            },
            "required": ["semantic_verdict", "matched_dimensions", "missing_dimensions",
                         "evidence_terms", "rationale"],
            "additionalProperties": False,
        },
    },
}


def _identity(row: Mapping[str, Any]) -> tuple[str, str]:
    doi = row.get("doi")
    pmcid = row.get("pmcid")
    return (
        doi.strip().casefold() if isinstance(doi, str) else "",
        pmcid.strip().casefold() if isinstance(pmcid, str) else "",
    )


def _valid_doi(doi: str) -> bool:
    return bool(doi) and not any(doi.startswith(prefix)
        for prefix in HISTORICAL_DOI_PREFIX_EXCLUSIONS)


def _valid_pmcid(pmcid: str) -> bool:
    return bool(re.fullmatch(r"pmc\d+", pmcid))


def _matched_count(relevance: Mapping[str, Any]) -> int:
    matched = relevance.get("matched_terms", {})
    if not isinstance(matched, Mapping):
        return 0
    return sum(bool(matched.get(dimension)) for dimension in PICO_DIMENSIONS)


def stage_topic_candidates(
    records: Iterable[Mapping[str, Any]],
    *,
    topic_spec: Mapping[str, Any],
    topic_id: str,
    known_dois: Iterable[str] = (),
    known_pmcids: Iterable[str] = (),
    observed_dois: Iterable[str] = (),
    observed_pmcids: Iterable[str] = (),
    min_matched_dimensions: int = DEFAULT_MIN_MATCHED_DIMENSIONS,
    source_rank_start: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stage eligible near-misses and explain every skipped row.

    A row is staged only when it has stable DOI+PMCID identity, an abstract of
    at least 400 characters, is outside the historical exclusion set, is not
    already known/observed, and matches at least ``min_matched_dimensions`` of
    the four PICO dimensions.  Rows matching all four are included too, but
    remain ``needs_semantic_review`` until the unchanged strict source gate is
    re-run by the caller.
    """
    if not isinstance(topic_id, str) or not topic_id.strip():
        raise ValueError("topic_id is required")
    if type(min_matched_dimensions) is not int or not 1 <= min_matched_dimensions <= MAX_MATCHED_DIMENSIONS:
        raise ValueError("min_matched_dimensions must be between 1 and 4")
    known_doi_set = {str(value).strip().casefold() for value in known_dois if str(value).strip()}
    known_pmcid_set = {str(value).strip().casefold() for value in known_pmcids if str(value).strip()}
    observed_doi_set = {str(value).strip().casefold() for value in observed_dois if str(value).strip()}
    observed_pmcid_set = {str(value).strip().casefold() for value in observed_pmcids if str(value).strip()}
    staged, exclusions = [], []
    local_dois, local_pmcids = set(), set()
    for offset, raw in enumerate(records):
        row = dict(raw) if isinstance(raw, Mapping) else None
        if row is None:
            exclusions.append({"source_rank": source_rank_start + offset, "reason": "invalid_record"})
            continue
        doi, pmcid = _identity(row)
        reason = None
        if not _valid_doi(doi):
            reason = "missing_or_forbidden_doi"
        elif not _valid_pmcid(pmcid):
            reason = "missing_pmcid"
        elif doi in known_doi_set or pmcid in known_pmcid_set:
            reason = "known_identity"
        elif doi in observed_doi_set or pmcid in observed_pmcid_set:
            reason = "observed_identity"
        elif doi in local_dois or pmcid in local_pmcids:
            reason = "duplicate_identity_in_topic"
        abstract = row.get("abstractText")
        if reason is None and (not isinstance(abstract, str) or len(abstract.strip()) < 400):
            reason = "abstract_missing_or_too_short"
        if reason is not None:
            exclusions.append({"source_rank": source_rank_start + offset,
                "doi": doi or None, "pmcid": pmcid or None, "reason": reason})
            continue
        relevance = assess_topic_pico_relevance(row, dict(topic_spec))
        matched_count = _matched_count(relevance)
        local_dois.add(doi)
        local_pmcids.add(pmcid)
        if matched_count < min_matched_dimensions:
            exclusions.append({"source_rank": source_rank_start + offset, "doi": doi,
                "pmcid": pmcid, "reason": "below_minimum_pico_recall",
                "matched_dimensions": matched_count, "pico_relevance": relevance})
            continue
        staged.append({
            "topic_id": topic_id,
            "doi": doi,
            "pmcid": pmcid,
            "title": row.get("title") if isinstance(row.get("title"), str) else "",
            "abstractText": abstract,
            "license": row.get("license") if isinstance(row.get("license"), str) else None,
            "source_rank": source_rank_start + offset,
            "matched_dimensions": matched_count,
            "pico_relevance": relevance,
            "candidate_status": "needs_semantic_review",
            "accepted_into_source_group": False,
            "label": None,
            "score": None,
        })
    staged.sort(key=lambda row: (-row["matched_dimensions"], row["source_rank"], row["doi"], row["pmcid"]))
    return staged, exclusions


def build_recall_ledger(
    topic_records: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    topic_specs: Mapping[str, Mapping[str, Any]],
    known_dois: Iterable[str] = (),
    known_pmcids: Iterable[str] = (),
    min_matched_dimensions: int = DEFAULT_MIN_MATCHED_DIMENSIONS,
    parent_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic, diagnostic-only high-recall ledger."""
    if not isinstance(topic_records, Mapping) or not isinstance(topic_specs, Mapping):
        raise ValueError("topic_records and topic_specs must be mappings")
    if set(topic_records) != set(topic_specs):
        raise ValueError("topic records/specs must contain the same topic IDs")
    all_dois = {str(value).strip().casefold() for value in known_dois if str(value).strip()}
    all_pmcids = {str(value).strip().casefold() for value in known_pmcids if str(value).strip()}
    topics, exclusions = {}, []
    for topic_id in sorted(topic_specs):
        staged, skipped = stage_topic_candidates(topic_records[topic_id], topic_spec=topic_specs[topic_id],
            topic_id=topic_id, known_dois=all_dois, known_pmcids=all_pmcids,
            min_matched_dimensions=min_matched_dimensions)
        topics[topic_id] = staged
        exclusions.extend({**row, "topic_id": topic_id} for row in skipped)
        # A source accepted/observed in an earlier topic cannot enter another
        # topic through the other identity half.
        for row in staged:
            all_dois.add(row["doi"])
            all_pmcids.add(row["pmcid"])
    body: dict[str, Any] = {
        "schema": RECALL_LEDGER_SCHEMA,
        "status": "diagnostic_only_needs_semantic_review",
        "parent_artifact_sha256": parent_artifact_sha256,
        "min_matched_dimensions": min_matched_dimensions,
        "topics": topics,
        "exclusions": exclusions,
        "candidate_count": sum(len(rows) for rows in topics.values()),
        "accepted_source_group_count": 0,
        "labels_created": False,
        "scores_created": False,
        "cohort_created": False,
        "network_calls": 0,
        "model_calls": 0,
    }
    body["ledger_sha256"] = sha256_obj(body)
    return body


def validate_recall_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if a recall ledger has been mutated into a scoring artifact."""
    if not isinstance(ledger, Mapping) or ledger.get("schema") != RECALL_LEDGER_SCHEMA:
        raise ValueError("unsupported recall ledger schema")
    expected = sha256_obj({key: value for key, value in ledger.items() if key != "ledger_sha256"})
    if ledger.get("ledger_sha256") != expected:
        raise ValueError("recall ledger hash mismatch")
    if ledger.get("status") != "diagnostic_only_needs_semantic_review":
        raise ValueError("recall ledger must remain diagnostic-only")
    for key in ("accepted_source_group_count", "labels_created", "scores_created", "cohort_created", "network_calls", "model_calls"):
        if ledger.get(key) not in ({"accepted_source_group_count": 0, "labels_created": False,
            "scores_created": False, "cohort_created": False, "network_calls": 0,
            "model_calls": 0}[key],):
            raise ValueError(f"recall ledger illegally contains {key}")
    identities: set[tuple[str, str]] = set()
    topics = ledger.get("topics")
    if not isinstance(topics, Mapping):
        raise ValueError("recall ledger topics must be a mapping")
    candidate_count = 0
    for topic_id, rows in topics.items():
        if not isinstance(topic_id, str) or not isinstance(rows, list):
            raise ValueError("malformed recall topic")
        for row in rows:
            if not isinstance(row, Mapping) or row.get("candidate_status") != "needs_semantic_review":
                raise ValueError("recall row is not staged for semantic review")
            identity = _identity(row)
            if not _valid_doi(identity[0]) or not _valid_pmcid(identity[1]) or identity in identities:
                raise ValueError("duplicate or invalid recall identity")
            if row.get("accepted_into_source_group") is not False or row.get("label") is not None or row.get("score") is not None:
                raise ValueError("recall row contains downstream decision fields")
            identities.add(identity)
            candidate_count += 1
    if ledger.get("candidate_count") != candidate_count:
        raise ValueError("recall candidate count mismatch")
    return {"status": "valid", "candidate_count": candidate_count,
        "topic_count": len(topics), "ledger_sha256": ledger["ledger_sha256"]}


def _review_prompt(candidate: Mapping[str, Any], topic_spec: Mapping[str, Any]) -> str:
    """Build a blinded, source-grounded semantic review prompt."""
    dimensions = {dimension: topic_spec.get(field, [])
        for dimension, field in PICO_DIMENSIONS.items()}
    return (
        "Review one retrieved research article as a candidate for this frozen PICO slot. "
        "Use only the supplied title and abstract. This is a recall review, not a source "
        "acceptance or answer-label decision. Mark uncertain when the abstract does not "
        "support a dimension. Copy short evidence terms exactly from the abstract.\n\n"
        f"PICO dimensions: {dimensions}\n"
        f"Title: {candidate.get('title', '')}\n"
        f"Abstract: {candidate.get('abstractText', '')}"
    )


def build_semantic_review_plan(
    ledger: Mapping[str, Any], *, ledger_file_sha256: str,
    topic_specs: Mapping[str, Mapping[str, Any]],
    model_id: str = "Qwen/Qwen3-8B",
    max_calls: int | None = None,
) -> dict[str, Any]:
    """Create a hash-bound, zero-call plan for semantic review of staged rows."""
    validate_recall_ledger(ledger)
    if not isinstance(ledger_file_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", ledger_file_sha256):
        raise ValueError("ledger_file_sha256 must be a lowercase SHA-256")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id is required")
    candidates = []
    for topic_id, rows in ledger["topics"].items():
        if topic_id not in topic_specs:
            raise ValueError(f"missing topic spec for {topic_id}")
        for row in rows:
            case_id = f"{topic_id}:{row['doi']}"
            prompt = _review_prompt(row, topic_specs[topic_id])
            candidates.append({"case_id": case_id, "topic_id": topic_id,
                "doi": row["doi"], "pmcid": row["pmcid"],
                "candidate_sha256": sha256_obj(row),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt": prompt})
    candidates.sort(key=lambda row: row["case_id"])
    if max_calls is None:
        max_calls = len(candidates)
    if type(max_calls) is not int or max_calls != len(candidates):
        raise ValueError("semantic review max_calls must equal staged candidate count")
    body: dict[str, Any] = {
        "schema": SEMANTIC_REVIEW_PLAN_SCHEMA,
        "status": "zero_call_plan_pending_review",
        "ledger_sha256": ledger["ledger_sha256"],
        "ledger_file_sha256": ledger_file_sha256,
        "model_id": model_id,
        "max_calls": max_calls,
        "max_physical_attempts": max_calls,
        "hard_cost_usd": "0.00",
        "response_format": SEMANTIC_REVIEW_FORMAT,
        "response_format_sha256": sha256_obj(SEMANTIC_REVIEW_FORMAT),
        "candidates": candidates,
        "accepted_source_group_count": 0,
        "labels_created": False,
        "scores_created": False,
        "cohort_created": False,
        "network_calls": 0,
        "model_calls": 0,
        "dispatch_authorized": False,
    }
    body["plan_sha256"] = sha256_obj(body)
    return body


def validate_semantic_review_plan(plan: Mapping[str, Any], ledger: Mapping[str, Any], *, ledger_file_sha256: str) -> dict[str, Any]:
    """Validate the plan before any provider is constructed or called."""
    validate_recall_ledger(ledger)
    if not isinstance(plan, Mapping) or plan.get("schema") != SEMANTIC_REVIEW_PLAN_SCHEMA:
        raise ValueError("unsupported semantic review plan")
    expected = sha256_obj({key: value for key, value in plan.items() if key != "plan_sha256"})
    if plan.get("plan_sha256") != expected or plan.get("ledger_sha256") != ledger.get("ledger_sha256"):
        raise ValueError("semantic review plan hash/ledger binding mismatch")
    if plan.get("ledger_file_sha256") != ledger_file_sha256:
        raise ValueError("semantic review plan file binding mismatch")
    if plan.get("status") != "zero_call_plan_pending_review" or plan.get("dispatch_authorized") is not False:
        raise ValueError("semantic review plan is not a zero-call pending plan")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or plan.get("max_calls") != len(candidates) or plan.get("max_physical_attempts") != len(candidates):
        raise ValueError("semantic review plan candidate budget mismatch")
    for key, expected_value in (("accepted_source_group_count", 0), ("labels_created", False),
                                ("scores_created", False), ("cohort_created", False),
                                ("network_calls", 0), ("model_calls", 0), ("hard_cost_usd", "0.00")):
        if plan.get(key) != expected_value:
            raise ValueError(f"semantic review plan illegally sets {key}")
    return {"status": "passed_zero_call_semantic_review_plan", "plan_sha256": plan["plan_sha256"],
        "candidate_count": len(candidates), "network_calls": 0, "model_calls": 0,
        "dispatch_authorized": False}


def parse_semantic_review_output(content: str, *, candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Parse one review without promoting it into source acceptance."""
    import json
    if not isinstance(content, str):
        raise ValueError("semantic review output must be text")
    value = json.loads(content)
    expected = set(SEMANTIC_REVIEW_FORMAT["json_schema"]["schema"]["required"])
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("semantic review output schema mismatch")
    if value["semantic_verdict"] not in {"accept_candidate", "reject_candidate", "uncertain"}:
        raise ValueError("invalid semantic review verdict")
    dims = set(PICO_DIMENSIONS)
    matched, missing = value["matched_dimensions"], value["missing_dimensions"]
    if (not isinstance(matched, list) or not isinstance(missing, list)
            or len(matched) != len(set(matched)) or len(missing) != len(set(missing))
            or not set(matched) <= dims or not set(missing) <= dims
            or set(matched) & set(missing)):
        raise ValueError("semantic review dimensions are invalid")
    if not isinstance(value["evidence_terms"], list) or any(
            not isinstance(term, str) or not term.strip() for term in value["evidence_terms"]):
        raise ValueError("semantic review evidence_terms are invalid")
    if not isinstance(value["rationale"], str) or not value["rationale"].strip():
        raise ValueError("semantic review rationale is required")
    return {**value, "case_id": f"{candidate['topic_id']}:{candidate['doi']}",
        "candidate_status": "semantic_reviewed_not_accepted",
        "accepted_into_source_group": False, "label": None, "score": None}

