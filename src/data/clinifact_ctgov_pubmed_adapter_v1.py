"""Strict, unlabeled ClinicalTrials.gov → PubMed CliniFact adapter (v1).

The adapter derives claim direction only from a uniquely matched registered
primary outcome and its two-group mean table. Review packets contain the claim,
PubMed title, and abstract only. Registry values and source identifiers remain
in private provenance. This module does not create labels or authorize
confirmation/promotion.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from ..util import sha256_obj

SOURCE_ADAPTER_SCHEMA = "clinifact-ctgov-pubmed-source-bundle-v1"
PACKET_SCHEMA = "clinifact-blind-ai-reference-packet-v1"
PROTOCOL_VERSION = "clinifact_trial_result_claim_evidence_v5"
EXTRACTION_ALGORITHM = "ctgov-unique-primary-two-group-mean-direction-v1"
ROLE_TARGETS = {"mipro_train": 30, "search_validation": 24, "confirmation": 90}

LABEL_PROMPT = (
    "You are an independent clinical-trial abstract claim verifier. Use only the supplied claim and publication title/abstract. "
    "Classify exactly one label: EVIDENCE means the abstract's reported results directly support the claim; "
    "INCONCLUSIVE means the abstract directly reports an inconclusive, non-significant, mixed, or explicitly uncertain result "
    "for the same outcome and comparison; NEI means the abstract does not provide enough directly relevant information to decide. "
    "Do not infer from background, plausibility, outside knowledge, or registry data. Do not use a dataset label. "
    "For EVIDENCE or INCONCLUSIVE, give one exact contiguous quote from the supplied abstract. For NEI, use an empty quote "
    "if no decisive passage exists. Return JSON only with exactly label, evidence_quote, and rationale."
)


def normalize_text(value: Any) -> str:
    ascii_text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", ascii_text.casefold()).split())


def make_history_exclusion_manifest(*, source_report_sha256: str,
                                    candidate_pool_sha256: str,
                                    excluded_nct_ids: Sequence[str],
                                    excluded_pmids: Sequence[str],
                                    excluded_claims: Sequence[str],
                                    candidate_pool_precleared: bool = True) -> dict[str, Any]:
    """Create a deterministic exclusion index; its completeness is caller-attested."""
    body = {
        "schema": "clinifact-source-history-exclusions-v1",
        "complete": True,
        "candidate_pool_precleared": candidate_pool_precleared,
        "source_report_sha256": source_report_sha256,
        "candidate_pool_sha256": candidate_pool_sha256,
        "excluded_nct_ids": sorted({str(x).upper() for x in excluded_nct_ids}),
        "excluded_pmids": sorted({str(x) for x in excluded_pmids}),
        "excluded_claims_normalized": sorted({normalize_text(x) for x in excluded_claims if normalize_text(x)}),
    }
    return {**body, "manifest_sha256": sha256_obj(body)}


def _validate_history(history: Mapping[str, Any], candidate_pool_sha256: str) -> None:
    unsigned = dict(history)
    digest = unsigned.pop("manifest_sha256", None)
    if (history.get("schema") != "clinifact-source-history-exclusions-v1"
            or history.get("complete") is not True
            or history.get("candidate_pool_precleared") is not True
            or not re.fullmatch(r"[0-9a-f]{64}", str(history.get("source_report_sha256") or ""))
            or history.get("candidate_pool_sha256") != candidate_pool_sha256
            or not digest or sha256_obj(unsigned) != digest):
        raise ValueError("complete, self-hashed exposure exclusions bound to this candidate pool are required")


def _primary_outcome_claim(study: Mapping[str, Any]) -> tuple[str | None, dict[str, Any] | None, str]:
    protocol = study.get("protocolSection") or {}
    primary = (protocol.get("outcomesModule") or {}).get("primaryOutcomes") or []
    measures = ((study.get("resultsSection") or {}).get("outcomeMeasuresModule") or {}).get("outcomeMeasures") or []
    # The production CT.gov detail queries project `protocolSection` and
    # `resultsSection`, so the top-level `hasResults` flag is normally absent
    # from otherwise complete captured records.  A present false flag remains
    # a contradiction and fails closed; an omitted flag is inferred only from
    # the actual, non-empty results section that carries the outcome table.
    if study.get("hasResults") is False or not study.get("resultsSection"):
        return None, None, "results_section_missing"
    if len(primary) != 1:
        return None, None, "registered_primary_outcome_not_unique"
    outcome = primary[0]
    matches = [(index, measure) for index, measure in enumerate(measures)
               if measure.get("type") == "PRIMARY"
               and normalize_text(measure.get("title")) == normalize_text(outcome.get("measure"))
               and normalize_text(measure.get("timeFrame")) == normalize_text(outcome.get("timeFrame"))]
    if len(matches) != 1:
        return None, None, "primary_result_measure_not_unique_exact_match"
    result_measure_index, measure = matches[0]
    if measure.get("paramType") != "MEAN":
        return None, None, "primary_result_measure_not_mean"
    groups = measure.get("groups") or []
    if len(groups) != 2:
        return None, None, "primary_measure_not_exactly_two_groups"
    group_ids = [str(group.get("id") or "") for group in groups]
    group_titles = [str(group.get("title") or "").strip() for group in groups]
    if (not all(group_ids) or len(set(group_ids)) != 2 or not all(group_titles)
            or len({normalize_text(x) for x in group_titles}) != 2):
        return None, None, "primary_measure_groups_not_unique_named"
    classes = measure.get("classes") or []
    categories = [category for cls in classes for category in (cls.get("categories") or [])]
    if len(classes) != 1 or len(categories) != 1:
        return None, None, "primary_measure_mean_category_not_unique"
    measurements = categories[0].get("measurements") or []
    values: dict[str, Decimal] = {}
    measurement_row_indices: dict[str, int] = {}
    for row_index, row in enumerate(measurements):
        gid = str(row.get("groupId") or "")
        raw_value = str(row.get("value") or "").strip()
        if gid not in group_ids or gid in values:
            return None, None, "primary_measure_group_values_not_unique"
        try:
            value = Decimal(raw_value)
        except (InvalidOperation, ValueError):
            return None, None, "primary_measure_value_not_numeric"
        if not value.is_finite():
            return None, None, "primary_measure_value_not_finite"
        values[gid] = value
        measurement_row_indices[gid] = row_index
    if set(values) != set(group_ids):
        return None, None, "primary_measure_group_value_missing"
    left_id, right_id = group_ids
    if values[left_id] == values[right_id]:
        return None, None, "primary_measure_values_tied"
    left, right = group_titles
    direction = "lower" if values[left_id] < values[right_id] else "higher"
    name = str(outcome.get("measure") or "").strip()
    timeframe = str(outcome.get("timeFrame") or "").strip()
    if not name or not timeframe:
        return None, None, "primary_outcome_name_or_timeframe_missing"
    claim = f"At {timeframe}, the mean {name} was {direction} for {left} than for {right}."
    details = {
        "primary_outcome": dict(outcome),
        "primary_outcome_row_index": 0,
        "result_measure_row_index": result_measure_index,
        "measurement_row_indices": measurement_row_indices,
        "result_measure": dict(measure),
        "groups": [dict(group) for group in groups],
        "measurement_values": {left_id: str(values[left_id]), right_id: str(values[right_id])},
        "direction": direction,
        "unit_of_measure": measure.get("unitOfMeasure"),
        "analyses": measure.get("analyses") or [],
    }
    return claim, details, ""


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def _component_ids(cases: list[dict[str, Any]]) -> list[str]:
    union = _UnionFind(len(cases))
    first: dict[tuple[str, str], int] = {}
    for index, case in enumerate(cases):
        keys = []
        for binding in case["private_bindings"]:
            keys.extend((("nct", binding["nct_id"]), ("pmid", binding["pmid"])))
        keys.append(("claim", normalize_text(case["input"]["claim"])))
        paper_text = normalize_text(case["input"]["title"] + " " + case["input"]["abstract"])
        keys.append(("publication_text", paper_text))
        for key in keys:
            previous = first.setdefault(key, index)
            union.union(index, previous)
    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(cases)):
        members[union.find(index)].append(index)
    out = [""] * len(cases)
    for indexes in members.values():
        identity = sorted({
            f"{binding['nct_id']}:{binding['pmid']}"
            for i in indexes for binding in cases[i]["private_bindings"]
        } | {normalize_text(cases[i]["input"]["claim"]) for i in indexes})
        component = sha256_obj(["clinifact-ctgov-pubmed-component-v1", identity])
        for i in indexes:
            out[i] = component
    return out


def build_source_bundle(ctgov_studies: Mapping[str, Mapping[str, Any]],
                        pubmed_crosswalk: Mapping[str, Mapping[str, Any]],
                        candidate_components: Sequence[Mapping[str, Any]], *,
                        source_hashes: Mapping[str, str],
                        history_exclusion: Mapping[str, Any],
                        seed: int = 20261017,
                        targets: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Build blind packets and separate provenance from frozen CT.gov/PubMed captures.

    `candidate_components` must already be the history-screened source pool.
    The returned pool is always exploratory and confirmation-ineligible until
    an independent, class-balanced AI reference and candidate-specific review
    are separately frozen.
    """
    source_hashes = dict(source_hashes)
    required_hashes = {"ctgov_capture_sha256", "pubmed_capture_sha256", "candidate_pool_sha256"}
    if (not required_hashes.issubset(source_hashes)
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(source_hashes[k])) for k in required_hashes)):
        raise ValueError("all source capture and candidate pool SHA-256 bindings are required")
    _validate_history(history_exclusion, source_hashes["candidate_pool_sha256"])
    target_map = dict(targets or ROLE_TARGETS)
    if set(target_map) != set(ROLE_TARGETS) or any(type(v) is not int or v <= 0 for v in target_map.values()):
        raise ValueError("role targets must define positive train/search-validation/confirmation capacities")
    excluded_nct = {str(x).upper() for x in history_exclusion.get("excluded_nct_ids", [])}
    excluded_pmids = {str(x) for x in history_exclusion.get("excluded_pmids", [])}
    excluded_claims = set(history_exclusion.get("excluded_claims_normalized", []))
    exclusions: list[dict[str, Any]] = []
    raw_cases: list[dict[str, Any]] = []
    for component_row_index, component in enumerate(candidate_components):
        component_id = str(component.get("provisional_component_id") or "")
        component_pmids = {str(x) for x in component.get("pmids", [])}
        for nct_id_raw in component.get("nct_ids", []):
            nct_id = str(nct_id_raw).upper()
            if nct_id in excluded_nct:
                exclusions.append({"component_row_index": component_row_index, "nct_id": nct_id,
                                   "reason": "historical_nct_exposure"})
                continue
            study = ctgov_studies.get(nct_id)
            if not study:
                exclusions.append({"component_row_index": component_row_index, "nct_id": nct_id,
                                   "reason": "ctgov_record_missing"})
                continue
            recorded_nct = str((((study.get("protocolSection") or {}).get("identificationModule") or {})
                                .get("nctId") or "")).upper()
            if recorded_nct != nct_id:
                exclusions.append({"component_row_index": component_row_index, "nct_id": nct_id,
                                   "recorded_nct_id": recorded_nct,
                                   "reason": "ctgov_nct_identity_mismatch"})
                continue
            claim, outcome_details, reason = _primary_outcome_claim(study)
            if reason:
                exclusions.append({"component_row_index": component_row_index, "nct_id": nct_id,
                                   "reason": reason})
                continue
            if normalize_text(claim) in excluded_claims:
                exclusions.append({"component_row_index": component_row_index, "nct_id": nct_id,
                                   "reason": "historical_claim_text_exposure"})
                continue
            nct_raw_sha = hashlib.sha256(json.dumps(
                study, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
            references = ((study.get("protocolSection") or {}).get("referencesModule") or {}).get("references") or []
            result_refs = {str(ref.get("pmid")): ref for ref in references
                           if ref.get("type") == "RESULT" and ref.get("pmid")}
            for pmid in sorted(component_pmids & set(result_refs)):
                if pmid in excluded_pmids:
                    exclusions.append({"component_row_index": component_row_index, "nct_id": nct_id,
                                       "pmid": pmid, "reason": "historical_pmid_exposure"})
                    continue
                record = pubmed_crosswalk.get(pmid)
                if not record or record.get("returned") is not True:
                    exclusions.append({"component_row_index": component_row_index, "nct_id": nct_id,
                                       "pmid": pmid, "reason": "pubmed_record_missing"})
                    continue
                if str(record.get("pmid") or "") != pmid:
                    exclusions.append({"component_row_index": component_row_index, "nct_id": nct_id,
                                       "pmid": pmid, "recorded_pmid": record.get("pmid"),
                                       "reason": "pubmed_pmid_identity_mismatch"})
                    continue
                link = next((row for row in record.get("links", [])
                             if str(row.get("nct_id", "")).upper() == nct_id), None)
                title, abstract = str(record.get("title") or "").strip(), str(record.get("abstract") or "").strip()
                citation = str((link or {}).get("result_reference_citation") or "")
                if (not link or link.get("result_reference_title_identity_match") is not True
                        or normalize_text(title) not in normalize_text(citation)):
                    exclusions.append({"component_row_index": component_row_index, "nct_id": nct_id,
                                       "pmid": pmid, "reason": "result_reference_title_mismatch"})
                    continue
                if not abstract:
                    exclusions.append({"component_row_index": component_row_index, "nct_id": nct_id,
                                       "pmid": pmid, "reason": "pubmed_abstract_missing"})
                    continue
                input_value = {"claim": claim, "title": title, "abstract": abstract}
                content_key = tuple(normalize_text(input_value[k]) for k in ("claim", "title", "abstract"))
                raw_cases.append({
                    "input": input_value,
                    "content_key": content_key,
                    "private_bindings": [{
                        "component_row_index": component_row_index,
                        "source_component_id": component_id,
                        "nct_id": nct_id,
                        "pmid": pmid,
                        "result_reference_citation": citation,
                        "raw_nct_record_sha256": nct_raw_sha,
                        "abstract_sha256": hashlib.sha256(abstract.encode("utf-8")).hexdigest(),
                        "primary_outcome": outcome_details["primary_outcome"],
                        "primary_outcome_row_index": outcome_details["primary_outcome_row_index"],
                        "result_measure_row_index": outcome_details["result_measure_row_index"],
                        "measurement_row_indices": outcome_details["measurement_row_indices"],
                        "result_title": outcome_details["result_measure"].get("title"),
                        "result_time_frame": outcome_details["result_measure"].get("timeFrame"),
                        "measurement_values": outcome_details["measurement_values"],
                        "groups": outcome_details["groups"],
                        "unit_of_measure": outcome_details["unit_of_measure"],
                        "analyses": outcome_details["analyses"],
                        "extraction_algorithm": EXTRACTION_ALGORITHM,
                    }],
                })
    # Collapse duplicate task text while retaining every identity in private provenance.
    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in raw_cases:
        existing = dedup.get(row["content_key"])
        if existing:
            existing["private_bindings"].extend(row["private_bindings"])
            exclusions.append({"reason": "duplicate_claim_title_abstract_text",
                               "merged_component_row_index": row["private_bindings"][0]["component_row_index"]})
        else:
            dedup[row["content_key"]] = row
    cases = list(dedup.values())
    component_ids = _component_ids(cases)
    for case, component_id in zip(cases, component_ids):
        case["component_id"] = component_id
        identity = sorted({f"{x['nct_id']}:{x['pmid']}" for x in case["private_bindings"]})
        case["task_id"] = "clinifact:ctgov-pubmed:v1:" + sha256_obj(identity)[:20]
        case["case_id"] = sha256_obj(["clinifact-ctgov-pubmed-case-v1", case["task_id"], case["input"]])[:24]
        case["input_sha256"] = sha256_obj(case["input"])
    independent_components = sorted({case["component_id"] for case in cases})
    minimum = sum(target_map.values())
    capacity = {
        "independent_component_count": len(independent_components),
        "required_by_role": target_map,
        "minimum_required": minimum,
        "eligible": len(independent_components) >= minimum,
        "eligible_for_confirmation": False,
        "reason": "candidate-preselection, blind references, class balance, and candidate-specific review are not complete",
    }
    protocol_sha = sha256_obj({
        "schema": SOURCE_ADAPTER_SCHEMA, "protocol": PROTOCOL_VERSION,
        "extraction_algorithm": EXTRACTION_ALGORITHM, "source_hashes": source_hashes,
        "history_exclusion_sha256": history_exclusion["manifest_sha256"],
        "seed": seed, "targets": target_map,
        "case_inputs": sorted(case["input_sha256"] for case in cases),
    })
    reviewer_packets = {}
    for slot, offset in (("reviewer_a", 0), ("reviewer_b", 1)):
        visible = [{"case_id": c["case_id"], "component_id": c["component_id"],
                    "task_id": c["task_id"], "input": c["input"],
                    "input_sha256": c["input_sha256"]} for c in cases]
        random.Random(seed + offset).shuffle(visible)
        unsigned = {"schema": PACKET_SCHEMA, "protocol_sha256": protocol_sha,
                    "reviewer_slot": slot, "instructions": LABEL_PROMPT, "cases": visible}
        reviewer_packets[slot] = {**unsigned, "packet_sha256": sha256_obj(unsigned)}
    return {
        "schema": SOURCE_ADAPTER_SCHEMA,
        "protocol": PROTOCOL_VERSION,
        "extraction_algorithm": EXTRACTION_ALGORITHM,
        "protocol_sha256": protocol_sha,
        "status": "unlabeled_source_pool",
        "candidate_binding": None,
        "source_hashes": source_hashes,
        "history_exclusion_sha256": history_exclusion["manifest_sha256"],
        "source_label_shown_to_reviewers": False,
        "reviewer_input_fields": ["claim", "title", "abstract"],
        "cases": [{k: c[k] for k in ("case_id", "component_id", "task_id", "input", "input_sha256")} for c in cases],
        "packets": reviewer_packets,
        "private_provenance": [{"case_id": c["case_id"], "component_id": c["component_id"],
                                 "task_id": c["task_id"], "bindings": c["private_bindings"]} for c in cases],
        "exclusions": exclusions,
        "capacity_gate": capacity,
        "eligible_for_confirmation": False,
        "official_test_read": False,
        "selection_gate_read": False,
    }


def allocate_disjoint_roles(component_ids: Sequence[str], *, targets: Mapping[str, int], seed: int,
                            candidate_binding: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Allocate disjoint component identities; confirmation requires a preselected candidate."""
    required = {"mipro_train", "search_validation", "confirmation"}
    if set(targets) != required or any(type(n) is not int or n <= 0 for n in targets.values()):
        raise ValueError("all three positive role capacities are required")
    if targets["confirmation"] and not candidate_binding:
        raise ValueError("confirmation allocation requires a preselected candidate binding")
    if candidate_binding and (not candidate_binding.get("candidate_id")
                              or not re.fullmatch(r"[0-9a-f]{64}", str(candidate_binding.get("candidate_report_sha256") or ""))):
        raise ValueError("candidate binding must identify a candidate report by SHA-256")
    unique = sorted(set(map(str, component_ids)))
    if len(unique) < sum(targets.values()):
        raise ValueError(f"independent component capacity {len(unique)} is below required {sum(targets.values())}")
    random.Random(seed).shuffle(unique)
    cursor = 0
    result: dict[str, Any] = {"seed": seed}
    for role in ("mipro_train", "search_validation", "confirmation"):
        result[role] = unique[cursor:cursor + targets[role]]
        cursor += targets[role]
    result["confirmation_binding"] = (dict(candidate_binding) if candidate_binding else None)
    result["confirmation_eligible"] = False
    return result
