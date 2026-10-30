"""Frozen, single-document selected-answer calibration packet and scorer.

This module is intentionally not wired into runtime or production attestations.
It validates future task-builder/reference outputs and scores already-recorded
model receipts without making network calls.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..data.g05_task_packet import _canonical_identity
from .grounding import evidence_sentence_spans
from .grounding_selected_answer import (
    normalize_selected_answer_result,
    selected_answer_support_messages,
)

PACKET_SCHEMA = "g05-selected-answer-support-packet-v1"
REFERENCE_SCHEMA = "g05-selected-answer-support-reference-v1"
SOURCE_NATIVE_REFERENCE_SCHEMA = "g05-selected-answer-support-source-native-reference-v1"
ABSTRACT_SOURCE_NATIVE_REFERENCE_SCHEMA = "g05-selected-answer-support-source-native-abstract-reference-v1"
MEDAESQA_DATASET = "g05-medaesqa-selected-answer-support"
MEDAESQA_ABSTRACT_DATASET = "g05-medaesqa-full-abstract-support"
MEDAESQA_LABEL_MAP = {
    "supporting": "supported",
    "contradicting": "contradicted",
    "neutral": "insufficient",
}
PLAN_SCHEMA = "g05-selected-answer-support-calibration-plan-v1"
REPORT_SCHEMA = "g05-selected-answer-support-calibration-report-v1"
TASK_CONTRACT = "selected_answer_support_v1"
SCOPE = "single_doc"
LABELS = ("supported", "contradicted", "insufficient")
THRESHOLDS = {
    "minimum_per_class": 30,
    "minimum_independent_groups_per_class": 30,
    "macro_f1": 0.80,
    "minimum_class_recall": 0.75,
    "false_support_rate": 0.10,
    "invalid_outputs": 0,
}
MAX_DOC_CHARS = 6000
MAX_COMPLETION_TOKENS = 512
# Reasoning models can spend the entire legacy 512-token output budget on
# hidden reasoning before emitting the required JSON object.  Keep 512 as the
# backwards-compatible default, while allowing a frozen provider plan to opt
# into a larger, auditable budget.
ALLOWED_COMPLETION_TOKENS = frozenset({512, 1024, 1536, 2048})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _without_hash(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def _case_input_payload(case: dict[str, Any]) -> dict[str, Any]:
    identity_field = "source_id" if "source_id" in case else "doi"
    return {key: case[key] for key in (
        "case_id", "task_id", "source_group_id", identity_field, "question",
        "selected_answer", "evidence")}


def seal_selected_answer_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Deterministically add evidence, case-input, and packet hashes."""
    result = dict(packet)
    cases = []
    for original in result.get("cases", []):
        case = dict(original)
        evidence = []
        for original_doc in case.get("evidence", []):
            doc = dict(original_doc)
            doc["content_sha256"] = hashlib.sha256(
                str(doc.get("text", "")).encode("utf-8")).hexdigest()
            evidence.append(doc)
        case["evidence"] = evidence
        case["input_sha256"] = _sha(_case_input_payload(case))
        cases.append(case)
    result["cases"] = cases
    result.pop("packet_sha256", None)
    result["packet_sha256"] = _sha(result)
    return result


def validate_selected_answer_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Validate a frozen task packet and its minimum independent cohort size."""
    packet_keys = {
        "schema", "dataset", "split", "task_contract", "scope",
        "source_pool_sha256", "publication_status_report_sha256",
        "sample_plan_sha256", "protocol_sha256", "cases", "packet_sha256",
    }
    if not isinstance(packet, dict):
        raise ValueError("selected-answer packet schema keys mismatch")
    dataset = packet.get("dataset")
    if dataset in {MEDAESQA_DATASET, MEDAESQA_ABSTRACT_DATASET}:
        packet_keys = packet_keys | {"source_id_type", "source_data_sha256", "history_scan_sha256"}
        identity_field = "source_id"
        if (packet.get("source_id_type") != "pmid"
                or packet.get("source_data_sha256") != packet.get("source_pool_sha256")
                or packet.get("history_scan_sha256") != packet.get("publication_status_report_sha256")):
            raise ValueError("MedAESQA source identity hashes or PMID identifier type mismatch")
    else:
        identity_field = "doi"
    case_keys = {
        "case_id", "task_id", "source_group_id", identity_field, "question",
        "selected_answer", "evidence", "input_sha256",
    }
    if set(packet) != packet_keys:
        raise ValueError("selected-answer packet schema keys mismatch")
    if (packet["schema"] != PACKET_SCHEMA
            or packet["dataset"] not in {"g05-plos-selected-answer-support", MEDAESQA_DATASET,
                                          MEDAESQA_ABSTRACT_DATASET}
            or packet["split"] != "calibration"
            or packet["task_contract"] != TASK_CONTRACT):
        raise ValueError("selected-answer packet task identity mismatch")
    if packet["scope"] != SCOPE:
        raise ValueError("top3 or other scope cannot use single_doc calibration")
    for field in ("source_pool_sha256", "publication_status_report_sha256",
                  "sample_plan_sha256", "protocol_sha256"):
        if not _valid_sha(packet[field]):
            raise ValueError(f"packet {field} is missing or invalid")
    cases = packet["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("selected-answer packet has no cases")
    case_ids: set[str] = set()
    task_ids: set[str] = set()
    groups: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != case_keys:
            raise ValueError("selected-answer case schema keys mismatch")
        for field in ("case_id", "task_id", "source_group_id", identity_field,
                      "question", "selected_answer"):
            if not isinstance(case[field], str) or not case[field].strip():
                raise ValueError(f"selected-answer case {field} is missing")
        group = case["source_group_id"].strip().casefold()
        source_id = case[identity_field].strip().casefold()
        if group != source_id:
            raise ValueError(f"source_group_id must equal the canonical {identity_field}")
        if dataset in {MEDAESQA_DATASET, MEDAESQA_ABSTRACT_DATASET} and not source_id.isdigit():
            raise ValueError("MedAESQA source IDs must be numeric PMIDs")
        if case["case_id"] in case_ids or case["task_id"] in task_ids:
            raise ValueError("duplicate case_id or task_id in selected-answer packet")
        case_ids.add(case["case_id"])
        task_ids.add(case["task_id"])
        groups.add(group)
        evidence = case["evidence"]
        if not isinstance(evidence, list) or len(evidence) != 1:
            raise ValueError("selected-answer packet requires exactly one evidence document per case")
        doc = evidence[0]
        if (not isinstance(doc, dict) or set(doc) != {"doc_id", "text", "content_sha256"}
                or not isinstance(doc["doc_id"], str) or not doc["doc_id"].strip()
                or not isinstance(doc["text"], str) or not doc["text"].strip()):
            raise ValueError("selected-answer evidence schema or content is invalid")
        actual_content_sha = hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
        if doc["content_sha256"] != actual_content_sha:
            raise ValueError("selected-answer evidence content hash mismatch")
        expected_doc_ids = {source_id}
        if dataset in {MEDAESQA_DATASET, MEDAESQA_ABSTRACT_DATASET}:
            expected_doc_ids.add(f"pmid:{source_id}")
        if doc["doc_id"].strip().casefold() not in expected_doc_ids:
            raise ValueError("evidence document identity does not match its source group")
        if case["input_sha256"] != _sha(_case_input_payload(case)):
            raise ValueError("selected-answer case input hash mismatch")
    if packet["packet_sha256"] != _sha(_without_hash(packet, "packet_sha256")):
        raise ValueError("selected-answer packet hash mismatch")
    return {"case_count": len(cases), "source_group_count": len(groups),
            "case_ids": [case["case_id"] for case in cases]}


def seal_selected_answer_reference(reference: dict[str, Any]) -> dict[str, Any]:
    result = dict(reference)
    result.pop("reference_sha256", None)
    result["reference_sha256"] = _sha(result)
    return result


def _validate_reference_span(case: dict[str, Any], sentence_ids: Any) -> bool:
    if not isinstance(sentence_ids, list) or not sentence_ids:
        return False
    if any(type(item) is not int for item in sentence_ids) or len(sentence_ids) != len(set(sentence_ids)):
        return False
    text = case["evidence"][0]["text"][:MAX_DOC_CHARS]
    available = {row["sentence_id"] for row in evidence_sentence_spans(text)}
    return all(item in available for item in sentence_ids)


def _valid_reference_label(row: Any, case: dict[str, Any]) -> bool:
    keys = {"verdict", "basis", "sentence_ids", "raw_response_sha256"}
    if not isinstance(row, dict) or set(row) != keys:
        return False
    if row["verdict"] not in LABELS or not _valid_sha(row["raw_response_sha256"]):
        return False
    basis_to_verdict = {
        "direct_finding_supports": "supported",
        "direct_finding_opposes": "contradicted",
        "evidence_gap_or_uncertainty": "insufficient",
        "topic_only_or_scope_mismatch": "insufficient",
    }
    if not isinstance(row["basis"], str):
        return False
    return (basis_to_verdict.get(row["basis"]) == row["verdict"]
            and _validate_reference_span(case, row["sentence_ids"]))


def validate_selected_answer_reference(
    packet: dict[str, Any], reference: dict[str, Any],
) -> dict[str, Any]:
    """Validate AI blind references, adjudication, exact spans, and group power."""
    validate_selected_answer_packet(packet)
    if isinstance(reference, dict) and reference.get("schema") == SOURCE_NATIVE_REFERENCE_SCHEMA:
        return _validate_source_native_reference(packet, reference)
    if (isinstance(reference, dict)
            and reference.get("schema") == ABSTRACT_SOURCE_NATIVE_REFERENCE_SCHEMA):
        return _validate_abstract_source_native_reference(packet, reference)
    keys = {"schema", "annotation_origin", "task_contract", "scope", "packet_sha256",
            "protocol_sha256", "reviewer_provenance", "per_case", "reference_sha256"}
    if not isinstance(reference, dict) or set(reference) != keys:
        raise ValueError("selected-answer reference schema keys mismatch")
    if (reference["schema"] != REFERENCE_SCHEMA or reference["annotation_origin"] != "ai"
            or reference["task_contract"] != TASK_CONTRACT or reference["scope"] != SCOPE
            or reference["packet_sha256"] != packet["packet_sha256"]
            or reference["protocol_sha256"] != packet["protocol_sha256"]):
        raise ValueError("selected-answer reference task or packet identity mismatch")
    if reference["reference_sha256"] != _sha(_without_hash(reference, "reference_sha256")):
        raise ValueError("selected-answer reference hash mismatch")
    provenance = reference["reviewer_provenance"]
    provenance_keys = {"reviewer_id", "provider", "model_family", "model_id", "prompt_sha256"}
    if (not isinstance(provenance, list) or len(provenance) != 2
            or any(not isinstance(row, dict) or set(row) != provenance_keys for row in provenance)):
        raise ValueError("selected-answer reference requires two reviewer provenance records")
    reviewers = {row["reviewer_id"]: row for row in provenance}
    if set(reviewers) != {"A", "B"}:
        raise ValueError("reviewers must be identified as A and B")
    canonical_reviewers = {}
    for row in provenance:
        if not _valid_sha(row["prompt_sha256"]):
            raise ValueError("reviewer provenance is incomplete")
        canonical_reviewers[row["reviewer_id"]] = _canonical_identity(
            row["provider"], row["model_id"], row["model_family"])
    if (canonical_reviewers["A"][2] == canonical_reviewers["B"][2]
            or canonical_reviewers["A"][1] == canonical_reviewers["B"][1]):
        raise ValueError("reference reviewers must use different model families and IDs")

    packet_by_id = {case["case_id"]: case for case in packet["cases"]}
    rows = reference["per_case"]
    if not isinstance(rows, list) or len(rows) != len(packet_by_id):
        raise ValueError("selected-answer reference case coverage mismatch")
    resolved: dict[str, list[str]] = {label: [] for label in LABELS}
    seen: set[str] = set()
    for row in rows:
        row_keys = {"case_id", "input_sha256", "source_group_id", "reviewers",
                    "adjudication", "resolved_verdict", "resolved_sentence_ids"}
        if not isinstance(row, dict) or set(row) != row_keys:
            raise ValueError("selected-answer per-case reference schema mismatch")
        case_id = row["case_id"]
        if case_id in seen or case_id not in packet_by_id:
            raise ValueError("duplicate or unknown selected-answer reference case")
        seen.add(case_id)
        case = packet_by_id[case_id]
        if (row["input_sha256"] != case["input_sha256"]
                or row["source_group_id"] != case["source_group_id"]):
            raise ValueError("selected-answer reference case input mismatch")
        pair = row["reviewers"]
        if not isinstance(pair, dict) or set(pair) != {"A", "B"}:
            raise ValueError("each reference case requires blind A/B labels")
        if not all(_valid_reference_label(pair[key], case) for key in ("A", "B")):
            raise ValueError("reference label or exact sentence span is invalid")
        a, b = pair["A"], pair["B"]
        adjudication = row["adjudication"]
        if a["verdict"] == b["verdict"]:
            if adjudication is not None or row["resolved_verdict"] != a["verdict"]:
                raise ValueError("consensus reference must resolve to A/B without adjudication")
            expected_spans = a["sentence_ids"]
        else:
            adjudication_keys = {"provider", "model_family", "model_id", "verdict", "basis",
                                 "sentence_ids", "raw_response_sha256"}
            if not isinstance(adjudication, dict) or set(adjudication) != adjudication_keys:
                raise ValueError("disagreement requires a third-model adjudication")
            adjudicator_identity = _canonical_identity(adjudication["provider"],
                adjudication["model_id"], adjudication["model_family"])
            if (adjudicator_identity[2] in {canonical_reviewers["A"][2],
                                            canonical_reviewers["B"][2]}
                    or adjudicator_identity[1] in {canonical_reviewers["A"][1],
                                                   canonical_reviewers["B"][1]}):
                raise ValueError("adjudicator must use a third model family and ID")
            adjudication_label = {key: adjudication.get(key) for key in
                                  ("verdict", "basis", "sentence_ids", "raw_response_sha256")}
            if not _valid_reference_label(adjudication_label, case):
                raise ValueError("adjudication label or exact sentence span is invalid")
            if row["resolved_verdict"] != adjudication["verdict"]:
                raise ValueError("resolved verdict does not match adjudication")
            expected_spans = adjudication["sentence_ids"]
        if (row["resolved_verdict"] not in LABELS
                or row["resolved_sentence_ids"] != expected_spans
                or not _validate_reference_span(case, row["resolved_sentence_ids"])):
            raise ValueError("resolved label or exact sentence span is invalid")
        resolved[row["resolved_verdict"]].append(case["source_group_id"].casefold())
    if seen != set(packet_by_id):
        raise ValueError("selected-answer reference case coverage mismatch")
    counts = {label: sum(1 for row in rows if row["resolved_verdict"] == label)
              for label in LABELS}
    group_counts = {label: len(set(resolved[label])) for label in LABELS}
    for label in LABELS:
        if counts[label] < THRESHOLDS["minimum_per_class"]:
            raise ValueError(f"reference requires minimum 30 cases per class: {label}")
        if group_counts[label] < THRESHOLDS["minimum_independent_groups_per_class"]:
            raise ValueError(f"reference requires minimum 30 independent DOI groups per class: {label}")
    return {"case_count": len(rows), "class_case_counts": counts,
            "class_independent_groups": group_counts}


def _validate_source_native_reference(
    packet: dict[str, Any], reference: dict[str, Any],
) -> dict[str, Any]:
    """Validate dataset-native human citation labels without fabricating AI provenance."""
    from ..data.g05_medaesqa_adapter import _canonical, _sha

    keys = {"schema", "annotation_origin", "task_contract", "scope", "packet_sha256",
            "protocol_sha256", "source_data_sha256", "history_scan_sha256",
            "label_mapping_sha256", "per_case", "reference_sha256"}
    if set(reference) != keys:
        raise ValueError("source-native selected-answer reference schema keys mismatch")
    if (packet.get("dataset") != MEDAESQA_DATASET
            or reference.get("annotation_origin") != "source_native_human"
            or reference.get("task_contract") != TASK_CONTRACT
            or reference.get("scope") != SCOPE
            or reference.get("packet_sha256") != packet.get("packet_sha256")
            or reference.get("protocol_sha256") != packet.get("protocol_sha256")
            or reference.get("source_data_sha256") != packet.get("source_data_sha256")
            or reference.get("history_scan_sha256") != packet.get("history_scan_sha256")
            or reference.get("label_mapping_sha256") != _sha(MEDAESQA_LABEL_MAP)):
        raise ValueError("source-native reference packet, source or label mapping mismatch")
    if reference.get("reference_sha256") != _sha(
            {key: value for key, value in reference.items() if key != "reference_sha256"}):
        raise ValueError("source-native reference hash mismatch")

    packet_by_id = {case["case_id"]: case for case in packet["cases"]}
    rows = reference.get("per_case")
    if not isinstance(rows, list) or len(rows) != len(packet_by_id):
        raise ValueError("source-native reference case coverage mismatch")
    label_to_verdict = MEDAESQA_LABEL_MAP
    resolved: dict[str, list[str]] = {label: [] for label in LABELS}
    seen: set[str] = set()
    for row in rows:
        row_keys = {"case_id", "source_group_id", "input_sha256", "source_label",
                    "resolved_verdict", "resolved_sentence_ids", "source_annotation",
                    "source_annotation_sha256"}
        if not isinstance(row, dict) or set(row) != row_keys:
            raise ValueError("source-native reference case schema mismatch")
        case_id = row.get("case_id")
        if case_id in seen or case_id not in packet_by_id:
            raise ValueError("duplicate or unknown source-native reference case")
        seen.add(case_id)
        case = packet_by_id[case_id]
        if (row.get("input_sha256") != case["input_sha256"]
                or row.get("source_group_id") != case["source_group_id"]):
            raise ValueError("source-native reference case input mismatch")
        source_label = row.get("source_label")
        verdict = row.get("resolved_verdict")
        if label_to_verdict.get(source_label) != verdict:
            raise ValueError("unsupported source-native relation label mapping")
        annotation = row.get("source_annotation")
        annotation_keys = {"question_id", "model_id", "answer_sentence_id", "pmid",
                           "source_label", "answer_sha256", "evidence_sha256"}
        if not isinstance(annotation, dict) or set(annotation) != annotation_keys:
            raise ValueError("source-native annotation binding schema mismatch")
        if (annotation.get("source_label") != source_label
                or annotation.get("pmid") != case["source_id"]
                or not all(isinstance(annotation.get(key), str) and annotation[key].strip()
                           for key in ("question_id", "model_id", "answer_sentence_id"))
                or annotation.get("answer_sha256") != hashlib.sha256(
                    case["selected_answer"].encode("utf-8")).hexdigest()
                or annotation.get("evidence_sha256") != hashlib.sha256(
                    case["evidence"][0]["text"].encode("utf-8")).hexdigest()
                or row.get("source_annotation_sha256") != _sha(annotation)):
            raise ValueError("source-native annotation is not bound to selected answer and evidence")
        if (row.get("resolved_sentence_ids") != [0]
                or not _validate_reference_span(case, [0])):
            raise ValueError("source-native reference evidence span is invalid")
        resolved[verdict].append(case["source_group_id"].casefold())
    if seen != set(packet_by_id):
        raise ValueError("source-native reference case coverage mismatch")
    counts = {label: sum(1 for row in rows if row["resolved_verdict"] == label)
              for label in LABELS}
    group_counts = {label: len(set(resolved[label])) for label in LABELS}
    for label in LABELS:
        if counts[label] < THRESHOLDS["minimum_per_class"]:
            raise ValueError(f"source-native reference requires minimum 30 cases per class: {label}")
        if group_counts[label] < THRESHOLDS["minimum_independent_groups_per_class"]:
            raise ValueError(f"source-native reference requires 30 independent source groups: {label}")
    return {"case_count": len(rows), "class_case_counts": counts,
            "class_independent_groups": group_counts}


def _validate_abstract_source_native_reference(
    packet: dict[str, Any], reference: dict[str, Any],
) -> dict[str, Any]:
    """Validate a source-native citation relation bound to a full abstract.

    The native label is the citation-assessment relation for one selected answer
    sentence.  It is intentionally a separate schema from the AI full-abstract
    reference: it supports a reproducible diagnostic calibration, but does not
    claim that the upstream excerpt annotation is a human gold label for every
    fact in the complete abstract.
    """
    from ..data.g05_medaesqa_adapter import _sha as source_sha
    keys = {"schema", "annotation_origin", "task_contract", "scope", "packet_sha256",
            "protocol_sha256", "source_data_sha256", "history_scan_sha256",
            "label_mapping_sha256", "reference_scope", "per_case", "reference_sha256"}
    if set(reference) != keys:
        raise ValueError("abstract source-native reference schema keys mismatch")
    if (packet.get("dataset") != MEDAESQA_ABSTRACT_DATASET
            or reference.get("schema") != ABSTRACT_SOURCE_NATIVE_REFERENCE_SCHEMA
            or reference.get("annotation_origin") != "source_native_citation_assessment"
            or reference.get("task_contract") != TASK_CONTRACT
            or reference.get("scope") != SCOPE
            or reference.get("reference_scope") != "citation_excerpt_bound_to_full_abstract"
            or reference.get("packet_sha256") != packet.get("packet_sha256")
            or reference.get("protocol_sha256") != packet.get("protocol_sha256")
            or reference.get("source_data_sha256") != packet.get("source_data_sha256")
            or reference.get("history_scan_sha256") != packet.get("history_scan_sha256")
            or reference.get("label_mapping_sha256") != source_sha(MEDAESQA_LABEL_MAP)):
        raise ValueError("abstract source-native reference packet or scope mismatch")
    if reference.get("reference_sha256") != _sha(
            _without_hash(reference, "reference_sha256")):
        raise ValueError("abstract source-native reference hash mismatch")
    packet_by_id = {case["case_id"]: case for case in packet["cases"]}
    rows = reference.get("per_case")
    if not isinstance(rows, list) or len(rows) != len(packet_by_id):
        raise ValueError("abstract source-native reference case coverage mismatch")
    resolved: dict[str, list[str]] = {label: [] for label in LABELS}
    seen: set[str] = set()
    for row in rows:
        row_keys = {"case_id", "source_group_id", "input_sha256", "source_label",
                    "resolved_verdict", "resolved_sentence_ids", "source_annotation",
                    "source_annotation_sha256"}
        if not isinstance(row, dict) or set(row) != row_keys:
            raise ValueError("abstract source-native per-case schema mismatch")
        case_id = row.get("case_id")
        if case_id in seen or case_id not in packet_by_id:
            raise ValueError("duplicate or unknown abstract source-native case")
        seen.add(case_id)
        case = packet_by_id[case_id]
        if (row.get("input_sha256") != case.get("input_sha256")
                or row.get("source_group_id") != case.get("source_group_id")):
            raise ValueError("abstract source-native case input mismatch")
        source_label = row.get("source_label")
        verdict = row.get("resolved_verdict")
        if MEDAESQA_LABEL_MAP.get(source_label) != verdict:
            raise ValueError("unsupported abstract source-native relation label mapping")
        annotation = row.get("source_annotation")
        annotation_keys = {"question_id", "model_id", "answer_sentence_id", "pmid",
                           "source_label", "answer_sha256", "evidence_sha256",
                           "abstract_sha256", "excerpt_match_mode"}
        if not isinstance(annotation, dict) or set(annotation) != annotation_keys:
            raise ValueError("abstract source annotation binding schema mismatch")
        if (annotation.get("source_label") != source_label
                or annotation.get("pmid") != case.get("source_id")
                or not all(isinstance(annotation.get(key), str) and annotation[key].strip()
                           for key in ("question_id", "model_id", "answer_sentence_id"))
                or annotation.get("answer_sha256") != hashlib.sha256(
                    case["selected_answer"].encode("utf-8")).hexdigest()
                or not _valid_sha(annotation.get("evidence_sha256"))
                or annotation.get("abstract_sha256") != hashlib.sha256(
                    case["evidence"][0]["text"].encode("utf-8")).hexdigest()
                or annotation.get("excerpt_match_mode") not in {"exact", "html_entity_normalized"}
                or not _valid_sha(annotation.get("abstract_sha256"))
                or row.get("source_annotation_sha256") != _sha(annotation)):
            raise ValueError("abstract source annotation is not bound to answer and document")
        if (row.get("resolved_sentence_ids") != [0]
                or not _validate_reference_span(case, [0])):
            raise ValueError("abstract source-native evidence span is invalid")
        resolved[verdict].append(str(case["source_group_id"]).casefold())
    if seen != set(packet_by_id):
        raise ValueError("abstract source-native reference case coverage mismatch")
    counts = {label: sum(1 for row in rows if row["resolved_verdict"] == label)
              for label in LABELS}
    group_counts = {label: len(set(resolved[label])) for label in LABELS}
    for label in LABELS:
        if counts[label] < THRESHOLDS["minimum_per_class"]:
            raise ValueError(f"abstract source-native reference requires minimum 30 cases: {label}")
        if group_counts[label] < THRESHOLDS["minimum_independent_groups_per_class"]:
            raise ValueError(f"abstract source-native reference requires 30 source groups: {label}")
    return {"case_count": len(rows), "class_case_counts": counts,
            "class_independent_groups": group_counts}


def _implementation_sha256(dataset: str | None = None) -> str:
    from . import grounding_selected_answer as adapter
    module_paths = [Path(__file__).resolve(), Path(adapter.__file__).resolve()]
    if dataset == MEDAESQA_DATASET:
        from ..data import g05_medaesqa_adapter as source_adapter
        module_paths.append(Path(source_adapter.__file__).resolve())
    elif dataset == MEDAESQA_ABSTRACT_DATASET:
        from ..data import g05_medaesqa_abstract_adapter as source_adapter
        module_paths.append(Path(source_adapter.__file__).resolve())
    return _sha({path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in module_paths})


def _build_plan(packet: dict[str, Any], reference: dict[str, Any], *, model_id: str,
                model_protocol_sha256: str, created_at: str, max_doc_chars: int,
                max_tokens: int, model_protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    requests = []
    for index, case in enumerate(packet["cases"]):
        system, user = selected_answer_support_messages(
            case["question"], case["selected_answer"], case["evidence"], max_doc_chars)
        prompt_sha = hashlib.sha256((system + "\0" + user).encode("utf-8")).hexdigest()
        requests.append({"call_index": index, "case_id": case["case_id"],
                         "input_sha256": case["input_sha256"],
                         "prompt_sha256": prompt_sha})
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "dataset": packet["dataset"],
        "task_contract": TASK_CONTRACT,
        "scope": SCOPE,
        "packet_sha256": packet["packet_sha256"],
        "reference_sha256": reference["reference_sha256"],
        "protocol_sha256": packet["protocol_sha256"],
        "model_id": model_id,
        "model_protocol_sha256": model_protocol_sha256,
        "scorer_implementation_sha256": _implementation_sha256(packet["dataset"]),
        "created_at": created_at,
        "max_doc_chars": max_doc_chars,
        "max_tokens": max_tokens,
        "thresholds": THRESHOLDS,
        "request_count": len(requests),
        "max_physical_attempts": len(requests),
        "retries_per_case": 0,
        "requests": requests,
        "labels_generated": False,
        "model_calls_during_planning": 0,
        "target_scorer_calls_during_planning": 0,
    }
    if model_protocol is not None:
        plan["model_protocol"] = dict(model_protocol)
    plan["plan_sha256"] = _sha(plan)
    return plan


def build_selected_answer_plan(
    packet: dict[str, Any], reference: dict[str, Any], *, model_id: str,
    model_protocol_sha256: str, created_at: str,
    max_doc_chars: int = MAX_DOC_CHARS,
    max_tokens: int = MAX_COMPLETION_TOKENS,
    model_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze exact selected-answer prompts and one-attempt-per-case budget."""
    validate_selected_answer_packet(packet)
    validate_selected_answer_reference(packet, reference)
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("target model_id is required")
    if not _valid_sha(model_protocol_sha256):
        raise ValueError("target model protocol SHA-256 is required")
    if not isinstance(created_at, str) or not created_at.strip():
        raise ValueError("plan creation timestamp is required")
    if type(max_doc_chars) is not int or max_doc_chars != MAX_DOC_CHARS:
        raise ValueError("calibration max_doc_chars must equal the production 6000-character contract")
    if type(max_tokens) is not int or max_tokens not in ALLOWED_COMPLETION_TOKENS:
        raise ValueError("calibration max_tokens must be one of the frozen budgets: 512, 1024, 1536, 2048")
    return _build_plan(packet, reference, model_id=model_id,
                       model_protocol_sha256=model_protocol_sha256,
                       created_at=created_at, max_doc_chars=max_doc_chars,
                       max_tokens=max_tokens, model_protocol=model_protocol)


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = {truth: {pred: 0 for pred in (*LABELS, "invalid_output")} for truth in LABELS}
    for row in rows:
        matrix[row["reference"]][row["prediction"]] += 1
    per_class = {}
    for label in LABELS:
        support = sum(matrix[label].values())
        predicted = sum(matrix[truth][label] for truth in LABELS)
        tp = matrix[label][label]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        per_class[label] = {"support": support, "precision": precision,
                            "recall": recall, "f1": f1}
    macro_f1 = sum(row["f1"] for row in per_class.values()) / len(LABELS)
    negative_total = sum(sum(matrix[label].values()) for label in LABELS if label != "supported")
    false_supports = sum(matrix[label]["supported"] for label in LABELS if label != "supported")
    invalid = sum(matrix[label]["invalid_output"] for label in LABELS)
    return {"matrix": matrix, "per_class": per_class, "macro_f1": macro_f1,
            "minimum_class_recall": min(row["recall"] for row in per_class.values()),
            "false_support_rate": false_supports / negative_total if negative_total else 1.0,
            "invalid_outputs": invalid}


def _receipt_valid(call: dict[str, Any], request: dict[str, Any], plan: dict[str, Any],
                   system: str, user: str) -> tuple[bool, str, int]:
    expected_prompt_sha = request["prompt_sha256"]
    attempts = call.get("physical_attempts")
    attempt_count = attempts if type(attempts) is int and attempts >= 0 else 0
    if call.get("plan_sha256") != plan["plan_sha256"]:
        return False, "plan_identity_mismatch", attempt_count
    if call.get("input_sha256") != request["input_sha256"]:
        return False, "input_identity_mismatch", attempt_count
    if (call.get("system") != system or call.get("user") != user
            or call.get("prompt_sha256") != expected_prompt_sha):
        return False, "prompt_identity_mismatch", attempt_count
    receipt = call.get("receipt")
    if not isinstance(receipt, dict):
        return False, "missing_call_receipt", attempt_count
    common = {
        "plan_sha256": plan["plan_sha256"],
        "input_sha256": request["input_sha256"],
        "prompt_sha256": expected_prompt_sha,
        "model_protocol_sha256": plan["model_protocol_sha256"],
        "physical_attempts": attempts,
    }
    if any(receipt.get(key) != value for key, value in common.items()):
        return False, "receipt_identity_mismatch", attempt_count
    if not isinstance(receipt.get("request_id"), str) or not receipt["request_id"].strip():
        return False, "missing_request_id", attempt_count
    status = call.get("status")
    if status == "succeeded":
        raw_output = call.get("raw_output")
        if raw_output is None:
            return False, "missing_scorer_output", attempt_count
        if receipt.get("raw_output_sha256") != _sha(raw_output):
            return False, "output_receipt_mismatch", attempt_count
        usage = receipt.get("usage")
        if not isinstance(usage, dict):
            return False, "missing_usage_receipt", attempt_count
        for key in ("prompt_tokens", "completion_tokens"):
            if type(usage.get(key)) is not int or usage[key] < 0:
                return False, "invalid_usage_receipt", attempt_count
        if usage["completion_tokens"] > plan["max_tokens"]:
            return False, "completion_token_budget_exceeded", attempt_count
        return True, "valid", attempt_count
    if status == "failed":
        if call.get("raw_output") is not None or not isinstance(call.get("error"), str) or not call["error"].strip():
            return False, "invalid_failed_call_record", attempt_count
        return False, "provider_call_failed", attempt_count
    return False, "invalid_call_status", attempt_count


def score_selected_answer_calibration(
    packet: dict[str, Any], reference: dict[str, Any], plan: dict[str, Any],
    call_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score exact production outputs/receipts; failures and invalids stay in denominator."""
    validate_selected_answer_packet(packet)
    validate_selected_answer_reference(packet, reference)
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported selected-answer calibration plan")
    expected_plan = _build_plan(
        packet, reference, model_id=plan.get("model_id"),
        model_protocol_sha256=plan.get("model_protocol_sha256"),
        created_at=plan.get("created_at"), max_doc_chars=plan.get("max_doc_chars"),
        max_tokens=plan.get("max_tokens"), model_protocol=plan.get("model_protocol"))
    if plan != expected_plan:
        raise ValueError("selected-answer calibration plan is stale, altered, or scorer-mismatched")
    if not isinstance(call_records, list):
        raise ValueError("call records must be a list")
    packet_cases = {case["case_id"]: case for case in packet["cases"]}
    references = {row["case_id"]: row["resolved_verdict"] for row in reference["per_case"]}
    requests = {row["case_id"]: row for row in plan["requests"]}
    grouped: dict[str, list[dict[str, Any]]] = {}
    unexpected_count = 0
    physical_total = 0
    retry_violations = 0
    request_ids: set[str] = set()
    for call in call_records:
        if not isinstance(call, dict):
            unexpected_count += 1
            continue
        case_id = call.get("case_id")
        attempts = call.get("physical_attempts")
        if type(attempts) is int and attempts >= 0:
            physical_total += attempts
            if attempts != 1:
                retry_violations += 1
        else:
            retry_violations += 1
        if case_id not in requests:
            unexpected_count += 1
            continue
        grouped.setdefault(case_id, []).append(call)
        receipt = call.get("receipt")
        if isinstance(receipt, dict) and isinstance(receipt.get("request_id"), str):
            if receipt["request_id"] in request_ids:
                retry_violations += 1
            request_ids.add(receipt["request_id"])

    rows = []
    details = []
    missing_count = 0
    duplicate_count = 0
    failed_count = 0
    invalid_receipt_count = 0
    for request in plan["requests"]:
        case_id = request["case_id"]
        case = packet_cases[case_id]
        calls = grouped.get(case_id, [])
        prediction = "invalid_output"
        reason = "missing_call_receipt"
        if not calls:
            missing_count += 1
        elif len(calls) != 1:
            duplicate_count += 1
            reason = "duplicate_call_receipt"
        else:
            call = calls[0]
            system, user = selected_answer_support_messages(
                case["question"], case["selected_answer"], case["evidence"],
                plan["max_doc_chars"])
            valid_receipt, reason, _ = _receipt_valid(call, request, plan, system, user)
            if not valid_receipt:
                if reason == "provider_call_failed":
                    failed_count += 1
                else:
                    invalid_receipt_count += 1
            else:
                normalized, parser_reason = normalize_selected_answer_result(
                    call["raw_output"], case["evidence"], plan["max_doc_chars"])
                if normalized is None:
                    reason = parser_reason
                else:
                    prediction = normalized["verdict"]
                    reason = "valid"
        rows.append({"case_id": case_id, "input_sha256": request["input_sha256"],
                     "source_group_id": case["source_group_id"],
                     "reference": references[case_id], "prediction": prediction})
        details.append({"case_id": case_id, "status": reason,
                        "prediction": prediction,
                        "call_receipt_count": len(calls)})

    metrics = _metrics(rows)
    class_groups = {label: len({row["source_group_id"].casefold() for row in rows
                               if row["reference"] == label}) for label in LABELS}
    all_calls_accounted = (len(call_records) == len(packet_cases)
                           and unexpected_count == 0 and duplicate_count == 0
                           and missing_count == 0)
    physical_budget_ok = physical_total == plan["request_count"]
    no_retries = retry_violations == 0
    thresholds_pass = (
        all(metrics["per_class"][label]["support"] >= THRESHOLDS["minimum_per_class"]
            for label in LABELS)
        and all(class_groups[label] >= THRESHOLDS["minimum_independent_groups_per_class"]
                for label in LABELS)
        and metrics["macro_f1"] >= THRESHOLDS["macro_f1"]
        and metrics["minimum_class_recall"] >= THRESHOLDS["minimum_class_recall"]
        and metrics["false_support_rate"] <= THRESHOLDS["false_support_rate"]
        and metrics["invalid_outputs"] == THRESHOLDS["invalid_outputs"]
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "completed" if all_calls_accounted and physical_budget_ok and no_retries else "incomplete_or_invalid_ledger",
        "dataset": packet["dataset"],
        "task_contract": TASK_CONTRACT,
        "scope": SCOPE,
        "packet_sha256": packet["packet_sha256"],
        "reference_sha256": reference["reference_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "model_id": plan["model_id"],
        "model_protocol_sha256": plan["model_protocol_sha256"],
        "scorer_implementation_sha256": plan["scorer_implementation_sha256"],
        "request_count_required": plan["request_count"],
        "request_count_recorded": len(call_records),
        "physical_attempt_budget": plan["max_physical_attempts"],
        "physical_attempts_recorded": physical_total,
        "retries_per_case": plan["retries_per_case"],
        "missing_call_count": missing_count,
        "duplicate_call_count": duplicate_count,
        "unexpected_call_count": unexpected_count,
        "failed_call_count": failed_count,
        "invalid_receipt_count": invalid_receipt_count,
        "retry_or_budget_violations": retry_violations,
        "ledger_complete": all_calls_accounted,
        "physical_budget_passed": physical_budget_ok,
        "no_retry_passed": no_retries,
        "class_independent_groups": class_groups,
        "thresholds": THRESHOLDS,
        **metrics,
        "case_results": rows,
        "call_results": details,
        "calibration_pass": bool(all_calls_accounted and physical_budget_ok
                                  and no_retries and thresholds_pass),
        "model_calls_during_scoring": physical_total,
    }
    report["report_sha256"] = _sha(report)
    return report
