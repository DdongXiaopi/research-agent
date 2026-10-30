"""Validate two blind grounding reviews and report agreement without adjudicating.

This is an offline scorer. It never creates a provider and never reads the
researcher-only benchmark-label key. Disagreements stay unresolved.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.util import sha256_obj


PACKET_SCHEMA = "task-input-grounding-human-reference-v2"
RESPONSE_SCHEMA = "task-input-grounding-human-reference-responses-v2"
PROTOCOL_CONTEXT_KEYS = {
    "schema", "dataset", "split", "sampling", "per_label",
    "offset_per_label", "judge_evidence_character_limit_per_document",
    "rubric_version", "case_input_sha256s", "source_hashes",
}
PUBMEDQA_REFERENCE_SOURCE_HASHES_V2 = {
    "calibration_selection_source_sha256": "06c9e4c98f3eb93da7bcc56a5ea3cacad3369e5c8abdff9b43ecc534da22859c",
    "task_input_evidence_source_sha256": "9ac40b0abc2ecf7a24a9b7af5161b970b4085beeae0d56392abad75432b6eedc",
    "reference_packet_builder_source_sha256": "e797ca0f6588f2184603e4e42b7504529df6d7b12d62f372f9207e006f87bf91",
    "human_reference_scorer_source_sha256": "dc24f50929585d6e11e32afedb7d7bea34902516c6e5a5c06ec7525c654ddfbf",
}
HEALTHVER_REFERENCE_SOURCE_HASHES_V1 = {
    "calibration_selection_source_sha256": "3c38d3889b1cf415a75ced9e30ace9d81378a8fc3f196dbda79b3b249082d1d9",
    "task_input_evidence_source_sha256": "454f9096dea381bc0d3398eb78f021e17dc3825124aa8134d52a24df2b820028",
    "reference_packet_builder_source_sha256": "a734bf8e595be1c784316c376638ce1c968de8d38db881633d7fbde1acbf9bf6",
    "human_reference_scorer_source_sha256": "5b5f70eed2f9718ea3dba2716be4f908365b89d1cc3381aa0746a80c77a1800d",
}
# Exact fingerprints of the last frozen v2/top3 builders before protocol v4.
# These historical implementations remain auditable when the validator changes;
# this is an allowlist of known tuples, never a blanket bypass of source checks.
HEALTHVER_REFERENCE_SOURCE_HASHES_V2_20261015 = {
    "calibration_selection_source_sha256": "3c38d3889b1cf415a75ced9e30ace9d81378a8fc3f196dbda79b3b249082d1d9",
    "task_input_evidence_source_sha256": "454f9096dea381bc0d3398eb78f021e17dc3825124aa8134d52a24df2b820028",
    "reference_packet_builder_source_sha256": "cc75600e7fe38cd8440b6371242deceeae2af71a911c04cec59fdade089c26be",
    "human_reference_scorer_source_sha256": "13dac56615690efcf8c302c06177a8aefa32eef514ec1d9d88cfcda1eddb1a43",
}
HEALTHVER_REFERENCE_SOURCE_HASHES_TOP3_20261015 = {
    **HEALTHVER_REFERENCE_SOURCE_HASHES_V2_20261015,
    "reference_packet_builder_source_sha256": "ec939b7dee52b2b380821069caf7f22b27f91056197cd1eae1758c0ca5b2a7ac",
}
HEALTHVER_SAME_CLAIM_TOP3_V1_SOURCE_HASHES_20261016 = {
    "calibration_selection_source_sha256": "5319eb1f0779367ed8811141f4386e83675314cb198625dffa7c0beb3087a90b",
    "task_input_evidence_source_sha256": "454f9096dea381bc0d3398eb78f021e17dc3825124aa8134d52a24df2b820028",
    "reference_packet_builder_source_sha256": "e2ba094777524ebad7cee8e8364d340019927899fad076fd6c3e4850fff1d749",
    "human_reference_scorer_source_sha256": "e01c4681377609256ca51d2e04bd0380c09417a2010c1dd166588f1dff6dacc0",
}
HEALTHVER_SAME_CLAIM_TOP3_V2_SOURCE_HASHES_20261016 = {
    "calibration_selection_source_sha256": "5319eb1f0779367ed8811141f4386e83675314cb198625dffa7c0beb3087a90b",
    "task_input_evidence_source_sha256": "454f9096dea381bc0d3398eb78f021e17dc3825124aa8134d52a24df2b820028",
    "reference_packet_builder_source_sha256": "f9a4950e8b24942586e9c87f18d3340eac526e4bb2d70718f3ad57148d2241ea",
    "human_reference_scorer_source_sha256": "8eab3d9ad325b2e01ca834a5ab8fcbdec735bad9628d827b9cd12a0a53c7cfb2",
}
VERDICTS = ("supported", "contradicted", "insufficient", "unable_to_judge")
ANNOTATION_KEYS = {
    "case_id", "verdict", "supporting_quote", "quote_doc_id", "reason",
    "confidence_1_to_3",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _current_source_hashes() -> dict[str, str]:
    """Return the immutable v2 PubMedQA fingerprints for legacy packets."""
    return dict(PUBMEDQA_REFERENCE_SOURCE_HASHES_V2)


def _validate_packet(packet: dict[str, Any], slot: str) -> dict[str, dict[str, Any]]:
    if packet.get("schema") != PACKET_SCHEMA:
        raise ValueError(f"{slot} packet schema is unsupported")
    if packet.get("reviewer_slot") != slot:
        raise ValueError(f"{slot} packet has the wrong reviewer slot")
    if not isinstance(packet.get("protocol_sha256"), str):
        raise ValueError(f"{slot} packet lacks a protocol hash")
    context = packet.get("protocol_context")
    if not isinstance(context, dict) or context.get("schema") not in {
            "task-input-grounding-reference-protocol-v2",
            "task-input-grounding-reference-protocol-v3",
            "task-input-grounding-reference-protocol-v4",
            "task-input-grounding-reference-protocol-v5",
            "task-input-grounding-reference-protocol-v6",
            "task-input-grounding-reference-protocol-v7"}:
        raise ValueError(f"{slot} packet has a malformed protocol context")
    version = context["schema"]
    expected_keys = (PROTOCOL_CONTEXT_KEYS if version.endswith("v2") else
                     PROTOCOL_CONTEXT_KEYS | {"case_count", "labels", "cohort_sha256"})
    if version.endswith("v4"):
        expected_keys |= {"sampling_seed", "history_exclusion"}
    if version.endswith("v5"):
        expected_keys |= {"documents_per_case", "exposure_audit", "source_label_semantics"}
    if version.endswith("v6"):
        expected_keys |= {"documents_per_case", "exposure_audit", "source_label_semantics",
                          "annotation_granularity", "aggregation_version"}
    if version.endswith("v7"):
        expected_keys |= {"documents_per_case", "exposure_audit", "source_label_semantics",
                          "annotation_granularity", "aggregation_version", "label_counts",
                          "parent_protocol_sha256", "parent_packet_sha256", "parent_plan_sha256",
                          "excluded_case_ids", "derivation_source_sha256"}
    if set(context) != expected_keys:
        raise ValueError(f"{slot} packet protocol fields are malformed")
    if packet["protocol_sha256"] != sha256_obj(context):
        raise ValueError(f"{slot} packet protocol identity does not match its context")
    dataset_ok = (context.get("dataset") == "pubmedqa" if version.endswith("v2")
                  else context.get("dataset") == "healthver"
                  and context.get("case_count", 0) >= 30
                  and context.get("labels") == ["supported", "contradicted", "insufficient"]
                  and isinstance(context.get("cohort_sha256"), str)
                  and len(context["cohort_sha256"]) == 64)
    if (not dataset_ok
            or context.get("split") != "dev_optimize"
            or type(context.get("per_label")) is not int
            or context["per_label"] < 1
            or type(context.get("offset_per_label")) is not int
            or context["offset_per_label"] < 0
            or type(context.get("judge_evidence_character_limit_per_document")) is not int
            or context["judge_evidence_character_limit_per_document"] < 1):
        raise ValueError(f"{slot} packet protocol scope is invalid")
    source_hashes = context.get("source_hashes")
    if (not isinstance(source_hashes, dict) or set(source_hashes) != {
                "calibration_selection_source_sha256",
                "task_input_evidence_source_sha256",
                "reference_packet_builder_source_sha256",
                "human_reference_scorer_source_sha256"}
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value)
                   for value in source_hashes.values())):
        raise ValueError(f"{slot} packet source fingerprints are malformed")
    legacy_source_hashes = None
    if version.endswith("v4"):
        from src.eval.healthver_calibration_cohort import RUBRIC, _source_hashes

        if context.get("rubric_version") != RUBRIC:
            raise ValueError(f"{slot} HealthVer recalibration rubric is unsupported")
        expected_source_hashes = _source_hashes()
    elif version.endswith("v2"):
        expected_source_hashes = PUBMEDQA_REFERENCE_SOURCE_HASHES_V2
    else:
        rubric_version = context.get("rubric_version")
        if rubric_version == "scientific-grounding-ai-reference-v1":
            # Preserve the frozen HealthVer v1 packet identities. New packets
            # use versioned builders so scorer changes cannot rewrite history.
            expected_source_hashes = HEALTHVER_REFERENCE_SOURCE_HASHES_V1
        elif rubric_version == "scientific-grounding-ai-reference-top3-v1":
            from scripts.prepare_healthver_top3_grounding_reference_packet import (
                _source_hashes as healthver_top3_source_hashes,
            )
            expected_source_hashes = healthver_top3_source_hashes()
            legacy_source_hashes = HEALTHVER_REFERENCE_SOURCE_HASHES_TOP3_20261015
        elif rubric_version == "scientific-grounding-ai-reference-same-claim-top3-v1":
            expected_source_hashes = HEALTHVER_SAME_CLAIM_TOP3_V1_SOURCE_HASHES_20261016
        elif rubric_version == "scientific-grounding-ai-reference-same-claim-top3-v2":
            from scripts.prepare_healthver_same_claim_top3_reference import (
                _source_hashes as healthver_same_claim_source_hashes,
            )
            expected_source_hashes = healthver_same_claim_source_hashes()
            if version.endswith("v6"):
                legacy_source_hashes = HEALTHVER_SAME_CLAIM_TOP3_V2_SOURCE_HASHES_20261016
        elif rubric_version == "scientific-grounding-ai-reference-v2":
            from scripts.prepare_healthver_grounding_reference_packet import (
                _source_hashes as healthver_source_hashes,
            )
            expected_source_hashes = healthver_source_hashes()
            legacy_source_hashes = HEALTHVER_REFERENCE_SOURCE_HASHES_V2_20261015
        else:
            raise ValueError(f"{slot} HealthVer rubric version is unsupported")
    if source_hashes != expected_source_hashes and source_hashes != legacy_source_hashes:
        raise ValueError(
            f"{slot} packet source fingerprints do not match the current implementation")
    unsigned = {key: value for key, value in packet.items()
                if key != "packet_sha256"}
    if packet.get("packet_sha256") != sha256_obj(unsigned):
        raise ValueError(f"{slot} packet self-hash is invalid")
    cases = packet.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{slot} packet has no cases")
    by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
                "case_id", "question", "evidence", "input_sha256"}:
            raise ValueError(f"{slot} packet contains malformed case fields")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in by_id:
            raise ValueError(f"{slot} packet has an empty or duplicate case ID")
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"{slot} packet has an empty question")
        docs = case.get("evidence")
        if not isinstance(docs, list) or not docs:
            raise ValueError(f"{slot} packet has no evidence for {case_id}")
        if version.endswith(("v5", "v6", "v7")) and len(docs) != context.get("documents_per_case"):
            raise ValueError(f"{slot} packet has the wrong document count for {case_id}")
        for doc in docs:
            if (not isinstance(doc, dict) or set(doc) != {"doc_id", "text"}
                    or not isinstance(doc.get("doc_id"), str)
                    or not isinstance(doc.get("text"), str)
                    or not doc["text"].strip()):
                raise ValueError(f"{slot} packet has malformed evidence for {case_id}")
        payload_hash = sha256_obj({
            "question": case["question"], "evidence": case["evidence"]})
        if case.get("input_sha256") != payload_hash:
            raise ValueError(f"{slot} packet input hash is invalid for {case_id}")
        by_id[case_id] = case
    expected_input_hashes = context.get("case_input_sha256s")
    actual_input_hashes = sorted(case["input_sha256"] for case in by_id.values())
    if (not isinstance(expected_input_hashes, list)
            or expected_input_hashes != actual_input_hashes
            or len(actual_input_hashes) != (
                2 * context["per_label"] if version.endswith("v2")
                else context["case_count"])):
        raise ValueError(f"{slot} packet inputs do not match the frozen protocol identity")
    scope = packet.get("scope")
    if (not isinstance(scope, dict) or scope.get("dataset") != context["dataset"]
            or scope.get("split") != context["split"]
            or scope.get("case_count") != len(by_id)):
        raise ValueError(f"{slot} packet scope does not match its protocol context")
    if version.endswith("v4"):
        from src.eval.healthver_calibration_cohort import validate_history_packet

        validate_history_packet(packet)
    if version.endswith("v7"):
        label_counts = context.get("label_counts")
        if (not isinstance(label_counts, dict)
                or set(label_counts) != {"supported", "contradicted", "insufficient"}
                or any(type(value) is not int or value < 1 for value in label_counts.values())
                or sum(label_counts.values()) != context.get("case_count")
                or min(label_counts.values()) < 3):
            raise ValueError(f"{slot} packet lacks balanced three-way source label coverage")
        scope_derivation = (packet.get("scope") or {}).get("derivation")
        if (not isinstance(scope_derivation, dict)
                or scope_derivation.get("parent_packet_sha256") != context.get("parent_packet_sha256")
                or scope_derivation.get("parent_protocol_sha256") != context.get("parent_protocol_sha256")
                or scope_derivation.get("parent_plan_sha256") != context.get("parent_plan_sha256")
                or scope_derivation.get("excluded_case_ids") != context.get("excluded_case_ids")):
            raise ValueError(f"{slot} packet derivation audit does not match its protocol")
    return by_id


def _validate_response(response: dict[str, Any], packet: dict[str, Any],
                       cases: dict[str, dict[str, Any]], slot: str) -> dict[str, dict[str, Any]]:
    if response.get("schema") != RESPONSE_SCHEMA:
        raise ValueError(f"{slot} response schema is unsupported")
    if response.get("protocol_sha256") != packet["protocol_sha256"]:
        raise ValueError(f"{slot} response protocol does not match its packet")
    if response.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError(f"{slot} response packet hash does not match its packet")
    reviewer_id = response.get("reviewer_id")
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise ValueError(f"{slot} reviewer ID is required")
    if response.get("blind_attestation") is not True:
        raise ValueError(f"{slot} reviewer must attest to blind independent annotation")
    annotations = response.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError(f"{slot} response annotations must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for annotation in annotations:
        if not isinstance(annotation, dict) or set(annotation) != ANNOTATION_KEYS:
            raise ValueError(f"{slot} response has malformed annotation fields")
        case_id = annotation.get("case_id")
        if not isinstance(case_id, str) or case_id not in cases:
            raise ValueError(f"{slot} response has an unknown case ID")
        if case_id in by_id:
            raise ValueError(f"{slot} response has duplicate annotation for {case_id}")
        verdict = annotation.get("verdict")
        if verdict not in VERDICTS:
            raise ValueError(f"{slot} response has invalid verdict for {case_id}")
        reason = annotation.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{slot} response lacks a reason for {case_id}")
        confidence = annotation.get("confidence_1_to_3")
        if type(confidence) is not int or confidence not in (1, 2, 3):
            raise ValueError(f"{slot} response has invalid confidence for {case_id}")
        quote = annotation.get("supporting_quote")
        doc_id = annotation.get("quote_doc_id")
        if verdict in {"supported", "contradicted"}:
            if not isinstance(quote, str) or len(_norm(quote)) < 20:
                raise ValueError(f"{slot} response needs an exact substantial quote for {case_id}")
            docs = [doc for doc in cases[case_id]["evidence"]
                    if doc["doc_id"] == doc_id]
            if len(docs) != 1 or _norm(quote) not in _norm(docs[0]["text"]):
                raise ValueError(f"{slot} response quote is not in supplied evidence for {case_id}")
        elif quote or doc_id:
            if not isinstance(quote, str) or not isinstance(doc_id, str):
                raise ValueError(f"{slot} response has malformed optional quote for {case_id}")
            docs = [doc for doc in cases[case_id]["evidence"]
                    if doc["doc_id"] == doc_id]
            if len(docs) != 1 or (quote and _norm(quote) not in _norm(docs[0]["text"])):
                raise ValueError(f"{slot} response quote is not in supplied evidence for {case_id}")
        by_id[case_id] = annotation
    if set(by_id) != set(cases):
        missing = sorted(set(cases) - set(by_id))
        raise ValueError(f"{slot} response is incomplete; missing {len(missing)} cases")
    return by_id


def _cohen_kappa(a: list[str], b: list[str]) -> tuple[float | None, float, float]:
    if len(a) != len(b) or not a:
        return None, 0.0, 0.0
    observed = sum(left == right for left, right in zip(a, b)) / len(a)
    labels = set(a) | set(b)
    expected = sum(
        (sum(value == label for value in a) / len(a))
        * (sum(value == label for value in b) / len(b))
        for label in labels
    )
    if expected == 1.0:
        # Both reviewers used the same single category throughout. Observed
        # agreement is informative, but kappa's normalization denominator is
        # zero and its value is mathematically undefined.
        return None, observed, expected
    return (observed - expected) / (1.0 - expected), observed, expected


def score_human_reference(packet_a: dict[str, Any], packet_b: dict[str, Any],
                          response_a: dict[str, Any],
                          response_b: dict[str, Any]) -> dict[str, Any]:
    """Validate two complete blind reviews and report agreement/consensus only."""
    cases_a = _validate_packet(packet_a, "reviewer_a")
    cases_b = _validate_packet(packet_b, "reviewer_b")
    if packet_a["protocol_sha256"] != packet_b["protocol_sha256"]:
        raise ValueError("reviewer packets belong to different protocols")
    if packet_a["protocol_context"] != packet_b["protocol_context"]:
        raise ValueError("reviewer packets have different protocol contexts")
    if set(cases_a) != set(cases_b):
        raise ValueError("reviewer packets contain different case sets")
    for case_id in cases_a:
        if cases_a[case_id] != cases_b[case_id]:
            raise ValueError(f"reviewer packets differ in task inputs for {case_id}")
    labels_a = _validate_response(response_a, packet_a, cases_a, "reviewer_a")
    labels_b = _validate_response(response_b, packet_b, cases_b, "reviewer_b")
    reviewer_a_id = response_a["reviewer_id"].strip()
    reviewer_b_id = response_b["reviewer_id"].strip()
    if reviewer_a_id.casefold() == reviewer_b_id.casefold():
        raise ValueError("two distinct reviewer IDs are required")

    case_ids = sorted(cases_a)
    a_values = [labels_a[case_id]["verdict"] for case_id in case_ids]
    b_values = [labels_b[case_id]["verdict"] for case_id in case_ids]
    kappa, observed, expected = _cohen_kappa(a_values, b_values)
    confusion = {left: {right: 0 for right in VERDICTS} for left in VERDICTS}
    rows = []
    for case_id in case_ids:
        a_label = labels_a[case_id]["verdict"]
        b_label = labels_b[case_id]["verdict"]
        confusion[a_label][b_label] += 1
        consensus = a_label if a_label == b_label else None
        rows.append({
            "case_id": case_id,
            "reviewer_a_verdict": a_label,
            "reviewer_b_verdict": b_label,
            "consensus_verdict": consensus,
            "consensus_usable_for_semantic_calibration": (
                consensus in {"supported", "contradicted", "insufficient"}),
            "reviewer_a_confidence": labels_a[case_id]["confidence_1_to_3"],
            "reviewer_b_confidence": labels_b[case_id]["confidence_1_to_3"],
        })
    agreed = sum(row["reviewer_a_verdict"] == row["reviewer_b_verdict"]
                 for row in rows)
    usable = sum(row["consensus_usable_for_semantic_calibration"] for row in rows)
    unable = sum(row["consensus_verdict"] == "unable_to_judge" for row in rows)
    unresolved = len(rows) - agreed
    response_hashes = {
        "reviewer_a": sha256_obj(response_a),
        "reviewer_b": sha256_obj(response_b),
    }
    unsigned = {
        "schema": "task-input-grounding-human-reference-agreement-v1",
        "protocol_sha256": packet_a["protocol_sha256"],
        "protocol_context": packet_a["protocol_context"],
        "reviewer_packet_sha256": {
            "reviewer_a": packet_a["packet_sha256"],
            "reviewer_b": packet_b["packet_sha256"],
        },
        "response_sha256": response_hashes,
        "reviewer_ids_distinct": True,
        "n_cases": len(rows),
        "n_agreement": agreed,
        "exact_agreement_rate": observed,
        "cohen_kappa_nominal": kappa,
        "cohen_kappa_expected_agreement": expected,
        "n_usable_consensus": usable,
        "n_unable_to_judge_consensus": unable,
        "n_disagreement_unresolved": unresolved,
        "agreement_matrix_reviewer_a_rows_reviewer_b_columns": confusion,
        "decision_policy": (
            "no adjudication or forced consensus; only matching supported, "
            "contradicted, and insufficient labels are usable; unable_to_judge "
            "consensus is excluded from semantic calibration"),
        "calibration_pass": None,
        "per_case": rows,
    }
    return {**unsigned, "report_sha256": sha256_obj(unsigned)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-a", type=Path, required=True)
    parser.add_argument("--packet-b", type=Path, required=True)
    parser.add_argument("--responses-a", type=Path, required=True)
    parser.add_argument("--responses-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score_human_reference(
        _load(args.packet_a), _load(args.packet_b),
        _load(args.responses_a), _load(args.responses_b))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "n_cases", "n_agreement", "exact_agreement_rate", "cohen_kappa_nominal",
        "n_usable_consensus", "n_disagreement_unresolved", "report_sha256")},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
