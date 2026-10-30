"""Merge an AI annotated natural-source supplement into the G06 capacity audit.

The historical census and the external Natural Instructions proxy have
different packet schemas and must never be concatenated by ad-hoc scripts.
This module makes the merge explicit, preserves each source's provenance, and
separates the class-capacity result from production eligibility.  It is an
evidence artifact for the AI-annotation policy; it does not turn an external
proxy into human gold or silently open the reviewer gate.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from ..util import sha256_obj

SCHEMA = "g06-natural-ai-reference-merge-v1"
MIN_REFERENCE = 3
_DIMENSIONS = {
    "assertion": ("claim_free", "contains_claims"),
    "contract": ("compatible", "incompatible"),
}


def _require_hash(value: Mapping[str, Any], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or len(result) != 64:
        raise ValueError(f"{field} is missing or malformed")
    return result


def _external_case_rows(packet: Mapping[str, Any], output: Mapping[str, Any]) -> list[dict[str, Any]]:
    if packet.get("schema") != "instruction-reviewer-external-source-packet-v1":
        raise ValueError("external packet schema is not supported")
    if output.get("schema") != "external-instruction-reference-v1":
        raise ValueError("external reference schema is not supported")
    if output.get("packet_sha256") != packet.get("packet_sha256"):
        raise ValueError("external reference is not bound to its packet")
    packet_cases = {row["case_id"]: row for row in packet.get("cases", [])}
    rows = output.get("cases")
    if not isinstance(rows, list) or set(row.get("case_id") for row in rows) != set(packet_cases):
        raise ValueError("external reference case coverage is incomplete")
    result = []
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("verdicts"), Mapping):
            raise ValueError("external reference row is malformed")
        verdicts = row["verdicts"]
        if any(verdicts.get(dim) not in (*labels, "uncertain")
               for dim, labels in _DIMENSIONS.items()):
            raise ValueError("external reference has an invalid verdict")
        originals = row.get("originals")
        if not isinstance(originals, list) or len(originals) != 2:
            raise ValueError("external reference requires two blind AI reviews")
        run_ids = []
        for review in originals:
            actor = review.get("actor") if isinstance(review, Mapping) else None
            if not isinstance(actor, Mapping) or actor.get("blind") is not True:
                raise ValueError("external review is not blind")
            run_id = actor.get("review_run_id")
            if not isinstance(run_id, str) or not run_id.strip():
                raise ValueError("external review run identity is missing")
            run_ids.append(run_id)
        if len(set(run_ids)) != 2:
            raise ValueError("external AI reviews must have separate run identities")
        result.append({
            "case_id": row["case_id"],
            "source_kind": "external_natural_instructions",
            "candidate_sha256": packet_cases[row["case_id"]]["candidate_sha256"],
            "verdicts": dict(verdicts),
            "review_provenance": [
                {"model_id": review["actor"].get("model_id"),
                 "review_run_id": review["actor"]["review_run_id"]}
                for review in originals
            ],
        })
    return sorted(result, key=lambda row: row["case_id"])


def _historical_rows(packet: Mapping[str, Any], reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    if packet.get("schema") != "natural-instruction-candidate-census-v1":
        raise ValueError("historical packet schema is not supported")
    if reference.get("schema") not in {
        "natural-instruction-frozen-reference-v1",
        "natural-instruction-frozen-reference-v2",
        "natural-instruction-frozen-reference-v3",
    } or reference.get("packet_sha256") != packet.get("packet_sha256"):
        raise ValueError("historical reference is not bound to its packet")
    packet_cases = {row["case_id"]: row for row in packet.get("cases", [])}
    rows = reference.get("cases")
    if not isinstance(rows, list) or set(row.get("case_id") for row in rows) != set(packet_cases):
        raise ValueError("historical reference case coverage is incomplete")
    result = []
    for row in rows:
        verdicts = row.get("verdicts")
        if not isinstance(verdicts, Mapping) or any(
                verdicts.get(dim) not in (*labels, "uncertain")
                for dim, labels in _DIMENSIONS.items()):
            raise ValueError("historical reference has an invalid verdict")
        result.append({
            "case_id": row["case_id"],
            "source_kind": "historical_candidate_census",
            "candidate_sha256": packet_cases[row["case_id"]]["candidate_sha256"],
            "verdicts": dict(verdicts),
            "review_provenance": [
                {"model_id": reference.get("reviewer_a", {}).get("model_id"),
                 "receipt_sha256": reference.get("reviewer_a", {}).get("receipt_sha256")},
                {"model_id": reference.get("reviewer_b", {}).get("model_id"),
                 "receipt_sha256": reference.get("reviewer_b", {}).get("receipt_sha256")},
            ],
        })
    return sorted(result, key=lambda row: row["case_id"])


def merge_references(historical_packet: Mapping[str, Any],
                     historical_reference: Mapping[str, Any],
                     external_packet: Mapping[str, Any],
                     external_reference: Mapping[str, Any]) -> dict[str, Any]:
    """Create a provenance-preserving capacity merge without opening promotion."""
    rows = _historical_rows(historical_packet, historical_reference)
    rows.extend(_external_case_rows(external_packet, external_reference))
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("merged reference case IDs overlap")
    coverage = {}
    for dimension, labels in _DIMENSIONS.items():
        counts = Counter(row["verdicts"][dimension] for row in rows)
        coverage[dimension] = {
            "approval_reference": counts[labels[0]],
            "rejection_reference": counts[labels[1]],
            "uncertain_reference": counts["uncertain"],
        }
    capacity_pass = all(
        coverage[dimension][key] >= MIN_REFERENCE
        for dimension in _DIMENSIONS
        for key in ("approval_reference", "rejection_reference")
    )
    body = {
        "schema": SCHEMA,
        "annotation_origin": "ai",
        "ai_annotation_policy_accepted": True,
        "natural_distribution": True,
        "source_packet_sha256": {
            "historical": _require_hash(historical_packet, "packet_sha256"),
            "external": _require_hash(external_packet, "packet_sha256"),
        },
        "source_reference_sha256": {
            "historical": _require_hash(historical_reference, "reference_sha256"),
            "external": sha256_obj(external_reference),
        },
        "cases": rows,
        "case_count": len(rows),
        "reference_class_coverage": coverage,
        "reference_class_coverage_sufficient": capacity_pass,
        "production_gate_eligible": False,
        "target_evaluation_authorized": False,
        "limitations": [
            "external rows are an AI-annotated natural-source supplement, not human gold",
            "external review uses separate blind runs but the same model family",
            "source-family independence and current-candidate representativeness remain unverified",
        ],
    }
    return {**body, "reference_sha256": sha256_obj(body)}


def validate_merged_reference(value: Mapping[str, Any]) -> None:
    if value.get("schema") != SCHEMA or value.get("annotation_origin") != "ai":
        raise ValueError("merged G06 reference schema or annotation origin is invalid")
    if value.get("reference_sha256") != sha256_obj(
            {key: item for key, item in value.items() if key != "reference_sha256"}):
        raise ValueError("merged G06 reference hash mismatch")
    if value.get("case_count") != len(value.get("cases", [])):
        raise ValueError("merged G06 reference case count mismatch")
    for dimension, labels in _DIMENSIONS.items():
        coverage = value.get("reference_class_coverage", {}).get(dimension, {})
        if any(not isinstance(coverage.get(key), int) for key in
               ("approval_reference", "rejection_reference", "uncertain_reference")):
            raise ValueError("merged G06 reference coverage is malformed")


__all__ = ["SCHEMA", "merge_references", "validate_merged_reference"]
