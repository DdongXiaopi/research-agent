"""Hash-bound human attribution corrections (Spec §5.2 / FR-104)."""

from __future__ import annotations

from typing import Any

from ..contracts.types import ATTRIBUTION_CLASSES, EXTENDED_ATTRIBUTION_CLASSES
from ..util import sha256_obj

REVIEW_PACKET_SCHEMA = "human-attribution-review-packet-v1"
ATTRIBUTION_VALUES = set(ATTRIBUTION_CLASSES) | set(EXTENDED_ATTRIBUTION_CLASSES)


def _failure_identity(row: dict[str, Any]) -> dict[str, Any]:
    # Corrections bind every machine-produced field. Derived human fields are
    # excluded so applying a document twice remains deterministic.
    return {key: value for key, value in row.items()
            if key not in {"human_correction", "effective_attributed_class",
                           "effective_target_artifact", "effective_mainline",
                           "effective_event_ids"}}


def _packet_identity(document: dict[str, Any]) -> str:
    """Recompute the stable identity after blank cells have been edited."""
    identity = {key: value for key, value in document.items()
                if key != "review_packet_id"}
    corrections = []
    for row in identity.get("corrections") or []:
        normalized = dict(row)
        normalized.update({"corrected_class": None, "reason": None, "reviewer": None,
                           "trace_event_ids": []})
        corrections.append(normalized)
    identity["corrections"] = corrections
    return sha256_obj(identity)


def build_attribution_review_packet(
        failures: list[dict[str, Any]], *, source_run_ids: list[str]) -> dict[str, Any]:
    """Create one pending, complete, immutable-identity review row per failure."""
    if not source_run_ids or not all(isinstance(value, str) and value for value in source_run_ids):
        raise ValueError("source run IDs must be nonempty strings")
    indexed: dict[str, dict[str, Any]] = {}
    corrections = []
    for row in failures:
        if not isinstance(row, dict) or not isinstance(row.get("failure_id"), str):
            raise ValueError("every failure row needs a stable failure_id")
        failure_id = row["failure_id"]
        if failure_id in indexed:
            raise ValueError("failure IDs must be unique")
        if row.get("source_run_id") not in source_run_ids:
            raise ValueError("failure source run is not declared in review packet")
        indexed[failure_id] = row
        trace_ids = row.get("trace_event_ids")
        if not isinstance(trace_ids, list):
            trace_ids = list(dict.fromkeys(row.get("event_ids") or []))
        corrections.append({
            "failure_id": failure_id,
            "source_failure_sha256": sha256_obj(_failure_identity(row)),
            "original_class": row.get("attributed_class"),
            "corrected_class": None,
            "reason": None,
            "reviewer": None,
            "trace_event_ids": [],
            "available_trace_event_ids": list(dict.fromkeys(
                str(value) for value in trace_ids if isinstance(value, str) and value)),
        })
    packet = {
        "schema": REVIEW_PACKET_SCHEMA,
        "source_run_ids": list(dict.fromkeys(source_run_ids)),
        "empty_source_run_ids": sorted(set(source_run_ids) -
                                       {row.get("source_run_id") for row in failures}),
        "source_failures_sha256": sha256_obj([
            _failure_identity(indexed[key]) for key in sorted(indexed)]),
        "failure_count": len(corrections),
        "corrections": corrections,
    }
    packet["review_packet_id"] = sha256_obj(packet)
    return packet


def apply_attribution_corrections(
        failures: list[dict[str, Any]], document: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate a human correction document and expose separate effective labels.

    The machine `attributed_class`, its supporting `event_ids`, and its causal
    flag remain untouched. Only a complete correction with exact source/hash
    identity, reviewer rationale, and trace references may set an effective
    class. Extended/system classes remain ineligible for candidate generation.
    """
    if not isinstance(document, dict) or document.get("schema") != REVIEW_PACKET_SCHEMA:
        raise ValueError("unsupported human attribution review schema")
    source_runs = document.get("source_run_ids")
    if (not isinstance(source_runs, list) or not source_runs
            or not all(isinstance(value, str) and value for value in source_runs)):
        raise ValueError("review packet source run IDs are invalid")
    if not isinstance(failures, list):
        raise ValueError("failures must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for row in failures:
        if not isinstance(row, dict) or not isinstance(row.get("failure_id"), str):
            raise ValueError("every failure row needs a stable failure_id")
        if row["failure_id"] in indexed:
            raise ValueError("source failure IDs are not unique")
        indexed[row["failure_id"]] = row
    expected_hash = sha256_obj([
        _failure_identity(indexed[key]) for key in sorted(indexed)])
    if document.get("source_failures_sha256") != expected_hash:
        raise ValueError("source failure hash-set mismatch; regenerate the review packet")
    actual_run_ids = {row.get("source_run_id") for row in failures}
    empty = document.get("empty_source_run_ids", [])
    if (not isinstance(empty, list) or not all(isinstance(value, str) for value in empty)
            or len(empty) != len(set(empty)) or actual_run_ids & set(empty)
            or actual_run_ids | set(empty) != set(source_runs)):
        raise ValueError("review packet source run IDs do not match failure provenance")
    rows = document.get("corrections")
    if (not isinstance(rows, list) or len(rows) != len(indexed)
            or document.get("failure_count") != len(indexed)):
        raise ValueError("review packet must cover every source failure exactly once")
    if document.get("review_packet_id") != _packet_identity(document):
        raise ValueError("review packet identity mismatch")
    correction_index: dict[str, dict[str, Any]] = {}
    for correction in rows:
        if not isinstance(correction, dict):
            raise ValueError("correction entry must be an object")
        if set(correction) != {
                "failure_id", "source_failure_sha256", "original_class",
                "corrected_class", "reason", "reviewer", "trace_event_ids",
                "available_trace_event_ids"}:
            raise ValueError("correction entry contains missing or unsupported fields")
        failure_id = correction.get("failure_id")
        if failure_id not in indexed or failure_id in correction_index:
            raise ValueError("review packet must cover every source failure exactly once")
        source = indexed[failure_id]
        if correction.get("source_failure_sha256") != sha256_obj(_failure_identity(source)):
            raise ValueError("source failure hash mismatch")
        if correction.get("original_class") != source.get("attributed_class"):
            raise ValueError("original class does not match machine attribution")
        trace_ids = source.get("trace_event_ids")
        if not isinstance(trace_ids, list):
            trace_ids = list(dict.fromkeys(source.get("event_ids") or []))
        expected_available = list(dict.fromkeys(
            str(value) for value in trace_ids if isinstance(value, str) and value))
        # This is generated provenance metadata, not an editable reviewer field.
        # Require the exact canonical list so duplicates or reordered IDs cannot
        # become a second, misleading representation of the source trace.
        if correction.get("available_trace_event_ids") != expected_available:
            raise ValueError("available trace event IDs differ from source failure")
        correction_index[failure_id] = correction
    if set(correction_index) != set(indexed):
        raise ValueError("review packet must cover every source failure exactly once")

    result = []
    for failure_id in sorted(indexed):
        source = indexed[failure_id]
        correction = correction_index[failure_id]
        corrected_class = correction.get("corrected_class")
        reason = correction.get("reason")
        reviewer = correction.get("reviewer")
        trace_event_ids = correction.get("trace_event_ids")
        available = set(correction["available_trace_event_ids"])
        pending = (corrected_class is None and reason is None and reviewer is None
                   and trace_event_ids == [])
        copy = dict(source)
        original = source.get("attributed_class")
        if pending:
            copy.update({
                "effective_attributed_class": original,
                "effective_target_artifact": source.get("target_artifact"),
                "effective_mainline": bool(source.get("mainline")),
                "effective_event_ids": list(source.get("event_ids") or []),
                "human_correction": {"applied": False, "original_class": original},
            })
            result.append(copy)
            continue
        if corrected_class not in ATTRIBUTION_VALUES:
            raise ValueError("corrected class is invalid")
        if not isinstance(reason, str) or not reason.strip() \
                or not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("applied correction requires reason and reviewer")
        if (not isinstance(trace_event_ids, list) or not trace_event_ids
                or len(trace_event_ids) != len(set(trace_event_ids))
                or not all(isinstance(value, str) and value for value in trace_event_ids)
                or not set(trace_event_ids).issubset(available)):
            raise ValueError("correction must cite available trace event IDs")
        from ..contracts.types import ATTRIBUTION_TARGET

        copy.update({
            "effective_attributed_class": corrected_class,
            "effective_target_artifact": ATTRIBUTION_TARGET.get(corrected_class),
            "effective_mainline": corrected_class in ATTRIBUTION_CLASSES,
            "effective_event_ids": list(trace_event_ids),
            "human_correction": {
                "applied": True, "original_class": original,
                "corrected_class": corrected_class, "reason": reason.strip(),
                "reviewer": reviewer.strip(), "trace_event_ids": list(trace_event_ids),
                "source_failure_sha256": correction["source_failure_sha256"],
                "review_packet_id": document.get("review_packet_id"),
                "attribution_is_causal": False,
            },
        })
        result.append(copy)
    return result
