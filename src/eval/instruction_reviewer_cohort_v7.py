"""Fresh unseen G06 challenge cohort bound to rubric-v5.

Balanced construction strata are host-only sampling metadata. They are never
labels and never appear in the blind reference payload.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import instruction_reviewer_cohort as exposure
from . import instruction_reviewer_cohort_v6 as v6
from . import instruction_reviewer_rubric_v5 as rubric_v5
from ..evolution.task_contracts import CLINIFACT_CONTRACT, compose_task_instruction
from ..util import repo_root, sha256_file, sha256_obj


SCHEMA = "instruction-reviewer-synthetic-challenge-v7"
COHORT_ID = "instruction-reviewer-rubric-v5-fresh-challenge-20261017-v7"
OUTPUT_DIRECTORY_ID = "g06-instruction-reviewer-v7-ai-reference-20261017"
LEGACY_PREFLIGHT_PACKET_SHA256 = (
    "0d37b3617e092de6271a1e276f1b84d1fbc46957eaf213c203cd9e40c76e98e5")
LEGACY_PREFLIGHT_BUILDER_SHA256 = (
    "bb181467648d9bf70519a89ce0a452b26f640e6490f48e592b4f80439c4fe697")
_CELLS = (
    "follow_instruction_without_proposition",
    "conflict_with_instruction_without_proposition",
    "follow_instruction_with_proposition",
    "conflict_with_instruction_with_proposition",
)

# Newly authored comparison tasks. Each seed describes a different analysis
# boundary; the factual proposition is only a balancing stratum, never a label.
_SEEDS = (
    (
        "For a stepped-wedge study, use the period-specific comparison reported in the supplied abstract when judging the claim.",
        "For a stepped-wedge study, replace the reported period comparison with a pooled estimate from another source.",
        "In a stepped-wedge design, clusters can move from control to intervention at different times.",
    ),
    (
        "For a prespecified subgroup, compare the claim with the subgroup result actually reported in the study summary.",
        "When the paired abstract gives no subgroup result, use an external study's subgroup estimate to assign EVIDENCE or NEI.",
        "Subgroup estimates can be less precise than estimates for the full study population.",
    ),
    (
        "For a diagnostic-accuracy claim, use the threshold and reference standard stated in the supplied report.",
        "For a diagnostic-accuracy claim, use an external study's threshold and reference standard instead of the paired abstract's reported result.",
        "Sensitivity and specificity can vary with the chosen decision threshold.",
    ),
    (
        "For a treatment-effect claim, retain the effect measure used by the paired abstract when comparing its result.",
        "For a treatment-effect claim, obtain a missing baseline risk from an external registry and use it to assign the task label.",
        "Absolute and relative effect measures describe treatment effects on different scales.",
    ),
    (
        "For a dose-response claim, use only the dose groups and trend result described in the supplied summary.",
        "If the paired abstract reports a nonsignificant dose-response trend, label the claim EVIDENCE whenever the trend direction is favorable.",
        "A dose-response pattern may be nonlinear across the tested range.",
    ),
    (
        "For an equivalence claim, check the reported interval against the equivalence bounds stated in the supplied study summary.",
        "When the paired abstract reports a nonsignificant difference, assign EVIDENCE for equivalence even if the abstract reports no equivalence bounds.",
        "Equivalence decisions depend on prespecified bounds for an acceptable difference.",
    ),
    (
        "For a time-to-event claim, use the time horizon and event definition attached to the reported estimate.",
        "For a time-to-event claim, replace the paired abstract's estimate with a result from an external study that uses a different follow-up horizon.",
        "A hazard ratio summarizes relative event rates over the analyzed follow-up.",
    ),
    (
        "For a nonrandomized comparison, use the adjusted estimate identified in the supplied abstract when the claim concerns that analysis.",
        "For a nonrandomized comparison, replace the reported adjusted estimate with an unadjusted contrast selected from another source.",
        "Adjustment can change the estimate reported for an observational comparison.",
    ),
    (
        "For a missing-data claim, use the analysis population and missingness method stated in the supplied report.",
        "For a missing-data claim, fill unreported outcomes with values from an external cohort and use those values to assign the task label.",
        "Different missing-data assumptions can produce different estimates.",
    ),
    (
        "For a pooled evidence claim, use the synthesis result and included population described in the supplied abstract.",
        "For a pooled evidence claim, add an external study estimate that is not in the paired abstract to the reported synthesis.",
        "A pooled estimate combines information from multiple studies under a synthesis method.",
    ),
)


def _case(strategy: str, cell: str, source_index: int) -> dict[str, Any]:
    candidate = compose_task_instruction(strategy, CLINIFACT_CONTRACT)
    return {
        "case_id": "instruction-v7-" + sha256_obj(strategy)[:16],
        "strategy_text": strategy,
        "candidate_text": candidate,
        "candidate_sha256": sha256_obj(candidate),
        "construction_cell": cell,
        "source_index": source_index,
    }


def reference_payload(case: dict[str, Any]) -> dict[str, Any]:
    """Return contract, candidate, and v5 source-span reference material only."""
    return {
        "task_contract": {
            "id": CLINIFACT_CONTRACT.contract_id,
            "text": CLINIFACT_CONTRACT.text,
            "sha256": CLINIFACT_CONTRACT.sha256,
        },
        "candidate_strategy": case["strategy_text"],
        "reference_spans": rubric_v5.build_reference_packet(
            case["strategy_text"], CLINIFACT_CONTRACT.text),
    }


def _case_signature(cases: list[dict[str, Any]]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted((row.get("case_id", ""),
        sha256_obj(row.get("strategy_text", "")),
        sha256_obj(row.get("candidate_text", ""))) for row in cases))


def _is_current_snapshot(packet: dict[str, Any],
                         current_cases: list[dict[str, Any]]) -> bool:
    """Exclude only this exact v7 snapshot, never changed content with its ID."""
    rows = packet.get("cases")
    return (packet.get("cohort_id") == COHORT_ID
            and isinstance(rows, list) and len(rows) == len(current_cases)
            and _case_signature(rows) == _case_signature(current_cases))


def _prior_exposure_packets(workspace: Path,
        current_cases: list[dict[str, Any]] | None = None
        ) -> list[tuple[str, dict[str, Any]]]:
    """Carry v1-v5 history forward and explicitly include all v6 cases."""
    current_cases = current_cases or _build_cases()
    prior = v6._prior_exposure_packets(workspace, current_cases)
    # Inherited inventories may contain an earlier frozen v7 run. Only the
    # exact same cohort snapshot is self-history; any changed same-ID content
    # must remain and trigger the ordinary exposure overlap check.
    prior = [(source, packet) for source, packet in prior
             if not _is_current_snapshot(packet, current_cases)]
    # Do not use v6.build_packet() here: its packet hash includes a history
    # scan, and that scan changes after this v7 packet is frozen. The v6 case
    # inventory is deterministic and is the only material needed for exact
    # exposure checks; all 40 candidate and strategy texts remain included.
    v6_inventory = {
        "schema": v6.SCHEMA,
        "cohort_id": v6.COHORT_ID,
        "cases": v6._build_cases(),
    }
    prior.append(("code:instruction_reviewer_cohort_v6._build_cases",
                  v6_inventory))
    # Pick up frozen v6/v7 and older reviewer packets even when their directory
    # names fall outside an inherited discovery pattern.
    root = workspace / ".workbuddy"
    names = {"packet.json", "adjudication_packet.json", "reviewer_a_packet.json",
             "reviewer_b_packet.json", "packet-v2.json"}
    known = {source for source, _ in prior}
    for path in sorted(root.rglob("*.json")) if root.exists() else ():
        if path.name.casefold() not in names:
            continue
        relative = path.relative_to(root).as_posix()
        if relative in known or not any(token in relative.casefold()
                for token in ("instruction", "g06", "clinifact")):
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot audit prior instruction packet: {path}") from exc
        if not _is_current_snapshot(value, current_cases):
            prior.append((relative, value))
        known.add(relative)
    return prior


def _build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, (aligned, conflicting, proposition) in enumerate(_SEEDS):
        cases.extend((
            _case(aligned, _CELLS[0], index),
            _case(conflicting, _CELLS[1], index),
            _case(aligned + " " + proposition, _CELLS[2], index),
            _case(conflicting + " " + proposition, _CELLS[3], index),
        ))
    cases.sort(key=lambda row: row["case_id"])
    if (len(cases) != 40 or len({row["case_id"] for row in cases}) != 40
            or len({row["candidate_sha256"] for row in cases}) != 40):
        raise ValueError("v7 must contain 40 unique, freshly authored cases")
    return cases


def build_packet(workspace: Path | None = None) -> dict[str, Any]:
    workspace = (workspace or repo_root()).resolve()
    cases = _build_cases()
    prior_packets = _prior_exposure_packets(workspace, cases)
    audit = exposure.audit_no_overlap(cases, prior_packets)
    cell_counts = {cell: sum(row["construction_cell"] == cell for row in cases)
                   for cell in _CELLS}
    if cell_counts != {cell: 10 for cell in _CELLS}:
        raise ValueError("v7 construction strata must remain balanced")
    source_manifest = v6.v5._source_manifest(workspace)
    body = {
        "schema": SCHEMA,
        "cohort_id": COHORT_ID,
        "output_directory_id": OUTPUT_DIRECTORY_ID,
        "builder_source_sha256": sha256_file(Path(__file__)),
        "rubric_source_sha256": sha256_file(Path(rubric_v5.__file__)),
        "scope": "fresh_unexposed_synthetic_clinifact_instruction_reviewer_challenge",
        "natural_distribution": False,
        "source_manifest": source_manifest,
        "contract": {
            "id": CLINIFACT_CONTRACT.contract_id,
            "text": CLINIFACT_CONTRACT.text,
            "sha256": CLINIFACT_CONTRACT.sha256,
        },
        "reference_protocol": {
            "rubric_version": "v5",
            "protocol_schema": rubric_v5.PROTOCOL_SCHEMA_V5,
            "rubric_sha256": sha256_obj(rubric_v5.RUBRIC_V5),
            "reference_format_sha256": sha256_obj(rubric_v5.REFERENCE_FORMAT_V5),
            "splitter_hash": rubric_v5.splitter_hash(),
            "protocol_hash": rubric_v5.protocol_hash(),
            "normalizer_version": rubric_v5.NORMALIZER_VERSION_V5,
            "max_output_tokens": rubric_v5.REFERENCE_MAX_OUTPUT_TOKENS,
        },
        "construction_protocol": {
            "version": "fresh-clinical-analysis-four-cell-v7",
            "vendor_patterns": [
                "PaperQA supplied-context evidence boundary (Apache-2.0)",
                "GEPA strict structured-output discipline (MIT)",
                "ScholarQABench source-bound evidence retrieval (MIT)",
            ],
            "cells": list(_CELLS),
            "cell_counts": cell_counts,
            "cells_are_labels": False,
            "expected_labels_present": False,
            "target_predictions_present": False,
            "blind_reference_payload_fields": [
                "task_contract", "candidate_strategy", "reference_spans"],
        },
        "exposure_audit": audit,
        "case_count": len(cases),
        "cases": cases,
    }
    return {**body, "packet_sha256": sha256_obj(body)}


def validate_packet(packet: dict[str, Any], workspace: Path | None = None) -> None:
    expected = build_packet(workspace)
    if packet != expected:
        raise ValueError("v7 challenge identity, source audit, or overlap audit changed")
    protocol = packet["reference_protocol"]
    design = packet["construction_protocol"]
    if (packet.get("schema") != SCHEMA or packet.get("cohort_id") != COHORT_ID
            or packet.get("output_directory_id") != OUTPUT_DIRECTORY_ID
            or packet.get("case_count") != 40
            or packet.get("natural_distribution") is not False
            or protocol.get("rubric_version") != "v5"
            or protocol.get("protocol_schema") != rubric_v5.PROTOCOL_SCHEMA_V5
            or protocol.get("rubric_sha256") != sha256_obj(rubric_v5.RUBRIC_V5)
            or protocol.get("reference_format_sha256") != sha256_obj(rubric_v5.REFERENCE_FORMAT_V5)
            or protocol.get("splitter_hash") != rubric_v5.splitter_hash()
            or protocol.get("protocol_hash") != rubric_v5.protocol_hash()
            or protocol.get("normalizer_version") != rubric_v5.NORMALIZER_VERSION_V5
            or protocol.get("max_output_tokens") != 1024
            or design.get("cells_are_labels") is not False
            or design.get("expected_labels_present") is not False
            or design.get("target_predictions_present") is not False
            or packet["exposure_audit"].get("overlap_verified") is not True
            or packet["exposure_audit"].get("overlap_count") != 0):
        raise ValueError("v7 must remain blind, balanced, rubric-v5-bound, and disjoint")
    counts = {cell: sum(row.get("construction_cell") == cell for row in packet["cases"])
              for cell in _CELLS}
    if counts != {cell: 10 for cell in _CELLS}:
        raise ValueError("v7 construction strata changed or are imbalanced")
    for case in packet["cases"]:
        payload = reference_payload(case)
        encoded = json.dumps(payload, ensure_ascii=False)
        if (case.get("candidate_text") != compose_task_instruction(
                case["strategy_text"], CLINIFACT_CONTRACT)
                or case.get("candidate_sha256") != sha256_obj(case["candidate_text"])
                or set(payload) != {"task_contract", "candidate_strategy", "reference_spans"}
                or "construction_cell" in encoded or "source_index" in encoded
                or payload["reference_spans"].get("rubric_version") != rubric_v5.RUBRIC_VERSION
                or payload["reference_spans"].get("protocol_schema") != rubric_v5.PROTOCOL_SCHEMA_V5):
            raise ValueError("v7 case or blinded rubric-v5 reference payload changed")


def validate_packet_integrity(packet: dict[str, Any], workspace: Path | None = None) -> None:
    """Verify a frozen v7 packet without repeating the historical exposure scan.

    Full source discovery and exposure auditing remain mandatory in
    ``validate_packet`` before a plan is frozen. Runner boundaries use this
    deterministic integrity check and still verify every bound source file.
    """
    workspace = (workspace or repo_root()).resolve()
    if not isinstance(packet, dict) or packet.get("schema") != SCHEMA:
        raise ValueError("unsupported v7 packet")
    body = {key: value for key, value in packet.items() if key != "packet_sha256"}
    if packet.get("packet_sha256") != sha256_obj(body):
        raise ValueError("v7 packet hash mismatch")

    def verify_bound_file(entry: dict[str, Any], field: str) -> None:
        relative = entry.get(field)
        digest = entry.get("sha256" if field == "path" else "license_sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("v7 source manifest entry is malformed")
        path = (workspace / relative).resolve()
        if not path.is_relative_to(workspace) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError("v7 source manifest file identity changed")

    legacy_preflight_identity = (
        packet.get("packet_sha256") == LEGACY_PREFLIGHT_PACKET_SHA256
        and packet.get("builder_source_sha256") == LEGACY_PREFLIGHT_BUILDER_SHA256)
    if ((packet.get("builder_source_sha256") != sha256_file(Path(__file__))
            and not legacy_preflight_identity)
            or packet.get("rubric_source_sha256") != sha256_file(Path(rubric_v5.__file__))):
        raise ValueError("v7 packet builder or rubric source changed")
    manifest = packet.get("source_manifest")
    expected_sources = {"paperqa_context_prompts", "gepa_json_task_prompt",
                        "gepa_terminal_task_prompt", "scholarqa_citation_correctness"}
    if not isinstance(manifest, dict) or set(manifest) != expected_sources:
        raise ValueError("v7 packet source manifest changed")
    for entry in manifest.values():
        if not isinstance(entry, dict):
            raise ValueError("v7 source manifest entry is malformed")
        verify_bound_file(entry, "path")
        verify_bound_file(entry, "license_path")

    protocol = packet.get("reference_protocol", {})
    design = packet.get("construction_protocol", {})
    cases = _build_cases()
    expected_design = {
        "version": "fresh-clinical-analysis-four-cell-v7",
        "vendor_patterns": [
            "PaperQA supplied-context evidence boundary (Apache-2.0)",
            "GEPA strict structured-output discipline (MIT)",
            "ScholarQABench source-bound evidence retrieval (MIT)",
        ],
        "cells": list(_CELLS),
        "cell_counts": {cell: 10 for cell in _CELLS},
        "cells_are_labels": False,
        "expected_labels_present": False,
        "target_predictions_present": False,
        "blind_reference_payload_fields": [
            "task_contract", "candidate_strategy", "reference_spans"],
    }
    if (packet.get("cohort_id") != COHORT_ID
            or packet.get("output_directory_id") != OUTPUT_DIRECTORY_ID
            or packet.get("scope") != "fresh_unexposed_synthetic_clinifact_instruction_reviewer_challenge"
            or packet.get("natural_distribution") is not False
            or packet.get("case_count") != 40
            or packet.get("cases") != cases
            or packet.get("contract") != {
                "id": CLINIFACT_CONTRACT.contract_id,
                "text": CLINIFACT_CONTRACT.text,
                "sha256": CLINIFACT_CONTRACT.sha256,
            }
            or design != expected_design
            or protocol.get("rubric_version") != "v5"
            or protocol.get("protocol_schema") != rubric_v5.PROTOCOL_SCHEMA_V5
            or protocol.get("rubric_sha256") != sha256_obj(rubric_v5.RUBRIC_V5)
            or protocol.get("reference_format_sha256") != sha256_obj(rubric_v5.REFERENCE_FORMAT_V5)
            or protocol.get("splitter_hash") != rubric_v5.splitter_hash()
            or protocol.get("protocol_hash") != rubric_v5.protocol_hash()
            or protocol.get("normalizer_version") != rubric_v5.NORMALIZER_VERSION_V5
            or protocol.get("max_output_tokens") != rubric_v5.REFERENCE_MAX_OUTPUT_TOKENS):
        raise ValueError("v7 frozen packet content or protocol identity changed")
    audit = packet.get("exposure_audit", {})
    if (audit.get("overlap_verified") is not True or audit.get("overlap_count") != 0
            or not isinstance(audit.get("prior_packet_count"), int)
            or audit.get("prior_packet_count", 0) < 0):
        raise ValueError("v7 frozen packet lacks a valid exposure-audit attestation")
    for case in cases:
        payload = reference_payload(case)
        encoded = json.dumps(payload, ensure_ascii=False)
        if (set(payload) != {"task_contract", "candidate_strategy", "reference_spans"}
                or "construction_cell" in encoded or "source_index" in encoded
                or payload["reference_spans"].get("rubric_version") != rubric_v5.RUBRIC_VERSION
                or payload["reference_spans"].get("protocol_schema") != rubric_v5.PROTOCOL_SCHEMA_V5):
            raise ValueError("v7 blind rubric-v5 reference payload changed")


