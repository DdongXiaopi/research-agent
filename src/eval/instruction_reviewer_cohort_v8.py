"""Fresh unseen G06 challenge cohort bound to exhaustive evidence-only rubric-v6.

Balanced construction strata are host-only sampling metadata. They are never
labels and never appear in the blind reference payload.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import instruction_reviewer_cohort as exposure
from . import instruction_reviewer_cohort_v7 as v7
from . import instruction_reviewer_rubric_v6 as rubric_v6
from ..evolution.task_contracts import CLINIFACT_CONTRACT, compose_task_instruction
from ..util import repo_root, sha256_file, sha256_obj


SCHEMA = "instruction-reviewer-synthetic-challenge-v8"
COHORT_ID = "instruction-reviewer-rubric-v6-exhaustive-fresh-challenge-20261018-v8"
OUTPUT_DIRECTORY_ID = "g06-instruction-reviewer-v8-rubric-v6-ai-reference-20261018-r2"
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
        "For a cluster crossover trial, compare the candidate outcome with the sequence-specific result stated in the paired abstract.",
        "For a cluster crossover trial, discard the paired abstract result and substitute a pooled estimate retrieved from another publication.",
        "In a crossover trial, treatment sequence and period can affect the reported comparison.",
    ),
    (
        "For a safety endpoint, check the event definition and estimate against the supplied study summary.",
        "When the supplied summary omits the event estimate, consult an external safety database and use it to select EVIDENCE or NEI.",
        "Safety estimates depend on how adverse events are defined and collected.",
    ),
    (
        "For a noninferiority result, use the margin and analysis population specified by the paired report.",
        "For a noninferiority result, replace the paired report margin with a margin taken from an outside guideline.",
        "A noninferiority conclusion depends on the prespecified margin and confidence interval.",
    ),
    (
        "For an interim analysis, compare the candidate only with the interim boundary and result described in the supplied abstract.",
        "If the supplied abstract does not report an interim boundary, take one from an external trial registry and use it to determine the task label.",
        "Interim analyses may use prespecified stopping boundaries.",
    ),
    (
        "For a composite endpoint, retain the component definition used in the supplied publication abstract.",
        "If the abstract reports no component breakdown, obtain component results from an external source and use them to label the candidate.",
        "Composite endpoints combine multiple event types according to a defined rule.",
    ),
    (
        "For a mediation claim, use the adjustment set and estimand defined in the paired paper summary.",
        "If the paper summary does not describe adjustment variables, take a preferred set from another article and use it to assign the label.",
        "Mediation estimates depend on the estimand and assumptions used in the analysis.",
    ),
    (
        "For a diagnostic test claim, use the reference standard and threshold reported in the paired study.",
        "For a diagnostic test claim, replace the paired study threshold and reference standard with values from another publication.",
        "Diagnostic accuracy measures depend on the target population and reference standard.",
    ),
    (
        "For a missing outcome, use only the imputation method and population described in the supplied study report.",
        "For a missing outcome, fill in absent values from an unrelated cohort and use those values to choose the task label.",
        "Missing-data conclusions can depend on the assumed mechanism and imputation model.",
    ),
    (
        "For a subgroup interaction, compare the interaction test reported in the paired abstract with the candidate.",
        "If the paired abstract does not report an interaction test, use a result from a separate study to assign the task label.",
        "A subgroup difference requires evidence of an interaction rather than separate within-group significance.",
    ),
    (
        "For a replication result, use the replication estimate and outcome definition specified in the supplied abstract.",
        "For a replication result, add a result from a different experiment that is absent from the paired abstract.",
        "Replication assessments depend on the original outcome and comparison being reproduced.",
    ),
)


def _case(strategy: str, cell: str, source_index: int) -> dict[str, Any]:
    candidate = compose_task_instruction(strategy, CLINIFACT_CONTRACT)
    return {
        "case_id": "instruction-v8-" + sha256_obj(strategy)[:16],
        "strategy_text": strategy,
        "candidate_text": candidate,
        "candidate_sha256": sha256_obj(candidate),
        "construction_cell": cell,
        "source_index": source_index,
    }


def reference_payload(case: dict[str, Any]) -> dict[str, Any]:
    """Return contract, candidate, and v6 exhaustive source-span reference material only."""
    return {
        "task_contract": {
            "id": CLINIFACT_CONTRACT.contract_id,
            "text": CLINIFACT_CONTRACT.text,
            "sha256": CLINIFACT_CONTRACT.sha256,
        },
        "candidate_strategy": case["strategy_text"],
        "reference_spans": rubric_v6.build_reference_packet(
            case["strategy_text"], CLINIFACT_CONTRACT.text),
    }


def _case_signature(cases: list[dict[str, Any]]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted((row.get("case_id", ""),
        sha256_obj(row.get("strategy_text", "")),
        sha256_obj(row.get("candidate_text", ""))) for row in cases))


def _is_current_snapshot(packet: dict[str, Any],
                         current_cases: list[dict[str, Any]]) -> bool:
    """Exclude only this exact v8 snapshot, never changed content with its ID."""
    rows = packet.get("cases")
    return (packet.get("cohort_id") == COHORT_ID
            and isinstance(rows, list) and len(rows) == len(current_cases)
            and _case_signature(rows) == _case_signature(current_cases))


def _prior_exposure_packets(workspace: Path,
        current_cases: list[dict[str, Any]] | None = None
        ) -> list[tuple[str, dict[str, Any]]]:
    """Carry v1-v7 history forward and explicitly include all v5/v6/v7 cases."""
    current_cases = current_cases or _build_cases()
    prior = v7._prior_exposure_packets(workspace, current_cases)
    # v7 inherits a v5 packet built from a live history scan. Once this v8
    # packet exists, that nested v5 packet's audit hash changes because v5 does
    # not know how to exclude future v8 snapshots. Exposure checks need the
    # historical case inventory, not a recursively history-dependent packet
    # receipt, so replace only that derived receipt with its deterministic
    # source inventory. All on-disk packets remain in the audit below.
    prior = [(source, packet) for source, packet in prior
             if source != "code:instruction_reviewer_cohort_v5.build_packet"]
    v5_cases = []
    for index, (aligned, conflicting, proposition) in enumerate(v7.v6.v5._BASES):
        v5_cases.extend((
            v7.v6.v5._case(aligned, v7.v6.v5._CELLS[0], index),
            v7.v6.v5._case(conflicting, v7.v6.v5._CELLS[1], index),
            v7.v6.v5._case(aligned + " " + proposition, v7.v6.v5._CELLS[2], index),
            v7.v6.v5._case(conflicting + " " + proposition, v7.v6.v5._CELLS[3], index),
        ))
    v5_cases.sort(key=lambda row: row["case_id"])
    v5_inventory = {
        "schema": v7.v6.v5.SCHEMA,
        "cohort_id": v7.v6.v5.COHORT_ID,
        "cases": v5_cases,
    }
    prior.append(("code:instruction_reviewer_cohort_v5._build_cases",
                  v5_inventory))
    # Inherited inventories may contain an earlier frozen v8 run. Only the
    # exact same cohort snapshot is self-history; any changed same-ID content
    # must remain and trigger the ordinary exposure overlap check.
    prior = [(source, packet) for source, packet in prior
             if not _is_current_snapshot(packet, current_cases)]
    # v7's deterministic inventory is included explicitly so the v8 history
    # covers all v5/v6/v7 candidate and strategy text even if older packet
    # discovery rules change.
    v7_inventory = {
        "schema": v7.SCHEMA,
        "cohort_id": v7.COHORT_ID,
        "cases": v7._build_cases(),
    }
    prior.append(("code:instruction_reviewer_cohort_v7._build_cases",
                  v7_inventory))
    # Pick up frozen v5/v6/v7 and older reviewer packets even when their directory
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
        raise ValueError("v8 must contain 40 unique, freshly authored cases")
    return cases


def build_packet(workspace: Path | None = None) -> dict[str, Any]:
    workspace = (workspace or repo_root()).resolve()
    cases = _build_cases()
    prior_packets = _prior_exposure_packets(workspace, cases)
    audit = exposure.audit_no_overlap(cases, prior_packets)
    cell_counts = {cell: sum(row["construction_cell"] == cell for row in cases)
                   for cell in _CELLS}
    if cell_counts != {cell: 10 for cell in _CELLS}:
        raise ValueError("v8 construction strata must remain balanced")
    source_manifest = v7.v6.v5._source_manifest(workspace)
    body = {
        "schema": SCHEMA,
        "cohort_id": COHORT_ID,
        "output_directory_id": OUTPUT_DIRECTORY_ID,
        "builder_source_sha256": sha256_file(Path(__file__)),
        "rubric_source_sha256": sha256_file(Path(rubric_v6.__file__)),
        "scope": "fresh_unexposed_synthetic_clinifact_instruction_reviewer_challenge",
        "natural_distribution": False,
        "source_manifest": source_manifest,
        "contract": {
            "id": CLINIFACT_CONTRACT.contract_id,
            "text": CLINIFACT_CONTRACT.text,
            "sha256": CLINIFACT_CONTRACT.sha256,
        },
        "reference_protocol": {
            "rubric_version": "v6",
            "protocol_schema": rubric_v6.PROTOCOL_SCHEMA_V6,
            "rubric_sha256": sha256_obj(rubric_v6.RUBRIC_V6),
            "reference_format_template_sha256": sha256_obj(rubric_v6.REFERENCE_FORMAT_V6),
            "splitter_hash": rubric_v6.splitter_hash(),
            "protocol_hash": rubric_v6.protocol_hash(),
            "normalizer_version": rubric_v6.NORMALIZER_VERSION_V6,
            "max_output_tokens": rubric_v6.REFERENCE_MAX_OUTPUT_TOKENS,
        },
        "construction_protocol": {
            "version": "fresh-clinical-analysis-four-cell-v8",
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
        raise ValueError("v8 challenge identity, source audit, or overlap audit changed")
    protocol = packet["reference_protocol"]
    design = packet["construction_protocol"]
    if (packet.get("schema") != SCHEMA or packet.get("cohort_id") != COHORT_ID
            or packet.get("output_directory_id") != OUTPUT_DIRECTORY_ID
            or packet.get("case_count") != 40
            or packet.get("natural_distribution") is not False
            or protocol.get("rubric_version") != "v6"
            or protocol.get("protocol_schema") != rubric_v6.PROTOCOL_SCHEMA_V6
            or protocol.get("rubric_sha256") != sha256_obj(rubric_v6.RUBRIC_V6)
            or protocol.get("reference_format_template_sha256") != sha256_obj(rubric_v6.REFERENCE_FORMAT_V6)
            or protocol.get("splitter_hash") != rubric_v6.splitter_hash()
            or protocol.get("protocol_hash") != rubric_v6.protocol_hash()
            or protocol.get("normalizer_version") != rubric_v6.NORMALIZER_VERSION_V6
            or protocol.get("max_output_tokens") != 1024
            or design.get("cells_are_labels") is not False
            or design.get("expected_labels_present") is not False
            or design.get("target_predictions_present") is not False
            or packet["exposure_audit"].get("overlap_verified") is not True
            or packet["exposure_audit"].get("overlap_count") != 0):
        raise ValueError("v8 must remain blind, balanced, rubric-v6-bound, and disjoint")
    counts = {cell: sum(row.get("construction_cell") == cell for row in packet["cases"])
              for cell in _CELLS}
    if counts != {cell: 10 for cell in _CELLS}:
        raise ValueError("v8 construction strata changed or are imbalanced")
    for case in packet["cases"]:
        payload = reference_payload(case)
        encoded = json.dumps(payload, ensure_ascii=False)
        if (case.get("candidate_text") != compose_task_instruction(
                case["strategy_text"], CLINIFACT_CONTRACT)
                or case.get("candidate_sha256") != sha256_obj(case["candidate_text"])
                or set(payload) != {"task_contract", "candidate_strategy", "reference_spans"}
                or "construction_cell" in encoded or "source_index" in encoded
                or payload["reference_spans"].get("rubric_version") != rubric_v6.RUBRIC_VERSION
                or payload["reference_spans"].get("protocol_schema") != rubric_v6.PROTOCOL_SCHEMA_V6):
            raise ValueError("v8 case or blinded rubric-v6 reference payload changed")


def validate_packet_integrity(packet: dict[str, Any], workspace: Path | None = None) -> None:
    """Verify a frozen v7 packet without repeating the historical exposure scan.

    Full source discovery and exposure auditing remain mandatory in
    ``validate_packet`` before a plan is frozen. Runner boundaries use this
    deterministic integrity check and still verify every bound source file.
    """
    workspace = (workspace or repo_root()).resolve()
    if not isinstance(packet, dict) or packet.get("schema") != SCHEMA:
        raise ValueError("unsupported v8 packet")
    body = {key: value for key, value in packet.items() if key != "packet_sha256"}
    if packet.get("packet_sha256") != sha256_obj(body):
        raise ValueError("v8 packet hash mismatch")

    def verify_bound_file(entry: dict[str, Any], field: str) -> None:
        relative = entry.get(field)
        digest = entry.get("sha256" if field == "path" else "license_sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("v8 source manifest entry is malformed")
        path = (workspace / relative).resolve()
        if not path.is_relative_to(workspace) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError("v8 source manifest file identity changed")

    legacy_preflight_identity = (
        packet.get("packet_sha256") == LEGACY_PREFLIGHT_PACKET_SHA256
        and packet.get("builder_source_sha256") == LEGACY_PREFLIGHT_BUILDER_SHA256)
    if ((packet.get("builder_source_sha256") != sha256_file(Path(__file__))
            and not legacy_preflight_identity)
            or packet.get("rubric_source_sha256") != sha256_file(Path(rubric_v6.__file__))):
        raise ValueError("v8 packet builder or rubric source changed")
    manifest = packet.get("source_manifest")
    expected_sources = {"paperqa_context_prompts", "gepa_json_task_prompt",
                        "gepa_terminal_task_prompt", "scholarqa_citation_correctness"}
    if not isinstance(manifest, dict) or set(manifest) != expected_sources:
        raise ValueError("v7 packet source manifest changed")
    for entry in manifest.values():
        if not isinstance(entry, dict):
            raise ValueError("v8 source manifest entry is malformed")
        verify_bound_file(entry, "path")
        verify_bound_file(entry, "license_path")

    protocol = packet.get("reference_protocol", {})
    design = packet.get("construction_protocol", {})
    cases = _build_cases()
    expected_design = {
        "version": "fresh-clinical-analysis-four-cell-v8",
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
            or protocol.get("rubric_version") != "v6"
            or protocol.get("protocol_schema") != rubric_v6.PROTOCOL_SCHEMA_V6
            or protocol.get("rubric_sha256") != sha256_obj(rubric_v6.RUBRIC_V6)
            or protocol.get("reference_format_template_sha256") != sha256_obj(rubric_v6.REFERENCE_FORMAT_V6)
            or protocol.get("splitter_hash") != rubric_v6.splitter_hash()
            or protocol.get("protocol_hash") != rubric_v6.protocol_hash()
            or protocol.get("normalizer_version") != rubric_v6.NORMALIZER_VERSION_V6
            or protocol.get("max_output_tokens") != rubric_v6.REFERENCE_MAX_OUTPUT_TOKENS):
        raise ValueError("v8 frozen packet content or protocol identity changed")
    audit = packet.get("exposure_audit", {})
    if (audit.get("overlap_verified") is not True or audit.get("overlap_count") != 0
            or not isinstance(audit.get("prior_packet_count"), int)
            or audit.get("prior_packet_count", 0) < 0):
        raise ValueError("v8 frozen packet lacks a valid exposure-audit attestation")
    for case in cases:
        payload = reference_payload(case)
        encoded = json.dumps(payload, ensure_ascii=False)
        if (set(payload) != {"task_contract", "candidate_strategy", "reference_spans"}
                or "construction_cell" in encoded or "source_index" in encoded
                or payload["reference_spans"].get("rubric_version") != rubric_v6.RUBRIC_VERSION
                or payload["reference_spans"].get("protocol_schema") != rubric_v6.PROTOCOL_SCHEMA_V6):
            raise ValueError("v7 blind rubric-v5 reference payload changed")


