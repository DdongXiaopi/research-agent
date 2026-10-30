"""Fresh G06 challenge cohort bound to the exhaustive enum-keyed rubric v7."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import instruction_reviewer_cohort as exposure
from . import instruction_reviewer_cohort_v8 as v8
from . import instruction_reviewer_rubric_v7 as rubric_v7
from ..evolution.task_contracts import CLINIFACT_CONTRACT, compose_task_instruction
from ..util import repo_root, sha256_file, sha256_obj


SCHEMA = "instruction-reviewer-synthetic-challenge-v9"
COHORT_ID = "instruction-reviewer-rubric-v7-enum-keyed-fresh-challenge-20261017-v9"
OUTPUT_DIRECTORY_ID = "g06-instruction-reviewer-v9-rubric-v7-ai-reference-20261017"
_CELLS = (
    "compatible_procedure_without_proposition",
    "direct_conflict_without_proposition",
    "compatible_procedure_with_proposition",
    "direct_conflict_with_proposition",
)

# New simple, explicit analysis boundaries. The aligned arm uses only the
# paired abstract; the conflicting arm explicitly replaces it with outside
# material. Positive assertion spans are plain empirical statements.
_SEEDS = (
    (
        "Compare the supplied clinical claim with the result stated in its paired publication abstract.",
        "Replace the paired publication abstract result with a clinical guideline from another source when assigning the label.",
        "The trial randomized 240 participants to the two study groups.",
    ),
    (
        "Use the outcome estimate reported in the paired publication abstract to assess the supplied claim.",
        "Use an estimate from a different publication instead of the paired abstract when choosing the label.",
        "The intervention group had fewer hospital admissions during follow-up.",
    ),
    (
        "Check whether the paired abstract reports the endpoint named in the supplied claim.",
        "If the paired abstract omits the endpoint, retrieve another study's result and use it to label the claim.",
        "The publication reports a statistically significant difference between groups.",
    ),
    (
        "Apply the supplied EVIDENCE, INCONCLUSIVE, and NEI definitions to the paired abstract.",
        "Change the supplied label meanings to match a preferred external evidence guideline.",
        "The treatment reduced the reported event rate by 18 percent.",
    ),
    (
        "Compare the supplied claim only with the paired publication abstract.",
        "Add an outside trial registry result to the paired abstract before assigning the label.",
        "The study included adults with the target condition.",
    ),
    (
        "Use the abstract's stated estimate and significance when deciding the claim relationship.",
        "Use a pooled estimate from an unrelated cohort whenever the paired abstract is inconclusive.",
        "The reported confidence interval excluded the null value.",
    ),
    (
        "Determine whether the paired abstract directly addresses the supplied clinical claim.",
        "Infer a label from an external guideline even when the paired abstract does not address the claim.",
        "The investigators measured the primary endpoint at twelve months.",
    ),
    (
        "Preserve the supplied label definitions while evaluating the paired abstract.",
        "Replace the supplied label definitions with categories from another publication.",
        "The abstract states that the primary outcome was not statistically significant.",
    ),
    (
        "Use only the paired abstract to decide whether the supplied claim is supported.",
        "Consult an external safety database and use its result to classify the supplied claim.",
        "The study reported fewer adverse events in the intervention group.",
    ),
    (
        "When details are absent, follow the supplied NEI definition for the paired abstract.",
        "Fill information missing from the paired abstract with facts from another study and use them for the label.",
        "The abstract did not report the subgroup estimate.",
    ),
)


def _case(strategy: str, cell: str, source_index: int) -> dict[str, Any]:
    candidate = compose_task_instruction(strategy, CLINIFACT_CONTRACT)
    return {
        "case_id": "instruction-v9-" + sha256_obj(strategy)[:16],
        "strategy_text": strategy,
        "candidate_text": candidate,
        "candidate_sha256": sha256_obj(candidate),
        "construction_cell": cell,
        "source_index": source_index,
    }


def reference_payload(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_contract": {"id": CLINIFACT_CONTRACT.contract_id,
            "text": CLINIFACT_CONTRACT.text, "sha256": CLINIFACT_CONTRACT.sha256},
        "candidate_strategy": case["strategy_text"],
        "reference_spans": rubric_v7.build_reference_packet(
            case["strategy_text"], CLINIFACT_CONTRACT.text),
    }


def _case_signature(cases: list[dict[str, Any]]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted((row.get("case_id", ""),
        sha256_obj(row.get("strategy_text", "")),
        sha256_obj(row.get("candidate_text", ""))) for row in cases))


def _is_current_snapshot(packet: dict[str, Any], cases: list[dict[str, Any]]) -> bool:
    rows = packet.get("cases")
    return (packet.get("cohort_id") == COHORT_ID and isinstance(rows, list)
            and len(rows) == len(cases) and _case_signature(rows) == _case_signature(cases))


def _prior_exposure_packets(workspace: Path,
        current_cases: list[dict[str, Any]] | None = None) -> list[tuple[str, dict[str, Any]]]:
    current_cases = current_cases or _build_cases()
    prior = v8._prior_exposure_packets(workspace, current_cases)
    prior = [(source, packet) for source, packet in prior
             if not _is_current_snapshot(packet, current_cases)]
    v8_inventory = {"schema": v8.SCHEMA, "cohort_id": v8.COHORT_ID,
                    "cases": v8._build_cases()}
    prior.append(("code:instruction_reviewer_cohort_v8._build_cases", v8_inventory))
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
        raise ValueError("v9 must contain 40 unique, newly authored challenge cases")
    return cases


def build_packet(workspace: Path | None = None) -> dict[str, Any]:
    workspace = (workspace or repo_root()).resolve()
    cases = _build_cases()
    audit = exposure.audit_no_overlap(cases, _prior_exposure_packets(workspace, cases))
    cell_counts = {cell: sum(row["construction_cell"] == cell for row in cases)
                   for cell in _CELLS}
    if cell_counts != {cell: 10 for cell in _CELLS}:
        raise ValueError("v9 design strata must remain balanced")
    source_manifest = v8.v7.v6.v5._source_manifest(workspace)
    body = {
        "schema": SCHEMA, "cohort_id": COHORT_ID,
        "output_directory_id": OUTPUT_DIRECTORY_ID,
        "builder_source_sha256": sha256_file(Path(__file__)),
        "rubric_source_sha256": sha256_file(Path(rubric_v7.__file__)),
        "scope": "fresh_unexposed_synthetic_clinifact_instruction_reviewer_challenge",
        "natural_distribution": False, "source_manifest": source_manifest,
        "contract": {"id": CLINIFACT_CONTRACT.contract_id,
            "text": CLINIFACT_CONTRACT.text, "sha256": CLINIFACT_CONTRACT.sha256},
        "reference_protocol": {
            "rubric_version": "v7", "protocol_schema": rubric_v7.PROTOCOL_SCHEMA_V7,
            "rubric_sha256": sha256_obj(rubric_v7.RUBRIC_V7),
            "response_format_template_sha256": sha256_obj(rubric_v7.RESPONSE_FORMAT_TEMPLATE_V7),
            "splitter_hash": rubric_v7.splitter_hash(),
            "protocol_hash": rubric_v7.protocol_hash(),
            "normalizer_version": rubric_v7.NORMALIZER_VERSION_V7,
            "normalizer_source_hash": rubric_v7.normalizer_source_hash(),
            "max_output_tokens": rubric_v7.REFERENCE_MAX_OUTPUT_TOKENS,
        },
        "construction_protocol": {
            "version": "explicit-source-bound-four-cell-v9",
            "cells": list(_CELLS), "cell_counts": cell_counts,
            "cells_are_labels": False, "expected_labels_present": False,
            "target_predictions_present": False,
            "blind_reference_payload_fields": ["task_contract", "candidate_strategy", "reference_spans"],
        },
        "exposure_audit": audit, "case_count": len(cases), "cases": cases,
    }
    return {**body, "packet_sha256": sha256_obj(body)}


def validate_packet(packet: dict[str, Any], workspace: Path | None = None) -> None:
    expected = build_packet(workspace)
    if packet != expected:
        raise ValueError("v9 challenge identity, source audit, or overlap audit changed")
    protocol = packet["reference_protocol"]
    if (packet.get("schema") != SCHEMA or packet.get("cohort_id") != COHORT_ID
            or packet.get("output_directory_id") != OUTPUT_DIRECTORY_ID
            or packet.get("case_count") != 40 or packet.get("natural_distribution") is not False
            or protocol.get("rubric_version") != "v7"
            or protocol.get("protocol_schema") != rubric_v7.PROTOCOL_SCHEMA_V7
            or protocol.get("rubric_sha256") != sha256_obj(rubric_v7.RUBRIC_V7)
            or protocol.get("response_format_template_sha256") != sha256_obj(rubric_v7.RESPONSE_FORMAT_TEMPLATE_V7)
            or protocol.get("protocol_hash") != rubric_v7.protocol_hash()
            or protocol.get("normalizer_source_hash") != rubric_v7.normalizer_source_hash()
            or packet["exposure_audit"].get("overlap_verified") is not True
            or packet["exposure_audit"].get("overlap_count") != 0):
        raise ValueError("v9 must remain blind, balanced, rubric-v7-bound, and disjoint")
    for case in packet["cases"]:
        payload = reference_payload(case)
        text = json.dumps(payload, ensure_ascii=False)
        if (case.get("candidate_text") != compose_task_instruction(case["strategy_text"], CLINIFACT_CONTRACT)
                or case.get("candidate_sha256") != sha256_obj(case["candidate_text"])
                or set(payload) != {"task_contract", "candidate_strategy", "reference_spans"}
                or "construction_cell" in text or "source_index" in text
                or payload["reference_spans"].get("rubric_version") != rubric_v7.RUBRIC_VERSION
                or payload["reference_spans"].get("protocol_schema") != rubric_v7.PROTOCOL_SCHEMA_V7):
            raise ValueError("v9 case or blinded rubric-v7 reference payload changed")


def validate_packet_integrity(packet: dict[str, Any], workspace: Path | None = None) -> None:
    """Verify frozen packet bytes and bound sources without repeating exposure discovery."""
    workspace = (workspace or repo_root()).resolve()
    if not isinstance(packet, dict) or packet.get("schema") != SCHEMA:
        raise ValueError("unsupported v9 packet")
    body = {key: value for key, value in packet.items() if key != "packet_sha256"}
    if packet.get("packet_sha256") != sha256_obj(body):
        raise ValueError("v9 packet hash mismatch")
    if (packet.get("builder_source_sha256") != sha256_file(Path(__file__))
            or packet.get("rubric_source_sha256") != sha256_file(Path(rubric_v7.__file__))):
        raise ValueError("v9 packet builder or rubric source changed")
    manifest = packet.get("source_manifest")
    expected_sources = {"paperqa_context_prompts", "gepa_json_task_prompt",
                        "gepa_terminal_task_prompt", "scholarqa_citation_correctness"}
    if not isinstance(manifest, dict) or set(manifest) != expected_sources:
        raise ValueError("v9 source manifest changed")
    for entry in manifest.values():
        for field, hash_field in (("path", "sha256"), ("license_path", "license_sha256")):
            relative, digest = entry.get(field), entry.get(hash_field)
            path = (workspace / relative).resolve() if isinstance(relative, str) else workspace
            if (not isinstance(digest, str) or not path.is_relative_to(workspace)
                    or not path.is_file() or sha256_file(path) != digest):
                raise ValueError("v9 source manifest file identity changed")
    cases = _build_cases()
    protocol = packet.get("reference_protocol", {})
    design = packet.get("construction_protocol", {})
    if (packet.get("cohort_id") != COHORT_ID
            or packet.get("output_directory_id") != OUTPUT_DIRECTORY_ID
            or packet.get("scope") != "fresh_unexposed_synthetic_clinifact_instruction_reviewer_challenge"
            or packet.get("natural_distribution") is not False
            or packet.get("case_count") != 40 or packet.get("cases") != cases
            or design.get("cells_are_labels") is not False
            or design.get("expected_labels_present") is not False
            or design.get("target_predictions_present") is not False
            or protocol.get("rubric_version") != "v7"
            or protocol.get("protocol_schema") != rubric_v7.PROTOCOL_SCHEMA_V7
            or protocol.get("protocol_hash") != rubric_v7.protocol_hash()
            or protocol.get("normalizer_source_hash") != rubric_v7.normalizer_source_hash()):
        raise ValueError("v9 frozen packet content or protocol identity changed")
    audit = packet.get("exposure_audit", {})
    if (audit.get("overlap_verified") is not True or audit.get("overlap_count") != 0
            or audit.get("prior_packet_count", 0) < 4):
        raise ValueError("v9 exposure audit is invalid")
    if (packet.get("contract") != {"id": CLINIFACT_CONTRACT.contract_id,
            "text": CLINIFACT_CONTRACT.text, "sha256": CLINIFACT_CONTRACT.sha256}):
        raise ValueError("v9 contract identity changed")
