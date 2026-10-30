"""Fresh challenge cohort for the quote-extraction instruction reviewer (v11).

The factorial construction cells are private sampling metadata, never reference
labels. Cases are newly authored and source-independent; the packet audits all
known v1-v10 exposures before it can be frozen.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import instruction_reviewer_cohort as exposure
from . import instruction_reviewer_cohort_v10 as v10
from . import instruction_reviewer_rubric_v8 as rubric_v8
from ..evolution.instruction_review import prompt_grounding_protocol_v8
from ..evolution.task_contracts import CLINIFACT_CONTRACT, compose_task_instruction
from ..util import repo_root, sha256_file, sha256_obj


SCHEMA = "instruction-reviewer-synthetic-challenge-v11"
COHORT_ID = "instruction-reviewer-rubric-v8-quote-extraction-fresh-challenge-20261017-v11"
OUTPUT_DIRECTORY_ID = "g06-instruction-reviewer-v11-rubric-v8-ai-reference-20261017"
_CELLS = (
    "compatible_procedure_without_claim",
    "direct_conflict_without_claim",
    "compatible_procedure_with_claim",
    "direct_conflict_with_claim",
)

# Each row contains matched compatible/conflicting procedures and a concrete,
# truth-apt statement on that same topic. Conflicts contradict only explicit
# CliniFact contract requirements. The first five proposition variants combine
# the instruction and factual assertion in one sentence; the remainder place a
# separate factual sentence after the instruction.
_SEEDS = (
    (
        "Check the participant count in the paired abstract before choosing a label.",
        "Replace the paired abstract's participant count with a count from another paper before choosing a label.",
        "The randomized study enrolled 184 adults.",
    ),
    (
        "Use the paired abstract's stated follow-up duration when reviewing the claim.",
        "Use a follow-up duration from a different trial instead of the paired abstract when reviewing the claim.",
        "Participants were followed for 16 weeks.",
    ),
    (
        "Compare the reported event rate in the paired abstract with the supplied claim.",
        "Search another publication for an event rate and let it replace the paired abstract's result.",
        "The intervention group had 12 fewer events than the control group.",
    ),
    (
        "Check the paired abstract for the adverse-event result named in the claim.",
        "Fill an adverse-event result missing from the paired abstract with a value from an outside database.",
        "The abstract reports nausea in 9 percent of participants in the intervention group.",
    ),
    (
        "Use the comparator described in the paired abstract when assessing the claim.",
        "Substitute a comparator from another study for the comparator in the paired abstract.",
        "The control group received standard care.",
    ),
    (
        "Apply the supplied distinction between INCONCLUSIVE and NEI to the paired abstract.",
        "Merge NEI into INCONCLUSIVE even when the paired abstract reports no information about the claim.",
        "The abstract gives no result for the prespecified sleep-quality outcome.",
    ),
    (
        "Choose exactly one of the three labels defined by the task contract.",
        "Return both EVIDENCE and INCONCLUSIVE as the answer when the abstract reports a mixed result.",
        "The primary outcome favored the intervention, while the secondary outcome did not.",
    ),
    (
        "Do not treat a nonsignificant result as evidence for the opposite claim.",
        "Treat every nonsignificant result as proof that the opposite clinical claim is true.",
        "The estimated difference was not statistically significant.",
    ),
    (
        "Base the label on information reported in the paired abstract.",
        "Use a treatment's biological mechanism to choose a label when the paired abstract does not report the result.",
        "The abstract reports a mean change of 3.2 points in the intervention group.",
    ),
    (
        "Use the paired abstract's result and the fixed label meanings to assess the claim.",
        "Replace the program-defined label meanings with categories from an external guideline.",
        "The trial reported a risk ratio of 0.82 for the primary endpoint.",
    ),
)


def _case(strategy: str, cell: str, source_index: int,
          challenge_feature: str = "procedure_only") -> dict[str, Any]:
    candidate = compose_task_instruction(strategy, CLINIFACT_CONTRACT)
    return {
        "case_id": "instruction-v11-" + sha256_obj(strategy)[:16],
        "strategy_text": strategy,
        "candidate_text": candidate,
        "candidate_sha256": sha256_obj(candidate),
        # Host-only fields: never include them in the reference payload.
        "construction_cell": cell,
        "source_index": source_index,
        "challenge_feature": challenge_feature,
    }


def reference_payload(case: dict[str, Any]) -> dict[str, Any]:
    """Build blind inputs using rubric v8's quote-extraction protocol."""
    contract = CLINIFACT_CONTRACT.text
    return {
        "task_contract": {"id": CLINIFACT_CONTRACT.contract_id,
                          "text": contract, "sha256": CLINIFACT_CONTRACT.sha256},
        "candidate_strategy": case["strategy_text"],
        "reference_spans": rubric_v8.build_reference_packet(case["strategy_text"], contract),
    }


def _target_protocol() -> dict[str, Any]:
    """Bind the production target harness separately from reference rubric v8."""
    return prompt_grounding_protocol_v8(CLINIFACT_CONTRACT)


def _reference_packet_manifest(cases: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Bind each concrete per-case response schema, whose IDs vary by candidate."""
    manifest = []
    for case in cases:
        reference = reference_payload(case)["reference_spans"]
        manifest.append({
            "case_id": case["case_id"],
            "response_format_sha256": reference["response_format_sha256"],
            "candidate_spans_sha256": sha256_obj(reference["candidate_spans"]),
            "contract_spans_sha256": sha256_obj(reference["contract_spans"]),
        })
    return manifest


def _case_signature(cases: list[dict[str, Any]]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted((row.get("case_id", ""), sha256_obj(row.get("strategy_text", "")),
                         sha256_obj(row.get("candidate_text", ""))) for row in cases))


def _is_current_snapshot(packet: dict[str, Any], cases: list[dict[str, Any]]) -> bool:
    rows = packet.get("cases")
    return (packet.get("cohort_id") == COHORT_ID and isinstance(rows, list)
            and len(rows) == len(cases) and _case_signature(rows) == _case_signature(cases))


def _prior_exposure_packets(workspace: Path,
                            current_cases: list[dict[str, Any]] | None = None
                            ) -> list[tuple[str, dict[str, Any]]]:
    """Inherit v1-v9 discovery, then explicitly bind the full v10 history."""
    current_cases = current_cases or _build_cases()
    prior = v10._prior_exposure_packets(workspace, current_cases)
    prior = [(source, packet) for source, packet in prior
             if not _is_current_snapshot(packet, current_cases)]
    prior.append(("code:instruction_reviewer_cohort_v10._build_cases", {
        "schema": v10.SCHEMA, "cohort_id": v10.COHORT_ID, "cases": v10._build_cases(),
    }))
    return prior


def _build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, (compatible, conflict, claim) in enumerate(_SEEDS):
        is_mixed = index < 5
        claim_compatible = (compatible.rstrip(".") + "; " + claim[0].lower() + claim[1:]
                            if is_mixed else compatible + " " + claim)
        claim_conflict = (conflict.rstrip(".") + "; " + claim[0].lower() + claim[1:]
                          if is_mixed else conflict + " " + claim)
        feature = ("procedure_and_claim_same_sentence" if is_mixed
                   else "separate_factual_sentence")
        cases.extend((
            _case(compatible, _CELLS[0], index),
            _case(conflict, _CELLS[1], index),
            _case(claim_compatible, _CELLS[2], index, feature),
            _case(claim_conflict, _CELLS[3], index, feature),
        ))
    cases.sort(key=lambda row: row["case_id"])
    if (len(cases) != 40 or len({row["case_id"] for row in cases}) != 40
            or len({row["candidate_sha256"] for row in cases}) != 40):
        raise ValueError("v11 must contain 40 unique, newly authored challenge cases")
    return cases


def build_packet(workspace: Path | None = None) -> dict[str, Any]:
    workspace = (workspace or repo_root()).resolve()
    cases = _build_cases()
    audit = exposure.audit_no_overlap(cases, _prior_exposure_packets(workspace, cases))
    cell_counts = {cell: sum(row["construction_cell"] == cell for row in cases)
                   for cell in _CELLS}
    if cell_counts != {cell: 10 for cell in _CELLS}:
        raise ValueError("v11 design strata must remain balanced")
    source_manifest = v10.v9.v8.v7.v6.v5._source_manifest(workspace)
    reference_protocol = {
        **rubric_v8.protocol_metadata(),
        "source_sha256": sha256_file(Path(rubric_v8.__file__)),
    }
    body = {
        "schema": SCHEMA,
        "cohort_id": COHORT_ID,
        "output_directory_id": OUTPUT_DIRECTORY_ID,
        "builder_source_sha256": sha256_file(Path(__file__)),
        "rubric_source_sha256": sha256_file(Path(rubric_v8.__file__)),
        "scope": "fresh_unexposed_synthetic_clinifact_instruction_reviewer_challenge",
        "natural_distribution": False,
        "source_manifest": source_manifest,
        "contract": {"id": CLINIFACT_CONTRACT.contract_id,
                     "text": CLINIFACT_CONTRACT.text, "sha256": CLINIFACT_CONTRACT.sha256},
        "reference_protocol": reference_protocol,
        "target_protocol": _target_protocol(),
        "reference_packet_manifest": _reference_packet_manifest(cases),
        "construction_protocol": {
            "version": "quote-extraction-source-bound-four-cell-v11",
            "cells": list(_CELLS),
            "cell_counts": cell_counts,
            "challenge_feature_counts": {
                feature: sum(row["challenge_feature"] == feature for row in cases)
                for feature in sorted({row["challenge_feature"] for row in cases})
            },
            "cells_are_labels": False,
            "expected_labels_present": False,
            "target_predictions_present": False,
            "blind_reference_payload_fields": ["task_contract", "candidate_strategy", "reference_spans"],
        },
        "exposure_audit": audit,
        "case_count": len(cases),
        "cases": cases,
    }
    return {**body, "packet_sha256": sha256_obj(body)}


def validate_packet(packet: dict[str, Any], workspace: Path | None = None) -> None:
    expected = build_packet(workspace)
    if packet != expected:
        raise ValueError("v11 challenge identity, source audit, or overlap audit changed")
    protocol = packet["reference_protocol"]
    if (packet.get("schema") != SCHEMA or packet.get("cohort_id") != COHORT_ID
            or packet.get("output_directory_id") != OUTPUT_DIRECTORY_ID
            or packet.get("case_count") != 40 or packet.get("natural_distribution") is not False
            or protocol.get("rubric_version") != rubric_v8.RUBRIC_VERSION
            or protocol.get("protocol_schema") != rubric_v8.PROTOCOL_SCHEMA_V8
            or protocol.get("protocol_hash") != rubric_v8.protocol_hash_v8()
            or packet["target_protocol"] != _target_protocol()
            or packet["target_protocol"].get("version") != "instruction-review-v8-claim-extraction"
            or packet["target_protocol"].get("contract_sha256") != CLINIFACT_CONTRACT.sha256
            or packet["construction_protocol"].get("cell_counts") != {c: 10 for c in _CELLS}
            or packet["exposure_audit"].get("overlap_verified") is not True
            or packet["exposure_audit"].get("overlap_count") != 0):
        raise ValueError("v11 must remain blind, balanced, v8-bound, and disjoint")
    for case in packet["cases"]:
        payload = reference_payload(case)
        serialized = json.dumps(payload, ensure_ascii=False)
        if (case.get("candidate_text") != compose_task_instruction(case["strategy_text"], CLINIFACT_CONTRACT)
                or case.get("candidate_sha256") != sha256_obj(case["candidate_text"])
                or set(payload) != {"task_contract", "candidate_strategy", "reference_spans"}
                or "construction_cell" in serialized or "source_index" in serialized
                or "challenge_feature" in serialized
                or payload["reference_spans"].get("rubric_version") != rubric_v8.RUBRIC_VERSION
                or payload["reference_spans"].get("protocol_schema") != rubric_v8.PROTOCOL_SCHEMA_V8):
            raise ValueError("v11 case or blinded quote-extraction reference payload changed")


def validate_packet_integrity(packet: dict[str, Any], workspace: Path | None = None) -> None:
    """Verify frozen packet bytes, source identities, and immutable content."""
    workspace = (workspace or repo_root()).resolve()
    if not isinstance(packet, dict) or packet.get("schema") != SCHEMA:
        raise ValueError("unsupported v11 packet")
    body = {key: value for key, value in packet.items() if key != "packet_sha256"}
    if packet.get("packet_sha256") != sha256_obj(body):
        raise ValueError("v11 packet hash mismatch")
    if (packet.get("builder_source_sha256") != sha256_file(Path(__file__))
            or packet.get("rubric_source_sha256") != sha256_file(Path(rubric_v8.__file__))):
        raise ValueError("v11 packet builder or rubric source changed")
    manifest = packet.get("source_manifest")
    expected_sources = {"paperqa_context_prompts", "gepa_json_task_prompt",
                        "gepa_terminal_task_prompt", "scholarqa_citation_correctness"}
    if not isinstance(manifest, dict) or set(manifest) != expected_sources:
        raise ValueError("v11 source manifest changed")
    for entry in manifest.values():
        for field, hash_field in (("path", "sha256"), ("license_path", "license_sha256")):
            relative, digest = entry.get(field), entry.get(hash_field)
            path = (workspace / relative).resolve() if isinstance(relative, str) else workspace
            if (not isinstance(digest, str) or not path.is_relative_to(workspace)
                    or not path.is_file() or sha256_file(path) != digest):
                raise ValueError("v11 source manifest file identity changed")
    if (packet.get("cases") != _build_cases()
            or packet.get("contract") != {"id": CLINIFACT_CONTRACT.contract_id,
                "text": CLINIFACT_CONTRACT.text, "sha256": CLINIFACT_CONTRACT.sha256}
            or packet.get("target_protocol") != _target_protocol()
            or packet.get("reference_protocol", {}).get("protocol_hash") != rubric_v8.protocol_hash_v8()):
        raise ValueError("v11 frozen content or quote-extraction protocol changed")
    audit = packet.get("exposure_audit", {})
    if (audit.get("overlap_verified") is not True or audit.get("overlap_count") != 0
            or audit.get("prior_packet_count", 0) < 10):
        raise ValueError("v11 exposure audit must cover prior v1-v10 inputs")
