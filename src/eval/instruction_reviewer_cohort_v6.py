"""Fresh unseen G06 challenge cohort for rubric-v4 reviewer diagnosis.

Construction cells are host-side sampling strata only. They are never emitted
to the blind reference payload and are not treated as expected labels.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import instruction_reviewer_cohort as exposure
from . import instruction_reviewer_cohort_v5 as v5
from . import instruction_reviewer_rubric_v4 as rubric_v4
from ..evolution.task_contracts import CLINIFACT_CONTRACT, compose_task_instruction
from ..util import repo_root, sha256_file, sha256_obj


SCHEMA = "instruction-reviewer-synthetic-challenge-v6"
COHORT_ID = "instruction-reviewer-rubric-v4-fresh-challenge-20261017-v6"
OUTPUT_DIRECTORY_ID = "g06-instruction-reviewer-v6-ai-reference-20261017"
_CELLS = (
    "procedure_contract_following",
    "procedure_contract_conflicting",
    "procedure_with_proposition_contract_following",
    "procedure_with_proposition_contract_conflicting",
)

# Each tuple is a newly authored task-specific procedure for a distinct clinical
# comparison dimension plus a factual proposition used only to create a balanced
# sampling design. The proposition is not a gold label and is not sent separately.
_SEEDS = (
    (
        "For participants who switch treatment after randomization, classify the claim using only the results reported in the supplied abstract.",
        "For participants who switch treatment after randomization, use a separate trial registry to replace the abstract's reported group assignment.",
        "A participant's assigned arm remains fixed in an intention-to-treat analysis after treatment discontinuation.",
    ),
    (
        "For withdrawals during follow-up, use the disposition counts stated in the supplied study summary and preserve its answer categories.",
        "For withdrawals during follow-up, import missing participant outcomes from another publication and substitute them for the supplied summary.",
        "Loss to follow-up can change the population represented by a reported analysis.",
    ),
    (
        "For a crossover design, judge the clinical claim from the sequence-specific findings described in the provided report.",
        "For a crossover design, infer the treatment effect from a drug's known mechanism when the provided report omits sequence findings.",
        "Residual treatment effects can persist into a later period of a crossover study.",
    ),
    (
        "For studies with several primary outcomes, use the result tied to the outcome named by the clinical claim.",
        "For studies with several primary outcomes, replace the named outcome with any secondary result that supports the claim.",
        "Testing multiple outcomes can increase the chance of observing at least one small p-value.",
    ),
    (
        "For a composite endpoint, inspect the outcome definition and reported finding in the supplied abstract before deciding.",
        "For a composite endpoint, decide from an external guideline which component should control the label even if the abstract reports another result.",
        "A composite endpoint combines more than one component event into a single measure.",
    ),
    (
        "For a noninferiority claim, apply only the evidence and decision labels available in the attached study abstract.",
        "For a noninferiority claim, borrow a margin from a different drug study and use it to change the attached abstract's result.",
        "A noninferiority conclusion depends on a margin specified for the comparison.",
    ),
    (
        "For cluster-assigned trials, use the group-level results actually described in the supplied abstract.",
        "For cluster-assigned trials, convert group-level findings into individual patient conclusions using outside epidemiology sources.",
        "Participants within one randomized cluster may share exposures or outcomes.",
    ),
    (
        "For reported safety events, follow the definitions and event counts given in the study summary.",
        "For reported safety events, add events from a product label and let those outside counts determine the task answer.",
        "Safety-event rates depend on both an event count and a defined observation population.",
    ),
    (
        "For an interim analysis, assess the claim from the findings and decision labels in the supplied report.",
        "For an interim analysis, apply a stopping boundary found in an external protocol even when the supplied report uses another rule.",
        "Repeated interim looks can affect the false-positive rate if a design does not account for them.",
    ),
    (
        "For adherence analyses, distinguish the reported assigned-group result from any separately described analysis.",
        "For adherence analyses, discard the assigned-group result and replace the task label with an outside per-protocol interpretation.",
        "An analysis restricted by adherence can compare groups that differ from the original randomized groups.",
    ),
)


def _case(strategy: str, cell: str, source_index: int) -> dict[str, Any]:
    candidate = compose_task_instruction(strategy, CLINIFACT_CONTRACT)
    return {
        "case_id": "instruction-v6-" + sha256_obj(strategy)[:16],
        "strategy_text": strategy,
        "candidate_text": candidate,
        "candidate_sha256": sha256_obj(candidate),
        # This is sampling metadata, intentionally excluded from the blind view.
        "construction_cell": cell,
        "source_index": source_index,
    }


def reference_payload(case: dict[str, Any]) -> dict[str, Any]:
    """Return only candidate and contract source spans for blind AI review."""
    return {
        "task_contract": {
            "id": CLINIFACT_CONTRACT.contract_id,
            "text": CLINIFACT_CONTRACT.text,
            "sha256": CLINIFACT_CONTRACT.sha256,
        },
        "candidate_strategy": case["strategy_text"],
        "reference_spans": rubric_v4.spans.build_reference_packet(
            case["strategy_text"], CLINIFACT_CONTRACT.text),
    }


def _case_signature(cases: list[dict[str, Any]]) -> tuple[tuple[str, str, str], ...]:
    return tuple(sorted((row.get("case_id", ""),
        sha256_obj(row.get("strategy_text", "")),
        sha256_obj(row.get("candidate_text", ""))) for row in cases))


def _is_current_snapshot(packet: dict[str, Any],
                         current_cases: list[dict[str, Any]]) -> bool:
    """Exclude only the exact same cohort snapshot, never a changed same-ID packet."""
    rows = packet.get("cases")
    return (packet.get("cohort_id") == COHORT_ID
            and isinstance(rows, list) and len(rows) == len(current_cases)
            and _case_signature(rows) == _case_signature(current_cases))


def _prior_exposure_packets(workspace: Path,
        current_cases: list[dict[str, Any]] | None = None
        ) -> list[tuple[str, dict[str, Any]]]:
    """Collect the historical audit set and always include the complete v5 cohort."""
    prior = v5._prior_exposure_packets(workspace)
    current_cases = current_cases or _build_cases()
    # Rebuilding/validating an already frozen v6 packet must not treat that
    # exact packet as its own historical exposure. Changed packets with the
    # same cohort ID remain in the audit and will fail overlap checks.
    prior = [(source, packet) for source, packet in prior
             if not _is_current_snapshot(packet, current_cases)]
    v5_packet = v5.build_packet(workspace)
    if not any(packet.get("cohort_id") == v5_packet.get("cohort_id")
               or packet.get("packet_sha256") == v5_packet.get("packet_sha256")
               for _source, packet in prior):
        prior.append(("code:instruction_reviewer_cohort_v5.build_packet", v5_packet))
    # Some older audited packets use filenames outside the v5 search pattern.
    # Search all known instruction packet shapes without loading unrelated JSON.
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
    cases = []
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
        raise ValueError("v6 must contain 40 unique, freshly authored cases")
    return cases


def build_packet(workspace: Path | None = None) -> dict[str, Any]:
    workspace = (workspace or repo_root()).resolve()
    cases = _build_cases()
    prior_packets = _prior_exposure_packets(workspace, cases)
    exposure_audit = exposure.audit_no_overlap(cases, prior_packets)
    cell_counts = {cell: sum(row["construction_cell"] == cell for row in cases)
                   for cell in _CELLS}
    if cell_counts != {cell: 10 for cell in _CELLS}:
        raise ValueError("v6 construction design must remain balanced across four cells")
    source_manifest = v5._source_manifest(workspace)
    body = {
        "schema": SCHEMA,
        "cohort_id": COHORT_ID,
        "builder_source_sha256": sha256_file(Path(__file__)),
        "preparation_cli_source_sha256": sha256_file(
            workspace / "scripts" / "prepare_instruction_reviewer_cohort_v6.py"),
        "rubric_source_sha256": sha256_file(Path(rubric_v4.__file__)),
        "scope": "fresh_unexposed_synthetic_clinifact_instruction_reviewer_challenge",
        "natural_distribution": False,
        "source_manifest": source_manifest,
        "contract": {
            "id": CLINIFACT_CONTRACT.contract_id,
            "text": CLINIFACT_CONTRACT.text,
            "sha256": CLINIFACT_CONTRACT.sha256,
        },
        "reference_protocol": {
            "rubric_version": "v4",
            "rubric_sha256": sha256_obj(rubric_v4.RUBRIC_V4),
            "reference_format_sha256": sha256_obj(rubric_v4.REFERENCE_FORMAT_V4),
            "splitter_hash": rubric_v4.splitter_hash(),
            "protocol_hash": rubric_v4.protocol_hash(),
            "normalizer_version": "instruction-reference-normalizer-v4",
            "max_output_tokens": 512,
        },
        "construction_protocol": {
            "version": "fresh-clinical-decision-four-cell-v6",
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
        "exposure_audit": exposure_audit,
        "case_count": len(cases),
        "cases": cases,
    }
    return {**body, "packet_sha256": sha256_obj(body)}


def validate_packet(packet: dict[str, Any], workspace: Path | None = None) -> None:
    expected = build_packet(workspace)
    if packet != expected:
        raise ValueError("v6 challenge identity, source audit, or overlap audit changed")
    protocol = packet["reference_protocol"]
    cells = packet["construction_protocol"]
    if (packet.get("schema") != SCHEMA or packet.get("case_count") != 40
            or packet.get("natural_distribution") is not False
            or protocol.get("rubric_version") != "v4"
            or protocol.get("rubric_sha256") != sha256_obj(rubric_v4.RUBRIC_V4)
            or protocol.get("reference_format_sha256") != sha256_obj(rubric_v4.REFERENCE_FORMAT_V4)
            or protocol.get("splitter_hash") != rubric_v4.splitter_hash()
            or protocol.get("protocol_hash") != rubric_v4.protocol_hash()
            or protocol.get("normalizer_version") != "instruction-reference-normalizer-v4"
            or protocol.get("max_output_tokens") != 512
            or cells.get("cells_are_labels") is not False
            or cells.get("expected_labels_present") is not False
            or cells.get("target_predictions_present") is not False
            or packet["exposure_audit"].get("overlap_verified") is not True
            or packet["exposure_audit"].get("overlap_count") != 0):
        raise ValueError("v6 must remain blind, balanced, rubric-v4-bound, and disjoint")
    counts = {cell: sum(row.get("construction_cell") == cell for row in packet["cases"])
              for cell in _CELLS}
    if counts != {cell: 10 for cell in _CELLS}:
        raise ValueError("v6 construction strata changed or are imbalanced")
    for case in packet["cases"]:
        if (case.get("candidate_text") != compose_task_instruction(
                case["strategy_text"], CLINIFACT_CONTRACT)
                or case.get("candidate_sha256") != sha256_obj(case["candidate_text"])
                or set(reference_payload(case)) != {
                    "task_contract", "candidate_strategy", "reference_spans"}
                or "construction_cell" in json.dumps(reference_payload(case))):
            raise ValueError("v6 case or blinded reference payload changed")
