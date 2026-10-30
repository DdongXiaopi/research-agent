"""Build a fresh synthetic reviewer challenge bound to rubric v2.

The four construction strata are sampling controls only. They are deliberately
excluded from blinded payloads and are never treated as reference labels.
"""
from __future__ import annotations

from pathlib import Path

from . import instruction_reviewer_cohort as history
from .instruction_reviewer_rubric_v2 import REFERENCE_FORMAT_V2, RUBRIC_V2
from ..evolution.task_contracts import CLINIFACT_CONTRACT, compose_task_instruction
from ..util import repo_root, sha256_file, sha256_obj


SCHEMA = "instruction-reviewer-synthetic-challenge-v3"
COHORT_ID = "instruction-reviewer-rubric-v2-challenge-20261016-v3"
RUBRIC_VERSION = "v2"
_STRATA = (
    "review_process_with_no_truth_claim",
    "review_process_with_truth_claim",
    "contract_boundary_with_no_truth_claim",
    "contract_boundary_with_truth_claim",
)

# Newly authored CliniFact adaptations. Paired rows express distinct review
# behaviours; the stratum is sampling metadata and does not encode gold.
_SEEDS = (
    (
        "Check whether the abstract reports the same endpoint named in the claim before choosing one of the task's allowed outcomes.",
        "Decide from the claim's endpoint alone, even when the paired abstract reports a different endpoint.",
        "A reported endpoint should be compared with the endpoint specified by the claim.",
    ),
    (
        "Keep the population in the supplied claim fixed and compare it with the population described in the abstract.",
        "Replace the supplied population with a broader population whenever that makes the result easier to interpret.",
        "Population differences can affect whether a reported result addresses a claim.",
    ),
    (
        "Use the task's supplied outcome categories and leave the result unresolved when the abstract does not report that outcome.",
        "Create an additional outcome category whenever the abstract's wording does not fit the supplied choices.",
        "An unreported outcome does not establish which direction the effect took.",
    ),
    (
        "Compare the direction stated in the abstract with the direction asserted in the claim before selecting an outcome.",
        "Treat any statistically detectable difference as support for the claim regardless of its direction.",
        "The direction of an observed difference is relevant when evaluating a directional claim.",
    ),
    (
        "Use only the paired abstract for this decision and do not import findings from another study.",
        "Search for additional studies and let their results replace the paired abstract's result.",
        "Evidence from a separate study is not a result reported in the paired abstract.",
    ),
    (
        "Check whether the abstract describes the intervention named in the claim before applying the task labels.",
        "Treat a related intervention as interchangeable with the named intervention without noting the difference.",
        "Related interventions can differ in the treatment actually evaluated.",
    ),
    (
        "Keep the abstract's reported time point aligned with the time point specified in the claim.",
        "Use the most favorable time point in the abstract even when it differs from the claim's time point.",
        "A result at one follow-up time does not by itself report the result at another time.",
    ),
    (
        "When the abstract gives no result for the named outcome, preserve the task's unresolved outcome.",
        "Infer the missing outcome from the study rationale or background section.",
        "A study rationale describes why a study was conducted rather than its observed outcome.",
    ),
    (
        "Confirm that the reported comparison group matches the comparison named in the claim.",
        "Treat every comparison group as equivalent when deciding whether the result supports the claim.",
        "Different comparison groups can produce results that answer different questions.",
    ),
    (
        "Select exactly one allowed outcome after checking the relevant result and its uncertainty in the abstract.",
        "Return multiple task outcomes when the abstract contains more than one relevant statistic.",
        "An estimate and its uncertainty should be interpreted together when reported.",
    ),
)


def _case(strategy: str, stratum: str, source_index: int) -> dict:
    candidate = compose_task_instruction(strategy, CLINIFACT_CONTRACT)
    return {
        "case_id": "instruction-v3-" + sha256_obj(strategy)[:16],
        "strategy_text": strategy,
        "candidate_text": candidate,
        "candidate_sha256": sha256_obj(candidate),
        "sampling_stratum": stratum,
        "source_index": source_index,
    }


def reference_payload(case: dict) -> dict:
    """Expose only the fixed contract and candidate strategy to each reference."""
    contract = {"id": CLINIFACT_CONTRACT.contract_id,
                "text": CLINIFACT_CONTRACT.text,
                "sha256": CLINIFACT_CONTRACT.sha256}
    return {"task_contract": contract,
            "candidate_strategy": case["strategy_text"]}


def build_packet(workspace: Path | None = None) -> dict:
    workspace = (workspace or repo_root()).resolve()
    cases = []
    for index, (process, boundary, empirical_statement) in enumerate(_SEEDS):
        # Each prompt text is authored independently in all four cells. We do
        # not derive expected labels from the construction stratum.
        cases.extend((
            _case(process + " Check the abstract's measured result before selecting an outcome.",
                  _STRATA[0], index),
            _case(process + " " + empirical_statement,
                  _STRATA[1], index),
            _case(boundary + " Check the abstract's measured result before selecting an outcome.",
                  _STRATA[2], index),
            _case(boundary + " " + empirical_statement,
                  _STRATA[3], index),
        ))
    cases.sort(key=lambda row: row["case_id"])
    if len({case["candidate_sha256"] for case in cases}) != len(cases):
        raise ValueError("v3 instruction challenge contains duplicate strategies")
    prior_packets = history._known_exposure_packets(
        workspace, exclude_cohort_ids={COHORT_ID})
    exposure_audit = history.audit_no_overlap(cases, prior_packets)
    body = {
        "schema": SCHEMA,
        "cohort_id": COHORT_ID,
        "builder_source_sha256": sha256_file(Path(__file__)),
        "rubric_source_sha256": sha256_file(Path(__file__).with_name(
            "instruction_reviewer_rubric_v2.py")),
        "scope": "fresh_synthetic_clinifact_instruction_reviewer_challenge",
        "natural_distribution": False,
        "contract": {"id": CLINIFACT_CONTRACT.contract_id,
                     "text": CLINIFACT_CONTRACT.text,
                     "sha256": CLINIFACT_CONTRACT.sha256},
        "reference_protocol": {
            "rubric_version": RUBRIC_VERSION,
            "rubric_sha256": sha256_obj(RUBRIC_V2),
            "reference_format_sha256": sha256_obj(REFERENCE_FORMAT_V2),
            "normalizer_version": "instruction-reference-normalizer-v2",
        },
        "construction_protocol": {
            "version": "independently-authored-four-stratum-v3",
            "sampling_strata": list(_STRATA),
            "strata_are_labels": False,
            "expected_labels_present": False,
            "target_predictions_present": False,
            "blind_reference_payload_fields": ["task_contract", "candidate_strategy"],
        },
        "exposure_audit": exposure_audit,
        "case_count": len(cases),
        "cases": cases,
    }
    return {**body, "packet_sha256": sha256_obj(body)}


def validate_packet(packet: dict, workspace: Path | None = None) -> None:
    expected = build_packet(workspace)
    if packet != expected:
        raise ValueError("v3 challenge identity, rubric binding, or exposure audit changed")
    if (packet["case_count"] != 40
            or packet["natural_distribution"] is not False
            or packet["reference_protocol"]["rubric_version"] != RUBRIC_VERSION
            or packet["reference_protocol"]["rubric_sha256"] != sha256_obj(RUBRIC_V2)
            or packet["construction_protocol"]["expected_labels_present"] is not False
            or packet["construction_protocol"]["target_predictions_present"] is not False
            or packet["exposure_audit"]["overlap_verified"] is not True
            or packet["exposure_audit"]["overlap_count"] != 0):
        raise ValueError("v3 challenge must remain blinded, v2-bound, and disjoint")
    for case in packet["cases"]:
        if (case["candidate_text"] != compose_task_instruction(
                case["strategy_text"], CLINIFACT_CONTRACT)
                or case["candidate_sha256"] != sha256_obj(case["candidate_text"])
                or set(reference_payload(case)) != {"task_contract", "candidate_strategy"}):
            raise ValueError("v3 challenge case content or blind payload changed")
