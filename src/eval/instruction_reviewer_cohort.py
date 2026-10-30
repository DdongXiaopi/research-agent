"""Build a licensed, unseen synthetic challenge for instruction-review calibration.

Cases are self-authored adaptations of general prompt-design patterns in the
MIT-licensed GEPA and Apache-2.0 PaperQA snapshots. No benchmark answers or
verbatim vendor prompt blocks are copied. The factorial design is a sampling
plan only; it does not contain reference labels and must never be scored as
gold before the independent AI reference procedure runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .ai_instruction_calibration import THRESHOLDS
from .natural_instruction_calibration import RUBRIC
from ..evolution.instruction_review import (
    PROMPT_GROUNDING_VERSION,
    prompt_grounding_protocol,
)
from ..evolution.task_contracts import CLINIFACT_CONTRACT, compose_task_instruction
from ..util import repo_root, sha256_file, sha256_obj


SCHEMA = "instruction-reviewer-vendor-adapted-cohort-v1"
COHORT_ID = "instruction-reviewer-vendor-adapted-20261016-v2"
_SUPERSEDED_COHORT_IDS = {"instruction-reviewer-vendor-adapted-20261016-v1"}
_LATER_COHORT_IDS = {"instruction-reviewer-rubric-v2-challenge-20261016-v3"}
_SOURCE_FILES = {
    "paperqa_context_prompts": {
        "path": "vendor/paper-qa-main/paper-qa-main/src/paperqa/prompts.py",
        "sha256": "48a504fd09feb608b647b574461f68c2cc4a98cc1a468d7c0e7741f81d199181",
        "license_path": "vendor/paper-qa-main/paper-qa-main/LICENSE",
        "license": "Apache-2.0",
        "license_sha256": "a10f591e3522fbfe867cfde0b35e0594f6f43a51b013540208a1469c9e0b0f62",
        "anchors": ["summary_prompt", "answer_iteration_prompt_template", "qa_prompt",
                    "select_paper_prompt", "citation_prompt"],
    },
    "gepa_json_task_prompt": {
        "path": "vendor/gepa-main/gepa-main/src/gepa/examples/anymaths-bench/prompt-templates/instruction_prompt.txt",
        "sha256": "f182ed0909543bb9b151c48d27b375298317aaccad661999917472115f781274",
        "license_path": "vendor/gepa-main/gepa-main/LICENSE",
        "license": "MIT",
        "license_sha256": "10c47467a961feb40adf3294fe27dd9cba79d4d1b7cf27173b1c34586d4126c3",
        "anchors": ["JSON output contract", "step-by-step solution instruction"],
    },
    "gepa_terminal_task_prompt": {
        "path": "vendor/gepa-main/gepa-main/src/gepa/examples/terminal-bench/prompt-templates/instruction_prompt.txt",
        "sha256": "7b33b73b8a6140fe77c2665af6f67c0474af3bb68ce952b71ce7d67a53e475b1",
        "license_path": "vendor/gepa-main/gepa-main/LICENSE",
        "license": "MIT",
        "license_sha256": "10c47467a961feb40adf3294fe27dd9cba79d4d1b7cf27173b1c34586d4126c3",
        "anchors": ["inspect prior command output", "batch command execution"],
    },
}

# Each seed is newly authored for the CliniFact runtime. The source anchors
# document which open-source design pattern informed the wording; they are not
# copied vendor text and are not reference labels.
_SEEDS = (
    ("paperqa_context_prompts", "context_bound", "Use only the supplied claim and paired abstract. Compare the reported result with the claim and choose one of the task's listed labels.", "Search for other publications and let them override the paired abstract when the sources disagree."),
    ("paperqa_context_prompts", "insufficient_context", "When the abstract does not report information addressing the claim, preserve NEI instead of filling gaps from memory.", "When the abstract does not report the outcome, convert NEI to INCONCLUSIVE."),
    ("paperqa_context_prompts", "evidence_summary", "Identify the result that bears on the claim, then apply the task's fixed label definitions without adding outside facts.", "Treat a highly cited or prestigious journal as stronger evidence than the result reported in the paired abstract."),
    ("paperqa_context_prompts", "current_context", "Base the decision on the claim and abstract present in this task; do not carry evidence forward from an earlier task.", "Reuse a prior answer even when its supporting abstract is absent from the current task."),
    ("paperqa_context_prompts", "source_selection", "Check that the reported population and outcome match the supplied claim before selecting a listed label.", "Search for additional papers and classify the claim from the broader literature rather than the supplied abstract."),
    ("gepa_json_task_prompt", "output_format", "Read the paired abstract and return only the program-constrained option index required by this task.", "Return a JSON object with a free-form explanation instead of the program-constrained option index."),
    ("gepa_json_task_prompt", "stepwise_review", "Check the relevant result in the supplied abstract before choosing exactly one listed label.", "Require at least 500 participants before any result can receive EVIDENCE."),
    ("gepa_terminal_task_prompt", "inspect_material", "Inspect the supplied abstract for the result relevant to the claim and use no external evidence.", "Use external consensus or a treatment mechanism to decide whenever the supplied result is unclear."),
    ("paperqa_context_prompts", "answer_boundary", "Summarize only the evidence needed to compare the claim with the supplied abstract; keep the task labels distinct.", "Merge NEI into INCONCLUSIVE whenever the abstract contains background information."),
    ("paperqa_context_prompts", "citation_boundary", "Do not invent study findings; choose a label only when the supplied abstract supports that decision under the contract.", "Treat any statistically significant result as EVIDENCE even when its direction does not support the claim."),
)

_EMPIRICAL_ASSERTIONS = (
    "All randomized trials eliminate confounding and establish causality.",
    "Every nonsignificant finding proves that an intervention has exactly no effect.",
    "A confidence interval excluding zero always establishes significance for every measure.",
    "Observational studies can never support a causal conclusion.",
    "A statistically significant result proves clinical importance in every population.",
    "Every meta-analysis is unbiased by construction.",
    "Any p-value below 0.05 is significant regardless of the prespecified threshold.",
    "Placebo groups are always equivalent to usual care.",
    "Improvement in a biomarker always guarantees longer survival.",
    "A large sample size removes every source of study bias.",
)

_DESIGN_BALANCE = {
    "procedural_strategy_without_empirical_proposition": 10,
    "contract_conflict_without_empirical_proposition": 10,
    "procedural_strategy_with_empirical_proposition": 10,
    "contract_conflict_with_empirical_proposition": 10,
}


def _sources(workspace: Path) -> dict:
    result = {}
    for source_id, record in _SOURCE_FILES.items():
        source_path = workspace / record["path"]
        license_path = workspace / record["license_path"]
        if not source_path.is_file() or not license_path.is_file():
            raise ValueError(f"required licensed vendor source is missing: {source_id}")
        source_hash, license_hash = sha256_file(source_path), sha256_file(license_path)
        if source_hash != record["sha256"] or license_hash != record["license_sha256"]:
            raise ValueError(f"vendored source or license changed: {source_id}")
        result[source_id] = {
            "path": record["path"], "sha256": source_hash,
            "license": record["license"], "license_path": record["license_path"],
            "license_sha256": license_hash, "source_anchors": list(record["anchors"]),
            "adaptation": "self-authored domain adaptation; no benchmark rows or verbatim prompt block copied",
        }
    return result


def _case(case_text: str, source_id: str, design_cell: str) -> dict:
    strategy = case_text.strip()
    digest = sha256_obj(strategy)
    return {
        "case_id": "vendor-adapted-" + digest[:16],
        "strategy_text": strategy,
        "candidate_text": compose_task_instruction(strategy, CLINIFACT_CONTRACT),
        "candidate_sha256": sha256_obj(compose_task_instruction(strategy, CLINIFACT_CONTRACT)),
        "source_ids": [source_id],
        # This is a sampling-design label only; it is excluded from reference
        # requests and is never substituted for an AI reference verdict.
        "design_cell": design_cell,
    }


def reference_payload(case: dict) -> dict:
    """Return only blinded reference inputs; omit provenance and design strata."""
    return {
        "task_contract": {"id": CLINIFACT_CONTRACT.contract_id,
                          "text": CLINIFACT_CONTRACT.text,
                          "sha256": CLINIFACT_CONTRACT.sha256},
        "candidate_strategy": case["strategy_text"],
    }


def _known_exposure_packets(workspace: Path, *,
                            exclude_cohort_ids: set[str] | None = None
                            ) -> list[tuple[str, dict]]:
    """Load prior instruction-review packets and known code challenge packets."""
    from .ai_instruction_calibration import build_packet as build_ai_challenge
    from .instruction_review_calibration import build_calibration_packet

    packets = [
        ("code:ai_instruction_calibration.build_packet", build_ai_challenge()),
        ("code:instruction_review_calibration.build_calibration_packet",
         build_calibration_packet()),
    ]
    excluded_ids = (set(exclude_cohort_ids) if exclude_cohort_ids is not None
                    else {COHORT_ID, *_SUPERSEDED_COHORT_IDS, *_LATER_COHORT_IDS})
    root = workspace / ".workbuddy"
    if not root.exists():
        return packets
    names = {"packet.json", "adjudication_packet.json"}
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root)
        if (path.name.casefold() not in names
                or not any("instruction" in part.casefold() for part in relative.parts)
                or any(part.startswith(cohort_id)
                       for part in relative.parts for cohort_id in excluded_ids)):
            continue
        try:
            packet = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot audit prior instruction packet: {path}") from exc
        packets.append((relative.as_posix(), packet))
    return packets


def _candidate_texts(value) -> Iterable[str]:
    if isinstance(value, dict):
        for key in ("candidate_text", "strategy_text", "candidate_strategy"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                yield text.strip()
        for child in value.values():
            yield from _candidate_texts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _candidate_texts(child)


def audit_no_overlap(cases: list[dict], prior_packets: list[tuple[str, dict]]) -> dict:
    """Require exact strategy/full-candidate disjointness from exposed packets."""
    prior_hashes: set[str] = set()
    packet_receipts = []
    for source, packet in prior_packets:
        texts = sorted(set(_candidate_texts(packet)))
        prior_hashes.update(sha256_obj(text) for text in texts)
        packet_receipts.append({"source": source, "packet_sha256": packet.get("packet_sha256"),
                                "candidate_text_count": len(texts),
                                "document_sha256": sha256_obj(packet)})
    overlaps = []
    for case in cases:
        for field in ("strategy_text", "candidate_text"):
            if sha256_obj(case[field]) in prior_hashes:
                overlaps.append({"case_id": case["case_id"], "field": field})
    if overlaps:
        raise ValueError(f"new instruction cohort overlaps exposed cases: {overlaps[:3]}")
    return {"prior_packet_count": len(packet_receipts),
            "prior_packet_receipts": packet_receipts,
            "exposed_candidate_text_count": len(prior_hashes),
            "overlap_count": 0, "overlap_verified": True}


def build_packet(workspace: Path | None = None) -> dict:
    workspace = (workspace or repo_root()).resolve()
    source_manifest = _sources(workspace)
    contract = {"id": CLINIFACT_CONTRACT.contract_id,
                "text": CLINIFACT_CONTRACT.text, "sha256": CLINIFACT_CONTRACT.sha256}
    cases = []
    for index, seed in enumerate(_SEEDS):
        source_id, anchor, compatible, incompatible = seed
        empirical = _EMPIRICAL_ASSERTIONS[index]
        cases.extend((
            _case(compatible, source_id, "procedural_strategy_without_empirical_proposition"),
            _case(incompatible, source_id, "contract_conflict_without_empirical_proposition"),
            _case(compatible + " General methodological note: " + empirical,
                  source_id, "procedural_strategy_with_empirical_proposition"),
            _case(incompatible + " General methodological note: " + empirical,
                  source_id, "contract_conflict_with_empirical_proposition"),
        ))
    cases.sort(key=lambda row: row["case_id"])
    if len({case["candidate_sha256"] for case in cases}) != len(cases):
        raise ValueError("instruction cohort contains duplicate candidate strategies")
    exposure_audit = audit_no_overlap(cases, _known_exposure_packets(workspace))
    body = {
        "schema": SCHEMA,
        "cohort_id": COHORT_ID,
        "builder_source_sha256": sha256_file(Path(__file__)),
        "preparation_cli_source_sha256": sha256_file(
            workspace / "scripts" / "prepare_instruction_reviewer_cohort.py"),
        "scope": "unseen_synthetic_clinifact_instruction_reviewer_challenge",
        "natural_distribution": False,
        "source_derived": True,
        "source_manifest": source_manifest,
        "contract": contract,
        "construction_protocol": {
            "version": "vendor-pattern-adaptation-factorial-v1",
            "source_use": "patterns only; all candidate strategies are newly authored adaptations",
            "factors": ["empirical proposition present/absent",
                        "task-contract conflict present/absent"],
            "design_balance": dict(_DESIGN_BALANCE),
            "design_cells_are_labels": False,
            "expected_labels_present": False,
            "target_predictions_present": False,
            "blind_reference_payload_fields": ["task_contract", "candidate_strategy"],
        },
        "reference_protocol": {
            "rubric_sha256": sha256_obj(RUBRIC),
            "target_review_protocol": prompt_grounding_protocol(CLINIFACT_CONTRACT),
            "target_max_tokens": 256,
            "thresholds": dict(THRESHOLDS),
            "two_distinct_reference_model_ids": True,
            "disagreements": "uncertain; do not force adjudication into decisive classes",
        },
        "exposure_audit": exposure_audit,
        "case_count": len(cases),
        "cases": cases,
    }
    return {**body, "packet_sha256": sha256_obj(body)}


def validate_packet(packet: dict, workspace: Path | None = None) -> None:
    expected = build_packet(workspace)
    if packet != expected:
        raise ValueError("instruction reviewer cohort identity, licenses, exposure audit, or protocol changed")
    if (packet["case_count"] < THRESHOLDS["min_cases"]
            or packet["natural_distribution"] is not False
            or packet["construction_protocol"]["expected_labels_present"] is not False
            or packet["construction_protocol"]["target_predictions_present"] is not False
            or packet["exposure_audit"]["overlap_verified"] is not True
            or packet["exposure_audit"]["overlap_count"] != 0):
        raise ValueError("instruction reviewer cohort is not a blinded independent challenge")
    for case in packet["cases"]:
        if (case["candidate_text"] != compose_task_instruction(case["strategy_text"],
                                                               CLINIFACT_CONTRACT)
                or case["candidate_sha256"] != sha256_obj(case["candidate_text"])
                or set(reference_payload(case)) != {"task_contract", "candidate_strategy"}):
            raise ValueError("instruction reviewer challenge case changed")

