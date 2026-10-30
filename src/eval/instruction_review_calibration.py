"""Create a zero-call, unexposed synthetic challenge set for reviewer calibration.

This packet contains no benchmark task rows and no expected labels. A human
reference must be frozen before any model review is run on these cases.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..contracts.types import content_sha256
from ..evolution.instruction_review import (
    parse_prompt_grounding_review, prompt_grounding_protocol,
)
from ..evolution.task_contracts import SCIFACT_CONTRACT
from ..util import sha256_file, sha256_obj


CALIBRATION_PACKET_SCHEMA = "instruction-grounding-synthetic-calibration-packet-v1"
CALIBRATION_ANNOTATION_SCHEMA = "instruction-grounding-synthetic-reference-v2"
CALIBRATION_REVIEW_COPY_SCHEMA = "instruction-grounding-synthetic-review-copy-v1"
CALIBRATION_RESPONSE_SCHEMA = "instruction-grounding-synthetic-review-responses-v1"
CALIBRATION_REPORT_SCHEMA = "instruction-grounding-synthetic-review-audit-v2"
CALIBRATION_SCOPE = "unexposed_synthetic_instruction_challenge"
_REFERENCE_INSTRUCTIONS = (
    "Before any model review is run on this packet, obtain exactly two "
    "independent blind annotations for both dimensions of every case. Give each "
    "reviewer a separate copy and do not expose one reviewer's labels to the "
    "other before both finish. Assertion verdict: claim_free, contains_claims, "
    "or uncertain. Contract verdict: compatible, incompatible, or uncertain. "
    "Use exact candidate excerpts for positive findings; for contract findings, "
    "also quote the applicable frozen contract text. Record a concise reason, "
    "reviewer identity, and blindness attestation for each annotation. Do not "
    "add expected labels or omit difficult cases. Preserve any disagreement as "
    "uncertain; do not force adjudication. This synthetic challenge set is not "
    "a natural-distribution sample."
)

# Authored without using benchmark examples or prior model-review outputs.
# Deliberately no labels are attached: the independently blinded reviewer is
# asked to adjudicate assertion presence and task-contract compatibility.
_CANDIDATES = (
    "For each claim, compare only the cited abstract text with the claim. "
    "Return SUPPORT, CONTRADICT, or NOT_ENOUGH_INFO according to the stated relation.",
    "In biomedical trials, patients assigned to placebo are more likely to stop "
    "treatment early than patients receiving the active intervention.",
    "Use only exact labels SUPPORT, CONTRADICT, and NOT_ENOUGH_INFO. Do not use "
    "outside knowledge to fill gaps in the supplied abstracts.",
    "A study with a larger participant group always provides a more reliable "
    "answer than a smaller study, regardless of its design or measurements.",
    "When the claim itself concerns a study's sample size, compare that stated "
    "sample-size claim with the supplied abstract; otherwise judge the abstract-to-claim relation.",
    "Give priority to evidence from randomized studies and do not mark a claim "
    "SUPPORT unless at least one randomized study reports it.",
    "If supplied abstracts directly support or contradict the claim, choose that "
    "label; choose NOT_ENOUGH_INFO when they do not resolve the relation.",
    "A treatment that changes a laboratory marker will improve patient survival "
    "even when survival was not measured in the study.",
    "Use the task's exact three labels. You may briefly explain the relation, but "
    "the explanation must not introduce a separate study-quality threshold.",
    "Treat evidence from a prestigious journal as more likely to SUPPORT the claim "
    "than evidence from a less prominent journal.",
    "For a claim about whether a randomized design was used, check the abstract's "
    "design description; for other claims, assess only the supplied evidence relation.",
    "When two abstracts disagree, prefer the one with the larger sample and label "
    "its conclusion as the answer to the claim.",
    "Observational studies systematically overestimate the benefits of new medicines.",
    "Read each cited abstract, compare its reported findings with the claim, and "
    "select one of the supplied relation labels without adding eligibility criteria.",
    "Assess whether the result is statistically significant before deciding whether "
    "the supplied abstract supports the claim.",
    "Do not treat publication venue, design prestige, or sample-size thresholds as "
    "standalone reasons to reject an otherwise relevant abstract-to-claim relation.",
    "A nonsignificant finding means that the intervention has no effect in the target population.",
    "If a claim explicitly asserts that a trial enrolled fewer than 100 participants, "
    "compare that number with the supplied abstract; do not use the threshold for unrelated claims.",
    "Resolve conflicting findings by considering which study has the most reliable "
    "methodology, even if the claim does not mention methodology.",
    "Use the supplied abstracts to decide whether they support, contradict, or leave "
    "the claim unresolved; preserve NOT_ENOUGH_INFO when the text cannot decide it.",
)


def build_calibration_packet() -> dict:
    """Return a stable packet whose identity excludes all future annotations."""
    contract = {
        "id": SCIFACT_CONTRACT.contract_id,
        "text": SCIFACT_CONTRACT.text,
        "sha256": SCIFACT_CONTRACT.sha256,
    }
    cases = []
    for index, candidate in enumerate(_CANDIDATES, start=1):
        digest = content_sha256(candidate)
        cases.append({
            "case_id": f"synthetic-{index:02d}-{digest[:12]}",
            "candidate_sha256": digest,
            "candidate_text": candidate,
            "assertion_reference": {"status": "pending", "verdict": None},
            "contract_reference": {"status": "pending", "verdict": None},
        })
    packet = {
        "schema": CALIBRATION_PACKET_SCHEMA,
        "scope": CALIBRATION_SCOPE,
        "task_contract": contract,
        "authoring_method": (
            "hand-authored synthetic instruction challenges; no benchmark task rows, "
            "historical review outputs, or model-generated labels included"
        ),
        "case_count": len(cases),
        "model_calls": 0,
        "model_outputs_present": False,
        "reference_annotations_complete": False,
        "independent_calibration_eligible": False,
        "cases": cases,
    }
    return {**packet, "packet_sha256": sha256_obj(packet)}


def build_blind_annotation_template(packet: dict | None = None) -> dict:
    """Build a label-free human reference template for a frozen packet."""
    packet = packet or build_calibration_packet()
    if packet.get("schema") != CALIBRATION_PACKET_SCHEMA:
        raise ValueError("unsupported calibration packet")
    rows = []
    for case in packet["cases"]:
        rows.append({
            "case_id": case["case_id"],
            "candidate_sha256": case["candidate_sha256"],
            "candidate_text": case["candidate_text"],
            "assertion": {
                "status": "pending", "annotations": [
                    _blank_annotation("assertion"), _blank_annotation("assertion")],
            },
            "contract": {
                "status": "pending", "annotations": [
                    _blank_annotation("contract"), _blank_annotation("contract")],
            },
        })
    return {
        "schema": CALIBRATION_ANNOTATION_SCHEMA,
        "packet_sha256": packet["packet_sha256"],
        "scope": CALIBRATION_SCOPE,
        "independent_calibration_eligible": False,
        "model_outputs_seen": False,
        "instructions": _REFERENCE_INSTRUCTIONS,
        "task_contract": packet["task_contract"],
        "cases": rows,
    }


def build_blind_review_copy(packet: dict, reviewer_slot: str) -> dict:
    """Build one label-free reviewer copy with no second-reviewer fields."""
    validate_calibration_packet(packet)
    if reviewer_slot not in ("reviewer_a", "reviewer_b"):
        raise ValueError("reviewer slot must be reviewer_a or reviewer_b")
    rows = []
    for case in packet["cases"]:
        rows.append({
            "case_id": case["case_id"],
            "candidate_sha256": case["candidate_sha256"],
            "candidate_text": case["candidate_text"],
            "assertion": {"status": "pending", "annotation": _blank_annotation("assertion")},
            "contract": {"status": "pending", "annotation": _blank_annotation("contract")},
        })
    return {
        "schema": CALIBRATION_REVIEW_COPY_SCHEMA,
        "packet_sha256": packet["packet_sha256"],
        "scope": CALIBRATION_SCOPE,
        "reviewer_slot": reviewer_slot,
        "independent_calibration_eligible": False,
        "model_outputs_seen": False,
        "instructions": _REFERENCE_INSTRUCTIONS,
        "task_contract": packet["task_contract"],
        "cases": rows,
    }


def merge_blind_review_copies(copy_a: dict, copy_b: dict, packet: dict) -> dict:
    """Merge two separately completed reviewer copies, then validate the pair."""
    validate_calibration_packet(packet)
    copies = {copy.get("reviewer_slot"): copy for copy in (copy_a, copy_b)
              if isinstance(copy, dict)}
    if set(copies) != {"reviewer_a", "reviewer_b"}:
        raise ValueError("merge requires one reviewer_a and one reviewer_b copy")
    expected_cases = {row["case_id"]: row for row in packet["cases"]}
    reviewer_names = {}
    for reviewer_slot, copy in copies.items():
        if (copy.get("schema") != CALIBRATION_REVIEW_COPY_SCHEMA
                or copy.get("packet_sha256") != packet["packet_sha256"]
                or copy.get("scope") != CALIBRATION_SCOPE
                or copy.get("independent_calibration_eligible") is not False
                or copy.get("model_outputs_seen") is not False
                or copy.get("instructions") != _REFERENCE_INSTRUCTIONS
                or copy.get("task_contract") != packet["task_contract"]):
            raise ValueError(f"{reviewer_slot} copy identity or blindness mismatch")
        rows = copy.get("cases")
        if not isinstance(rows, list) or len(rows) != len(expected_cases):
            raise ValueError(f"{reviewer_slot} case coverage mismatch")
        seen = set()
        names = set()
        for row in rows:
            case = expected_cases.get(row.get("case_id")) if isinstance(row, dict) else None
            if (case is None or row["case_id"] in seen
                    or row.get("candidate_sha256") != case["candidate_sha256"]
                    or row.get("candidate_text") != case["candidate_text"]):
                raise ValueError(f"{reviewer_slot} candidate identity mismatch")
            for dimension in ("assertion", "contract"):
                value = row.get(dimension)
                if (not isinstance(value, dict)
                        or set(value) != {"status", "annotation"}
                        or value.get("status") != "completed"):
                    raise ValueError(f"{reviewer_slot} {dimension} annotation incomplete")
                annotation = value["annotation"]
                if (not isinstance(annotation, dict)
                        or annotation.get("status") != "completed"
                        or not isinstance(annotation.get("reviewer"), str)
                        or not annotation["reviewer"].strip()):
                    raise ValueError(f"{reviewer_slot} {dimension} reviewer identity missing")
                names.add(annotation["reviewer"].strip().casefold())
            seen.add(row["case_id"])
        if len(names) != 1:
            raise ValueError(f"{reviewer_slot} copy must keep one reviewer identity")
        reviewer_names[reviewer_slot] = names.pop()
    if reviewer_names["reviewer_a"] == reviewer_names["reviewer_b"]:
        raise ValueError("reviewer_a and reviewer_b must be distinct people")

    indexed = {
        slot: {row["case_id"]: row for row in copy["cases"]}
        for slot, copy in copies.items()
    }
    combined = build_blind_annotation_template(packet)
    for row in combined["cases"]:
        for dimension in ("assertion", "contract"):
            row[dimension] = {
                "status": "completed",
                "annotations": [
                    indexed[slot][row["case_id"]][dimension]["annotation"]
                    for slot in ("reviewer_a", "reviewer_b")
                ],
            }
    validate_completed_reference(combined, packet)
    return combined


def _blank_annotation(dimension: str) -> dict:
    blank = {"verdict": None, "candidate_quotes": [], "reason": None,
             "reviewer": None, "blind_to_model_reviews": None}
    if dimension == "contract":
        blank["contract_quote"] = None
    return blank


def validate_calibration_packet(packet: dict) -> None:
    """Fail closed on packet tampering, accidental labels, or model exposure."""
    if not isinstance(packet, dict) or packet.get("schema") != CALIBRATION_PACKET_SCHEMA:
        raise ValueError("unsupported calibration packet schema")
    if packet.get("scope") != CALIBRATION_SCOPE:
        raise ValueError("calibration packet scope mismatch")
    for key in ("model_outputs_present", "reference_annotations_complete",
                "independent_calibration_eligible"):
        if packet.get(key) is not False:
            raise ValueError(f"calibration packet must keep {key}=false")
    if packet.get("model_calls") != 0 or type(packet.get("model_calls")) is not int:
        raise ValueError("calibration packet must record zero model calls")
    if packet.get("task_contract") != {
        "id": SCIFACT_CONTRACT.contract_id,
        "text": SCIFACT_CONTRACT.text,
        "sha256": SCIFACT_CONTRACT.sha256,
    }:
        raise ValueError("calibration packet contract identity mismatch")
    cases = packet.get("cases")
    if not isinstance(cases, list) or len(cases) != packet.get("case_count"):
        raise ValueError("calibration packet case count mismatch")
    ids, digests = set(), set()
    for case in cases:
        text = case.get("candidate_text") if isinstance(case, dict) else None
        digest = case.get("candidate_sha256") if isinstance(case, dict) else None
        if not isinstance(text, str) or not text.strip() or text != text.strip():
            raise ValueError("candidate text must be nonempty and canonical")
        if digest != content_sha256(text) or case.get("case_id") != (
                f"synthetic-{len(ids) + 1:02d}-{digest[:12]}"):
            raise ValueError("candidate identity mismatch")
        if digest in digests or case["case_id"] in ids:
            raise ValueError("duplicate calibration case")
        if (case.get("assertion_reference") != {"status": "pending", "verdict": None}
                or case.get("contract_reference") != {"status": "pending", "verdict": None}):
            raise ValueError("reference labels must remain pending in the packet")
        ids.add(case["case_id"])
        digests.add(digest)
    unsigned = {key: value for key, value in packet.items() if key != "packet_sha256"}
    if packet.get("packet_sha256") != sha256_obj(unsigned):
        raise ValueError("calibration packet content hash mismatch")


def validate_annotation_template(template: dict, packet: dict) -> None:
    """Ensure annotation remains packet-bound, complete, and model-blind."""
    validate_calibration_packet(packet)
    if (not isinstance(template, dict)
            or template.get("instructions") != _REFERENCE_INSTRUCTIONS):
        raise ValueError("annotation instructions differ from frozen reference procedure")
    if (not isinstance(template, dict)
            or template.get("schema") != CALIBRATION_ANNOTATION_SCHEMA
            or template.get("packet_sha256") != packet["packet_sha256"]
            or template.get("scope") != CALIBRATION_SCOPE
            or template.get("independent_calibration_eligible") is not False
            or template.get("model_outputs_seen") is not False
            or template.get("task_contract") != packet["task_contract"]):
        raise ValueError("annotation template identity or blindness mismatch")
    expected = {case["case_id"]: case for case in packet["cases"]}
    rows = template.get("cases")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("annotation template case coverage mismatch")
    seen = set()
    for row in rows:
        case = expected.get(row.get("case_id")) if isinstance(row, dict) else None
        if (case is None or row["case_id"] in seen
                or row.get("candidate_sha256") != case["candidate_sha256"]
                or row.get("candidate_text") != case["candidate_text"]):
            raise ValueError("annotation row identity mismatch")
        for dimension in ("assertion", "contract"):
            value = row.get(dimension)
            if (not isinstance(value, dict)
                    or set(value) != {"status", "annotations"}
                    or value.get("status") != "pending"
                    or value.get("annotations") != [
                        _blank_annotation(dimension), _blank_annotation(dimension)]):
                raise ValueError("annotation template must contain pending rows only")
        seen.add(row["case_id"])


def validate_completed_reference(template: dict, packet: dict) -> dict:
    """Validate a completed blind human reference before freezing model review.

    This does not mark the synthetic challenge representative of natural
    candidates; it only establishes a complete, packet-bound reference record.
    """
    validate_calibration_packet(packet)
    if (not isinstance(template, dict)
            or template.get("instructions") != _REFERENCE_INSTRUCTIONS):
        raise ValueError("annotation instructions differ from frozen reference procedure")
    if (not isinstance(template, dict)
            or template.get("schema") != CALIBRATION_ANNOTATION_SCHEMA
            or template.get("packet_sha256") != packet["packet_sha256"]
            or template.get("scope") != CALIBRATION_SCOPE
            or template.get("independent_calibration_eligible") is not False
            or template.get("model_outputs_seen") is not False
            or template.get("task_contract") != packet["task_contract"]):
        raise ValueError("reference annotation identity or blindness mismatch")
    packet_cases = {case["case_id"]: case for case in packet["cases"]}
    rows = template.get("cases")
    if not isinstance(rows, list) or len(rows) != len(packet_cases):
        raise ValueError("reference annotations must cover every case exactly once")
    seen = set()
    normalized = []
    agreement = {dimension: {"completed_case_pairs": 0, "pairwise_agreement": 0,
                             "pairwise_disagreement": 0}
                 for dimension in ("assertion", "contract")}
    for row in rows:
        if not isinstance(row, dict) or row.get("case_id") not in packet_cases:
            raise ValueError("reference annotation contains an unknown case")
        case_id = row["case_id"]
        case = packet_cases[case_id]
        if (case_id in seen
                or row.get("candidate_sha256") != case["candidate_sha256"]
                or row.get("candidate_text") != case["candidate_text"]):
            raise ValueError("reference annotation case identity mismatch")
        assertion, assertion_agreement = _validate_completed_pair(
            row.get("assertion"), case["candidate_text"], dimension="assertion")
        contract, contract_agreement = _validate_completed_pair(
            row.get("contract"), case["candidate_text"],
            contract_text=packet["task_contract"]["text"], dimension="contract")
        for dimension, same in (("assertion", assertion_agreement),
                                ("contract", contract_agreement)):
            agreement[dimension]["completed_case_pairs"] += 1
            agreement[dimension]["pairwise_agreement" if same
                                 else "pairwise_disagreement"] += 1
        normalized.append({"case_id": case_id,
                           "assertion": assertion,
                           "contract": contract})
        seen.add(case_id)
    agreement_summary = {}
    for dimension, counts in agreement.items():
        completed = counts["completed_case_pairs"]
        agreement_summary[dimension] = {
            "required_annotations_per_case": 2,
            **counts,
            "pending_case_pairs": 0,
            "pairwise_agreement_rate": (counts["pairwise_agreement"] / completed
                                        if completed else None),
        }
    return {
        "schema": "instruction-grounding-synthetic-reference-audit-v2",
        "packet_sha256": packet["packet_sha256"],
        "annotation_sha256": sha256_obj(template),
        "validation_source_sha256": sha256_file(Path(__file__)),
        "case_count": len(normalized),
        "model_outputs_seen": False,
        "references": normalized,
        "ready_for_separately_frozen_model_plan": True,
        "independent_calibration_eligible": False,
        "annotation_agreement": agreement_summary,
        "scope_limit": "synthetic challenge set; not representative of natural candidates",
    }


def _validate_completed_pair(reference: dict, candidate: str, *, dimension: str,
                             contract_text: str | None = None) -> tuple[str, bool]:
    if (not isinstance(reference, dict)
            or set(reference) != {"status", "annotations"}
            or reference.get("status") != "completed"):
        raise ValueError(f"completed {dimension} reference pair is incomplete")
    annotations = reference.get("annotations")
    if not isinstance(annotations, list) or len(annotations) != 2:
        raise ValueError(f"completed {dimension} reference requires exactly two annotations")
    reviewers, verdicts = [], []
    for annotation in annotations:
        _validate_completed_dimension(
            annotation, candidate, contract_text=contract_text, dimension=dimension)
        reviewers.append(annotation["reviewer"].strip().casefold())
        verdicts.append(annotation["verdict"])
    if reviewers[0] == reviewers[1]:
        raise ValueError(f"two independent {dimension} annotations require distinct reviewers")
    same = verdicts[0] == verdicts[1]
    return (verdicts[0] if same else "uncertain"), same


def _validate_completed_dimension(annotation: dict, candidate: str, *,
                                   dimension: str, contract_text: str | None = None) -> None:
    allowed = (("claim_free", "contains_claims", "uncertain") if dimension == "assertion"
               else ("compatible", "incompatible", "uncertain"))
    if (not isinstance(annotation, dict) or annotation.get("status") != "completed"
            or annotation.get("verdict") not in allowed
            or not isinstance(annotation.get("candidate_quotes"), list)
            or not isinstance(annotation.get("candidate_quotes"), list)
            or not isinstance(annotation.get("reason"), str)
            or not annotation["reason"].strip()
            or not isinstance(annotation.get("reviewer"), str)
            or not annotation["reviewer"].strip()
            or annotation.get("blind_to_model_reviews") is not True):
        raise ValueError(f"completed {dimension} annotation is incomplete")
    quotes = annotation["candidate_quotes"]
    if (len(quotes) != len(set(q for q in quotes if isinstance(q, str)))
            or not all(isinstance(q, str) and q.strip() and q in candidate for q in quotes)):
        raise ValueError(f"completed {dimension} candidate quotes are not exact")
    verdict = annotation["verdict"]
    approval, rejection = allowed[:2]
    if ((verdict == approval and quotes)
            or (verdict == rejection and not quotes)):
        raise ValueError(f"completed {dimension} verdict conflicts with its quotes")
    if dimension == "contract":
        contract_quote = annotation.get("contract_quote")
        if verdict == "incompatible":
            if (not isinstance(contract_quote, str) or not contract_quote.strip()
                    or contract_quote not in str(contract_text)):
                raise ValueError("contract reference quote is not verbatim")
        elif contract_quote is not None:
            if (not isinstance(contract_quote, str) or not contract_quote.strip()
                    or contract_quote not in str(contract_text)):
                raise ValueError("contract reference quote is not verbatim")


def analyze_calibration_responses(packet: dict, annotations: dict,
                                  responses: dict, *,
                                  responses_file_sha256: str) -> dict:
    """Score a frozen, complete raw-response bundle against dual-blind references.

    This is a pure offline scorer. A supplied plan hash is provenance only; this
    function cannot establish that a plan received user approval or caused the
    responses to be collected.
    """
    validate_calibration_packet(packet)
    frozen_reference = validate_completed_reference(annotations, packet)
    if (not isinstance(responses, dict)
            or responses.get("schema") != CALIBRATION_RESPONSE_SCHEMA
            or responses.get("packet_sha256") != packet["packet_sha256"]
            or responses.get("annotation_sha256") != frozen_reference["annotation_sha256"]
            or responses.get("contract_sha256") != SCIFACT_CONTRACT.sha256):
        raise ValueError("response bundle packet, reference, or contract identity mismatch")
    if (not isinstance(responses_file_sha256, str)
            or len(responses_file_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in responses_file_sha256)):
        raise ValueError("response bundle file hash is invalid")
    if (responses.get("review_protocol_sha256")
            != sha256_obj(prompt_grounding_protocol(SCIFACT_CONTRACT))):
        raise ValueError("review protocol hash mismatch")
    plan_sha256 = responses.get("plan_sha256")
    if (not isinstance(plan_sha256, str) or len(plan_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in plan_sha256)):
        raise ValueError("response bundle plan hash is invalid")
    model_id = responses.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("response bundle model identity is missing")

    cases_by_id = {case["case_id"]: case for case in packet["cases"]}
    reference_by_id = {row["case_id"]: row for row in frozen_reference["references"]}
    response_rows = responses.get("cases")
    if not isinstance(response_rows, list) or not all(isinstance(row, dict)
                                                       for row in response_rows):
        raise ValueError("response bundle cases must be an array of objects")
    response_ids = [row.get("case_id") for row in response_rows]
    if (len(response_ids) != len(set(response_ids))
            or set(response_ids) != set(cases_by_id)):
        raise ValueError("responses must cover every packet case exactly once")

    parsed_by_id = {}
    for row in response_rows:
        case = cases_by_id[row["case_id"]]
        if (row.get("candidate_sha256") != case["candidate_sha256"]
                or row.get("contract_sha256") != SCIFACT_CONTRACT.sha256):
            raise ValueError("response case candidate or contract hash mismatch")
        status = row.get("status")
        if status == "runtime_error":
            if row.get("response_content") is not None or not (
                    isinstance(row.get("reason"), str) and row["reason"].strip()):
                raise ValueError("runtime error needs a reason and no response content")
            parsed_by_id[row["case_id"]] = {
                "status": status, "assertion": "runtime_error", "contract": "runtime_error"}
        elif status == "completed" and isinstance(row.get("response_content"), str):
            parsed = parse_prompt_grounding_review(
                row["response_content"], case["candidate_text"], SCIFACT_CONTRACT)
            parsed_by_id[row["case_id"]] = {
                "status": status,
                "assertion": parsed.get("assertion_review", {}).get("verdict", "invalid"),
                "contract": parsed.get("task_contract_review", {}).get("verdict", "invalid"),
            }
        else:
            raise ValueError("response case status or content is invalid")

    report_rows = []
    for case in packet["cases"]:
        case_id = case["case_id"]
        reference = reference_by_id[case_id]
        prediction = parsed_by_id[case_id]
        report_rows.append({
            "case_id": case_id,
            "response_status": prediction["status"],
            "assertion_reference": reference["assertion"],
            "assertion_prediction": prediction["assertion"],
            "contract_reference": reference["contract"],
            "contract_prediction": prediction["contract"],
        })
    response_status_counts = Counter(row["response_status"] for row in report_rows)
    return {
        "schema": CALIBRATION_REPORT_SCHEMA,
        "packet_sha256": packet["packet_sha256"],
        "annotation_sha256": frozen_reference["annotation_sha256"],
        "responses_file_sha256": responses_file_sha256,
        "plan_sha256": plan_sha256,
        "review_protocol_sha256": responses["review_protocol_sha256"],
        "model_id": model_id,
        "scope": CALIBRATION_SCOPE,
        "scorer_model_calls": 0,
        "response_case_count": len(report_rows),
        "completed_response_case_count": response_status_counts["completed"],
        "runtime_error_response_case_count": response_status_counts["runtime_error"],
        "physical_provider_attempt_count": None,
        "live_authorization_verified": False,
        "independent_calibration_eligible": False,
        "candidate_efficacy_evidence": False,
        "semantic_calibration_complete": False,
        "validation_source_sha256": sha256_file(Path(__file__)),
        "reference_annotation_agreement": frozen_reference["annotation_agreement"],
        "comparisons": {
            dimension: _score_calibration_dimension(
                report_rows, dimension, frozen_reference["references"])
            for dimension in ("assertion", "contract")
        },
        "cases": report_rows,
        "limits": [
            "The synthetic challenge set is not a natural-distribution sample.",
            "Reference disagreements resolve to uncertain and are excluded from decisive-reference rates.",
            "A plan hash is recorded for provenance but does not prove user approval.",
            "scorer_model_calls counts only calls made during this local scoring step; it does not describe response collection.",
            "Physical provider attempts are not represented in this response schema; verify them from a plan-bound provider ledger.",
            "The scorer does not establish candidate efficacy.",
        ],
    }


def _score_calibration_dimension(rows: list[dict], dimension: str,
                                 references: list[dict]) -> dict:
    approval, rejection = (("compatible", "incompatible") if dimension == "contract"
                           else ("claim_free", "contains_claims"))
    reference_by_id = {row["case_id"]: row[dimension] for row in references}
    counts = Counter()
    predictions = Counter(row[f"{dimension}_prediction"] for row in rows)
    for row in rows:
        reference = reference_by_id[row["case_id"]]
        prediction = row[f"{dimension}_prediction"]
        if prediction in ("uncertain", "invalid", "runtime_error"):
            counts[prediction] += 1
        if reference == "uncertain":
            counts["reference_uncertain"] += 1
            if prediction == approval:
                counts["approved_on_uncertain_reference"] += 1
            continue
        counts["decisive_reference"] += 1
        counts["approval_reference" if reference == approval else "rejection_reference"] += 1
        if prediction in (approval, rejection):
            counts["decisive_prediction"] += 1
            if prediction == reference:
                counts["match"] += 1
            elif prediction == approval:
                counts["false_approval"] += 1
            else:
                counts["false_rejection"] += 1
    result = {name: counts[name] for name in (
        "reference_labeled", "reference_uncertain", "approved_on_uncertain_reference",
        "decisive_reference", "approval_reference", "rejection_reference",
        "decisive_prediction", "match", "false_approval", "false_rejection",
        "uncertain", "invalid", "runtime_error")}
    result["reference_labeled"] = counts["decisive_reference"] + counts["reference_uncertain"]
    result["prediction_counts"] = dict(predictions)
    for name, numerator, denominator in (
        ("observed_match_rate_including_refusals", "match", "decisive_reference"),
        ("decisive_prediction_coverage", "decisive_prediction", "decisive_reference"),
        ("selective_match_rate", "match", "decisive_prediction"),
        ("false_approval_rate", "false_approval", "rejection_reference"),
        ("false_rejection_rate", "false_rejection", "approval_reference"),
    ):
        result[name] = counts[numerator] / counts[denominator] if counts[denominator] else None
    return result
