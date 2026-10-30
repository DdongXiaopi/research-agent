"""Offline instruction-review diagnostics; exposed cases never become calibration gold.

No provider, task loader or retrieval client is imported. Historical processed
verdicts and raw responses have different provenance and are reported separately.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

from ..contracts.types import content_sha256
from ..evolution.instruction_review import (
    parse_prompt_grounding_review, prompt_grounding_protocol,
)
from ..evolution.task_contracts import SCIFACT_CONTRACT
from ..util import sha256_file, sha256_obj

PACKET_SCHEMA = "instruction-grounding-development-review-packet-v1"
ANNOTATION_SCHEMA = "instruction-grounding-reference-annotations-v2"
RESPONSE_SCHEMA = "instruction-grounding-responses-v1"
ROLE = "exposed_development_regression_only"
ANNOTATION_INSTRUCTIONS = (
    "Review each instruction against the frozen contract. Label assertion presence "
    "and decision-contract compatibility separately. Use exact candidate excerpts; "
    "a contract rejection also needs a verbatim contract excerpt. Record reviewer, "
    "reason and blindness attestation. Assign separate copies to two reviewers and "
    "merge them only after both finish; do not expose one reviewer's labels to the other. "
    "Do not consult historical model verdicts or provisional labels. Disagreements remain "
    "uncertain. These exposed cases remain development diagnostics."
)


def _nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _digest(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _quotes(quotes, text: str) -> bool:
    return (isinstance(quotes, list)
            and len(quotes) == len(set(q for q in quotes if isinstance(q, str)))
            and all(_nonempty(q) and q in text for q in quotes))


def _annotation(annotation: dict, candidate: str, contract_text: str,
                *, dimension="contract", completed=False) -> None:
    allowed = (("compatible", "incompatible", "uncertain") if dimension == "contract"
               else ("claim_free", "contains_claims", "uncertain"))
    approval, rejection, _ = allowed
    if not isinstance(annotation, dict) or annotation.get("verdict") not in allowed:
        raise ValueError(f"invalid {dimension} reference verdict")
    quotes = annotation.get("candidate_quotes")
    if (not _quotes(quotes, candidate)
            or annotation["verdict"] == rejection and not quotes
            or annotation["verdict"] == approval and quotes
            or not _nonempty(annotation.get("reason"))):
        raise ValueError(f"invalid {dimension} reference evidence")
    if dimension == "contract" and (
        not _nonempty(annotation.get("contract_quote"))
        or annotation["contract_quote"] not in contract_text
    ):
        raise ValueError("reference contract quote is not verbatim")
    if completed and (not _nonempty(annotation.get("reviewer"))
                      or annotation.get("blind_to_model_reviews") is not True):
        raise ValueError("completed reference requires reviewer and blindness attestation")


def validate_packet(packet: dict) -> None:
    if not isinstance(packet, dict) or packet.get("schema") != PACKET_SCHEMA:
        raise ValueError("unsupported development packet schema")
    for marker in ("independent_calibration_eligible", "candidate_efficacy_evidence",
                   "provisional_annotation_is_independent"):
        if packet.get(marker) is not False:
            raise ValueError(f"exposed development packet must keep {marker}=false")
    if type(packet.get("model_calls")) is not int or packet["model_calls"] != 0:
        raise ValueError("development packet must record zero model calls")
    if not _digest(packet.get("source_report_sha256")):
        raise ValueError("source report digest is missing")
    contract = packet.get("task_contract")
    if contract != {"id": SCIFACT_CONTRACT.contract_id, "text": SCIFACT_CONTRACT.text,
                    "sha256": SCIFACT_CONTRACT.sha256}:
        raise ValueError("development packet differs from the frozen task contract")
    cases = packet.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("development packet must contain cases")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or case.get("role") != ROLE:
            raise ValueError("unsupported case scope; exposed cases cannot be independent")
        text = case.get("candidate_text")
        digest = case.get("candidate_sha256")
        if (not _nonempty(text) or text != text.strip() or not _digest(digest)
                or content_sha256(text) != digest or case.get("case_id") != digest):
            raise ValueError("candidate identity does not match canonical instruction")
        if digest in seen:
            raise ValueError("duplicate candidate case")
        seen.add(digest)
        _annotation(case.get("provisional_contract_annotation"), text, contract["text"])
        # In-packet pending labels are not reference data. Completed labels must
        # arrive in a separately frozen, hash-bound annotation document.
        pending = case.get("independent_annotation")
        if (not isinstance(pending, dict) or pending.get("status") != "pending"
                or pending.get("verdict") is not None):
            raise ValueError("in-packet independent annotation must remain pending")
        historical = case.get("historical_review")
        if not isinstance(historical, dict) or historical.get("candidate_sha256") != digest:
            raise ValueError("historical review candidate identity mismatch")
        if not _nonempty(historical.get("model_id")):
            raise ValueError("historical review model identity is missing")
        if historical.get("verdict") not in ("claim_free", "contains_claims", "uncertain", "invalid"):
            raise ValueError("historical assertion verdict is invalid")
        nested = historical.get("task_contract_review")
        if not isinstance(nested, dict) or (
            nested.get("candidate_sha256") != digest
            or nested.get("contract_id") != SCIFACT_CONTRACT.contract_id
            or nested.get("contract_sha256") != SCIFACT_CONTRACT.sha256
        ):
            raise ValueError("historical review contract identity mismatch")
        if nested.get("verdict") not in ("compatible", "incompatible", "uncertain", "invalid"):
            raise ValueError("historical contract verdict is invalid")


def prepare_offline_plan(packet: dict, packet_sha256: str) -> dict:
    validate_packet(packet)
    if not _digest(packet_sha256):
        raise ValueError("packet file digest is invalid")
    protocol = prompt_grounding_protocol(SCIFACT_CONTRACT)
    plan = {
        "schema": "instruction-grounding-offline-plan-v1",
        "packet_sha256": packet_sha256,
        "case_ids": [case["case_id"] for case in packet["cases"]],
        "scope": ROLE,
        "review_protocol": protocol,
        "review_protocol_sha256": sha256_obj(protocol),
        "scoring_source_sha256": sha256_file(Path(__file__)),
        "model_calls": 0, "logical_request_cap": 0, "physical_request_cap": 0,
        "live_authorized": False, "independent_calibration_eligible": False,
        "candidate_efficacy_evidence": False,
        "requirements_for_semantic_calibration": [
            "Freeze a separate, unexposed instruction corpus and its sampling scope.",
            "Obtain independent reference annotations before inspecting model verdicts.",
            "Freeze model, decoding, retry policy, scoring and request caps in a new live plan.",
            "Obtain explicit user approval for that exact plan before any API request.",
        ],
    }
    return {**plan, "plan_sha256": sha256_obj(plan)}


def blind_annotation_template(packet: dict, packet_sha256: str) -> dict:
    validate_packet(packet)
    if not _digest(packet_sha256):
        raise ValueError("packet file digest is invalid")
    return {
        "schema": ANNOTATION_SCHEMA, "packet_sha256": packet_sha256,
        "contract_sha256": SCIFACT_CONTRACT.sha256, "scope": ROLE,
        "independent_calibration_eligible": False,
        "instructions": ANNOTATION_INSTRUCTIONS,
        "task_contract": packet["task_contract"],
        "cases": [{
            "case_id": case["case_id"], "candidate_sha256": case["candidate_sha256"],
            "candidate_text": case["candidate_text"],
            "assertion": {"status": "pending", "annotations": [
                _blank_annotation("assertion"), _blank_annotation("assertion")]},
            "contract": {"status": "pending", "annotations": [
                _blank_annotation("contract"), _blank_annotation("contract")]},
        } for case in packet["cases"]],
    }


def _blank_annotation(dimension: str) -> dict:
    blank = {"verdict": None, "candidate_quotes": [], "reason": None,
             "reviewer": None, "blind_to_model_reviews": None}
    if dimension == "contract":
        blank["contract_quote"] = None
    return blank


def _bound_rows(document: dict, schema: str, packet: dict, digest: str) -> dict:
    if (not isinstance(document, dict) or document.get("schema") != schema
            or document.get("packet_sha256") != digest):
        raise ValueError("review document schema or packet hash mismatch")
    cases = document.get("cases")
    if not isinstance(cases, list) or not all(isinstance(row, dict) for row in cases):
        raise ValueError("review document cases must be an array of objects")
    ids = [row.get("case_id") for row in cases]
    if (not all(_digest(case_id) for case_id in ids)
            or len(ids) != len(set(ids))
            or set(ids) != {case["case_id"] for case in packet["cases"]}):
        raise ValueError("review document must cover every frozen case exactly once")
    indexed = {row["case_id"]: row for row in cases}
    for case in packet["cases"]:
        if indexed[case["case_id"]].get("candidate_sha256") != case["candidate_sha256"]:
            raise ValueError("review document candidate hash mismatch")
    return indexed


def _references(packet: dict, packet_sha: str, annotations: dict | None) -> tuple[dict, dict]:
    if annotations is None:
        return {}, {dimension: {
            "required_annotations_per_case": 2, "completed_case_pairs": 0,
            "pending_case_pairs": len(packet["cases"]), "pairwise_agreement": 0,
            "pairwise_disagreement": 0, "pairwise_agreement_rate": None,
        } for dimension in ("assertion", "contract")}
    if (not isinstance(annotations, dict)
            or annotations.get("instructions") != ANNOTATION_INSTRUCTIONS
            or annotations.get("contract_sha256") != SCIFACT_CONTRACT.sha256
            or annotations.get("scope") != ROLE
            or annotations.get("independent_calibration_eligible") is not False
            or annotations.get("task_contract") != packet["task_contract"]):
        raise ValueError("reference document contract or development scope mismatch")
    rows = _bound_rows(annotations, ANNOTATION_SCHEMA, packet, packet_sha)
    references = {}
    agreement = {dimension: Counter() for dimension in ("assertion", "contract")}
    for case in packet["cases"]:
        row = rows[case["case_id"]]
        if row.get("candidate_text") != case["candidate_text"]:
            raise ValueError("reference document candidate text mismatch")
        refs = {}
        for dimension in ("assertion", "contract"):
            ref = row.get(dimension)
            if not isinstance(ref, dict) or ref.get("status") not in ("pending", "completed"):
                raise ValueError("reference status must be pending or completed")
            votes = ref.get("annotations")
            if not isinstance(votes, list) or len(votes) != 2:
                raise ValueError("each reference dimension requires exactly two annotations")
            if ref["status"] == "pending":
                if votes != [_blank_annotation(dimension), _blank_annotation(dimension)]:
                    raise ValueError("pending reference must contain two blank annotations")
                agreement[dimension]["pending_case_pairs"] += 1
                continue
            reviewers, verdicts = [], []
            for vote in votes:
                _annotation(vote, case["candidate_text"], SCIFACT_CONTRACT.text,
                            dimension=dimension, completed=True)
                reviewers.append(vote["reviewer"].strip().casefold())
                verdicts.append(vote["verdict"])
            if reviewers[0] == reviewers[1]:
                raise ValueError("two independent annotations require distinct reviewers")
            agreement[dimension]["completed_case_pairs"] += 1
            same = verdicts[0] == verdicts[1]
            agreement[dimension]["pairwise_agreement" if same
                                 else "pairwise_disagreement"] += 1
            refs[dimension] = verdicts[0] if same else "uncertain"
        references[case["case_id"]] = refs
    diagnostics = {}
    for dimension, counts in agreement.items():
        completed = counts["completed_case_pairs"]
        diagnostics[dimension] = {
            "required_annotations_per_case": 2,
            "completed_case_pairs": completed,
            "pending_case_pairs": counts["pending_case_pairs"],
            "pairwise_agreement": counts["pairwise_agreement"],
            "pairwise_disagreement": counts["pairwise_disagreement"],
            "pairwise_agreement_rate": (counts["pairwise_agreement"] / completed
                                        if completed else None),
        }
    return references, diagnostics


def _comparison(rows: list[dict], references: dict, dimension: str) -> dict:
    approval, rejection = (("compatible", "incompatible") if dimension == "contract"
                           else ("claim_free", "contains_claims"))
    counts = Counter()
    predictions = Counter(row[dimension] for row in rows)
    for row in rows:
        ref = references.get(row["case_id"], {}).get(dimension)
        pred = row[dimension]
        if ref is None:
            counts["reference_pending"] += 1
            continue
        counts["reference_labeled"] += 1
        if ref == "uncertain":
            counts["reference_uncertain"] += 1
            if pred == approval:
                counts["approved_on_uncertain_reference"] += 1
            continue
        counts["decisive_reference"] += 1
        counts["approval_reference" if ref == approval else "rejection_reference"] += 1
        if pred in (approval, rejection):
            counts["decisive_prediction"] += 1
            if pred == ref:
                counts["match"] += 1
            elif pred == approval:
                counts["false_approval"] += 1
            else:
                counts["false_rejection"] += 1
        else:
            counts[pred if pred in ("uncertain", "invalid", "runtime_error") else "invalid"] += 1
    result = {name: counts[name] for name in (
        "reference_labeled", "reference_pending", "reference_uncertain",
        "approved_on_uncertain_reference", "decisive_reference", "approval_reference",
        "rejection_reference", "decisive_prediction", "match", "false_approval",
        "false_rejection", "uncertain", "invalid", "runtime_error",
    )}
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


def analyze_development_packet(packet: dict, packet_sha256: str, *,
                               annotations: dict | None = None,
                               responses: dict | None = None) -> dict:
    """Reparse every raw response, retain all refusals and never infer missing gold."""
    plan = prepare_offline_plan(packet, packet_sha256)
    references, annotation_agreement = _references(packet, packet_sha256, annotations)
    provisional = {case["case_id"]: {
        "contract": case["provisional_contract_annotation"]["verdict"],
    } for case in packet["cases"]}
    historical = [{
        "case_id": case["case_id"], "model_id": case["historical_review"]["model_id"],
        "assertion": case["historical_review"].get("verdict", "invalid"),
        "contract": case["historical_review"]["task_contract_review"].get("verdict", "invalid"),
        "response_provenance": "historical_processed_verdict; raw response unavailable",
        "schema_reparsed": False,
    } for case in packet["cases"]]
    raw_rows = None
    if responses is not None:
        rows = _bound_rows(responses, RESPONSE_SCHEMA, packet, packet_sha256)
        if responses.get("review_protocol_sha256") != plan["review_protocol_sha256"]:
            raise ValueError("raw response review protocol hash mismatch; freeze a new plan")
        if not _nonempty(responses.get("model_id")):
            raise ValueError("raw response model identity is missing")
        raw_rows = []
        for case in packet["cases"]:
            row = rows[case["case_id"]]
            if row.get("contract_sha256") != SCIFACT_CONTRACT.sha256:
                raise ValueError("raw response contract hash mismatch")
            status = row.get("status")
            if status == "runtime_error":
                if row.get("response_content") is not None or not _nonempty(row.get("reason")):
                    raise ValueError("runtime error needs reason and no response content")
                parsed = None
                assertion = contract = "runtime_error"
            elif status == "completed" and isinstance(row.get("response_content"), str):
                parsed = parse_prompt_grounding_review(
                    row["response_content"], case["candidate_text"], SCIFACT_CONTRACT)
                assertion = parsed.get("assertion_review", {}).get("verdict", "invalid")
                contract = parsed.get("task_contract_review", {}).get("verdict", "invalid")
            else:
                raise ValueError("raw response status or content is invalid")
            raw_rows.append({"case_id": case["case_id"], "assertion": assertion,
                             "contract": contract, "parsed": parsed,
                             "model_id": responses["model_id"], "status": status})

    def comparisons(rows):
        return {
            "provisional_contract_comparison": _comparison(rows, provisional, "contract"),
            "blinded_development_reference_comparison": {
                dimension: _comparison(rows, references, dimension)
                for dimension in ("assertion", "contract")
            },
        }

    return {
        "schema": "instruction-grounding-development-audit-v1",
        "packet_sha256": packet_sha256, "offline_plan": plan,
        "model_calls": 0, "scope": ROLE,
        "candidate_efficacy_evidence": False, "independent_calibration_eligible": False,
        "semantic_calibration_complete": False,
        "reference_annotation_agreement": annotation_agreement,
        "historical_processed_reviews": {"cases": historical, **comparisons(historical)},
        "raw_response_replay": ({"cases": raw_rows, **comparisons(raw_rows)}
                                if raw_rows is not None else None),
        "limits": [
            "Provisional contract labels are agent analysis, not independent gold.",
            "Blinded annotations of exposed cases remain development reference evidence.",
            "Historical processed verdicts cannot establish raw-response schema validity.",
            "Observed development rates do not estimate population semantic accuracy.",
            "No benchmark task inputs or sealed data are read; no API requests are made.",
            "This audit and zero-request plan do not authorize a live experiment.",
        ],
    }
