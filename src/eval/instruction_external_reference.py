"""AI reference runner for the external-natural instruction proxy.

This intentionally produces diagnostic evidence only.  It reuses the shared
AI-job ledger and the existing two-reviewer/adjudication contract, while the
external packet itself remains ineligible for the generated-candidate gate.
"""
from __future__ import annotations

import json
import re
from typing import Any

from . import ai_jobs
from .annotation_policy import validate_actor
from .instruction_reviewer_external_source import (
    reference_payload, validate_packet,
)
from .instruction_review_calibration import _validate_completed_dimension
from ..util import sha256_obj


RUBRIC = (
    "Review the candidate strategy against only the supplied immutable task contract. "
    "Judge assertion presence separately from contract compatibility. Do not use source "
    "dataset labels, examples, or outside knowledge. assertion.verdict is claim_free, "
    "contains_claims, or uncertain; contract.verdict is compatible, incompatible, or uncertain. "
    "For contains_claims/incompatible, quote exact contiguous candidate text; incompatible "
    "also quotes exact contract text. For approvals, candidate_quotes must be empty. "
    "Preserve uncertainty rather than guessing. Return exactly one JSON object with assertion "
    "and contract fields."
)

REFERENCE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "external_instruction_reference",
        "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "required": ["assertion", "contract"],
            "properties": {
                "assertion": {"$ref": "#/$defs/assertion"},
                "contract": {"$ref": "#/$defs/contract"},
            },
            "$defs": {
                "assertion": {
                    "type": "object", "additionalProperties": False,
                    "required": ["verdict", "candidate_quotes", "reason"],
                    "properties": {
                        "verdict": {"type": "string", "enum": ["claim_free", "contains_claims", "uncertain"]},
                        "candidate_quotes": {"type": "array", "items": {"type": "string"}},
                        "reason": {"type": "string", "minLength": 1},
                    },
                },
                "contract": {
                    "type": "object", "additionalProperties": False,
                    "required": ["verdict", "candidate_quotes", "reason", "contract_quote"],
                    "properties": {
                        "verdict": {"type": "string", "enum": ["compatible", "incompatible", "uncertain"]},
                        "candidate_quotes": {"type": "array", "items": {"type": "string"}},
                        "reason": {"type": "string", "minLength": 1},
                        "contract_quote": {"type": ["string", "null"]},
                    },
                },
            },
        },
    },
}


def _decode(raw: Any) -> dict[str, Any]:
    text = raw.strip() if isinstance(raw, str) else raw
    if isinstance(text, str):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.I | re.S)
        text = match.group(1) if match else text
        text = json.loads(text)
    if not isinstance(text, dict) or set(text) != {"assertion", "contract"}:
        raise ValueError("external reference requires assertion and contract")
    return text


def _validated(raw: Any, case: dict[str, Any], packet: dict[str, Any], reviewer: str) -> dict[str, Any]:
    value = _decode(raw)
    candidate = case["strategy_text"]
    contract = packet["contract"]["text"]
    result = {}
    for dimension in ("assertion", "contract"):
        annotation = dict(value[dimension])
        annotation["status"] = "completed"
        annotation["reviewer"] = reviewer
        annotation["blind_to_model_reviews"] = True
        _validate_completed_dimension(annotation, candidate, dimension=dimension,
                                      contract_text=contract)
        result[dimension] = {
            "verdict": value[dimension]["verdict"],
            "candidate_quotes": list(value[dimension]["candidate_quotes"]),
            "reason": value[dimension]["reason"],
            **({"contract_quote": value[dimension].get("contract_quote")}
               if dimension == "contract" else {}),
        }
    return result


def _validated_with_safe_quote_repair(raw: Any, case: dict[str, Any],
                                      packet: dict[str, Any], reviewer: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Apply evidence-only normalization for an explicit rejection.

    Small models can emit a rejection verdict and rationale but omit the
    required verbatim evidence fields.  Supplying the complete immutable
    candidate/contract text preserves the model's verdict and only completes
    the audit evidence.  Verdict changes are never repaired here.  Approval
    evidence is cleared to its contractually required empty form, while an
    invalid rejection quote is replaced only by complete trusted text and
    recorded as a normalization event.
    """
    value = _decode(raw)
    repairs: list[dict[str, str]] = []
    candidate = case["strategy_text"]
    contract = packet["contract"]["text"]
    for dimension in ("assertion", "contract"):
        annotation = value[dimension]
        if not isinstance(annotation.get("reason"), str) or not annotation["reason"].strip():
            value[dimension] = {
                **annotation,
                "reason": f"model_reason_omitted_{dimension}; verdict retained",
            }
            repairs.append({"dimension": dimension,
                            "repair": "fill_structural_reason_marker"})
    for dimension, rejection in (("assertion", "contains_claims"),
                                 ("contract", "incompatible")):
        annotation = value[dimension]
        approval = "claim_free" if dimension == "assertion" else "compatible"
        if annotation.get("verdict") == approval and annotation.get("candidate_quotes"):
            value[dimension] = {**annotation, "candidate_quotes": []}
            annotation = value[dimension]
            repairs.append({"dimension": dimension,
                            "repair": "clear_approval_candidate_quotes"})
        if annotation.get("verdict") == rejection and not annotation.get("candidate_quotes"):
            value[dimension] = {**annotation, "candidate_quotes": [candidate]}
            annotation = value[dimension]
            repairs.append({"dimension": dimension,
                            "repair": "candidate_full_text_for_missing_rejection_quote"})
        if (annotation.get("verdict") == rejection
                and isinstance(annotation.get("candidate_quotes"), list)
                and annotation.get("candidate_quotes")
                and not all(isinstance(quote, str) and quote.strip() and quote in candidate
                            for quote in annotation["candidate_quotes"])):
            value[dimension] = {**value[dimension], "candidate_quotes": [candidate]}
            annotation = value[dimension]
            repairs.append({"dimension": dimension,
                            "repair": "candidate_full_text_for_invalid_rejection_quote"})
        if (dimension == "contract" and annotation.get("verdict") in {"compatible", "uncertain"}
                and annotation.get("contract_quote") is not None):
            value[dimension] = {**value[dimension], "contract_quote": None}
            annotation = value[dimension]
            repairs.append({"dimension": dimension,
                            "repair": "clear_non_rejection_contract_quote"})
        if (dimension == "contract" and annotation.get("verdict") == rejection
                and not annotation.get("contract_quote")
                and isinstance(annotation.get("contract_quotes"), list)):
            exact = next((quote for quote in annotation["contract_quotes"]
                          if isinstance(quote, str) and quote.strip() and quote in contract), None)
            value[dimension] = {**value[dimension],
                                "contract_quote": exact or contract}
            annotation = value[dimension]
            repairs.append({"dimension": dimension,
                            "repair": "map_contract_quotes_to_contract_quote" if exact
                            else "contract_full_text_for_invalid_rejection_quote"})
        if (dimension == "contract" and annotation.get("verdict") == rejection
                and (not isinstance(annotation.get("contract_quote"), str)
                     or annotation.get("contract_quote") not in contract)):
            value[dimension] = {**value[dimension], "contract_quote": contract}
            repairs.append({"dimension": dimension,
                            "repair": "contract_full_text_for_invalid_rejection_quote"})
    return _validated(value, case, packet, reviewer), repairs


def reference_plan(packet: dict[str, Any], cfg: dict[str, Any], *,
                   max_cost_usd: float = 0.0, zero_price_only: bool = True) -> dict[str, Any]:
    validate_packet(packet)
    # Two reviewers + two bounded corrections per reviewer + one adjudicator +
    # two bounded adjudicator corrections per case.  The correction slots are
    # reserved before dispatch, so malformed JSON/quote combinations cannot
    # consume an unplanned request or leave an ambiguous partial reference.
    max_calls = 8 * len(packet["cases"])
    return ai_jobs.build_plan("instruction_reference", packet, cfg,
        max_calls=max_calls, tokens=900, max_input_chars=20000,
        max_cost_usd=max_cost_usd, response_format=REFERENCE_FORMAT,
        zero_price_only=zero_price_only)


def reference_execute(packet: dict[str, Any], cfg: dict[str, Any], plan: dict[str, Any], *, provider=None) -> dict[str, Any]:
    validate_packet(packet)
    expected = reference_plan(packet, cfg, max_cost_usd=plan["max_cost_usd"],
                              zero_price_only=plan.get("zero_price_only", False))
    if expected != plan:
        raise ValueError("external reference frozen plan changed")

    def _call_validated(call, role: str, case: dict[str, Any], *, reviewer: str,
                        correction_suffix: str = ""):
        payload = reference_payload(packet, case)
        raw, provenance = call(role, payload, RUBRIC + correction_suffix)
        try:
            return _validated(raw, case, packet, reviewer), provenance, None
        except (TypeError, ValueError) as error:
            try:
                dimensions, repairs = _validated_with_safe_quote_repair(
                    raw, case, packet, reviewer)
                return dimensions, provenance, [{"reason": str(error), "role": role,
                                                 "attempt": 0, **repair}
                                                for repair in repairs]
            except (TypeError, ValueError):
                pass
            correction_notes = []
            for attempt in range(2):
                correction_payload = {
                    **payload, "invalid_output": raw,
                    "validation_error": str(error),
                    "correction_attempt": attempt + 1,
                }
                corrected, correction_provenance = call(
                    "instruction_reference_correction", correction_payload,
                    RUBRIC + " Correct the invalid draft. A compatible/claim_free verdict "
                    "MUST have candidate_quotes=[], while incompatible/contains_claims "
                    "MUST have at least one exact candidate quote. Only incompatible may "
                    "have a non-null contract_quote. Return only the required JSON object.")
                try:
                    return (_validated(corrected, case, packet, reviewer), correction_provenance,
                            correction_notes + [{"reason": str(error), "role": role,
                                                 "attempt": attempt + 1}])
                except (TypeError, ValueError) as next_error:
                    try:
                        dimensions, repairs = _validated_with_safe_quote_repair(
                            corrected, case, packet, reviewer)
                        return (dimensions, correction_provenance,
                                correction_notes + [{"reason": str(next_error),
                                                     "role": "instruction_reference_correction",
                                                     "attempt": attempt + 1, **repair}
                                                    for repair in repairs])
                    except (TypeError, ValueError):
                        pass
                    correction_notes.append({"reason": str(next_error),
                                             "role": "instruction_reference_correction",
                                             "attempt": attempt + 1})
                    raw, error = corrected, next_error
            raise error

    def worker(call):
        reviews: dict[str, dict[str, Any]] = {}
        for reviewer in ("reviewer_a", "reviewer_b"):
            reviews[reviewer] = {}
            for case in packet["cases"]:
                dimensions, provenance, correction = _call_validated(
                    call, "instruction_reference", case, reviewer=reviewer)
                actor = validate_actor(reviewer, "independent_ai", provenance)
                reviews[reviewer][case["case_id"]] = {
                    "dimensions": dimensions, "actor": actor,
                    **({"correction": correction} if correction else {})}
        resolved = []
        for case in packet["cases"]:
            pair = [reviews[name][case["case_id"]] for name in ("reviewer_a", "reviewer_b")]
            disputes = [dimension for dimension in ("assertion", "contract")
                        if pair[0]["dimensions"][dimension]["verdict"] !=
                        pair[1]["dimensions"][dimension]["verdict"]]
            adjudication = None
            if disputes:
                payload = {
                    **reference_payload(packet, case),
                    "original_reviews": [row["dimensions"] for row in pair],
                }
                raw, provenance = call("instruction_adjudicator", payload,
                    RUBRIC + " Two references disagree; preserve uncertain when unresolved.")
                try:
                    dimensions = _validated(raw, case, packet, "adjudicator")
                    correction = None
                except (TypeError, ValueError) as error:
                    try:
                        dimensions, repairs = _validated_with_safe_quote_repair(
                            raw, case, packet, "adjudicator")
                        correction = [{"reason": str(error),
                                       "role": "instruction_adjudicator",
                                       "attempt": 0, **repair}
                                      for repair in repairs]
                    except (TypeError, ValueError):
                        correction = []
                        for attempt in range(2):
                            corrected, provenance = call("instruction_adjudicator_correction", {
                                **payload, "invalid_output": raw,
                                "validation_error": str(error), "correction_attempt": attempt + 1},
                                RUBRIC + " Correct the adjudication. Approval verdicts require "
                                "candidate_quotes=[], rejection verdicts require exact candidate "
                                "quotes; preserve uncertain when unresolved.")
                            try:
                                dimensions = _validated(corrected, case, packet, "adjudicator")
                                correction.append({"reason": str(error),
                                                   "role": "instruction_adjudicator",
                                                   "attempt": attempt + 1})
                                break
                            except (TypeError, ValueError) as next_error:
                                try:
                                    dimensions, repairs = _validated_with_safe_quote_repair(
                                        corrected, case, packet, "adjudicator")
                                    correction.extend([{"reason": str(next_error),
                                                        "role": "instruction_adjudicator_correction",
                                                        "attempt": attempt + 1, **repair}
                                                       for repair in repairs])
                                    break
                                except (TypeError, ValueError):
                                    correction.append({"reason": str(next_error),
                                                       "role": "instruction_adjudicator_correction",
                                                       "attempt": attempt + 1})
                                    raw, error = corrected, next_error
                        else:
                            raise error
                adjudication = {"dimensions": dimensions,
                                "provenance": {**provenance, "blind": False,
                                               "other_review_outputs_consulted": True,
                                               **({"correction": correction} if correction else {})}}
            verdicts = {dimension: (
                adjudication["dimensions"][dimension]["verdict"] if adjudication and dimension in disputes
                else pair[0]["dimensions"][dimension]["verdict"])
                for dimension in ("assertion", "contract")}
            resolved.append({"case_id": case["case_id"], "originals": pair,
                             "disputed_dimensions": disputes, "adjudication": adjudication,
                             "verdicts": verdicts})
        return {"schema": "external-instruction-reference-v1",
                "packet_sha256": packet["packet_sha256"], "annotation_origin": "ai",
                "production_gate_eligible": False,
                "reference_rubric_sha256": sha256_obj(RUBRIC), "cases": resolved}

    return ai_jobs.execute(plan, packet, cfg, worker, provider=provider)


__all__ = ["RUBRIC", "REFERENCE_FORMAT", "reference_plan", "reference_execute"]
