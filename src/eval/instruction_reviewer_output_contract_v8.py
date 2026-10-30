"""Reference-only v8 output contract for claim-level reviewer evidence.

This module is intentionally isolated from production candidate gates. It
validates model output against host-owned source spans and derives verdicts.
"""
from __future__ import annotations

import json
from typing import Any

from .instruction_reviewer_output_contract_v7 import CLAIM_KINDS, ISSUE_KINDS


SPAN_LABELS = ("no_claim", "contains_claims", "uncertain")
CONTRACT_LABELS = ("no_conflict", "uncertain")
MAX_CLAIMS = 64
MIN_QUOTE_CHARS = 8
DELIMITER = "|"


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties,
            "required": required, "additionalProperties": False}


def _validate_ids(ids: list[str] | None, name: str, *, required: bool) -> list[str]:
    if ids is None and not required:
        return []
    if not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError(f"{name}_ids_must_be_nonempty_strings")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate_{name}_id")
    if any(DELIMITER in value for value in ids):
        raise ValueError(f"{name}_id_contains_reserved_delimiter")
    if required and not ids:
        raise ValueError(f"{name}_ids_required")
    return list(ids)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _span_map(spans: list[dict[str, str]], source: str) -> dict[str, str]:
    if not isinstance(spans, list) or not spans:
        raise ValueError(f"{source}_span_inventory_required")
    result: dict[str, str] = {}
    for span in spans:
        if (not isinstance(span, dict) or set(span) != {"id", "text"}
                or not isinstance(span["id"], str) or not span["id"]
                or DELIMITER in span["id"] or not isinstance(span["text"], str)
                or not span["text"]):
            raise ValueError(f"invalid_{source}_span_inventory")
        if span["id"] in result:
            raise ValueError(f"duplicate_{source}_span_id")
        result[span["id"]] = span["text"]
    return result


def _contract_values(contract_ids: list[str]) -> list[str]:
    return [*CONTRACT_LABELS, *[
        f"direct_conflict{DELIMITER}{contract_id}{DELIMITER}{kind}"
        for contract_id in contract_ids for kind in ISSUE_KINDS
    ]]


def build_response_format_v8(candidate_ids: list[str],
                             contract_ids: list[str] | None = None) -> dict[str, Any]:
    """Build a strict schema, with contract dimension only when supplied."""
    candidates = _validate_ids(candidate_ids, "candidate", required=True)
    contracts = _validate_ids(contract_ids, "contract", required=False)
    if contracts and not candidates:
        raise ValueError("candidate_ids_required")
    labels = _object({cid: {"type": "string", "enum": list(SPAN_LABELS)}
                      for cid in candidates}, candidates)
    claims = {"type": "array", "maxItems": MAX_CLAIMS,
              "items": _object({
                  "span_id": {"type": "string", "enum": candidates},
                  "quote": {"type": "string", "minLength": MIN_QUOTE_CHARS},
                  "kind": {"type": "string", "enum": list(CLAIM_KINDS)},
              }, ["span_id", "quote", "kind"])}
    assertion = _object({"span_labels": labels, "claims": claims},
                        ["span_labels", "claims"])
    properties: dict[str, Any] = {"assertion": assertion}
    required = ["assertion"]
    if contracts:
        contract_values = _contract_values(contracts)
        properties["contract"] = _object({cid: {"type": "string", "enum": contract_values}
                                            for cid in candidates}, candidates)
        required.append("contract")
    schema = _object(properties, required)
    response_format = {"type": "json_schema", "json_schema": {
        "name": "instruction_reviewer_output_contract_v8",
        "strict": True,
        "schema": schema,
    }}
    if len(json.dumps(response_format, ensure_ascii=False)) > 120000:
        raise ValueError("schema_total_string_limit_exceeded")
    return response_format


def _decode_contract(value: Any, contracts: dict[str, str]
                     ) -> tuple[str, str | None, str | None]:
    if not isinstance(value, str):
        raise ValueError("contract_value_must_be_string")
    if value in CONTRACT_LABELS:
        return value, None, None
    parts = value.split(DELIMITER)
    if (len(parts) == 3 and parts[0] == "direct_conflict"
            and parts[1] in contracts and parts[2] in ISSUE_KINDS):
        return "direct_conflict", parts[1], parts[2]
    raise ValueError("invalid_contract_annotation")


def validate_output_v8(raw: str, candidate_spans: list[dict[str, str]],
                       contract_spans: list[dict[str, str]] | None = None
                       ) -> dict[str, Any]:
    """Validate complete span labels and unique, exact claim quotes.

    Raises ``ValueError`` on every malformed or ambiguous result, so callers
    can fail closed without treating parser failures as negative evidence.
    """
    candidate = _span_map(candidate_spans, "candidate")
    contracts = {} if contract_spans is None else _span_map(contract_spans, "contract")
    if not isinstance(raw, str):
        raise ValueError("json_text_required")
    try:
        parsed = json.loads(raw, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid_json") from exc
    except ValueError:
        raise
    except TypeError as exc:
        raise ValueError("invalid_json") from exc
    expected_top = {"assertion", "contract"} if contract_spans is not None else {"assertion"}
    if not isinstance(parsed, dict) or set(parsed) != expected_top:
        raise ValueError("top_level_schema_mismatch")
    assertion = parsed["assertion"]
    if not isinstance(assertion, dict) or set(assertion) != {"span_labels", "claims"}:
        raise ValueError("assertion_schema_mismatch")
    labels = assertion["span_labels"]
    if not isinstance(labels, dict):
        raise ValueError("span_labels_must_be_object")
    if set(labels) != set(candidate):
        if set(candidate) - set(labels):
            raise ValueError("span_labels_candidate_coverage_missing")
        raise ValueError("span_labels_candidate_coverage_extra")
    if any(label not in SPAN_LABELS for label in labels.values()):
        raise ValueError("invalid_span_label")
    claims = assertion["claims"]
    if not isinstance(claims, list):
        raise ValueError("claims_must_be_array")
    if len(claims) > MAX_CLAIMS:
        raise ValueError("claims_limit_exceeded")
    seen_quotes: set[tuple[str, str]] = set()
    claims_by_span: dict[str, list[dict[str, str]]] = {cid: [] for cid in candidate}
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"span_id", "quote", "kind"}:
            raise ValueError("invalid_claim_shape")
        span_id, quote, kind = claim["span_id"], claim["quote"], claim["kind"]
        if span_id not in candidate:
            raise ValueError("claim_span_id_unknown")
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError("claim_quote_required")
        if len(quote.strip()) < MIN_QUOTE_CHARS:
            raise ValueError("quote_too_short_to_express_proposition")
        if kind not in CLAIM_KINDS:
            raise ValueError("invalid_claim_kind")
        occurrence_count = candidate[span_id].count(quote)
        if occurrence_count == 0:
            raise ValueError("quote_not_found")
        if occurrence_count != 1:
            raise ValueError("quote_not_unique")
        quote_key = (span_id, quote)
        if quote_key in seen_quotes:
            raise ValueError("duplicate_claim_quote")
        seen_quotes.add(quote_key)
        claims_by_span[span_id].append({"span_id": span_id, "quote": quote, "kind": kind})
    for span_id, label in labels.items():
        count = len(claims_by_span[span_id])
        if label == "contains_claims" and count == 0:
            raise ValueError("contains_claims_requires_quote")
        if label == "no_claim" and count:
            raise ValueError("no_claim_has_quote")
        if label == "uncertain" and count:
            raise ValueError("uncertain_has_quote")

    decoded_contract: dict[str, tuple[str, str | None, str | None]] = {}
    if contract_spans is not None:
        mapping = parsed["contract"]
        if not isinstance(mapping, dict):
            raise ValueError("contract_must_be_object")
        if set(mapping) != set(candidate):
            if set(candidate) - set(mapping):
                raise ValueError("contract_candidate_coverage_missing")
            raise ValueError("contract_candidate_coverage_extra")
        decoded_contract = {cid: _decode_contract(value, contracts)
                            for cid, value in mapping.items()}
    return {"span_labels": labels, "claims": claims_by_span,
            "contract": decoded_contract}


def normalize_output_v8(raw: str, candidate_spans: list[dict[str, str]],
                        contract_spans: list[dict[str, str]] | None = None
                        ) -> dict[str, Any]:
    candidate = _span_map(candidate_spans, "candidate")
    contracts = {} if contract_spans is None else _span_map(contract_spans, "contract")
    decoded = validate_output_v8(raw, candidate_spans, contract_spans)
    claims = [{"candidate": {"id": span_id, "text": candidate[span_id]},
               "quote": claim["quote"], "kind": claim["kind"]}
              for span_id, span_claims in decoded["claims"].items()
              for claim in span_claims]
    uncertain = [{"id": span_id, "text": candidate[span_id]}
                 for span_id, label in decoded["span_labels"].items()
                 if label == "uncertain"]
    assertion_verdict = ("contains_claims" if claims else "uncertain" if uncertain
                         else "claim_free")
    result: dict[str, Any] = {
        "assertion": {
            "verdict": assertion_verdict,
            "claims": claims,
            "uncertain": uncertain,
            "verdict_reason_code": {
                "claim_free": "all_candidate_spans_no_claim",
                "contains_claims": "host_verified_claim_quote_found",
                "uncertain": "uncertain_candidate_span_without_claim_quote",
            }[assertion_verdict],
        },
        "source_coverage": {
            "assertion": decoded["span_labels"],
            "claim_quotes": decoded["claims"],
        },
    }
    if contract_spans is not None:
        issues = [{"candidate": {"id": cid, "text": candidate[cid]},
                   "contract": {"id": contract_id, "text": contracts[contract_id]},
                   "kind": kind}
                  for cid, (status, contract_id, kind) in decoded["contract"].items()
                  if status == "direct_conflict"]
        contract_uncertain = [{"id": cid, "text": candidate[cid]}
                              for cid, (status, _, _) in decoded["contract"].items()
                              if status == "uncertain"]
        contract_verdict = "incompatible" if issues else "uncertain" if contract_uncertain else "compatible"
        result["contract"] = {
            "verdict": contract_verdict,
            "issues": issues[:1],
            "uncertain": contract_uncertain,
            "verdict_reason_code": {
                "compatible": "no_explicit_direct_contract_conflict",
                "incompatible": "explicit_direct_contract_conflict_found",
                "uncertain": "direct_conflict_boundary_unclear",
            }[contract_verdict],
        }
        result["all_direct_conflicts"] = issues
        result["source_coverage"]["contract"] = decoded["contract"]
    return result
