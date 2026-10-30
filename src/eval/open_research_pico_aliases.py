"""Deterministic, pre-registered PICO terminology expansion for G09.

The original G09 source gate intentionally required literal matches for the
four frozen PICO dimensions.  That gate is safe but rejects explicit lexical
variants such as ``postmenopausal women`` for an adult-women slot.  This module
keeps the same fail-closed shape while allowing a *plan-bound* alias table.
Aliases are supplied by the frozen plan; no stemming, embeddings, model
judgment, or inferred synonym is used at runtime.
"""
from __future__ import annotations

import re
from typing import Mapping, Sequence

PICO_ALIAS_POLICY_VERSION = "g09-pico-explicit-alias-v1"
PICO_DIMENSIONS = ("condition", "population", "intervention", "outcome")
_FIELDS = {name: f"{name}_terms" for name in PICO_DIMENSIONS}


def _normalised(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _terms(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{label} terms must be strings")
        text = " ".join(item.split())
        if not text or len(text) > 160 or any(ord(c) < 32 or ord(c) == 127 for c in text):
            raise ValueError(f"invalid {label} term")
        normalised = _normalised(text)
        if not normalised:
            raise ValueError(f"{label} contains an empty term")
        result.append(normalised)
    return tuple(dict.fromkeys(result))


def validate_alias_table(aliases: Mapping[str, object]) -> dict[str, list[str]]:
    """Validate and canonicalise one topic's explicit alias table."""
    if not isinstance(aliases, Mapping) or set(aliases) != set(PICO_DIMENSIONS):
        raise ValueError("PICO alias table must contain exactly four dimensions")
    result: dict[str, list[str]] = {}
    for dimension in PICO_DIMENSIONS:
        result[dimension] = list(_terms(aliases[dimension], label=f"{dimension} aliases"))
    return result


def assess_topic_pico_alias_relevance(record: Mapping[str, object], aliases: Mapping[str, object]) -> dict:
    """Apply a deterministic title/abstract alias gate.

    A source is eligible only when at least one pre-registered alias appears
    in the normalised title or abstract for every PICO dimension.  The output
    retains matched aliases and missing dimensions for audit receipts.
    """
    if not isinstance(record, Mapping):
        raise ValueError("source record must be an object")
    table = validate_alias_table(aliases)
    title = record.get("title") if isinstance(record.get("title"), str) else ""
    abstract = record.get("abstractText") if isinstance(record.get("abstractText"), str) else ""
    text = _normalised(f"{title} {abstract}")
    padded = f" {text} "
    matched: dict[str, list[str]] = {}
    missing: list[str] = []
    for dimension in PICO_DIMENSIONS:
        values = [term for term in table[dimension] if f" {term} " in padded]
        matched[dimension] = values
        if not values:
            missing.append(dimension)
    return {
        "policy_version": PICO_ALIAS_POLICY_VERSION,
        "eligible": not missing,
        "matched_aliases": matched,
        "missing_dimensions": missing,
    }


def aliases_from_topic_spec(topic_spec: Mapping[str, object], aliases: Mapping[str, object]) -> dict:
    """Return a plan-safe alias payload bound to one topic's identity."""
    if not isinstance(topic_spec, Mapping) or not isinstance(aliases, Mapping):
        raise ValueError("topic spec and aliases must be objects")
    topic_id = topic_spec.get("topic_id")
    if not isinstance(topic_id, str) or not topic_id:
        raise ValueError("topic spec requires a topic_id")
    return {"topic_id": topic_id, "policy_version": PICO_ALIAS_POLICY_VERSION,
            "aliases": validate_alias_table(aliases)}
