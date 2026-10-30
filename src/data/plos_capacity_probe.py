"""Deterministic metadata capacity indicators; never assign semantic labels."""

from __future__ import annotations

import ast
import re
from collections import Counter
from typing import Any


NCT_PATTERN = re.compile(r"(?<![A-Z0-9])NCT[0-9]{8}(?![0-9])", re.I)
P_VALUE_PATTERN = re.compile(
    r"(?<![A-Z])p\s*(?:=|<|>|≤|≥)\s*(?:0?\.\d+|1(?:\.0+)?)", re.I)
CI_PATTERN = re.compile(r"95\s*%\s*(?:CI|confidence interval)|confidence interval", re.I)
PRIMARY_PATTERN = re.compile(r"primary\s+(?:outcome|endpoint)", re.I)


def normalize_abstract_text(value: Any) -> str:
    """Read normalized arrays and old stringified-list snapshots consistently."""
    if isinstance(value, list):
        return " ".join(item.strip() for item in value
                         if isinstance(item, str) and item.strip())
    if not isinstance(value, str):
        return ""
    text = value.strip()
    # Earlier capture code accidentally stored ``str(list[str])``. Parse only a
    # literal list of strings; arbitrary/malformed values remain plain text.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return " ".join(item.strip() for item in parsed if item.strip())
    return text


def summarize_plos_capacity(
    source_rows: list[dict[str, Any]], eligible_dois: set[str],
) -> dict[str, Any]:
    rows = [row for row in source_rows
            if str(row.get("doi") or "").casefold() in eligible_dois]
    nct_matches: dict[str, set[str]] = {}
    counts: Counter[str] = Counter()
    for row in rows:
        abstract = normalize_abstract_text(row.get("abstract"))
        ncts = {value.upper() for value in NCT_PATTERN.findall(abstract)}
        if ncts:
            nct_matches[str(row.get("doi") or "").casefold()] = ncts
        has_p = bool(P_VALUE_PATTERN.search(abstract))
        has_ci = bool(CI_PATTERN.search(abstract))
        has_primary = bool(PRIMARY_PATTERN.search(abstract))
        counts["abstract_with_nct"] += bool(ncts)
        counts["abstract_with_p_value_pattern"] += has_p
        counts["abstract_with_ci_pattern"] += has_ci
        counts["abstract_mentions_primary_outcome"] += has_primary
        counts["abstract_with_nct_and_p_value_pattern"] += bool(ncts) and has_p
    return {
        "record_count": len(rows),
        "unique_nct_count": len({nct for ncts in nct_matches.values() for nct in ncts}),
        "indicator_counts": dict(sorted(counts.items())),
        "nct_by_doi": {doi: sorted(ncts) for doi, ncts in sorted(nct_matches.items())},
        "these_are_metadata_indicators_not_trial_identity_or_label_verification": True,
        "task_labels_generated": False,
    }
