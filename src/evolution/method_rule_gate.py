"""Deterministic guard for a small class of unsafe statistical instructions."""

from __future__ import annotations

import re

METHOD_RULE_GATE_VERSION = "ci-zero-and-reported-absence-null-v2"


_CI = re.compile(r"(?i)(?:confidence\s+interval|\bci\b)")
_ZERO_EXCLUSION = re.compile(
    r"(?i)(?:does\s+not\s+include|does\s+not\s+contain|"
    r"exclude(?:s|d|ing)?|excluding)\s+(?:zero|0)\b")
_SIGNIFICANCE = re.compile(r"(?i)\b(?:significant|significance|p-value)\b")
_ZERO_NULL_EXCEPTIONS = re.compile(
    r"(?i)\b(?:risk\s+difference|mean\s+difference|absolute\s+difference|"
    r"difference\s+scale)\b"
)
_ABSENT_SIGNIFICANCE = re.compile(
    r"(?i)(?:\b(?:does\s+not|doesn't|fails?\s+to|is\s+not|isn't|not)\s+"
    r"(?:explicitly\s+)?(?:report|mention|state|specify|provide)\b"
    r".{0,80}\b(?:statistical\s+)?significance\b|"
    r"\b(?:no|without|absence\s+of|lack\s+of)\s+(?:a\s+)?"
    r"(?:reported\s+)?(?:statistical\s+)?significance\b|"
    r"\b(?:no|without|absence\s+of|lack\s+of)\s+(?:a\s+)?p[- ]?values?\b)"
)
_NULL_OUTCOME = re.compile(
    r"(?i)\b(?:no\s+significant\s+difference|no\s+difference|"
    r"not\s+significant|non[- ]significant)\b"
)
_NULL_ASSIGNMENT = re.compile(
    r"(?i)\b(?:choose|select|label|classify|answer|treat|consider|"
    r"assume|conclude|means?|implies?|indicates?)\b.{0,80}"
    r"\b(?:no\s+significant\s+difference|no\s+difference|"
    r"not\s+significant|non[- ]significant)\b"
)
_NEGATED_NULL_INFERENCE = re.compile(
    r"(?i)\b(?:do\s+not|don't|never|avoid)\s+(?:automatically\s+)?"
    r"(?:infer|assume|conclude|treat|classify|choose|select|label|map)"
    r"\b.{0,80}\b(?:no\s+significant\s+difference|no\s+difference|"
    r"not\s+significant|non[- ]significant)\b"
)


def validate_statistical_method_rules(content: str) -> dict:
    """Reject CI-zero instructions that do not name a zero-null statistic.

    The deterministic guard catches two unsupported equivalences: a CI that
    excludes zero universally implies significance, and silence about reported
    significance implies a null effect. Other method claims still require review.
    """
    text = str(content or "")
    sentences = re.split(r"(?<=[.!?;])\s+|\n+", text)
    hits = []
    for sentence in sentences:
        ci_zero_hit = (_CI.search(sentence) and _ZERO_EXCLUSION.search(sentence)
                       and _SIGNIFICANCE.search(sentence)
                       and not _ZERO_NULL_EXCEPTIONS.search(sentence))
        absence_null_hit = (
            _ABSENT_SIGNIFICANCE.search(sentence)
            and _NULL_OUTCOME.search(sentence)
            and _NULL_ASSIGNMENT.search(sentence)
            and not _NEGATED_NULL_INFERENCE.search(sentence)
        )
        if ci_zero_hit or absence_null_hit:
            hits.append(sentence.strip())
    return {
        "passed": not hits,
        "reason": ("no_unsupported_null_effect_rule" if not hits else
                   ("absence_of_significance_mapped_to_null_effect"
                    if any(_ABSENT_SIGNIFICANCE.search(hit) for hit in hits)
                    else "confidence_interval_null_value_unspecified")),
        "hits": hits,
    }
