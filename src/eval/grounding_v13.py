"""Evidence selection followed by a whole-proposition, citation-grounded judge."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from .grounding_v11 import _validate_selection
from .grounding_v12 import _direct_relations, claim_relation_messages_v12


def claim_relation_messages_v13(question: str, docs: Sequence[Dict[str, Any]],
                                evidence_selection: Any,
                                max_doc_chars: int = 6000) -> Tuple[str, str]:
    system, user = claim_relation_messages_v12(question, docs, max_doc_chars)
    hints = f"\n\nHigh-recall candidate sentence IDs by document and claim facet (hints only; inspect the complete indexed documents above):\n{evidence_selection}"
    return system, user + hints


def parse_v13_verdict(evidence_selection: Any, claim_relation: Any,
                      docs: Sequence[Dict[str, Any]], answer: str,
                      *, max_doc_chars: int = 6000) -> tuple[bool, str]:
    _, selection_reason = _validate_selection(evidence_selection, docs, max_doc_chars)
    if selection_reason != "valid":
        return False, selection_reason
    relation, reason = _direct_relations(claim_relation, docs, max_doc_chars)
    if reason not in {"valid", "conflicting_document_relations"}:
        return False, reason
    if relation == "insufficient":
        return False, "proposition_uncertain"
    answer_value = answer.strip().casefold() if isinstance(answer, str) else ""
    if answer_value not in {"yes", "no"}:
        return False, "proposition_requires_yes_no"
    expected = "supported" if answer_value == "yes" else "contradicted"
    if relation == expected:
        return True, "proposition_" + relation + "_whole_claim_with_indexed_evidence"
    return False, "proposition_supports_opposite"
