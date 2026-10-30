"""Frozen task-decision contracts used to review evolved instructions."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

from ..contracts.types import content_sha256


@dataclass(frozen=True)
class TaskContract:
    contract_id: str
    text: str

    @property
    def sha256(self) -> str:
        return content_sha256(self.text)


SCIFACT_CONTRACT = TaskContract(
    contract_id="scifact_claim_verification_v1",
    text=(
        "Given a scientific claim and the cited abstracts supplied in the task, "
        "decide whether those abstracts support the claim, contradict it, or do "
        "not provide enough information. Use the task evidence and claim relation "
        "as the decision basis. The task does not ask for a separate assessment "
        "of clinical relevance, experimental rigor, study quality, or statistical "
        "significance as an independent decision criterion; such a criterion is "
        "appropriate only when it is part of the claim being verified. Preserve "
        "the exact labels SUPPORT, CONTRADICT, and NOT_ENOUGH_INFO."
    ),
)

EVIDENCE_INFERENCE_CONTRACT = TaskContract(
    contract_id="evidence_inference_effect_direction_v1",
    text=(
        "Use only the task's question, answer options, and supplied material. "
        "Identify the specified intervention, comparator, and outcome, then "
        "judge the reported effect of the intervention relative to the comparator "
        "on that outcome. Choose exactly one listed label: significantly increased, "
        "significantly decreased, or no significant difference. Match both the "
        "direction and significance stated in the supplied material; do not infer "
        "statistical significance from direction or magnitude alone. If the supplied "
        "material does not establish direction or significance, do not invent one; "
        "choose the listed label best supported by the material."
    ),
)

HEALTHVER_CONTRACT = TaskContract(
    contract_id="healthver_claim_abstract_entailment_v1",
    text=(
        "Use only the supplied biomedical claim, answer options, and its paired "
        "abstract. Decide whether the abstract explicitly supports the claim, "
        "explicitly refutes it, or establishes neither. Preserve the exact labels "
        "SUPPORT, CONTRADICT, and NOT_ENOUGH_INFO. Do not treat missing details "
        "or related background information as evidence for or against the claim, "
        "and do not add facts from outside the supplied abstract."
    ),
)

HEALTHFC_CONTRACT = TaskContract(
    contract_id="healthfc_fixed_evidence_claim_verification_v1",
    text=(
        "Use only the supplied clinical claim and fixed evidence sentences. "
        "Decide whether the supplied evidence supports the claim, refutes it, "
        "or is insufficient to decide. Choose exactly one listed label: "
        "SUPPORTED, NEI, or REFUTED. NEI is the correct choice when the material "
        "is insufficient, unclear, or does not directly address the claim; do not "
        "infer a clinical conclusion beyond the fixed evidence."
    ),
)

CLINIFACT_CONTRACT = TaskContract(
    contract_id="clinifact_trial_result_claim_evidence_v4",
    text=(
        "Use only the supplied clinical research claim and its paired publication "
        "abstract. Apply these program-owned, immutable CliniFact label "
        "definitions:\n"
        "EVIDENCE: the abstract reports a statistically significant result that "
        "supports the claim.\n"
        "INCONCLUSIVE: the abstract reports a nonsignificant or otherwise "
        "inconclusive result for the claim.\n"
        "NEI: the abstract is background or does not report information that "
        "addresses the claim.\n"
        "Keep INCONCLUSIVE distinct from NEI. Do not reinterpret a "
        "nonsignificant result as evidence that the opposite claim is true. "
        "Choose exactly one listed label: EVIDENCE, INCONCLUSIVE, or NEI. "
        "The runtime accepts one program-constrained integer option index. "
        "It rejects malformed or missing answers and never infers an answer "
        "from surrounding prose."
    ),
)
CLINIFACT_MIPRO_STRATEGY_SEED = (
    "Compare the supplied clinical research claim with the reported results "
    "in its paired publication abstract and classify only the relationship "
    "supported by those results."
)


def compose_task_instruction(strategy: str, contract: TaskContract | None) -> str:
    """Compose runtime instructions with an immutable, program-owned contract.

    The optimizer may rewrite ``strategy`` arbitrarily. Required label meanings
    and output protocol are appended by code and never depend on the proposal
    model reproducing particular wording.
    """
    if contract is None:
        text = str(strategy or "")
        if not text.strip():
            raise ValueError("task strategy cannot be empty")
        return text
    text = str(strategy or "").strip()
    if not text:
        raise ValueError("task strategy cannot be empty")
    marker = (f"\n\n--- IMMUTABLE TASK CONTRACT "
              f"{contract.contract_id} sha256={contract.sha256} ---\n")
    suffix = marker + contract.text.strip()
    if text.endswith(suffix):
        return text
    return text + suffix


def has_immutable_task_contract(prompt: str, contract: TaskContract | None) -> bool:
    """Check the exact code-owned contract suffix without parsing model prose."""
    if contract is None:
        return True
    marker = (f"\n\n--- IMMUTABLE TASK CONTRACT "
              f"{contract.contract_id} sha256={contract.sha256} ---\n")
    return str(prompt or "").endswith(marker + contract.text.strip())


def task_instruction_strategy(prompt: str, contract: TaskContract | None) -> str:
    """Recover editable strategy from a composed instruction without parsing prose.

    A contract suffix is removed only when its exact code-owned marker and text
    match. A marker with altered or partial contract content is rejected rather
    than treated as ordinary model-authored strategy.
    """
    text = str(prompt or "").strip()
    if contract is None:
        if not text:
            raise ValueError("task strategy cannot be empty")
        return text
    marker = (f"\n\n--- IMMUTABLE TASK CONTRACT "
              f"{contract.contract_id} sha256={contract.sha256} ---\n")
    suffix = marker + contract.text.strip()
    if text.endswith(suffix):
        text = text[:-len(suffix)].strip()
    elif "--- IMMUTABLE TASK CONTRACT " in text:
        raise ValueError("task instruction contains a malformed immutable contract")
    if not text:
        raise ValueError("task strategy cannot be empty")
    return text

PUBMEDQA_THREEWAY_CONTRACT = TaskContract(
    contract_id="pubmedqa_abstract_question_threeway_v1",
    text=(
        "Use only the supplied question, answer options, and PubMed abstract. "
        "Preserve PubMedQA's three-way label meanings: yes when the reported "
        "experiments or results support the answer in the context of the question; "
        "no when they do not; maybe when results across conditions, interventions, "
        "or observations give mixed answers. Do not replace maybe with a general "
        "insufficient-evidence label. Choose exactly one listed label: yes, no, "
        "or maybe."
    ),
)

BIOASQ_YESNO_CONTRACT = TaskContract(
    contract_id="bioasq_yesno_binary_answer_v1",
    text=(
        "Use only the supplied BioASQ question, the two listed answer options, "
        "and the task-provided scientific material. Preserve the source-native yes/no "
        "semantics: choose yes when the material supports the proposition in the "
        "question and choose no when it does not. Choose exactly one listed label: "
        "yes or no. This task has no abstention, maybe, insufficient, or "
        "unanswerable label; unresolved wording must still be mapped to the "
        "better-supported listed option. Never emit a label absent from the options."
    ),
)

_CONTRACTS = {
    "scifact_claim_verification": SCIFACT_CONTRACT,
    "evidence_inference_v2_abstract": EVIDENCE_INFERENCE_CONTRACT,
    "healthver_entailment_task_input": HEALTHVER_CONTRACT,
    "healthfc_fixed_evidence": HEALTHFC_CONTRACT,
    "clinifact_fixed_abstract": CLINIFACT_CONTRACT,
    "pubmedqa_threeway_probe": PUBMEDQA_THREEWAY_CONTRACT,
    "bioasq_yesno_fresh_20261019": BIOASQ_YESNO_CONTRACT,
}
TASK_CONTRACT_RULE_VERSION = "task-decision-scope-guard-v2"

_SCIFACT_METHOD_TOPICS = re.compile(
    r"\b(?:statistical\s+(?:rigou?r|significance|significance\s+threshold)"
    r"|biological\s+mechanisms?"
    r"|clinical\s+relevance"
    r"|experimental\s+(?:rigou?r|quality)"
    r"|study\s+types?"
    r"|(?:study|methodological)\s+(?:quality|rigou?r|methodology|methods?|design)"
    r"|animal\s+models?"
    r"|randomi[sz]ed\s+controlled\s+trials?"
    r"|\bRCTs?\b|meta[- ]analyses)\b",
    re.IGNORECASE,
)
_SCIFACT_DECISION_CUES = re.compile(
    r"\b(?:focus(?:ed|es|ing)?\s+(?:exclusively\s+)?on"
    r"|prioriti[sz]\w*|emphasi[sz]\w*|rely\s+on|based\s+on|through|via"
    r"|assess\w*|evaluat\w*|consider\w*|weigh\w*|judge\w*|analy[sz]\w*"
    r"|decid\w*|determin\w*|select\w*"
    r"|only\s+if|require\w*|must\s+(?:have|include|show|demonstrate)"
    r"|lack\w*|insufficient|irrelevant|fail\w*\s+to"
    r"|threshold|criterion|criteria|standard)\b",
    re.IGNORECASE,
)
_SCIFACT_CLAIM_CONCEPT = re.compile(
    r"\bclaim\s+(?:itself\s+)?(?:is\s+about|concerns|asserts|asks\s+whether)\s+",
    re.IGNORECASE,
)
_SCIFACT_CONDITIONAL_SCOPE = re.compile(
    r"\bonly\s+when\s+(?:it|that\s+criterion)\s+is\s+part\s+of\s+the\s+claim\b",
    re.IGNORECASE,
)
_SCIFACT_CONTRAST = re.compile(
    r"\b(?:but|however|yet|instead|whereas|nevertheless)\b|[;:\n]",
    re.IGNORECASE,
)


def _scifact_clause_excludes_topic(clause: str, topic: str) -> bool:
    """Match a negation and exclusion to this topic, not a nearby topic."""
    escaped_topic = re.escape(topic)
    explicit = re.compile(
        rf"\b(?:do\s+not|don't|does\s+not|doesn't|must\s+not|should\s+not|"
        rf"never|not)\b.{{0,70}}\b{escaped_topic}\b.{{0,70}}"
        rf"\b(?:as\s+(?:an?\s+)?)?(?:separate|independent|additional|extra)\s+"
        rf"(?:decision\s+)?criteria?\b",
        re.IGNORECASE,
    )
    direct_negation = re.compile(
        rf"\b(?:do\s+not|don't|does\s+not|doesn't|must\s+not|should\s+not|"
        rf"never)\b.{{0,45}}\b{escaped_topic}\b",
        re.IGNORECASE,
    )
    if explicit.search(clause) or direct_negation.search(clause):
        return True

    for claim_match in _SCIFACT_CLAIM_CONCEPT.finditer(clause):
        after_claim = clause[claim_match.end():claim_match.end() + 80]
        if re.match(rf"\s*(?:the\s+)?{escaped_topic}\b", after_claim,
                    flags=re.IGNORECASE):
            return True
    return bool(_SCIFACT_CONDITIONAL_SCOPE.search(clause))


def validate_task_contract_instruction(
    prompt: str,
    contract: TaskContract | None,
) -> dict[str, Any]:
    """Reject explicit criteria that exceed the registered task decision contract.

    The language-model contract reviewer is useful for semantic review, but it
    can miss added decision standards. This deterministic guard catches the
    SciFact-specific methodology criteria that the task does not request.
    Mentions that explicitly exclude a criterion, or bind that exact concept to
    the claim being verified, remain allowed. Exceptions are matched per topic
    so a nearby exemption cannot authorize another decision criterion.
    """
    if contract is None or contract.contract_id != SCIFACT_CONTRACT.contract_id:
        return {"passed": True, "reason": "no_task_contract_rule", "violations": []}

    violations: list[str] = []
    text = str(prompt or "")
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        clauses = _SCIFACT_CONTRAST.split(sentence)
        for clause in clauses:
            topics = list(_SCIFACT_METHOD_TOPICS.finditer(clause))
            if not topics or not _SCIFACT_DECISION_CUES.search(clause):
                continue
            if any(not _scifact_clause_excludes_topic(clause, match.group(0))
                   for match in topics):
                violations.append("independent_methodology_criterion")
                break
        if violations:
            break

    return {
        "passed": not violations,
        "rule_version": TASK_CONTRACT_RULE_VERSION,
        "reason": ("no_unrequested_independent_criteria" if not violations else
                   "unrequested_methodology_standard_as_decision_criterion"),
        "violations": violations,
    }


def task_contract_for_examples(
    examples: Sequence[dict[str, Any]],
) -> TaskContract | None:
    """Return the frozen contract for one homogeneous, registered task source."""
    datasets = {str(item.get("dataset") or "") for item in examples
                if item.get("dataset")}
    if not datasets:
        return None
    if len(datasets) != 1:
        raise ValueError("task-contract review requires a homogeneous dataset scope")
    dataset = next(iter(datasets))
    return _CONTRACTS.get(dataset)


def validate_task_contract_candidate(
    candidate: str,
    examples: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the registered deterministic task rule to a proposed artifact."""
    return validate_task_contract_instruction(
        candidate, task_contract_for_examples(examples))
