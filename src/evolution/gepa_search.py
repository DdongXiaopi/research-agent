"""Actual GEPA Prompt search on task-supplied inputs and visible training tasks."""

from __future__ import annotations

import uuid
import re
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..config import config_fingerprint
from ..contracts.types import Candidate, content_id, content_sha256
from ..data.split import load_tasks
from ..eval.metrics import total_cost
from ..eval.options import inject_all
from ..eval.task_rows import validate_task_answer_row
from ..governance.policy import static_gate
from ..runtime.input_evidence import task_input_docs
from ..runtime.call_ledger import call_component
from ..runtime.provider import HttpProvider, build_provider
from ..runtime.runner import ArtifactState, TaskRunner
from ..util import artifacts_path, read_json, sha256_obj, write_json
from .gepa_adapter import (TaskExecutionFailure, build_metric, build_optimizer, run_reflection,
                           summarize_chat_usage, summarize_vendor_proposals)
from .prompt_rewrite import validate_rewrite
from .reflection import unified_diff
from .method_rule_gate import (METHOD_RULE_GATE_VERSION,
                               validate_statistical_method_rules)
from .task_contracts import (TASK_CONTRACT_RULE_VERSION,
                            compose_task_instruction,
                            has_immutable_task_contract,
                            task_contract_for_examples,
                            task_instruction_strategy,
                            validate_task_contract_candidate)

TRACE_FEEDBACK_RULE_VERSION = "task-input-trace-asi-v10-evidence-bounded-basis"
FEEDBACK_RULE_VERSION = "task-material-targeted-results-v6"
REFLECTION_GUIDANCE_VERSION = "task-label-contract-aware-v5"
OBJECTIVE_SELECTION_POLICY_VERSION = "accuracy-primary-pareto-maximin-v1"
TASK_INPUT_FINGERPRINT_VERSION = "task-input-payload-sha256-v1"
GEPA_CANDIDATE_BOUNDARY_VERSION = "mutable-strategy-code-composed-contract-v1"


def build_gepa_proposal_task_contract(
        examples: Sequence[Dict[str, Any]]) -> tuple[Optional[str], Dict[str, Any]]:
    """Build an explicit, hashable GEPA reflection context from a frozen contract."""
    contract = task_contract_for_examples(examples)
    if contract is None:
        return None, {
            "task_contract_id": None,
            "task_contract_sha256": None,
            "context_sha256": None,
            "injected_into_reflection_system_message": False,
            "student_runtime_prompt_modified": False,
        }
    context = (
        f"Frozen task decision contract [id={contract.contract_id}; "
        f"sha256={contract.sha256}]:\n{contract.text}"
    )
    return context, {
        "task_contract_id": contract.contract_id,
        "task_contract_sha256": contract.sha256,
        "context_sha256": content_sha256(context),
        "injected_into_reflection_system_message": True,
        "student_runtime_prompt_modified": False,
    }


def gepa_reflection_system_message(task_label_semantics: str = "",
                                  task_contract_context: str = "") -> str:
    """Keep reflection focused on transferable behavior instead of broad heuristics."""
    message = (
        "Rewrite the complete reusable scientific-assistant system prompt. "
        "Use the task feedback, preserve the task's supplied answer-label and output "
        "contracts. Work only with task-provided inputs and do not rely on unavailable "
        "capabilities or hidden answer information. Treat trace IDs as audit metadata. "
        "AttributionHint is heuristic, and AlternativeFailureMechanisms are "
        "possibilities rather than confirmed causes.\n\n"
        "PossibleTargetResultSentences are lexical matches from the supplied "
        "material, not verified evidence labels. Check whether each sentence "
        "reports the exact outcome, exposure, comparator, and population asked; "
        "prefer adjusted analyses over unadjusted trends for association claims. "
        "Do not infer causality from association or transfer a result to another "
        "subgroup.\n\n"
        "For each error, distinguish the claim being asked (for example, a measured "
        "difference, treatment benefit, risk, association, prediction, or causal "
        "effect) from the evidence reported in the study. Match the reported result "
        "to the exact comparison or relationship asked. A nonsignificant result or "
        "an explicit statement of no effect can support NO for that tested claim; "
        "statistical significance is not required when the study directly reports "
        "the target result. Do not generalize that result to a different population, "
        "outcome, subgroup, or causal claim the study did not test.\n\n"
        "Association or correlation between two constructs does not establish that "
        "they are equivalent, identical, or interchangeable. For an equivalence or "
        "identity claim, require evidence that directly tests equivalence/agreement "
        "or otherwise distinguishes the two options; do not treat a significant "
        "correlation alone as proof of equivalence. Do not treat higher answer "
        "coverage as an improvement by itself: an answer replacing abstention must "
        "be supported by evidence that distinguishes the answer options.\n\n"
        "When the material directly supports or contradicts the tested claim, choose "
        "the corresponding label from the task's provided options. If the material "
        "cannot resolve the claim and the task provides an unanswerable option, use "
        "that option; otherwise follow the task's supplied label semantics. Never "
        "invent an answer label that is absent from the provided options. Do not switch "
        "an evidence-supported answer to an abstention label solely because a result "
        "is nonsignificant, indirect in wording, or not stated in one exact sentence.\n\n"
        "Preserve behaviors that work on other examples. Prefer a small, targeted "
        "change over adding a long list of generic rules. Generalize a strategy only "
        "when the feedback supports it across examples or the material directly "
        "establishes the rule. Compare plausible failure mechanisms with the task "
        "trace and answer feedback; never copy task IDs or study-specific facts into "
        "the reusable prompt. Return only the replacement prompt in one fenced block."
    )
    semantics = str(task_label_semantics or "").strip()
    if len(semantics) > 4000:
        raise ValueError("GEPA task label semantics must not exceed 4000 characters")
    if semantics:
        message += (
            "\n\nTask-specific answer-label semantics supplied by the experiment "
            "protocol (apply these labels exactly; do not reinterpret them as a "
            "generic confidence or abstention scale):\n" + semantics
        )
    contract_context = str(task_contract_context or "").strip()
    if len(contract_context) > 4000:
        raise ValueError("GEPA task contract context must not exceed 4000 characters")
    if contract_context:
        message += (
            "\n\nApply this frozen task decision contract as a hard boundary when "
            "rewriting the reusable prompt. Preserve its evidence scope, decision "
            "criterion, and exact label distinctions; do not broaden it into a "
            "generic heuristic or change the runtime task prompt:\n" + contract_context
        )
    return message


def feedback_claim_relation_hint(example: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Expose a narrow, label-free guard for equivalence/identity questions."""
    question = str(example.get("question") or "").lower()
    if re.search(r"\b(?:equivalen\w*|identical|interchangeable)\b|\bsame\s+as\b",
                 question):
        return {
            "claim_relation": "equivalence_or_identity",
            "decision_guard": (
                "Association or correlation alone does not establish equivalence or "
                "identity; require direct evidence that distinguishes the options."),
            "is_heuristic": "true",
        }
    return None


class MetricBudgetExceeded(RuntimeError):
    """Abort GEPA instead of recording a fabricated zero for an unevaluated task."""


_NEGATIVE_RESULT_RE = re.compile(
    r"\b(?:no significant|not significant|nonsignificant|not associated|"
    r"no association|no difference|no increase|no reduction|did not|"
    r"was not|were not|none of|unrelated|no evidence|no protective)\b",
    re.I)
_RESULT_MARKER_RE = re.compile(
    r"\b(?:significant|significance|associated|association|correlated|"
    r"correlation|difference|effect|risk|rate|odds ratio|confidence interval|"
    r"higher|lower|increased|decreased|increase|decrease|reduced|greater|"
    r"less|trend|multivariable|multivariate|adjusted)\b|"
    r"\bp\s*[<=>]\s*0?\.\d+|\b\d+(?:\.\d+)?\s*%",
    re.I)
_NON_RESULT_INTENT_RE = re.compile(
    r"\b(?:it is generally assumed|generally assumed|hypothesized|hypothesis)\b|"
    r"\b(?:we|the authors)\s+(?:aimed|sought|designed|performed|conducted|"
    r"studied|included|reviewed|enrolled|recruited|evaluated|investigated|assessed|"
    r"compared)\b|"
    r"\b(?:to|in order to)\s+(?:investigate|evaluate|estimate|assess|determine|"
    r"examine)\b|"
    r"\b(?:patients|subjects|participants)\s+(?:were|who were)\s+"
    r"(?:included|enrolled|recruited)\b|"
    r"\b(?:was|were|is|are)\s+measured\s+in\s+\d+\s+"
    r"(?:patients|subjects|participants)\b|"
    r"\bwe\s+measured\b|"
    r"\b(?:measurements|assessments)\s+were\s+(?:performed|made)\b",
    re.I)
_QUESTION_STOPWORDS = {
    "does", "did", "do", "is", "are", "was", "were", "be", "been",
    "being", "the", "a", "an", "of", "to", "in", "on", "for", "with",
    "and", "or", "after", "before", "among", "between", "from", "by",
    "have", "has", "had", "this", "that", "these", "those", "it", "its",
    "their", "they", "we", "our", "same", "different", "impact", "effect",
    "current", "following", "using", "versus", "than", "what", "which",
}


def feedback_target_result_sentences(example: Dict[str, Any],
                                      max_items: int = 4) -> List[Dict[str, Any]]:
    """Select question-relevant result sentences without consulting any labels.

    This is lexical evidence triage for GEPA reflection, not an entailment
    classifier. Each item retains the original sentence and matched question
    terms so the reflector can inspect the surrounding task material.
    """
    question_text = str(example.get("question") or "").lower()
    question_terms = {
        token for token in re.findall(r"[a-z0-9]+", question_text)
        if len(token) > 2 and token not in _QUESTION_STOPWORDS
    }
    material = str(example.get("evidence") or example.get("context") or "")
    # Resolve only explicit long-form (ACRONYM) definitions in the supplied
    # material. This lets a question's full concept match later results that
    # use the defined acronym, without importing an external terminology map.
    for match in re.finditer(r"\b([^()]{1,100})\s*\(([A-Z][A-Z0-9-]{1,9})\)",
                             material):
        preceding_text, acronym = match.groups()
        words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", preceding_text.lower())
        for start in range(max(0, len(words) - 9), len(words) - 1):
            long_terms = [word for word in words[start:]
                          if word not in _QUESTION_STOPWORDS]
            initials = "".join(token[0] for token in long_terms).lower()
            if (len(long_terms) >= 2 and initials == acronym.lower()
                    and set(long_terms).issubset(question_terms)):
                question_terms.add(acronym.lower())
                break
    ranked = []
    for index, sentence in enumerate(re.split(r"(?<=[.!?])\s+", material)):
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        overlap = sorted(question_terms & sentence_terms)
        markers = sorted({match.group(0).lower()
                          for match in _RESULT_MARKER_RE.finditer(sentence)})
        # A single shared token (e.g. "position" or "neuropsychological") plus
        # a generic result word was enough to surface unrelated/method sentences.
        # Allow one target token only when the sentence explicitly reports an
        # adjusted/multivariable analysis, which is a meaningful result qualifier.
        # Full task material remains available when no sentence clears this hint.
        adjusted_marker = bool(re.search(r"\b(?:adjusted|multivariable|multivariate)\b",
                                         sentence, re.I))
        if len(overlap) < 2 and not (len(overlap) == 1 and adjusted_marker):
            continue
        if not markers or _NON_RESULT_INTENT_RE.search(sentence):
            continue
        score = 2 * len(overlap) + min(len(markers), 4)
        ranked.append((score, -index, {
            "sentence": sentence[:700],
            "matched_question_terms": overlap[:8],
            "result_markers": markers[:8],
        }))
    ranked.sort(reverse=True)
    return [item for _, _, item in ranked[:max(0, int(max_items))]]


def feedback_evidence_signals(example: Dict[str, Any], max_items: int = 3) -> List[str]:
    """Highlight possible negative-result sentences, without reading the label.

    These are reflection hints, not a factual entailment judge. The optimizer
    still sees the complete available task material and the scored answer.
    """
    material = str(example.get("evidence") or example.get("context") or "")
    sentences = re.split(r"(?<=[.!?])\s+", material)
    return [s.strip()[:450] for s in sentences if _NEGATIVE_RESULT_RE.search(s)][:max_items]


_GENERIC_QUESTION_TOKENS = frozenset({
    # Common English function words.
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "how", "if", "in", "into", "is", "it", "its", "may", "might",
    "more", "most", "of", "on", "or", "our", "should", "that", "the", "their",
    "then", "there", "these", "this", "those", "through", "to", "under", "was",
    "we", "were", "what", "when", "which", "who", "will", "with", "within",
    "would",
    # Repeated wording in the Evidence Inference task-question template.
    "abstract", "answer", "based", "choose", "comparator", "effect", "evidence",
    "inference", "intervention", "label", "material", "option", "outcome", "reported",
    "result", "significance", "significant", "significantly", "statistical", "statistically",
    "supplied", "task", "treatment", "trial", "versus",
})


def training_question_overlap(prompt: str,
                              examples: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Flag rare, informative verbatim question fragments in a reusable prompt.

    Fixed task-question templates repeat generic phrases such as "the reported
    effect of the intervention" across most examples. A raw exact 5-gram match
    falsely treats those shared words as copied task content. Require at least
    three non-generic terms and suppress grams repeated across the visible
    question set; specific copied details remain blocked.
    """
    prompt_tokens = re.findall(r"[a-z0-9]+", prompt.lower())
    if len(prompt_tokens) < 5:
        return None
    candidate_grams = {
        tuple(prompt_tokens[i:i + 5]) for i in range(len(prompt_tokens) - 4)
    }
    gram_task_ids: dict[tuple[str, ...], set[str]] = {}
    for index, example in enumerate(examples):
        question = re.findall(
            r"[a-z0-9]+", str(example.get("question") or "").lower())
        task_id = str(example.get("task_id") or f"example-{index}")
        for i in range(len(question) - 4):
            gram = tuple(question[i:i + 5])
            if len({token for token in gram
                    if token not in _GENERIC_QUESTION_TOKENS}) < 3:
                continue
            gram_task_ids.setdefault(gram, set()).add(task_id)

    # Repeated wording is part of the input format, not a task-specific leak.
    common_gram_frequency = max(2, int(len(examples) * 0.05))
    for gram in sorted(candidate_grams):
        task_ids = gram_task_ids.get(gram)
        if task_ids and len(task_ids) <= common_gram_frequency:
            task_id = sorted(task_ids)[0]
            return (f"verbatim training-question phrase from {task_id}: "
                    + " ".join(gram))
    return None


def task_label_contract_conflict(prompt: str,
                                 examples: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Reject explicit answer-label instructions unsupported by task options."""
    available = {
        str(option).strip().casefold()
        for example in examples for option in (example.get("options") or [])
    }
    for label in ("insufficient", "abstain", "unanswerable"):
        if label in available:
            continue
        pattern = (rf"\b(?:answer|respond|use|select|choose)\b"
                   rf"[^.!?\n]{{0,40}}\b{label}\b")
        if re.search(pattern, prompt, re.I):
            return (f"candidate instructs the model to emit {label!r}, but that label "
                    "is absent from the visible task options")
    # A task may have a substantive NEI class without exposing runtime
    # abstention as an answer. Do not let a candidate turn that valid class into
    # a no-answer outcome when the evidence is uncertain or incomplete.
    abstention_options = {
        label for label in available
        if (any(token in label for token in ("abstain", "insufficient", "unanswerable"))
            or "no answer" in label)
    }
    if not abstention_options:
        action = re.compile(r"\b(?:abstain(?:ing|s)?|abstention|withhold(?:ing)?\s+"
                            r"(?:an?\s+)?answer|refuse\s+to\s+answer)\b", re.I)
        condition = re.compile(r"\b(?:if|when|unless|whenever|in\s+case)\b", re.I)
        negation = re.compile(
            r"(?:\bdo\s+not|\bdon't|\bnever|\bavoid|\bnot\s+(?:an?\s+)?)\s*$",
            re.I,
        )
        for match in action.finditer(prompt):
            start = max(prompt.rfind(mark, 0, match.start())
                        for mark in (".", "!", "?", ";", "\n")) + 1
            end_candidates = [position for mark in (".", "!", "?", ";", "\n")
                              if (position := prompt.find(mark, match.end())) >= 0]
            end = min(end_candidates) if end_candidates else len(prompt)
            clause = prompt[start:end]
            before_action = clause[:match.start() - start]
            # Negative rules such as “do not abstain; choose NEI” preserve the
            # task's answer contract and should not be rejected.
            if negation.search(before_action[-32:]):
                continue
            if condition.search(clause):
                return ("candidate instructs runtime abstention for a conditional "
                        "case, but abstention is absent from the visible task options")
    # Catch task-specific enumerations too. The old check only guarded common
    # abstention words, so an EI instruction such as “choose significantly
    # increased/decreased/no significant difference” passed against HealthVer's
    # SUPPORT/CONTRADICT/NOT_ENOUGH_INFO options.
    listing = re.search(
        r"\b(?:choose|select|answer|respond with)\s+(?:exactly\s+)?"
        r"(?:one|a single|the)\s+(?:of\s+)?(?:the\s+)?(?:listed\s+)?"
        r"(?:answer\s+)?(?:options?|labels?)"
        r"(?:\s+that\s+best\s+matches?)?\s*:\s*([^\n.]+)",
        prompt, re.I,
    )
    if listing:
        proposed_labels = [
            re.sub(r"[^a-z0-9]+", " ", item.casefold()).strip()
            for item in re.split(r",|\s+(?:or|and)\s+", listing.group(1), flags=re.I)
        ]
        proposed_labels = [item for item in proposed_labels if item]
        if proposed_labels:
            normalized_available = {
                re.sub(r"[^a-z0-9]+", " ", item).strip()
                for item in available
            }
            unsupported = [item for item in proposed_labels
                           if item not in normalized_available]
            if unsupported:
                return ("candidate enumerates answer labels absent from the visible task "
                        f"options: {unsupported[:3]}")

    datasets = {str(example.get("dataset") or "") for example in examples}
    if "clinifact_fixed_abstract" in datasets:
        if datasets != {"clinifact_fixed_abstract"}:
            return "CliniFact label contract cannot be checked on a mixed-dataset candidate scope"
        contract = task_contract_for_examples(examples)
        if has_immutable_task_contract(prompt, contract):
            # MIPRO's runtime contract is appended by code and hash-identified;
            # do not re-infer its exact labels or output type from prose.
            return None
        definitions: dict[str, str] = {}
        for label in ("EVIDENCE", "INCONCLUSIVE", "NEI"):
            # MIPRO emits complete instructions with ordinary Markdown labels
            # ("**EVIDENCE**:") and sometimes inline parenthetical definitions.
            # Parse those forms as definitions, then apply the same semantic
            # checks below. An enumeration alone still is not a definition.
            match = re.search(
                rf"\b{label}\b(?:\*\*|__)?\s*(?::|means?|is)\s*"
                rf"(.*?)(?:[.!?\n]|$)", prompt, re.I,
            )
            if match is None:
                match = re.search(
                    rf"\b{label}\b(?:\*\*|__)?\s*\(([^)]+)\)",
                    prompt, re.I,
                )
            if match is None:
                return (f"CliniFact candidate must define {label} in the prompt "
                        "to preserve its source-specific task semantics")
            definitions[label] = match.group(1).casefold()
        negated = re.compile(
            r"\b(?:not|no|never|without|isn't|doesn't|cannot|can't)\b"
            r"(?:\W+\w+){0,3}\W*$", re.I)

        def has_unnegated(pattern: re.Pattern[str], text: str) -> bool:
            return any(not negated.search(text[max(0, match.start() - 40):match.start()])
                       for match in pattern.finditer(text))

        supports_claim = re.compile(r"\bsupport\w*\b", re.I)
        significant_result = re.compile(r"\bstatistically\s+significant\b", re.I)
        nonsignificant_result = re.compile(
            r"\b(?:non[- ]?significant|not\s+statistically\s+significant|"
            r"inconclusive)\b", re.I)
        no_result = re.compile(
            r"\b(?:background|not\s+enough\s+information|no\s+information|"
            r"does\s+not\s+(?:report|address)|doesn't\s+(?:report|address))\b", re.I)
        if (not has_unnegated(supports_claim, definitions["EVIDENCE"])
                or not has_unnegated(significant_result, definitions["EVIDENCE"])):
            return ("CliniFact EVIDENCE must mean a statistically significant result "
                    "that supports the claim")
        if not has_unnegated(nonsignificant_result, definitions["INCONCLUSIVE"]):
            return "CliniFact INCONCLUSIVE must mean a reported non-significant/inconclusive result"
        if not has_unnegated(no_result, definitions["NEI"]):
            return "CliniFact NEI must mean background or no reported information addressing the claim"
        if (has_unnegated(no_result, definitions["INCONCLUSIVE"])
                or has_unnegated(nonsignificant_result, definitions["NEI"])):
            return "CliniFact INCONCLUSIVE and NEI meanings must remain distinct"
        if not (re.search(r"\b(?:exactly\s+one|one)\s+ANSWER\s+line\b", prompt, re.I)
                and re.search(r"\b(?:option\s+number|numeric\s+option)\b", prompt, re.I)):
            return "CliniFact candidate must require one numeric ANSWER line"
    return None


def feedback_material(example: Dict[str, Any], limit: int = 3000) -> str:
    """Keep the beginning and conclusion when long task material is summarized."""
    material = str(example.get("evidence") or example.get("context") or "")
    if len(material) <= limit:
        return material
    half = (limit - 80) // 2
    return (material[:half] + "\n[... middle omitted from reflection feedback ...]\n"
            + material[-half:])


def task_input_fingerprint(example: Dict[str, Any]) -> str:
    """Hash the exact task payload components used by task_input evaluation.

    Exclude the gold answer so this fingerprint proves input immutability without
    encoding evaluation labels. The underlying evidence document includes the
    runtime's trimming, source URL, and content hash representation.
    """
    return sha256_obj({
        "dataset": str(example.get("dataset", "")),
        "task_id": str(example.get("task_id", "")),
        "question": str(example.get("question", "")),
        "options": [str(option) for option in example.get("options", [])],
        "task_input_docs": task_input_docs(example),
    })


def task_trace_side_info(outcome: Any) -> Dict[str, Any]:
    """Expose compact, label-safe execution diagnostics to GEPA as actionable feedback.

    GEPA's ASI is where runtime traces become optimization guidance. Keep only
    event IDs and allowlisted metadata; raw prompts, query text, and judge output
    are excluded. Attribution is a content-derived hint, not a verified cause.
    """
    from .attribution import (ATTRIBUTION_RULE_VERSION, attribute,
                              validated_answer_trace_ids)

    payload_fields = {
        "task.start": (),
        "skill.candidates": ("candidates", "candidate_scores", "recall_min_score",
                             "routing_mode"),
        "skill.loaded": ("name", "behaviors", "effective"),
        "input.evidence": ("doc_ids", "hashes"),
        "tool.call": ("doc_ids", "top_k", "query_source", "source_query_names",
                      "cited", "precision"),
        "model.call": ("mode", "answer_index"),
        "task.answer.selected": ("raw_answer_index", "answer_index",
                                  "selected_option_sha256", "abstained"),
        "citation.support": ("supported", "reason"),
        "evidence.top3_consistency": ("consistent", "reason"),
        "plan.failed": ("plan_kind", "planner", "instruction_source",
                         "failure_code"),
        "task.end": ("correct", "status"),
    }
    events = list(getattr(outcome, "events", []) or [])
    task_id = str(getattr(outcome, "task_id", "") or "")
    task_scoped = [event for event in events
                   if isinstance(event, dict) and event.get("task_id") == task_id]
    if task_scoped:
        events = task_scoped
    elif any(isinstance(event, dict) and event.get("task_id")
             for event in events):
        # A shared trace with no matching task events must not leak a sibling's
        # execution into this failure's optimizer feedback.
        events = []
    answer_trace_ids = validated_answer_trace_ids(outcome, events)
    trace_events = []
    run_ids = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        kind = str(event.get("event_type") or "")
        if kind not in payload_fields and kind != "budget.exceeded":
            continue
        if (kind == "model.call"
                and str(event.get("event_id") or "") not in answer_trace_ids):
            continue
        if (kind == "task.answer.selected"
                and str(event.get("event_id") or "") not in answer_trace_ids):
            continue
        if event.get("run_id"):
            run_ids.add(str(event["run_id"]))
        row = {key: event[key] for key in
               ("event_id", "event_type", "parent_event_id", "tool_name")
               if event.get(key) not in (None, "")}
        if event.get("error") not in (None, ""):
            # Provider/tool exception bodies may echo request material or
            # credentials. The optimizer only needs to know a failure occurred.
            row["failure_reported"] = True
        payload = event.get("payload") or {}
        safe_payload = {key: payload[key] for key in payload_fields.get(kind, ())
                        if key in payload}
        if safe_payload:
            row["payload"] = safe_payload
        trace_events.append(row)

    sources = []
    doc_ids = list(getattr(outcome, "visible_evidence_doc_ids", []) or [])
    hashes = list(getattr(outcome, "visible_evidence_hashes", []) or [])
    # Preserve compatibility with lightweight test doubles and old in-memory
    # outcomes while keeping persisted TaskOutcome retrieval fields clean.
    if not doc_ids:
        if str(getattr(outcome, "input_mode", "")) == "task_input":
            doc_ids = list(getattr(outcome, "input_doc_ids", []) or [])
            hashes = list(getattr(outcome, "input_passage_hashes", []) or [])
        else:
            doc_ids = list(getattr(outcome, "retrieved_doc_ids", []) or [])
            hashes = list(getattr(outcome, "retrieved_passage_hashes", []) or [])
    for index, doc_id in enumerate(doc_ids):
        sources.append({"doc_id": str(doc_id),
                        "content_sha256": str(hashes[index]) if index < len(hashes) else ""})
    try:
        attribution_hint = attribute(outcome)
    except (AttributeError, TypeError, ValueError):
        attribution_hint = "unknown"
    result = {
        "TaskID": str(getattr(outcome, "task_id", "")),
        "Dataset": str(getattr(outcome, "dataset", "")),
        "Split": str(getattr(outcome, "split", "")),
        "InputMode": str(getattr(outcome, "input_mode", "")),
        "RunID": next(iter(sorted(run_ids)), ""),
        "AttributionHint": attribution_hint,
        "AttributionRuleVersion": ATTRIBUTION_RULE_VERSION,
        "AttributionHintIsHeuristic": True,
        "SkillCandidates": list(getattr(outcome, "skill_candidates", []) or []),
        "SkillRecalled": str(getattr(outcome, "skill_recalled", "") or ""),
        "SkillBehaviors": list(getattr(outcome, "skill_behaviors", []) or []),
        "EvidenceSources": sources,
        "TraceEvents": trace_events,
        "RuntimeFailureReported": bool(getattr(outcome, "error", "")),
    }
    if attribution_hint == "plan.failed":
        result.update({
            "AttributionHintBasis": (
                "A task-answer output failed runtime format validation; this is a "
                "traceable output-format failure, not evidence that the system Prompt "
                "caused it. The invalid GEPA task outcome is not converted into an "
                "answer score."),
            "AlternativeFailureMechanisms": [
                "The recalled Skill procedure may be mismatched or misapplied.",
                "The answer selection or abstention strategy may be unsuitable.",
                "The task may exceed model capability or be ambiguous under the supplied material.",
            ],
        })
    elif attribution_hint == "unknown":
        result.update({
            "AttributionHintBasis": (
                "The available task-scoped trace cannot establish a component-specific "
                "cause for this answer failure. Skill loading or behavior markers, "
                "when present, do not establish correct procedure application or whether "
                "the Prompt/model/task caused the failure."),
            "AlternativeFailureMechanisms": [
                "A recalled Skill procedure may be mismatched or misapplied.",
                "The system Prompt or answer-selection strategy may be unsuitable.",
                "The supplied evidence may be ambiguous or insufficient.",
                "The task may exceed the answer model's capability.",
            ],
        })
    elif attribution_hint is not None:
        result["AttributionHintBasis"] = (
            "This label follows the versioned content-derived routing rule; it is "
            "a diagnostic hint, not an independently validated causal finding.")
    return result


def ranked_challenger_indices(
        scores: Sequence[float],
        objective_scores: Optional[Sequence[Dict[str, float]]] = None) -> List[int]:
    """Rank scalar challengers or the accuracy-safe Pareto front.

    Multiobjective candidates must not regress primary accuracy. Among those,
    retain only candidates that improve on the baseline and are non-dominated
    across the reported higher-is-better objectives. Order that front by primary
    score, then maximin objective delta, then total delta for deterministic replay.
    """
    if len(scores) < 2:
        return []
    baseline_score = float(scores[0])
    if objective_scores is None:
        best = max(float(x) for x in scores)
        return sorted((i for i in range(1, len(scores))
                       if float(scores[i]) >= baseline_score - 1e-9
                       and float(scores[i]) >= best - 1e-9),
                      key=lambda i: (-float(scores[i]), i))

    if len(objective_scores) != len(scores):
        raise ValueError("objective validation scores must align with candidates")
    names = set(objective_scores[0])
    if "accuracy" not in names or not names:
        raise ValueError("multiobjective selection requires an accuracy objective")
    vectors: List[Dict[str, float]] = []
    for vector in objective_scores:
        if set(vector) != names:
            raise ValueError("objective validation score keys must be consistent")
        numeric = {name: float(value) for name, value in vector.items()}
        if any(not math.isfinite(value) for value in numeric.values()):
            raise ValueError("objective validation scores must be finite")
        vectors.append(numeric)
    if any(abs(float(score) - vector["accuracy"]) > 1e-9
           for score, vector in zip(scores, vectors)):
        raise ValueError("primary scores must match the accuracy objective")
    baseline = vectors[0]
    eligible: List[int] = []
    for idx in range(1, len(vectors)):
        current = vectors[idx]
        if current["accuracy"] < baseline["accuracy"] - 1e-9:
            continue
        if all(current[name] <= baseline[name] + 1e-9 for name in names):
            continue
        eligible.append(idx)

    front = []
    for idx in eligible:
        current = vectors[idx]
        dominated = any(
            other != idx and
            all(vectors[other][name] >= current[name] - 1e-9 for name in names) and
            any(vectors[other][name] > current[name] + 1e-9 for name in names)
            for other in eligible)
        if not dominated:
            front.append(idx)

    def rank(idx: int) -> tuple:
        deltas = [vectors[idx][name] - baseline[name] for name in names]
        return (-vectors[idx]["accuracy"], -min(deltas), -sum(deltas), idx)

    return sorted(front, key=rank)


def visible_examples(cfg: Dict[str, Any], datasets: Sequence[str],
                     limit: Optional[int]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split an offset/limit slice or an exact ID cohort into train and val.

    `evolution.gepa.task_offset` is applied independently inside each dataset,
    before the existing deterministic combined ordering and train/val split.
    When `visible_task_ids` is set, it replaces offset/limit selection and is
    validated against the requested datasets' dev_optimize rows.
    """
    examples: List[Dict[str, Any]] = []
    gp = ((cfg.get("evolution") or {}).get("gepa") or {})
    offset = int(gp.get("task_offset", 0))
    if offset < 0:
        raise ValueError("GEPA task_offset must be non-negative")
    requested_ids = gp.get("visible_task_ids")
    requested_train_ids = gp.get("visible_train_task_ids")
    requested_validation_ids = gp.get("visible_validation_task_ids")
    if requested_train_ids is not None or requested_validation_ids is not None:
        if requested_ids is not None:
            raise ValueError("GEPA exact train/validation IDs cannot be combined with visible_task_ids")
        if offset:
            raise ValueError("GEPA exact train/validation IDs cannot be combined with task_offset")
        if (not isinstance(requested_train_ids, list) or not requested_train_ids
                or not isinstance(requested_validation_ids, list)
                or not requested_validation_ids):
            raise ValueError("GEPA exact train and validation task IDs must both be non-empty lists")
        train_ids = [str(task_id) for task_id in requested_train_ids]
        validation_ids = [str(task_id) for task_id in requested_validation_ids]
        combined_ids = train_ids + validation_ids
        if any(not task_id for task_id in combined_ids):
            raise ValueError("GEPA exact cohort IDs cannot contain empty IDs")
        if len(set(combined_ids)) != len(combined_ids):
            raise ValueError("GEPA exact train/validation IDs must be unique and disjoint")
        if limit is not None and limit < len(combined_ids):
            raise ValueError("GEPA limit cannot truncate exact train/validation IDs")
        requested = set(combined_ids)
        found: Dict[str, Dict[str, Any]] = {}
        for dataset in datasets:
            for row in load_tasks(dataset, splits=["dev_optimize"]):
                task_id = str(row["task_id"])
                if task_id in requested:
                    if task_id in found:
                        raise ValueError(f"GEPA visible task ID is ambiguous: {task_id}")
                    found[task_id] = row
        missing = requested - set(found)
        if missing:
            raise ValueError("GEPA exact train/validation IDs must belong to requested "
                             "dev_optimize data: " + ", ".join(sorted(missing)))
        train = [found[task_id] for task_id in train_ids]
        val = [found[task_id] for task_id in validation_ids]
        if cfg.get("evaluation", {}).get("insufficient_option", True):
            train, val = inject_all(train), inject_all(val)
        return train, val
    if requested_ids is not None:
        if offset:
            raise ValueError("GEPA visible_task_ids cannot be combined with task_offset")
        if not isinstance(requested_ids, list) or not requested_ids:
            raise ValueError("GEPA visible_task_ids must be a non-empty list")
        normalized_ids = [str(task_id) for task_id in requested_ids]
        if any(not task_id for task_id in normalized_ids):
            raise ValueError("GEPA visible_task_ids cannot contain empty IDs")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("GEPA visible_task_ids must be unique")
        if limit is not None and limit < len(normalized_ids):
            raise ValueError("GEPA limit cannot truncate visible_task_ids")
        requested = set(normalized_ids)
        found: Dict[str, Dict[str, Any]] = {}
        for dataset in datasets:
            for row in load_tasks(dataset, splits=["dev_optimize"]):
                task_id = str(row["task_id"])
                if task_id in requested:
                    if task_id in found:
                        raise ValueError(f"GEPA visible task ID is ambiguous: {task_id}")
                    found[task_id] = row
        missing = requested - set(found)
        if missing:
            raise ValueError("GEPA visible_task_ids must belong to requested dev_optimize data: "
                             + ", ".join(sorted(missing)))
        examples = [found[task_id] for task_id in normalized_ids]
    for dataset in datasets:
        if requested_ids is not None:
            continue
        rows = sorted(load_tasks(dataset, splits=["dev_optimize"]),
                      key=lambda t: t["task_id"])
        stop = None if limit is None else offset + limit
        rows = rows[offset:stop]
        if cfg.get("evaluation", {}).get("insufficient_option", True):
            rows = inject_all(rows)
        examples.extend(rows)
    if len(examples) < 3:
        raise ValueError("GEPA needs at least 3 visible dev_optimize tasks")
    examples.sort(key=lambda t: (str(t["dataset"]), str(t["task_id"])))
    if bool(gp.get("stratify_by_gold_label", False)):
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for example in examples:
            options = example.get("options") or []
            try:
                gold = str(options[int(example["correct_index"])]).strip().casefold()
            except (IndexError, KeyError, TypeError, ValueError):
                raise ValueError(
                    "stratified GEPA split requires a valid gold option on every task")
            if not gold:
                raise ValueError("stratified GEPA split found an empty gold label")
            by_label.setdefault(gold, []).append(example)
        rare = sorted(label for label, rows in by_label.items() if len(rows) < 2)
        if rare:
            raise ValueError(
                "stratified GEPA split needs at least two examples per gold label; "
                f"singleton_labels={rare}")
        train, val = [], []
        for label in sorted(by_label):
            rows = sorted(by_label[label],
                          key=lambda task: (str(task["dataset"]), str(task["task_id"])))
            # Each label needs representation in both scopes, including n=2.
            validation_indices = set(range(2, len(rows), 3)) or {len(rows) - 1}
            train.extend(row for index, row in enumerate(rows)
                         if index not in validation_indices)
            val.extend(row for index, row in enumerate(rows)
                       if index in validation_indices)
        train.sort(key=lambda t: (str(t["dataset"]), str(t["task_id"])))
        val.sort(key=lambda t: (str(t["dataset"]), str(t["task_id"])))
    else:
        val = [x for i, x in enumerate(examples) if i % 3 == 2]
        train = [x for i, x in enumerate(examples) if i % 3 != 2]
    if not train or not val:
        raise ValueError("GEPA training/validation split is empty")
    return train, val


def replay_prompt_candidate(cfg: Dict[str, Any], state: ArtifactState,
                            datasets: Sequence[str], limit: Optional[int],
                            failure_ids: Sequence[str], report_path: str,
                            candidate_index: int) -> Candidate:
    """Reconstruct a dev-selected GEPA candidate without another stochastic search.

    The source report must match this exact release, protocol and visible train/val
    cohort. The outer round still applies its own static and evaluation gates.
    """
    if (cfg.get("runtime") or {}).get("input_mode") != "task_input":
        raise ValueError("GEPA replay requires task_input mode")
    source = read_json(Path(report_path))
    train, val = visible_examples(cfg, datasets, limit)
    if (source.get("release_id") != state.release_id or
            source.get("protocol_sha256") != config_fingerprint(cfg) or
            source.get("train_ids") != [x["task_id"] for x in train] or
            source.get("val_ids") != [x["task_id"] for x in val]):
        raise ValueError("GEPA report release, protocol or visible-dev cohort mismatch")
    if source.get("status") in ("searching", "budget_exhausted_no_selection"):
        raise ValueError("GEPA report has no completed, within-budget search")
    if int(source.get("evaluator_attempts", 0)) > int(
            source.get("requested_max_metric_calls", -1)):
        raise ValueError("GEPA report exceeded its task-evaluation budget")
    scores = source.get("val_aggregate_scores") or []
    texts = source.get("candidate_texts") or []
    contract = task_contract_for_examples(train + val)
    is_clinifact = bool(contract and contract.contract_id ==
                        "clinifact_trial_result_claim_evidence_v4")
    if is_clinifact and source.get("candidate_boundary_version") != \
            GEPA_CANDIDATE_BOUNDARY_VERSION:
        raise ValueError("GEPA report candidate boundary is missing or outdated")
    if is_clinifact and (
            source.get("task_contract_id") != contract.contract_id or
            source.get("task_contract_sha256") != contract.sha256):
        raise ValueError("GEPA report task contract identity mismatch")
    multi_objective = bool(source.get("multi_objective", False))
    objective_scores = source.get("val_aggregate_subscores") if multi_objective else None
    if multi_objective and source.get("objective_selection_policy") != \
            OBJECTIVE_SELECTION_POLICY_VERSION:
        raise ValueError("GEPA objective selection policy is missing or outdated")
    ranked = ranked_challenger_indices(scores, objective_scores)
    if len(scores) != len(texts) or candidate_index not in ranked:
        raise ValueError("index is not a best, nonregressing visible-dev challenger")
    strategy = str(texts[candidate_index]).strip()
    if not strategy:
        raise ValueError("GEPA challenger strategy is empty")
    runtime_prompt = compose_task_instruction(strategy, contract)
    if is_clinifact:
        strategy_hashes = source.get("candidate_strategy_sha256") or []
        recorded_prompts = source.get("candidate_runtime_prompts") or []
        recorded_hashes = source.get("candidate_runtime_prompt_sha256") or []
        if (candidate_index >= len(strategy_hashes) or
                strategy_hashes[candidate_index] != sha256_obj(strategy) or
                candidate_index >= len(recorded_prompts) or
                candidate_index >= len(recorded_hashes) or
                str(recorded_prompts[candidate_index]) != runtime_prompt or
                recorded_hashes[candidate_index] != sha256_obj(runtime_prompt)):
            raise ValueError("GEPA report runtime prompt identity mismatch")
    if runtime_prompt == state.system_prompt.strip():
        raise ValueError("GEPA challenger is empty or identical to baseline")
    overlap = training_question_overlap(strategy, train + val)
    task_contract_gate = validate_task_contract_candidate(runtime_prompt, train + val)
    label_conflict = task_label_contract_conflict(runtime_prompt, train + val)
    reason = validate_rewrite(runtime_prompt, state.system_prompt, cfg)
    method_rules = validate_statistical_method_rules(runtime_prompt)
    if (overlap or not task_contract_gate["passed"] or label_conflict or reason
            or not method_rules["passed"]):
        raise ValueError("GEPA challenger invalid: "
                         + str(overlap or (task_contract_gate["reason"]
                                           if not task_contract_gate["passed"] else None) or
                               label_conflict or reason or
                               method_rules["reason"]))
    tied = float(scores[candidate_index]) <= float(scores[0]) + 1e-9
    return Candidate(
        candidate_id=content_id("cand", state.release_id, "system_prompt", runtime_prompt),
        parent_release_id=state.release_id, artifact_type="system_prompt",
        artifact_id="system_prompt", diff=unified_diff(state.system_prompt, runtime_prompt),
        rationale="Pinned GEPA visible-dev challenger replay",
        status="draft", new_content=runtime_prompt, attributed_class="plan.failed",
        source_failure_ids=list(failure_ids),
        generator={"backend": "pinned-gepa-replay", "is_gepa": True,
                   "search_report": str(Path(report_path).resolve()),
                   "search_report_sha256": sha256_obj(source),
                   "selected_idx": candidate_index, "validation_tie": tied,
                   "selected_val_score": scores[candidate_index],
                   "total_metric_calls": source.get("total_metric_calls")},
        content_sha256=content_sha256(runtime_prompt))


def search_prompt_candidate(cfg: Dict[str, Any], state: ArtifactState,
                            datasets: Sequence[str], limit: Optional[int],
                            seed: int, failure_ids: Sequence[str], *,
                            report_path_callback: Optional[Callable[[Path], None]] = None
                            ) -> Optional[Candidate]:
    """Use pinned GEPA for mutation/Pareto; outer round owns gate and review."""
    if (cfg.get("runtime") or {}).get("input_mode") != "task_input":
        raise ValueError("mainline GEPA search currently requires task_input mode")
    if (cfg.get("evaluation") or {}).get("main_metric", "accuracy") != "accuracy":
        raise ValueError("GEPA task score is accuracy; selection gate must use accuracy too")
    gp = (cfg.get("evolution") or {}).get("gepa") or {}
    use_evidence_signals = bool(gp.get("feedback_evidence_signals", False))
    multi_objective = bool(gp.get("multi_objective", False))
    task_label_semantics = str(gp.get("reflection_label_semantics") or "").strip()
    if len(task_label_semantics) > 4000:
        raise ValueError("GEPA task label semantics must not exceed 4000 characters")
    train, val = visible_examples(cfg, datasets, limit)
    proposal_task_contract_context, proposal_task_contract = (
        build_gepa_proposal_task_contract(train + val))
    task_contract = task_contract_for_examples(train + val)
    # Runtime composition is enabled for contracts whose visible labels are
    # otherwise easy for a free-form proposer to broaden (for example, turning
    # a binary yes/no task into an abstention task).  Older EI/HealthVer/SciFact
    # releases keep their historical prompt bytes; their contracts still gate
    # proposals and are included in the reflection context.
    runtime_contract = (task_contract if task_contract and task_contract.contract_id in {
        "clinifact_trial_result_claim_evidence_v4",
        "bioasq_yesno_binary_answer_v1",
    } else None)
    seed_strategy = task_instruction_strategy(state.system_prompt, runtime_contract)
    reflection_system_message = gepa_reflection_system_message(
        task_label_semantics, proposal_task_contract_context or "")
    reflection_system_message_sha256 = content_sha256(reflection_system_message)
    run_id = "gepa-prompt-" + uuid.uuid4().hex[:12]
    out_dir = artifacts_path("search", "gepa", run_id)
    out_dir.mkdir(parents=True, exist_ok=False)
    report_path = out_dir / "report.json"
    if report_path_callback is not None:
        report_path_callback(report_path)
    task_provider = build_provider(cfg, seed=seed)
    reflection_cfg = dict((cfg.get("models") or {}).get("reflection") or {})
    if reflection_cfg.pop("provider", None) != "http":
        raise ValueError("actual GEPA requires an HTTP reflection model")
    reflection_cfg["max_tokens"] = max(1200, int(reflection_cfg.get("max_tokens") or 0))
    reflection_provider = HttpProvider(cache=False, **reflection_cfg)
    metric = build_metric(multi_objective=multi_objective)
    max_metric_calls = int(gp.get("max_metric_calls", 120))
    max_candidate_proposals = int(gp.get("max_candidate_proposals", 8))
    reflection_minibatch_size = int(gp.get("reflection_minibatch_size", 5))
    acceptance_criterion = str(gp.get("acceptance_criterion", "strict_improvement"))
    use_merge = bool(gp.get("use_merge", False))
    merge_max_invocations = int(gp.get("merge_max_invocations", 5))
    merge_val_overlap_floor = int(gp.get("merge_val_overlap_floor", 5))
    validation_reserve = len(val)
    optimizer_max_metric_calls = max_metric_calls - validation_reserve
    if optimizer_max_metric_calls <= len(val):
        raise ValueError(
            "GEPA task-evaluation budget must cover seed validation, at least one "
            "proposal evaluation, and one full candidate validation "
            f"(need > {2 * len(val)} calls; got {max_metric_calls})")
    metric_attempts = 0
    task_model_evaluations = 0
    task_outcomes: List[Any] = []
    task_evaluation_rows: List[Dict[str, Any]] = []
    pre_evaluation_rejections: List[Dict[str, Any]] = []
    validation_keys = {(str(x.get("dataset", "")), str(x.get("task_id", "")))
                       for x in val}
    reflection_usage: List[Dict[str, Any]] = []

    def record_pre_evaluation_rejection(candidate_text: str,
                                       example: Dict[str, Any],
                                       partition: str, gate_name: str,
                                       reason: str,
                                       runtime_prompt: Optional[str] = None) -> None:
        pre_evaluation_rejections.append({
            "dataset": str(example.get("dataset", "")),
            "task_id": str(example.get("task_id", "")),
            "partition": partition,
            "candidate_sha256": sha256_obj(candidate_text),
            "candidate_strategy_sha256": sha256_obj(candidate_text),
            "runtime_prompt_sha256": sha256_obj(
                runtime_prompt if runtime_prompt is not None else candidate_text),
            "gate": gate_name,
            "reason": str(reason),
        })

    def reflection_lm(prompt: Any) -> str:
        if isinstance(prompt, list):
            user = "\n\n".join(str(x.get("content", "")) for x in prompt)
        else:
            user = str(prompt)
        with call_component("gepa_reflection"):
            measured = reflection_provider.chat_measured(
                reflection_system_message, user,
                max_tokens=1400)
        reflection_usage.append({
            "prompt_tokens": measured.prompt_tokens,
            "completion_tokens": measured.completion_tokens,
            "latency_ms": measured.latency_ms,
            "cache_hit": measured.cache_hit,
            "usage_available": measured.usage_available,
            "cost": measured.cost,
        })
        report["reflection_usage"] = summarize_chat_usage(reflection_usage)
        report["reflection_provider_stats"] = (
            dict(reflection_provider.stats)
            if isinstance(getattr(reflection_provider, "stats", None), dict) else {})
        write_json(report_path, report)
        return measured.content

    def evaluator(candidate: str, example: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        nonlocal metric_attempts, task_model_evaluations
        partition = ("validation" if (str(example.get("dataset", "")),
                                       str(example.get("task_id", ""))) in validation_keys
                     else "train")
        metric_attempts += 1
        if metric_attempts > max_metric_calls:
            # A placeholder score would be treated as a real task outcome by
            # GEPA and can silently corrupt its valset/Pareto state. Abort the
            # current search; the caller records the budget failure separately.
            raise MetricBudgetExceeded(
                f"GEPA requested task evaluation {metric_attempts} above hard "
                f"budget {max_metric_calls}")
        proposed_strategy = task_instruction_strategy(candidate, runtime_contract)
        proposed = compose_task_instruction(proposed_strategy, runtime_contract)
        if proposed != state.system_prompt.strip():
            task_contract_gate = validate_task_contract_candidate(
                proposed, [example])
            if not task_contract_gate["passed"]:
                record_pre_evaluation_rejection(
                    proposed, example, partition, "task_contract",
                    str(task_contract_gate["reason"]), proposed)
                return 0.0, {
                    "Feedback": "Candidate rejected before evaluation: "
                                + task_contract_gate["reason"],
                    "Status": "task_contract_reject",
                    "TaskContractGate": task_contract_gate,
                }
            overlap = training_question_overlap(proposed, train + val)
            if overlap:
                record_pre_evaluation_rejection(
                    proposed_strategy, example, partition, "training_overlap", overlap,
                    proposed)
                return 0.0, {"Feedback": "Candidate rejected before evaluation: "
                             + overlap, "Status": "training_overlap"}
            label_conflict = task_label_contract_conflict(proposed, train + val)
            if label_conflict:
                record_pre_evaluation_rejection(
                    proposed_strategy, example, partition, "task_label_contract",
                    label_conflict, proposed)
                return 0.0, {"Feedback": "Candidate rejected before evaluation: "
                             + label_conflict, "Status": "task_label_contract"}
            method_rules = validate_statistical_method_rules(proposed)
            if not method_rules["passed"]:
                record_pre_evaluation_rejection(
                    proposed_strategy, example, partition, "method_rule",
                    str(method_rules["reason"]), proposed)
                return 0.0, {"Feedback": "Candidate rejected before evaluation: "
                             + method_rules["reason"],
                             "Status": "method_rule_reject",
                             "MethodRuleGate": method_rules}
            trial = Candidate(
                candidate_id=content_id("cand", state.release_id, "system_prompt", proposed),
                parent_release_id=state.release_id, artifact_type="system_prompt",
                artifact_id="system_prompt",
                diff=unified_diff(state.system_prompt, proposed), new_content=proposed)
            reason = validate_rewrite(proposed, state.system_prompt, cfg)
            gate = static_gate(trial, cfg, state)
            if reason or not gate["passed"]:
                failed = [x["check"] for x in gate["results"] if not x["passed"]]
                rejection_reason = str(reason or ", ".join(failed))
                record_pre_evaluation_rejection(
                    proposed_strategy, example, partition,
                    "rewrite" if reason else "static", rejection_reason, proposed)
                return 0.0, {"Feedback": "Candidate rejected before evaluation: "
                             + rejection_reason, "Status": "static_reject"}
        trial_state = ArtifactState(
            release_id=state.release_id, system_prompt=proposed, registry=state.registry,
            task_demos=state.task_demos)
        task_model_evaluations += 1
        outcome = TaskRunner(cfg, trial_state, provider=task_provider, seed=seed).run(example)
        task_outcomes.append(outcome)
        if str(getattr(outcome, "status", "")) in {"ok", "abstain"}:
            try:
                outcome_row = (outcome.to_dict() if callable(
                    getattr(outcome, "to_dict", None)) else vars(outcome))
                validate_task_answer_row(
                    outcome_row, example, context="GEPA task evaluation",
                    require_reported_gold_index=True)
            except (AttributeError, TypeError, ValueError) as exc:
                raise TaskExecutionFailure(
                    str(getattr(outcome, "task_id", example.get("task_id", "unknown"))),
                    "invalid_outcome") from exc
        report["task_evaluation_cost"] = total_cost(task_outcomes)
        write_json(report_path, report)
        score, side = metric(outcome)
        task_evaluation_rows.append({
            "dataset": str(example.get("dataset", "")),
            "task_id": str(example.get("task_id", "")),
            "partition": partition,
            "candidate_sha256": sha256_obj(proposed),
            "candidate_strategy_sha256": sha256_obj(proposed_strategy),
            "runtime_prompt_sha256": sha256_obj(proposed),
            "task_input_sha256": task_input_fingerprint(example),
            "task_input_fingerprint_version": TASK_INPUT_FINGERPRINT_VERSION,
            "score": float(score),
            "status": str(outcome.status),
            "correct": bool(outcome.correct),
            "answer_index": int(outcome.answer_index),
        })
        report["task_evaluation_rows"] = list(task_evaluation_rows)
        write_json(report_path, report)
        if partition == "validation":
            # GEPA evaluates validation examples to rank candidates, but the
            # reflective proposer must not receive their gold labels or traces.
            # Keep only the multi-objective values needed by the optimizer.
            objective_scores = side.get("scores")
            side = {
                "EvaluationPartition": "validation",
                "Feedback": "Held-out validation evaluation; diagnostics withheld.",
            }
            if objective_scores is not None:
                side["scores"] = objective_scores
        else:
            expected = example["options"][int(example["correct_index"])]
            got = (example["options"][outcome.answer_index]
                   if 0 <= outcome.answer_index < len(example["options"]) else "ABSTAIN")
            side.update({
                "EvaluationPartition": "train",
                "Question": str(example["question"])[:400],
                "Options": list(example["options"]),
                "TaskMaterial": feedback_material(example),
                "Expected": expected, "Selected": got,
                "Feedback": ("Correct" if outcome.correct else
                             f"Selected {got}; expected {expected}; status={outcome.status}"),
            })
            side.update(task_trace_side_info(outcome))
            if use_evidence_signals:
                relation_hint = feedback_claim_relation_hint(example)
                if relation_hint:
                    side["TargetRelationHint"] = relation_hint
                side["PossibleNegativeResultSentences"] = feedback_evidence_signals(example)
                side["PossibleTargetResultSentences"] = (
                    feedback_target_result_sentences(example))
        return score, side

    config = build_optimizer(
        {"seed": seed,
           "max_candidate_proposals": max_candidate_proposals,
           "reflection_minibatch_size": reflection_minibatch_size,
           "acceptance_criterion": acceptance_criterion,
           "multi_objective": multi_objective,
           "use_merge": use_merge,
           "merge_max_invocations": merge_max_invocations,
           "merge_val_overlap_floor": merge_val_overlap_floor},
        reflection_lm, str(out_dir / "state"),
        # GEPA checks its stopper between iterations, then may run a complete
        # valset evaluation for an accepted proposal. Reserve one full val pass
        # so the hard external cap can include that evaluation.
        max_metric_calls=optimizer_max_metric_calls)
    feedback_version = (TRACE_FEEDBACK_RULE_VERSION + "+" +
        REFLECTION_GUIDANCE_VERSION + (
        "+" + FEEDBACK_RULE_VERSION if use_evidence_signals else "")
    )
    report = {
        "run_id": run_id, "status": "searching",
        "optimizer": "pinned vendor GEPA optimize_anything",
        "task_model_id": (cfg.get("models") or {}).get("task", {}).get("model_id"),
        "reflection_model_id": reflection_cfg.get("model_id"),
        "input_mode": "task_input", "protocol_sha256": config_fingerprint(cfg),
        "task_input_fingerprint_version": TASK_INPUT_FINGERPRINT_VERSION,
        "max_metric_calls": max_metric_calls,
        "optimizer_max_metric_calls": optimizer_max_metric_calls,
        "validation_reserve": validation_reserve,
        "max_candidate_proposals": max_candidate_proposals,
        "reflection_minibatch_size": reflection_minibatch_size,
        "task_offset": int(gp.get("task_offset", 0)),
        "visible_task_ids": (list(gp["visible_task_ids"])
                             if gp.get("visible_task_ids") is not None else None),
        "acceptance_criterion": acceptance_criterion,
        "multi_objective": multi_objective,
        "objective_selection_policy": (
            OBJECTIVE_SELECTION_POLICY_VERSION if multi_objective else None),
        "objective_names": (["accuracy", "coverage", "answer_quality"]
                            if multi_objective else ["accuracy"]),
        "use_merge": use_merge,
         "merge_max_invocations": merge_max_invocations if use_merge else None,
         "merge_val_overlap_floor": merge_val_overlap_floor if use_merge else None,
        "feedback_rule_version": feedback_version,
        "method_rule_gate_version": METHOD_RULE_GATE_VERSION,
        "task_contract_rule_version": TASK_CONTRACT_RULE_VERSION,
        "candidate_boundary_version": GEPA_CANDIDATE_BOUNDARY_VERSION,
        "candidate_strategy_seed_sha256": sha256_obj(seed_strategy),
        "task_contract_id": (runtime_contract.contract_id
                             if runtime_contract else None),
        "task_contract_sha256": (runtime_contract.sha256
                                 if runtime_contract else None),
        "proposal_task_contract": proposal_task_contract,
        "reflection_system_message_sha256": reflection_system_message_sha256,
        "reflection_label_semantics_sha256": (
            sha256_obj(task_label_semantics) if task_label_semantics else None),
        "optimizer_protocol_sha256": sha256_obj({
            "config": cfg, "feedback_rule_version": feedback_version,
            "method_rule_gate_version": METHOD_RULE_GATE_VERSION,
            "task_contract_rule_version": TASK_CONTRACT_RULE_VERSION,
            "candidate_boundary_version": GEPA_CANDIDATE_BOUNDARY_VERSION,
            "proposal_task_contract": proposal_task_contract,
            "reflection_system_message_sha256": reflection_system_message_sha256,
            "objective_selection_policy": (OBJECTIVE_SELECTION_POLICY_VERSION
                                            if multi_objective else None)}),
        "release_id": state.release_id,
        "train_ids": [x["task_id"] for x in train],
        "val_ids": [x["task_id"] for x in val],
        "selection_gate_read": False,
        "reflection_usage": summarize_chat_usage(reflection_usage),
        "reflection_provider_stats": (
            dict(reflection_provider.stats)
            if isinstance(getattr(reflection_provider, "stats", None), dict) else {}),
        "task_evaluation_cost": total_cost(task_outcomes),
        "task_evaluation_rows": task_evaluation_rows,
        "pre_evaluation_rejections": pre_evaluation_rejections,
    }
    write_json(report_path, report)
    try:
        result = run_reflection(
            seed_candidate=seed_strategy, evaluator=evaluator,
            trainset=train, valset=val, config=config,
            objective="Improve task-grounded scientific question answering on supplied task material.",
            background="Optimize the complete system prompt using task inputs and execution feedback. "
                       "Only task-provided inputs are available at runtime. Do not rely on "
                       "unavailable capabilities or hidden answer information.")
    except MetricBudgetExceeded as exc:
        report.update({
            "status": "budget_exhausted_no_selection",
            "budget_abort_reason": str(exc),
            "optimizer_run_aborted": True,
            "total_metric_calls": None,
            "requested_max_metric_calls": max_metric_calls,
            "evaluator_attempts": metric_attempts,
            "budgeted_metric_calls": max_metric_calls,
            "budgeted_task_evaluations": task_model_evaluations,
            "budget_guard_rejections": 1,
            "val_aggregate_scores": None,
            "best_val_score": None,
            "val_scores_valid": False,
            "reflection_usage": summarize_chat_usage(reflection_usage),
            "reflection_provider_stats": (
                dict(reflection_provider.stats)
                if isinstance(getattr(reflection_provider, "stats", None), dict) else {}),
            "task_evaluation_cost": total_cost(task_outcomes),
        })
        write_json(report_path, report)
        return None
    except Exception as exc:
        # GEPA/vendor failures can occur after full validation has started (for
        # example, a platform-default text encoding rejects a Unicode prompt).
        # Persist an explicit terminal invalid status so partial task rows cannot
        # be mistaken for a completed search or reusable candidate.
        report.update({
            "status": "optimizer_error_no_selection",
            "optimizer_run_aborted": True,
            "optimizer_error_type": type(exc).__name__,
            # Keep only the exception class; str(exc) can contain raw provider
            # response bodies, request details, or credentials.
            "optimizer_error_detail_omitted": True,
            "total_metric_calls": None,
            "requested_max_metric_calls": max_metric_calls,
            "evaluator_attempts": metric_attempts,
            "budgeted_metric_calls": min(metric_attempts, max_metric_calls),
            "budgeted_task_evaluations": task_model_evaluations,
            "val_scores_valid": False,
            "val_aggregate_scores": None,
            "val_aggregate_subscores": None,
            "best_val_score": None,
            "selected_idx": None,
            "selected_sha256": None,
            "task_evaluation_rows": [],
            "selection_gate_read": False,
            "reflection_usage": summarize_chat_usage(reflection_usage),
            "reflection_provider_stats": (
                dict(reflection_provider.stats)
                if isinstance(getattr(reflection_provider, "stats", None), dict) else {}),
            "task_evaluation_cost": total_cost(task_outcomes),
        })
        if isinstance(exc, TaskExecutionFailure):
            report["task_execution_failure"] = dict(exc.failure)
        write_json(report_path, report)
        raise
    best = result.best_candidate
    if isinstance(best, dict):
        best = next(iter(best.values()))
    best_strategy = task_instruction_strategy(best, runtime_contract)
    candidate_texts = [
        task_instruction_strategy(
            str(next(iter(x.values()))) if isinstance(x, dict) else str(x),
            runtime_contract)
        for x in result.candidates]
    candidate_runtime_prompts = [
        compose_task_instruction(text, runtime_contract) for text in candidate_texts]
    report.update({
        "status": "selected", "best_sha256": sha256_obj(best_strategy),
        "best_strategy_sha256": sha256_obj(best_strategy),
        "best_runtime_prompt_sha256": sha256_obj(
            compose_task_instruction(best_strategy, runtime_contract)),
        "best_val_score": result.val_aggregate_scores[result.best_idx],
        "n_candidates": len(result.candidates),
        "candidate_texts": candidate_texts,
        "candidate_strategy_sha256": [sha256_obj(x) for x in candidate_texts],
        "candidate_runtime_prompts": candidate_runtime_prompts,
        "candidate_runtime_prompt_sha256": [
            sha256_obj(x) for x in candidate_runtime_prompts],
        "val_aggregate_scores": list(result.val_aggregate_scores),
        "val_aggregate_subscores": (
            [dict(x) for x in result.val_aggregate_subscores]
            if getattr(result, "val_aggregate_subscores", None) is not None else None),
        "total_metric_calls": result.total_metric_calls,
        "requested_max_metric_calls": max_metric_calls,
        "evaluator_attempts": metric_attempts,
        "budgeted_metric_calls": min(metric_attempts, max_metric_calls),
        "budgeted_task_evaluations": task_model_evaluations,
        "budget_guard_rejections": max(0, metric_attempts - max_metric_calls),
        "reflection_usage": summarize_chat_usage(reflection_usage),
        "reflection_provider_stats": (
            dict(reflection_provider.stats)
            if isinstance(getattr(reflection_provider, "stats", None), dict) else {}),
        "task_evaluation_cost": total_cost(task_outcomes),
        "vendor_run_log": str(out_dir / "state" / "run_log.json"),
    })
    report.update(summarize_vendor_proposals(out_dir / "state" / "run_log.json"))
    if metric_attempts > max_metric_calls:
        report["status"] = "budget_exhausted_no_selection"
        # This should be unreachable because the evaluator aborts on overrun.
        # Keep a defensive invalidation in case an adapter swallows the abort.
        report["val_scores_valid"] = False
        report["val_aggregate_scores"] = None
        report["best_val_score"] = None
        write_json(report_path, report)
        return None
    report["val_scores_valid"] = True
    objective_scores = (getattr(result, "val_aggregate_subscores", None)
                        if multi_objective else None)
    if multi_objective and not objective_scores:
        report.update({"status": "invalid_objective_validation",
                       "val_scores_valid": False,
                       "objective_selection_error":
                           "multiobjective search returned no validation objective scores"})
        write_json(report_path, report)
        return None
    try:
        selection_order = ranked_challenger_indices(
            result.val_aggregate_scores, objective_scores)
    except ValueError as exc:
        report.update({"status": "invalid_objective_validation",
                       "val_scores_valid": False,
                       "objective_selection_error": str(exc)})
        write_json(report_path, report)
        return None
    selected_idx = None
    selected_text = ""
    rejected: List[Dict[str, Any]] = []
    for idx in selection_order:
        strategy = candidate_texts[idx].strip()
        proposed = candidate_runtime_prompts[idx]
        if not strategy or not proposed or proposed == state.system_prompt.strip():
            continue
        task_contract_gate = validate_task_contract_candidate(proposed, train + val)
        overlap = training_question_overlap(strategy, train + val)
        label_conflict = task_label_contract_conflict(proposed, train + val)
        reason = validate_rewrite(proposed, state.system_prompt, cfg)
        method_rules = validate_statistical_method_rules(proposed)
        trial = Candidate(
            candidate_id=content_id("cand", state.release_id, "system_prompt", proposed),
            parent_release_id=state.release_id, artifact_type="system_prompt",
            artifact_id="system_prompt", diff=unified_diff(state.system_prompt, proposed),
            new_content=proposed)
        gate = static_gate(trial, cfg, state)
        if (not task_contract_gate["passed"] or overlap or label_conflict or reason
                or not method_rules["passed"]
                or not gate["passed"]):
            rejected.append({"index": idx,
                             "reason": (None if task_contract_gate["passed"] else
                                        task_contract_gate["reason"])
                             or overlap or label_conflict or reason or
                             (None if method_rules["passed"] else
                              method_rules["reason"]) or
                             ", ".join(x["check"] for x in gate["results"]
                                       if not x["passed"])})
            continue
        selected_idx, selected_text = idx, proposed
        break
    report["final_static_rejections"] = rejected
    if selected_idx is None:
        report["status"] = "no_change"
        write_json(report_path, report)
        return None
    tied = float(result.val_aggregate_scores[selected_idx]) <= float(
        result.val_aggregate_scores[0]) + 1e-9
    report.update({"status": "selected_tied_challenger" if tied else "selected",
                   "selected_idx": selected_idx,
                   "selected_strategy_sha256": sha256_obj(candidate_texts[selected_idx]),
                   "selected_runtime_prompt_sha256": sha256_obj(selected_text),
                   "selected_sha256": sha256_obj(selected_text),
                   "selected_val_score": result.val_aggregate_scores[selected_idx],
                   "selected_val_subscores": (
                       dict(result.val_aggregate_subscores[selected_idx])
                       if getattr(result, "val_aggregate_subscores", None) is not None
                       else None),
                   "selection_policy": (
                       "accuracy-primary Pareto front ranked by maximin objective delta"
                       if multi_objective else
                       "best nonbaseline dev validation, including ties")})
    cand = Candidate(
        candidate_id=content_id("cand", state.release_id, "system_prompt", selected_text),
        parent_release_id=state.release_id,
        artifact_type="system_prompt", artifact_id="system_prompt",
        diff=unified_diff(state.system_prompt, selected_text),
        rationale=("Pinned GEPA Pareto tie challenger on visible dev validation"
                   if tied else "Pinned GEPA selected a full Prompt replacement on visible task inputs"),
        status="draft", new_content=selected_text, attributed_class="plan.failed",
        source_failure_ids=list(failure_ids),
        generator={"backend": "pinned-gepa", "is_gepa": True,
                   "search_report": str(report_path),
                   "n_candidates": len(result.candidates),
                   "selected_idx": selected_idx,
                   "validation_tie": tied,
                   "total_metric_calls": result.total_metric_calls},
        content_sha256=content_sha256(selected_text))
    write_json(report_path, report)
    return cand
