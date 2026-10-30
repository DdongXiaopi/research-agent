"""Thin DSPy MIPROv2 adapter that keeps task execution inside Pi TaskRunner.

The optional ``dspy[optuna]`` dependency is loaded lazily. MIPRO may optimize
the predictor instruction and labeled demonstrations, but this module never
answers a task through a second LM path: each evaluation calls TaskRunner in
``task_input`` mode with the frozen task evidence.
"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Sequence
import hashlib
import importlib
import importlib.metadata
import inspect
import threading

from ..runtime.call_ledger import (record_operation_end, reserve_physical_attempts,
                                   reset_active_operation, set_active_operation,
                                   start_operation)
from ..runtime.runner import ArtifactState, TaskRunner
from ..runtime.provider import (published_price_for_endpoint,
                                published_tariff_metadata,
                                HttpProvider, ProviderError)
from ..runtime.skill_routes import FrozenSkillRouteCache
from ..eval.task_rows import validate_task_answer_row
from .fewshot import parse_demos_bundle, serialize_demos_bundle
from .task_contracts import TaskContract, compose_task_instruction

MIPRO_SELECTION_POLICY_VERSION = "class-aware-search-validation-v2"
MIPRO_VIEW_DATA_BATCH_SIZE = 10
MIPRO_TASK_INPUT_PREDICTOR_COUNT = 1
_MIPRO_PROPOSER_HOOK_LOCK = threading.Lock()
_MIPRO_TASK_CONTRACT_CONTEXT = ContextVar(
    "mipro_task_contract_context", default=None)
MIPRO_PROPOSER_GUIDANCE = {
    "program_aware_proposer": True,
    "data_aware_proposer": True,
    # Vendor DSPy offers a generic "high stakes" tip that produced an
    # unsupported clinical-impact framing in the SciFact exploration. Scientific
    # task instructions should be grounded in the supplied examples instead.
    "tip_aware_proposer": False,
    "fewshot_aware_proposer": True,
}


def validate_mipro_optimizer_runtime() -> dict[str, str]:
    """Fail before paid calls when active dependencies cannot run vendored DSPy."""
    found: dict[str, str] = {}
    for package in ("dspy", "openai"):
        try:
            found[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"MIPRO dependency {package} is not installed; use the "
                "project-locked optimizer environment") from exc
    from packaging.version import Version
    if Version(found["openai"]) < Version("1.66.2"):
        raise RuntimeError(
            f"MIPRO dependency openai=={found['openai']} is below the "
            "vendored DSPy minimum 1.66.2; use the project-locked "
            "optimizer environment")
    return found


def mipro_proposer_guidance(*, prompt_only: bool = False) -> dict[str, bool]:
    """Return the frozen task-input proposal guidance used by MIPROv2."""
    guidance = dict(MIPRO_PROPOSER_GUIDANCE)
    if prompt_only:
        guidance["fewshot_aware_proposer"] = False
    return guidance


def _proposal_context_report(task_contract_context: str | None,
                             optimizer: Any) -> dict[str, Any]:
    """Bind the planned immutable prefix to digests of realized proposal input."""
    observations = list(getattr(
        optimizer, "_rsi_proposal_context_observations", []) or [])
    if not task_contract_context:
        return {
            "context_sha256": None,
            "injected_into_data_aware_proposer": False,
            "data_summary_observation_schema": None,
            "data_summary_observations": [],
        }
    return {
        "context_sha256": hashlib.sha256(
            task_contract_context.strip().encode("utf-8")).hexdigest(),
        "injected_into_data_aware_proposer": True,
        "data_summary_observation_schema": "mipro-generated-summary-digest-v1",
        "data_summary_observations": observations,
    }


def proposal_context_failure_report(contract_identity: dict[str, Any],
                                     error: BaseException) -> dict[str, Any]:
    """Retain only captured summary digests when a proposal fails mid-call."""
    observations = getattr(
        error, "rsi_proposal_context_observations", [])
    if not isinstance(observations, (list, tuple)):
        observations = []
    return {
        **contract_identity,
        "data_summary_observations": [
            dict(item) for item in observations if isinstance(item, dict)
        ],
    }


def _canonical_mipro_instruction(text: str) -> str:
    # Signature.instructions exposes inspect.cleandoc(__doc__). Canonical
    # proposal identity must use that same view, including for the incumbent.
    return inspect.cleandoc(str(text or "")).strip()


def _mipro_runtime_prompt(program: Any) -> str:
    """Render the mutable proposal with any program-owned frozen contract."""
    instruction = str(program.predict.signature.instructions or "")
    parent = getattr(program, "base_state", None)
    if parent is not None and hasattr(parent, "system_prompt"):
        original = str(parent.system_prompt or "")
        if _canonical_mipro_instruction(instruction) == _canonical_mipro_instruction(original):
            instruction = original
    return compose_task_instruction(instruction, getattr(program, "task_contract", None))


def require_mipro_v2():
    """Load the locked upstream optimizer and fail with an actionable message."""
    try:
        import dspy
        from dspy.teleprompt import MIPROv2
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "MIPROv2 requires requirements-mipro.txt (locked vendor DSPy + optuna)") from exc
    if getattr(dspy, "__version__", None) != "3.4.0b1":
        raise RuntimeError(
            "MIPROv2 adapter is locked to vendored DSPy 3.4.0b1; "
            f"found {getattr(dspy, '__version__', 'unknown')}")
    pinned = (Path(__file__).resolve().parents[2]
              / "vendor/dspy-main/dspy-main/dspy")
    optimizer_module = importlib.import_module("dspy.teleprompt.mipro_optimizer_v2")
    for module, relative in ((dspy, "__init__.py"),
                             (optimizer_module, "teleprompt/mipro_optimizer_v2.py")):
        loaded_path = getattr(module, "__file__", None)
        if not loaded_path or Path(loaded_path).resolve() != (pinned / relative).resolve():
            raise RuntimeError(
                "MIPROv2 requires the vendored DSPy source; "
                "same-version packages from another installation are not the frozen backend")
    if optimizer_module.MIPROv2 is not MIPROv2:
        raise RuntimeError("MIPROv2 export does not match the vendored DSPy source")
    return dspy, MIPROv2, optuna


def build_mipro_examples(tasks: Sequence[dict], *, allowed_ids: set[str] | None = None) -> list:
    """Map frozen task rows to DSPy examples with only train labels as targets."""
    dspy, _, _ = require_mipro_v2()
    examples = []
    seen = set()
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        options = task.get("options") or []
        answer_index = task.get("correct_index")
        if not task_id or task_id in seen:
            raise ValueError("MIPRO task IDs must be non-empty and unique")
        if allowed_ids is not None and task_id not in {str(value) for value in allowed_ids}:
            raise ValueError(f"MIPRO task is outside the caller's frozen ID scope: {task_id}")
        if task.get("split") != "dev_optimize":
            raise ValueError("MIPRO task examples must come from dev_optimize")
        if type(answer_index) is not int or not 0 <= answer_index < len(options):
            raise ValueError(f"MIPRO task has an invalid gold option: {task_id}")
        if not str(task.get("question") or "").strip():
            raise ValueError(f"MIPRO task is missing question text: {task_id}")
        evidence = str(task.get("evidence") or task.get("context") or "")
        if not evidence.strip():
            raise ValueError(f"MIPRO task is missing supplied evidence: {task_id}")
        seen.add(task_id)
        examples.append(dspy.Example(
            task_id=task_id, question=str(task["question"]), evidence=evidence,
            options=list(options), answer_index=answer_index,
        ).with_inputs("task_id", "question", "evidence", "options"))
    if not examples:
        raise ValueError("MIPRO requires at least one training task")
    return examples


def make_mipro_accuracy_metric() -> Callable:
    """Return the optimizer metric; class-wise gates remain an outer project gate."""
    def metric(example, prediction, trace=None):
        try:
            predicted = int(prediction.answer_index)
            expected = int(example.answer_index)
        except (AttributeError, TypeError, ValueError):
            return False
        return predicted == expected
    return metric


def summarize_mipro_eval_results(results: Sequence[Any]) -> dict[str, Any]:
    """Summarize DSPy results and preserve task-level predictions for audit."""
    by_label: dict[str, dict[str, int]] = {}
    total = {"n": 0, "answered": 0, "correct": 0}
    items: list[dict[str, Any]] = []
    for item in results:
        try:
            example, prediction, score = item
            options = list(getattr(example, "options", []) or [])
            gold_index = int(getattr(example, "answer_index"))
            predicted_index = int(getattr(prediction, "answer_index"))
            if not 0 <= gold_index < len(options):
                raise ValueError("MIPRO evaluation example has an invalid gold option")
            label = str(options[gold_index])
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("MIPRO evaluator returned malformed per-example results") from exc
        bucket = by_label.setdefault(label, {"n": 0, "answered": 0, "correct": 0})
        answered = 0 <= predicted_index < len(options)
        correct = bool(score)
        bucket["n"] += 1
        total["n"] += 1
        if answered:
            bucket["answered"] += 1
            total["answered"] += 1
        if correct:
            bucket["correct"] += 1
            total["correct"] += 1
        items.append({
            "task_id": str(getattr(example, "task_id", "")),
            "gold_answer_index": gold_index,
            "gold_label": label,
            "predicted_answer_index": predicted_index,
            "predicted_label": (str(options[predicted_index]) if answered else None),
            "answered": answered,
            "correct": correct,
            "selected_skill": str(getattr(prediction, "selected_skill", "") or ""),
        })
    if not total["n"]:
        raise ValueError("MIPRO evaluator returned no per-example results")

    def axes(bucket: dict[str, int]) -> dict[str, float | int]:
        n, answered, correct = bucket["n"], bucket["answered"], bucket["correct"]
        return {
            "n": n,
            "accuracy": correct / n,
            "coverage": answered / n,
            "answered_precision": correct / answered if answered else 0.0,
        }

    return {**axes(total), "per_gold_label": {
        label: axes(values) for label, values in sorted(by_label.items())},
        "items": items}


def mipro_class_aware_gain(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    """Mirror the project's selection gate on nested search-validation only."""
    if float(candidate.get("accuracy", -1)) <= float(baseline.get("accuracy", -1)) + 1e-12:
        return False
    candidate_labels = candidate.get("per_gold_label") or {}
    baseline_labels = baseline.get("per_gold_label") or {}
    if not baseline_labels or set(candidate_labels) != set(baseline_labels):
        return False
    for label, base in baseline_labels.items():
        current = candidate_labels[label]
        for axis in ("accuracy", "coverage", "answered_precision"):
            if float(current.get(axis, -1)) + 1e-12 < float(base.get(axis, 0)):
                return False
    return True


class MiproTaskBudget:
    """Shared hard cap across all DSPy deep-copied candidate programs."""

    def __init__(self, limit: int):
        if limit <= 0:
            raise ValueError("MIPRO task-evaluation budget must be positive")
        self.limit = int(limit)
        self.calls = 0
        self.exceeded = False
        self.failure = ""
        self.failure_detail: dict[str, Any] | None = None

    def __deepcopy__(self, memo):
        # MIPRO deep-copies its student for each trial; the evaluator budget
        # must remain global to the compile call.
        return self

    def reserve(self) -> bool:
        if self.calls >= self.limit:
            self.exceeded = True
            return False
        self.calls += 1
        return True

    def record_failure(self, task_id: str, exc: Exception) -> None:
        """Keep task execution failures separate from model score outcomes."""
        detail: dict[str, Any] = {
            "task_id": str(task_id),
            "error_type": type(exc).__name__,
            "error": str(exc)[:1200],
            "task_evaluation_calls_so_far": self.calls,
            "task_evaluation_budget": self.limit,
        }
        attempts = getattr(exc, "provider_http_attempts", None)
        if isinstance(attempts, int) and attempts >= 0:
            detail["provider_http_attempts"] = attempts
        statuses = getattr(exc, "provider_statuses", None)
        if isinstance(statuses, (list, tuple)):
            detail["provider_statuses"] = [int(value) for value in statuses
                                            if isinstance(value, int)]
        provider_error = getattr(exc, "provider_error", None)
        if isinstance(provider_error, dict):
            detail["provider_error"] = {
                key: str(provider_error[key])[:512]
                for key in ("code", "type", "message")
                if isinstance(provider_error.get(key), (str, int))
            }
        self.failure_detail = detail
        self.failure = f"{detail['error_type']}: {detail['error']}"


class MiproTaskEvaluationError(RuntimeError):
    """Raised when infrastructure failure made an optimizer score incomplete."""

    def __init__(self, failure_detail: dict[str, Any]):
        self.failure_detail = dict(failure_detail)
        task_id = self.failure_detail.get("task_id", "unknown")
        error = self.failure_detail.get("error", "task evaluation failed")
        super().__init__(f"MIPRO task evaluation failed for {task_id}: {error}")


def _fail_closed_candidate_eval(delegate: Callable, budget: MiproTaskBudget,
                                *args, **kwargs):
    """Undo DSPy's catch-all zero-score behavior for failed task executions."""
    result = delegate(*args, **kwargs)
    if budget.failure_detail is not None:
        raise MiproTaskEvaluationError(budget.failure_detail)
    return result


def make_gated_mipro_v2(candidate_validator: Callable[[str], Any],
                        train_ids: set[str],
                        program_validator: Callable[[Any], Any] | None = None,
                        balanced_demo_labels: Sequence[str] = (),
                        require_balanced_demo_bundle: bool = False,
                        allow_empty_demo_candidate: bool = False,
                        task_contract_context: str | None = None,
                        candidate_composer: Callable[[str], str] | None = None,
                        frozen_empty_demo_pool: bool = False):
    """Subclass the pinned upstream extension point to gate instructions pre-eval."""
    _, upstream_mipro, optuna = require_mipro_v2()

    class GatedMIPROv2(upstream_mipro):
        def _propose_instructions(
                self, program, trainset, demo_candidates, view_data_batch_size,
                program_aware_proposer, data_aware_proposer, tip_aware_proposer,
                fewshot_aware_proposer, num_instruct_candidates):
            if task_contract_context and not data_aware_proposer:
                raise ValueError(
                    "MIPRO task contract context requires the upstream data-aware proposer")
            # Pinned DSPy sets its proposal count to len(demo_candidates[0])
            # when a demo pool exists. A frozen pool containing just [[]] thus
            # collapses N requested instructions to one incumbent slot. Pass
            # no demo candidates to the proposer while retaining [[]] for the
            # later, truly frozen optimizer search space.
            proposal_demos = None if frozen_empty_demo_pool else demo_candidates
            proposal_fewshot_aware = (
                False if frozen_empty_demo_pool else fewshot_aware_proposer)
            if task_contract_context:
                # DSPy 3.4.0b1 passes GroundedProposer.data_summary as the
                # `dataset_description` field to its instruction generator.
                # Add the frozen contract there so proposal generation sees it.
                # Runtime evaluation composes the same immutable contract after
                # each strategy, independent of what the proposer returns.
                # The pinned optimizer imports this class at module scope, so
                # serialize the temporary hook across concurrent searches.
                with _MIPRO_PROPOSER_HOOK_LOCK:
                    mipro_module = importlib.import_module(
                        "dspy.teleprompt.mipro_optimizer_v2")
                    upstream_proposer = mipro_module.GroundedProposer
                    context = str(task_contract_context).strip()
                    context_observations = []

                    class ContractAwareGroundedProposer(upstream_proposer):
                        def __init__(self, *args, **kwargs):
                            super().__init__(*args, **kwargs)
                            # This class is temporarily visible through the
                            # upstream module global. Only the logical proposal
                            # call that installed the contract may receive it;
                            # unrelated concurrent DSPy users retain the
                            # ordinary dataset-only summary.
                            proposal_context = _MIPRO_TASK_CONTRACT_CONTEXT.get()
                            if proposal_context is None:
                                return
                            summary = getattr(self, "data_summary", None)
                            if (not getattr(self, "use_dataset_summary", False)
                                    or not isinstance(summary, str) or not summary.strip()):
                                raise RuntimeError(
                                    "MIPRO could not build the data-aware proposal context")
                            train_summary = summary.strip()
                            self.data_summary = (
                                proposal_context + "\n\nDataset summary:\n"
                                + train_summary)
                            # The summary is generated by DSPy's upstream LM and
                            # affects every instruction proposal. Keep a safe
                            # digest of the exact text passed downstream so the
                            # compile report can identify the realized context
                            # without persisting potentially sensitive content.
                            context_observations.append({
                                "dataset_summary_sha256": hashlib.sha256(
                                    train_summary.encode("utf-8")).hexdigest(),
                                "dataset_summary_char_count": len(train_summary),
                                "proposal_generator_data_summary_sha256": hashlib.sha256(
                                    self.data_summary.encode("utf-8")).hexdigest(),
                                "proposal_generator_data_summary_char_count": len(
                                    self.data_summary),
                            })

                    context_token = _MIPRO_TASK_CONTRACT_CONTEXT.set(context)
                    mipro_module.GroundedProposer = ContractAwareGroundedProposer
                    try:
                        proposals = super()._propose_instructions(
                            program, trainset, proposal_demos, view_data_batch_size,
                            program_aware_proposer, data_aware_proposer, tip_aware_proposer,
                            proposal_fewshot_aware, num_instruct_candidates)
                    except BaseException as exc:
                        # Preserve only safe digests on the original failure so
                        # the runner can attach realized input provenance to
                        # its fail-closed terminal report.
                        try:
                            setattr(exc, "rsi_proposal_context_observations",
                                    list(context_observations))
                        except (AttributeError, TypeError):
                            pass
                        raise
                    finally:
                        mipro_module.GroundedProposer = upstream_proposer
                        _MIPRO_TASK_CONTRACT_CONTEXT.reset(context_token)
                        self._rsi_proposal_context_observations = list(
                            context_observations)
            else:
                proposals = super()._propose_instructions(
                    program, trainset, proposal_demos, view_data_batch_size,
                    program_aware_proposer, data_aware_proposer, tip_aware_proposer,
                    proposal_fewshot_aware, num_instruct_candidates)
            gated = {}
            proposal_audit = getattr(self, "_rsi_instruction_proposal_audit", [])
            for index, candidates in proposals.items():
                accepted = []
                incumbent_text = str(getattr(
                    getattr(getattr(program, "predict", None), "signature", None),
                    "instructions", "") or "").strip()
                for proposal_index, text in enumerate(candidates):
                    if not isinstance(text, str) or not text.strip():
                        proposal_audit.append({
                            "predictor_index": int(index),
                            "proposal_index": proposal_index,
                            "decision": "rejected",
                            "reason": "proposal_must_be_a_nonempty_string",
                            "observed_type": type(text).__name__,
                        })
                        continue
                    candidate_text = text
                    canonical_strategy = _canonical_mipro_instruction(candidate_text)
                    proposal_hash = hashlib.sha256(
                        canonical_strategy.encode("utf-8")).hexdigest()
                    # Whitespace-only variants of the incumbent are not novel
                    # instructions and should not consume a MIPRO search slot.
                    if canonical_strategy == _canonical_mipro_instruction(incumbent_text):
                        proposal_audit.append({
                            "predictor_index": int(index),
                            "proposal_index": proposal_index,
                            "instruction_sha256": proposal_hash,
                            "decision": "rejected",
                            "reason": "duplicate_of_incumbent_after_whitespace_normalization",
                        })
                        continue
                    # The optimizer controls only its mutable text. Review the
                    # exact runtime instruction after program-owned contracts
                    # have been composed deterministically.
                    checked_text = (candidate_composer(candidate_text)
                                    if candidate_composer else candidate_text)
                    result = candidate_validator(checked_text)
                    runtime_instruction_hash = hashlib.sha256(
                        _canonical_mipro_instruction(checked_text).encode("utf-8")
                    ).hexdigest()
                    # Keep the optimizer's mutable signature identity distinct
                    # from the composed runtime instruction reviewed by gates.
                    # DSPy trial logs identify the former; safety gates identify
                    # the latter. Coverage joins on the former and retains both.
                    validator_records = getattr(candidate_validator, "records", None)
                    if isinstance(validator_records, list) and validator_records:
                        gate_record = validator_records[-1]
                        if isinstance(gate_record, dict):
                            gate_record["proposal_instruction_sha256"] = proposal_hash
                            gate_record["proposal_instruction_hash_scope"] = (
                                "canonical_mutable_dspy_instruction")
                            gate_record["runtime_instruction_sha256"] = (
                                runtime_instruction_hash)
                    passed = (result is True or
                              isinstance(result, dict) and result.get("passed") is True)
                    proposal_audit.append({
                        "predictor_index": int(index),
                        "proposal_index": proposal_index,
                        "instruction_sha256": proposal_hash,
                        "instruction_hash_scope": "canonical_mutable_dspy_instruction",
                        "runtime_instruction_sha256": runtime_instruction_hash,
                        "decision": "accepted" if passed else "rejected",
                        "reason": ("all_pre_evaluation_gates_passed" if passed else
                                   str(result.get("reason") or "candidate_gate_rejected")
                                   if isinstance(result, dict) else "candidate_gate_rejected"),
                    })
                    if passed and candidate_text.strip() not in {
                            item.strip() for item in accepted}:
                        accepted.append(candidate_text)
                if not accepted:
                    if frozen_empty_demo_pool:
                        self._rsi_instruction_proposal_audit = proposal_audit
                        error = ValueError(
                            "prompt-only MIPRO produced no novel instruction "
                            "passing pre-evaluation gates")
                        error.rsi_instruction_proposal_audit = list(proposal_audit)
                        raise error
                    # Preserve a legitimate few-shot-only search when all
                    # generated instructions are duplicates or fail gates.
                    if incumbent_text:
                        accepted.append(incumbent_text)
                        proposal_audit.append({
                            "predictor_index": int(index),
                            "proposal_index": None,
                            "instruction_sha256": hashlib.sha256(
                                incumbent_text.encode("utf-8")).hexdigest(),
                            "decision": "fallback",
                            "reason": "no_novel_instruction_survived_pre_evaluation_gates",
                        })
                    else:
                        self._rsi_instruction_proposal_audit = proposal_audit
                        raise ValueError(
                            f"MIPRO produced no instruction passing static/grounding gates for predictor {index}")
                gated[index] = accepted
            self._rsi_instruction_proposal_audit = proposal_audit
            return gated

        def _bootstrap_fewshot_examples(
                self, program, trainset, seed, teacher, *, num_fewshot_candidates,
                max_bootstrapped_demos, max_labeled_demos, max_errors,
                metric_threshold):
            if max_bootstrapped_demos != 0:
                raise ValueError(
                    "Pi task-input MIPRO must not synthesize bootstrapped rationales; "
                    "use gold-labeled train demonstrations only")
            if frozen_empty_demo_pool:
                predictors=list(program.predictors())
                if not predictors or any(getattr(p,"demos",[]) for p in predictors):
                    raise ValueError("Prompt-only MIPRO requires an unchanged empty parent demo pool")
                self._rsi_demo_candidate_pool={"source":"frozen_empty_parent_demo_pool",
                    "candidate_set_count":len(predictors),"require_balanced_demo_bundle":False,
                    "predictors":{str(i):[{"task_ids":[],"demo_count":0,"class_coverage":0,
                        "label_counts":{},"balanced_class_bundle":False}] for i in range(len(predictors))}}
                return {i:[[]] for i in range(len(predictors))}
            # DSPy 3.4.0b1's helper always emits three base sets (zero-shot,
            # labeled-only, and unshuffled labeled). Any value above three
            # enters its shuffled BootstrapFewShot branch, which requires a
            # positive max_bootstrapped_demos. We add extra class-balanced
            # labeled sets below, so keep the upstream call in its supported
            # no-synthetic-rationale mode.
            upstream_candidate_sets = min(int(num_fewshot_candidates), 3)
            demo_candidates = super()._bootstrap_fewshot_examples(
                program, trainset, seed, teacher,
                num_fewshot_candidates=upstream_candidate_sets,
                max_bootstrapped_demos=0, max_labeled_demos=max_labeled_demos,
                max_errors=max_errors, metric_threshold=metric_threshold)
            labels = tuple(str(label) for label in balanced_demo_labels)
            if labels and max_labeled_demos >= len(labels):
                examples_by_label = {label: [] for label in labels}
                for example in sorted(trainset, key=lambda item: str(getattr(item, "task_id", ""))):
                    options = list(getattr(example, "options", []) or [])
                    answer_index = getattr(example, "answer_index", None)
                    if type(answer_index) is int and 0 <= answer_index < len(options):
                        label = str(options[answer_index])
                        if label in examples_by_label:
                            examples_by_label[label].append(example)
                if all(examples_by_label[label] for label in labels):
                    for predictor_sets in (demo_candidates or {}).values():
                        existing = {
                            tuple(str(getattr(item, "task_id", "")) for item in demo_set)
                            for demo_set in predictor_sets or []
                        }
                        for candidate_index in range(max(1, num_fewshot_candidates)):
                            demo_set = [
                                examples_by_label[label][candidate_index % len(examples_by_label[label])]
                                for label in labels
                            ]
                            key = tuple(str(getattr(item, "task_id", "")) for item in demo_set)
                            if key not in existing:
                                predictor_sets.append(demo_set)
                                existing.add(key)
            for predictor_sets in (demo_candidates or {}).values():
                for demo_set in predictor_sets or []:
                    for example in demo_set:
                        if str(getattr(example, "task_id", "")) not in frozen_train_ids:
                            raise ValueError("MIPRO produced a demonstration outside train IDs")
            # Optimizer may choose an empty set only when the parent has no
            # demos; that yields a Prompt-only candidate. Nonempty labeled
            # sets yield demos-only candidates only when the Prompt is unchanged.
            def has_all_labels_once(demo_set) -> bool:
                if not labels or len(demo_set) != len(labels):
                    return False
                observed = []
                for example in demo_set:
                    options = list(getattr(example, "options", []) or [])
                    answer_index = getattr(example, "answer_index", None)
                    if type(answer_index) is not int or not 0 <= answer_index < len(options):
                        return False
                    observed.append(str(options[answer_index]))
                return len(observed) == len(set(observed)) and set(observed) == set(labels)

            filtered = {}
            for index, predictor_sets in (demo_candidates or {}).items():
                eligible = [demo_set for demo_set in predictor_sets if demo_set]
                if allow_empty_demo_candidate:
                    # With an empty parent demo bundle, the empty optimizer
                    # choice permits Prompt-only MIPRO candidates. Nonempty
                    # proposals that change both components are separately
                    # rejected by the parent-relative program gate.
                    eligible.insert(0, [])
                if require_balanced_demo_bundle:
                    eligible = [demo_set for demo_set in eligible
                                if not demo_set or has_all_labels_once(demo_set)]
                if eligible:
                    filtered[index] = eligible
            if not filtered or len(filtered) != len(demo_candidates or {}):
                raise ValueError("MIPRO produced no eligible labeled train-demo candidate")
            pool_audit = {}
            for index, predictor_sets in filtered.items():
                audited_sets = []
                for demo_set in predictor_sets:
                    label_counts = {}
                    task_ids = []
                    for example in demo_set:
                        task_id = str(getattr(example, "task_id", ""))
                        answer_index = int(getattr(example, "answer_index"))
                        options = list(getattr(example, "options", []) or [])
                        label = (str(options[answer_index])
                                 if 0 <= answer_index < len(options)
                                 else str(answer_index))
                        task_ids.append(task_id)
                        label_counts[label] = label_counts.get(label, 0) + 1
                    audited_sets.append({
                        "task_ids": task_ids,
                        "demo_count": len(task_ids),
                        "class_coverage": len(label_counts),
                        "label_counts": dict(sorted(label_counts.items())),
                        # Audit actual labels, not whether this exact task ID
                        # ordering was generated by our supplemental builder.
                        # DSPy may already supply the same balanced labels in
                        # a different order.
                        "balanced_class_bundle": has_all_labels_once(demo_set),
                    })
                pool_audit[str(index)] = audited_sets
            self._rsi_demo_candidate_pool = {
                "source": "locked_DSPy_MIPROv2_labeled_train_demo_candidates",
                "require_balanced_demo_bundle": require_balanced_demo_bundle,
                "candidate_set_count": sum(len(items) for items in pool_audit.values()),
                "predictors": pool_audit,
            }
            return filtered

        def _select_and_insert_instructions_and_demos(
                self, candidate_program, instruction_candidates, demo_candidates,
                trial, trial_logs, trial_num):
            selected = super()._select_and_insert_instructions_and_demos(
                candidate_program, instruction_candidates, demo_candidates,
                trial, trial_logs, trial_num)
            candidate_program._rsi_mipro_trial_metadata = {
                "console_trial_number": int(trial_num) + 1,
                "optimizer_trial_number": int(getattr(trial, "number", trial_num - 1)),
                "parameters": dict(getattr(trial, "params", {}) or {}),
                "chosen_parameters": list(selected),
            }
            if program_validator is not None:
                result = program_validator(candidate_program)
                passed = (result is True or isinstance(result, dict)
                          and result.get("passed") is True)
                if not passed:
                    gate_result = result if isinstance(result, dict) else {}
                    static_gate = gate_result.get("static_gate") or {}
                    failed_checks = [
                        {"check": str(item.get("check") or "unknown"),
                         "detail": str(item.get("detail") or "")[:500]}
                        for item in static_gate.get("results", [])
                        if isinstance(item, dict) and item.get("passed") is False
                    ]
                    trial_logs[trial_num]["rsi_candidate_gate"] = {
                        "passed": False,
                        "reason": str(gate_result.get("reason") or (
                            "static_gate_rejected"
                            if static_gate.get("passed") is False else
                            "task_program_gate_rejected")),
                        "candidate_id": gate_result.get("candidate_id"),
                        "content_sha256": gate_result.get("content_sha256"),
                        "failed_checks": failed_checks,
                        "demo_gate_reason": (
                            (gate_result.get("task_demo_rendering_gate") or {}).get("reason")
                            if isinstance(gate_result.get("task_demo_rendering_gate"), dict)
                            else None),
                        "optimizer_trial_number": getattr(
                            candidate_program, "_rsi_mipro_trial_metadata", {}).get(
                                "optimizer_trial_number"),
                        "chosen_parameters": getattr(
                            candidate_program, "_rsi_mipro_trial_metadata", {}).get(
                                "chosen_parameters", []),
                    }
                    # This combination is inadmissible, but it should not
                    # discard other already-gated combinations in the study.
                    # Optuna records the trial as PRUNED and continues without
                    # running benchmark answers for this program.
                    raise optuna.TrialPruned(
                        "MIPRO assembled task_program failed static/grounding gates before scoring")
            return selected

    # Closure is frozen at construction so a compile cannot change its data scope.
    frozen_train_ids = frozenset(str(value) for value in train_ids)
    if not frozen_train_ids:
        raise ValueError("MIPRO train ID fence cannot be empty")
    GatedMIPROv2.__name__ = "GatedMIPROv2"
    return GatedMIPROv2


class PiTaskInputProgram:
    """Factory for a DSPy Module whose forward path is the project's Pi runner."""

    @staticmethod
    def create(cfg: dict, state: ArtifactState, tasks: Sequence[dict],
               train_ids: set[str], execution_seed: int,
               task_budget: MiproTaskBudget,
               frozen_skill_routes: FrozenSkillRouteCache | None = None,
               task_contract: TaskContract | None = None):
        dspy, _, _ = require_mipro_v2()
        runtime = cfg.get("runtime") or {}
        if (str(runtime.get("provider") or "").lower() != "pi"
                or str(runtime.get("input_mode") or "").lower() != "task_input"):
            raise ValueError("MIPRO adapter requires the project Pi task_input runtime")
        active_tools = set((cfg.get("tools") or {}).get("active") or [])
        if "academic.search" in active_tools:
            raise ValueError("MIPRO task-input evaluation must not use retrieval as optimizer input")
        by_id = {str(task.get("task_id") or ""): task for task in tasks}
        if "" in by_id or len(by_id) != len(tasks):
            raise ValueError("MIPRO task map contains duplicate or invalid IDs")
        train_ids = {str(value) for value in train_ids}
        if not train_ids or not train_ids <= set(by_id):
            raise ValueError("MIPRO source tasks must cover all frozen train IDs")

        class _Program(dspy.Module):
            def __init__(self):
                super().__init__()
                signature = dspy.Signature(
                    "task_id, question, evidence, options -> answer_index"
                ).with_instructions(state.system_prompt)
                self.predict = dspy.Predict(signature)
                self.cfg = cfg
                self.base_state = state
                self.task_contract = task_contract
                self.task_by_id = by_id
                self.train_ids = train_ids
                self.execution_seed = int(execution_seed)
                self.task_budget = task_budget
                self.frozen_skill_routes = frozen_skill_routes
                self.last_outcome = None

            def _runtime_demos(self) -> list[dict]:
                demos = []
                for index, example in enumerate(self.predict.demos):
                    task_id = str(getattr(example, "task_id", ""))
                    if task_id not in self.train_ids:
                        raise ValueError("MIPRO demonstration is outside frozen train IDs")
                    task = self.task_by_id[task_id]
                    try:
                        label = int(getattr(example, "answer_index"))
                    except (TypeError, ValueError) as exc:
                        raise ValueError("MIPRO demonstration has a malformed gold label") from exc
                    if label != int(task["correct_index"]):
                        raise ValueError("MIPRO demonstration label differs from frozen train gold")
                    # Use the frozen task verbatim. The optimizer cannot rewrite
                    # its question, evidence, options, split or answer label.
                    demos.append({"task_id": task_id, "task": task,
                                  "correct_index": label})
                return parse_demos_bundle(serialize_demos_bundle(demos)) if demos else []

            def forward(self, task_id, question, evidence, options):
                task_id = str(task_id)
                task = self.task_by_id.get(task_id)
                if task is None:
                    raise ValueError("MIPRO evaluation task is outside the frozen task cohort")
                if (str(question) != str(task.get("question") or "")
                        or str(evidence) != str(task.get("evidence") or task.get("context") or "")
                        or list(options) != list(task.get("options") or [])):
                    raise ValueError("MIPRO changed frozen task input fields")
                if not self.task_budget.reserve():
                    raise RuntimeError("MIPRO exhausted its shared Pi task-evaluation budget")
                candidate_state = ArtifactState(
                    release_id=self.base_state.release_id,
                    system_prompt=_mipro_runtime_prompt(self),
                    registry=self.base_state.registry,
                    task_demos=self._runtime_demos(),
                )
                try:
                    outcome = TaskRunner(
                        self.cfg, candidate_state, seed=self.execution_seed,
                        frozen_skill_routes=self.frozen_skill_routes,
                    ).run(task)
                    self.last_outcome = validate_task_answer_row(
                        outcome.to_dict(), task,
                        context=f"mipro:{self.execution_seed}:{task_id}",
                        require_reported_gold_index=True)
                except Exception as exc:
                    self.task_budget.record_failure(task_id, exc)
                    raise
                return dspy.Prediction(
                    answer_index=int(outcome.answer_index),
                    selected_skill=str(getattr(outcome, "skill_recalled", "") or ""),
                )

        return _Program()


class _ProjectHttpEngine:
    """Adapt DSPy lm15 requests to the project's metered requests provider."""

    def __init__(self, provider: HttpProvider):
        self.provider = provider

    @staticmethod
    def _text_parts(parts: Any) -> str:
        values = []
        for part in parts or ():
            if getattr(part, "type", None) != "text":
                raise ProviderError("MIPRO proposer only supports text request parts")
            values.append(str(part.text))
        return "".join(values)

    def complete(self, request: Any):
        from dspy.lm15 import Message, Response, TransportError, Usage

        if request.tools:
            raise ProviderError("MIPRO proposer does not support tool calls")
        messages: list[dict[str, str]] = []
        if request.system:
            if isinstance(request.system, str):
                system_text = request.system
            else:
                system_text = self._text_parts(request.system)
            if system_text:
                messages.append({"role": "system", "content": system_text})
        for message in request.messages:
            if message.role not in {"system", "developer", "user", "assistant"}:
                raise ProviderError(
                    f"MIPRO proposer does not support message role {message.role!r}")
            messages.append({"role": message.role,
                             "content": self._text_parts(message.parts)})
        body = self.provider._body(messages)
        config = request.config
        for field_name, wire_name in (
            ("temperature", "temperature"), ("max_tokens", "max_tokens"),
            ("top_p", "top_p"), ("stop", "stop"),
            ("response_format", "response_format"),
        ):
            value = getattr(config, field_name, None)
            if value is not None and value != ():
                body[wire_name] = value
        try:
            payload = self.provider._post(body)
            choice = payload["choices"][0]
            message = choice["message"]
            content = message.get("content")
            if not isinstance(content, str):
                raise ProviderError(
                    "MIPRO proposer response lacks plain-text message content")
        except ProviderError as exc:
            raise TransportError(str(exc)) from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"MIPRO proposer response structure invalid: {exc}") from exc
        usage_payload = payload.get("usage") or {}
        prompt_tokens = usage_payload.get("prompt_tokens")
        completion_tokens = usage_payload.get("completion_tokens")
        total_tokens = usage_payload.get("total_tokens")
        reason = choice.get("finish_reason") or "stop"
        if reason == "tool_calls":
            reason = "tool_call"
        if reason not in {"stop", "length", "tool_call", "content_filter", "error"}:
            reason = "stop"
        return Response(
            id=str(payload.get("id") or "mipro-proposer"),
            model=str(payload.get("model") or self.provider.model_id),
            message=Message.assistant(content), finish_reason=reason,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens,
                        total_tokens=total_tokens),
        )


def make_openai_compatible_lm(cfg: dict, section: str = "reflection"):
    """Build a DSPy LM through the project's metered HTTP transport."""
    dspy, _, _ = require_mipro_v2()
    from dspy.utils.callback import BaseCallback
    model_cfg = (cfg.get("models") or {}).get(section) or {}
    if str(model_cfg.get("provider") or "").lower() != "http":
        raise ValueError(f"MIPRO proposer requires an HTTP-compatible {section} model")
    model_id = str(model_cfg.get("model_id") or "").strip()
    base_url = str(model_cfg.get("base_url") or "").strip()
    key_env = str(model_cfg.get("api_key_env") or "").strip()
    import os
    api_key = os.environ.get(key_env) if key_env else None
    if not model_id or not base_url or not api_key:
        raise RuntimeError(
            f"MIPRO proposer config is incomplete; set the configured {key_env or 'API key'} environment variable")
    call_limit = int(((cfg.get("evolution") or {}).get("mipro") or {}).get(
        "max_proposer_calls", 24))
    if call_limit <= 0:
        raise ValueError("MIPRO proposer-call budget must be positive")

    class CallBudget(BaseCallback):
        def __init__(self):
            self.calls = 0
            self.limit = call_limit
            self.exceeded = False
            self.operations = {}

        def on_lm_start(self, call_id, instance, inputs):
            if self.calls >= self.limit:
                self.exceeded = True
                raise RuntimeError("MIPRO exhausted its proposer-model call budget")
            operation_id = start_operation("mipro_proposer", "dspy", model_id)
            token = set_active_operation(operation_id)
            self.operations[str(call_id)] = (operation_id, token)
            self.calls += 1

        def on_lm_end(self, call_id, outputs, exception=None):
            operation_id, token = self.operations.pop(str(call_id), (None, None))
            record_operation_end(
                operation_id,
                status="failed" if exception is not None else "success",
                error_type=type(exception).__name__ if exception is not None else "",
            )
            if token is not None:
                reset_active_operation(token)

    callback = CallBudget()
    max_retries = int(model_cfg.get("max_retries", 2))
    if not 0 <= max_retries <= 2:
        raise ValueError("MIPRO proposer max_retries must be between 0 and 2")
    # Retry-after-first is converted to total HTTP attempts. DSPy-level retries
    # are disabled so every physical request is emitted by HttpProvider and
    # recorded in the shared request ledger. The preflight reserves the same
    # (logical-call-budget * attempts-per-call) worst-case ceiling.
    provider = HttpProvider(
        model_id=model_id, base_url=base_url, api_key=api_key,
        temperature=float(model_cfg.get("temperature", 0.2)),
        max_tokens=int(model_cfg.get("max_tokens", 1200)),
        timeout=float(model_cfg.get("timeout", 120.0)),
        max_retries=max_retries + 1,
        enable_thinking=bool(model_cfg.get("enable_thinking", False)),
        extra_body=model_cfg.get("extra_body") or {}, cache=False,
    )
    lm = dspy.LM(
        "openai/" + model_id, engine=_ProjectHttpEngine(provider),
        temperature=float(model_cfg.get("temperature", 0.2)),
        max_tokens=int(model_cfg.get("max_tokens", 1200)),
        cache=False, num_retries=0, callbacks=[callback],
    )
    # Expose only counters and a cap; never retain or serialize the API key.
    lm._rsi_call_budget = callback
    lm._rsi_model_id = model_id
    lm._rsi_base_url = base_url
    return lm


def summarize_mipro_usage(usage_tracker: Any, reported_calls: int,
                          model_id: str = "", base_url: str = "") -> dict[str, Any]:
    """Summarize DSPy usage records without treating missing usage as zero."""
    usage_data = getattr(usage_tracker, "usage_data", {}) or {}
    totals_by_model = usage_tracker.get_total_tokens() if usage_tracker else {}
    by_model = {}
    observed = 0
    allowed_fields = {
        "prompt_tokens", "completion_tokens", "total_tokens",
        "cached_tokens", "cache_creation_tokens", "reasoning_tokens",
    }
    for model, entries in usage_data.items():
        entries = list(entries or [])
        observed += len(entries)
        totals = totals_by_model.get(model) or {}
        token_totals = {
            key: value for key, value in totals.items()
            if key in allowed_fields and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
        by_model[str(model)] = {
            "usage_record_count": len(entries),
            "token_totals": token_totals,
        }
    expected = max(0, int(reported_calls))
    complete = observed == expected
    matching = [totals_by_model.get(model) or {}
                for model in totals_by_model
                if model_id and str(model).endswith(model_id)]
    token_totals = matching[0] if len(matching) == 1 else {}
    price = published_price_for_endpoint(model_id, base_url) if model_id and base_url else None
    token_usage_complete = (
        complete and (expected == 0 or
                      "prompt_tokens" in token_totals and "completion_tokens" in token_totals))
    tariff_cost = None
    if price is not None and token_usage_complete:
        tariff_cost = round((float(token_totals.get("prompt_tokens") or 0)
                             * price["prompt"]
                             + float(token_totals.get("completion_tokens") or 0)
                             * price["completion"]) / 1_000_000.0, 9)
    summary = {
        "reported_call_count": expected,
        "observed_usage_record_count": observed,
        "missing_usage_record_count": max(0, expected - observed),
        "usage_records_complete": complete,
        "by_model": by_model,
        "tariff_cost_usd": tariff_cost,
        "tariff_cost_is_account_invoice": False if tariff_cost is not None else None,
    }
    if tariff_cost is None:
        summary["tariff_cost_unavailable_reason"] = (
            "usage records incomplete, token totals ambiguous, or endpoint/model has no published tariff")
    else:
        summary.update(published_tariff_metadata(model_id, base_url))
    return summary


def select_mipro_fewshot_tie_candidate(compiled: Any, train_ids: set[str]):
    """Apply a transparent tie-break for the prompt × labeled-demo experiment.

    Upstream MIPROv2 keeps its initial zero-shot program unless a trial scores
    strictly higher. When a few-shot trial ties that incumbent, the optimizer
    has still evaluated a valid few-shot candidate, but returns zero-shot and
    hides the tied alternative as the winner. Select the highest-ranked
    already-evaluated train-demo candidate only when it is not worse than the
    upstream incumbent. The separate internal confirmation gate still decides
    whether that candidate is usable.
    """
    incumbent_score = getattr(compiled, "score", None)
    if incumbent_score is None:
        return None, {"policy": "upstream_winner_only", "reason": "missing_incumbent_score"}
    try:
        incumbent_score = float(incumbent_score)
    except (TypeError, ValueError):
        return None, {"policy": "upstream_winner_only", "reason": "invalid_incumbent_score"}

    candidates = getattr(compiled, "candidate_programs", None) or []
    eligible = []
    for order, row in enumerate(candidates):
        if not isinstance(row, dict):
            continue
        try:
            score = float(row.get("score"))
        except (TypeError, ValueError):
            continue
        if score + 1e-9 < incumbent_score:
            continue
        program = row.get("program")
        if program is None:
            continue
        artifact = extract_task_program(program, train_ids)
        if not artifact["task_demos"]:
            # A prompt-only program is a valid FR-201 candidate, but it is not
            # a few-shot tie-break candidate over the zero-shot incumbent.
            continue
        demo_rows = [item["task"] for item in artifact["task_demos"]]
        demo_labels = [int(item["correct_index"]) for item in artifact["task_demos"]]
        label_coverage = len(set(demo_labels))
        vote_support = sum(
            int((task.get("metadata") or {}).get("valid_abstract_label_vote_count", 1))
            for task in demo_rows
        )
        eligible.append((score, label_coverage, vote_support, -order, program, artifact))

    if not eligible:
        return None, {
            "policy": "prefer_evaluated_train_demo_on_equal_score",
            "reason": "no_train_demo_candidate_at_or_above_zero_shot_score",
            "incumbent_score": incumbent_score,
        }
    score, label_coverage, vote_support, _, program, artifact = max(
        eligible, key=lambda item: (item[0], item[1], item[2], item[3]))
    return program, {
        "policy": "prefer_evaluated_train_demo_on_equal_score",
        "tie_break_policy": "score_then_train_demo_class_coverage_then_annotation_support_then_order",
        "reason": ("fewshot_candidate_tied_upstream_zero_shot_incumbent"
                   if abs(score - incumbent_score) <= 1e-9
                   else "fewshot_candidate_exceeded_upstream_incumbent"),
        "incumbent_score": incumbent_score,
        "selected_validation_score": score,
        "selected_demo_ids": [str(row["task_id"]) for row in artifact["task_demos"]],
        "selected_demo_class_coverage": label_coverage,
        "selected_demo_annotation_vote_support": vote_support,
    }


def summarize_mipro_trial_log(compiled: Any, train_ids: set[str]) -> dict[str, Any]:
    """Extract auditable trial choices from the pinned optimizer's saved logs."""
    logs = getattr(compiled, "trial_logs", None) or {}
    records = []
    for raw_number, entry in sorted(
            logs.items(), key=lambda item: (0, int(item[0])) if str(item[0]).isdigit()
            else (1, str(item[0]))):
        if not isinstance(entry, dict):
            continue
        program = entry.get("full_eval_program") or entry.get("mb_program")
        if program is None:
            record = {
                "trial_number": str(raw_number),
                "score": entry.get("full_eval_score", entry.get("mb_score")),
                "program_available": False,
            }
            candidate_gate = entry.get("rsi_candidate_gate")
            if isinstance(candidate_gate, dict):
                record["evaluation_type"] = "pruned_before_task_evaluation"
                record["candidate_gate"] = candidate_gate
            records.append(record)
            continue
        demos = list(getattr(getattr(program, "predict", None), "demos", []) or [])
        task_by_id = getattr(program, "task_by_id", {}) or {}
        demo_ids = []
        labels = {}
        for example in demos:
            task_id = str(getattr(example, "task_id", ""))
            if not task_id or task_id not in train_ids or task_id not in task_by_id:
                raise ValueError("MIPRO trial log contains a demonstration outside frozen train IDs")
            task = task_by_id[task_id]
            answer_index = int(getattr(example, "answer_index"))
            if answer_index != int(task.get("correct_index", -2)):
                raise ValueError("MIPRO trial demo label differs from frozen train gold")
            options = list(task.get("options") or [])
            if not 0 <= answer_index < len(options):
                raise ValueError("MIPRO trial demo answer index is outside its frozen options")
            label = str(options[answer_index])
            demo_ids.append(task_id)
            labels[label] = labels.get(label, 0) + 1
        instructions = str(getattr(
            getattr(getattr(program, "predict", None), "signature", None),
            "instructions", "") or "")
        records.append({
            "trial_number": int(raw_number) if str(raw_number).isdigit() else str(raw_number),
            "score": entry.get("full_eval_score", entry.get("mb_score")),
            "evaluation_type": "full" if "full_eval_score" in entry else "minibatch",
            "program_available": True,
            "instruction_sha256": hashlib.sha256(
                _canonical_mipro_instruction(instructions).encode("utf-8")).hexdigest(),
            "demo_task_ids": demo_ids,
            "demo_count": len(demo_ids),
            "demo_class_coverage": len(labels),
            "demo_label_counts": dict(sorted(labels.items())),
        })
    return {
        "source": "locked_DSPy_MIPROv2_trial_logs",
        "trial_record_count": len(records),
        "records": records,
    }


def summarize_instruction_search_coverage(
        candidate_instruction_gates: Sequence[dict[str, Any]],
        trial_audit: dict[str, Any], baseline_prompt: str,
        minimum_generated_instructions_evaluated: int = 1,
        selected_prompt: str | None = None) -> dict[str, Any]:
    """Check whether grounded novel prompt proposals reached full evaluation."""
    if minimum_generated_instructions_evaluated < 0:
        raise ValueError("minimum generated instruction coverage cannot be negative")
    baseline_hash = hashlib.sha256(
        _canonical_mipro_instruction(baseline_prompt).encode("utf-8")).hexdigest()
    proposed_hashes = sorted({
        str(record.get("proposal_instruction_sha256") or hashlib.sha256(
            _canonical_mipro_instruction(
                str(record.get("candidate_text") or "")).encode("utf-8")).hexdigest())
        for record in candidate_instruction_gates
        if record.get("passed") is True and record.get("content_sha256")
        and str(record.get("candidate_text") or "").strip()
        and str(record.get("proposal_instruction_sha256") or hashlib.sha256(
            _canonical_mipro_instruction(
                str(record.get("candidate_text") or "")).encode("utf-8")).hexdigest())
        != baseline_hash
    })
    evaluated_hashes = sorted({
        str(record.get("instruction_sha256"))
        for record in trial_audit.get("records", [])
        if record.get("evaluation_type") == "full"
        and record.get("instruction_sha256")
    })
    covered = sorted(set(proposed_hashes) & set(evaluated_hashes))
    unseen = sorted(set(proposed_hashes) - set(evaluated_hashes))
    selected_hash = (hashlib.sha256(
        _canonical_mipro_instruction(selected_prompt).encode("utf-8")).hexdigest()
                     if selected_prompt is not None else None)
    return {
        "policy": "grounded_novel_prompt_must_reach_full_validation-v2",
        "baseline_instruction_sha256": baseline_hash,
        "proposed_novel_instruction_count": len(proposed_hashes),
        "proposed_novel_instruction_sha256s": proposed_hashes,
        "full_evaluated_instruction_sha256s": evaluated_hashes,
        "evaluated_novel_instruction_count": len(covered),
        "evaluated_novel_instruction_sha256s": covered,
        "unevaluated_novel_instruction_sha256s": unseen,
        "minimum_generated_instructions_evaluated": minimum_generated_instructions_evaluated,
        "passed": len(covered) >= minimum_generated_instructions_evaluated,
        "coverage_complete": not unseen,
        "selected_instruction_sha256": selected_hash,
        "selected_instruction_was_full_evaluated": (
            selected_hash in evaluated_hashes if selected_hash is not None else None),
    }


def select_class_aware_mipro_candidate(
        evaluated_programs: Sequence[tuple[Any, dict[str, Any]]]) -> tuple[Any | None, dict[str, Any]]:
    """Select a proposed program only if it passes the class-aware gain gate.

    DSPy still proposes and scores programs with its pinned upstream MIPROv2
    implementation. This project-level selector uses the same already-paid,
    full search-validation answers and does not inspect internal confirmation.
    """
    rows = [row for _, row in evaluated_programs]
    baselines = [(program, row) for program, row in evaluated_programs
                 if not row.get("demo_ids") and not row.get("candidate_id")]
    if not baselines:
        return None, {"policy": MIPRO_SELECTION_POLICY_VERSION,
                      "reason": "default_program_full_evaluation_missing",
                      "evaluated_full_programs": len(rows)}
    baseline_program, baseline_row = baselines[0]
    baseline_metrics = baseline_row.get("search_validation_metrics")
    if not isinstance(baseline_metrics, dict):
        return None, {"policy": MIPRO_SELECTION_POLICY_VERSION,
                      "reason": "default_program_metrics_missing",
                      "evaluated_full_programs": len(rows)}
    eligible = []
    rejected = []
    for program, row in evaluated_programs:
        if not row.get("candidate_id") and not row.get("demo_ids"):
            continue
        current = row.get("search_validation_metrics")
        if isinstance(current, dict) and mipro_class_aware_gain(current, baseline_metrics):
            eligible.append((program, row))
        else:
            rejected.append({
                "trial": row.get("trial"),
                "candidate_id": row.get("candidate_id"),
                "reason": ("class_aware_search_validation_gate_rejected"
                           if isinstance(current, dict) else "candidate_metrics_missing"),
                "search_validation_metrics": current,
            })
    if not eligible:
        return baseline_program, {
            "policy": MIPRO_SELECTION_POLICY_VERSION,
            "reason": "no_candidate_passed_search_validation_gate",
            "baseline_search_validation_metrics": baseline_metrics,
            "evaluated_full_programs": len(rows),
            "rejected_candidates": rejected,
        }

    def rank(item):
        _, row = item
        metrics = row["search_validation_metrics"]
        per_label = metrics["per_gold_label"]
        weakest_class_accuracy = min(float(value["accuracy"])
                                     for value in per_label.values())
        return (float(metrics["accuracy"]), weakest_class_accuracy,
                float(metrics["answered_precision"]), float(metrics["coverage"]),
                float(row.get("score_percent", 0)))

    selected_program, selected_row = max(eligible, key=rank)
    return selected_program, {
        "policy": MIPRO_SELECTION_POLICY_VERSION,
        "reason": "best_candidate_passed_search_validation_gate",
        "baseline_search_validation_metrics": baseline_metrics,
        "selected_candidate_id": selected_row.get("candidate_id"),
        "selected_trial": selected_row.get("trial"),
        "selected_search_validation_metrics": selected_row["search_validation_metrics"],
        "evaluated_full_programs": len(rows),
        "eligible_candidates": len(eligible),
        "rejected_candidates": rejected,
    }


def compile_mipro_task_input(*, cfg: dict, state: ArtifactState,
                             train_tasks: Sequence[dict], validation_tasks: Sequence[dict],
                             execution_tasks: Sequence[dict], train_ids: set[str],
                             validation_ids: set[str], candidate_validator: Callable[[str], Any],
                             execution_seed: int, task_evaluation_budget: int,
                             num_candidates: int = 3, num_trials: int = 4,
                             max_labeled_demos: int = 3,
                             minimum_generated_instructions_evaluated: int = 1,
                             program_validator: Callable[[Any], Any] | None = None,
                             balanced_demo_labels: Sequence[str] = (),
                             require_balanced_demo_bundle: bool = False,
                             trial_checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
                             frozen_skill_routes: FrozenSkillRouteCache | None = None,
                             task_contract_context: str | None = None,
                             task_contract: TaskContract | None = None,
                             prompt_only: bool = False):
    """Run bounded upstream MIPROv2 while all scored answers use Pi TaskRunner.

    ``validation_tasks`` are nested search-validation rows, never discovery or
    confirmation rows. The separate ``execution_tasks`` map only supplies the
    frozen inputs to the adapter and must match exactly train + search-val IDs;
    it must never contain internal candidate/confirmation rows.
    """
    dspy, _, _ = require_mipro_v2()
    if num_candidates < 2 or num_trials < 1 or not 1 <= max_labeled_demos <= 3:
        raise ValueError("MIPRO requires >=2 candidates, >=1 trial, and 1..3 labeled demos")
    if minimum_generated_instructions_evaluated < 0:
        raise ValueError("minimum generated instruction coverage cannot be negative")
    if prompt_only and state.task_demos:
        raise ValueError("Prompt-only MIPRO currently requires the frozen empty parent demo pool")
    train_ids = {str(value) for value in train_ids}
    validation_ids = {str(value) for value in validation_ids}
    if not train_ids or not validation_ids or train_ids & validation_ids:
        raise ValueError("MIPRO train and nested-validation ID scopes must be non-empty and disjoint")
    trainset = build_mipro_examples(train_tasks, allowed_ids=train_ids)
    valset = build_mipro_examples(validation_tasks, allowed_ids=validation_ids)
    if {str(row.task_id) for row in trainset} != train_ids:
        raise ValueError("MIPRO train rows do not exactly match frozen train IDs")
    if {str(row.task_id) for row in valset} != validation_ids:
        raise ValueError("MIPRO validation rows do not exactly match nested-validation IDs")
    execution_rows = list(execution_tasks)
    execution_task_ids = [str(row.get("task_id") or "") for row in execution_rows]
    expected_execution_ids = train_ids | validation_ids
    if (not all(execution_task_ids)
            or len(execution_task_ids) != len(set(execution_task_ids))
            or set(execution_task_ids) != expected_execution_ids
            or any(row.get("split") != "dev_optimize" for row in execution_rows)):
        raise ValueError(
            "MIPRO execution task map must exactly match frozen train + search-validation IDs")
    # No-minibatch MIPRO performs one default full evaluation plus one full
    # evaluation per Bayesian trial. Fail before model calls if that lower bound
    # exceeds the declared project TaskRunner budget.
    expected_evals = (num_trials + 1) * len(valset)
    if expected_evals > task_evaluation_budget:
        raise ValueError(
            f"MIPRO plan needs at least {expected_evals} Pi task evaluations, "
            f"above the configured budget {task_evaluation_budget}")
    budget = MiproTaskBudget(task_evaluation_budget)
    completed_full_evals: list[tuple[Any, dict[str, Any]]] = []
    student = PiTaskInputProgram.create(
        cfg, state, execution_rows, train_ids, execution_seed, budget,
        frozen_skill_routes=frozen_skill_routes, task_contract=task_contract)
    predictor_count = len(student.predictors())
    if predictor_count != MIPRO_TASK_INPUT_PREDICTOR_COUNT:
        raise RuntimeError(
            "MIPRO proposer budget model assumes exactly "
            f"{MIPRO_TASK_INPUT_PREDICTOR_COUNT} predictor; program has "
            f"{predictor_count}")
    optimizer_type = make_gated_mipro_v2(
        candidate_validator, train_ids, program_validator=program_validator,
        balanced_demo_labels=balanced_demo_labels,
        require_balanced_demo_bundle=require_balanced_demo_bundle,
        allow_empty_demo_candidate=not bool(state.task_demos),
        task_contract_context=task_contract_context,
        frozen_empty_demo_pool=prompt_only,
        candidate_composer=(lambda text: compose_task_instruction(text, task_contract))
        if task_contract else None)
    prompt_lm = make_openai_compatible_lm(cfg, "reflection")
    optimizer = optimizer_type(
        metric=make_mipro_accuracy_metric(), prompt_model=prompt_lm,
        task_model=prompt_lm, auto=None, num_candidates=num_candidates,
        num_threads=1, max_errors=1, seed=execution_seed,
        max_bootstrapped_demos=0, max_labeled_demos=max_labeled_demos,
        track_stats=True, verbose=False,
    )
    # DSPy 3.4.0b1 catches every evaluator exception and converts it into a
    # legitimate 0.0 score. Temporarily wrap its imported helper so a TaskRunner
    # failure recorded by the student aborts the compile instead of entering
    # Bayesian optimization as a candidate score.
    mipro_module = importlib.import_module("dspy.teleprompt.mipro_optimizer_v2")
    upstream_eval = mipro_module.eval_candidate_program

    def checkpointed_fail_closed_eval(*args, **kwargs):
        result = _fail_closed_candidate_eval(
            upstream_eval, budget, *args, **kwargs)
        batch_size = args[0] if args else kwargs.get("batch_size")
        candidate_program = (args[2] if len(args) > 2
                             else kwargs.get("candidate_program"))
        if int(batch_size or 0) >= len(valset) and candidate_program is not None:
            row = make_mipro_trial_checkpoint(
                candidate_program, float(result.score), train_ids=train_ids,
                release_id=state.release_id, evaluation_count=len(valset),
                evaluation_metrics=summarize_mipro_eval_results(result.results),
            )
            completed_full_evals.append((candidate_program, row))
            if trial_checkpoint_callback is not None:
                trial_checkpoint_callback(row)
        return result

    mipro_module.eval_candidate_program = checkpointed_fail_closed_eval
    try:
        proposer_retries = int((cfg.get("models", {}).get("reflection") or {}).get(
            "max_retries", 2))
        reserve_physical_attempts(
            count=prompt_lm._rsi_call_budget.limit * (proposer_retries + 1),
            reason=("bounded MIPRO HTTP attempts; each attempt is emitted and "
                    "recorded by the project HttpProvider"),
            reservation_key="mipro_proposer_retry_upper_bound",
        )
        with dspy.track_usage() as usage_tracker:
            compiled = optimizer.compile(
                student, trainset=trainset, valset=valset, num_trials=num_trials,
                max_bootstrapped_demos=0, max_labeled_demos=max_labeled_demos,
                seed=execution_seed, minibatch=False,
                view_data_batch_size=MIPRO_VIEW_DATA_BATCH_SIZE,
                **mipro_proposer_guidance(prompt_only=prompt_only),
            )
    except BaseException as exc:
        # A proposal may have succeeded before a later task evaluation or
        # checkpoint fails. Preserve the realized proposal-input digests on
        # every compile failure, not only failures thrown by the proposer.
        observations = getattr(
            optimizer, "_rsi_proposal_context_observations", None)
        if observations is None:
            observations = getattr(
                exc, "rsi_proposal_context_observations", [])
        try:
            setattr(exc, "rsi_proposal_context_observations",
                    list(observations or []))
        except (AttributeError, TypeError):
            pass
        raise
    finally:
        mipro_module.eval_candidate_program = upstream_eval
    if budget.failure_detail is not None:
        raise MiproTaskEvaluationError(budget.failure_detail)
    if budget.exceeded:
        raise RuntimeError("MIPRO exceeded the Pi task-evaluation budget; candidate discarded")
    if prompt_lm._rsi_call_budget.exceeded:
        raise RuntimeError("MIPRO exceeded the proposer-model call budget; candidate discarded")
    optimizer_score = getattr(compiled, "score", None)
    trial_audit = summarize_mipro_trial_log(compiled, train_ids)
    selected_program, selection = select_class_aware_mipro_candidate(
        completed_full_evals)
    if selected_program is not None:
        compiled = selected_program
    selected_prompt = str(getattr(
        getattr(getattr(compiled, "predict", None), "signature", None),
        "instructions", "") or "")
    instruction_search_coverage = summarize_instruction_search_coverage(
        getattr(candidate_validator, "records", []), trial_audit,
        str(state.system_prompt),
        minimum_generated_instructions_evaluated=minimum_generated_instructions_evaluated,
        selected_prompt=selected_prompt)
    from .task_program import (changed_task_program_components,
                              serialize_task_program)
    artifact = (extract_task_program(compiled, train_ids)
                if selected_program is not None else None)
    changed_components = (changed_task_program_components(serialize_task_program(
        artifact["system_prompt"], artifact["task_demos"]), state)
        if artifact is not None else [])
    if selected_program is None:
        candidate_status = "no_class_aware_candidate"
        candidate_status_reason = selection["reason"]
    elif len(changed_components) == 1:
        candidate_status = "candidate"
        candidate_status_reason = None
    else:
        artifact = None
        candidate_status = "no_single_component_candidate"
        candidate_status_reason = (
            "incumbent_or_selected_program_changes_"
            + ("no" if not changed_components else "multiple")
            + "_parent_components")
    usage = summarize_mipro_usage(
        usage_tracker, prompt_lm._rsi_call_budget.calls,
        model_id=str(getattr(prompt_lm, "_rsi_model_id", "")),
        base_url=str(getattr(prompt_lm, "_rsi_base_url", "")))
    return {
        "artifact": artifact,
        "candidate_status": candidate_status,
        "candidate_status_reason": candidate_status_reason,
        "score": optimizer_score,
        "selection": selection,
        "task_evaluations": budget.calls,
        "task_evaluation_budget": budget.limit,
        "prompt_model_calls": prompt_lm._rsi_call_budget.calls,
        "prompt_model_call_budget": prompt_lm._rsi_call_budget.limit,
        "prompt_model_budget_exceeded": prompt_lm._rsi_call_budget.exceeded,
        "prompt_model_usage": usage,
        "demo_candidate_pool": getattr(optimizer, "_rsi_demo_candidate_pool", {}),
        "search_space": "prompt_only_frozen_empty_demo_pool_v1" if prompt_only else "instruction_x_labeled_demo_v1",
        "demo_bundle_constraint": {
            "require_balanced_demo_bundle": require_balanced_demo_bundle,
            "required_labels": list(balanced_demo_labels),
        },
        "trial_audit": trial_audit,
        "instruction_search_coverage": instruction_search_coverage,
        "instruction_proposal_audit": getattr(
            optimizer, "_rsi_instruction_proposal_audit", []),
        "proposal_task_contract": _proposal_context_report(
            task_contract_context, optimizer),
        "optimizer": {"name": "DSPy MIPROv2", "version": "3.4.0b1",
                      "num_candidates": num_candidates, "num_trials": num_trials,
                      "max_labeled_demos": max_labeled_demos,
                      "max_bootstrapped_demos": 0, "minibatch": False,
                      "proposal_guidance": mipro_proposer_guidance(prompt_only=prompt_only)},
    }


def extract_task_program(program: Any, train_ids: set[str]) -> dict:
    """Convert a MIPRO result to a Prompt-plus-demos candidate envelope."""
    prompt = _mipro_runtime_prompt(program)
    if not prompt.strip():
        raise ValueError("MIPRO returned an empty predictor instruction")
    demos = []
    seen = set()
    for example in program.predict.demos:
        task_id = str(getattr(example, "task_id", ""))
        if not task_id or task_id in seen or task_id not in train_ids:
            raise ValueError("MIPRO returned duplicate or non-train demonstrations")
        seen.add(task_id)
        demos.append({"task_id": task_id,
                      "task": program.task_by_id[task_id],
                      "correct_index": int(getattr(example, "answer_index"))})
    return {"system_prompt": prompt,
            "task_demos": (parse_demos_bundle(serialize_demos_bundle(demos))
                           if demos else [])}


def make_mipro_trial_checkpoint(program: Any, score: float, *,
                                train_ids: set[str], release_id: str,
                                evaluation_count: int,
                                evaluation_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    """Snapshot one fully scored MIPRO program before its next trial starts."""
    from ..contracts.types import content_id, content_sha256
    from .task_program import serialize_task_program

    prompt = _mipro_runtime_prompt(program)
    demos = list(program.predict.demos or [])
    demo_ids = [str(getattr(example, "task_id", "")) for example in demos]
    # The default zero-shot program is a useful score checkpoint, but it is
    # not an optimizer proposal. Prompt-only proposals have trial metadata and
    # must still be retained even though they have no labeled demonstrations.
    has_trial_metadata = bool(getattr(program, "_rsi_mipro_trial_metadata", None))
    parent = getattr(program, "base_state", None)
    if parent is not None:
        from .task_program import changed_task_program_components
        extracted = extract_task_program(program, train_ids)
        artifact = (extracted if changed_task_program_components(
            serialize_task_program(extracted["system_prompt"], extracted["task_demos"]),
            parent) else None)
    else:
        artifact = (extract_task_program(program, train_ids)
                    if demos or has_trial_metadata else None)
    serialized = (serialize_task_program(artifact["system_prompt"], artifact["task_demos"])
                  if artifact is not None else None)
    candidate_id = (content_id("cand", release_id, "task_program", serialized)
                    if serialized is not None else None)
    candidate_content_sha256 = (content_sha256(serialized)
                                if serialized is not None else None)
    labels = {}
    for example in demos:
        options = list(getattr(example, "options", []) or [])
        answer_index = getattr(example, "answer_index", None)
        if type(answer_index) is int and 0 <= answer_index < len(options):
            label = str(options[answer_index])
            labels[label] = labels.get(label, 0) + 1
    return {
        "schema_version": 1,
        "evaluation_type": "full_search_validation",
        "evaluation_count": int(evaluation_count),
        "score": float(score),
        "score_percent": float(score),
        "search_validation_metrics": evaluation_metrics,
        "trial": dict(getattr(program, "_rsi_mipro_trial_metadata", {}) or {}),
        "candidate_id": candidate_id,
        "candidate_content_sha256": candidate_content_sha256,
        "instruction_sha256": content_sha256(_canonical_mipro_instruction(prompt)),
        "instruction_hash_scope": "canonical DSPy instruction; exact runtime bytes hashed separately",
        "runtime_system_prompt_sha256": content_sha256(prompt),
        "demo_ids": demo_ids,
        "demo_label_counts": dict(sorted(labels.items())),
        "candidate": artifact,
    }
