#!/usr/bin/env python3
"""Run bounded task-input MIPROv2 search on fixed supplied evidence.

Evidence Inference mode reads only its frozen article-disjoint train cohort.
HealthVer mode reads the combined public upstream splits re-assigned by claim /
abstract component; its upstream test is materialized and is never described as
blind confirmation. SciFact mode reads only labeled train/dev claims after
prior-use component exclusions. All modes use fixed evidence and disable retrieval.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

# Capture process-provided credentials for audit metadata. The user has
# explicitly authorized necessary live research calls to use configured project
# dotenv credentials; this is not an authorization gate. Runtime presence,
# frozen-plan identity, data scope and request budgets remain mandatory.
_PROCESS_ENV_AT_START = frozenset(os.environ)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_pubmedqa_prompt_discovery import run_prompt_grounding_review  # noqa: E402
from src.config import config_fingerprint, load_config  # noqa: E402
from src.contracts.types import Candidate, content_id, content_sha256  # noqa: E402
from src.data.evidence_inference_cohort import LABELS as EI_LABELS  # noqa: E402
from src.data.evidence_inference_v2 import (  # noqa: E402
    DATASET as EI_DATASET, load_tasks as load_ei_tasks,
)
from src.data.healthver_task_input import (  # noqa: E402
    DATASET as HEALTHVER_DATASET, LABELS as HEALTHVER_LABELS,
    load_tasks as load_healthver_tasks,
)
from src.data.healthfc_task_input import (  # noqa: E402
    DATASET as HEALTHFC_DATASET, LABELS_ORDERED as HEALTHFC_LABELS,
    SOURCE_SHA256 as HEALTHFC_SOURCE_SHA256,
    AUDIT_PATH as HEALTHFC_AUDIT_PATH,
    load_tasks as load_healthfc_tasks,
)
from src.data.healthfc_history_scan import (  # noqa: E402
    DEFAULT_REPORT_PATH as HEALTHFC_HISTORY_REPORT_PATH,
)
from src.data.clinifact_task_input import (  # noqa: E402
    DATASET as CLINIFACT_DATASET, LABELS as CLINIFACT_LABELS,
    load_tasks as load_clinifact_tasks,
)
from src.data.clinifact_history_scan import (  # noqa: E402
    scan_clinifact_history,
)
from src.data.scifact_task_input import (  # noqa: E402
    DATASET as SCIFACT_DATASET, LABELS as SCIFACT_LABELS,
    SOURCE_SHA256 as SCIFACT_SOURCE_SHA256,
    load_tasks as load_scifact_tasks,
)
from src.eval.stats import mcnemar_exact, paired_bootstrap  # noqa: E402
from src.eval.task_rows import validate_task_answer_row  # noqa: E402
from src.evolution.mipro_task_input import (  # noqa: E402
    MIPRO_SELECTION_POLICY_VERSION, MIPRO_TASK_INPUT_PREDICTOR_COUNT,
    MIPRO_VIEW_DATA_BATCH_SIZE, compile_mipro_task_input,
    extract_task_program, proposal_context_failure_report, require_mipro_v2,
    mipro_proposer_guidance, validate_mipro_optimizer_runtime,
)
from src.evolution.task_input_gates import (  # noqa: E402
    make_task_input_prompt_validator, validate_task_demo_rendering,
    validate_task_input_prompt,
)
from src.evolution.task_contracts import (  # noqa: E402
    CLINIFACT_MIPRO_STRATEGY_SEED, compose_task_instruction,
    task_contract_for_examples)
from src.evolution.task_program import serialize_task_program  # noqa: E402
from src.evolution.evidence_inference_promotion import (  # noqa: E402
    PROTOCOL as EI_MIPRO_PROTOCOL,
    compute_mipro_plan_sha256,
)
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.provider import build_provider  # noqa: E402
from src.runtime.runner import ArtifactState, TaskRunner  # noqa: E402
from src.runtime.skill_routes import (  # noqa: E402
    FrozenSkillRouteCache,
)
from src.runtime.call_ledger import (  # noqa: E402
    call_component, seed_legacy_usage, summarize_ledger,
)
from src.util import (  # noqa: E402
    artifacts_path, path, read_json, read_jsonl, sha256_file, sha256_obj,
    write_json,
)

RELEASE_ID = "task-input-evidence-inference-v1"
DATASET = EI_DATASET
HTTP_MODEL_MAX_ATTEMPTS = 5
MIPRO_PROPOSER_MAX_RETRIES = 2
MIN_VOTES_FOR_SELECTION_GATE = 2
LABELS = EI_LABELS
load_tasks = load_ei_tasks
COHORT_PATH = ROOT / "artifacts" / "probe" / "evidence_inference_nested_cohort.json"
MANIFEST_PATH = path("data", "manifests", f"{DATASET}.json")
STUDY_NAMESPACE = "evidence_inference_mipro"


def configure_dataset_adapter(name: str) -> None:
    """Bind one explicit source protocol before constructing the MIPRO plan."""
    global DATASET, LABELS, load_tasks, COHORT_PATH, MANIFEST_PATH, STUDY_NAMESPACE
    global RELEASE_ID
    from src.data.research_profiles import profile_for
    profile = profile_for(name)
    DATASET, LABELS, load_tasks = profile.dataset, profile.labels, profile.loader
    COHORT_PATH, MANIFEST_PATH = profile.cohort_path, profile.manifest_path
    STUDY_NAMESPACE, RELEASE_ID = profile.study_namespace, profile.release_id


def prepare_config(config_path: str, *, proposer_calls: int,
                   routing_mode: str = "model",
                   task_model_id: str = "Qwen/Qwen3-32B",
                   instruction_review_model_id: str | None = None,
                   pi_timeout_max_attempts: int = 1,
                   task_request_cooldown_seconds: float = 30.0) -> dict[str, Any]:
    cfg = copy.deepcopy(load_config(config_path))
    if not 1 <= int(pi_timeout_max_attempts) <= 3:
        raise ValueError("Pi timeout attempts must be between 1 and 3")
    if not 0 <= task_request_cooldown_seconds <= 60:
        raise ValueError("task request cooldown must be between 0 and 60 seconds")
    if task_model_id not in {"Qwen/Qwen3-32B", "Qwen/Qwen3-8B"}:
        raise ValueError("unsupported task-input MIPRO model")
    if instruction_review_model_id not in {None, "Qwen/Qwen3-8B"}:
        raise ValueError(
            "only an explicit Qwen/Qwen3-8B exploratory instruction-review override is supported")
    cfg.setdefault("models", {}).setdefault("task", {})["model_id"] = task_model_id
    cfg.setdefault("models", {}).setdefault("task", {})["provider"] = "http"
    if DATASET == CLINIFACT_DATASET:
        # CliniFact answer shape is enforced by the provider API schema, not by
        # asking the evolving prompt or model to emit a parseable text marker.
        cfg["models"]["task"]["structured_output"] = "json_schema"
    # HttpProvider interprets max_retries as total attempts (not retries after
    # the first call). Use the same bounded five-attempt ceiling already
    # reserved for Pi-backed task answers so transient router TLS failures do
    # not abort an otherwise healthy MIPRO trial after only two attempts.
    cfg["models"]["task"]["max_retries"] = HTTP_MODEL_MAX_ATTEMPTS
    cfg.setdefault("models", {}).setdefault("reflection", {})["provider"] = "http"
    # MIPRO proposer retries are expressed as retries after the first request;
    # the project-owned adapter converts them to HttpProvider total attempts.
    cfg["models"]["reflection"]["max_retries"] = MIPRO_PROPOSER_MAX_RETRIES
    cfg.setdefault("skills", {}).update({
        "routing_mode": routing_mode,
        "router_provider": "http",
        "router_model_id": "Qwen/Qwen3-8B",
    })
    if DATASET == CLINIFACT_DATASET:
        # The candidate instruction reviewer has an independent calibration
        # protocol; its default stays separate from the skill router. An explicit
        # 8B override is only for exploratory smoke runs and cannot establish
        # reviewer validity or candidate eligibility.
        cfg["skills"]["instruction_review_model_id"] = (
            instruction_review_model_id or "Qwen/Qwen3-32B")
    cfg.setdefault("runtime", {}).update({
        "provider": "pi",
        "input_mode": "task_input",
        "task_input_max_chars": 30000,
        "pi_task_demos_max_block_chars": 3500,
        "pi_task_demo_max_example_chars": 400,
        # The previous balanced-demo search hit three consecutive SiliconFlow
        # 50609/429 responses on a 32B validation call despite the bounded
        # exponential retry. Slow successful requests further for the next
        # exploratory run so the same failure does not recur as a burst.
        "pi_request_cooldown_seconds": float(task_request_cooldown_seconds),
        # Bound physical provider requests and expose every retry in Pi traces.
        # The provider adapter honors Retry-After and caps SDK retries at two.
        # Two prior 32B MIPRO runs exhausted three provider attempts on a
        # transient 50609/429 task. Give system-busy recovery up to four
        # exponential-backoff retries before failing closed.
        "pi_provider_max_retries": 4,
        # SiliconFlow can return transient 50609/429 without Retry-After.
        # Back off between bounded retries instead of sending them as a burst.
        "pi_provider_429_backoff_base_seconds": 60.0,
        "http_provider_transport_backoff_base_seconds": 15.0,
        "pi_timeout_max_attempts": int(pi_timeout_max_attempts),
        "http_no_cache_required": True,
    })
    cfg["runtime"].pop("pi_fake_response", None)
    cfg.setdefault("evaluation", {}).update({
        "datasets": [DATASET],
        "insufficient_option": False,
        "main_metric": "accuracy",
        "primary_axis": "accuracy",
        "sealed_pairing": False,
        # The EI adapter enforces its own manifest/task-ID hashes; it is not
        # registered in the legacy multi-dataset freeze manifest.
        "require_split_freeze": False,
    })
    cfg.setdefault("gates", {}).setdefault("accept", {}).update({
        "require_classwise_non_regression": True,
        "require_strict_accuracy_gain": True,
    })
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg.get("tools", {}).get("active", [])
        if name != "academic.search"]
    evolution = cfg.setdefault("evolution", {})
    evolution["allowed_artifacts"] = ["task_program"]
    evolution.setdefault("fewshot", {}).update({"enabled": True, "k": 3})
    evolution.setdefault("mipro", {}).update({"max_proposer_calls": proposer_calls})
    # Three complete abstract-level examples can exceed the ordinary prompt
    # artifact limit. The atomic task_program still caps the combined artifact
    # at max_prompt_chars + 6000, so align its demo component with that ceiling.
    cfg.setdefault("gates", {}).setdefault("static", {})[
        "max_task_demos_chars"] = int(
            cfg["gates"]["static"].get("max_prompt_chars", 8000)) + 6000
    return cfg


def apply_physical_request_budget(
    cfg: dict[str, Any], budget: int, *, proposer_retries_after_first: int = 0,
) -> None:
    """Apply explicit per-provider retry bounds before computing the plan.

    The plan builder then checks the resulting worst-case physical-attempt
    count against ``budget``. HTTP task calls remain single-attempt in this
    bounded mode; DSPy's proposer can use a small, explicitly budgeted retry
    count for transient LMTransportError/LMServerError failures.
    """
    if int(budget) <= 0:
        raise ValueError("physical HTTP request budget must be positive")
    retries = int(proposer_retries_after_first)
    if not 0 <= retries <= MIPRO_PROPOSER_MAX_RETRIES:
        raise ValueError(
            "MIPRO proposer retries-after-first must be between 0 and "
            f"{MIPRO_PROPOSER_MAX_RETRIES}")
    cfg.setdefault("runtime", {}).update({
        "pi_provider_max_retries": 0,
        "pi_timeout_max_attempts": 1,
    })
    cfg.setdefault("models", {}).setdefault("task", {})["max_retries"] = 1
    cfg.setdefault("models", {}).setdefault("reflection", {})[
        "max_retries"] = retries


def baseline_task_config_fingerprint(cfg: dict[str, Any]) -> str:
    """Fingerprint task-answer semantics, excluding optimizer-only retry policy."""
    task_cfg = copy.deepcopy(cfg)
    task_cfg.setdefault("models", {}).setdefault("reflection", {})[
        "max_retries"] = 0
    return config_fingerprint(task_cfg)


def validate_physical_budget_extension(plan: dict[str, Any], physical_ceiling: int) -> None:
    """Reject resume extensions that escape a frozen physical request ceiling."""
    budget = plan.get("physical_request_budget")
    if budget is not None and int(physical_ceiling) > int(budget):
        raise ValueError(
            "physical budget extension cannot exceed the frozen "
            "--physical-request-budget")


def cohort_is_exploratory(cohort: dict[str, Any]) -> bool:
    """Return whether this cohort can support only exploratory evidence.

    HealthVer's project-facing split is exhausted, and EI cohorts with fewer
    than two annotation votes per included task are too weak for the project's
    official candidate-selection gate.
    """
    minimum_votes = int(cohort.get("minimum_annotation_votes", 1))
    dataset = str(cohort.get("dataset") or DATASET)
    return dataset != EI_DATASET or minimum_votes < MIN_VOTES_FOR_SELECTION_GATE


def build_resume_runtime_override(plan: dict[str, Any], backoff: float) -> dict[str, Any]:
    """Record resume settings using the frozen plan's actual policy fields."""
    retry_policy = plan["provider_retry_policy"]
    return {
        "plan_sha256": plan["plan_sha256"],
        "pi_provider_retries_after_first_attempt": retry_policy[
            "pi_provider_retries_after_first_attempt"],
        "pi_provider_429_backoff_base_seconds": backoff,
        "http_task_router_configured_max_attempts": retry_policy[
            "http_task_router_configured_max_attempts"],
        "mipro_proposer_http_attempts_per_call": retry_policy[
            "mipro_proposer_http_attempts_per_call"],
        "mipro_proposer_retries_after_first": retry_policy[
            "mipro_proposer_retries_after_first"],
        "http_provider_429_backoff_base_seconds": backoff,
        "reason": "bounded resume backoff for retryable provider failures",
    }


def compute_plan_sha256(plan: dict[str, Any]) -> str:
    """Hash experiment semantics while omitting launcher and wrapper metadata.

    Whether a credential was explicitly present can only be known at live
    launch. It is an execution attestation, not an experiment parameter, so it
    must not create a different study identity from the zero-call preflight.
    The top-level `status` field on a saved preflight is a wrapper marker added
    after plan construction, not part of the frozen experiment semantics.
    """
    return compute_mipro_plan_sha256(plan)


def validate_resume_plan(current_plan: dict[str, Any],
                         frozen_plan: dict[str, Any], *,
                         report_status: str,
                         resume_study_id: str,
                         phase_checkpoint_verified: bool = False) -> dict[str, Any]:
    """Accept resume only when the saved plan and current protocol are identical.

    The credential-presence attestation is execution metadata and is excluded
    from the plan hash. All experiment semantics, including the protocol source
    fingerprint and baseline identity, remain binding during resume.
    """
    frozen_sha = frozen_plan.get("plan_sha256")
    if (not isinstance(frozen_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", frozen_sha)
            or compute_plan_sha256(frozen_plan) != frozen_sha):
        raise ValueError("frozen plan hash does not match its saved contents")
    if frozen_sha[:16] != resume_study_id:
        raise ValueError("resume study id does not match its frozen plan")
    if (report_status != "internal_confirmation_failed"
            and not (phase_checkpoint_verified and report_status in {
                "baseline_in_progress", "baseline_failed", "search_completed"})):
        raise ValueError("resume is only allowed for an interrupted internal confirmation")
    current_protocol = current_plan.get("config_sha256")
    frozen_protocol = frozen_plan.get("config_sha256")
    if (not isinstance(current_protocol, str)
            or not re.fullmatch(r"[0-9a-f]{64}", current_protocol)
            or not isinstance(frozen_protocol, str)
            or not re.fullmatch(r"[0-9a-f]{64}", frozen_protocol)
            or current_protocol != frozen_protocol):
        raise ValueError(
            "resume protocol fingerprint differs from the frozen plan; create a new plan")
    ignored_fields = {"plan_sha256", "api_key_explicit_at_launch"}
    current_semantics = {k: v for k, v in current_plan.items()
                         if k not in ignored_fields}
    frozen_semantics = {k: v for k, v in frozen_plan.items()
                        if k not in ignored_fields}
    if current_semantics != frozen_semantics:
        differing = sorted(k for k in set(current_semantics) | set(frozen_semantics)
                           if current_semantics.get(k) != frozen_semantics.get(k))
        raise ValueError("resume plan differs in experiment fields: " + ", ".join(differing))
    return dict(frozen_plan)


def validate_phase_checkpoint(checkpoint: dict[str, Any], plan: dict[str, Any],
                              ledger_path: Path) -> dict[str, Any]:
    """Resume only a durable phase with fully reconciled dispatched requests.

    DSPy has no trial-state resume. In-flight search is deliberately excluded;
    completed compiled candidates and baseline rows can be replayed as receipts.
    """
    identity = {k: v for k, v in checkpoint.items() if k != "checkpoint_sha256"}
    if (checkpoint.get("schema") != "mipro-phase-checkpoint-v1"
            or sha256_obj(identity) != checkpoint.get("checkpoint_sha256")
            or checkpoint.get("plan_sha256") != plan.get("plan_sha256")
            or checkpoint.get("status") not in {
                "baseline_in_progress", "baseline_failed", "search_completed"}
            or (checkpoint.get("incumbent_gate") or {}).get("passed") is not True):
        raise ValueError("missing, stale, corrupt or non-resumable MIPRO phase checkpoint")
    accounting = summarize_ledger(ledger_path)
    if (accounting.get("logical_incomplete") or accounting.get("physical_incomplete")
            or set(accounting.get("execution_plan_sha256s") or []) - {plan["plan_sha256"]}):
        raise ValueError("MIPRO resume blocked: dispatched request outcome unknown or ledger identity changed")
    if checkpoint["status"] in {"baseline_in_progress", "baseline_failed"}:
        if (accounting.get("logical_by_component") or {}).get("mipro_proposer", 0):
            raise ValueError("MIPRO search already started; baseline checkpoint cannot replay it")
        rows = read_jsonl(Path(checkpoint["internal_baseline_checkpoint"]))
        prefilled = checkpoint.get("prefilled_baseline_rows", 0)
        if (not isinstance(prefilled, int) or prefilled < 0 or
                len(rows) != prefilled + (accounting.get("logical_by_component") or {}).get("task_answer", 0)):
            raise ValueError("MIPRO baseline receipts do not reconcile with dispatched task calls")
        if sha256_obj(rows[:prefilled]) != checkpoint.get(
                "prefilled_baseline_sha256", sha256_obj([])):
            raise ValueError("pre-existing baseline receipts changed")
    elif checkpoint["status"] == "search_completed":
        compiled = Path(checkpoint.get("compiled_candidate_path", ""))
        if (not compiled.is_file() or sha256_file(compiled) != checkpoint.get("compiled_candidate_sha256")
                or read_json(compiled).get("plan_sha256") != plan["plan_sha256"]):
            raise ValueError("completed MIPRO search is missing its immutable compiled receipt")
    return dict(checkpoint)


def persist_phase_checkpoint(path_: Path, plan: dict[str, Any], status: str,
                              baseline_checkpoint: Path, incumbent_gate: dict,
                              candidate_instruction_gates: list[dict]) -> None:
    previous = read_json(path_) if path_.exists() else None
    if previous:
        if (previous.get("plan_sha256") != plan["plan_sha256"] or
                sha256_obj({k: v for k, v in previous.items() if k != "checkpoint_sha256"})
                != previous.get("checkpoint_sha256")):
            raise ValueError("phase checkpoint identity changed")
        prefilled = previous.get("prefilled_baseline_rows", 0)
        prefilled_sha = previous.get("prefilled_baseline_sha256", sha256_obj([]))
    else:
        prefilled_rows = read_jsonl(baseline_checkpoint) if baseline_checkpoint.exists() else []
        prefilled = len(prefilled_rows)
        prefilled_sha = sha256_obj(prefilled_rows)
    body = {"schema": "mipro-phase-checkpoint-v1", "plan_sha256": plan["plan_sha256"],
            "status": status, "internal_baseline_checkpoint": str(baseline_checkpoint),
            "prefilled_baseline_rows": prefilled, "prefilled_baseline_sha256": prefilled_sha,
            "incumbent_gate": incumbent_gate,
            "candidate_instruction_gates": candidate_instruction_gates}
    if status == "search_completed":
        compiled = path_.parent / "compiled-candidate.json"
        body.update({"compiled_candidate_path": str(compiled),
                     "compiled_candidate_sha256": sha256_file(compiled)})
    body["checkpoint_sha256"] = sha256_obj(body)
    write_json(path_, body)


def build_proposal_task_contract(task_contract):
    """Tell MIPRO that the runtime owns the immutable contract component."""
    if task_contract is None:
        return None, {
            "task_contract_id": None,
            "task_contract_sha256": None,
            "context_sha256": None,
            "injected_into_data_aware_proposer": False,
            "context_format": "frozen-task-contract-plus-train-summary-v1",
            "train_summary_retained": True,
            "student_task_prompt_modified": False,
            "data_summary_observation_schema": None,
        }
    contract_header = (
        "Frozen task decision contract "
        f"(id={task_contract.contract_id}; "
        f"sha256={task_contract.sha256}):\n{task_contract.text}\n\n")
    if task_contract.contract_id == "clinifact_trial_result_claim_evidence_v4":
        context = (contract_header
                   + "Instruction proposal rule: generate only a reasoning strategy. "
                   "The runtime composes the immutable contract above after every "
                   "strategy, so do not restate or redefine labels or output formatting. "
                   "Strategy text is reviewed together with the composed contract "
                   "before evaluation.")
        context_format = "program-owned-contract-plus-mutable-strategy-v3"
    else:
        context = (contract_header
                   + "Instruction proposal rule: generate a complete replacement "
                   "instruction that preserves the frozen task decision contract. "
                   "A proposal that omits or contradicts its requirements is rejected.")
        context_format = "frozen-task-contract-plus-train-summary-v2"
    return context, {
        "task_contract_id": task_contract.contract_id,
        "task_contract_sha256": task_contract.sha256,
        "context_sha256": hashlib.sha256(
            context.strip().encode("utf-8")).hexdigest(),
        "injected_into_data_aware_proposer": True,
        "context_format": context_format,
        "train_summary_retained": True,
        "student_task_prompt_modified": task_contract.contract_id == (
            "clinifact_trial_result_claim_evidence_v4"),
        "contract_composed_by_runtime_code": task_contract.contract_id == (
            "clinifact_trial_result_claim_evidence_v4"),
        "proposal_text_must_restate_contract": task_contract.contract_id != (
            "clinifact_trial_result_claim_evidence_v4"),
        "data_summary_observation_schema": (
            "mipro-generated-summary-digest-v1"),
    }


def _configured_api_envs(cfg: dict[str, Any]) -> list[str]:
    names: set[str] = set()
    for section in ("task", "reflection"):
        model = (cfg.get("models") or {}).get(section) or {}
        if str(model.get("provider") or "http").lower() == "http":
            key_env = str(model.get("api_key_env") or "").strip()
            if key_env:
                names.add(key_env)
    skills = cfg.get("skills") or {}
    if (str(skills.get("routing_mode") or "lexical").lower() == "model"
            and str(skills.get("router_provider") or "http").lower() == "http"):
        router_key_env = str(skills.get("router_api_key_env") or "SILICONFLOW_API_KEY").strip()
        if router_key_env:
            names.add(router_key_env)
    return sorted(names)


def validate_mipro_proposer_config(cfg: dict[str, Any]) -> None:
    """Fail before baseline calls if the live MIPRO proposer is not executable."""
    model = (cfg.get("models") or {}).get("reflection") or {}
    missing = [name for name in ("model_id", "base_url", "api_key_env")
               if not str(model.get(name) or "").strip()]
    if str(model.get("model_id") or "").strip().lower().startswith("fake-"):
        missing.append("live_model_id")
    if str(model.get("provider") or "").lower() != "http":
        missing.insert(0, "provider=http")
    if missing:
        raise ValueError(
            "MIPRO requires a configured HTTP proposer before any model calls; "
            "missing reflection fields: " + ", ".join(missing))
    retries = int(model.get("max_retries", MIPRO_PROPOSER_MAX_RETRIES))
    if not 0 <= retries <= MIPRO_PROPOSER_MAX_RETRIES:
        raise ValueError(
            "MIPRO proposer retries-after-first must be between 0 and "
            f"{MIPRO_PROPOSER_MAX_RETRIES}")


def _dotenv_only_api_envs(cfg: dict[str, Any]) -> list[str]:
    """Return configured keys loaded after launch (typically from project .env)."""
    return [name for name in _configured_api_envs(cfg)
            if name not in _PROCESS_ENV_AT_START and os.environ.get(name)]


def require_launcher_credentials(cfg: dict[str, Any]) -> None:
    missing = [name for name in _configured_api_envs(cfg)
               if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "missing configured model credentials for " + ", ".join(missing)
        )


def physical_http_attempt_upper_bounds(
    cfg: dict[str, Any], *, task_evaluations: int, router_calls: int,
    proposer_calls: int, grounding_calls: int,
) -> dict[str, int]:
    """Count maximum wire requests using each provider's implemented semantics.

    Pi's ``providerMaxRetries`` means retries after the first attempt; the
    bridge timeout setting means total subprocess attempts. ``HttpProvider``'s
    legacy ``max_retries`` config is implemented as total attempts (``range``
    over that value), so it must not receive an extra ``+1`` here.
    """
    runtime = cfg.get("runtime") or {}
    models = cfg.get("models") or {}
    task_model = models.get("task") or {}
    reflection = models.get("reflection") or {}
    pi_retries = max(0, min(4, int(runtime.get("pi_provider_max_retries", 4))))
    pi_timeout_attempts = int(runtime.get("pi_timeout_max_attempts", 1))
    if not 1 <= pi_timeout_attempts <= 3:
        raise ValueError("Pi bridge timeout attempts must be between 1 and 3")
    http_task_attempts = int(task_model.get("max_retries", 6))
    proposer_retries_after_first = int(
        reflection.get("max_retries", MIPRO_PROPOSER_MAX_RETRIES))
    if http_task_attempts < 1:
        raise ValueError("HTTP provider max_retries must configure at least one attempt")
    if not 0 <= proposer_retries_after_first <= MIPRO_PROPOSER_MAX_RETRIES:
        raise ValueError(
            "MIPRO proposer retries-after-first must be between 0 and "
            f"{MIPRO_PROPOSER_MAX_RETRIES}")
    # The custom project engine uses HttpProvider's total-attempt max_retries,
    # while the CLI and frozen protocol express retries after the first call.
    http_proposer_attempts = proposer_retries_after_first + 1
    pi_attempts_per_logical = (pi_retries + 1) * pi_timeout_attempts
    return {
        "task_answer": int(task_evaluations) * pi_attempts_per_logical,
        "skill_router": int(router_calls) * http_task_attempts,
        "mipro_proposer": int(proposer_calls) * http_proposer_attempts,
        "grounding": int(grounding_calls) * pi_attempts_per_logical,
    }


def mipro_proposer_lm_call_upper_bounds(
    *, train_example_count: int, instruction_candidate_count: int,
    predictor_count: int = 1, view_data_batch_size: int = 10,
    program_aware: bool = True, data_aware: bool = True,
) -> dict[str, int]:
    """Bound pinned DSPy 3.4.0b1 LM calls needed to build its instruction set.

    The pinned GroundedProposer optionally makes a dataset-summary call, up to
    nine follow-up batch summaries, and a final summary call. It then generates
    one instruction for each requested candidate and predictor; with
    program-aware mode, each generated instruction also makes DescribeProgram
    and DescribeModule calls. This is derived from the vendored implementation,
    not from provider retry counts (which are accounted for separately).
    """
    if train_example_count < 1:
        raise ValueError("MIPRO proposer requires at least one train example")
    if instruction_candidate_count < 1 or predictor_count < 1:
        raise ValueError("MIPRO proposer requires candidates and predictors")
    if view_data_batch_size < 1:
        raise ValueError("MIPRO dataset-summary batch size must be positive")

    # create_dataset_summary always requests the first descriptor and the final
    # summary. Its loop emits at most max_calls - 1 follow-ups because it checks
    # `calls >= max_calls` before dispatching the current follow-up.
    summary_followups = max(0, (train_example_count - 1) // view_data_batch_size)
    dataset_summary_calls = (
        2 + min(9, summary_followups) if data_aware else 0)
    calls_per_instruction = 1 + (2 if program_aware else 0)
    instruction_generation_calls = (
        instruction_candidate_count * predictor_count * calls_per_instruction)
    return {
        "dataset_summary": dataset_summary_calls,
        "instruction_generation": instruction_generation_calls,
        "total": dataset_summary_calls + instruction_generation_calls,
    }


def require_sufficient_mipro_proposer_budget(
    configured_budget: int, call_upper_bounds: dict[str, int],
) -> None:
    required = int(call_upper_bounds["total"])
    if configured_budget < required:
        raise ValueError(
            "MIPRO proposer-call budget cannot complete the pinned DSPy "
            "instruction proposal stage: "
            f"needs at least {required} LM calls "
            f"(summary={call_upper_bounds['dataset_summary']}, "
            "program-aware instruction generation="
            f"{call_upper_bounds['instruction_generation']}), "
            f"configured {configured_budget}")


def mipro_instruction_review_call_upper_bounds(*, instruction_candidate_count: int,
                                              predictor_count: int = 1) -> dict[str, int]:
    """Reserve the incumbent plus every proposed instruction before task calls.

    Pinned DSPy replaces slot zero with the incumbent; the remaining slots
    can all require review. Program
    assembly and final extraction reuse the exact recorded instruction review;
    they do not dispatch new reviews. Static rejections/duplicates may reduce
    realized calls, but cannot justify underfunding an unobserved proposal set.
    """
    if (type(instruction_candidate_count) is not int or instruction_candidate_count < 2
            or type(predictor_count) is not int or predictor_count != 1):
        raise ValueError("instruction review bound requires >=2 candidates and the pinned single predictor")
    generated = instruction_candidate_count - 1
    return {"incumbent": 1, "generated_instructions": generated, "total": 1 + generated}


def require_sufficient_mipro_instruction_review_budget(configured_budget: int,
                                                       bounds: dict[str, int]) -> None:
    if type(configured_budget) is not int or configured_budget < bounds["total"]:
        raise ValueError(
            "MIPRO candidate-grounding budget cannot complete the frozen proposal stage: "
            f"needs at least {bounds['total']} calls "
            f"(incumbent={bounds['incumbent']}, generated={bounds['generated_instructions']}), "
            f"configured {configured_budget}; rejected before model dispatch")


def load_frozen_cohort(
        cohort_path: Path,
        train_rows: list[dict[str, Any]],
        *,
        clinifact_history_report: Path | None = None):
    from src.data.research_cohort import CohortContext, validate_cohort
    context = CohortContext(DATASET, MANIFEST_PATH, ROOT, LABELS,
                            HEALTHFC_HISTORY_REPORT_PATH, read_json, sha256_file, sha256_obj)
    return validate_cohort(context, cohort_path, train_rows,
                           clinifact_history_report=clinifact_history_report)

def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot score an empty task scope")
    n = len(rows)
    answered = sum(int(row.get("answer_index", -1)) >= 0 for row in rows)
    correct = sum(bool(row.get("correct")) for row in rows)
    per_label = {}
    for label in LABELS:
        subset = [row for row in rows if str(row.get("gold_label")) == label]
        if not subset:
            raise ValueError(f"scoring scope lacks label: {label}")
        sub_answered = sum(int(row.get("answer_index", -1)) >= 0 for row in subset)
        sub_correct = sum(bool(row.get("correct")) for row in subset)
        per_label[label] = {
            "n": len(subset), "accuracy": sub_correct / len(subset),
            "coverage": sub_answered / len(subset),
            "answered_precision": sub_correct / sub_answered if sub_answered else 0.0,
        }
    return {
        "n": n, "accuracy": correct / n, "coverage": answered / n,
        "answered_precision": correct / answered if answered else 0.0,
        "per_gold_label": per_label,
    }


def class_aware_gain(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    if candidate["accuracy"] <= baseline["accuracy"] + 1e-12:
        return False
    for label, base in baseline["per_gold_label"].items():
        current = candidate["per_gold_label"].get(label) or {}
        for key in ("accuracy", "coverage", "answered_precision"):
            if float(current.get(key, -1)) + 1e-12 < float(base[key]):
                return False
    return True


def selection_gate_eligibility_reason(dataset: str, exploratory_only: bool,
                                      candidate_status: str) -> str:
    """Describe this dataset's actual restriction without a cross-dataset fallback."""
    if dataset == EI_DATASET and exploratory_only:
        return "insufficient_annotation_votes_for_selection_gate"
    candidate_restrictions = {
        "fr201_nonpromotable_component": "fr201_candidate_category_not_system_prompt",
        "instruction_search_coverage_failed": "minimum_grounded_novel_prompt_coverage_not_met",
        "selected_prompt_not_full_evaluated": "selected_candidate_prompt_not_full_evaluated",
    }
    if candidate_status in candidate_restrictions:
        return candidate_restrictions[candidate_status]
    if dataset == EI_DATASET:
        if candidate_status == "internal_gate_passed":
            return "eligible_after_internal_gate"
        if candidate_status == "internal_gate_rejected":
            return "internal_class_aware_gain_gate_rejected"
        return "no_admissible_candidate"
    return {
        HEALTHVER_DATASET: "healthver_project_split_exhausted_exploratory_only",
        CLINIFACT_DATASET: "clinifact_single_vote_exploratory_only",
        SCIFACT_DATASET: "scifact_exploratory_task_source_no_confirmation",
        HEALTHFC_DATASET: "healthfc_exploratory_task_source_no_confirmation",
    }.get(dataset, "unsupported_dataset_for_selection_gate")


def paired_internal_analysis(
        baseline_rows_by_seed: dict[int, list[dict[str, Any]]],
        candidate_rows_by_seed: dict[int, list[dict[str, Any]]],
        seeds: list[int], *, bootstrap_samples: int = 10_000) -> dict[str, Any]:
    """Report paired inference without treating repeated task calls as new items.

    McNemar is reported separately for each execution label over unique task
    IDs. The aggregate interval resamples task IDs as clusters and averages
    repeated-label accuracy within each task before resampling.
    """
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("paired analysis requires unique execution labels")
    if set(baseline_rows_by_seed) != set(seeds) or set(candidate_rows_by_seed) != set(seeds):
        raise ValueError("paired analysis execution labels differ")

    baseline_maps: dict[int, dict[str, dict[str, Any]]] = {}
    candidate_maps: dict[int, dict[str, dict[str, Any]]] = {}
    task_ids: set[str] | None = None
    per_label = {}
    for seed in seeds:
        baseline_map = {str(row.get("task_id") or ""): row
                        for row in baseline_rows_by_seed[seed]}
        candidate_map = {str(row.get("task_id") or ""): row
                         for row in candidate_rows_by_seed[seed]}
        if (not baseline_map or "" in baseline_map or "" in candidate_map
                or len(baseline_map) != len(baseline_rows_by_seed[seed])
                or len(candidate_map) != len(candidate_rows_by_seed[seed])):
            raise ValueError(f"paired analysis has invalid or duplicate task IDs for label {seed}")
        if set(baseline_map) != set(candidate_map):
            raise ValueError(f"baseline/candidate task IDs differ for label {seed}")
        if task_ids is None:
            task_ids = set(baseline_map)
        elif task_ids != set(baseline_map):
            raise ValueError("execution labels do not cover the same unique task IDs")
        for task_id in sorted(baseline_map):
            if baseline_map[task_id].get("gold_label") != candidate_map[task_id].get("gold_label"):
                raise ValueError(f"baseline/candidate gold label differs for {task_id}")
        baseline_maps[seed] = baseline_map
        candidate_maps[seed] = candidate_map
        mc = mcnemar_exact(
            [float(bool(baseline_map[task_id].get("correct"))) for task_id in sorted(task_ids)],
            [float(bool(candidate_map[task_id].get("correct"))) for task_id in sorted(task_ids)],
        )
        per_label[str(seed)] = {"n_paired_unique_tasks": len(task_ids), **mc}

    assert task_ids is not None
    ordered_ids = sorted(task_ids)
    baseline_task_accuracy = [
        sum(float(bool(baseline_maps[seed][task_id].get("correct"))) for seed in seeds) / len(seeds)
        for task_id in ordered_ids
    ]
    candidate_task_accuracy = [
        sum(float(bool(candidate_maps[seed][task_id].get("correct"))) for seed in seeds) / len(seeds)
        for task_id in ordered_ids
    ]
    transition_counts = {
        "unchanged_all_labels": 0,
        "improved_without_regression": 0,
        "regressed_without_improvement": 0,
        "mixed_improvements_and_regressions": 0,
    }
    for task_id in ordered_ids:
        deltas = [int(bool(candidate_maps[seed][task_id].get("correct")))
                  - int(bool(baseline_maps[seed][task_id].get("correct")))
                  for seed in seeds]
        improved = any(delta > 0 for delta in deltas)
        regressed = any(delta < 0 for delta in deltas)
        if not improved and not regressed:
            transition_counts["unchanged_all_labels"] += 1
        elif improved and regressed:
            transition_counts["mixed_improvements_and_regressions"] += 1
        elif improved:
            transition_counts["improved_without_regression"] += 1
        else:
            transition_counts["regressed_without_improvement"] += 1

    return {
        "unit_of_analysis": "unique task ID; repeated execution labels are clustered within task",
        "unique_task_count": len(ordered_ids),
        "repeated_execution_label_count": len(seeds),
        "paired_row_count_for_context_only": len(ordered_ids) * len(seeds),
        "per_execution_label_mcnemar": per_label,
        "task_clustered_accuracy_bootstrap": paired_bootstrap(
            baseline_task_accuracy, candidate_task_accuracy,
            B=bootstrap_samples, rng_seed=20261005),
        "task_transition_consistency": transition_counts,
    }


def _load_rows(path_: Path, seeds: list[int], task_by_id: dict[str, dict],
               *, candidate_id: str | None = None) -> dict[tuple[int, str], dict]:
    saved: dict[tuple[int, str], dict] = {}
    if not path_.exists():
        return saved
    for raw in read_jsonl(path_):
        key = (int(raw.get("seed", -1)), str(raw.get("task_id") or ""))
        if key in saved or key[0] not in seeds or key[1] not in task_by_id:
            raise ValueError(f"duplicate or out-of-scope task checkpoint row: {key}")
        if candidate_id is not None and raw.get("candidate_id") != candidate_id:
            raise ValueError("candidate checkpoint belongs to another candidate")
        validated = validate_task_answer_row(
            raw, task_by_id[key[1]], context=f"EI-MIPRO-checkpoint:{key}",
            require_reported_gold_index=True)
        # The generic row validator case-folds labels for compatibility across
        # tasks. This runner's metrics use the frozen dataset label strings,
        # so restore the exact label from the saved task options on checkpoint
        # loads (fresh calls already store this exact value).
        task = task_by_id[key[1]]
        gold_index = int(task["correct_index"])
        validated["gold_label"] = str(task["options"][gold_index])
        saved[key] = validated
    return saved


def _provider_failure_metadata(exc: Exception) -> dict[str, Any]:
    """Keep safe provider diagnostics in run reports after a failed task."""
    attempts = getattr(exc, "provider_attempts", None)
    if not isinstance(attempts, list) or not attempts:
        return {}
    allowed_headers = {
        "retry-after", "retry-after-ms", "x-request-id", "request-id",
        "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests", "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens",
    }
    safe_attempts = []
    for item in attempts:
        if not isinstance(item, dict):
            continue
        headers = item.get("headers")
        safe = {
            "status": item.get("status") if isinstance(item.get("status"), int) else None,
            "duration_ms": item.get("durationMs") if isinstance(item.get("durationMs"), int) else None,
            "headers": {
                str(key): str(value)[:256]
                for key, value in (headers.items() if isinstance(headers, dict) else [])
                if str(key).lower() in allowed_headers
            },
        }
        detail = item.get("providerError")
        if isinstance(detail, dict):
            safe["provider_error"] = {
                key: _redact_provider_error_text(detail[key])
                for key in ("code", "type", "message")
                if isinstance(detail.get(key), (str, int))
            }
        safe_attempts.append(safe)
    return {
        "provider_http_attempts": len(attempts),
        "provider_statuses": [item["status"] for item in safe_attempts
                               if item.get("status") is not None],
        "provider_attempts": safe_attempts,
    }


def _redact_provider_error_text(value: Any) -> str:
    text = str(value)
    text = re.sub(r"\bBearer\s+[^\s,;]+", "Bearer [REDACTED]", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:sk|sf)-[A-Za-z0-9._-]{8,}\b", "[REDACTED]", text)
    text = re.sub(r"\b(api[_ -]?key|token)\s*[:=]\s*[^\s,;]+",
                  r"\1=[REDACTED]", text, flags=re.IGNORECASE)
    return text[:512]


def _backfill_legacy_mipro_ledger(
        *, ledger_path: Path, study_dir: Path, plan: dict[str, Any],
        baseline_checkpoint: Path, compile_checkpoint: Path,
        candidate_checkpoint: Path, internal_ids: set[str]) -> dict[str, Any] | None:
    """Conservatively seed one pre-ledger EI MIPRO run from its saved artifacts."""
    report_path = study_dir / "report.json"
    if ledger_path.exists():
        if report_path.exists():
            previous = read_json(report_path)
            if ((previous.get("plan") or {}).get("plan_sha256")
                    == plan.get("plan_sha256")):
                return ((previous.get("request_budget") or {}).get("legacy_backfill"))
        return None
    if not compile_checkpoint.exists():
        return None
    if not report_path.exists():
        return None
    previous_report = read_json(report_path)
    if ((previous_report.get("plan") or {}).get("plan_sha256")
            != plan.get("plan_sha256")
            or previous_report.get("status") != "internal_confirmation_failed"):
        return None
    compiled = read_json(compile_checkpoint)
    mipro = compiled.get("mipro") or {}
    baseline_rows = len(read_jsonl(baseline_checkpoint))
    candidate_rows = len(read_jsonl(candidate_checkpoint))
    compile_gate_calls = sum(
        1 for record in (compiled.get("candidate_instruction_gates") or [])
        if isinstance(record, dict)
        and isinstance(record.get("grounding_review"), dict)
        and isinstance(record["grounding_review"].get("token_usage"), dict)
    )
    session_root = artifacts_path("pi_sessions")
    compile_time = compile_checkpoint.stat().st_mtime
    failed_sessions = 0
    if session_root.exists():
        for session_file in session_root.rglob("*.jsonl"):
            try:
                if session_file.stat().st_mtime < compile_time:
                    continue
                raw = session_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if ("429 status code" in raw
                    and any(f'"taskId":"{task_id}"' in raw for task_id in internal_ids)):
                failed_sessions += 1
    if failed_sessions <= 0:
        return None

    task_calls = baseline_rows + int(mipro.get("task_evaluations") or 0) \
        + candidate_rows + failed_sessions
    router_calls = task_calls if plan.get("routing_mode") == "model" else 0
    proposer_calls = int(mipro.get("prompt_model_calls") or 0)
    # Each process restart reruns the incumbent grounding check before loading
    # the cached compile artifact; the initial check is among compile reviews.
    grounding_calls = compile_gate_calls + max(0, failed_sessions - 1)
    logical_by_component = {
        "task_answer": task_calls,
        "skill_router": router_calls,
        "mipro_proposer": proposer_calls,
        "grounding": grounding_calls,
    }
    logical_total = sum(logical_by_component.values())
    known_failed_http = failed_sessions * 3
    unknown_reservation = max(
        0, logical_total * 3 - known_failed_http)
    seeded = seed_legacy_usage(
        ledger_path, logical_by_component=logical_by_component,
        known_http_statuses=[429] * known_failed_http,
        unknown_physical_reservation=unknown_reservation,
    )
    if not seeded:
        return None
    return {
        "source": "legacy_checkpoint_backfill",
        "logical_by_component": logical_by_component,
        "logical_operations": logical_total,
        "known_http_429_attempts": known_failed_http,
        "unknown_physical_attempt_upper_bound_reserved": unknown_reservation,
        "physical_exact": False,
    }


def _score_repeated_scope(cfg: dict, state: ArtifactState, scope: list[dict],
                          seeds: list[int], path_: Path, *,
                          frozen_skill_routes: FrozenSkillRouteCache | None = None,
                          candidate_id: str | None = None) -> dict[int, list[dict]]:
    ids = {str(row["task_id"]) for row in scope}
    task_by_id = {str(row["task_id"]): row for row in scope}
    saved = _load_rows(path_, seeds, task_by_id, candidate_id=candidate_id)
    out: dict[int, list[dict]] = {}
    for seed in seeds:
        provider = build_provider(cfg, seed=seed)
        rows = []
        for task_id in sorted(ids):
            key = (seed, task_id)
            row = saved.get(key)
            if row is None:
                result = TaskRunner(
                    cfg, state, provider=provider, seed=seed,
                    frozen_skill_routes=frozen_skill_routes,
                ).run(task_by_id[task_id])
                row = validate_task_answer_row(
                    result.to_dict(), task_by_id[task_id],
                    context=f"EI-MIPRO:{seed}:{task_id}",
                    require_reported_gold_index=True)
                row.update({
                    "seed": seed,
                    "gold_label": str(task_by_id[task_id]["options"][
                        task_by_id[task_id]["correct_index"]]),
                })
                if candidate_id is not None:
                    row["candidate_id"] = candidate_id
                with path_.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                saved[key] = row
            rows.append(row)
        out[seed] = rows
    return out


def _baseline_cache_identity(plan: dict[str, Any], state: ArtifactState,
                            internal_scope: list[dict[str, Any]]) -> dict[str, Any]:
    """Identify baseline outputs by task-execution semantics, not optimizer knobs.

    MIPRO candidate count, trial count, proposal seed and demo ceiling do not
    affect incumbent answers. Reuse is safe only while the task protocol,
    execution labels, dataset rows, model/runtime source fingerprint and
    incumbent artifact remain identical.
    """
    task_rows = sorted(internal_scope, key=lambda row: str(row["task_id"]))
    return {
        "schema_version": 1,
        "protocol": plan["protocol"],
        "dataset": plan["dataset"],
        # Proposer settings do not affect incumbent TaskRunner outputs. Keep
        # the cache bound to the task-answer configuration so changing a
        # bounded DSPy retry policy does not force an identical 63-row baseline
        # to be purchased again. For old plans, the normalized fingerprint is
        # byte-for-byte the former config fingerprint because retries were 0.
        "config_sha256": plan.get(
            "baseline_task_config_sha256", plan["config_sha256"]),
        "release_id": state.release_id,
        "system_prompt_sha256": content_sha256(state.system_prompt),
        "registry_fingerprint": state.registry.fingerprint(),
        "task_demos_sha256": sha256_obj(state.task_demos),
        "cohort_sha256": plan["cohort_sha256"],
        "internal_scope_sha256": sha256_obj(task_rows),
        "task_model": plan["task_model"],
        "routing_mode": plan["routing_mode"],
        "execution_labels": [int(value) for value in plan["execution_labels"]],
        "http_response_cache_disabled_for_run": plan[
            "http_response_cache_disabled_for_run"],
    }


def _baseline_checkpoint_for(identity: dict[str, Any], *, create: bool) -> tuple[Path, str]:
    fingerprint = sha256_obj(identity)
    directory = artifacts_path("probe", "evidence_inference_mipro_baselines", fingerprint)
    checkpoint = directory / "internal-baseline.jsonl"
    manifest = directory / "identity.json"
    if manifest.exists():
        saved = read_json(manifest)
        if saved != {"fingerprint": fingerprint, "identity": identity}:
            raise ValueError("baseline cache identity mismatch; refusing cross-protocol reuse")
    elif checkpoint.exists():
        raise ValueError("baseline checkpoint exists without identity manifest; refusing reuse")
    if create:
        directory.mkdir(parents=True, exist_ok=True)
        if not manifest.exists():
            write_json(manifest, {"fingerprint": fingerprint, "identity": identity})
    return checkpoint, fingerprint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--dataset-adapter", choices=["evidence_inference", "healthver", "scifact", "healthfc", "clinifact"],
                        default="evidence_inference")
    parser.add_argument("--cohort", type=Path)
    parser.add_argument(
        "--clinifact-history-report", type=Path,
        default=ROOT / "artifacts/probe/clinifact_history_scan_20261008.json",
        help="history scan report bound to a CliniFact cohort",
    )
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--num-trials", type=int, default=4)
    parser.add_argument(
        "--minimum-generated-instructions-evaluated", type=int, default=1,
        help="require at least this many distinct grounded novel prompts to receive full validation",
    )
    parser.add_argument("--max-labeled-demos", type=int, default=3)
    parser.add_argument(
        "--require-balanced-demo-bundle", action="store_true",
        help="restrict MIPRO to three train demos with one example per EI label",
    )
    parser.add_argument("--mipro-task-evaluation-budget", type=int, default=120)
    parser.add_argument("--proposer-call-budget", type=int, default=24)
    parser.add_argument(
        "--mipro-proposer-retries-after-first", type=int, default=0,
        help="bounded project HTTP retries per proposer LM call; included in the physical request ceiling",
    )
    parser.add_argument("--grounding-call-budget", type=int, default=12)
    parser.add_argument(
        "--physical-request-budget", type=int,
        help="cap worst-case dispatched HTTP attempts; disables automatic retries",
    )
    parser.add_argument("--execution-labels", nargs="+", type=int,
                        default=[202610051, 202610052, 202610053])
    parser.add_argument("--seed", type=int, default=202610051)
    parser.add_argument("--extend-logical-budget", type=int)
    parser.add_argument("--extend-physical-budget", type=int)
    parser.add_argument(
        "--resume-study-id",
        help="resume one frozen study after checking all non-fingerprint plan fields",
    )
    parser.add_argument("--resume-429-backoff-base-seconds", type=float)
    parser.add_argument(
        "--expected-plan-sha256",
        help="required with --execute; must match the previously reviewed zero-call plan",
    )
    parser.add_argument("--routing-mode", choices=["lexical", "model"], default="model")
    parser.add_argument("--task-model-id", choices=["Qwen/Qwen3-32B", "Qwen/Qwen3-8B"],
                        default="Qwen/Qwen3-32B")
    parser.add_argument(
        "--instruction-review-model-id", choices=["Qwen/Qwen3-8B"],
        help="exploratory-only reviewer override; does not establish reviewer calibration")
    parser.add_argument("--pi-timeout-max-attempts", choices=[1, 2, 3], type=int,
                        default=1,
                        help="bounded retries after a Pi task-call timeout")
    parser.add_argument("--task-request-cooldown-seconds", type=float, default=30.0,
                        help="frozen pause after each successful Pi task request (0 to 60 seconds)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true",
                      help="write and print the zero-call plan for review")
    mode.add_argument("--execute", action="store_true",
                      help="execute only the plan identified by --expected-plan-sha256")
    args = parser.parse_args(argv)
    if args.dataset_adapter == "evidence_inference" and args.cohort is None:
        parser.error(
            "Evidence Inference MIPRO requires an explicit frozen --cohort; "
            "the repository default cohort may be stale")
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256 or ""):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    if args.resume_study_id and args.validate_only:
        parser.error("--resume-study-id requires --execute")

    configure_dataset_adapter(args.dataset_adapter)
    if args.cohort is None:
        args.cohort = COHORT_PATH

    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("task-input MIPRO requires an isolated RSI_HOME")
    if (args.num_candidates < 2 or args.num_trials < 1 or
            args.minimum_generated_instructions_evaluated < 0 or
            args.mipro_task_evaluation_budget <= 0 or args.proposer_call_budget <= 0 or
            args.grounding_call_budget <= 0 or not 1 <= args.max_labeled_demos <= 3):
        raise ValueError("invalid MIPRO candidate/trial/demo/call budget")
    if not 0 <= args.mipro_proposer_retries_after_first <= MIPRO_PROPOSER_MAX_RETRIES:
        raise ValueError(
            "MIPRO proposer retries-after-first must be between 0 and "
            f"{MIPRO_PROPOSER_MAX_RETRIES}")
    if args.require_balanced_demo_bundle and args.max_labeled_demos != len(LABELS):
        raise ValueError("balanced demo bundles require one example per task label")
    if not args.execution_labels or len(set(args.execution_labels)) != len(args.execution_labels):
        raise ValueError("execution labels must be a non-empty unique list")
    if ((args.extend_logical_budget is None)
            != (args.extend_physical_budget is None)):
        raise ValueError("logical and physical budget extensions must be provided together")
    resume_backoff = (args.resume_429_backoff_base_seconds
                      if args.resume_429_backoff_base_seconds is not None
                      else (15.0 if args.resume_study_id else 0.0))
    if resume_backoff < 0 or resume_backoff > 60:
        raise ValueError("429 backoff base must be between 0 and 60 seconds")
    if args.resume_429_backoff_base_seconds is not None and not args.resume_study_id:
        raise ValueError("429 backoff override requires --resume-study-id")

    cfg = prepare_config(args.config, proposer_calls=args.proposer_call_budget,
                         routing_mode=args.routing_mode,
                         task_model_id=args.task_model_id,
                         instruction_review_model_id=args.instruction_review_model_id,
                         pi_timeout_max_attempts=args.pi_timeout_max_attempts,
                         task_request_cooldown_seconds=args.task_request_cooldown_seconds)
    if args.physical_request_budget is not None:
        apply_physical_request_budget(
            cfg, args.physical_request_budget,
            proposer_retries_after_first=args.mipro_proposer_retries_after_first)
    elif args.mipro_proposer_retries_after_first:
        cfg.setdefault("models", {}).setdefault("reflection", {})[
            "max_retries"] = args.mipro_proposer_retries_after_first
    validate_mipro_proposer_config(cfg)
    state = load_state(RELEASE_ID)
    if state.release_id != RELEASE_ID:
        raise ValueError("loaded release differs from the task-input baseline release")
    if state.task_demos:
        raise ValueError("this system_prompt search requires a frozen empty parent demo pool")
    active_path = artifacts_path("releases", "ACTIVE")
    if not active_path.exists() or active_path.read_text(encoding="utf-8").strip() != RELEASE_ID:
        raise ValueError("isolated RSI_HOME ACTIVE must point to the task-input baseline release")
    train_rows = load_tasks(splits=["dev_optimize"])
    cohort, scopes, task_by_id = load_frozen_cohort(
        args.cohort.resolve(), train_rows,
        clinifact_history_report=args.clinifact_history_report.resolve())
    train = scopes["mipro_train"]
    search_validation = scopes["search_validation"]
    internal_scope_name = ("internal_confirmation" if DATASET == EI_DATASET
                           else "internal_candidate")
    internal = scopes[internal_scope_name]
    if set(task_by_id) != {str(row["task_id"]) for row in train_rows}:
        raise ValueError("train adapter task map is inconsistent")
    train_ids = {str(row["task_id"]) for row in train}
    search_ids = {str(row["task_id"]) for row in search_validation}
    internal_ids = {str(row["task_id"]) for row in internal}
    if train_ids & search_ids or train_ids & internal_ids or search_ids & internal_ids:
        raise ValueError("nested MIPRO scopes overlap")
    proposer_guidance = mipro_proposer_guidance(prompt_only=True)
    proposer_call_upper_bounds = mipro_proposer_lm_call_upper_bounds(
        train_example_count=len(train_ids),
        instruction_candidate_count=args.num_candidates,
        predictor_count=MIPRO_TASK_INPUT_PREDICTOR_COUNT,
        view_data_batch_size=MIPRO_VIEW_DATA_BATCH_SIZE,
        program_aware=proposer_guidance["program_aware_proposer"],
        data_aware=proposer_guidance["data_aware_proposer"],
    )
    require_sufficient_mipro_proposer_budget(
        args.proposer_call_budget, proposer_call_upper_bounds)
    instruction_review_call_upper_bounds = mipro_instruction_review_call_upper_bounds(
        instruction_candidate_count=args.num_candidates,
        predictor_count=MIPRO_TASK_INPUT_PREDICTOR_COUNT)
    require_sufficient_mipro_instruction_review_budget(
        args.grounding_call_budget, instruction_review_call_upper_bounds)

    prompt_cfg = copy.deepcopy(cfg)
    prompt_cfg.setdefault("evolution", {})["allowed_artifacts"] = ["system_prompt"]
    candidate_gate_examples = train + search_validation
    task_contract = task_contract_for_examples(train)
    task_contract_context, proposal_task_contract = build_proposal_task_contract(
        task_contract)
    runtime_contract = (task_contract if DATASET == CLINIFACT_DATASET else None)
    if runtime_contract is not None:
        # Use the same deterministic composition for incumbent and candidates;
        # label meanings and answer protocol no longer depend on proposer text.
        state = ArtifactState(
            release_id=state.release_id,
            system_prompt=compose_task_instruction(
                CLINIFACT_MIPRO_STRATEGY_SEED, runtime_contract),
            registry=state.registry, task_demos=state.task_demos)
    baseline_static = validate_task_input_prompt(
        state.system_prompt, candidate_gate_examples, prompt_cfg, state)
    if baseline_static.get("reason") not in {
            "grounding_review_missing_or_failed",
            "task_contract_review_missing_or_failed"}:
        raise ValueError("incumbent prompt fails label/static gate: " + str(baseline_static.get("reason")))

    if (args.num_trials + 1) * len(search_ids) > args.mipro_task_evaluation_budget:
        raise ValueError("MIPRO evaluation budget is below the full-validation lower bound")
    confirmation_evaluations = len(internal_ids) * len(args.execution_labels) * 2
    minimum_task_evaluations = (args.num_trials + 1) * len(search_ids) + confirmation_evaluations
    maximum_task_evaluations = args.mipro_task_evaluation_budget + confirmation_evaluations
    frozen_router_task_count = (
        len(search_ids) + len(internal_ids)
        if args.routing_mode == "model"
        and not str((cfg.get("skills") or {}).get("pinned_skill") or "")
        and state.registry.skills
        else 0
    )
    maximum_router_calls = frozen_router_task_count
    # The logical-operation ceiling and the actual HTTP-attempt ceiling are
    # separate. Compute the latter from the provider adapters' actual retry
    # semantics; a failed request still consumes one physical attempt.
    physical_http_attempts_by_component = physical_http_attempt_upper_bounds(
        cfg,
        task_evaluations=maximum_task_evaluations,
        router_calls=maximum_router_calls,
        proposer_calls=args.proposer_call_budget,
        grounding_calls=args.grounding_call_budget,
    )
    maximum_physical_http_requests = sum(
        physical_http_attempts_by_component.values())
    if (args.physical_request_budget is not None
            and maximum_physical_http_requests > args.physical_request_budget):
        raise ValueError(
            "worst-case physical HTTP attempts exceed --physical-request-budget: "
            f"{maximum_physical_http_requests} > {args.physical_request_budget}")
    pi_provider_retries = max(0, min(4, int(
        cfg["runtime"].get("pi_provider_max_retries", 4))))
    pi_timeout_attempts = int(cfg["runtime"].get("pi_timeout_max_attempts", 1))
    http_task_attempts = int(cfg["models"]["task"].get("max_retries", 6))
    proposer_retries_after_first = int(cfg["models"]["reflection"].get(
        "max_retries", MIPRO_PROPOSER_MAX_RETRIES))
    http_proposer_attempts = proposer_retries_after_first + 1
    exploratory_only = cohort_is_exploratory(cohort)
    plan = {
        "plan_schema_version": 4,
        "request_budget_version": (
            "bounded-physical-attempt-budget-v1"
            if args.physical_request_budget is not None
            else "logical-plus-wire-attempts-v2"),
        "protocol": (EI_MIPRO_PROTOCOL
                     if DATASET == EI_DATASET else
                     str(cohort.get("protocol_version") or
                         "healthver-component-disjoint-mipro-task-input-v2")),
        "dataset": DATASET, "release_id": state.release_id,
        "cohort_sha256": cohort["cohort_sha256"],
        "task_model": cfg["models"]["task"]["model_id"],
        "proposer_model": cfg["models"]["reflection"]["model_id"],
        "instruction_review_model": (cfg.get("skills") or {}).get(
            "instruction_review_model_id") or (cfg.get("skills") or {}).get(
            "router_model_id"),
        "routing_mode": args.routing_mode,
        "skill_route_cache_policy": "freeze_first_model_route_per_task_v1",
        "skill_route_cache_unique_task_count": frozen_router_task_count,
        "scopes": {name: len(rows) for name, rows in scopes.items()},
        "class_counts": {name: entry["class_counts"]
                          for name, entry in cohort["scopes"].items()},
        "execution_labels": args.execution_labels,
        "execution_labels_are_independent_rng_seeds": False,
        "mipro_execution_seed": args.seed,
        "mipro_num_candidates": args.num_candidates,
        "mipro_num_trials": args.num_trials,
        "minimum_generated_instructions_evaluated": (
            args.minimum_generated_instructions_evaluated),
        "mipro_max_labeled_demos": args.max_labeled_demos,
        "mipro_balanced_demo_labels": list(LABELS),
        "mipro_require_balanced_demo_bundle": args.require_balanced_demo_bundle,
        "mipro_search_space": "prompt_only_frozen_empty_demo_pool_v1",
        "mipro_demo_pool_optimized": False,
        "mipro_candidate_selection_policy": MIPRO_SELECTION_POLICY_VERSION,
        "mipro_proposer_configuration": {
            **proposer_guidance,
            "predictor_count": MIPRO_TASK_INPUT_PREDICTOR_COUNT,
            "view_data_batch_size": MIPRO_VIEW_DATA_BATCH_SIZE,
        },
        "proposal_task_contract": proposal_task_contract,
        "mipro_persist_completed_trial_scores": True,
        "task_demo_block_char_limit": int(cfg["runtime"][
            "pi_task_demos_max_block_chars"]),
        "task_demo_example_char_limit": int(cfg["runtime"][
            "pi_task_demo_max_example_chars"]),
        "http_response_cache_disabled_for_run": True,
        "mipro_full_validation_lower_bound": (args.num_trials + 1) * len(search_ids),
        "internal_baseline_and_candidate_evaluations": confirmation_evaluations,
        "minimum_task_evaluations_total": minimum_task_evaluations,
        "maximum_task_evaluations_total": maximum_task_evaluations,
        "minimum_annotation_votes": int(cohort.get("minimum_annotation_votes", 1)),
        "selection_gate_eligible": not exploratory_only,
        "cohort_confirmation_eligible": bool(
            cohort.get("eligible_for_confirmation", False)),
        "exploratory_search_only": exploratory_only,
        "candidate_confirmation_allowed": (
            bool(cohort.get("eligible_for_confirmation", False))
            and not exploratory_only),
        "candidate_release_allowed": False,
        "reviewer_model_override_is_calibration_evidence": False,
        "maximum_task_model_calls": maximum_task_evaluations,
        "maximum_skill_router_model_calls": maximum_router_calls,
        "proposer_model_call_ceiling": args.proposer_call_budget,
        "mipro_proposer_lm_call_upper_bounds": proposer_call_upper_bounds,
        "mipro_proposer_lm_call_budget_sufficient": True,
        "grounding_model_call_ceiling": args.grounding_call_budget,
        "instruction_review_call_upper_bounds": instruction_review_call_upper_bounds,
        "instruction_review_budget_sufficient": True,
        "maximum_configured_model_calls": (
            maximum_task_evaluations + maximum_router_calls
            + args.proposer_call_budget + args.grounding_call_budget),
        "model_call_ceiling_counts_logical_operations": True,
        "request_budget_semantics": {
            "logical_operation": "one provider operation; may issue multiple HTTP attempts",
            "physical_http_attempt": "one dispatched HTTP request, including failed responses and transport attempts",
            "logical_operations_are_not_http_attempts": True,
        },
        "provider_retry_policy": {
            "pi_successful_request_cooldown_seconds": float(
                cfg["runtime"].get("pi_request_cooldown_seconds", 0)),
            "http_transport_backoff_base_seconds": float(
                cfg["runtime"].get("http_provider_transport_backoff_base_seconds", 0)),
            "pi_provider_retries_after_first_attempt": pi_provider_retries,
            "pi_provider_attempts_per_bridge": pi_provider_retries + 1,
            "pi_provider_429_backoff_base_seconds": float(
                cfg["runtime"].get("pi_provider_429_backoff_base_seconds", 0)),
            "pi_bridge_timeout_attempts": pi_timeout_attempts,
            "http_task_router_configured_max_attempts": http_task_attempts,
            "mipro_proposer_http_attempts_per_call": http_proposer_attempts,
            "mipro_proposer_retries_after_first": proposer_retries_after_first,
            "retry_after_honored": True,
            "maximum_retry_after_ms": 60000,
        },
        "maximum_physical_http_requests_by_component": physical_http_attempts_by_component,
        "maximum_physical_http_requests": maximum_physical_http_requests,
        "physical_request_budget": args.physical_request_budget,
        "physical_request_budget_compliant": (
            maximum_physical_http_requests <= args.physical_request_budget
            if args.physical_request_budget is not None else None),
        "mipro_task_evaluation_budget": args.mipro_task_evaluation_budget,
        "proposer_call_budget": args.proposer_call_budget,
        "grounding_call_budget": args.grounding_call_budget,
        "api_key_configured": {
            section: bool(os.environ.get(str((cfg["models"].get(section) or {}).get("api_key_env") or "")))
            for section in ("task", "reflection")
        },
        "api_key_explicit_at_launch": {
            name: name in _PROCESS_ENV_AT_START
            for name in _configured_api_envs(cfg)
        },
        "selection_gate_read": False,
        # Project-owned official holdouts remain sealed in this search runner.
        # HealthVer's public upstream split materialization is disclosed under
        # separate source_* fields below; it is not the project official holdout.
        "official_validation_read": False,
        "official_test_read": False,
        "project_selection_gate_or_confirmation_read": False,
        # These source-material flags are distinct from project holdout reads.
        # CliniFact re-groups its public upstream train and validation rows;
        # the old official_validation_read flag above names the sealed project
        # holdout, not the public upstream source split.
        "source_upstream_validation_materialized": DATASET in {
            HEALTHVER_DATASET, SCIFACT_DATASET, CLINIFACT_DATASET},
        "source_upstream_train_and_validation_regrouped": DATASET == CLINIFACT_DATASET,
        "source_upstream_test_materialized": DATASET == HEALTHVER_DATASET,
        "source_test_is_blind_confirmation": False if DATASET in {HEALTHVER_DATASET, SCIFACT_DATASET, HEALTHFC_DATASET, CLINIFACT_DATASET} else None,
        "data_source_scope": ("public HealthVer; all upstream splits materialized; exploration only"
                              if DATASET == HEALTHVER_DATASET else
                              "public SciFact labeled train/dev; prior-use connected components excluded; test sealed; exploration only"
                              if DATASET == SCIFACT_DATASET else
                              "HealthFC fixed supplied evidence; history-screened and claim/article-grouped; no official holdout; exploration only"
                              if DATASET == HEALTHFC_DATASET else
                              "CliniFact CC-BY fixed PubMed abstracts; train/validation re-grouped by claim/trial/PMID; exact-history screened; official test sealed; exploration only"
                              if DATASET == CLINIFACT_DATASET else
                              "Evidence Inference train-only nested article cohort"),
        "retrieval_used": False,
        "exploratory_only": exploratory_only,
        "model_calls": 0,
    }
    protocol_sha = config_fingerprint(cfg)
    plan["config_sha256"] = protocol_sha
    plan["baseline_task_config_sha256"] = baseline_task_config_fingerprint(cfg)
    baseline_identity = _baseline_cache_identity(plan, state, internal)
    plan["baseline_cache_fingerprint"] = sha256_obj(baseline_identity)
    plan["plan_sha256"] = compute_plan_sha256(plan)
    resume_report = None
    if args.resume_study_id:
        resume_dir = artifacts_path("probe", STUDY_NAMESPACE,
                                    args.resume_study_id)
        frozen_plan_path = resume_dir / "plan.json"
        frozen_report_path = resume_dir / "report.json"
        if not frozen_plan_path.exists():
            raise ValueError("resume study must have a frozen plan")
        frozen_plan = read_json(frozen_plan_path)
        phase_path = resume_dir / "phase-checkpoint.json"
        phase_verified = False
        resume_report = read_json(frozen_report_path) if frozen_report_path.exists() else {}
        if resume_report.get("status") != "internal_confirmation_failed":
            resume_report = validate_phase_checkpoint(read_json(phase_path), frozen_plan,
                resume_dir / "request-ledger.jsonl")
            phase_verified = True
        plan = validate_resume_plan(
            plan, frozen_plan,
            report_status=str(resume_report.get("status") or ""),
            resume_study_id=args.resume_study_id,
            phase_checkpoint_verified=phase_verified)

    if args.validate_only:
        preflight_path = artifacts_path(
            "probe", STUDY_NAMESPACE, "preflight-" + plan["plan_sha256"][:16] + ".json")
        write_json(preflight_path, {"status": "valid", **plan})
        print(json.dumps({"status": "resume_compatible" if args.resume_study_id else "valid",
                          "preflight_path": str(preflight_path),
                          **plan}, ensure_ascii=False))
        return 0

    if args.expected_plan_sha256 != plan["plan_sha256"]:
        raise ValueError(
            "expected plan SHA does not match the recomputed MIPRO plan; no model calls made")

    # The pinned optimizer is an execution dependency only. Keep preflight
    # usable in lightweight environments so a user can inspect and approve a
    # complete plan before installing/activating the live-search runtime.
    dspy_module, _, _ = require_mipro_v2()
    optimizer_versions = validate_mipro_optimizer_runtime()
    execution_runtime_identity = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "dspy_version": optimizer_versions["dspy"],
        "dspy_source_path": str(Path(dspy_module.__file__).resolve()),
        "openai_version": optimizer_versions["openai"],
        "httpx_version": importlib.metadata.version("httpx"),
        "certifi_version": importlib.metadata.version("certifi"),
        "mipro_proposer_transport": "project-HttpProvider-requests-v1",
    }

    if DATASET == HEALTHFC_DATASET and cohort.get("api_transfer_approved") is not True:
        raise PermissionError(
            "HealthFC is licensed CC BY-NC-ND 4.0 and cloud API transfer has not been approved; "
            "refusing to send claim/evidence text to a provider")

    require_launcher_credentials(cfg)

    # The HTTP cache key excludes execution seed. With temperature=0, caching
    # would turn repeated confirmation labels into copied responses.
    os.environ["RSI_HTTP_NO_CACHE"] = "1"
    configured_429_backoff = float(cfg["runtime"].get(
        "pi_provider_429_backoff_base_seconds", 0))
    os.environ["RSI_PROVIDER_TRANSPORT_BACKOFF_BASE_SECONDS"] = str(
        float(cfg["runtime"].get(
            "http_provider_transport_backoff_base_seconds", 0)))
    os.environ["RSI_PROVIDER_429_BACKOFF_BASE_SECONDS"] = str(
        args.resume_429_backoff_base_seconds
        if args.resume_study_id and args.resume_429_backoff_base_seconds is not None
        else (resume_backoff if args.resume_study_id else configured_429_backoff))

    study_id = plan["plan_sha256"][:16]
    study_dir = artifacts_path("probe", STUDY_NAMESPACE, study_id)
    study_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = study_dir / "request-ledger.jsonl"
    extension_path = study_dir / "budget-extension.json"
    base_logical = int(plan["maximum_configured_model_calls"])
    base_physical = int(plan["maximum_physical_http_requests"])
    if args.extend_logical_budget is not None:
        if (args.extend_logical_budget < base_logical
                or args.extend_physical_budget < base_physical):
            raise ValueError("budget extension cannot reduce the frozen plan ceiling")
        validate_physical_budget_extension(plan, args.extend_physical_budget)
        extension = {
            "plan_sha256": plan["plan_sha256"],
            "base_logical_ceiling": base_logical,
            "base_physical_ceiling": base_physical,
            "logical_ceiling": args.extend_logical_budget,
            "physical_ceiling": args.extend_physical_budget,
            "reason": "account for bounded resume overhead after provider 429 failures",
        }
        if extension_path.exists() and read_json(extension_path) != extension:
            raise ValueError("an incompatible request-budget extension already exists")
        write_json(extension_path, extension)
    elif extension_path.exists():
        extension = read_json(extension_path)
        if (extension.get("plan_sha256") != plan["plan_sha256"]
                or int(extension.get("base_logical_ceiling", 0)) != base_logical
                or int(extension.get("base_physical_ceiling", 0)) != base_physical):
            raise ValueError("saved request-budget extension belongs to another plan")
        validate_physical_budget_extension(
            plan, int(extension.get("physical_ceiling", 0)))
    else:
        extension = {
            "plan_sha256": plan["plan_sha256"],
            "base_logical_ceiling": base_logical,
            "base_physical_ceiling": base_physical,
            "logical_ceiling": base_logical,
            "physical_ceiling": base_physical,
            "reason": "frozen plan ceiling",
        }
    report_path = study_dir / "report.json"
    plan_path = study_dir / "plan.json"
    write_json(plan_path, plan)
    seeds = list(args.execution_labels)
    if resume_report is not None:
        baseline_checkpoint = Path(resume_report["internal_baseline_checkpoint"])
        expected_checkpoint, _ = _baseline_checkpoint_for(baseline_identity, create=False)
        if baseline_checkpoint.resolve() != expected_checkpoint.resolve():
            raise ValueError("resume baseline checkpoint does not match frozen cache identity")
        if not baseline_checkpoint.exists():
            raise ValueError("frozen internal baseline checkpoint is missing")
        if (resume_report.get("status") == "internal_confirmation_failed"
                and len(read_jsonl(baseline_checkpoint)) != len(internal_ids) * len(seeds)):
            raise ValueError("frozen internal baseline checkpoint is incomplete")
        baseline_fingerprint = str(plan["baseline_cache_fingerprint"])
    else:
        baseline_checkpoint, baseline_fingerprint = _baseline_checkpoint_for(
            baseline_identity, create=True)
    route_cache_scope = {
        "baseline_cache_fingerprint": baseline_fingerprint,
        "search_task_ids": sorted(search_ids),
        "internal_task_ids": sorted(internal_ids),
        "policy": "freeze_first_model_route_per_task_v1",
    }
    route_cache_scope_sha256 = sha256_obj(route_cache_scope)
    route_cache_path = artifacts_path(
        "probe", "mipro_skill_routes", route_cache_scope_sha256 + ".json")
    frozen_skill_routes = FrozenSkillRouteCache(
        path=route_cache_path, scope_sha256=route_cache_scope_sha256)
    candidate_checkpoint = study_dir / "internal-candidate.jsonl"
    compile_checkpoint = study_dir / "compiled-candidate.json"
    trial_checkpoint = study_dir / "completed-mipro-trials.jsonl"
    legacy_backfill = _backfill_legacy_mipro_ledger(
        ledger_path=ledger_path, study_dir=study_dir, plan=plan,
        baseline_checkpoint=baseline_checkpoint,
        compile_checkpoint=compile_checkpoint,
        candidate_checkpoint=candidate_checkpoint,
        internal_ids=internal_ids,
    )
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(plan["plan_sha256"])
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path.resolve())
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(extension["logical_ceiling"])
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(extension["physical_ceiling"])

    def persist_completed_trial(row: dict[str, Any]) -> None:
        entry = {
            **row,
            "study_id": study_id,
            "plan_sha256": plan["plan_sha256"],
        }
        with trial_checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    resume_runtime_override = None
    if args.resume_study_id:
        resume_runtime_override = build_resume_runtime_override(plan, resume_backoff)
        override_path = study_dir / "resume-runtime-overrides.json"
        if override_path.exists() and read_json(override_path) != resume_runtime_override:
            raise ValueError("resume runtime override differs from the saved override")
        write_json(override_path, resume_runtime_override)

    def with_request_accounting(report: dict[str, Any]) -> dict[str, Any]:
        report["request_accounting"] = summarize_ledger(ledger_path)
        report["execution_runtime_identity"] = execution_runtime_identity
        report["frozen_skill_route_cache"] = {
            "path": str(route_cache_path),
            "scope_sha256": route_cache_scope_sha256,
            "route_count": len(frozen_skill_routes),
            "policy": "freeze_first_model_route_per_task_v1",
        }
        report["request_budget"] = {
            **extension,
            "legacy_backfill": legacy_backfill,
            "resume_runtime_overrides": resume_runtime_override,
        }
        return report

    # Incumbent and every MIPRO proposal must pass the same label/static/measured
    # factual-assertion and task-contract gates before benchmark scoring.
    def reviewer(prompt):
        with call_component("grounding"):
            if task_contract is None:
                return run_prompt_grounding_review(prompt, cfg, "system_prompt")
            return run_prompt_grounding_review(
                prompt, cfg, "system_prompt", task_contract=task_contract)
    candidate_validator = make_task_input_prompt_validator(
        candidate_gate_examples, prompt_cfg, state, reviewer,
        grounding_call_budget=args.grounding_call_budget,
    )
    if resume_report is not None:
        incumbent_gate = resume_report.get("incumbent_gate") or {}
        if incumbent_gate.get("passed") is not True:
            raise ValueError("frozen resume report lacks a passing incumbent gate")
        candidate_validator.records[:] = resume_report.get("candidate_instruction_gates") or []
    else:
        try:
            incumbent_gate = candidate_validator(state.system_prompt)
        except Exception as exc:
            incumbent_gate = {
                "passed": False, "reason": "incumbent_gate_error",
                "error_type": type(exc).__name__, "error": str(exc)[:1200],
            }
    if not incumbent_gate.get("passed"):
        failure = {
            "protocol": plan["protocol"], "plan": plan,
            "status": "incumbent_gate_failed", "study_id": study_id,
            "incumbent_gate": incumbent_gate,
            "candidate_instruction_gates": candidate_validator.records,
            "grounding_model_calls": candidate_validator.grounding_calls(),
            "task_evaluations": 0,
            "selection_gate_read": False, "official_test_read": False,
            "retrieval_used": False,
        }
        report_path = study_dir / "report.json"
        write_json(report_path, with_request_accounting(failure))
        print(json.dumps({"report": str(report_path), "status": failure["status"],
                          "incumbent_gate_reason": incumbent_gate.get("reason")},
                         ensure_ascii=False))
        return 2

    program_cfg = copy.deepcopy(cfg)
    program_cfg["evolution"]["allowed_artifacts"] = ["task_program"]
    program_gate_records: list[dict[str, Any]] = []

    # Measure and freeze the internal baseline only after the incumbent passes
    # the same gates required of candidates.
    phase_path = study_dir / "phase-checkpoint.json"
    if resume_report is None or resume_report.get("status") != "search_completed":
        persist_phase_checkpoint(phase_path, plan, "baseline_in_progress",
            baseline_checkpoint, incumbent_gate, candidate_validator.records)
    try:
        baseline_rows = _score_repeated_scope(
            cfg, state, internal, seeds, baseline_checkpoint,
            frozen_skill_routes=frozen_skill_routes)
    except Exception as exc:
        persist_phase_checkpoint(phase_path, plan, "baseline_failed",
            baseline_checkpoint, incumbent_gate, candidate_validator.records)
        failure = {
            "protocol": plan["protocol"], "plan": plan,
            "status": "baseline_failed", "study_id": study_id,
            "error_type": type(exc).__name__, "error": str(exc)[:1200],
            **_provider_failure_metadata(exc),
            "internal_baseline_checkpoint": str(baseline_checkpoint),
            "incumbent_gate": incumbent_gate,
            "candidate_instruction_gates": candidate_validator.records,
            "grounding_model_calls": candidate_validator.grounding_calls(),
            "selection_gate_read": False, "official_test_read": False,
            "retrieval_used": False,
        }
        report_path = study_dir / "report.json"
        write_json(report_path, with_request_accounting(failure))
        print(json.dumps({"report": str(report_path), "status": failure["status"],
                          "error_type": failure["error_type"]}, ensure_ascii=False))
        return 2
    if maximum_router_calls and not internal_ids <= set(
            frozen_skill_routes.snapshot()["routes"]):
        missing_routes = sorted(
            internal_ids - set(frozen_skill_routes.snapshot()["routes"]))
        raise RuntimeError(
            "internal baseline is missing its paired frozen Skill route records: "
            + ", ".join(missing_routes[:5]))
    baseline_metrics = {str(seed): metrics(rows) for seed, rows in baseline_rows.items()}

    def validate_program(program):
        artifact = extract_task_program(program, train_ids)
        prompt = str(artifact["system_prompt"])
        passed_prompt = any(
            record.get("passed") is True and record.get("content_sha256") == content_sha256(prompt)
            for record in candidate_validator.records
        )
        if not passed_prompt:
            result = {"passed": False, "reason": "program_prompt_not_grounded_before_scoring"}
            program_gate_records.append(result)
            return result
        from src.evolution.task_program import (changed_task_program_components,
                                                serialize_task_program)
        content = serialize_task_program(prompt, artifact["task_demos"])
        changed_components = changed_task_program_components(content, state)
        if changed_components != ["system_prompt"]:
            result = {
                "passed": False,
                "reason": "fr201_candidate_category_not_system_prompt",
                "changed_components": changed_components,
                "prompt_grounding_reviewed": True,
            }
            program_gate_records.append(result)
            return result
        demo_gate = validate_task_demo_rendering(artifact["task_demos"], program_cfg)
        if not demo_gate.get("passed"):
            result = {"passed": False,
                      "reason": "task_demo_rendering_gate_rejected",
                      "task_demo_rendering_gate": demo_gate}
            program_gate_records.append(result)
            return result
        candidate = Candidate(
            candidate_id=content_id("cand", state.release_id, "task_program", content),
            parent_release_id=state.release_id, artifact_type="task_program",
            artifact_id="task_program", diff="MIPROv2 prompt × labeled train demos",
            new_content=content, content_sha256=content_sha256(content),
        )
        gate = static_gate(candidate, program_cfg, state)
        record = {"passed": bool(gate.get("passed")), "static_gate": gate,
                  "reason": ("all_pre_evaluation_gates_passed"
                             if gate.get("passed") is True else "static_gate_rejected"),
                  "task_demo_rendering_gate": demo_gate,
                  "candidate_id": candidate.candidate_id,
                  "content_sha256": candidate.content_sha256,
                  "prompt_grounding_reviewed": True}
        program_gate_records.append(record)
        return record

    persist_phase_checkpoint(phase_path, plan, "search_in_progress",
        baseline_checkpoint, incumbent_gate, candidate_validator.records)
    if compile_checkpoint.exists():
        cached = read_json(compile_checkpoint)
        if cached.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("compiled MIPRO candidate belongs to a different protocol plan")
        mipro = cached.get("mipro") or {}
        mipro["artifact"] = cached.get("artifact")
        candidate_validator.records[:] = cached.get("candidate_instruction_gates") or []
        program_gate_records = cached.get("candidate_program_gates") or []
    else:
        try:
            mipro = compile_mipro_task_input(
                cfg=cfg, state=state, train_tasks=train,
                validation_tasks=search_validation,
                execution_tasks=candidate_gate_examples,
                train_ids=train_ids, validation_ids=search_ids,
                candidate_validator=candidate_validator,
                program_validator=validate_program,
                execution_seed=args.seed,
                task_evaluation_budget=args.mipro_task_evaluation_budget,
                num_candidates=args.num_candidates, num_trials=args.num_trials,
                minimum_generated_instructions_evaluated=(
                    args.minimum_generated_instructions_evaluated),
                max_labeled_demos=args.max_labeled_demos,
                balanced_demo_labels=LABELS,
                require_balanced_demo_bundle=args.require_balanced_demo_bundle,
                trial_checkpoint_callback=persist_completed_trial,
                frozen_skill_routes=frozen_skill_routes,
                task_contract_context=task_contract_context,
                task_contract=runtime_contract,
                prompt_only=True,
            )
        except Exception as exc:
            failure = {
                "protocol": plan["protocol"], "plan": plan,
                "status": "candidate_search_failed", "study_id": study_id,
                "error_type": type(exc).__name__, "error": str(exc)[:1200],
                "proposal_task_contract": proposal_context_failure_report(
                    proposal_task_contract, exc),
                **_provider_failure_metadata(exc),
                "task_evaluation_failure": getattr(exc, "failure_detail", None),
                "incumbent_gate": incumbent_gate,
                "candidate_instruction_gates": candidate_validator.records,
                "instruction_proposal_audit": getattr(
                    exc, "rsi_instruction_proposal_audit", []),
                "candidate_program_gates": program_gate_records,
                "grounding_model_calls": candidate_validator.grounding_calls(),
                "internal_baseline_metrics_by_execution_label": baseline_metrics,
                "compile_checkpoint": str(compile_checkpoint),
                "completed_mipro_trials_checkpoint": str(trial_checkpoint),
                "completed_mipro_trial_checkpoint_rows": (
                    len(read_jsonl(trial_checkpoint)) if trial_checkpoint.exists() else 0),
                "internal_baseline_checkpoint": str(baseline_checkpoint),
                "selection_gate_read": False, "official_test_read": False,
                "retrieval_used": False,
            }
            report_path = study_dir / "report.json"
            write_json(report_path, with_request_accounting(failure))
            print(json.dumps({"report": str(report_path), "status": failure["status"],
                              "error_type": failure["error_type"]}, ensure_ascii=False))
            return 2
        write_json(compile_checkpoint, {
            "plan_sha256": plan["plan_sha256"],
            "candidate_instruction_gates": candidate_validator.records,
            "candidate_program_gates": program_gate_records,
            "mipro": {key: value for key, value in mipro.items() if key != "artifact"},
            "artifact": mipro.get("artifact"),
        })

    persist_phase_checkpoint(phase_path, plan, "search_completed",
        baseline_checkpoint, incumbent_gate, candidate_validator.records)
    artifact = mipro.get("artifact")
    candidate_id = None
    candidate_state = None
    candidate_metrics = None
    paired_analysis = None
    candidate_status = str(mipro.get("candidate_status") or "no_candidate")
    coverage = mipro.get("instruction_search_coverage") or {}
    # The evaluator/optimizer logs hash the mutable DSPy signature instruction.
    # The extracted artifact may additionally contain the composed immutable
    # task contract, so comparing its runtime-content hash to those trial hashes
    # would mix identity scopes and falsely reject a fully evaluated candidate.
    candidate_prompt_was_evaluated = bool(
        coverage.get("selected_instruction_was_full_evaluated"))
    candidate_search_artifact = artifact
    if artifact:
        from src.evolution.task_program import (changed_task_program_components,
                                                serialize_task_program)
        selected_program_content = serialize_task_program(
            artifact["system_prompt"], artifact["task_demos"])
        selected_components = changed_task_program_components(
            selected_program_content, state)
        if selected_components != ["system_prompt"]:
            # Fail closed even if a stale or malformed optimizer checkpoint
            # bypassed the per-trial program validator.
            artifact = None
            candidate_status = "fr201_nonpromotable_component"
    if artifact and coverage.get("passed") is not True:
        # Coverage is an acceptance gate, not just report metadata. Do not
        # spend internal candidate-evaluation budget when the configured
        # minimum of grounded novel instructions did not reach full search
        # validation. Preserve the selected search artifact for diagnosis.
        artifact = None
        candidate_status = "instruction_search_coverage_failed"
    elif artifact and not candidate_prompt_was_evaluated:
        artifact = None
        candidate_status = "selected_prompt_not_full_evaluated"
    if artifact:
        content = serialize_task_program(artifact["system_prompt"], artifact["task_demos"])
        candidate_id = content_id("cand", state.release_id, "task_program", content)
        final_prompt_gate = next((record for record in candidate_validator.records
                                  if record.get("content_sha256") == content_sha256(artifact["system_prompt"])
                                  and record.get("passed") is True), None)
        if final_prompt_gate is None:
            raise RuntimeError("selected EI prompt lacks a passing pre-evaluation label/static/grounding record")
        candidate = Candidate(
            candidate_id=candidate_id, parent_release_id=state.release_id,
            artifact_type="task_program", artifact_id="task_program",
            diff="MIPROv2 selected prompt-only task instruction",
            new_content=content, content_sha256=content_sha256(content),
        )
        final_gate = static_gate(candidate, program_cfg, state)
        if not final_gate.get("passed"):
            raise RuntimeError("selected EI program failed final static gate")
        candidate_state = ArtifactState(
            release_id=state.release_id, system_prompt=artifact["system_prompt"],
            registry=state.registry, task_demos=artifact["task_demos"],
        )
        try:
            candidate_rows = _score_repeated_scope(
                cfg, candidate_state, internal, seeds, candidate_checkpoint,
                candidate_id=candidate_id,
                frozen_skill_routes=frozen_skill_routes,
            )
        except Exception as exc:
            failure = {
                "protocol": plan["protocol"], "plan": plan,
                "status": "internal_confirmation_failed", "study_id": study_id,
                "error_type": type(exc).__name__, "error": str(exc)[:1200],
                **_provider_failure_metadata(exc),
                "candidate_id": candidate_id,
                "incumbent_gate": incumbent_gate,
                "internal_baseline_metrics_by_execution_label": baseline_metrics,
                "candidate_instruction_gates": candidate_validator.records,
                "candidate_program_gates": program_gate_records,
                "grounding_model_calls": candidate_validator.grounding_calls(),
                "compile_checkpoint": str(compile_checkpoint),
                "internal_baseline_checkpoint": str(baseline_checkpoint),
                "internal_candidate_checkpoint": str(candidate_checkpoint),
                "selection_gate_read": False, "official_test_read": False,
                "retrieval_used": False,
            }
            report_path = study_dir / "report.json"
            write_json(report_path, with_request_accounting(failure))
            print(json.dumps({"report": str(report_path), "status": failure["status"],
                              "error_type": failure["error_type"]}, ensure_ascii=False))
            return 2
        candidate_metrics = {str(seed): metrics(rows) for seed, rows in candidate_rows.items()}
        paired_analysis = paired_internal_analysis(baseline_rows, candidate_rows, seeds)
        accepted = all(class_aware_gain(candidate_metrics[str(seed)], baseline_metrics[str(seed)])
                       for seed in seeds)
        candidate_status = "internal_gate_passed" if accepted else "internal_gate_rejected"

    report = {
        "protocol": plan["protocol"], "plan": plan,
        "status": "completed", "study_id": study_id,
        "plan_artifact": str(plan_path),
        "compile_checkpoint": str(compile_checkpoint),
        "completed_mipro_trials_checkpoint": str(trial_checkpoint),
        "completed_mipro_trial_checkpoint_rows": (
            len(read_jsonl(trial_checkpoint)) if trial_checkpoint.exists() else 0),
        "internal_baseline_checkpoint": str(baseline_checkpoint),
        "internal_candidate_checkpoint": str(candidate_checkpoint),
        "release_id": state.release_id, "cohort_path": str(args.cohort.resolve()),
        "candidate_id": candidate_id, "candidate_status": candidate_status,
        "candidate": artifact if candidate_id else None,
        "candidate_search_artifact": (
            candidate_search_artifact
            if candidate_status in {
                "fr201_nonpromotable_component",
                "instruction_search_coverage_failed",
                "selected_prompt_not_full_evaluated",
            } else None),
        "prompt_search_coverage_passed": coverage.get("passed"),
        "selected_candidate_prompt_was_full_evaluated": candidate_prompt_was_evaluated,
        "incumbent_gate": incumbent_gate,
        "baseline_internal_metrics_by_execution_label": baseline_metrics,
        "candidate_internal_metrics_by_execution_label": candidate_metrics,
        "internal_paired_analysis": paired_analysis,
        "candidate_instruction_gates": candidate_validator.records,
        "candidate_program_gates": program_gate_records,
        "grounding_model_calls": candidate_validator.grounding_calls(),
        "mipro": {key: value for key, value in mipro.items() if key != "artifact"},
        "selection_gate_read": False,
        "official_validation_read": False,
        "official_test_read": False,
        "source_upstream_test_materialized": DATASET == HEALTHVER_DATASET,
        "source_test_is_blind_confirmation": False if DATASET in {HEALTHVER_DATASET, SCIFACT_DATASET, HEALTHFC_DATASET, CLINIFACT_DATASET} else None,
        "exploratory_only": bool(plan["exploratory_only"]),
        "retrieval_used": False,
        "confirmation_eligible": False,
        "eligible_for_official_selection_gate": (
            DATASET == EI_DATASET and not plan["exploratory_only"]
            and candidate_status == "internal_gate_passed"),
        "selection_gate_eligibility_reason": selection_gate_eligibility_reason(
            DATASET, bool(plan["exploratory_only"]), candidate_status),
        "execution_labels_are_independent_rng_seeds": False,
    }
    report_path = study_dir / "report.json"
    write_json(report_path, with_request_accounting(report))
    print(json.dumps({"report": str(report_path), "candidate_status": candidate_status,
                      "candidate_id": candidate_id,
                      "eligible_for_official_selection_gate": report["eligible_for_official_selection_gate"],
                      "selection_gate_eligibility_reason": report[
                          "selection_gate_eligibility_reason"],
                      "official_validation_test_read": (
                          False if DATASET == EI_DATASET else None),
                      "official_validation_read": False,
                      "official_test_read": False,
                      "source_upstream_test_materialized": DATASET == HEALTHVER_DATASET,
                      "source_test_is_blind_confirmation": (
                          False if DATASET in {HEALTHVER_DATASET, SCIFACT_DATASET, HEALTHFC_DATASET, CLINIFACT_DATASET} else None)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
