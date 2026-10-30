"""实验配置加载与解析（Spec §9）。

- 加载 YAML 后必须把 ${ENV_VAR} 全部解析为真实值，运行记录中不得残留占位符
- 解析后的完整配置写入 manifest / run 产物
"""

from __future__ import annotations

import os
import re
import hashlib
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .util import path, read_json, sha256_obj

DEFAULT_CONFIG_PATH = path("configs", "experiment.yaml")
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
TASK_INPUT_PROTOCOL_SOURCES = (
    "src/config.py",
    "src/eval/assessment_policy.py",
    "src/eval/harness.py", "src/eval/final_eval.py", "src/eval/metrics.py",
    "src/data/freeze.py", "src/data/split.py", "src/governance/release.py",
    "src/util.py", "src/evolution/campaign.py",
    "src/evolution/candidate_memory.py",
    "src/runtime/provider.py", "src/runtime/runner.py",
    "src/runtime/trace.py", "scripts/audit_task_execution.py",
    "src/runtime/skill_routes.py",
    "src/runtime/skills.py", "src/runtime/skill_bundle.py",
    "src/data/research_profiles.py",
    "src/data/research_cohort.py",
    # Retrieval scoring is part of the evaluated behavior. Keep both the
    # fixture reranker and the live PubMed snapshot ranker in task identities.
    "src/retrieval/gateway.py", "src/retrieval/pubmed_snapshot.py",
    "src/governance/skill_migration.py", "src/evolution/skill_curation.py",
    "src/governance/review.py", "src/evolution/apply.py",
    "scripts/research_assistant.py", "src/runtime/research_assistant.py",
    "src/runtime/research_tools.py", "src/retrieval/literature.py",
    "src/runtime/docker_worker.py", "src/runtime/worker_contract.py",
    "tools/docker-worker/worker.py", "tools/docker-worker/Dockerfile", "scripts/deploy_docker_worker.py",
    "src/eval/open_benchmark.py", "src/eval/open_research.py",
    "src/eval/annotation_policy.py",
    "src/eval/ai_jobs.py", "src/eval/ai_annotation.py", "src/eval/ai_grounding_reference.py",
    "src/eval/ai_instruction_calibration.py", "src/eval/instruction_review_calibration.py",
    "src/eval/native_instruction_reference.py", "scripts/ai_instruction_review_native.py",
    "scripts/ai_instruction_review.py",
    "src/eval/grounding_calibration.py", "src/eval/grounding_v14.py",
    "src/eval/grounding_v15.py",
    "src/eval/grounding_v16.py",
    "src/eval/healthver_reference.py", "src/eval/healthver_calibration_cohort.py",
    "scripts/prepare_healthver_grounding_recalibration.py", "scripts/ai_grounding.py",
    "scripts/run_clinifact_candidate_confirmation.py",
    "scripts/run_clinifact_candidate_bundle_confirmation.py",
    "scripts/run_clinifact_g03_ablation.py",
    "scripts/run_g09_recall_semantic_review.py",
    "scripts/complete_g09_adjudications_with_provider.py",
    "src/eval/selected_answer_calibration_runner.py",
    "scripts/score_task_input_grounding_ai_reference.py",
    "scripts/build_healthver_source_reference.py",
    "scripts/audit_healthver_source_reference_calibration.py",
    "scripts/audit_healthver_ai_reference_triplet.py",
    "requirements-optimizers.constraints.txt",
    "src/runtime/tool_policy.py",
    "src/runtime/call_ledger.py",
    "src/runtime/input_evidence.py", "src/evolution/round.py",
    "src/evolution/attribution.py",
    "src/evolution/attribution_review.py",
    "src/evolution/counterfactual.py", "src/evolution/reflection.py",
    "src/evolution/crystallize.py",
    "src/evolution/gepa_search.py", "src/evolution/gepa_adapter.py",
    "src/evolution/prompt_rewrite.py",
    "src/evolution/gepa_skill_search.py", "src/eval/grounding.py",
    "src/evolution/fewshot.py", "src/evolution/task_demo_search.py",
    "src/evolution/task_program.py",
    "src/evolution/mipro_task_input.py",
    "src/evolution/task_input_gates.py",
    "src/evolution/task_contracts.py",
    "scripts/smoke_offline.py",
    "scripts/probe_fixed_demos_unseen_dev.py",
    "scripts/probe_frozen_gepa_prompt_unseen_dev.py",
    "scripts/probe_gepa_budget_invalidated_candidate.py",
    "scripts/probe_mipro_demo_effect.py",
    "scripts/probe_pubmed_reranker.py",
    "scripts/probe_pi_abstain_policy.py",
    "scripts/probe_pi_decision_rule.py",
    "scripts/probe_pi_decision_rule_arms.py",
    "scripts/probe_pi_rule_gate.py",
    "scripts/probe_pi_rule_stability.py",
    "scripts/probe_pi_task_demos.py",
    "scripts/probe_pi_task_input.py",
    "scripts/probe_pubmedqa_threeway_demos.py",
    "scripts/probe_task_demo_bundle_comparison.py",
    "scripts/probe_task_input_skill_counterfactual.py",
    "scripts/probe_task_input_skill_routing.py",
    "scripts/run_real_candidate_factorial.py",
    "src/evolution/instruction_review.py",
    "src/eval/instruction_review_audit.py",
    "src/evolution/method_rule_gate.py",
    "src/data/evidence_inference_v2.py",
    "src/data/evidence_inference_cohort.py",
    "src/data/healthver_task_input.py",
    "scripts/prepare_healthver_task_input.py",
    "scripts/prepare_healthver_mipro_cohort.py",
    "configs/task_input/system_prompt.healthver-v1.txt",
    "src/data/healthfc_task_input.py",
    "src/data/healthfc_cohort.py",
    "src/data/healthfc_history_scan.py",
    "src/data/healthfc_source_audit.py",
    "scripts/prepare_healthfc_exploratory_cohort.py",
    "scripts/scan_healthfc_history.py",
    "scripts/audit_healthfc_source.py",
    "configs/task_input/system_prompt.healthfc-v1.txt",
    "src/data/clinifact_task_input.py",
    "src/data/clinifact_cohort.py",
    "src/data/clinifact_history_scan.py",
    "src/data/clinifact_source_audit.py",
    "data/manifests/clinifact_figshare_license.json",
    "data/manifests/clinifact_figshare_article_27188109.json",
    "scripts/prepare_clinifact_mipro_cohort.py",
    "scripts/scan_clinifact_history.py",
    "scripts/audit_clinifact_source.py",
    "configs/prompts/system_prompt.clinifact.v0.txt",
    "configs/prompts/system_prompt.clinifact.v1.txt",
    "src/data/scifact_task_input.py",
    "src/data/scifact_cohort.py",
    "scripts/prepare_scifact_mipro_cohort.py",
    "configs/prompts/system_prompt.scifact.v0.txt",
    "src/data/healthver_factorial_exploration.py",
    "src/data/split.py",
    "src/eval/task_rows.py",
    "src/eval/run_manifest.py",
    "src/eval/options.py", "src/eval/harness.py", "src/eval/metrics.py", "src/eval/final_eval.py",
    "src/eval/gate_verdict.py", "src/governance/promotion.py",
    "src/governance/policy.py", "src/governance/review.py",
    "src/evolution/apply.py", "src/governance/release.py",
    "src/contracts/types.py",
    # Research experiment entry points are protocol code too: changing how a
    # cohort is split, candidates are gated, or discovery rows are paired must
    # never silently reuse a prior experiment fingerprint.
    "scripts/pubmedqa_threeway_protocol.py",
    "scripts/prepare_pubmedqa_v2_cohort.py",
    "scripts/run_pubmedqa_threeway_class_aware_search.py",
    "scripts/run_pubmedqa_v2_discovery_baseline.py",
    "scripts/run_pubmedqa_train_only_gepa.py",
    "scripts/run_pubmedqa_train_only_skill_gepa.py",
    "scripts/run_pubmedqa_v2_joint_search.py",
    "scripts/run_pubmedqa_v2_mipro_search.py",
    "scripts/evaluate_pubmedqa_prompt_discovery.py",
    "scripts/diagnose_pubmedqa_label_compatible_prompt.py",
    # Keep every direct Python scoring entrypoint in source-bound task-input
    # identities, including historical diagnostics that cannot select candidates.
    "scripts/pilot_bioasq_protocol.py",
    "scripts/pilot_model_runtime_control.py",
    "scripts/pilot_pubmed_evidence_ledger.py",
    "scripts/pilot_pubmed_protocol_pair.py",
    "scripts/pilot_pubmed_reranked_answer.py",
    "scripts/pilot_pubmed_snapshot.py",
    "scripts/pilot_pubmed_topk.py",
    "scripts/pilot_pubmed_window_grounding.py",
    "scripts/probe_chunking_real.py",
    "scripts/probe_context_truncation.py",
    "scripts/probe_cost_gate.py",
    "scripts/probe_doc_id_leak.py",
    "scripts/real_probe.py",
    "scripts/rerun_baseline.py",
    "scripts/rerun_baseline_protocol_change.py",
    "scripts/test_error_strategies.py",
    "src/eval/ablation.py",
    "src/eval/multiseed.py",
    "src/eval/controlled_experiment.py",
    # Legacy proposal/evaluation pilots are hard-disabled, but remain pinned
    # so re-enabling or changing their behavior invalidates task-input plans.
    "scripts/pilot_rewrite_iterative.py",
    "scripts/pilot_rewrite_fewshot.py",
    "scripts/confirm_frontier_candidate.py",
    "scripts/a1_reselect.py",
    "scripts/a1_pairwise.py",
    "scripts/verify_rewrite_holdout.py",
    "scripts/replay_gepa_challenger.py",
    "scripts/prepare_evidence_inference_task_input.py",
    "scripts/prepare_evidence_inference_nested_cohort.py",
    "src/evolution/evidence_inference_confirmation.py",
    "scripts/run_evidence_inference_mipro_search.py",
    "src/evolution/evidence_inference_promotion.py",
    "scripts/evaluate_evidence_inference_selection_gate.py",
    "scripts/prepare_healthver_task_input.py",
    "scripts/prepare_healthver_mipro_cohort.py",
    "scripts/prepare_healthver_fresh_component_cohort.py",
    "scripts/prepare_healthver_factorial_exploration.py",
    "scripts/prepare_healthver_support_regression_probe.py",
    "scripts/run_healthver_fixed_candidate_probe.py",
    "scripts/audit_healthver_mipro_rationales.py",
    "scripts/audit_healthver_fixed_probe_rationales.py",
    "scripts/build_healthver_rationale_review_queue.py",
    "scripts/prepare_healthver_remaining_search_probe.py",
    "scripts/run_gepa_prompt_search.py",
    "scripts/run_gepa_skill_search.py",
    "scripts/run_gepa_task_input_skill_search.py",
    "scripts/run_evidence_inference_gepa_prompt_search.py",
    "scripts/run_evidence_inference_gepa_skill_search.py",
    "scripts/run_evidence_inference_gepa_joint_search.py",
    "scripts/run_evidence_inference_fixed_candidate_probe.py",
    # This entry point defines the real Pi-backed GEPA ablation arms, task
    # offsets, seed semantics and joint demo-search budget. Those are all
    # candidate/evaluation protocol decisions and must invalidate old runs.
    "scripts/run_task_input_factorial.py",
    # Historical release-factor ablation remains fingerprinted even though
    # the shared evaluator now rejects it without a frozen execution context.
    "scripts/real_factorial.py",
    "configs/task_input/system_prompt.evidence-inference-v1.txt",
    "configs/task_input/system_prompt.healthver-v1.txt",
    "scripts/rsi.py",
    "scripts/prepare_attribution_review.py",
    "requirements-mipro.txt", "UPSTREAM_LOCK.yml",
    "vendor/gepa-main/gepa-main/pyproject.toml",
    "vendor/gepa-main/gepa-main/uv.lock",
    "vendor/gepa-main/gepa-main/LICENSE",
    "vendor/dspy-main/dspy-main/dspy/teleprompt/mipro_optimizer_v2.py",
    "vendor/dspy-main/dspy-main/dspy/propose/grounded_proposer.py",
    "vendor/dspy-main/dspy-main/dspy/propose/dataset_summary_generator.py",
    "vendor/dspy-main/dspy-main/dspy/propose/utils.py",
    "vendor/dspy-main/dspy-main/dspy/propose/propose_base.py",
    "vendor/dspy-main/dspy-main/dspy/teleprompt/utils.py",
    "vendor/dspy-main/dspy-main/dspy/predict/predict.py",
    "vendor/dspy-main/dspy-main/dspy/signatures/signature.py",
)


class ConfigError(RuntimeError):
    pass


_PI_RUNTIME_SOURCE_ROOTS = {
    "pi-agent-core": "tools/pi-bridge/node_modules/@earendil-works/pi-agent-core",
    "pi-ai": "tools/pi-bridge/node_modules/@earendil-works/pi-ai",
}


def _source_tree_sha256(root: Path) -> str:
    """Hash a vendored source tree by normalized relative path and file bytes."""
    if not root.is_dir():
        raise ConfigError(f"pinned upstream source tree is missing: {root}")
    entries = []
    for source in sorted(root.rglob("*")):
        if (not source.is_file() or "__pycache__" in source.parts
                or source.suffix == ".pyc"):
            continue
        entries.append((source.relative_to(root).as_posix(),
                        hashlib.sha256(source.read_bytes()).hexdigest()))
    if not entries:
        raise ConfigError(f"pinned upstream source tree is empty: {root}")
    return sha256_obj(entries)


def _locked_upstream_source_trees(runtime_provider: str | None = None, *,
                                  runtime_only: bool = False) -> Dict[str, str]:
    """Verify common pins and the selected provider's actual installed trees."""
    lock_path = path("UPSTREAM_LOCK.yml")
    try:
        lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read upstream source lock: {type(exc).__name__}") from exc
    upstreams = lock.get("upstreams") if isinstance(lock, dict) else None
    if not isinstance(upstreams, list):
        raise ConfigError("UPSTREAM_LOCK.yml has no upstreams list")
    fingerprints = {}
    for entry in upstreams:
        if not isinstance(entry, dict) or not entry.get("source_root"):
            continue
        declared_provider = entry.get("runtime_provider")
        if declared_provider and declared_provider != runtime_provider:
            continue
        if runtime_only and not declared_provider:
            continue
        name = str(entry.get("name") or "")
        upstream_path = Path(str(entry.get("path") or ""))
        source_root = Path(str(entry["source_root"]))
        expected = str(entry.get("source_sha256") or "")
        if (not name or upstream_path.is_absolute() or source_root.is_absolute()
                or ".." in upstream_path.parts or ".." in source_root.parts
                or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            raise ConfigError("UPSTREAM_LOCK.yml has an invalid source-tree pin")
        if name in fingerprints:
            raise ConfigError("UPSTREAM_LOCK.yml has a duplicate source-tree pin")
        if declared_provider == "pi" and (
                name not in _PI_RUNTIME_SOURCE_ROOTS
                or (upstream_path / source_root).as_posix() != _PI_RUNTIME_SOURCE_ROOTS[name]):
            raise ConfigError("UPSTREAM_LOCK.yml Pi pin does not identify the bridge's installed package")
        root = path(str(upstream_path / source_root))
        actual = _source_tree_sha256(root)
        if actual != expected:
            raise ConfigError(
                f"pinned {name} source tree differs from UPSTREAM_LOCK.yml; "
                "review and explicitly update the vendored revision pin")
        fingerprints[name] = actual
    if runtime_provider == "pi" and not set(_PI_RUNTIME_SOURCE_ROOTS) <= fingerprints.keys():
        raise ConfigError("UPSTREAM_LOCK.yml is missing required Pi installed-package pins")
    return fingerprints


def verify_pi_runtime_identity() -> Dict[str, str]:
    """Fail before bridge dispatch if an installed SDK differs from its pin."""
    return _locked_upstream_source_trees("pi", runtime_only=True)


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    p = str(config_path or DEFAULT_CONFIG_PATH)
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping: {p}")
    resolved = resolve_placeholders(raw)
    validate_config(resolved, source=p)
    return resolved


_ROOT_KEYS = {
    "spec_version", "domain_profile", "models", "runtime", "tools",
    "retrieval", "skills", "evaluation", "evolution", "gates", "review",
    "execution", "privacy",
}
_SECTION_KEYS = {
    "models": {"task", "reflection", "judge"},
    "runtime": {"provider", "disable_pi_builtin_tools", "max_model_calls",
                "max_tool_calls", "fake_noise", "input_mode", "protocol_epoch",
                "task_input_max_chars", "pi_task_demos_max_block_chars",
                "pi_task_demo_max_example_chars", "pi_request_cooldown_seconds",
                "pi_provider_max_retries", "pi_timeout_max_attempts",
                "http_no_cache_required", "open_research_structured_parse_failures"},
    "tools": {"active", "raw_shell", "raw_network"},
    "retrieval": {"backend", "top_k", "corpus_version", "doc_id_scheme", "chunking"},
    "skills": {"release_id", "registry", "mutable_groups", "frozen_groups",
               "crystallized_group", "description_evolution_groups", "recall_min_score",
               "routing_mode", "router_provider", "router_model_id",
               "instruction_review_model_id"},
    "evaluation": {"manifests", "datasets", "transfer_datasets", "primary_final_datasets",
                   "final_holdout_access", "sealed_pairing", "seed", "seeds",
                   "require_split_freeze", "insufficient_option", "max_tasks_per_round",
                   "main_metric", "primary_axis", "primary_axis_tolerance_pp", "bootstrap",
                   "task_offsets", "task_offset", "max_task_evaluations", "semantic_citation",
                   "persist_selection_per_task", "annotation_conservative_repairs"},
    "evolution": {"allowed_artifacts", "max_candidates_per_round", "max_rounds",
                  "reflection_backend", "reflection_max_attempts", "behavior_channels",
                  "gepa", "crystallization", "fewshot_enabled", "fewshot_k", "proposer",
                  "mipro"},
    "gates": {"static", "accept"},
    "review": {"release", "final_holdout_unlock"},
    "execution": {"enabled", "max_expr_chars", "max_nodes", "max_output_chars",
                  "allow_latex", "wsl2", "docker_worker"},
    "privacy": {"mode", "redact_before_persist"},
}
_MODEL_KEYS = {"provider", "model_id", "base_url", "api_key_env", "temperature",
               "max_tokens", "enable_thinking", "timeout", "max_retries",
               "max_context_chars", "max_doc_chars", "structured_output",
               "send_run_seed"}
_NESTED_KEYS = {
    "retrieval.chunking": {"enabled", "budget_chars", "target_chars", "max_chars",
                           "max_chunks", "option_weight"},
    "evaluation.bootstrap": {"B", "rng_seed"},
    "evaluation.semantic_citation": {"enabled", "mode", "model_id", "cache", "max_doc_chars", "contrastive", "check_top3", "prompt_version", "calibration_attestations"},
    "evaluation.semantic_citation.calibration_attestation": {"dataset", "scope", "metrics", "report_path", "report_sha256", "packet_sha256", "task_contract", "runtime_scorer_sha256"},
    "evolution.behavior_channels": {"enabled", "owner", "enforce_in_evolution"},
    "evolution.behavior_channels.owner": {"prompt", "skill_body", "description"},
    "evolution.gepa": {"enabled", "budget", "max_metric_calls",
                        "max_candidate_proposals", "reflection_minibatch_size"},
    "evolution.crystallization": {"min_procedure_steps", "min_trigger_recall",
                                  "max_description_similarity", "max_cross_domain_trigger",
                                  "max_non_target_trigger", "cross_domain_negative_limit"},
    "gates.static": {"max_prompt_chars", "max_description_chars", "max_body_chars",
                     "min_procedure_steps", "max_procedure_steps", "forbidden_patterns",
                     "require_single_artifact", "max_task_demos_chars"},
    "gates.accept": {"require_static_gate", "main_metric_non_regression",
                     "require_human_review", "gate_min_delta_pp",
                     "gate_min_prob_delta_gt_0", "tolerance",
                     "require_classwise_non_regression", "require_strict_accuracy_gain",
                     "minimum"},
    "gates.accept.tolerance": {"cost_pct", "p95_latency_pct", "citation_precision_pp",
                               "unsupported_answer_rate_pp", "citation_support_rate_pp",
                               "answer_evidence_consistency_rate_pp", "required", "strict"},
    "execution.wsl2": {"enabled", "note"},
    "execution.docker_worker": {"enabled", "image", "context", "receipt_path", "receipt_sha256", "timeout_seconds", "memory_mb", "pids_limit"},
    "evolution.mipro": {"enabled", "max_proposer_calls", "num_candidates", "num_trials",
                        "max_labeled_demos"},
    "evolution.proposer": {"mode", "enabled"},
}


def _mapping(value: Any, where: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping")
    return value


def _check_keys(value: Dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"{where} contains unknown key(s): {', '.join(unknown)}")


def _required(value: Dict[str, Any], keys: set[str], where: str) -> None:
    missing = sorted(keys - set(value))
    if missing:
        raise ConfigError(f"{where} is missing required key(s): {', '.join(missing)}")


def _enum(value: Any, choices: set[str], where: str) -> None:
    if not isinstance(value, str) or value not in choices:
        raise ConfigError(f"{where} must be one of: {', '.join(sorted(choices))}")


def _boolean(value: Any, where: str) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be boolean")


def _string(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty string")


def _string_list(value: Any, where: str, *, allow_empty: bool = True) -> None:
    if (not isinstance(value, list) or (not allow_empty and not value)
            or any(not isinstance(item, str) or not item.strip() for item in value)):
        raise ConfigError(f"{where} must be a list of non-empty strings")


def _integer(value: Any, where: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{where} must be an integer >= {minimum}")


def _number(value: Any, where: str, *, minimum: float | None = None,
            maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(f"{where} must be a finite number")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{where} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{where} must be <= {maximum}")


def validate_config(cfg: Dict[str, Any], *, source: str = "config") -> None:
    """Validate the supported experiment-config shape before any runtime uses it.

    Unknown keys are rejected so a misspelling cannot silently fall back to a
    runtime default. Experimental runners may still derive new configs after
    loading the validated base file.
    """
    root = _mapping(cfg, source)
    _check_keys(root, _ROOT_KEYS, source)
    _required(root, _ROOT_KEYS, source)
    _string(root["spec_version"], f"{source}.spec_version")
    _string(root["domain_profile"], f"{source}.domain_profile")

    models = _mapping(root["models"], f"{source}.models")
    _check_keys(models, _SECTION_KEYS["models"], f"{source}.models")
    _required(models, {"task", "reflection", "judge"}, f"{source}.models")
    for role in ("task", "reflection", "judge"):
        model = _mapping(models[role], f"{source}.models.{role}")
        _check_keys(model, _MODEL_KEYS, f"{source}.models.{role}")
        _required(model, {"provider", "model_id"}, f"{source}.models.{role}")
        _enum(model["provider"], {"fake", "http", "pi"},
              f"{source}.models.{role}.provider")
        _string(model["model_id"], f"{source}.models.{role}.model_id")
        if "temperature" in model:
            _number(model["temperature"], f"{source}.models.{role}.temperature",
                    minimum=0, maximum=2)
        for key in ("max_tokens", "timeout", "max_context_chars", "max_doc_chars"):
            if key in model:
                _integer(model[key], f"{source}.models.{role}.{key}", minimum=1)
        if "max_retries" in model:
            _integer(model["max_retries"], f"{source}.models.{role}.max_retries")
        if "enable_thinking" in model:
            _boolean(model["enable_thinking"], f"{source}.models.{role}.enable_thinking")
        if "send_run_seed" in model:
            _boolean(model["send_run_seed"], f"{source}.models.{role}.send_run_seed")
        if "structured_output" in model:
            _enum(model["structured_output"], {"json_schema"},
                  f"{source}.models.{role}.structured_output")

    for section, keys in _SECTION_KEYS.items():
        if section == "models":
            continue
        value = _mapping(root[section], f"{source}.{section}")
        _check_keys(value, keys, f"{source}.{section}")
    runtime, tools = root["runtime"], root["tools"]
    _required(runtime, {"provider", "max_model_calls", "max_tool_calls"},
              f"{source}.runtime")
    _enum(runtime["provider"], {"fake", "http", "pi"}, f"{source}.runtime.provider")
    for key in ("max_model_calls", "max_tool_calls"):
        _integer(runtime[key], f"{source}.runtime.{key}", minimum=1)
    if "input_mode" in runtime and runtime["input_mode"] not in {"retrieval", "task_input"}:
        raise ConfigError(f"{source}.runtime.input_mode must be retrieval or task_input")
    if "protocol_epoch" in runtime:
        _enum(runtime["protocol_epoch"], {"public-task-input-v1"},
              f"{source}.runtime.protocol_epoch")
        if runtime.get("input_mode") != "task_input":
            raise ConfigError("public task-input protocol requires input_mode=task_input")
    for key in ("disable_pi_builtin_tools",):
        if key in runtime:
            _string_list(runtime[key], f"{source}.runtime.{key}")
    for key in ("task_input_max_chars", "pi_task_demos_max_block_chars",
                "pi_task_demo_max_example_chars", "pi_provider_max_retries",
                "pi_timeout_max_attempts"):
        if key in runtime:
            _integer(runtime[key], f"{source}.runtime.{key}", minimum=0 if key=="pi_provider_max_retries" else 1)
    if runtime.get("pi_provider_max_retries",0)>4:
        raise ConfigError("runtime.pi_provider_max_retries exceeds the supported upper bound 4")
    if runtime.get("pi_timeout_max_attempts",1)>3:
        raise ConfigError("runtime.pi_timeout_max_attempts exceeds the supported upper bound 3")
    if "pi_request_cooldown_seconds" in runtime:
        _number(runtime["pi_request_cooldown_seconds"],
                f"{source}.runtime.pi_request_cooldown_seconds", minimum=0)
    if "http_no_cache_required" in runtime:
        _boolean(runtime["http_no_cache_required"], f"{source}.runtime.http_no_cache_required")
    if "open_research_structured_parse_failures" in runtime:
        _boolean(runtime["open_research_structured_parse_failures"],
                 f"{source}.runtime.open_research_structured_parse_failures")
    if "fake_noise" in runtime:
        _number(runtime["fake_noise"], f"{source}.runtime.fake_noise", minimum=0)
    _required(tools, {"active", "raw_shell", "raw_network"}, f"{source}.tools")
    if "active" not in tools:
        raise ConfigError(f"{source}.tools.active is required")
    _string_list(tools["active"], f"{source}.tools.active")
    from .runtime.tool_policy import CONTROLLED_TOOLS
    unsupported_tools = sorted(set(tools["active"]) - CONTROLLED_TOOLS)
    if unsupported_tools:
        raise ConfigError(
            f"{source}.tools.active contains unsupported tools: {unsupported_tools}")
    for key in ("raw_shell", "raw_network"):
        _enum(tools[key], {"allow", "deny"}, f"{source}.tools.{key}")

    retrieval = root["retrieval"]
    _required(retrieval, {"backend", "top_k", "corpus_version"}, f"{source}.retrieval")
    for key in ("backend", "corpus_version"):
        _string(retrieval[key], f"{source}.retrieval.{key}")
    _integer(retrieval["top_k"], f"{source}.retrieval.top_k", minimum=1)
    if "doc_id_scheme" in retrieval:
        _string(retrieval["doc_id_scheme"], f"{source}.retrieval.doc_id_scheme")
    if "chunking" in retrieval:
        chunking = _mapping(retrieval["chunking"], f"{source}.retrieval.chunking")
        _check_keys(chunking, _NESTED_KEYS["retrieval.chunking"],
                    f"{source}.retrieval.chunking")
        if "enabled" in chunking:
            _boolean(chunking["enabled"], f"{source}.retrieval.chunking.enabled")
        for key in ("budget_chars", "target_chars", "max_chars", "max_chunks", "option_weight"):
            if key in chunking:
                _integer(chunking[key], f"{source}.retrieval.chunking.{key}", minimum=1)
    skills = root["skills"]
    _required(skills, {"release_id", "registry", "mutable_groups", "frozen_groups"},
              f"{source}.skills")
    for key in ("release_id", "registry"):
        _string(skills[key], f"{source}.skills.{key}")
    for key in ("mutable_groups", "frozen_groups", "description_evolution_groups"):
        if key in skills:
            _string_list(skills[key], f"{source}.skills.{key}")
    if "recall_min_score" in skills:
        _number(skills["recall_min_score"], f"{source}.skills.recall_min_score", minimum=0)
    for key in ("crystallized_group", "router_model_id",
                "instruction_review_model_id"):
        if key in skills:
            _string(skills[key], f"{source}.skills.{key}")
    if "routing_mode" in skills:
        _enum(skills["routing_mode"], {"lexical", "model"}, f"{source}.skills.routing_mode")
    if "router_provider" in skills:
        _enum(skills["router_provider"], {"fake", "http", "pi"},
              f"{source}.skills.router_provider")

    evaluation = root["evaluation"]
    _required(evaluation, {"datasets", "manifests", "seed", "seeds", "main_metric"},
              f"{source}.evaluation")
    for key in ("datasets", "manifests", "transfer_datasets", "primary_final_datasets"):
        if key in evaluation:
            _string_list(evaluation[key], f"{source}.evaluation.{key}")
    _integer(evaluation["seed"], f"{source}.evaluation.seed", minimum=0)
    if not isinstance(evaluation["seeds"], list) or not evaluation["seeds"]:
        raise ConfigError(f"{source}.evaluation.seeds must be a non-empty integer list")
    for i, seed in enumerate(evaluation["seeds"]):
        _integer(seed, f"{source}.evaluation.seeds[{i}]")
    _string(evaluation["main_metric"], f"{source}.evaluation.main_metric")
    if "semantic_citation" in evaluation:
        semantic = _mapping(evaluation["semantic_citation"], f"{source}.evaluation.semantic_citation")
        _check_keys(semantic, _NESTED_KEYS["evaluation.semantic_citation"], f"{source}.evaluation.semantic_citation")
        for key in ("enabled", "cache", "contrastive", "check_top3"):
            if key in semantic:
                _boolean(semantic[key], f"{source}.evaluation.semantic_citation.{key}")
        if "mode" in semantic:
            _enum(semantic["mode"], {"answer", "proposition"}, f"{source}.evaluation.semantic_citation.mode")
        if "prompt_version" in semantic:
            _enum(semantic["prompt_version"], {"v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"},
                  f"{source}.evaluation.semantic_citation.prompt_version")
        if "model_id" in semantic:
            _string(semantic["model_id"], f"{source}.evaluation.semantic_citation.model_id")
        if "max_doc_chars" in semantic:
            _integer(semantic["max_doc_chars"], f"{source}.evaluation.semantic_citation.max_doc_chars", minimum=1)
        if "calibration_attestations" in semantic:
            attestations = semantic["calibration_attestations"]
            if not isinstance(attestations, list):
                raise ConfigError(f"{source}.evaluation.semantic_citation.calibration_attestations must be a list")
            seen_datasets = set()
            for index, item in enumerate(attestations):
                where = f"{source}.evaluation.semantic_citation.calibration_attestations[{index}]"
                row = _mapping(item, where)
                _check_keys(row, _NESTED_KEYS["evaluation.semantic_citation.calibration_attestation"], where)
                _required(row, _NESTED_KEYS["evaluation.semantic_citation.calibration_attestation"], where)
                _string(row["dataset"], f"{where}.dataset")
                if row["dataset"] in seen_datasets:
                    raise ConfigError(f"{where}.dataset duplicates another calibration attestation")
                seen_datasets.add(row["dataset"])
                _enum(row["scope"], {"single_doc", "top3"}, f"{where}.scope")
                _string_list(row["metrics"], f"{where}.metrics", allow_empty=False)
                _string(row["report_path"], f"{where}.report_path")
                for digest in ("report_sha256", "packet_sha256", "runtime_scorer_sha256"):
                    if not isinstance(row[digest], str) or not re.fullmatch(r"[0-9a-f]{64}", row[digest]):
                        raise ConfigError(f"{where}.{digest} must be a lowercase SHA-256")
                _enum(row["task_contract"], {"selected_answer_support_v1"},
                      f"{where}.task_contract")
        if semantic.get("check_top3") and semantic.get("mode", "answer") != "proposition":
            raise ConfigError("semantic_citation.check_top3 requires proposition mode")
    for key in ("sealed_pairing", "require_split_freeze", "insufficient_option",
                "persist_selection_per_task", "annotation_conservative_repairs"):
        if key in evaluation:
            _boolean(evaluation[key], f"{source}.evaluation.{key}")
    if "max_tasks_per_round" in evaluation:
        _integer(evaluation["max_tasks_per_round"],
                 f"{source}.evaluation.max_tasks_per_round", minimum=1)
    if "bootstrap" in evaluation:
        bootstrap = _mapping(evaluation["bootstrap"], f"{source}.evaluation.bootstrap")
        _check_keys(bootstrap, _NESTED_KEYS["evaluation.bootstrap"],
                    f"{source}.evaluation.bootstrap")
        if "B" in bootstrap:
            _integer(bootstrap["B"], f"{source}.evaluation.bootstrap.B", minimum=1)
        if "rng_seed" in bootstrap:
            _integer(bootstrap["rng_seed"], f"{source}.evaluation.bootstrap.rng_seed")
    for key in ("task_offset", "max_task_evaluations"):
        if key in evaluation:
            _integer(evaluation[key], f"{source}.evaluation.{key}")
    if "task_offsets" in evaluation:
        offsets = _mapping(evaluation["task_offsets"], f"{source}.evaluation.task_offsets")
        for split, offset in offsets.items():
            _integer(offset, f"{source}.evaluation.task_offsets.{split}")
    if "primary_axis_tolerance_pp" in evaluation:
        _number(evaluation["primary_axis_tolerance_pp"],
                f"{source}.evaluation.primary_axis_tolerance_pp", minimum=0)
    for key in ("primary_axis", "final_holdout_access"):
        if key in evaluation:
            _string(evaluation[key], f"{source}.evaluation.{key}")

    evolution = root["evolution"]
    _required(evolution, {"allowed_artifacts", "max_candidates_per_round", "max_rounds",
                          "reflection_backend"}, f"{source}.evolution")
    _string_list(evolution["allowed_artifacts"], f"{source}.evolution.allowed_artifacts",
                 allow_empty=False)
    for key in ("max_candidates_per_round", "max_rounds"):
        _integer(evolution[key], f"{source}.evolution.{key}", minimum=1)
    _string(evolution["reflection_backend"], f"{source}.evolution.reflection_backend")
    for key in ("fewshot_enabled",):
        if key in evolution:
            _boolean(evolution[key], f"{source}.evolution.{key}")
    for key in ("fewshot_k", "reflection_max_attempts"):
        if key in evolution:
            _integer(evolution[key], f"{source}.evolution.{key}", minimum=1)
    if "behavior_channels" in evolution:
        channels = _mapping(evolution["behavior_channels"],
                            f"{source}.evolution.behavior_channels")
        _check_keys(channels, _NESTED_KEYS["evolution.behavior_channels"],
                    f"{source}.evolution.behavior_channels")
        if "enabled" in channels:
            _boolean(channels["enabled"], f"{source}.evolution.behavior_channels.enabled")
        if "enforce_in_evolution" in channels:
            _boolean(channels["enforce_in_evolution"],
                     f"{source}.evolution.behavior_channels.enforce_in_evolution")
        if "owner" in channels:
            owners = _mapping(channels["owner"],
                              f"{source}.evolution.behavior_channels.owner")
            _check_keys(owners, _NESTED_KEYS["evolution.behavior_channels.owner"],
                        f"{source}.evolution.behavior_channels.owner")
            for name, behaviors in owners.items():
                _string_list(behaviors,
                             f"{source}.evolution.behavior_channels.owner.{name}")
    if "gepa" in evolution:
        gepa = _mapping(evolution["gepa"], f"{source}.evolution.gepa")
        _check_keys(gepa, _NESTED_KEYS["evolution.gepa"], f"{source}.evolution.gepa")
        if "enabled" in gepa:
            _boolean(gepa["enabled"], f"{source}.evolution.gepa.enabled")
        for key in ("max_metric_calls", "max_candidate_proposals", "reflection_minibatch_size"):
            if key in gepa:
                _integer(gepa[key], f"{source}.evolution.gepa.{key}", minimum=1)
        for key in ("budget",):
            if key in gepa:
                _string(gepa[key], f"{source}.evolution.gepa.{key}")
    if "crystallization" in evolution:
        crystal = _mapping(evolution["crystallization"],
                           f"{source}.evolution.crystallization")
        _check_keys(crystal, _NESTED_KEYS["evolution.crystallization"],
                    f"{source}.evolution.crystallization")
        if "min_procedure_steps" in crystal:
            _integer(crystal["min_procedure_steps"],
                     f"{source}.evolution.crystallization.min_procedure_steps", minimum=1)
        for key in ("min_trigger_recall", "max_description_similarity",
                    "max_cross_domain_trigger", "max_non_target_trigger"):
            if key in crystal:
                _number(crystal[key], f"{source}.evolution.crystallization.{key}",
                        minimum=0, maximum=1)
        if "cross_domain_negative_limit" in crystal:
            _integer(crystal["cross_domain_negative_limit"],
                     f"{source}.evolution.crystallization.cross_domain_negative_limit")
    if "mipro" in evolution:
        mipro = _mapping(evolution["mipro"], f"{source}.evolution.mipro")
        _check_keys(mipro, _NESTED_KEYS["evolution.mipro"], f"{source}.evolution.mipro")
        if "enabled" in mipro:
            _boolean(mipro["enabled"], f"{source}.evolution.mipro.enabled")
        for key in ("max_proposer_calls", "num_candidates", "num_trials", "max_labeled_demos"):
            if key in mipro:
                _integer(mipro[key], f"{source}.evolution.mipro.{key}", minimum=1)
    if "proposer" in evolution and isinstance(evolution["proposer"], dict):
        proposer = evolution["proposer"]
        _check_keys(proposer, _NESTED_KEYS["evolution.proposer"],
                    f"{source}.evolution.proposer")
        if "enabled" in proposer:
            _boolean(proposer["enabled"], f"{source}.evolution.proposer.enabled")
        if "mode" in proposer:
            _string(proposer["mode"], f"{source}.evolution.proposer.mode")
    elif "proposer" in evolution:
        _string(evolution["proposer"], f"{source}.evolution.proposer")
    gates = root["gates"]
    _required(gates, {"static", "accept"}, f"{source}.gates")
    for section, required in (("static", {"max_prompt_chars", "require_single_artifact"}),
                              ("accept", {"require_static_gate", "require_human_review"})):
        child = _mapping(gates[section], f"{source}.gates.{section}")
        _check_keys(child, _NESTED_KEYS[f"gates.{section}"], f"{source}.gates.{section}")
        _required(child, required, f"{source}.gates.{section}")
    static = gates["static"]
    for key in ("max_prompt_chars", "max_description_chars", "max_body_chars",
                "max_procedure_steps", "max_task_demos_chars"):
        if key in static:
            _integer(static[key], f"{source}.gates.static.{key}", minimum=1)
    if "min_procedure_steps" in static:
        _integer(static["min_procedure_steps"],
                 f"{source}.gates.static.min_procedure_steps", minimum=1)
    if ("max_procedure_steps" in static and "min_procedure_steps" in static
            and static["max_procedure_steps"] < static["min_procedure_steps"]):
        raise ConfigError(f"{source}.gates.static.max_procedure_steps must be >= min_procedure_steps")
    _boolean(static["require_single_artifact"],
             f"{source}.gates.static.require_single_artifact")
    if "forbidden_patterns" in static:
        _string_list(static["forbidden_patterns"], f"{source}.gates.static.forbidden_patterns")
    accept = gates["accept"]
    for key in ("require_static_gate", "require_human_review", "main_metric_non_regression",
                "require_classwise_non_regression", "require_strict_accuracy_gain"):
        if key in accept:
            _boolean(accept[key], f"{source}.gates.accept.{key}")
    for key in ("gate_min_delta_pp", "gate_min_prob_delta_gt_0"):
        if key in accept:
            _number(accept[key], f"{source}.gates.accept.{key}", minimum=0,
                    maximum=1 if key.endswith("prob_delta_gt_0") else None)
    if "tolerance" in accept:
        tolerance = _mapping(accept["tolerance"], f"{source}.gates.accept.tolerance")
        _check_keys(tolerance, _NESTED_KEYS["gates.accept.tolerance"],
                    f"{source}.gates.accept.tolerance")
        for key, value in tolerance.items():
            if key == "required":
                _string_list(value, f"{source}.gates.accept.tolerance.required")
                continue
            if key == "strict":
                _boolean(value, f"{source}.gates.accept.tolerance.strict")
                continue
            _number(value, f"{source}.gates.accept.tolerance.{key}", minimum=0)
    if "minimum" in accept:
        minimum = _mapping(accept["minimum"], f"{source}.gates.accept.minimum")
        _check_keys(minimum, {"citation_support_rate", "answer_evidence_consistency_rate"},
                    f"{source}.gates.accept.minimum")
        for key, value in minimum.items():
            _number(value, f"{source}.gates.accept.minimum.{key}", minimum=0, maximum=1)
    for section in ("review", "execution", "privacy"):
        for key, value in root[section].items():
            if key in {"release", "final_holdout_unlock", "mode"}:
                _string(value, f"{source}.{section}.{key}")
    _required(root["review"], {"release", "final_holdout_unlock"}, f"{source}.review")
    _required(root["execution"], {"enabled", "max_expr_chars", "max_nodes",
                                   "max_output_chars"}, f"{source}.execution")
    _required(root["privacy"], {"mode", "redact_before_persist"}, f"{source}.privacy")
    _boolean(root["execution"]["enabled"], f"{source}.execution.enabled")
    for key in ("max_expr_chars", "max_nodes", "max_output_chars"):
        _integer(root["execution"][key], f"{source}.execution.{key}", minimum=1)
    if "allow_latex" in root["execution"]:
        _boolean(root["execution"]["allow_latex"], f"{source}.execution.allow_latex")
    if "docker_worker" in root["execution"]:
        worker = _mapping(root["execution"]["docker_worker"], f"{source}.execution.docker_worker")
        _check_keys(worker, _NESTED_KEYS["execution.docker_worker"], f"{source}.execution.docker_worker")
        _required(worker, {"enabled"}, f"{source}.execution.docker_worker")
        _boolean(worker["enabled"], f"{source}.execution.docker_worker.enabled")
        if worker["enabled"]:
            _required(worker, {"image","context","receipt_path","receipt_sha256"}, f"{source}.execution.docker_worker")
            for key in ("image","context","receipt_path","receipt_sha256"):
                _string(worker[key], f"{source}.execution.docker_worker.{key}")
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", worker["image"]) or not re.fullmatch(r"[0-9a-f]{64}", worker["receipt_sha256"]):
                raise ConfigError("docker_worker requires immutable image and receipt hashes")
        for key, low, high in (("timeout_seconds",1,120),("memory_mb",128,1024),("pids_limit",16,128)):
            if key in worker:
                _integer(worker[key], f"{source}.execution.docker_worker.{key}", minimum=low)
                if worker[key]>high:
                    raise ConfigError("docker_worker resource limit exceeds upper bound: "+key)
    _boolean(root["privacy"]["redact_before_persist"],
             f"{source}.privacy.redact_before_persist")
    if "wsl2" in root["execution"]:
        wsl2 = _mapping(root["execution"]["wsl2"], f"{source}.execution.wsl2")
        _check_keys(wsl2, _NESTED_KEYS["execution.wsl2"], f"{source}.execution.wsl2")
        if "enabled" in wsl2:
            _boolean(wsl2["enabled"], f"{source}.execution.wsl2.enabled")
        if "note" in wsl2:
            _string(wsl2["note"], f"{source}.execution.wsl2.note")
    _string(root["privacy"]["mode"], f"{source}.privacy.mode")


def resolve_placeholders(obj: Any, env: Dict[str, str] | None = None) -> Any:
    """递归展开 ${VAR}；缺失则抛错（不允许静默保留占位符）。"""
    env = dict(os.environ if env is None else env)
    return _walk(obj, env)


def _walk(node: Any, env: Dict[str, str]) -> Any:
    if isinstance(node, dict):
        return {k: _walk(v, env) for k, v in node.items()}
    if isinstance(node, list):
        return [_walk(v, env) for v in node]
    if isinstance(node, str):
        return _sub(node, env)
    return node


def _sub(text: str, env: Dict[str, str]) -> str:
    def repl(m: "re.Match[str]") -> str:
        name = m.group(1)
        if name not in env or env[name] == "":
            raise ConfigError(f"unresolved config placeholder: ${{{name}}}")
        return env[name]

    return _PLACEHOLDER.sub(repl, text)


def assert_no_placeholder(obj: Any) -> None:
    leftovers: List[str] = []

    def walk(n: Any) -> None:
        if isinstance(n, dict):
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
        elif isinstance(n, str):
            for m in _PLACEHOLDER.finditer(n):
                leftovers.append(m.group(0))

    walk(obj)
    if leftovers:
        raise ConfigError(f"manifest must not contain unresolved placeholders: {sorted(set(leftovers))}")


def config_fingerprint(cfg: Dict[str, Any]) -> str:
    """配置指纹：写入 run manifest，保证运行可重放。"""
    runtime = cfg.get("runtime") or {}
    use_pi = str(runtime.get("provider", "")).lower() == "pi"
    task_input = str(runtime.get("input_mode", "")).lower() == "task_input"
    if runtime.get("protocol_epoch") == "public-task-input-v1":
        if not task_input:
            raise ConfigError("public-task-input-v1 requires task_input")
        from .eval.run_manifest import source_tree_fingerprint
        upstream = (_locked_upstream_source_trees("pi", runtime_only=True) if use_pi else {})
        return sha256_obj({"protocol_epoch": "public-task-input-v1", "config": cfg,
                           "project_sources": source_tree_fingerprint(),
                           "installed_runtime_sources": upstream})
    if use_pi or task_input:
        # Pi is an out-of-process runtime.  Its bridge and locked package are
        # part of the experimental protocol even when YAML is unchanged.
        # Task-input also pins the evaluator, evidence conversion and gates so
        # judge changes cannot reuse old candidate/measurement identifiers.
        files = list(TASK_INPUT_PROTOCOL_SOURCES)
        if use_pi:
            files.extend(["tools/pi-bridge/run.mjs",
                          "tools/pi-bridge/package-lock.json",
                          "src/runtime/pi_bridge.py", "src/runtime/pi_trace.py"])
        source = {name: hashlib.sha256(path(name).read_bytes()).hexdigest()
                  for name in sorted(set(files))}
        upstream_trees = _locked_upstream_source_trees("pi" if use_pi else None)
        return sha256_obj({"config": cfg,
                           "task_protocol_sources": source,
                           "locked_upstream_source_trees": upstream_trees})
    return sha256_obj(cfg)


def deep_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
