"""Zero-call freeze for a CliniFact prompt + Skill-description search.

This module only freezes the inputs for a future GEPA description search.  It
does not call a model, score a task, or produce a candidate bundle.  The
resulting plan is deliberately tied to the sandbox release that owns the Skill
snapshot; callers must not silently resolve the repository ACTIVE release.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..contracts.types import content_id, content_sha256
from ..evolution.task_program import serialize_task_program
from ..runtime.provider import published_price_for_endpoint
from ..util import sha256_file, sha256_obj

PLAN_SCHEMA = "clinifact-prompt-skill-gepa-plan-v1"
PROTOCOL = "clinifact-prompt-skill-description-gepa-v1"
DATASET = "clinifact_fixed_abstract"
SPLIT = "dev_optimize"
DEFAULT_TASK_MODEL = "Qwen/Qwen3-8B"
DEFAULT_REFLECTION_MODEL = "Qwen/Qwen3-8B"
SOURCE_FILES = (
    "src/eval/clinifact_skill_candidate_plan.py",
    "src/eval/clinifact_skill_candidate_preflight.py",
    "src/eval/clinifact_skill_candidate_execute.py",
    "scripts/run_clinifact_skill_candidate_generation.py",
    "scripts/prepare_clinifact_skill_candidate_plan.py",
    "src/evolution/gepa_skill_search.py",
    "src/evolution/gepa_search.py",
    "src/evolution/task_program.py",
    "src/runtime/skills.py",
    "src/governance/policy.py",
)


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256")
    return value


def _validate_gepa_config(config_path: Path) -> dict[str, Any]:
    """Fail closed on retrieval, metric, or final-holdout config drift."""
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError("GEPA config is unreadable") from exc
    if not isinstance(cfg, dict):
        raise ValueError("GEPA config must be a mapping")
    runtime = cfg.get("runtime") or {}
    if runtime.get("input_mode") != "task_input":
        raise ValueError("CliniFact GEPA config requires task_input mode")
    if runtime.get("provider") != "pi":
        raise ValueError("CliniFact GEPA config requires the Pi task-input runtime")
    if (cfg.get("retrieval") or {}).get("backend") != "offline-fixture":
        raise ValueError("CliniFact GEPA config must use fixed offline task material")
    evaluation = cfg.get("evaluation") or {}
    if evaluation.get("main_metric") != "accuracy":
        raise ValueError("CliniFact Skill GEPA requires accuracy as its task score")
    if evaluation.get("primary_final_datasets"):
        raise ValueError("plan-only Skill GEPA config must not include final datasets")
    if evaluation.get("final_holdout_access") != "evaluator_only":
        raise ValueError("final holdout must remain evaluator-only")
    if (evaluation.get("datasets") or []) != [DATASET]:
        raise ValueError("CliniFact GEPA config must bind only clinifact_fixed_abstract")
    models = cfg.get("models") or {}
    selected = []
    endpoints = []
    for role in ("task", "reflection"):
        model = models.get(role) or {}
        model_id = model.get("model_id")
        if model.get("provider") != "http" or not str(model_id or "").strip():
            raise ValueError("task and reflection models must use an HTTP-compatible model")
        base_url = str(model.get("base_url") or "").strip()
        api_key_env = str(model.get("api_key_env") or "").strip()
        if not base_url or not api_key_env:
            raise ValueError("task and reflection HTTP models require base_url and api_key_env")
        pricing = model.get("pricing")
        if published_price_for_endpoint(str(model_id), base_url) is None:
            if not isinstance(pricing, dict) or any(
                    key not in pricing for key in ("prompt", "completion", "cache_read", "cache_write")):
                raise ValueError(
                    "unknown model tariff; provide explicit prompt/completion/cache rates")
        endpoints.append(base_url.rstrip("/"))
        selected.append(model_id)
    if selected[0] != selected[1]:
        raise ValueError("task and reflection models must be the same model for one GEPA budget")
    if endpoints[0] != endpoints[1]:
        raise ValueError("task and reflection models must use the same endpoint for one GEPA budget")
    return cfg


def _self_hash(value: dict[str, Any]) -> str:
    digest = value.get("plan_sha256")
    unsigned = dict(value)
    unsigned.pop("plan_sha256", None)
    if not isinstance(digest, str) or sha256_obj(unsigned) != digest:
        raise ValueError("GEPA plan self-hash is invalid")
    return digest


def _candidate(report: dict[str, Any], report_file_sha256: str) -> dict[str, Any]:
    if report.get("candidate_status") != "internal_gate_passed":
        raise ValueError("candidate report is not an internal preselection")
    if any(report.get(k) is not False for k in (
            "official_test_read", "official_validation_read", "selection_gate_read")):
        raise ValueError("candidate report read a protected evaluation scope")
    candidate = report.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("candidate report lacks candidate material")
    prompt = str(candidate.get("system_prompt") or "")
    demos = candidate.get("task_demos")
    if not prompt.strip() or not isinstance(demos, list):
        raise ValueError("candidate must contain a prompt and task-demo list")
    serialized = serialize_task_program(prompt, demos)
    release_id = str(report.get("release_id") or (report.get("plan") or {}).get("release_id") or "")
    if not release_id:
        raise ValueError("candidate report has no parent release identity")
    expected_id = content_id("cand", release_id, "task_program", serialized)
    if report.get("candidate_id") != expected_id:
        raise ValueError("candidate report task-program identity does not reconstruct")
    return {
        "candidate_id": expected_id,
        "candidate_report_sha256": _sha(report_file_sha256, "candidate_report_sha256"),
        "candidate_prompt_sha256": content_sha256(prompt),
        "candidate_task_program_sha256": content_sha256(serialized),
        "parent_release_id": release_id,
        "task_demos_sha256": sha256_obj(demos),
    }


def _release_snapshot(release_dir: Path, *, expected_release_id: str,
                      skill_name: str) -> dict[str, Any]:
    release_dir = release_dir.resolve()
    release_file = release_dir / "release.json"
    registry_file = release_dir / "skills" / "registry.yaml"
    prompt_file = release_dir / "system_prompt.txt"
    if not release_file.is_file() or not registry_file.is_file() or not prompt_file.is_file():
        raise ValueError("explicit sandbox release is incomplete")
    release = json.loads(release_file.read_text(encoding="utf-8"))
    if release.get("release_id") != expected_release_id:
        raise ValueError("sandbox release id differs from candidate parent release")
    raw = yaml.safe_load(registry_file.read_text(encoding="utf-8")) or {}
    rows = raw.get("skills") or []
    match = next((row for row in rows if row.get("name") == skill_name), None)
    if not isinstance(match, dict):
        raise ValueError("target Skill is absent from the explicit sandbox registry")
    if bool(match.get("frozen")) or match.get("group") not in {"G-main", "G-adj"}:
        raise ValueError("target Skill is frozen or outside description-evolution groups")
    description = str(match.get("description") or "").strip()
    if not description:
        raise ValueError("target Skill has no description seed")
    body_file = release_dir / "skills" / "bodies" / f"{skill_name}.md"
    if not body_file.is_file():
        raise ValueError("target Skill body snapshot is missing")
    registry_sha = sha256_file(registry_file)
    return {
        "release_dir": str(release_dir),
        "release_file_sha256": sha256_file(release_file),
        "registry_file_sha256": registry_sha,
        "prompt_file_sha256": sha256_file(prompt_file),
        "release_id": expected_release_id,
        "skill_name": skill_name,
        "skill_group": str(match.get("group")),
        "skill_description_sha256": content_sha256(description),
        "skill_body_sha256": sha256_file(body_file),
        "skill_triggers_sha256": sha256_obj(list(match.get("triggers") or [])),
        "skill_frozen": False,
    }


def _task_manifest(rows: Iterable[dict[str, Any]], task_limit: int) -> dict[str, Any]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: str(row.get("task_id", "")))
    if task_limit < 6 or len(ordered) < task_limit:
        raise ValueError("CliniFact plan needs at least six dev_optimize tasks")
    selected = ordered[:task_limit]
    if any(row.get("dataset") != DATASET or row.get("split") != SPLIT for row in selected):
        raise ValueError("GEPA task scope must contain only CliniFact dev_optimize rows")
    ids = [str(row.get("task_id") or "") for row in selected]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("GEPA task IDs must be unique and non-empty")
    materials = {task_id: sha256_obj(row) for task_id, row in zip(ids, selected)}
    n_train = max(3, int(task_limit * 0.8))
    return {
        "dataset": DATASET, "split": SPLIT, "task_count": task_limit,
        "train_task_ids": ids[:n_train], "validation_task_ids": ids[n_train:],
        "task_ids": ids, "task_material_sha256": materials,
        "task_manifest_sha256": sha256_obj({"task_ids": ids, "task_material_sha256": materials}),
    }


def build_plan(*, candidate_report: dict[str, Any], candidate_report_path: Path,
               candidate_report_file_sha256: str,
               release_dir: Path, skill_name: str, task_rows: Iterable[dict[str, Any]],
               config_path: Path, task_model_id: str | None = None,
               reflection_model_id: str | None = None,
               max_metric_calls: int = 120, max_candidate_proposals: int = 8,
               reflection_minibatch_size: int = 5, seed: int = 20261018,
               task_limit: int = 30) -> dict[str, Any]:
    if min(max_metric_calls, max_candidate_proposals, reflection_minibatch_size) < 1:
        raise ValueError("GEPA budgets must be positive")
    cand = _candidate(candidate_report, candidate_report_file_sha256)
    snapshot = _release_snapshot(Path(release_dir), expected_release_id=cand["parent_release_id"],
                                 skill_name=skill_name)
    if type(task_limit) is not int or task_limit < 6:
        raise ValueError("task_limit must be an integer >= 6")
    tasks = _task_manifest(task_rows, task_limit)
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise ValueError("GEPA config file is missing")
    config = _validate_gepa_config(config_path)
    config_task_model = str((config.get("models") or {}).get("task", {}).get("model_id") or "")
    config_reflection_model = str((config.get("models") or {}).get("reflection", {}).get("model_id") or "")
    task_model_id = task_model_id or config_task_model
    reflection_model_id = reflection_model_id or config_reflection_model
    if task_model_id != config_task_model or reflection_model_id != config_reflection_model:
        raise ValueError("plan model identities must match the frozen GEPA config")
    source_hashes = {name: sha256_file(Path(__file__).resolve().parents[2] / name)
                     for name in SOURCE_FILES}
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA, "protocol": PROTOCOL,
        "status": "plan_only_pending_skill_generation", "model_calls": 0,
        "network_calls": 0, "dispatch_authorized": False,
        "candidate_prompt": cand, "candidate_report_path": str(candidate_report_path.resolve()),
        "release_snapshot": snapshot, "target_artifact": "skill_description",
        "task_scope": tasks,
        "gepa": {
            "task_model_id": task_model_id, "reflection_model_id": reflection_model_id,
            "config_path": str(config_path), "config_sha256": sha256_file(config_path),
            "max_metric_calls": int(max_metric_calls),
            "max_candidate_proposals": int(max_candidate_proposals),
            "reflection_minibatch_size": int(reflection_minibatch_size),
            "seed": int(seed), "candidate_generation_only": True,
            "evaluation_required_after_generation": True,
        },
        "source_code_sha256": source_hashes,
        "fail_closed": [
            "do not resolve ACTIVE or another release implicitly",
            "do not call GEPA before a separate execution approval and fresh ledger",
            "do not create a bundle until a real skill_description candidate is generated",
            "do not reuse the old single-artifact confirmation report as bundle confirmation",
            "do not read selection_gate, official validation, or official test material",
        ],
    }
    plan["plan_sha256"] = sha256_obj(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA or plan.get("protocol") != PROTOCOL:
        raise ValueError("unsupported CliniFact Skill GEPA plan")
    _self_hash(plan)
    if plan.get("status") != "plan_only_pending_skill_generation" or plan.get("model_calls") != 0 \
            or plan.get("network_calls") != 0 or plan.get("dispatch_authorized") is not False:
        raise ValueError("plan is not zero-call and dispatch-disabled")
    if plan.get("target_artifact") != "skill_description":
        raise ValueError("plan target must be skill_description")
    if plan.get("release_snapshot", {}).get("skill_frozen") is not False:
        raise ValueError("plan target Skill is frozen")
    task_scope = plan.get("task_scope") or {}
    if task_scope.get("dataset") != DATASET or task_scope.get("split") != SPLIT:
        raise ValueError("plan task scope is outside CliniFact dev_optimize")
    if task_scope.get("task_count") != len(task_scope.get("task_ids") or []):
        raise ValueError("plan task count does not match task IDs")


