"""Plan-bound execution for the CliniFact Skill-description GEPA arm.

The executor deliberately owns no planning policy.  It consumes one immutable
plan, one exact approval, and one exclusive write-ahead ledger, then invokes
the existing pinned GEPA implementation against the explicit sandbox release.
It never creates a confirmation bundle or reads a protected evaluation split.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from ..config import load_config
from ..data.split import load_tasks
from ..runtime.runner import ArtifactState
from ..runtime.skills import load_registry
from ..runtime.call_ledger import summarize_ledger
from ..util import sha256_file, sha256_obj, write_json
from ..evolution.gepa_skill_search import search_skill_description_candidate
from ..contracts.types import content_sha256
from .clinifact_skill_candidate_preflight import preflight

EXECUTION_SCHEMA = "clinifact-prompt-skill-gepa-execution-v1"


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _load_state(plan: dict[str, Any], cfg: dict[str, Any]) -> ArtifactState:
    snapshot = plan["release_snapshot"]
    release_dir = Path(snapshot["release_dir"]).resolve()
    try:
        release_dir.relative_to((Path(__file__).resolve().parents[2] / "artifacts" / "sandboxes").resolve())
    except ValueError as exc:
        raise ValueError("Skill GEPA must load the explicit sandbox release, never ACTIVE") from exc
    registry_path = release_dir / "skills" / "registry.yaml"
    registry = load_registry(str(registry_path))
    if registry.release_id != snapshot["release_id"]:
        raise ValueError("sandbox registry release id differs from the frozen plan")
    # The sandbox registry is intentionally independent from ACTIVE.  Its
    # historical body_path values may still be repository-relative release
    # paths, so make the explicit sandbox snapshot the runtime source of
    # truth and inline every active body before Pi materializes Skills.
    for skill in registry.skills:
        body_file = release_dir / "skills" / "bodies" / f"{skill.name}.md"
        if not body_file.is_file():
            raise ValueError(f"sandbox Skill body is missing: {skill.name}")
        # Inline frozen and retired snapshots too.  Routing normally excludes
        # retired skills, but inlining every registry entry makes a malformed
        # route unable to escape through a historical artifacts/releases path.
        skill.body = body_file.read_text(encoding="utf-8")
    # Skill-description search is the second component of the frozen
    # prompt+Skill candidate.  Its task runner must see the exact prompt
    # candidate that the plan bound, rather than silently falling back to the
    # sandbox baseline prompt.  The report file/hash was checked by preflight;
    # re-check the two content identities at this boundary as well.
    report = json.loads(Path(plan["candidate_report_path"]).read_text(encoding="utf-8"))
    candidate = report.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("candidate report has no candidate material")
    prompt = str(candidate.get("system_prompt") or "")
    demos = candidate.get("task_demos")
    bound = plan["candidate_prompt"]
    if not prompt.strip() or not isinstance(demos, list):
        raise ValueError("candidate prompt material is incomplete")
    if content_sha256(prompt) != bound["candidate_prompt_sha256"]:
        raise ValueError("candidate prompt content differs from the frozen plan")
    if sha256_obj(demos) != bound["task_demos_sha256"]:
        raise ValueError("candidate task demos differ from the frozen plan")
    return ArtifactState(snapshot["release_id"], prompt, registry, task_demos=demos)


def execute(plan_path: Path, *, expected_plan_sha256: str,
            approval_path: Path, ledger_path: Path, output_path: Path,
            search_fn: Callable[..., Any] = search_skill_description_candidate) -> dict[str, Any]:
    """Run exactly one approved plan and persist a candidate-generation receipt."""
    plan_path = plan_path.resolve()
    approval_path = approval_path.resolve()
    ledger_path = ledger_path.resolve()
    output_path = output_path.resolve()
    checked = preflight(plan_path, expected_plan_sha256=expected_plan_sha256,
                        ledger_path=ledger_path, approval_path=approval_path)
    if not checked["dispatch_authorized"]:
        raise ValueError("GEPA execution requires an exact dispatch approval")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    cfg = load_config(plan["gepa"]["config_path"])
    # Reserve the complete hard budget before the first physical request.
    metric_budget = int(plan["gepa"]["max_metric_calls"])
    proposal_budget = int(plan["gepa"]["max_candidate_proposals"])
    logical_budget = metric_budget + proposal_budget + 1
    physical_budget = logical_budget * 8
    ledger = {
        "schema": EXECUTION_SCHEMA,
        "status": "reserved",
        "plan_sha256": expected_plan_sha256,
        "plan_file_sha256": sha256_file(plan_path),
        "approval_file_sha256": sha256_file(approval_path),
        "max_metric_calls": metric_budget,
        "max_candidate_proposals": proposal_budget,
        "max_logical_calls": logical_budget,
        "max_physical_attempts": physical_budget,
        "model_calls": 0,
        "network_calls": 0,
        "protected_reads": checked["protected_reads"],
    }
    _exclusive_json(ledger_path, ledger)
    call_ledger_path = ledger_path.with_name("provider-call-ledger.jsonl")
    if call_ledger_path.exists():
        raise ValueError("provider call ledger already exists; refusing reuse")
    call_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    call_ledger_path.touch(exist_ok=False)
    state = _load_state(plan, cfg)
    task_count = int(plan["task_scope"]["task_count"])
    tasks = load_tasks("clinifact_fixed_abstract", splits=["dev_optimize"])
    result: dict[str, Any] = {
        "schema": EXECUTION_SCHEMA,
        "status": "candidate_generation_started",
        "plan_sha256": expected_plan_sha256,
        "ledger_path": str(ledger_path),
        "provider_call_ledger_path": str(call_ledger_path),
        "candidate_id": None,
        "candidate": None,
        "protected_reads": checked["protected_reads"],
    }
    # One logical operation is reserved for the run boundary; GEPA then owns
    # metric evaluations and proposal/reflection calls within the frozen plan.
    env_names = (
        "RSI_CALL_LEDGER_PATH", "RSI_CALL_LEDGER_MAX_LOGICAL",
        "RSI_CALL_LEDGER_MAX_PHYSICAL", "RSI_CALL_LEDGER_MAX_COMPONENT_TASK_ANSWER",
        "RSI_CALL_LEDGER_MAX_COMPONENT_PROVIDER_HTTP",
        "RSI_ACTIVE_EXECUTION_PLAN_SHA256", "RSI_CALL_LEDGER_RESERVE_PI_PHYSICAL",
    )
    previous_env = {name: os.environ.get(name) for name in env_names}
    try:
        os.environ.update({
            "RSI_CALL_LEDGER_PATH": str(call_ledger_path),
            "RSI_CALL_LEDGER_MAX_LOGICAL": str(logical_budget),
            "RSI_CALL_LEDGER_MAX_PHYSICAL": str(physical_budget),
            "RSI_CALL_LEDGER_MAX_COMPONENT_TASK_ANSWER": str(metric_budget),
            "RSI_CALL_LEDGER_MAX_COMPONENT_PROVIDER_HTTP": str(logical_budget),
            "RSI_ACTIVE_EXECUTION_PLAN_SHA256": expected_plan_sha256,
            "RSI_CALL_LEDGER_RESERVE_PI_PHYSICAL": "1",
        })
        candidate = search_fn(
            cfg, state, ["clinifact_fixed_abstract"], task_count,
            int(plan["gepa"]["seed"]), [], plan["release_snapshot"]["skill_name"])
        if candidate is None:
            result["status"] = "no_skill_candidate"
        else:
            result["status"] = "skill_candidate_generated"
            result["candidate_id"] = candidate.candidate_id
            result["candidate"] = candidate.to_dict()
            result["search_report_path"] = candidate.generator.get("search_report")
        ledger["status"] = result["status"]
        ledger["candidate_id"] = result["candidate_id"]
        ledger["model_calls"] = None
        ledger["network_calls"] = None
    except Exception as exc:
        result.update({"status": "failed_closed", "error_type": type(exc).__name__})
        ledger.update({"status": "failed_closed", "error_type": type(exc).__name__})
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        accounting = summarize_ledger(call_ledger_path)
        result["request_accounting"] = accounting
        result["provider_call_ledger_sha256"] = sha256_file(call_ledger_path)
        ledger["request_accounting"] = accounting
        ledger["model_calls"] = accounting["logical_operations"]
        ledger["network_calls"] = accounting["physical_http_attempts"]
    # Persist the terminal state after candidate generation.  The reservation
    # file is write-once at dispatch time, but leaving it at ``reserved`` after
    # a normal no-candidate result makes recovery/audit indistinguishable from
    # an interrupted run.  Keep the result hash independent of the later
    # ledger hash to avoid a circular identity.
    ledger["result_sha256"] = sha256_obj(result)
    write_json(ledger_path, ledger)
    result["ledger_sha256"] = sha256_file(ledger_path)
    write_json(output_path, result)
    return result

