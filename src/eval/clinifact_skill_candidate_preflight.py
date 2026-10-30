"""Zero-call preflight for the frozen CliniFact Skill GEPA plan."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..util import sha256_file, sha256_obj
from .clinifact_skill_candidate_plan import PLAN_SCHEMA, validate_plan

APPROVAL_SCHEMA = "clinifact-prompt-skill-gepa-approval-v1"
PREFLIGHT_SCHEMA = "clinifact-prompt-skill-gepa-preflight-v1"


def _self_hash(value: dict[str, Any], field: str, label: str) -> str:
    digest = value.get(field)
    unsigned = dict(value)
    unsigned.pop(field, None)
    if not isinstance(digest, str) or sha256_obj(unsigned) != digest:
        raise ValueError(f"{label} self-hash is invalid")
    return digest


def _verify_plan_inputs(plan: dict[str, Any]) -> None:
    candidate = plan["candidate_prompt"]
    report_path = Path(plan["candidate_report_path"]).resolve()
    if not report_path.is_file() or sha256_file(report_path) != candidate["candidate_report_sha256"]:
        raise ValueError("candidate report is missing or changed")
    gepa = plan["gepa"]
    config_path = Path(gepa["config_path"]).resolve()
    if not config_path.is_file() or sha256_file(config_path) != gepa["config_sha256"]:
        raise ValueError("GEPA config is missing or changed")
    snapshot = plan["release_snapshot"]
    release_dir = Path(snapshot["release_dir"]).resolve()
    checks = {
        "release.json": snapshot["release_file_sha256"],
        "skills/registry.yaml": snapshot["registry_file_sha256"],
        "system_prompt.txt": snapshot["prompt_file_sha256"],
        f"skills/bodies/{snapshot['skill_name']}.md": snapshot["skill_body_sha256"],
    }
    for relative, expected in checks.items():
        path = release_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"sandbox release input changed: {relative}")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in (plan.get("source_code_sha256") or {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"GEPA source input changed: {relative}")


def _verify_approval(approval: dict[str, Any], *, plan_sha256: str,
                     plan_file_sha256: str) -> None:
    if approval.get("schema") != APPROVAL_SCHEMA or approval.get("status") != "approved":
        raise ValueError("approval is missing or not approved")
    if approval.get("plan_sha256") != plan_sha256 or approval.get("plan_file_sha256") != plan_file_sha256:
        raise ValueError("approval is not bound to this exact plan")
    if approval.get("dispatch_authorized") is not True:
        raise ValueError("approval does not authorize dispatch")
    if any(approval.get(key) is not False for key in (
            "official_test_read", "official_validation_read", "selection_gate_read")):
        raise ValueError("approval reads a protected evaluation scope")
    _self_hash(approval, "approval_sha256", "approval")


def preflight(plan_path: Path, *, expected_plan_sha256: str,
              ledger_path: Path, approval_path: Path | None = None) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    if not plan_path.is_file():
        raise ValueError("plan file is missing")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("plan_sha256") != expected_plan_sha256:
        raise ValueError("plan self-hash differs from the execution challenge")
    _verify_plan_inputs(plan)
    ledger_path = ledger_path.resolve()
    if ledger_path.exists():
        raise ValueError("execution ledger already exists; refusing reuse")
    authorized = False
    approval_sha = None
    if approval_path is not None:
        approval_path = approval_path.resolve()
        if not approval_path.is_file():
            raise ValueError("approval file is missing")
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        _verify_approval(approval, plan_sha256=expected_plan_sha256,
                         plan_file_sha256=sha256_file(plan_path))
        approval_sha = sha256_file(approval_path)
        authorized = True
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "dispatch_authorized" if authorized else "pending_root_approval",
        "plan_path": str(plan_path), "plan_sha256": expected_plan_sha256,
        "ledger_path": str(ledger_path), "approval_path": str(approval_path) if approval_path else None,
        "approval_file_sha256": approval_sha,
        "model_calls": 0, "network_calls": 0,
        "dispatch_authorized": authorized,
        "protected_reads": {"official_test_read": False,
                             "official_validation_read": False,
                             "selection_gate_read": False},
    }
