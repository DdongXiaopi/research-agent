"""Build and validate the prompt+Skill candidate bundle used by G03.

The search executors intentionally produce single-artifact candidates.  G03
needs a second, immutable object that binds the two candidates to one parent
release and spells out the A0/A1/A2 intervention states.  This module is
offline-only: it reads reports and a release snapshot and never calls a
provider or reads an evaluation split.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts.types import content_sha256, content_id
from ..runtime.skills import load_registry
from ..util import read_json, sha256_file, sha256_obj
from .clinifact_skill_candidate_plan import _candidate as validate_prompt_candidate

BUNDLE_SCHEMA = "g03-confirmed-candidate-bundle-v1"
PROMPT_ARTIFACT = "system_prompt"
SKILL_ARTIFACT = "skill_description"


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256")
    return value


def _self_hash(bundle: dict[str, Any]) -> str:
    digest = bundle.get("candidate_bundle_sha256")
    unsigned = dict(bundle)
    unsigned.pop("candidate_bundle_sha256", None)
    if not isinstance(digest, str) or sha256_obj(unsigned) != digest:
        raise ValueError("candidate bundle self-hash is invalid")
    return digest


def _release_baseline(release_dir: Path, *, parent_release_id: str,
                      skill_name: str) -> tuple[str, str]:
    release_dir = release_dir.resolve()
    release = read_json(release_dir / "release.json")
    if release.get("release_id") != parent_release_id:
        raise ValueError("candidate parent release differs from explicit release snapshot")
    prompt_path = release_dir / "system_prompt.txt"
    registry_path = release_dir / "skills" / "registry.yaml"
    if not prompt_path.is_file() or not registry_path.is_file():
        raise ValueError("explicit release snapshot is incomplete")
    registry = load_registry(str(registry_path))
    skill = registry.get(skill_name)
    if skill is None or skill.frozen or skill.lifecycle != "active":
        raise ValueError("target Skill is absent, frozen, or inactive")
    return content_sha256(prompt_path.read_text(encoding="utf-8")), content_sha256(skill.description)


def build_candidate_bundle(*, prompt_report: dict[str, Any], prompt_report_path: Path,
                           skill_result: dict[str, Any], skill_result_path: Path,
                           release_dir: Path, skill_name: str,
                           candidate_bundle_id: str | None = None) -> dict[str, Any]:
    """Create a hash-bound bundle from two independently generated candidates."""
    prompt = validate_prompt_candidate(prompt_report, sha256_file(prompt_report_path))
    if skill_result.get("schema") != "clinifact-prompt-skill-gepa-execution-v1":
        raise ValueError("Skill result does not use the current GEPA execution schema")
    if skill_result.get("status") != "skill_candidate_generated":
        raise ValueError("Skill result is not a generated candidate")
    skill = skill_result.get("candidate")
    if not isinstance(skill, dict) or skill.get("artifact_type") != SKILL_ARTIFACT:
        raise ValueError("Skill result does not contain a skill_description candidate")
    if skill.get("status") not in {"draft", "evaluated"}:
        raise ValueError("Skill candidate has an unsupported lifecycle status")
    generator = skill.get("generator") or {}
    if generator.get("validation_tie") is True:
        raise ValueError("Skill candidate is only tied on validation and cannot enter a confirmation bundle")
    if generator.get("strict_validation_improvement") is False:
        raise ValueError("Skill candidate does not have a strict validation improvement")
    parent = str(skill.get("parent_release_id") or prompt["parent_release_id"])
    if parent != prompt["parent_release_id"]:
        raise ValueError("prompt and Skill candidates have different parent releases")
    if skill.get("artifact_id") != skill_name:
        raise ValueError("Skill candidate targets a different Skill")
    new_description = str(skill.get("new_content") or "").strip()
    if not new_description:
        raise ValueError("Skill candidate has empty description")
    skill_sha = content_sha256(new_description)
    if skill.get("content_sha256") != skill_sha:
        raise ValueError("Skill candidate content hash does not reconstruct")
    baseline_prompt, baseline_skill = _release_baseline(
        release_dir, parent_release_id=parent, skill_name=skill_name)
    if prompt["candidate_prompt_sha256"] == baseline_prompt:
        raise ValueError("prompt candidate is identical to the parent prompt")
    if skill_sha == baseline_skill:
        raise ValueError("Skill candidate is identical to the parent description")
    prompt_member = {
        "role": "prompt", "candidate_id": prompt["candidate_id"],
        "artifact_type": PROMPT_ARTIFACT, "artifact_id": PROMPT_ARTIFACT,
        "parent_release_id": parent, "content_sha256": prompt["candidate_prompt_sha256"],
    }
    skill_member = {
        "role": "skill_description", "candidate_id": str(skill_result["candidate_id"]),
        "artifact_type": SKILL_ARTIFACT, "artifact_id": skill_name,
        "parent_release_id": parent, "content_sha256": skill_sha,
    }
    members = [prompt_member, skill_member]
    bundle_id = candidate_bundle_id or content_id(
        "bundle", parent, prompt_member["candidate_id"], skill_member["candidate_id"],
        prompt_member["content_sha256"], skill_member["content_sha256"])
    bundle: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "candidate_bundle_id": bundle_id,
        "parent_release_id": parent,
        "skill_name": skill_name,
        "members": members,
        "baseline": {"system_prompt_sha256": baseline_prompt,
                      "skill_description_sha256": baseline_skill},
        "interventions": {
            "A0": {"system_prompt_sha256": baseline_prompt,
                   "skill_description_sha256": baseline_skill},
            "A1": {"system_prompt_sha256": prompt_member["content_sha256"],
                   "skill_description_sha256": baseline_skill},
            "A2": {"system_prompt_sha256": baseline_prompt,
                   "skill_description_sha256": skill_member["content_sha256"]},
        },
        "source_reports": {
            "prompt_report_path": str(prompt_report_path.resolve()),
            "prompt_report_sha256": prompt["candidate_report_sha256"],
            "skill_result_path": str(skill_result_path.resolve()),
            "skill_result_sha256": sha256_file(skill_result_path),
            "skill_search_report_path": str(skill.get("generator", {}).get("search_report") or ""),
        },
        "candidate_bundle_sha256": "",
    }
    bundle["candidate_bundle_sha256"] = sha256_obj({
        key: value for key, value in bundle.items() if key != "candidate_bundle_sha256"
    })
    validate_candidate_bundle(bundle)
    return bundle


def validate_candidate_bundle(bundle: dict[str, Any]) -> None:
    """Validate bundle identity and the three single-component interventions."""
    if bundle.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("unsupported candidate bundle schema")
    _self_hash(bundle)
    members = bundle.get("members")
    if not isinstance(members, list) or len(members) != 2:
        raise ValueError("candidate bundle must contain exactly two members")
    roles = {row.get("role"): row for row in members if isinstance(row, dict)}
    if set(roles) != {"prompt", "skill_description"}:
        raise ValueError("candidate bundle must contain prompt and skill_description")
    parent = bundle.get("parent_release_id")
    if not parent or any(row.get("parent_release_id") != parent for row in members):
        raise ValueError("candidate bundle member parent release mismatch")
    for role, row in roles.items():
        expected = PROMPT_ARTIFACT if role == "prompt" else SKILL_ARTIFACT
        if row.get("artifact_type") != expected or not row.get("candidate_id"):
            raise ValueError(f"invalid {role} candidate identity")
        _sha(row.get("content_sha256"), f"{role} content_sha256")
    baseline = bundle.get("baseline")
    arms = bundle.get("interventions")
    if not isinstance(baseline, dict) or not isinstance(arms, dict) or set(arms) != {"A0", "A1", "A2"}:
        raise ValueError("candidate bundle baseline/intervention states are incomplete")
    for key in ("system_prompt_sha256", "skill_description_sha256"):
        _sha(baseline.get(key), f"baseline.{key}")
    if arms["A0"] != baseline:
        raise ValueError("A0 must equal the parent release state")
    if arms["A1"].get("system_prompt_sha256") != roles["prompt"]["content_sha256"] \
            or arms["A1"].get("skill_description_sha256") != baseline["skill_description_sha256"]:
        raise ValueError("A1 must contain only the prompt candidate")
    if arms["A2"].get("system_prompt_sha256") != baseline["system_prompt_sha256"] \
            or arms["A2"].get("skill_description_sha256") != roles["skill_description"]["content_sha256"]:
        raise ValueError("A2 must contain only the Skill-description candidate")


__all__ = ["BUNDLE_SCHEMA", "build_candidate_bundle", "validate_candidate_bundle"]
