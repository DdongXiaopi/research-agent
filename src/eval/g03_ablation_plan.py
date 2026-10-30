"""Zero-call, candidate-bound plan validation for the G03 A0/A1/A2 study.

This module freezes a post-confirmation intervention bundle and a disjoint
visible development cohort. It does not score tasks, call a model, or consume a
cohort. The bundle attestation contract deliberately fails closed until G02 can
attest the exact prompt+description bundle used by A1/A2.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from ..util import sha256_file, sha256_obj

PROTOCOL = "g03-fixed-candidate-a0-a1-a2-v1"
PLAN_SCHEMA = "g03-fixed-candidate-ablation-plan-v1"
BUNDLE_SCHEMA = "g03-confirmed-candidate-bundle-v1"
ATTESTATION_SCHEMA = "g03-candidate-bundle-confirmation-report-v1"
COHORT_SCHEMA = "g03-frozen-development-cohort-v1"
ARMS = ("A0", "A1", "A2")
PROTOCOL_SOURCES = (
    "src/eval/g03_ablation_plan.py",
    "src/eval/ablation.py",
    "src/eval/harness.py",
    "src/eval/stats.py",
    "scripts/plan_g03_ablation.py",
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _self_hashed(value: dict[str, Any], field: str, context: str) -> str:
    unsigned = dict(value)
    digest = unsigned.pop(field, None)
    if not isinstance(digest, str) or sha256_obj(unsigned) != digest:
        raise ValueError(f"{context} self-hash is invalid")
    return digest


def _protocol_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {name: sha256_file(root / name) for name in PROTOCOL_SOURCES}


def _validate_bundle(bundle: dict[str, Any]) -> tuple[str, list[dict[str, str]], dict[str, dict[str, str]]]:
    if bundle.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("candidate bundle must use the G03 confirmed-bundle schema")
    bundle_sha = _self_hashed(bundle, "candidate_bundle_sha256", "candidate bundle")
    bundle_id = bundle.get("candidate_bundle_id")
    parent = bundle.get("parent_release_id")
    if not isinstance(bundle_id, str) or not bundle_id or not isinstance(parent, str) or not parent:
        raise ValueError("candidate bundle requires candidate_bundle_id and parent_release_id")
    raw_members = bundle.get("members")
    if not isinstance(raw_members, list) or len(raw_members) != 2:
        raise ValueError("G03 bundle requires exactly one prompt and one description candidate")
    members: list[dict[str, str]] = []
    roles: dict[str, dict[str, str]] = {}
    for item in raw_members:
        if not isinstance(item, dict):
            raise ValueError("candidate bundle members must be objects")
        role = item.get("role")
        if role not in {"prompt", "skill_description"} or role in roles:
            raise ValueError("candidate bundle must have unique prompt and skill_description roles")
        if item.get("parent_release_id") != parent:
            raise ValueError("candidate bundle member parent release differs from bundle")
        member = {
            "role": role,
            "candidate_id": str(item.get("candidate_id") or ""),
            "artifact_type": str(item.get("artifact_type") or ""),
            "artifact_id": str(item.get("artifact_id") or ""),
            "content_sha256": _require_sha(item.get("content_sha256"), f"{role} content_sha256"),
        }
        expected_type = "system_prompt" if role == "prompt" else "skill_description"
        if (not member["candidate_id"] or not member["artifact_id"]
                or member["artifact_type"] != expected_type):
            raise ValueError(f"{role} candidate identity is incomplete or has the wrong artifact type")
        roles[role] = member
        members.append(member)
    if set(roles) != {"prompt", "skill_description"}:
        raise ValueError("G03 bundle must include both prompt and description interventions")

    raw_arms = bundle.get("interventions")
    if not isinstance(raw_arms, dict) or set(raw_arms) != set(ARMS):
        raise ValueError("candidate bundle must define exactly A0, A1, and A2 interventions")
    arms: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        value = raw_arms[arm]
        if not isinstance(value, dict) or set(value) != {
                "system_prompt_sha256", "skill_description_sha256"}:
            raise ValueError(f"{arm} must bind prompt and skill-description state hashes")
        arms[arm] = {key: _require_sha(value[key], f"{arm}.{key}") for key in value}
    baseline = arms["A0"]
    if (arms["A1"]["system_prompt_sha256"] != roles["prompt"]["content_sha256"]
            or arms["A1"]["skill_description_sha256"] != baseline["skill_description_sha256"]
            or arms["A1"]["system_prompt_sha256"] == baseline["system_prompt_sha256"]):
        raise ValueError("A1 must change only the confirmed prompt intervention")
    if (arms["A2"]["skill_description_sha256"] != roles["skill_description"]["content_sha256"]
            or arms["A2"]["system_prompt_sha256"] != baseline["system_prompt_sha256"]
            or arms["A2"]["skill_description_sha256"] == baseline["skill_description_sha256"]):
        raise ValueError("A2 must change only the confirmed Skill-description intervention")
    return bundle_sha, members, arms


def _validate_confirmation(report: dict[str, Any], *, bundle: dict[str, Any],
                           bundle_sha: str, members: list[dict[str, str]],
                           target_model_id: str, target_protocol: str) -> tuple[str, str, str]:
    if report.get("schema") != ATTESTATION_SCHEMA:
        raise ValueError("G03 requires a candidate-bundle confirmation attestation, not a single-artifact report")
    plan = report.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("candidate confirmation attestation lacks its frozen plan")
    plan_sha = _self_hashed(plan, "plan_sha256", "candidate confirmation plan")
    if (report.get("status") != "independent_confirmation_passed"
            or report.get("independent_confirmation_passed") is not True
            or report.get("confirmation_eligible") is not True):
        raise ValueError("candidate bundle has no passing independent confirmation attestation")
    if (report.get("candidate_bundle_id") != bundle.get("candidate_bundle_id")
            or report.get("candidate_bundle_sha256") != bundle_sha
            or report.get("parent_release_id") != bundle.get("parent_release_id")
            or report.get("candidate_members") != members
            or plan.get("candidate_bundle_id") != bundle.get("candidate_bundle_id")
            or plan.get("candidate_bundle_sha256") != bundle_sha
            or plan.get("candidate_members") != members
            or plan.get("parent_release_id") != bundle.get("parent_release_id")):
        raise ValueError("confirmation attestation is not bound to exact G03 candidate bundle contents")
    if (report.get("target_model_id") != target_model_id
            or report.get("source_search_model_id") != target_model_id
            or report.get("target_protocol") != target_protocol
            or plan.get("target_model_id") != target_model_id
            or plan.get("target_protocol") != target_protocol):
        raise ValueError("confirmation model/protocol differs from the requested G03 target")
    if any(report.get(field) is not False for field in (
            "official_test_read", "official_validation_read", "selection_gate_read")):
        raise ValueError("candidate confirmation attestation read a protected evaluation scope")
    source_cohort_sha = _require_sha(report.get("source_cohort_sha256"),
                                     "confirmation source_cohort_sha256")
    confirmation_cohort_sha = _require_sha(report.get("confirmation_cohort_sha256"),
                                           "confirmation_cohort_sha256")
    accounting = report.get("request_accounting")
    if (not isinstance(accounting, dict)
            or accounting.get("logical_incomplete") != 0
            or accounting.get("physical_incomplete") != 0
            or accounting.get("physical_attempts_exact") is not True
            or not _HEX64.fullmatch(str(report.get("request_ledger_sha256") or ""))):
        raise ValueError("candidate confirmation request ledger is not complete and attested")
    if plan.get("source_cohort_sha256") != source_cohort_sha or plan.get(
            "confirmation_cohort_sha256") != confirmation_cohort_sha:
        raise ValueError("confirmation plan/report cohort identities differ")
    return plan_sha, source_cohort_sha, confirmation_cohort_sha


def _validate_cohort(cohort: dict[str, Any], *, bundle_sha: str,
                     source_cohort_sha: str, confirmation_cohort_sha: str) -> tuple[str, list[dict[str, str]]]:
    if cohort.get("schema") != COHORT_SCHEMA:
        raise ValueError("G03 requires a frozen development-only cohort manifest")
    cohort_sha = _self_hashed(cohort, "cohort_sha256", "G03 cohort")
    if cohort.get("candidate_bundle_sha256") != bundle_sha:
        raise ValueError("G03 cohort is not post-selection bound to this candidate bundle")
    if cohort.get("split") not in {"dev_optimize", "development"}:
        raise ValueError("G03 cohort must be development-only; gate/final scopes are forbidden")
    if cohort.get("seed_semantics") not in {
            "provider_seed_forwarded",
            "distinct_provider_requests_uncontrolled_sampling"}:
        raise ValueError("G03 cohort must declare whether seeds reach the provider")
    if any(cohort.get(field) is not False for field in (
            "official_test_read", "official_validation_read", "selection_gate_read",
            "final_holdout_read")):
        raise ValueError("G03 cohort records a protected evaluation scope read")
    exclusions = cohort.get("excluded_cohort_sha256s")
    if (not isinstance(exclusions, list)
            or not {source_cohort_sha, confirmation_cohort_sha}.issubset(
                {str(value) for value in exclusions})):
        raise ValueError("G03 cohort must exclude both candidate source and confirmation cohorts")
    rows = cohort.get("task_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("G03 cohort must contain non-empty frozen task rows")
    frozen_rows: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("task_id"), str) or not row["task_id"]:
            raise ValueError("G03 cohort rows require unique non-empty task IDs")
        frozen_rows.append({"task_id": row["task_id"],
                            "material_sha256": _require_sha(row.get("material_sha256"),
                                                             "task material_sha256")})
    if len({row["task_id"] for row in frozen_rows}) != len(frozen_rows):
        raise ValueError("G03 cohort contains duplicate task IDs")
    return cohort_sha, frozen_rows


def build_g03_ablation_plan(bundle: dict[str, Any], confirmation_report: dict[str, Any],
                            cohort: dict[str, Any], *,
                            candidate_bundle_file_sha256: str,
                            confirmation_report_file_sha256: str,
                            cohort_file_sha256: str,
                            target_model_id: str, target_protocol: str,
                            seeds: Sequence[int], logical_limit: int,
                            physical_limit: int, ledger_path: str,
                            bootstrap_replicates: int = 10000,
                            bootstrap_seed: int = 20261017) -> dict[str, Any]:
    """Rebuild a zero-call execution plan from exact candidate/cohort materials."""
    bundle_sha, members, interventions = _validate_bundle(bundle)
    bundle_file_sha = _require_sha(candidate_bundle_file_sha256,
                                   "candidate_bundle_file_sha256")
    confirmation_file_sha = _require_sha(confirmation_report_file_sha256,
                                         "confirmation_report_file_sha256")
    cohort_file_sha = _require_sha(cohort_file_sha256, "cohort_file_sha256")
    if not isinstance(target_model_id, str) or not target_model_id.strip():
        raise ValueError("target_model_id is required")
    if not isinstance(target_protocol, str) or not target_protocol.strip():
        raise ValueError("target_protocol is required")
    confirmation_plan_sha, source_cohort_sha, confirmation_cohort_sha = _validate_confirmation(
        confirmation_report, bundle=bundle, bundle_sha=bundle_sha, members=members,
        target_model_id=target_model_id, target_protocol=target_protocol)
    cohort_sha, rows = _validate_cohort(
        cohort, bundle_sha=bundle_sha, source_cohort_sha=source_cohort_sha,
        confirmation_cohort_sha=confirmation_cohort_sha)

    seed_values = list(seeds)
    if (len(seed_values) < 3 or any(type(seed) is not int for seed in seed_values)
            or len(set(seed_values)) != len(seed_values)):
        raise ValueError("G03 requires at least three distinct integer execution seeds")
    for value, name in ((logical_limit, "logical_limit"),
                        (physical_limit, "physical_limit"),
                        (bootstrap_replicates, "bootstrap_replicates")):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(bootstrap_seed) is not int:
        raise ValueError("bootstrap_seed must be an integer")
    minimum_logical = len(rows) * len(ARMS) * len(seed_values)
    if logical_limit < minimum_logical or physical_limit < logical_limit:
        raise ValueError("G03 ledger limits do not cover all paired arm/seed calls and retry capacity")
    ledger = Path(ledger_path)
    if not ledger.is_absolute():
        raise ValueError("G03 request ledger path must be absolute")
    if ledger.exists():
        raise FileExistsError("G03 request ledger path already exists; refusing reuse")

    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "protocol": PROTOCOL,
        "status": "valid_zero_call_preflight",
        "model_calls": 0,
        "candidate_bundle_id": bundle["candidate_bundle_id"],
        "candidate_bundle_sha256": bundle_sha,
        "candidate_bundle_file_sha256": bundle_file_sha,
        "candidate_members": members,
        "intervention_state_sha256": {
            arm: sha256_obj({"arm": arm, "parent_release_id": bundle["parent_release_id"],
                             **interventions[arm]}) for arm in ARMS},
        "arms": list(ARMS),
        "parent_release_id": bundle["parent_release_id"],
        "confirmation_report_sha256": confirmation_file_sha,
        "confirmation_plan_sha256": confirmation_plan_sha,
        "source_cohort_sha256": source_cohort_sha,
        "confirmation_cohort_sha256": confirmation_cohort_sha,
        "cohort_sha256": cohort_sha,
        "cohort_file_sha256": cohort_file_sha,
        "cohort_id": cohort.get("cohort_id"),
        "dataset": cohort.get("dataset"),
        "split": cohort.get("split"),
        "task_count": len(rows),
        "task_ids_sha256": sha256_obj([row["task_id"] for row in rows]),
        "task_materials_sha256": sha256_obj(rows),
        "target_model_id": target_model_id,
        "target_protocol": target_protocol,
        "seeds": seed_values,
        "seed_semantics": cohort.get("seed_semantics"),
        "protocol_source_sha256": _protocol_source_hashes(),
        "paired_statistics": {"method": "paired_task_bootstrap_percentile_v1",
                               "bootstrap_replicates": bootstrap_replicates,
                               "bootstrap_seed": bootstrap_seed,
                               "pairing_key": "(task_id, seed)",
                               "shared_sample_required": True},
        "request_ledger": {"path": str(ledger), "logical_limit": logical_limit,
                           "physical_limit": physical_limit,
                           "minimum_logical_operations": minimum_logical},
        "logical_limit": logical_limit,
        "physical_limit": physical_limit,
        "minimum_logical_operations": minimum_logical,
        "official_test_read": False,
        "official_validation_read": False,
        "selection_gate_read": False,
        "final_holdout_read": False,
        "plan_sha256": "",
    }
    plan["plan_sha256"] = sha256_obj({key: value for key, value in plan.items()
                                      if key != "plan_sha256"})
    return plan


def validate_g03_plan(plan: dict[str, Any]) -> None:
    """Check plan serialization/self-hash before a future executor is added."""
    if plan.get("schema") != PLAN_SCHEMA or plan.get("protocol") != PROTOCOL:
        raise ValueError("unsupported G03 plan schema/protocol")
    digest = _self_hashed(plan, "plan_sha256", "G03 plan")
    rebuilt_unsigned = dict(plan)
    rebuilt_unsigned.pop("plan_sha256")
    if sha256_obj(rebuilt_unsigned) != digest:
        raise ValueError("G03 plan serialization identity changed")
    if (plan.get("status") != "valid_zero_call_preflight"
            or plan.get("model_calls") != 0
            or plan.get("arms") != list(ARMS)
            or plan.get("official_test_read") is not False
            or plan.get("official_validation_read") is not False
            or plan.get("selection_gate_read") is not False
            or plan.get("final_holdout_read") is not False):
        raise ValueError("G03 plan violates the frozen zero-call/scope contract")
    if plan.get("protocol_source_sha256") != _protocol_source_hashes():
        raise ValueError("G03 protocol source changed after the plan was frozen")
