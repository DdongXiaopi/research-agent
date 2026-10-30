"""Paired discovery-only check for a preselected PubMedQA three-way prompt."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_pubmedqa_threeway_class_aware_search import _configure  # noqa: E402
from src.config import config_fingerprint, load_config  # noqa: E402
from src.contracts.types import Candidate, content_id, content_sha256  # noqa: E402
from src.data.split import load_tasks  # noqa: E402
from src.evolution.gepa_search import task_label_contract_conflict  # noqa: E402
from src.evolution.instruction_review import (  # noqa: E402
    PROMPT_GROUNDING_MAX_SPANS, PROMPT_GROUNDING_MAX_TOKENS,
    parse_prompt_grounding_review,
    prompt_grounding_messages, prompt_grounding_protocol,
    prompt_grounding_response_format, _model_strategy,
)
from src.eval.instruction_reviewer_rubric_v3 import split_source  # noqa: E402
from src.evolution.task_demo_search import _metrics, _nonregresses  # noqa: E402
from src.evolution.task_program import parse_task_program  # noqa: E402
from src.eval.task_rows import validate_task_answer_row  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.provider import build_provider  # noqa: E402
from src.runtime.call_ledger import call_component, summarize_ledger  # noqa: E402
from src.runtime.runner import ArtifactState, TaskRunner  # noqa: E402
from src.runtime.skills import with_body  # noqa: E402
from src.util import artifacts_path, ensure_dir, read_json, read_jsonl, sha256_file, sha256_obj, write_json  # noqa: E402
from scripts.pubmedqa_threeway_protocol import verify_dataset_snapshot  # noqa: E402


def validate_candidate_protocol(source_path: Path, source: dict,
                                candidate_record: dict) -> None:
    """Fail closed unless candidate and paired baseline share one frozen protocol."""
    verify_dataset_snapshot(source)
    partitions = {}
    for name in ("train_ids", "discovery_ids", "confirmation_ids"):
        raw_ids = source.get(name)
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError(f"frozen source must contain a nonempty {name} list")
        ids = [str(value) for value in raw_ids]
        if len(ids) != len(set(ids)):
            raise ValueError(f"frozen source {name} contain duplicate IDs")
        partitions[name] = set(ids)
    if (partitions["train_ids"] & partitions["discovery_ids"]
            or partitions["train_ids"] & partitions["confirmation_ids"]
            or partitions["discovery_ids"] & partitions["confirmation_ids"]):
        raise ValueError("frozen source train/discovery/confirmation partitions overlap")
    candidate_train_ids = candidate_record.get("train_ids")
    if (not isinstance(candidate_train_ids, list)
            or [str(value) for value in candidate_train_ids]
            != [str(value) for value in source["train_ids"]]):
        raise ValueError("candidate train IDs differ from the frozen source cohort")
    if candidate_record.get("discovery_ids_accessed") is not False:
        raise ValueError("candidate report does not certify discovery exclusion")
    if candidate_record.get("confirmation_ids_accessed") is not False:
        raise ValueError("candidate report does not certify confirmation exclusion")
    release_id = str(source.get("release_id") or "")
    if not release_id or str(candidate_record.get("release_id") or "") != release_id:
        raise ValueError("candidate and discovery source releases differ")
    routing_mode = str(source.get("skill_routing_mode", "lexical"))
    if candidate_record.get("skill_routing_mode") != routing_mode:
        raise ValueError("candidate and discovery source Skill routing modes differ")
    if candidate_record.get("source_report_sha256") != sha256_file(source_path):
        raise ValueError("candidate was generated from a different frozen source report")
    candidate = candidate_record.get("candidate") or {}
    artifact_type = str(candidate_record.get("target_artifact")
                        or candidate.get("artifact_type") or "")
    if artifact_type not in {"system_prompt", "skill_body", "task_program"}:
        raise ValueError("candidate target artifact is unsupported")
    if str(candidate.get("artifact_type") or "") != artifact_type:
        raise ValueError("candidate target artifact metadata is inconsistent")
    if str(candidate.get("parent_release_id") or "") != release_id:
        raise ValueError("candidate parent release differs from the discovery source")
    content = candidate.get("new_content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("candidate replacement content is empty")
    expected_content_sha = content_sha256(content)
    if candidate.get("content_sha256") != expected_content_sha:
        raise ValueError("candidate content hash does not match replacement content")
    identity_artifact = (str(candidate.get("artifact_id") or "")
                         if artifact_type == "skill_body" else artifact_type)
    expected_candidate_id = content_id(
        "cand", release_id, identity_artifact, content)
    if candidate.get("candidate_id") != expected_candidate_id:
        raise ValueError("candidate ID does not match its parent, artifact and content")
    if (candidate_record.get("candidate_id") is not None
            and candidate_record.get("candidate_id") != expected_candidate_id):
        raise ValueError("candidate report ID differs from embedded candidate identity")
    if (candidate_record.get("candidate_sha256") is not None
            and candidate_record.get("candidate_sha256") != expected_content_sha):
        raise ValueError("candidate report content hash differs from embedded content")


def apply_discovery_candidate(candidate: Candidate, state) -> ArtifactState:
    if candidate.artifact_type == "system_prompt":
        return ArtifactState(state.release_id, candidate.new_content,
                             state.registry, state.task_demos)
    if candidate.artifact_type == "skill_body":
        skill = state.registry.get(candidate.artifact_id)
        if skill is None or skill.frozen:
            raise ValueError("candidate target Skill is missing or frozen")
        return ArtifactState(state.release_id, state.system_prompt,
                             with_body(state.registry, candidate.artifact_id,
                                       candidate.new_content), state.task_demos)
    if candidate.artifact_type == "task_program":
        program = parse_task_program(candidate.new_content)
        return ArtifactState(state.release_id, program["system_prompt"],
                             state.registry, program["task_demos"])
    raise ValueError("candidate target artifact is unsupported")


def validate_paired_baseline_rows(rows: list[dict], run_record: dict,
                                  train_ids: set[str], discovery_ids: set[str],
                                  confirmation_ids: set[str],
                                  tasks_by_id: dict[str, dict]) -> dict[str, dict]:
    """Require one internally consistent baseline answer per declared task."""
    task_ids = [str(row.get("task_id") or "") for row in rows]
    present = set(task_ids)
    if not task_ids or len(task_ids) != len(present):
        raise ValueError("baseline per_task rows are empty or contain duplicate task IDs")
    if present & confirmation_ids:
        raise ValueError("baseline per_task rows contain confirmation IDs")
    includes_train = run_record.get("per_task_includes_train")
    if includes_train is True:
        expected = train_ids | discovery_ids
    elif includes_train is False:
        expected = discovery_ids
    else:
        # Older discovery-only reports predate the explicit scope marker. Infer
        # only one of the two documented scopes; reject any wider/ambiguous file.
        allowed = (discovery_ids, train_ids | discovery_ids)
        if present not in allowed:
            raise ValueError("baseline per_task scope is neither discovery-only nor train+discovery")
        expected = present
    if present != expected:
        raise ValueError("baseline per_task IDs do not match the declared paired scope")
    expected_count = run_record.get("run_task_count")
    if expected_count is not None and int(expected_count) != len(rows):
        raise ValueError("baseline run_task_count disagrees with per_task rows")
    indexed = {str(row["task_id"]): row for row in rows}
    validated_rows = {}
    for task_id in expected:
        task = tasks_by_id.get(task_id)
        if task is None:
            raise ValueError(f"baseline task is unavailable for answer verification: {task_id}")
        validated_rows[task_id] = validate_task_answer_row(
            indexed[task_id], task, context=f"baseline:{run_record.get('seed')}:{task_id}",
            require_reported_gold_index=True)
    return {task_id: validated_rows[task_id] for task_id in discovery_ids}


def run_prompt_grounding_review(prompt: str, cfg: dict,
                                artifact_type: str = "system_prompt",
                                task_contract=None) -> dict:
    """Use a measured configured router to reject factual prompt assertions before discovery."""
    strategy = _model_strategy(prompt, task_contract)
    candidate_span_count = len(split_source(strategy, "candidate"))
    if candidate_span_count > PROMPT_GROUNDING_MAX_SPANS:
        review = parse_prompt_grounding_review(
            "{}", prompt, task_contract, require_span_output=True)
        review.update({
            "passed": False,
            "schema_valid": False,
            "verdict": "invalid",
            "reason": "candidate_span_limit_exceeded",
            "candidate_span_count": candidate_span_count,
            "candidate_span_limit": PROMPT_GROUNDING_MAX_SPANS,
            "provider_call_made": False,
            "billable": False,
            "response_content": None,
            "token_usage": {"prompt": 0, "completion": 0},
            "cost": {},
            "review_protocol_sha256": sha256_obj(
                prompt_grounding_protocol(task_contract)),
            "candidate_sha256": content_sha256(str(prompt or "").strip()),
            "artifact_type": artifact_type,
            "assertion_scan_complete": False,
        })
        return review
    grounding_cfg = copy.deepcopy(cfg)
    router_provider = str((cfg.get("skills") or {}).get("router_provider", "http")).lower()
    if router_provider == "pi":
        raise ValueError("prompt grounding requires a measured non-Pi provider")
    grounding_cfg.setdefault("runtime", {})["provider"] = router_provider
    router_model = ((cfg.get("skills") or {}).get("instruction_review_model_id")
                    or (cfg.get("skills") or {}).get("router_model_id"))
    if router_model:
        grounding_cfg.setdefault("models", {}).setdefault("task", {})[
            "model_id"] = str(router_model)
    if router_provider == "http":
        task_cfg = grounding_cfg.setdefault("models", {}).setdefault("task", {})
        extra_body = copy.deepcopy(task_cfg.get("extra_body") or {})
        extra_body["response_format"] = prompt_grounding_response_format(task_contract)
        task_cfg["extra_body"] = extra_body
    provider = build_provider(grounding_cfg)
    if not hasattr(provider, "chat_measured"):
        raise ValueError("prompt grounding provider does not expose measured calls")
    system, user = prompt_grounding_messages(prompt, task_contract)
    with call_component("grounding"):
        result = provider.chat_measured(
            system, user, max_tokens=PROMPT_GROUNDING_MAX_TOKENS)
    review = parse_prompt_grounding_review(result.content, prompt, task_contract,
        require_span_output=True)
    review.update({
        "response_content": result.content,
        "review_protocol_sha256": sha256_obj(prompt_grounding_protocol(task_contract)),
        "scope": "pre-discovery candidate instruction artifact assertion scan",
        "artifact_type": artifact_type,
        "candidate_sha256": content_sha256(str(prompt or "").strip()),
        "task_contract_review": ({
            **review.get("task_contract_review", {}),
            "candidate_sha256": content_sha256(str(prompt or "").strip()),
        } if task_contract is not None else None),
        "method": "separate measured verifier call; candidate artifact only, no benchmark evidence supplied",
        "claim_level_evidence_verification": False,
        "candidate_span_count": candidate_span_count,
        "candidate_span_limit": PROMPT_GROUNDING_MAX_SPANS,
        "provider_call_made": True,
        "billable": True,
        "assertion_scan_complete": review["verdict"] in {"claim_free", "contains_claims"},
        "model_id": getattr(provider, "model_id", None),
        "token_usage": {"prompt": result.prompt_tokens,
                         "completion": result.completion_tokens},
        "cost": result.cost,
    })
    return review


def build_discovery_execution_plan(*, source_path: Path, candidate_report_path: Path,
                                   source: dict,
                                   candidate: Candidate, cfg: dict, state,
                                   static_gate_result: dict, dataset_hashes: tuple[str, str],
                                   discovery_ids: list[str], seeds: list[int]) -> dict:
    """Freeze the exact preselected discovery evaluation without provider calls."""
    runtime = cfg.get("runtime") or {}
    task_model = (cfg.get("models") or {}).get("task") or {}
    router_provider = str((cfg.get("skills") or {}).get(
        "router_provider", "http")).lower()
    if router_provider != "http":
        raise ValueError("frozen discovery grounding currently requires router_provider=http")
    max_model_calls = int(runtime.get("max_model_calls", 12))
    http_attempts = int(task_model.get("max_retries", 6))
    if max_model_calls < 1 or http_attempts < 1:
        raise ValueError("frozen discovery request bounds must be positive")
    # Keep the same semantics as the project MIPRO request planner: Pi retries
    # are retries-after-first, while bridge timeout attempts are total attempts;
    # HttpProvider.max_retries is also a total-attempt count.
    pi_retries = max(0, min(4, int(runtime.get("pi_provider_max_retries", 4))))
    pi_bridge_attempts = max(1, min(3, int(runtime.get("pi_timeout_max_attempts", 2))))
    pi_attempts_per_operation = (pi_retries + 1) * pi_bridge_attempts
    task_attempts_per_operation = max(pi_attempts_per_operation, http_attempts)
    task_runs = len(discovery_ids) * len(seeds)
    max_task_operations = task_runs * max_model_calls
    logical_upper = 1 + max_task_operations  # one grounding review plus TaskRunner calls
    physical_upper = (max_task_operations * task_attempts_per_operation
                      + http_attempts)  # grounding uses the measured HTTP provider
    if not discovery_ids or len(discovery_ids) != len(set(discovery_ids)):
        raise ValueError("discovery plan needs unique task IDs")
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError("discovery plan needs at least three distinct execution labels")
    unsigned = {
        "schema": "pubmedqa-prompt-discovery-execution-plan-v1",
        "role": "preselected_candidate_internal_discovery_only",
        "source_report_sha256": sha256_file(source_path),
        "candidate_report_sha256": sha256_file(candidate_report_path),
        "candidate_id": candidate.candidate_id,
        "candidate_artifact_type": candidate.artifact_type,
        "candidate_sha256": candidate.content_sha256,
        "parent_release_id": state.release_id,
        "registry_sha256": state.registry.fingerprint(),
        "config_sha256": config_fingerprint(cfg),
        "task_provider": str(runtime.get("provider") or ""),
        "task_model_id": str(task_model.get("model_id") or ""),
        "grounding_provider": router_provider,
        "grounding_model_id": str((cfg.get("skills") or {}).get(
            "router_model_id") or task_model.get("model_id") or ""),
        "skill_routing_mode": str((cfg.get("skills") or {}).get(
            "routing_mode") or "lexical"),
        "dataset": str(source.get("dataset") or ""),
        "dataset_manifest_sha256": dataset_hashes[0],
        "dataset_normalized_sha256": dataset_hashes[1],
        "train_ids": [str(value) for value in source.get("train_ids") or []],
        "discovery_ids": list(discovery_ids),
        "confirmation_ids": [str(value) for value in source.get("confirmation_ids") or []],
        "execution_labels": list(seeds),
        "candidate_static_gate_sha256": sha256_obj(static_gate_result),
        "candidate_static_gate_passed": static_gate_result.get("passed") is True,
        "review_protocol_sha256": sha256_obj(prompt_grounding_protocol()),
        "grounding_max_tokens": PROMPT_GROUNDING_MAX_TOKENS,
        "task_runner_count_upper_bound": task_runs,
        "model_operations_per_task_runner_upper_bound": max_model_calls,
        "grounding_model_operations_upper_bound": 1,
        "http_max_attempts_per_operation": http_attempts,
        "pi_provider_retries_after_first": pi_retries,
        "pi_timeout_subprocess_attempts": pi_bridge_attempts,
        "pi_provider_http_attempts_per_operation_upper_bound": pi_attempts_per_operation,
        "taskrunner_http_attempts_per_operation_upper_bound": task_attempts_per_operation,
        "logical_request_cap": logical_upper,
        "physical_request_cap": physical_upper,
        "request_ledger_required": True,
        "pi_physical_reservations_required": True,
        "selection_gate_read": False,
        "final_holdout_read": False,
        "confirmation_ids_read": False,
        "retrieval_used": False,
        "model_calls_during_preflight": 0,
        "live_authorized": False,
        "explicit_user_approval_required_before_execution": True,
        "independent_confirmation_eligible": False,
        "candidate_efficacy_evidence": False,
    }
    return {**unsigned, "plan_sha256": sha256_obj(unsigned)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True,
                        help="v5 exploratory report with disjoint train/discovery/confirmation IDs")
    parser.add_argument("--candidate-report", required=True,
                        help="completed train-only GEPA report containing one selected candidate")
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--skill-routing-mode", choices=["lexical", "model"],
                        default=None)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[202610051, 202610052, 202610053])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true",
                      help="freeze and write the zero-call plan")
    mode.add_argument("--execute", action="store_true",
                      help="execute only the exact approved plan supplied below")
    parser.add_argument("--expected-plan-sha256", default="",
                        help="required with --execute; must match the recomputed frozen plan")
    args = parser.parse_args()
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("discovery evaluation requires an explicit isolated RSI_HOME")
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("provide at least three distinct execution labels")

    source_path = Path(args.source_report).resolve()
    candidate_report_path = Path(args.candidate_report).resolve()
    source, candidate_record = read_json(source_path), read_json(candidate_report_path)
    dataset_hashes = verify_dataset_snapshot(source)
    validate_candidate_protocol(source_path, source, candidate_record)
    selected_candidate = Candidate(**{
        key: value for key, value in (candidate_record.get("candidate") or {}).items()
        if key in Candidate.__annotations__})
    if not selected_candidate.new_content.strip():
        raise ValueError("candidate report has no replacement artifact")
    discovery_ids = [str(x) for x in source.get("discovery_ids") or []]
    train_ids = set(str(x) for x in source.get("train_ids") or [])
    confirmation_ids = set(str(x) for x in source.get("confirmation_ids") or [])
    if len(discovery_ids) < 3 or len(set(discovery_ids)) != len(discovery_ids):
        raise ValueError("source report must contain unique discovery IDs")
    if set(discovery_ids) & (train_ids | confirmation_ids):
        raise ValueError("discovery IDs overlap train or confirmation IDs")
    if candidate_record.get("discovery_ids_accessed") is not False:
        raise ValueError("candidate must come from a train-only search report")

    train_tasks = {str(task["task_id"]): task for task in
                     load_tasks(str(source["dataset"]), splits=["dev_optimize"])
                     if str(task["task_id"]) in train_ids}
    if set(train_tasks) != train_ids:
        raise ValueError("one or more train IDs are missing from dev_optimize")
    candidate_instructions = (
        parse_task_program(selected_candidate.new_content)["system_prompt"]
        if selected_candidate.artifact_type == "task_program"
        else selected_candidate.new_content)
    if selected_candidate.artifact_type == "task_program":
        program = parse_task_program(selected_candidate.new_content)
        demo_ids = {str(row["task_id"]) for row in program["task_demos"]}
        if not demo_ids or not demo_ids <= train_ids:
            raise ValueError("task-program demonstrations must come only from train IDs")
        if demo_ids & (set(discovery_ids) | confirmation_ids):
            raise ValueError("task-program demonstrations overlap discovery or confirmation")
    label_conflict = task_label_contract_conflict(
        candidate_instructions, list(train_tasks.values()))
    if label_conflict:
        raise ValueError("candidate failed train-only task-label grounding gate; discovery not accessed: "
                         + label_conflict)

    cfg = _configure(args.config)
    routing_mode = args.skill_routing_mode or str(
        source.get("skill_routing_mode", "lexical"))
    if routing_mode not in {"lexical", "model"}:
        raise ValueError("source report has an invalid skill_routing_mode")
    if candidate_record.get("skill_routing_mode") != routing_mode:
        raise ValueError("candidate and discovery Skill routing modes differ")
    cfg.setdefault("skills", {})["routing_mode"] = routing_mode
    cfg["evolution"]["allowed_artifacts"] = [selected_candidate.artifact_type]
    cfg["evolution"]["fewshot"].update({
        "enabled": selected_candidate.artifact_type == "task_program",
        "k": 3 if selected_candidate.artifact_type == "task_program" else 0,
    })
    cfg["evolution"].setdefault("gepa", {})["visible_task_ids"] = list(
        candidate_record.get("train_ids") or [])
    cfg["evaluation"]["insufficient_option"] = False
    state = load_state(source["release_id"])
    gate = static_gate(selected_candidate, cfg, state)
    if not gate.get("passed"):
        raise RuntimeError("candidate failed static gate; discovery not accessed")

    plan = build_discovery_execution_plan(
        source_path=source_path, candidate_report_path=candidate_report_path,
        source=source,
        candidate=selected_candidate, cfg=cfg, state=state,
        static_gate_result=gate, dataset_hashes=dataset_hashes,
        discovery_ids=discovery_ids, seeds=args.seeds)
    plan_path = artifacts_path(
        "probe", "pubmedqa_threeway_v4",
        "discovery-plan-" + plan["plan_sha256"] + ".json")
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError("saved discovery plan differs from the recomputed plan")
    else:
        write_json(plan_path, plan)
    if args.validate_only:
        print(json.dumps({
            "status": "validated_not_run", "plan": str(plan_path),
            "plan_sha256": plan["plan_sha256"],
            "task_runs": plan["task_runner_count_upper_bound"],
            "logical_request_cap": plan["logical_request_cap"],
            "physical_request_cap": plan["physical_request_cap"],
            "model_calls": 0, "selection_gate_read": False,
            "final_holdout_read": False,
        }, ensure_ascii=False))
        return 0
    if args.expected_plan_sha256 != plan["plan_sha256"]:
        raise ValueError("expected plan hash does not match the current discovery plan")

    run_id = "pqa-prompt-discovery-" + uuid.uuid4().hex[:12]
    run_dir = artifacts_path("probe", "pubmedqa_threeway_v4", run_id)
    ensure_dir(run_dir)
    ledger_path = run_dir / "call-ledger.jsonl"
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(plan["plan_sha256"])
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path)
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(plan["logical_request_cap"])
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(plan["physical_request_cap"])
    os.environ["RSI_CALL_LEDGER_MAX_COMPONENT_GROUNDING"] = "1"
    os.environ["RSI_CALL_LEDGER_RESERVE_PI_PHYSICAL"] = "1"

    grounding_review = run_prompt_grounding_review(
        candidate_instructions, cfg, selected_candidate.artifact_type)
    grounding_review_path = artifacts_path(
        "probe", "pubmedqa_threeway_v4",
        "prompt-grounding-" + uuid.uuid4().hex[:12] + ".json")
    ensure_dir(grounding_review_path.parent)
    write_json(grounding_review_path, {
        "protocol": "prompt-claim-free-grounding-gate-v1",
        "source_report_sha256": sha256_file(source_path),
        "candidate_report_sha256": sha256_file(candidate_report_path),
        "candidate_id": selected_candidate.candidate_id,
        "candidate_sha256": selected_candidate.content_sha256,
        "execution_plan_sha256": plan["plan_sha256"],
        "execution_plan_path": str(plan_path),
        "candidate_static_gate": gate,
        "grounding_review": grounding_review,
        "discovery_labels_loaded": False,
        "request_ledger": summarize_ledger(ledger_path),
    })
    grounding_review["artifact"] = str(grounding_review_path)
    if not grounding_review["passed"]:
        raise RuntimeError("candidate failed or did not resolve the pre-discovery prompt grounding gate; discovery not accessed: "
                           + str(grounding_review["reason"]))

    # Only after static and prompt-grounding gates pass do we load discovery labels.
    discovery_tasks = {str(task["task_id"]): task for task in
                       load_tasks(str(source["dataset"]), splits=["dev_optimize"])
                       if str(task["task_id"]) in set(discovery_ids)}
    if set(discovery_tasks) != set(discovery_ids):
        raise ValueError("one or more discovery IDs are missing from dev_optimize")
    tasks = discovery_tasks
    paired_tasks = {**train_tasks, **discovery_tasks}

    baseline_source = source
    if not source.get("baseline_runs") and source.get("source_run_report"):
        baseline_source = read_json(Path(source["source_run_report"]))
    for partition in ("train_ids", "discovery_ids", "confirmation_ids"):
        if partition in baseline_source and partition in source:
            if {str(x) for x in baseline_source.get(partition) or []} != {
                    str(x) for x in source.get(partition) or []}:
                raise ValueError(f"baseline source {partition} differ from frozen source report")
    baseline_run_records = {int(item["seed"]): item
                            for item in baseline_source.get("baseline_runs") or []}
    if len(baseline_run_records) != len(baseline_source.get("baseline_runs") or []):
        raise ValueError("source baseline contains duplicate seed records")
    if any(seed not in baseline_run_records for seed in args.seeds):
        raise ValueError("source report lacks paired baseline run for every seed")
    baseline_by_seed = {}
    for seed in args.seeds:
        record = baseline_run_records[seed]
        rows = read_jsonl(artifacts_path(
            "runs", str(record["run_id"]), "per_task.jsonl"))
        baseline_by_seed[seed] = validate_paired_baseline_rows(
            rows, record, train_ids, set(discovery_ids), confirmation_ids,
            paired_tasks)
        if set(baseline_by_seed[seed]) != set(discovery_ids):
            raise ValueError(f"baseline run {seed} does not cover discovery IDs")

    checkpoint_path = run_dir / "checkpoint.jsonl"
    saved = {}
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                key = (int(row["seed"]), str(row["task_id"]))
                if key in saved:
                    raise ValueError(f"candidate checkpoint contains duplicate row: {key}")
                if key[0] not in args.seeds or key[1] not in tasks:
                    raise ValueError(f"candidate checkpoint contains out-of-scope row: {key}")
                if row.get("candidate_id") != selected_candidate.candidate_id:
                    raise ValueError(f"candidate checkpoint identity mismatch: {key}")
                if row.get("execution_plan_sha256") != plan["plan_sha256"]:
                    raise ValueError(f"candidate checkpoint plan mismatch: {key}")
                try:
                    saved[key] = validate_task_answer_row(
                        row, tasks[key[1]], context=f"candidate-checkpoint:{key[0]}:{key[1]}",
                        require_reported_gold_index=True)
                except ValueError as exc:
                    invalid_report = {
                        "protocol": "pubmedqa-threeway-candidate-discovery-v2-exploratory",
                        "status": "invalid_candidate_evaluation",
                        "source_report": str(source_path),
                        "source_report_sha256": sha256_file(source_path),
                        "candidate_report": str(candidate_report_path),
                        "candidate_report_sha256": sha256_file(candidate_report_path),
                        "candidate_id": selected_candidate.candidate_id,
                        "invalid_task_id": key[1], "execution_label": key[0],
                        "validation_error": str(exc),
                        "candidate_checkpoint": str(checkpoint_path),
                        "execution_plan_sha256": plan["plan_sha256"],
                        "execution_plan_path": str(plan_path),
                        "request_ledger": summarize_ledger(ledger_path),
                        "confirmation_ids_accessed": False,
                        "eligible_for_internal_confirmation": False,
                        "eligible_for_independent_confirmation": False,
                        "published": False,
                    }
                    write_json(run_dir / "report.json", invalid_report)
                    print(json.dumps({"report": str(run_dir / "report.json"),
                                      "status": invalid_report["status"],
                                      "task_id": key[1], "seed": key[0],
                                      "validation_error": str(exc)}, ensure_ascii=False))
                    return 2

    candidate_state = apply_discovery_candidate(selected_candidate, state)
    for seed in args.seeds:
        provider = build_provider(cfg, seed=seed)
        for task_id in discovery_ids:
            key = (seed, task_id)
            if key in saved and saved[key].get("status") == "ok":
                continue
            runner = TaskRunner(cfg, candidate_state, provider=provider, seed=seed)
            outcome = runner.run(tasks[task_id])
            row = outcome.to_dict()
            row.update({
                "seed": seed,
                "gold_label": str(tasks[task_id]["options"][
                    tasks[task_id]["correct_index"]]).strip().casefold(),
                "candidate_id": selected_candidate.candidate_id,
                "config_sha256": config_fingerprint(cfg),
                "execution_plan_sha256": plan["plan_sha256"],
            })
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            saved[key] = row
            try:
                saved[key] = validate_task_answer_row(
                    row, tasks[task_id], context=f"candidate:{seed}:{task_id}",
                    require_reported_gold_index=True)
            except ValueError as exc:
                invalid_report = {
                    "protocol": "pubmedqa-threeway-candidate-discovery-v2-exploratory",
                    "status": "invalid_candidate_evaluation",
                    "source_report": str(source_path),
                    "source_report_sha256": sha256_file(source_path),
                    "candidate_report": str(candidate_report_path),
                    "candidate_report_sha256": sha256_file(candidate_report_path),
                    "candidate_id": selected_candidate.candidate_id,
                    "invalid_task_id": task_id, "execution_label": seed,
                    "validation_error": str(exc),
                    "candidate_checkpoint": str(checkpoint_path),
                    "execution_plan_sha256": plan["plan_sha256"],
                    "execution_plan_path": str(plan_path),
                    "request_ledger": summarize_ledger(ledger_path),
                    "confirmation_ids_accessed": False,
                    "eligible_for_internal_confirmation": False,
                    "eligible_for_independent_confirmation": False,
                    "published": False,
                }
                write_json(run_dir / "report.json", invalid_report)
                print(json.dumps({"report": str(run_dir / "report.json"),
                                  "status": invalid_report["status"],
                                  "task_id": task_id, "seed": seed,
                                  "validation_error": str(exc)}, ensure_ascii=False))
                return 2

    by_seed = {}
    for seed in args.seeds:
        candidate_rows = []
        for task_id in discovery_ids:
            row = saved[(seed, task_id)]
            base = baseline_by_seed[seed][task_id]
            candidate_rows.append({
                "task_id": task_id,
                "gold_label": row["gold_label"],
                "answer_index": int(row.get("answer_index", -1)),
                "correct": bool(row.get("correct")),
                "status": row.get("status"),
                "baseline_answer_index": int(base.get("answer_index", -1)),
                "baseline_correct": bool(base.get("correct")),
            })
        baseline_rows = [{
            "task_id": task_id,
            "gold_label": str(tasks[task_id]["options"][tasks[task_id]["correct_index"]])
                .strip().casefold(),
            "answer_index": int(baseline_by_seed[seed][task_id].get("answer_index", -1)),
            "correct": bool(baseline_by_seed[seed][task_id].get("correct")),
        } for task_id in discovery_ids]
        candidate_metrics = _metrics(candidate_rows)
        baseline_metrics = _metrics(baseline_rows)
        by_seed[str(seed)] = {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "nonregresses": _nonregresses(candidate_metrics, baseline_metrics),
            "paired_rows": candidate_rows,
        }
    gate_passed = all(row["nonregresses"] for row in by_seed.values())
    baseline_accuracy = sum(row["baseline"]["accuracy"] for row in by_seed.values()) / len(by_seed)
    candidate_accuracy = sum(row["candidate"]["accuracy"] for row in by_seed.values()) / len(by_seed)
    strict_accuracy_gain = candidate_accuracy > baseline_accuracy + 1e-9
    report = {
        "protocol": "pubmedqa-threeway-candidate-discovery-v2-exploratory",
        "source_report": str(source_path),
        "source_report_sha256": sha256_file(source_path),
        "candidate_report": str(candidate_report_path),
        "candidate_report_sha256": sha256_file(candidate_report_path),
        "candidate_id": selected_candidate.candidate_id,
        "candidate_sha256": selected_candidate.content_sha256,
        "execution_plan_sha256": plan["plan_sha256"],
        "execution_plan_path": str(plan_path),
        "request_ledger_path": str(ledger_path),
        "request_ledger": summarize_ledger(ledger_path),
        "target_artifact": selected_candidate.artifact_type,
        "target_skill": (selected_candidate.artifact_id
                         if selected_candidate.artifact_type == "skill_body" else None),
        "candidate_static_gate": gate,
        "candidate_grounding_review": grounding_review,
        "dataset": source["dataset"],
        "discovery_ids": discovery_ids,
        "train_ids_accessed": False,
        "confirmation_ids_accessed": False,
        "selection_gate_accessed": False,
        "final_holdout_accessed": False,
        "retrieval_used": False,
        "synthetic_insufficient_option": False,
        "release_id": state.release_id,
        "skill_routing_mode": routing_mode,
        "execution_labels": list(args.seeds),
        "model_rng_seed_forwarded": False,
        "results_by_seed": by_seed,
        "all_seed_class_aware_nonregression": gate_passed,
        "mean_baseline_accuracy": baseline_accuracy,
        "mean_candidate_accuracy": candidate_accuracy,
        "strict_primary_accuracy_improvement": strict_accuracy_gain,
        "eligible_for_internal_confirmation": gate_passed and strict_accuracy_gain,
        "eligible_for_independent_confirmation": False,
        "plan_hash_verified": True,
        "approval_attestation": "out-of-band; not verified by this runner",
        "published": False,
        "candidate_checkpoint": str(checkpoint_path),
    }
    output_path = run_dir / "report.json"
    write_json(output_path, report)
    print(json.dumps({
        "report": str(output_path),
        "candidate_id": selected_candidate.candidate_id,
        "all_seed_class_aware_nonregression": gate_passed,
        "strict_primary_accuracy_improvement": strict_accuracy_gain,
        "metrics": {seed: {
            "baseline_accuracy": row["baseline"]["accuracy"],
            "candidate_accuracy": row["candidate"]["accuracy"],
            "baseline_per_class": {label: value["accuracy"] for label, value
                                   in row["baseline"]["per_gold_label"].items()},
            "candidate_per_class": {label: value["accuracy"] for label, value
                                    in row["candidate"]["per_gold_label"].items()},
            "transitions": {
                "improved": sum(not item["baseline_correct"] and item["correct"]
                                for item in row["paired_rows"]),
                "regressed": sum(item["baseline_correct"] and not item["correct"]
                                 for item in row["paired_rows"]),
            },
        } for seed, row in by_seed.items()},
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
