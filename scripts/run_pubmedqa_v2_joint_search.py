"""Run a frozen-cohort GEPA × few-shot task-program search (exploratory only)."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_pubmedqa_prompt_discovery import (  # noqa: E402
    run_prompt_grounding_review,
)
from scripts.pubmedqa_threeway_protocol import verify_dataset_snapshot  # noqa: E402
from src.config import config_fingerprint, load_config  # noqa: E402
from src.contracts.types import Candidate, content_id, content_sha256  # noqa: E402
from src.data.split import load_tasks  # noqa: E402
from src.evolution.gepa_search import visible_examples  # noqa: E402
from src.evolution.instruction_review import prompt_grounding_protocol  # noqa: E402
from src.evolution.task_demo_search import (  # noqa: E402
    search_task_demo_candidate, split_demo_validation,
)
from src.evolution.task_program import parse_task_program  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.util import (artifacts_path, read_json, read_jsonl, sha256_file,
                      sha256_obj, write_json)  # noqa: E402

JOINT_SEARCH_PROTOCOL = "pubmedqa-threeway-v2-train-only-gepa-fewshot-joint-v2"


def validate_inputs(source_path: Path, prompt_path: Path,
                    source: dict, prompt_report: dict
                    ) -> tuple[Candidate | None, list[str]]:
    verify_dataset_snapshot(source)
    if source.get("protocol") != "pubmedqa-threeway-v2-skill-routing-baseline-v2":
        raise ValueError("source report is not a v2 train-plus-discovery baseline")
    if prompt_report.get("source_report_sha256") != sha256_file(source_path):
        raise ValueError("Prompt candidate and baseline use different frozen source reports")
    if prompt_report.get("release_id") != source.get("release_id"):
        raise ValueError("Prompt candidate and baseline releases differ")
    if prompt_report.get("skill_routing_mode") != source.get("skill_routing_mode"):
        raise ValueError("Prompt candidate and baseline Skill routing modes differ")
    if prompt_report.get("target_artifact") != "system_prompt":
        raise ValueError("train-only Prompt report targets a different artifact")
    if prompt_report.get("discovery_ids_accessed") is not False or \
            prompt_report.get("confirmation_ids_accessed") is not False:
        raise ValueError("Prompt candidate report is not train-only")
    train_ids = [str(x) for x in source.get("train_ids") or []]
    discovery_ids = set(str(x) for x in source.get("discovery_ids") or [])
    confirmation_ids = set(str(x) for x in source.get("confirmation_ids") or [])
    if (set(train_ids) & (discovery_ids | confirmation_ids)
            or discovery_ids & confirmation_ids):
        raise ValueError("frozen train/discovery/confirmation partitions overlap")
    required = set(train_ids) | discovery_ids
    if source.get("confirmation_labels_accessed") is not False:
        raise ValueError("baseline source does not attest confirmation-label isolation")
    if source.get("train_answers_accessed") is not True:
        raise ValueError("joint search requires explicitly measured train baselines")
    evaluated_ids = [str(x) for x in source.get("evaluated_ids") or []]
    if len(evaluated_ids) != len(set(evaluated_ids)) or set(evaluated_ids) != required:
        raise ValueError("baseline evaluated IDs differ from frozen train plus discovery")
    if prompt_report.get("train_ids") != train_ids:
        raise ValueError("Prompt candidate train IDs differ from the frozen cohort")
    candidate_data = prompt_report.get("candidate")
    candidate = (Candidate(**{
        key: value for key, value in candidate_data.items()
        if key in Candidate.__annotations__}) if isinstance(candidate_data, dict) else None)
    if candidate is None:
        if prompt_report.get("candidate_status") not in {"no_candidate", "no_change"}:
            raise ValueError("Prompt report has no candidate but does not record a no-change result")
    elif candidate.artifact_type != "system_prompt" or not candidate.new_content.strip():
        raise ValueError("joint search requires a non-empty system_prompt candidate")
    baseline_runs = source.get("baseline_runs") or []
    seen_seeds = set()
    run_seeds = []
    for item in baseline_runs:
        seed = int(item["seed"])
        if seed in seen_seeds:
            raise ValueError("baseline report contains duplicate execution labels")
        seen_seeds.add(seed)
        run_seeds.append(seed)
        if not item.get("per_task_includes_train"):
            raise ValueError("joint search requires train-plus-discovery baseline per_task runs")
        if int(item.get("run_task_count", -1)) != len(required):
            raise ValueError("baseline run task count does not equal train plus discovery")
    if len(baseline_runs) < 3:
        raise ValueError("joint search requires at least three paired baseline seeds")
    if run_seeds != [int(seed) for seed in source.get("execution_labels") or []]:
        raise ValueError("baseline runs do not match source execution labels")
    return candidate, train_ids


def validate_baseline_task_rows(source: dict, train_ids: list[str]) -> None:
    """Inspect paired rows only after the Prompt candidate passes both pre-search gates."""
    required = set(train_ids) | set(str(x) for x in source.get("discovery_ids") or [])
    confirmation_ids = set(str(x) for x in source.get("confirmation_ids") or [])
    for item in source.get("baseline_runs") or []:
        rows = read_jsonl(artifacts_path("runs", str(item["run_id"]), "per_task.jsonl"))
        task_ids = [str(row.get("task_id") or "") for row in rows]
        present = set(task_ids)
        if len(task_ids) != len(present) or present != required:
            raise ValueError(
                f"baseline run {item['run_id']} must contain exactly train plus discovery rows")
        if present & confirmation_ids:
            raise ValueError(f"baseline run {item['run_id']} contains confirmation rows")


def configure(source: dict, config_path: str, train_ids: list[str],
              max_metric_calls: int, max_candidate_sets: int) -> dict:
    cfg = copy.deepcopy(load_config(config_path))
    cfg.setdefault("runtime", {}).update({
        "provider": "pi", "input_mode": "task_input", "task_input_max_chars": 30000,
        "pi_request_cooldown_seconds": 6,
    })
    cfg["runtime"].pop("pi_fake_response", None)
    cfg.setdefault("evaluation", {}).update({
        "insufficient_option": False, "main_metric": "accuracy",
        "primary_axis": "accuracy", "sealed_pairing": False,
    })
    cfg.setdefault("skills", {})["routing_mode"] = str(source["skill_routing_mode"])
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg.get("tools", {}).get("active", [])
        if name != "academic.search"]
    cfg.setdefault("evolution", {})["allowed_artifacts"] = ["task_program"]
    cfg["evolution"].setdefault("gepa", {}).update({
        "visible_task_ids": train_ids, "stratify_by_gold_label": True,
    })
    cfg["evolution"].setdefault("fewshot", {}).update({
        "enabled": True, "k": 3, "max_candidate_sets": max_candidate_sets,
        "max_metric_calls": max_metric_calls,
        "baseline_allowed_task_ids": (
            [*train_ids, *[str(x) for x in source.get("discovery_ids") or []]]),
    })
    return cfg


def build_joint_search_execution_plan(*, source_path: Path, prompt_path: Path,
                                      source: dict,
                                      prompt_candidate: Candidate | None,
                                      cfg: dict, state, static_gate_result: dict,
                                      train_ids: list[str], examples: list[dict],
                                      validation: list[dict], metric_calls: int,
                                      candidate_sets: int, seed: int) -> dict:
    """Freeze the exact visible joint-search scope and request ceilings offline."""
    runtime = cfg.get("runtime") or {}
    task_model = (cfg.get("models") or {}).get("task") or {}
    logical_per_runner = int(runtime.get("max_model_calls", 0))
    pi_retries = max(0, min(4, int(runtime.get("pi_provider_max_retries", 4))))
    pi_bridge_attempts = max(1, min(3, int(runtime.get("pi_timeout_max_attempts", 1))))
    pi_attempts = (pi_retries + 1) * pi_bridge_attempts
    http_attempts = max(1, int(task_model.get("max_retries", 1)))
    if logical_per_runner <= 0:
        raise ValueError("joint search requires a positive TaskRunner model-call cap")
    search_validation, internal_validation = split_demo_validation(validation)
    if not examples or not search_validation or not internal_validation:
        raise ValueError("joint search plan requires non-empty disjoint train/search/inner-validation scopes")
    if (set(str(row["task_id"]) for row in examples)
            & set(str(row["task_id"]) for row in validation)):
        raise ValueError("joint search train and validation task IDs overlap")
    task_operation_cap = int(metric_calls) * logical_per_runner
    logical_cap = task_operation_cap + 1  # one measured prompt-grounding operation
    task_attempts = max(pi_attempts, http_attempts)
    physical_cap = task_operation_cap * task_attempts + http_attempts
    instruction_variants = int(
        prompt_candidate is not None
        and prompt_candidate.new_content != state.system_prompt) + 1
    baseline_seeds = sorted(int(item["seed"]) for item in source.get("baseline_runs") or [])
    if not baseline_seeds or len(baseline_seeds) != len(set(baseline_seeds)):
        raise ValueError("joint search needs unique frozen baseline execution labels")
    dataset_manifest_sha256, dataset_normalized_sha256 = verify_dataset_snapshot(source)
    baseline_run_hashes = {}
    for item in source.get("baseline_runs") or []:
        run_id = str(item["run_id"])
        rows_path = artifacts_path("runs", run_id, "per_task.jsonl")
        if not rows_path.is_file():
            raise ValueError(f"frozen baseline per-task rows are missing for {run_id}")
        baseline_run_hashes[str(int(item["seed"]))] = sha256_file(rows_path)
    unsigned = {
        "schema": "pubmedqa-v2-joint-search-execution-plan-v1",
        "protocol": JOINT_SEARCH_PROTOCOL,
        "source_report_sha256": sha256_file(source_path),
        "prompt_candidate_report_sha256": sha256_file(prompt_path),
        "source_dataset": str(source.get("dataset") or ""),
        "source_manifest_sha256": dataset_manifest_sha256,
        "source_normalized_sha256": dataset_normalized_sha256,
        "prompt_candidate_id": prompt_candidate.candidate_id if prompt_candidate else None,
        "prompt_candidate_sha256": (content_sha256(prompt_candidate.new_content)
                                     if prompt_candidate else sha256_obj(state.system_prompt)),
        "prompt_candidate_artifact_type": (prompt_candidate.artifact_type
                                            if prompt_candidate else "system_prompt"),
        "prompt_static_gate_sha256": sha256_obj(static_gate_result),
        "prompt_static_gate_passed": static_gate_result.get("passed") is True,
        "prompt_grounding_protocol_sha256": sha256_obj(
            prompt_grounding_protocol()),
        "parent_release_id": state.release_id,
        "registry_sha256": state.registry.fingerprint(),
        "config_sha256": config_fingerprint(cfg),
        "skill_routing_mode": str(source["skill_routing_mode"]),
        "baseline_execution_labels": baseline_seeds,
        "baseline_per_task_sha256_by_label": baseline_run_hashes,
        "search_seed": int(seed),
        "train_ids": sorted(str(row["task_id"]) for row in examples),
        "outer_validation_ids": sorted(str(row["task_id"]) for row in validation),
        "task_demo_search_validation_ids": sorted(
            str(row["task_id"]) for row in search_validation),
        "task_demo_inner_validation_ids": sorted(
            str(row["task_id"]) for row in internal_validation),
        "source_confirmation_ids_read": False,
        "selection_gate_read": False,
        "final_holdout_read": False,
        "official_test_read": False,
        "retrieval_used": False,
        "candidate_artifact_type": "task_program",
        "instruction_variants_upper_bound": instruction_variants,
        "candidate_set_upper_bound": int(candidate_sets),
        "metric_call_budget": int(metric_calls),
        "task_runner_model_calls_per_metric_upper_bound": logical_per_runner,
        "task_runner_count_upper_bound": int(metric_calls),
        "grounding_model_operation_upper_bound": 1,
        "request_budget_semantics": {
            "logical": "one provider operation",
            "physical": "one dispatched HTTP attempt, including failed attempts",
        },
        "provider_attempts": {
            "pi_retries_after_first": pi_retries,
            "pi_bridge_total_attempts": pi_bridge_attempts,
            "pi_http_attempts_per_operation": pi_attempts,
            "http_total_attempts_per_operation": http_attempts,
            "task_operation_attempt_upper_bound": task_attempts,
        },
        "logical_request_cap": logical_cap,
        "physical_request_cap": physical_cap,
        "request_ledger_required": True,
        "pi_physical_reservations_required": True,
        "model_calls_during_preflight": 0,
        "live_authorized": False,
        "explicit_user_approval_required_before_execution": True,
        "confirmation_eligible": False,
        "candidate_efficacy_evidence": False,
    }
    if not unsigned["prompt_static_gate_passed"]:
        raise ValueError("prompt candidate failed static gate before joint search plan")
    return {**unsigned, "plan_sha256": sha256_obj(unsigned)}


@contextmanager
def request_ledger_environment(plan: dict, ledger_path: Path):
    names = (
        "RSI_ACTIVE_EXECUTION_PLAN_SHA256", "RSI_CALL_LEDGER_PATH", "RSI_CALL_LEDGER_MAX_LOGICAL",
        "RSI_CALL_LEDGER_MAX_PHYSICAL", "RSI_CALL_LEDGER_MAX_COMPONENT_GROUNDING",
        "RSI_CALL_LEDGER_RESERVE_PI_PHYSICAL",
    )
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({
        "RSI_ACTIVE_EXECUTION_PLAN_SHA256": str(plan["plan_sha256"]),
        "RSI_CALL_LEDGER_PATH": str(ledger_path),
        "RSI_CALL_LEDGER_MAX_LOGICAL": str(plan["logical_request_cap"]),
        "RSI_CALL_LEDGER_MAX_PHYSICAL": str(plan["physical_request_cap"]),
        "RSI_CALL_LEDGER_MAX_COMPONENT_GROUNDING": "1",
        "RSI_CALL_LEDGER_RESERVE_PI_PHYSICAL": "1",
    })
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True,
                        help="paired v2 baseline report with train rows in per_task")
    parser.add_argument("--prompt-candidate-report", required=True,
                        help="train-only GEPA report from this exact source report")
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--metric-calls", type=int, default=360)
    parser.add_argument("--candidate-sets", type=int, default=3)
    parser.add_argument("--seed", type=int, default=202610051)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true",
                      help="write a zero-call, self-hashed execution plan")
    mode.add_argument("--execute", action="store_true",
                      help="run only the exact plan reviewed by its SHA-256")
    parser.add_argument("--expected-plan-sha256", default="",
                        help="required with --execute; hash must match the recomputed plan")
    args = parser.parse_args()
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("joint PubMedQA search requires an explicit isolated RSI_HOME")
    if args.metric_calls <= 0 or args.candidate_sets <= 0:
        raise ValueError("metric-call and candidate-set budgets must be positive")

    source_path = Path(args.source_report).resolve()
    prompt_path = Path(args.prompt_candidate_report).resolve()
    source, prompt_report = read_json(source_path), read_json(prompt_path)
    if source.get("synthetic_preflight_only") and args.execute:
        raise ValueError("synthetic preflight fixture cannot be used for candidate search")
    prompt_candidate, train_ids = validate_inputs(
        source_path, prompt_path, source, prompt_report)
    cfg = configure(source, args.config, train_ids,
                    args.metric_calls, args.candidate_sets)
    state = load_state(str(source["release_id"]))
    if (prompt_candidate is not None
            and state.release_id != prompt_candidate.parent_release_id):
        raise ValueError("loaded release differs from the Prompt candidate parent")
    examples, validation = visible_examples(
        cfg, [str(source["dataset"])], len(train_ids))
    prompt_text = prompt_candidate.new_content if prompt_candidate else state.system_prompt
    prompt_gate_candidate = prompt_candidate or Candidate(
        candidate_id=content_id("cand", state.release_id, "system_prompt", prompt_text),
        parent_release_id=state.release_id, artifact_type="system_prompt",
        artifact_id="system_prompt", new_content=prompt_text,
        rationale="Baseline prompt used because train-only GEPA selected no challenger")
    prompt_cfg = copy.deepcopy(cfg)
    prompt_cfg["evolution"]["allowed_artifacts"] = ["system_prompt"]
    prompt_gate = static_gate(prompt_gate_candidate, prompt_cfg, state)
    if not prompt_gate.get("passed"):
        raise RuntimeError("GEPA Prompt candidate failed static gate before joint search plan")
    validate_baseline_task_rows(source, train_ids)
    plan = build_joint_search_execution_plan(
        source_path=source_path, prompt_path=prompt_path, source=source,
        prompt_candidate=prompt_candidate,
        cfg=cfg, state=state, static_gate_result=prompt_gate,
        train_ids=train_ids, examples=examples, validation=validation,
        metric_calls=args.metric_calls, candidate_sets=args.candidate_sets,
        seed=args.seed)
    plan_path = artifacts_path(
        "probe", "pubmedqa_v2_skill_routing", "plans",
        "joint-search-" + plan["plan_sha256"] + ".json")
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError("saved joint-search plan differs from the recomputed plan")
    else:
        write_json(plan_path, plan)
    if args.validate_only:
        print(json.dumps({
            "status": "validated_not_run", "plan": str(plan_path),
            "plan_sha256": plan["plan_sha256"],
            "logical_request_cap": plan["logical_request_cap"],
            "physical_request_cap": plan["physical_request_cap"],
            "static_gate_passed": plan["prompt_static_gate_passed"],
            "model_calls": 0, "selection_gate_read": False,
            "final_holdout_read": False,
        }, ensure_ascii=False))
        return 0
    if args.expected_plan_sha256 != plan["plan_sha256"]:
        raise ValueError("expected plan hash does not match the recomputed joint-search plan")

    run_dir = artifacts_path("probe", "pubmedqa_v2_skill_routing",
                             "joint-search-" + plan["plan_sha256"])
    output = run_dir / "report.json"
    ledger_path = run_dir / "call-ledger.jsonl"
    if output.exists() or ledger_path.exists():
        raise FileExistsError("this joint-search plan has already been executed or partially started")
    run_dir.mkdir(parents=True, exist_ok=False)

    baseline_run_ids = {str(source["dataset"]): {
        str(int(item["seed"])): str(item["run_id"])
        for item in source["baseline_runs"]}}
    discovery_ids = set(str(x) for x in source.get("discovery_ids") or [])
    confirmation_ids = set(str(x) for x in source.get("confirmation_ids") or [])
    if set(train_ids) & (discovery_ids | confirmation_ids):
        raise ValueError("frozen train IDs overlap discovery or confirmation")
    with request_ledger_environment(plan, ledger_path):
        grounding = run_prompt_grounding_review(prompt_text, cfg)
        if not grounding.get("passed"):
            raise RuntimeError("GEPA Prompt candidate failed claim-free grounding gate")

        candidate = search_task_demo_candidate(
            cfg, state, [str(source["dataset"])], len(train_ids), args.seed,
            [], baseline_run_ids, instruction_candidate=prompt_candidate)
        if candidate is not None:
            program = parse_task_program(candidate.new_content)
            demo_ids = {str(row["task_id"]) for row in program["task_demos"]}
            if not demo_ids or not demo_ids <= set(train_ids):
                raise RuntimeError("joint candidate demos are not confined to train IDs")
            if demo_ids & (discovery_ids | confirmation_ids):
                raise RuntimeError("joint candidate contains a discovery/confirmation demo")

        report = {
            "protocol": JOINT_SEARCH_PROTOCOL,
            "execution_plan_sha256": plan["plan_sha256"],
            "execution_plan_path": str(plan_path),
            "request_ledger_path": str(ledger_path),
            "request_ledger": summarize_ledger(ledger_path),
            "source_report": str(source_path),
            "source_report_sha256": sha256_file(source_path),
            "prompt_candidate_report": str(prompt_path),
            "prompt_candidate_report_sha256": sha256_file(prompt_path),
            "dataset": source["dataset"],
            "train_ids": train_ids,
            "discovery_ids_accessed": False,
            "confirmation_ids_accessed": False,
            "selection_gate_accessed": False,
            "final_holdout_accessed": False,
            "release_id": state.release_id,
            "skill_routing_mode": source["skill_routing_mode"],
            "target_artifact": "task_program",
            "prompt_candidate_id": (prompt_candidate.candidate_id
                                    if prompt_candidate else None),
            "prompt_search_status": prompt_report.get("candidate_status"),
            "joint_instruction_mode": ("gepa_candidate_x_demos"
                                       if prompt_candidate else "baseline_prompt_x_demos"),
            "prompt_static_gate": prompt_gate,
            "prompt_grounding_review": grounding,
            "config_sha256": config_fingerprint(cfg),
            "candidate": candidate.to_dict() if candidate else None,
            "candidate_status": candidate.status if candidate else "no_candidate",
            "candidate_static_gate": candidate.generator.get("selected_static_gate")
                if candidate else None,
            "search_report_path": candidate.generator.get("search_report")
                if candidate else None,
            "eligible_for_discovery": bool(candidate),
        }
        write_json(output, report)
    print(json.dumps({"report": str(output),
                      "candidate_id": candidate.candidate_id if candidate else None,
                      "candidate_status": report["candidate_status"],
                      "discovery_confirmation_accessed": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
