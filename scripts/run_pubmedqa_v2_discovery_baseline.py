"""Run a frozen Skill v2 baseline on train/discovery IDs, never confirmation.

This creates paired baseline runs for GEPA train feedback, few-shot control,
and later discovery comparison. Training IDs can be included explicitly so
the few-shot search has control answers; confirmation remains unread.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_fingerprint, load_config  # noqa: E402
from src.data.split import load_tasks  # noqa: E402
from src.eval.task_rows import validate_task_answer_row  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.provider import build_provider  # noqa: E402
from src.runtime.runner import TaskRunner  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from scripts.pubmedqa_threeway_protocol import verify_dataset_snapshot  # noqa: E402
from src.util import (  # noqa: E402
    artifacts_path, ensure_dir, read_json, sha256_file, sha256_obj, write_json, write_jsonl,
)


def baseline_identity(plan: dict, config_sha256: str, seeds: list[int],
                      release_id: str, evaluated_ids: list[str] | None = None) -> str:
    task_ids = evaluated_ids if evaluated_ids is not None else plan["discovery_ids"]
    return sha256_obj({
        "protocol": "pubmedqa-threeway-v2-skill-routing-baseline-v2",
        "cohort_plan_sha256": sha256_obj(plan),
        "config_sha256": config_sha256,
        "execution_labels": [int(seed) for seed in seeds],
        "release_id": release_id,
        "skill_routing_mode": plan["skill_routing_mode"],
        "evaluated_ids": sorted(str(task_id) for task_id in task_ids),
    })


def build_execution_plan(*, plan_path: Path, plan: dict, tasks: dict,
                         cfg: dict, state, seeds: list[int],
                         evaluated_ids: list[str], include_train: bool) -> dict:
    task_model = (cfg.get("models") or {}).get("task") or {}
    skills = cfg.get("skills") or {}
    logical_task = len(seeds) * len(evaluated_ids)
    model_router = (str(skills.get("routing_mode", "lexical")) == "model"
                    and bool(getattr(state.registry, "skills", []))
                    and not str(skills.get("pinned_skill") or ""))
    logical_router = logical_task if model_router else 0
    retries = max(0, min(4, int(cfg.get("runtime", {}).get(
        "pi_provider_max_retries", 4))))
    bridge_attempts = max(1, min(3, int(cfg.get("runtime", {}).get(
        "pi_timeout_max_attempts", 2))))
    pi_attempts = (retries + 1) * bridge_attempts
    router_attempts = max(1, int(task_model.get("max_retries", 6)))
    logical_by_component = {"task_answer": logical_task}
    physical_by_component = {"task_answer": logical_task * pi_attempts}
    if model_router:
        logical_by_component["skill_router"] = logical_router
        physical_by_component["skill_router"] = logical_router * router_attempts
    unsigned = {
        "schema": "pubmedqa-v2-discovery-baseline-plan-v1",
        "protocol": "pubmedqa-threeway-v2-skill-routing-baseline-v3-plan-bound",
        "role": "discovery_baseline_not_selection_or_confirmation",
        "cohort_plan_path": str(plan_path),
        "cohort_plan_sha256": sha256_file(plan_path),
        "dataset": str(plan["dataset"]),
        "dataset_snapshot": {
            "manifest_sha256": sha256_file(ROOT / "data/manifests/pubmedqa_threeway_probe.json"),
            "normalized_sha256": sha256_file(ROOT / "data/normalized/pubmedqa_threeway_probe.jsonl"),
        },
        "task_material_sha256": sha256_obj([tasks[task_id] for task_id in evaluated_ids]),
        "train_ids": [str(x) for x in plan.get("train_ids", [])],
        "discovery_ids": [str(x) for x in plan.get("discovery_ids", [])],
        "confirmation_ids": [str(x) for x in plan.get("confirmation_ids", [])],
        "evaluated_ids": list(evaluated_ids),
        "train_baseline_included": bool(include_train),
        "execution_labels": [int(seed) for seed in seeds],
        "release_id": state.release_id,
        "parent_system_prompt_sha256": sha256_obj(state.system_prompt),
        "parent_registry_sha256": state.registry.fingerprint(),
        "skill_routing_mode": str(skills.get("routing_mode", "lexical")),
        "config_sha256": config_fingerprint(cfg),
        "task_provider": str(cfg.get("runtime", {}).get("provider", "")),
        "task_model_id": str(task_model.get("model_id") or ""),
        "task_answer_operations_upper_bound": logical_task,
        "skill_router_operations_upper_bound": logical_router,
        "pi_provider_retries_after_first": retries,
        "pi_timeout_subprocess_attempts": bridge_attempts,
        "pi_http_attempts_per_operation_upper_bound": pi_attempts,
        "router_http_attempts_per_operation_upper_bound": router_attempts,
        "logical_by_component": logical_by_component,
        "physical_by_component": physical_by_component,
        "logical_request_cap": sum(logical_by_component.values()),
        "physical_request_cap": sum(physical_by_component.values()),
        "request_ledger_required": True,
        "selection_gate_read": False,
        "confirmation_labels_accessed": False,
        "final_holdout_read": False,
        "model_calls_during_preflight": 0,
        "live_authorized": False,
        "explicit_user_approval_required_before_execution": True,
    }
    return {**unsigned, "plan_sha256": sha256_obj(unsigned)}


def verify_execution_plan(plan: dict, expected_sha256: str) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    actual = sha256_obj(unsigned)
    if plan.get("plan_sha256") != actual or expected_sha256 != actual:
        raise ValueError("expected plan hash does not match current baseline execution plan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=(
        "artifacts/sandboxes/pubmedqa-threeway-v2-skill-routing/cohort-plan.json"))
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--release", default=None)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[202610051, 202610052, 202610053])
    parser.add_argument("--include-train-baseline", action="store_true",
                        help="also baseline train IDs for GEPA/few-shot control; never runs confirmation IDs")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true",
                      help="freeze and validate the baseline execution plan without model calls")
    mode.add_argument("--execute", action="store_true",
                      help="execute only after challenging the exact preflight hash and budgets")
    parser.add_argument("--expected-plan-sha256", default="")
    parser.add_argument("--logical-request-budget", type=int, default=0)
    parser.add_argument("--physical-request-budget", type=int, default=0)
    args = parser.parse_args()
    if args.logical_request_budget < 0 or args.physical_request_budget < 0:
        parser.error("request budgets must be non-negative")
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256 or ""):
        parser.error("--execute requires --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("discovery baseline requires an explicit isolated RSI_HOME")
    if len(args.seeds) < 3 or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("provide at least three distinct execution labels")

    plan_path = Path(args.plan).resolve()
    plan = read_json(plan_path)
    if plan.get("protocol") != "pubmedqa-threeway-v2-skill-routing-cohort-v1":
        raise ValueError("plan is not the frozen Skill v2 cohort protocol")
    verify_dataset_snapshot(plan)
    train_ids = [str(x) for x in plan.get("train_ids", [])]
    discovery_ids = [str(x) for x in plan.get("discovery_ids", [])]
    confirmation_ids = [str(x) for x in plan.get("confirmation_ids", [])]
    if not discovery_ids or len(discovery_ids) != len(set(discovery_ids)):
        raise ValueError("plan must contain unique discovery IDs")
    if (set(train_ids) & set(discovery_ids)
            or set(confirmation_ids) & (set(train_ids) | set(discovery_ids))):
        raise ValueError("frozen plan partitions overlap")

    cfg = copy.deepcopy(load_config(args.config))
    routing_mode = str(plan.get("skill_routing_mode", ""))
    if routing_mode not in {"lexical", "model"}:
        raise ValueError("plan has unsupported Skill routing mode")
    cfg.setdefault("skills", {})["routing_mode"] = routing_mode
    cfg.setdefault("runtime", {}).update({
        "provider": "pi", "input_mode": "task_input", "task_input_max_chars": 30000,
        "pi_request_cooldown_seconds": 6,
    })
    cfg["runtime"].pop("pi_fake_response", None)
    cfg.setdefault("evaluation", {}).update({
        "insufficient_option": False, "main_metric": "accuracy",
        "primary_axis": "accuracy", "sealed_pairing": False,
    })
    cfg.setdefault("evolution", {})["allowed_artifacts"] = ["system_prompt"]
    cfg["evolution"].setdefault("fewshot", {}).update({"enabled": False, "k": 0})
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg["tools"].get("active", []) if name != "academic.search"]

    release_id = args.release or str(plan["release_id"])
    if release_id != str(plan["release_id"]):
        raise ValueError("release ID differs from frozen cohort plan")
    state = load_state(release_id)
    if len(state.registry.skills) == 0:
        raise ValueError("baseline release has no Skills; expected the v2 registry")
    if len(state.registry.skills) != 10:
        raise ValueError(f"expected ten v2 Skills, found {len(state.registry.skills)}")

    available = {str(task["task_id"]): task for task in
                 load_tasks(str(plan["dataset"]), splits=["dev_optimize"])}
    if not set(discovery_ids + train_ids) <= available.keys():
        raise ValueError("one or more train/discovery IDs are absent from dev_optimize")
    evaluated_ids = (train_ids + discovery_ids if args.include_train_baseline
                     else discovery_ids)
    tasks = {task_id: available[task_id] for task_id in evaluated_ids}
    dataset_manifest = ROOT / "data/manifests/pubmedqa_threeway_probe.json"
    dataset_normalized = ROOT / "data/normalized/pubmedqa_threeway_probe.jsonl"
    execution_plan = build_execution_plan(
        plan_path=plan_path, plan=plan, tasks=tasks, cfg=cfg, state=state,
        seeds=args.seeds, evaluated_ids=evaluated_ids,
        include_train=args.include_train_baseline)
    if args.execute:
        verify_execution_plan(execution_plan, args.expected_plan_sha256)
        if (args.logical_request_budget != execution_plan["logical_request_cap"]
                or args.physical_request_budget != execution_plan["physical_request_cap"]):
            raise ValueError("request budgets must exactly match the frozen plan")
    execution_plan_path = artifacts_path(
        "probe", "pubmedqa_v2_skill_routing",
        "baseline-preflight-" + execution_plan["plan_sha256"] + ".json")
    ensure_dir(execution_plan_path.parent)
    if execution_plan_path.exists():
        if read_json(execution_plan_path) != execution_plan:
            raise ValueError("saved baseline execution plan differs from recomputed plan")
    else:
        write_json(execution_plan_path, execution_plan)
    if args.validate_only:
        print(json.dumps({
            "status": "validated_not_run",
            "cohort_plan_sha256": sha256_file(plan_path),
            "execution_plan": str(execution_plan_path),
            "plan_sha256": execution_plan["plan_sha256"],
            "release_id": state.release_id,
            "skill_count": len(state.registry.skills),
            "discovery_count": len(discovery_ids),
            "evaluated_task_count": len(tasks),
            "train_baseline_included": args.include_train_baseline,
            "train_task_count_in_per_task": len(train_ids) if args.include_train_baseline else 0,
            "skill_routing_mode": routing_mode,
            "config_sha256": config_fingerprint(cfg),
            "logical_request_cap": execution_plan["logical_request_cap"],
            "physical_request_cap": execution_plan["physical_request_cap"],
            "model_calls": 0,
        }, ensure_ascii=False))
        return 0

    identity = baseline_identity(plan, config_fingerprint(cfg), args.seeds,
                                state.release_id, evaluated_ids)
    run_id = "pqa-v2-discovery-baseline-" + identity[:12]
    run_dir = artifacts_path("probe", "pubmedqa_v2_skill_routing", run_id)
    ensure_dir(run_dir)
    ledger_path = run_dir / "request-ledger.jsonl"
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(execution_plan["plan_sha256"])
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path.resolve())
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(execution_plan["logical_request_cap"])
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(execution_plan["physical_request_cap"])
    for component, count in execution_plan["logical_by_component"].items():
        os.environ["RSI_CALL_LEDGER_MAX_COMPONENT_" + component.upper()] = str(count)
    checkpoint_path = run_dir / "checkpoint.jsonl"
    saved = {}
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                key = (int(row["seed"]), str(row["task_id"]))
                if key[0] not in args.seeds or key[1] not in tasks:
                    raise ValueError(f"baseline checkpoint contains out-of-scope row: {key}")
                # This file is an append-only attempt log. Retried failures
                # legitimately repeat a key; the latest attempt is the resume
                # state, while every successful attempt is still validated.
                if row.get("status") in {"ok", "abstain"}:
                    saved[key] = validate_task_answer_row(
                        row, tasks[key[1]], context=f"baseline-checkpoint:{key[0]}:{key[1]}",
                        require_reported_gold_index=True)
                else:
                    saved[key] = row

    baseline_runs = []
    for seed in args.seeds:
        provider = build_provider(cfg, seed=seed)
        outcomes = []
        for task_id in evaluated_ids:
            row = saved.get((seed, task_id))
            if row is None or row.get("status") not in {"ok", "abstain"}:
                outcome = TaskRunner(cfg, state, provider=provider, seed=seed).run(tasks[task_id])
                status = str(getattr(outcome, "status", "error"))
                if status in {"ok", "abstain"}:
                    row = outcome.to_dict()
                    row.update({
                        "seed": seed,
                        "gold_label": str(tasks[task_id]["options"][
                            tasks[task_id]["correct_index"]]).strip().casefold(),
                        "config_sha256": config_fingerprint(cfg),
                    })
                    row = validate_task_answer_row(
                        row, tasks[task_id], context=f"baseline:{seed}:{task_id}",
                        require_reported_gold_index=True)
                    row.update({"seed": seed, "config_sha256": config_fingerprint(cfg)})
                else:
                    # Keep a retryable failure marker without persisting provider text.
                    row = {
                        "task_id": task_id, "dataset": str(tasks[task_id]["dataset"]),
                        "split": str(tasks[task_id]["split"]), "status": status[:40],
                        "seed": seed, "config_sha256": config_fingerprint(cfg),
                    }
                with checkpoint_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                saved[(seed, task_id)] = row
            if row.get("status") in {"ok", "abstain"}:
                row = validate_task_answer_row(
                    row, tasks[task_id], context=f"baseline:{seed}:{task_id}",
                    require_reported_gold_index=True)
                saved[(seed, task_id)] = row
            outcomes.append(row)
        failed = [str(row.get("task_id")) for row in outcomes
                  if row.get("status") not in {"ok", "abstain"}]
        if failed:
            raise RuntimeError(
                f"baseline seed {seed} has non-ok task rows; no baseline report written: "
                + ", ".join(failed))
        if [str(row.get("task_id")) for row in outcomes] != evaluated_ids:
            raise RuntimeError(f"baseline seed {seed} does not cover the requested task IDs exactly")
        task_rows = [{key: value for key, value in row.items()
                      if key not in {"seed", "config_sha256"}}
                     for row in outcomes]
        seed_run_id = f"pqa-v2-baseline-{seed}-{identity[:8]}"
        seed_run_dir = artifacts_path("runs", seed_run_id)
        ensure_dir(seed_run_dir)
        write_jsonl(seed_run_dir / "per_task.jsonl", task_rows)
        discovery_rows = [row for row in outcomes
                          if str(row.get("task_id")) in set(discovery_ids)]
        correct = sum(bool(row.get("correct")) for row in discovery_rows)
        answered = sum(int(row.get("answer_index", -1)) >= 0 for row in discovery_rows)
        baseline_runs.append({
            "seed": seed,
            "run_id": seed_run_id,
            "task_count": len(discovery_rows),
            "run_task_count": len(outcomes),
            "metric_task_ids": discovery_ids,
            "per_task_includes_train": args.include_train_baseline,
            "metrics": {
                "accuracy": correct / len(discovery_rows),
                "coverage": answered / len(discovery_rows),
                "answered_precision": correct / answered if answered else 0.0,
            },
        })

    report = {
        "protocol": "pubmedqa-threeway-v2-skill-routing-baseline-v3-plan-bound",
        "cohort_plan": str(plan_path),
        "cohort_plan_sha256": sha256_file(plan_path),
        "dataset": str(plan["dataset"]),
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "dataset_normalized_sha256": sha256_file(dataset_normalized),
        "release_id": state.release_id,
        "skill_routing_mode": routing_mode,
        "config_sha256": config_fingerprint(cfg),
        "train_ids": train_ids,
        "discovery_ids": discovery_ids,
        "confirmation_ids": confirmation_ids,
        "train_labels_accessed": args.include_train_baseline,
        "confirmation_labels_accessed": False,
        "train_answers_accessed": args.include_train_baseline,
        "train_ids_in_per_task": train_ids if args.include_train_baseline else [],
        "per_task_run_scope": "train_plus_discovery" if args.include_train_baseline else "discovery_only",
        "selection_gate_accessed": False,
        "final_holdout_accessed": False,
        "retrieval_used": False,
        "published": False,
        "execution_labels": list(args.seeds),
        "evaluated_ids": evaluated_ids,
        "baseline_runs": baseline_runs,
        "checkpoint": str(checkpoint_path),
        "baseline_identity_sha256": identity,
        "execution_plan": str(execution_plan_path),
        "execution_plan_sha256": execution_plan["plan_sha256"],
        "request_ledger_path": str(ledger_path),
        "request_ledger_summary": summarize_ledger(ledger_path),
    }
    report_path = run_dir / "report.json"
    write_json(report_path, report)
    print(json.dumps({
        "report": str(report_path), "discovery_count": len(discovery_ids),
        "release_id": state.release_id, "skill_count": len(state.registry.skills),
        "skill_routing_mode": routing_mode, "baseline_runs": baseline_runs,
        "train_baseline_included": args.include_train_baseline,
        "confirmation_labels_accessed": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
