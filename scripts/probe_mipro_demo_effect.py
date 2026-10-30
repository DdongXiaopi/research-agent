#!/usr/bin/env python3
"""Isolate the effect of labeled-demo count for the frozen EI incumbent prompt.

Exploratory only: uses nested train-only demo subsets and the already-used
24-row search-validation scope; it never reads internal confirmation or test.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_evidence_inference_mipro_search import (  # noqa: E402
    load_frozen_cohort, prepare_config,
)
from src.config import config_fingerprint  # noqa: E402
from src.contracts.types import Candidate, content_id, content_sha256  # noqa: E402
from src.data.evidence_inference_v2 import load_tasks  # noqa: E402
from src.eval.stats import mcnemar_exact, paired_bootstrap  # noqa: E402
from src.evolution.fewshot import parse_demos_bundle, serialize_demos_bundle  # noqa: E402
from src.evolution.task_program import serialize_task_program  # noqa: E402
from src.evolution.task_input_gates import validate_task_demo_rendering  # noqa: E402
from src.eval.task_rows import validate_task_answer_row  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.runtime.call_ledger import summarize_ledger  # noqa: E402
from src.runtime.runner import ArtifactState, TaskRunner  # noqa: E402
from src.util import (  # noqa: E402
    append_jsonl, artifacts_path, sha256_file, sha256_obj,
    read_json, read_jsonl, sha256_file, write_json,
)


RELEASE_ID = "task-input-evidence-inference-v1"


def build_nested_demo_arms(
        selected_ids: list[str], task_by_id: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if len(selected_ids) != 3 or len(set(selected_ids)) != 3:
        raise ValueError("demo dose probe needs exactly three unique train demo IDs")
    examples = []
    for task_id in selected_ids:
        task = task_by_id.get(str(task_id))
        if task is None or task.get("split") != "dev_optimize":
            raise ValueError("dose probe demo escaped the frozen train scope")
        answer = task.get("correct_index")
        if type(answer) is not int:
            raise ValueError("dose probe demo has no frozen integer label")
        examples.append({"task_id": str(task_id), "task": task,
                         "correct_index": answer})
    return {
        f"demos_{count}": parse_demos_bundle(serialize_demos_bundle(examples[:count]))
        if count else []
        for count in range(4)
    }


def summarize_rows(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    arms: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("seed") == seed:
            if row.get("status") not in {"ok", "abstain"}:
                raise ValueError("demo-dose contains a failed task outcome")
            arms.setdefault(str(row["arm"]), []).append(row)
    def index_unique(values: list[dict[str, Any]], arm: str) -> dict[str, dict[str, Any]]:
        indexed = {str(row["task_id"]): row for row in values}
        if len(indexed) != len(values):
            raise ValueError(f"{arm} contains duplicate task rows")
        return indexed

    baseline = sorted(arms.get("demos_0", []), key=lambda row: row["task_id"])
    baseline_by_id = index_unique(baseline, "demos_0")
    report: dict[str, Any] = {"arms": {}}
    for name, values in sorted(arms.items()):
        values = sorted(values, key=lambda row: row["task_id"])
        candidate_by_id = index_unique(values, name)
        by_label = {}
        for label in sorted({str(row["gold_label"]) for row in values}):
            selected = [row for row in values if row["gold_label"] == label]
            by_label[label] = {
                "n": len(selected),
                "accuracy": (sum(bool(row["correct"]) for row in selected) / len(selected)
                             if selected else None),
            }
        correct = [float(bool(row["correct"])) for row in values]
        item = {
            "n": len(values),
            "accuracy": sum(correct) / len(correct) if correct else None,
            "correct": sum(correct),
            "per_gold_label": by_label,
        }
        if name != "demos_0":
            paired_ids = sorted(set(baseline_by_id) & set(candidate_by_id))
            transitions = []
            for task_id in paired_ids:
                before_row = baseline_by_id[task_id]
                after_row = candidate_by_id[task_id]
                before_correct = bool(before_row["correct"])
                after_correct = bool(after_row["correct"])
                transitions.append({
                    "task_id": task_id,
                    "gold_label": str(after_row["gold_label"]),
                    "baseline_answer_index": int(before_row["answer_index"]),
                    "candidate_answer_index": int(after_row["answer_index"]),
                    "baseline_correct": before_correct,
                    "candidate_correct": after_correct,
                    "answer_changed": (int(before_row["answer_index"])
                                        != int(after_row["answer_index"])),
                    "correctness_transition": (
                        "improved" if after_correct and not before_correct else
                        "regressed" if before_correct and not after_correct else
                        "unchanged_correct" if after_correct else
                        "unchanged_incorrect"),
                })
            transition_counts = {
                outcome: sum(row["correctness_transition"] == outcome
                             for row in transitions)
                for outcome in ("improved", "regressed", "unchanged_correct",
                                "unchanged_incorrect")
            }
            complete_pairs = set(baseline_by_id) == set(candidate_by_id)
            item["paired_answer_transitions"] = {
                "complete": complete_pairs,
                "paired_count": len(paired_ids),
                "baseline_only_task_ids": sorted(
                    set(baseline_by_id) - set(candidate_by_id)),
                "candidate_only_task_ids": sorted(
                    set(candidate_by_id) - set(baseline_by_id)),
                "answer_changed_count": sum(row["answer_changed"]
                                             for row in transitions),
                "correctness_transitions": transition_counts,
                "rows": transitions,
            }
        if (name != "demos_0" and item["paired_answer_transitions"]["complete"]
                and values and baseline):
            after = [float(bool(candidate_by_id[task_id]["correct"]))
                     for task_id in sorted(candidate_by_id)]
            before = [float(bool(baseline_by_id[task_id]["correct"]))
                      for task_id in sorted(baseline_by_id)]
            item["paired_vs_zero_demo"] = {
                "delta_pp": (sum(a - b for a, b in zip(after, before))
                             / len(before) * 100.0),
                "mcnemar_exact": mcnemar_exact(before, after),
                "paired_bootstrap": paired_bootstrap(
                    before, after, B=10000, rng_seed=seed),
            }
        report["arms"][name] = item
    return report


def preflight_demo_program_arms(
        cfg: dict[str, Any], state: Any,
        demo_arms: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Apply the shared FR-203 gate to each non-baseline nested demo program."""
    records = {}
    for name, demos in demo_arms.items():
        if not demos:
            continue  # demos_0 is the frozen parent baseline, not a candidate.
        bundle = serialize_demos_bundle(demos)
        content = serialize_task_program(state.system_prompt, bundle)
        candidate = Candidate(
            candidate_id=content_id(
                "cand", state.release_id, "task_program", content),
            parent_release_id=state.release_id,
            artifact_type="task_program", artifact_id="task_program",
            diff=f"evaluate exploratory {name} demo-dose arm",
            new_content=content, content_sha256=content_sha256(content))
        gate = static_gate(candidate, cfg, state)
        failed_checks = [row["check"] for row in gate.get("results", [])
                         if row.get("passed") is not True]
        record = {
            "candidate_id": candidate.candidate_id,
            "content_sha256": candidate.content_sha256,
            "static_gate_passed": gate.get("passed") is True,
            "failed_checks": failed_checks,
            "static_gate": gate,
        }
        records[name] = record
        if not record["static_gate_passed"]:
            raise ValueError(
                f"demo-dose candidate failed static gate ({name}): "
                + ", ".join(failed_checks))
    return records


def build_probe_protocol(*, cfg: dict[str, Any], source_study_id: str,
                         source_plan: dict[str, Any], cohort_sha256: str,
                         task_model: str, task_temperature: float,
                         seed: int, search_validation_count: int,
                         demo_ids: list[str], candidate_preflight: dict[str, dict[str, Any]],
                         source_report_sha256: str = "",
                         cohort_file_sha256: str = "",
                         task_ids: list[str] | None = None,
                         task_material_sha256: str = "",
                         release_state: dict[str, Any] | None = None,
                         render_gates: dict[str, Any] | None = None,
                         logical_by_component: dict[str, int] | None = None,
                         physical_by_component: dict[str, int] | None = None) -> dict[str, Any]:
    """Bind the diagnostic identity to code/config and exact gated candidate arms."""
    protocol = {
        "protocol": "ei-mipro-fixed-incumbent-nested-demo-dose-v4",
        "config_sha256": config_fingerprint(cfg),
        "source_study_id": source_study_id,
        "source_plan_sha256": source_plan.get("plan_sha256"),
        "dataset": source_plan.get("dataset"),
        "cohort_sha256": cohort_sha256,
        "task_model": task_model,
        "task_temperature": float(task_temperature),
        "routing_mode": "model",
        "seed_label": int(seed),
        "seed_semantics": "execution label only; HTTP provider does not forward seed",
        "sampling_replication": "not_independent_at_temperature_zero",
        "search_validation_count": int(search_validation_count),
        "demo_source": "highest-scoring 3-demo trial, nested prefixes",
        "demo_ids": list(demo_ids),
        "source_report_sha256": source_report_sha256,
        "cohort_file_sha256": cohort_file_sha256,
        "task_ids": list(task_ids or []),
        "task_material_sha256": task_material_sha256,
        "release_state": release_state or {},
        "render_gates": render_gates or {},
        "logical_by_component": logical_by_component or {},
        "physical_by_component": physical_by_component or {},
        "logical_request_cap": sum((logical_by_component or {}).values()),
        "physical_request_cap": sum((physical_by_component or {}).values()),
        "candidate_preflight": {
            arm: {key: record[key] for key in (
                "candidate_id", "content_sha256", "static_gate_passed", "failed_checks")}
            for arm, record in sorted(candidate_preflight.items())
        },
        "arms": ["demos_0", "demos_1", "demos_2", "demos_3"],
        "selection_gate_read": False,
        "official_validation_test_read": False,
    }
    protocol["protocol_sha256"] = sha256_obj(protocol)
    return protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--source-study-id", required=True)
    parser.add_argument("--seed", type=int, default=202610081,
                        help="execution label; the HTTP provider does not receive this seed")
    parser.add_argument("--logical-budget", type=int, default=None)
    parser.add_argument("--physical-budget", type=int, default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-plan-sha256", default="")
    args = parser.parse_args(argv)
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256 or ""):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")

    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("MIPRO demo effect probe requires an isolated RSI_HOME")
    source_dir = artifacts_path("probe", "evidence_inference_mipro", args.source_study_id)
    source_report_path = source_dir / "report.json"
    if not source_report_path.exists():
        raise ValueError("source MIPRO report is missing")
    source_report = read_json(source_report_path)
    plan = source_report.get("plan") or {}
    if (source_report.get("status") != "completed"
            or source_report.get("candidate_status") not in {
                "no_candidate_no_labeled_demos",  # original report schema
                "incumbent_won_without_task_demos",
            }
            or plan.get("task_model") != "Qwen/Qwen3-8B"):
        raise ValueError("source study is not a completed 8B no-candidate MIPRO search")

    records = ((source_report.get("mipro") or {}).get("trial_audit") or {}).get("records") or []
    measured = [record for record in records
                if record.get("program_available") is True
                and int(record.get("demo_count") or 0) == 3]
    if not measured:
        raise ValueError("source MIPRO search did not score a 3-demo program")
    winner = max(measured, key=lambda record: float(record.get("score") or 0.0))
    selected_ids = [str(value) for value in winner.get("demo_task_ids") or []]
    if len(selected_ids) != 3:
        raise ValueError("winning MIPRO demo bundle has an unexpected size")

    cfg = prepare_config(args.config, proposer_calls=24, routing_mode="model",
                         task_model_id="Qwen/Qwen3-8B")
    state = load_state(RELEASE_ID)
    train_rows = load_tasks(splits=["dev_optimize"])
    cohort, scopes, task_by_id = load_frozen_cohort(args.cohort.resolve(), train_rows)
    train_ids = {str(row["task_id"]) for row in scopes["mipro_train"]}
    search_ids = {str(row["task_id"]) for row in scopes["search_validation"]}
    if not set(selected_ids) <= train_ids:
        raise ValueError("MIPRO-selected demos do not come only from nested train")
    if not search_ids or search_ids & train_ids:
        raise ValueError("search-validation scope is invalid or overlaps train")

    demo_arms = build_nested_demo_arms(selected_ids, task_by_id)
    candidate_preflight = preflight_demo_program_arms(cfg, state, demo_arms)
    render_gates = {}
    for name, demos in demo_arms.items():
        if demos:
            gate = validate_task_demo_rendering(demos, cfg)
            if not gate.get("passed"):
                raise ValueError(f"demo rendering gate rejected {name}: {gate}")
            render_gates[name] = gate

    state_by_arm = {
        name: ArtifactState(state.release_id, state.system_prompt,
                            state.registry, demos)
        for name, demos in demo_arms.items()
    }
    tasks = sorted(scopes["search_validation"], key=lambda row: str(row["task_id"]))
    task_operations = len(tasks) * len(state_by_arm)
    router_operations = (task_operations if (
        str((cfg.get("skills") or {}).get("routing_mode") or "lexical") == "model"
        and bool(state.registry.skills)
        and not str((cfg.get("skills") or {}).get("pinned_skill") or "")) else 0)
    runtime = cfg.get("runtime") or {}
    task_cfg = (cfg.get("models") or {}).get("task") or {}
    pi_attempts = ((max(0, min(4, int(runtime.get("pi_provider_max_retries", 4)))) + 1)
                   * max(1, min(3, int(runtime.get("pi_timeout_max_attempts", 1)))))
    router_attempts = max(1, int(task_cfg.get("max_retries", 6)))
    logical_by_component = {"task_answer": task_operations}
    physical_by_component = {"task_answer": task_operations * pi_attempts}
    if router_operations:
        logical_by_component["skill_router"] = router_operations
        physical_by_component["skill_router"] = router_operations * router_attempts
    protocol = build_probe_protocol(
        cfg=cfg, source_study_id=args.source_study_id, source_plan=plan,
        cohort_sha256=cohort["cohort_sha256"], task_model="Qwen/Qwen3-8B",
        task_temperature=float(cfg.get("models", {}).get("task", {}).get(
            "temperature", 0.0)), seed=args.seed,
        search_validation_count=len(search_ids), demo_ids=selected_ids,
        candidate_preflight=candidate_preflight,
        source_report_sha256=sha256_file(source_report_path),
        cohort_file_sha256=sha256_file(args.cohort.resolve()),
        task_ids=[str(task["task_id"]) for task in tasks],
        task_material_sha256=sha256_obj(tasks),
        release_state={"release_id": state.release_id,
                       "system_prompt_sha256": sha256_obj(state.system_prompt),
                       "registry_sha256": state.registry.fingerprint(),
                       "task_demos_sha256": sha256_obj(state.task_demos)},
        render_gates=render_gates,
        logical_by_component=logical_by_component,
        physical_by_component=physical_by_component)
    plan_hash = protocol["protocol_sha256"]
    if args.execute:
        if args.expected_plan_sha256 != plan_hash:
            raise ValueError("execution plan hash mismatch")
        if (args.logical_budget != protocol["logical_request_cap"]
                or args.physical_budget != protocol["physical_request_cap"]):
            raise ValueError("request budgets must exactly match the frozen plan")
    out_dir = artifacts_path("probe", "evidence_inference_mipro_demo_ablation",
                             protocol["protocol_sha256"][:16])
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "plan.json"
    rows_path = out_dir / "rows.jsonl"
    ledger_path = out_dir / "request-ledger.jsonl"
    if plan_path.exists() and read_json(plan_path) != protocol:
        raise ValueError("existing demo-dose artifact has a different frozen protocol")
    write_json(plan_path, protocol)

    if args.validate_only:
        print(json.dumps({"status": "validated_not_run", "plan": str(plan_path),
                          "plan_sha256": plan_hash,
                          "logical_request_cap": protocol["logical_request_cap"],
                          "physical_request_cap": protocol["physical_request_cap"],
                          "model_calls": 0, "selection_gate_read": False,
                          "official_validation_test_read": False}, ensure_ascii=False))
        return 0

    task_key_env = str(task_cfg.get("api_key_env") or "").strip()
    if not task_key_env or not os.environ.get(task_key_env):
        raise RuntimeError("demo-dose execution is missing the configured task-model credential")

    os.environ["RSI_HTTP_NO_CACHE"] = "1"
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(plan_hash)
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path.resolve())
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(protocol["logical_request_cap"])
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(protocol["physical_request_cap"])
    os.environ["RSI_CALL_LEDGER_RESERVE_PI_PHYSICAL"] = "1"
    for component, count in protocol["logical_by_component"].items():
        os.environ["RSI_CALL_LEDGER_MAX_COMPONENT_" + component.upper()] = str(count)
    os.environ.setdefault("RSI_PROVIDER_429_BACKOFF_BASE_SECONDS", "15")
    existing = {(str(row.get("arm")), str(row.get("task_id")), int(row.get("seed", -1)))
                for row in read_jsonl(rows_path)}
    task_by_id = {str(task["task_id"]): task for task in tasks}
    arms = set(state_by_arm)
    for row in read_jsonl(rows_path):
        task = task_by_id.get(str(row.get("task_id")))
        if (task is None or str(row.get("arm")) not in arms
                or int(row.get("seed", -1)) != args.seed
                or str(row.get("dataset", "")) != str(task.get("dataset", ""))
                or str(row.get("split", "")) != str(task.get("split", "dev_optimize"))):
            raise ValueError("existing demo-dose rows escape the frozen plan")
        validate_task_answer_row(row, task, context="demo-dose checkpoint",
                                 require_reported_gold_index=True)
    try:
        for arm, arm_state in state_by_arm.items():
            for task in tasks:
                key = (arm, str(task["task_id"]), args.seed)
                if key in existing:
                    continue
                outcome = TaskRunner(cfg, arm_state, seed=args.seed).run(task)
                row = {
                    "arm": arm, "task_id": str(outcome.task_id),
                    "dataset": str(outcome.dataset), "split": str(outcome.split),
                    "gold_label": str(task["gold_label"]),
                    "answer_index": int(outcome.answer_index),
                    "gold_index": int(outcome.gold_index),
                    "correct": bool(outcome.correct),
                    "status": str(outcome.status), "seed": args.seed,
                }
                if (str(outcome.task_id) != str(task["task_id"])
                        or str(outcome.dataset) != str(task.get("dataset", ""))
                        or str(outcome.split) != str(task.get("split", "dev_optimize"))):
                    raise ValueError("demo-dose outcome identity differs from frozen task")
                row = validate_task_answer_row(
                    row, task, context="demo-dose evaluation",
                    require_reported_gold_index=True)
                append_jsonl(rows_path, row)
                existing.add(key)
    except Exception as exc:
        failure = {
            "status": "failed", "protocol": protocol,
            "error_type": type(exc).__name__,
            "request_accounting": summarize_ledger(ledger_path),
            "selection_gate_read": False, "official_validation_test_read": False,
        }
        write_json(out_dir / "report.json", failure)
        raise

    rows = read_jsonl(rows_path)
    expected = {(arm, task_id, args.seed) for arm in state_by_arm
                for task_id in task_by_id}
    observed = [(str(row.get("arm")), str(row.get("task_id")),
                 int(row.get("seed", -1))) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("demo-dose result rows do not exactly cover the frozen plan")
    report = {
        "status": "completed", "protocol": protocol,
        "source_study_candidate_status": source_report.get("candidate_status"),
        "demo_source_trial_number": winner.get("trial_number"),
        "demo_source_trial_accuracy": winner.get("score"),
        "candidate_prompt_held_constant": True,
        "candidate_preflight": candidate_preflight,
        "search_validation_is_exploratory_and_reused": True,
        "results": summarize_rows(rows, args.seed),
        "request_accounting": summarize_ledger(ledger_path),
        "selection_gate_read": False, "official_validation_test_read": False,
    }
    write_json(out_dir / "report.json", report)
    print(json.dumps({"report": str(out_dir / "report.json"),
                      "status": report["status"], "results": report["results"],
                      "request_accounting": report["request_accounting"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
