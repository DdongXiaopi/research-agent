"""Calibrate the evidence judge on the task-input protocol's visible optimize split.

This measures whether the judge accepts the benchmark's labeled yes/no answer
from the exact supplied evidence while rejecting the opposite answer. It is a
task-level calibration proxy, not an entailment oracle or confirmation result.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rsi import load_config
from src.data.split import load_tasks
from src.eval.grounding import (parse_proposition_verdict,
                                proposition_judge_messages)
from src.runtime.input_evidence import task_input_docs
from src.runtime.provider import build_provider
from src.util import sha256_obj


def select_cases(per_label: int, offset_per_label: int = 0) -> list[dict]:
    tasks = load_tasks("pubmedqa", ["dev_optimize"])
    by_label = {"yes": [], "no": []}
    for task in tasks:
        answer = str(task["options"][int(task["correct_index"])]).strip().casefold()
        if answer not in by_label or not task_input_docs(task):
            continue
        by_label[answer].append(task)
    selected = []
    for label, rows in by_label.items():
        ranked = sorted(rows, key=lambda row: hashlib.sha256(
            str(row["task_id"]).encode("utf-8")).hexdigest())
        if len(ranked) < offset_per_label + per_label:
            raise ValueError(f"not enough {label} dev_optimize tasks: {len(ranked)}")
        selected.extend(ranked[offset_per_label:offset_per_label + per_label])
    return selected


def build_protocol(cases: list[dict], model_id: str, prompt_version: str,
                   provider_config: dict) -> tuple[str, dict[str, tuple[str, str]]]:
    """Bind resumable rows to the exact requests, labels, config, and judge code."""
    requests = {}
    task_inputs = []
    for task in cases:
        task_id = str(task["task_id"])
        docs = task_input_docs(task)
        system, user = proposition_judge_messages(
            task["question"], docs, prompt_version=prompt_version)
        requests[task_id] = (system, user)
        task_inputs.append({
            "task_id": task_id,
            "gold_answer": str(task["options"][int(task["correct_index"])])
            .strip().casefold(),
            "system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
            "user_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(),
        })
    grounding_path = Path(inspect.getsourcefile(parse_proposition_verdict))
    protocol = sha256_obj({
        "schema": "task-input-grounding-calibration-protocol-v2",
        "dataset": "pubmedqa", "split": "dev_optimize",
        "task_inputs": task_inputs,
        "model_id": model_id,
        "prompt_version": prompt_version,
        "provider_config_sha256": sha256_obj(provider_config),
        "calibration_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "grounding_module_sha256": hashlib.sha256(grounding_path.read_bytes()).hexdigest(),
        "max_tokens": 450,
    })
    return protocol, requests


def validate_checkpoint_row(row: dict, protocol: str, task: dict) -> None:
    task_id = str(task["task_id"])
    gold = str(task["options"][int(task["correct_index"])])
    gold = gold.strip().casefold()
    if row.get("protocol") != protocol:
        raise ValueError("task-input grounding checkpoint protocol changed")
    if row.get("task_id") != task_id:
        raise ValueError("task-input grounding checkpoint task identity changed")
    if row.get("gold_answer") != gold or row.get("opposite_answer") != (
            "no" if gold == "yes" else "yes"):
        raise ValueError("task-input grounding checkpoint label changed")
    if (not isinstance(row.get("gold_supported"), bool)
            or not isinstance(row.get("opposite_supported"), bool)
            or any(not isinstance(row.get(key), str)
                   for key in ("gold_reason", "opposite_reason", "raw_verdict"))
            or any(type(row.get(key)) is not int or row[key] < 0
                   for key in ("prompt_tokens", "completion_tokens", "evidence_chars"))):
        raise ValueError("malformed task-input grounding checkpoint row")
    expected_evidence_chars = sum(len(doc["text"]) for doc in task_input_docs(task))
    if row["evidence_chars"] != expected_evidence_chars:
        raise ValueError("task-input grounding checkpoint evidence changed")


def load_checkpoint_rows(checkpoint: Path, task_by_id: dict[str, dict],
                         protocol: str) -> dict[str, dict]:
    saved: dict[str, dict] = {}
    if not checkpoint.exists():
        return saved
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("task_id"), str):
            raise ValueError("malformed task-input grounding checkpoint row")
        if row["task_id"] in saved:
            raise ValueError("duplicate task-input grounding checkpoint row")
        if row["task_id"] not in task_by_id:
            raise ValueError("checkpoint contains a task outside this calibration cohort")
        validate_checkpoint_row(row, protocol, task_by_id[row["task_id"]])
        saved[row["task_id"]] = row
    return saved


def build_execution_plan(cases: list[dict], model_id: str, prompt_version: str,
                         provider_config: dict, protocol: str,
                         requests: dict[str, tuple[str, str]],
                         saved_rows: dict[str, dict], *,
                         output_path: str,
                         max_http_attempts: int,
                         logical_request_budget: int | None = None,
                         physical_request_budget: int | None = None) -> dict[str, Any]:
    """Freeze exact task requests and logical/physical call ceilings."""
    task_ids = [str(task["task_id"]) for task in cases]
    if len(set(task_ids)) != len(task_ids) or set(requests) != set(task_ids):
        raise ValueError("grounding plan tasks and exact requests do not match")
    if set(saved_rows) - set(task_ids):
        raise ValueError("grounding checkpoint is outside the planned cohort")
    if type(max_http_attempts) is not int or max_http_attempts < 1:
        raise ValueError("max_http_attempts must be a positive integer")
    pending_ids = sorted(set(task_ids) - set(saved_rows))
    logical_upper = len(pending_ids)
    physical_upper = logical_upper * max_http_attempts
    logical_budget = (logical_upper if logical_request_budget is None
                      else logical_request_budget)
    physical_budget = (physical_upper if physical_request_budget is None
                       else physical_request_budget)
    if (type(logical_budget) is not int or logical_budget < logical_upper
            or type(physical_budget) is not int or physical_budget < physical_upper):
        raise ValueError("request budgets are below the frozen worst-case ceiling")

    grounding_path = Path(inspect.getsourcefile(parse_proposition_verdict)).resolve()
    evidence_path = Path(inspect.getsourcefile(task_input_docs)).resolve()
    split_path = Path(inspect.getsourcefile(load_tasks)).resolve()
    source_hashes = {
        "calibration_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "grounding_module_sha256": hashlib.sha256(grounding_path.read_bytes()).hexdigest(),
        "task_input_evidence_module_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "split_loader_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    }
    request_rows = [
        {"task_id": task_id,
         "system_sha256": hashlib.sha256(requests[task_id][0].encode("utf-8")).hexdigest(),
         "user_sha256": hashlib.sha256(requests[task_id][1].encode("utf-8")).hexdigest()}
        for task_id in pending_ids
    ]
    plan = {
        "schema": "task-input-grounding-calibration-execution-plan-v1",
        "dataset": "pubmedqa",
        "split": "dev_optimize",
        "output_path": str(Path(output_path).resolve()),
        "selection": "task-id SHA-256 hash, balanced yes/no, supplied evidence only",
        "task_ids": sorted(task_ids),
        "pending_task_ids": pending_ids,
        "cached_task_ids": sorted(saved_rows),
        "task_count": len(task_ids),
        "pending_task_count": logical_upper,
        "model_id": model_id,
        "judge_prompt_version": prompt_version,
        "max_tokens_per_request": 450,
        "max_http_attempts_per_logical_request": max_http_attempts,
        "logical_request_upper_bound": logical_upper,
        "logical_request_budget": logical_budget,
        "physical_request_upper_bound": physical_upper,
        "physical_request_budget": physical_budget,
        "protocol_sha256": protocol,
        "provider_config_sha256": sha256_obj(provider_config),
        "task_input_request_hashes": request_rows,
        "checkpoint_rows_sha256": sha256_obj({
            task_id: saved_rows[task_id] for task_id in sorted(saved_rows)}),
        "source_hashes": source_hashes,
        "selection_gate_read": False,
        "official_validation_read": False,
        "official_test_read": False,
        "model_calls_during_preflight": 0,
        "live_authorized": False,
        "explicit_user_approval_required_before_execution": True,
    }
    return {**plan, "plan_sha256": sha256_obj(plan)}


def verify_execution_plan(saved_plan: dict, current_plan: dict,
                          expected_sha256: str) -> None:
    saved_unsigned = {key: value for key, value in saved_plan.items()
                      if key != "plan_sha256"}
    if saved_plan.get("plan_sha256") != sha256_obj(saved_unsigned):
        raise ValueError("saved grounding plan self-hash is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or ""):
        raise ValueError("grounding execution requires --expected-plan-sha256")
    if expected_sha256 != saved_plan.get("plan_sha256"):
        raise ValueError("expected grounding plan hash does not match the saved plan")
    if saved_plan != current_plan:
        raise ValueError("current grounding protocol differs from the frozen plan")


def _load_provider_config(model_id: str, max_http_attempts: int) -> dict:
    cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
    cfg["models"]["task"]["model_id"] = model_id
    # HttpProvider interprets max_retries as total attempts, including the first.
    cfg["models"]["task"]["max_retries"] = max_http_attempts
    return cfg


def _write_execution_state(path: Path, state: dict[str, Any], *, create: bool = False) -> None:
    """Persist request reservations before dispatch; never replay a consumed plan."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(state, ensure_ascii=False, indent=2) + "\n"
    if create:
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise ValueError(
                "grounding execution state already exists; do not replay a consumed plan") from exc
        return
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-label", type=int, default=30)
    parser.add_argument("--offset-per-label", type=int, default=0,
                        help="skip earlier hash-ranked tasks for disjoint validation")
    parser.add_argument("--model-id", default="Qwen/Qwen3-32B")
    parser.add_argument("--prompt-version", choices=("v2", "v3", "v4"), default="v4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path,
                        help="frozen plan path (defaults to OUTPUT.plan.json)")
    parser.add_argument("--max-http-attempts", type=int, choices=range(1, 7), default=1,
                        help="total HTTP attempts per task; defaults to one, with no retries")
    parser.add_argument("--logical-request-budget", type=int,
                        help="upper bound for uncached task requests; defaults to exact need")
    parser.add_argument("--physical-request-budget", type=int,
                        help="upper bound for HTTP attempts; defaults to exact worst case")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true",
                      help="write an inspectable zero-call plan without creating a provider")
    mode.add_argument("--execute", action="store_true",
                      help="execute only a saved plan matching --expected-plan-sha256")
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args(argv)
    if not 1 <= args.per_label <= 100 or args.offset_per_label < 0:
        parser.error("per-label must be 1..100")
    if args.execute:
        if not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256 or ""):
            parser.error("--execute requires --expected-plan-sha256")
        if args.logical_request_budget is None or args.physical_request_budget is None:
            parser.error("--execute requires explicit logical and physical request budgets")
    elif args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    if args.logical_request_budget is not None and args.logical_request_budget < 0:
        parser.error("logical request budget must be nonnegative")
    if args.physical_request_budget is not None and args.physical_request_budget < 0:
        parser.error("physical request budget must be nonnegative")

    execution_state_path = Path(str(args.output) + ".execution.json")
    if execution_state_path.exists():
        raise ValueError(
            "this output path already has a consumed execution plan; use a new output "
            "path and obtain approval for its new plan")

    cases = select_cases(args.per_label, args.offset_per_label)
    cfg = _load_provider_config(args.model_id, args.max_http_attempts)
    protocol, requests = build_protocol(
        cases, args.model_id, args.prompt_version, cfg)
    checkpoint = args.output.with_suffix(args.output.suffix + ".partial.jsonl")
    task_by_id = {str(task["task_id"]): task for task in cases}
    saved = load_checkpoint_rows(checkpoint, task_by_id, protocol)
    current_plan = build_execution_plan(
        cases, args.model_id, args.prompt_version, cfg, protocol, requests, saved,
        output_path=str(args.output),
        max_http_attempts=args.max_http_attempts,
        logical_request_budget=args.logical_request_budget,
        physical_request_budget=args.physical_request_budget)
    plan_path = args.plan or Path(str(args.output) + ".plan.json")
    if args.validate_only:
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(current_plan, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        print(json.dumps({
            "status": "validated_not_run",
            "plan": str(plan_path),
            "plan_sha256": current_plan["plan_sha256"],
            "task_count": current_plan["task_count"],
            "pending_task_count": current_plan["pending_task_count"],
            "logical_request_budget": current_plan["logical_request_budget"],
            "physical_request_budget": current_plan["physical_request_budget"],
            "model_calls": 0,
        }, ensure_ascii=False, indent=2))
        return 0

    if not plan_path.exists():
        raise ValueError("frozen grounding plan does not exist; run --validate-only first")
    frozen_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    verify_execution_plan(frozen_plan, current_plan, args.expected_plan_sha256)
    if args.output.exists():
        raise FileExistsError("grounding report already exists; choose a new --output path")

    # Provider construction is deliberately below all frozen-plan checks.
    provider = build_provider(cfg)
    provider.cache = False
    execution_state = {
        "schema": "task-input-grounding-execution-state-v1",
        "plan_sha256": current_plan["plan_sha256"],
        "status": "started",
        "logical_request_budget": current_plan["logical_request_budget"],
        "physical_request_budget": current_plan["physical_request_budget"],
        "reserved_physical_http_attempts": 0,
        "actual_physical_http_attempts": 0,
        "attempted_task_ids": [],
        "completed_task_ids": [],
        "failure": None,
    }
    _write_execution_state(execution_state_path, execution_state, create=True)

    rows = []
    new_model_calls = 0
    physical_http_attempts = 0
    for task in cases:
        task_id = str(task["task_id"])
        if task_id in saved:
            rows.append(saved[task_id])
            continue
        docs = task_input_docs(task)
        system, user = requests[task_id]
        if len(execution_state["attempted_task_ids"]) >= current_plan["logical_request_budget"]:
            raise RuntimeError("grounding logical request budget exhausted")
        reservation = current_plan["max_http_attempts_per_logical_request"]
        if (execution_state["reserved_physical_http_attempts"] + reservation
                > current_plan["physical_request_budget"]):
            raise RuntimeError("grounding physical request budget exhausted")
        execution_state["reserved_physical_http_attempts"] += reservation
        execution_state["attempted_task_ids"].append(task_id)
        _write_execution_state(execution_state_path, execution_state)
        try:
            result = provider.chat_measured(system, user, max_tokens=450)
        except Exception as exc:
            attempts = int(getattr(provider, "_last_http_attempts", reservation) or 0)
            attempts = min(max(0, attempts), reservation)
            physical_http_attempts += attempts
            execution_state["actual_physical_http_attempts"] = physical_http_attempts
            execution_state["failure"] = {
                "task_id": task_id,
                "exception_type": type(exc).__name__,
                "error_detail_omitted": True,
            }
            execution_state["status"] = "failed_no_replay"
            _write_execution_state(execution_state_path, execution_state)
            raise RuntimeError(
                f"grounding provider call failed for {task_id}; details omitted; "
                "the reserved plan cannot be replayed") from None
        attempts = int(getattr(provider, "_last_http_attempts", 1) or 0)
        attempts = min(max(0, attempts), reservation)
        physical_http_attempts += attempts
        execution_state["actual_physical_http_attempts"] = physical_http_attempts
        execution_state["completed_task_ids"].append(task_id)
        _write_execution_state(execution_state_path, execution_state)
        new_model_calls += 1
        gold = str(task["options"][int(task["correct_index"])]).strip().casefold()
        opposite = "no" if gold == "yes" else "yes"
        gold_supported, gold_reason = parse_proposition_verdict(
            result.content, docs, gold)
        opposite_supported, opposite_reason = parse_proposition_verdict(
            result.content, docs, opposite)
        row = {
            "protocol": protocol, "task_id": task_id, "gold_answer": gold,
            "gold_supported": gold_supported, "gold_reason": gold_reason,
            "opposite_answer": opposite,
            "opposite_supported": opposite_supported,
            "opposite_reason": opposite_reason,
            "raw_verdict": result.content,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "evidence_chars": sum(len(doc["text"]) for doc in docs),
        }
        rows.append(row)
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    execution_state["status"] = "completed"
    _write_execution_state(execution_state_path, execution_state)

    yes = [row for row in rows if row["gold_answer"] == "yes"]
    no = [row for row in rows if row["gold_answer"] == "no"]
    false_support = [row for row in rows if row["opposite_supported"]]
    report = {
        "status": "exploratory_task_input_grounding_calibration",
        "dataset": "pubmedqa", "split": "dev_optimize",
        "model_id": args.model_id, "protocol": protocol,
        "plan_sha256": current_plan["plan_sha256"],
        "logical_request_budget": current_plan["logical_request_budget"],
        "physical_request_budget": current_plan["physical_request_budget"],
        "physical_request_upper_bound": current_plan["physical_request_upper_bound"],
        "physical_http_attempts": physical_http_attempts,
        "judge_prompt_version": args.prompt_version,
        "selection": ("task-id SHA-256 hash, balanced yes/no, offset per label="
                      f"{args.offset_per_label}, supplied evidence only"),
        "n_tasks": len(rows), "n_model_calls": new_model_calls,
        "n_cached_rows": len(rows) - new_model_calls,
        "gold_answer_support_rate": sum(row["gold_supported"] for row in rows) / len(rows),
        "gold_answer_support_by_label": {
            label: {"supported": sum(row["gold_supported"] for row in group),
                   "n": len(group)}
            for label, group in (("yes", yes), ("no", no))},
        "opposite_false_support_rate": len(false_support) / len(rows),
        "opposite_false_support_task_ids": [row["task_id"] for row in false_support],
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "completion_tokens": sum(row["completion_tokens"] for row in rows),
        "rows": rows,
        "limit": "Known PubMedQA labels are answer correctness labels, not direct human entailment labels. This is a task-matched evaluator diagnostic on an explored split, not a grounding gold benchmark or confirmation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "n_tasks", "gold_answer_support_rate", "gold_answer_support_by_label",
        "opposite_false_support_rate", "opposite_false_support_task_ids")},
        ensure_ascii=False, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
