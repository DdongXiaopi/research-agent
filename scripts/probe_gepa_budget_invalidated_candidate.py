"""Replay a GEPA candidate whose validation was invalidated by budget exhaustion.

This is a diagnostic-only replay on the already exposed dev_optimize validation
IDs. It must not be used to select/promote the candidate or represented as new
independent evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_fingerprint, load_config  # noqa: E402
from src.contracts.types import Candidate, content_id, content_sha256  # noqa: E402
from src.eval.options import inject_all  # noqa: E402
from src.evolution.gepa_search import visible_examples  # noqa: E402
from src.evolution.prompt_rewrite import validate_rewrite  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.runner import ArtifactState, TaskRunner  # noqa: E402
from src.util import artifacts_path, read_json, sha256_obj, write_json  # noqa: E402


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    answered = sum(int(row.get("answer_index", -1)) >= 0 for row in rows)
    correct = sum(bool(row.get("correct")) for row in rows)
    answered_correct = sum(
        int(row.get("answer_index", -1)) >= 0 and bool(row.get("correct"))
        for row in rows)
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "coverage": answered / n if n else 0.0,
        "answered_precision": answered_correct / answered if answered else 0.0,
        "abstentions": n - answered,
        "wrong_answered": answered - answered_correct,
        "runtime_errors": sum(row.get("status") == "error" for row in rows),
    }


def setup_task_input_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("runtime", {}).update({
        "provider": "pi", "input_mode": "task_input", "task_input_max_chars": 30000})
    cfg["runtime"].pop("pi_fake_response", None)
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg.get("tools", {}).get("active", [])
        if name != "academic.search"]
    cfg.setdefault("evaluation", {})["main_metric"] = "accuracy"
    cfg["evaluation"]["primary_axis"] = "accuracy"
    cfg["evaluation"]["sealed_pairing"] = False
    return cfg


def replay(*, config_path: str, report_path: str, candidate_index: int,
           seed: int) -> Dict[str, Any]:
    raise RuntimeError(
        "This legacy GEPA diagnostic replay is disabled: it scores a candidate "
        "on 60 exposed validation tasks without a frozen execution-plan hash "
        "or enforced request ledger. No report, config, task rows, release "
        "state, or TaskRunner will be loaded. Use a reviewed plan-bound "
        "diagnostic entrypoint instead."
    )
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("Set RSI_HOME to the isolated GEPA sandbox")
    search_report = read_json(report_path)
    if search_report.get("status") != "budget_exhausted_no_selection":
        raise ValueError("This probe only accepts a budget-invalidated GEPA report")
    if search_report.get("selection_gate_read") is not False:
        raise ValueError("Refusing a search report that read selection_gate")
    candidates = search_report.get("candidate_texts") or []
    if not 0 < candidate_index < len(candidates):
        raise ValueError("candidate_index must name a nonbaseline candidate")

    cfg = setup_task_input_config(load_config(config_path))
    state = load_state(search_report.get("release_id"))
    train, val = visible_examples(cfg, ["pubmedqa"], limit=60)
    expected_val_ids = [str(row["task_id"]) for row in val]
    if expected_val_ids != [str(x) for x in search_report.get("val_ids", [])]:
        raise ValueError("Visible validation IDs do not match the search report")

    prompt = str(candidates[candidate_index]).strip()
    candidate = Candidate(
        candidate_id=content_id("cand", state.release_id, "system_prompt", prompt),
        parent_release_id=state.release_id, artifact_type="system_prompt",
        artifact_id="system_prompt", diff="diagnostic replay only",
        new_content=prompt, content_sha256=content_sha256(prompt))
    reason = validate_rewrite(prompt, state.system_prompt, cfg)
    static = static_gate(candidate, cfg, state)
    if reason or not static.get("passed"):
        raise ValueError("Candidate failed pre-replay gate: "
                         + str(reason or static.get("reason") or "static gate"))

    seed_outputs_dir = (Path(report_path).parent / "state" / "iterations" / "seed" / "outputs")
    baseline_rows: List[Dict[str, Any]] = []
    for index, task in enumerate(val):
        saved = read_json(seed_outputs_dir / f"{index}.json")
        side = saved[2]
        if str(side.get("TaskID")) != str(task["task_id"]):
            raise ValueError(f"Baseline val row {index} does not match task ID")
        baseline_rows.append({
            "task_id": str(task["task_id"]),
            "answer_index": int(side.get("AnswerIndex", -1)),
            "gold_index": int(task["correct_index"]),
            "correct": str(side.get("Feedback", "")).lower() == "correct",
            "status": str(side.get("Status", "unknown")),
            "selected": str(side.get("Selected", "")),
            "run_id": str(side.get("RunID", "")),
        })

    trial_state = ArtifactState(
        release_id=state.release_id, system_prompt=prompt,
        registry=state.registry, task_demos=state.task_demos)
    candidate_rows: List[Dict[str, Any]] = []
    for task in val:
        outcome = TaskRunner(cfg, trial_state, seed=seed).run(task)
        options = task.get("options") or []
        selected = (options[outcome.answer_index]
                    if 0 <= outcome.answer_index < len(options) else "ABSTAIN")
        candidate_rows.append({
            "task_id": str(outcome.task_id), "answer_index": outcome.answer_index,
            "gold_index": outcome.gold_index, "correct": bool(outcome.correct),
            "status": outcome.status, "selected": selected,
            "cited": list(outcome.cited), "citation_precision": outcome.citation_precision,
            "error": outcome.error, "run_id": str(outcome.events[0].get("run_id", "")
                                                       if outcome.events else ""),
            "pi_session_path": outcome.pi_session_path,
        })

    baseline_by_id = {row["task_id"]: row for row in baseline_rows}
    transitions = {"correct_to_wrong": 0, "wrong_to_correct": 0,
                  "answer_to_abstain": 0, "abstain_to_answer": 0,
                  "unchanged_correctness": 0}
    for row in candidate_rows:
        base = baseline_by_id[row["task_id"]]
        if base["correct"] and not row["correct"]:
            transitions["correct_to_wrong"] += 1
        elif not base["correct"] and row["correct"]:
            transitions["wrong_to_correct"] += 1
        else:
            transitions["unchanged_correctness"] += 1
        if base["answer_index"] >= 0 and row["answer_index"] < 0:
            transitions["answer_to_abstain"] += 1
        elif base["answer_index"] < 0 and row["answer_index"] >= 0:
            transitions["abstain_to_answer"] += 1

    return {
        "protocol": "budget-invalidated-gepa-candidate-diagnostic-replay-v1",
        "diagnostic_only": True,
        "candidate_selection_allowed": False,
        "search_report": str(Path(report_path).resolve()),
        "search_report_sha256": sha256_obj(search_report),
        "search_protocol_sha256": search_report.get("protocol_sha256"),
        "config_sha256": config_fingerprint(cfg),
        "release_id": state.release_id, "candidate_index": candidate_index,
        "candidate_sha256": content_sha256(prompt), "seed": seed,
        "dataset": "pubmedqa", "split": "dev_optimize",
        "task_ids": expected_val_ids,
        "metrics": {"baseline": summarize_rows(baseline_rows),
                    "candidate": summarize_rows(candidate_rows)},
        "transitions": transitions,
        "baseline_rows": baseline_rows, "candidate_rows": candidate_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--search-report", required=True)
    parser.add_argument("--candidate-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261035)
    args = parser.parse_args()
    result = replay(config_path=args.config, report_path=args.search_report,
                    candidate_index=args.candidate_index, seed=args.seed)
    path = artifacts_path("probe", "gepa_budget_diagnostic",
                          "replay-" + uuid.uuid4().hex[:12] + ".json")
    write_json(path, result)
    print(json.dumps({"path": str(path), "diagnostic_only": True,
                      "metrics": result["metrics"],
                      "transitions": result["transitions"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
