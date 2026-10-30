"""Reconstruct the task-demo nested-confirmation error transitions.

This is a read-only diagnostic over already-used dev_optimize confirmation IDs.
It is not a fresh evaluation and must not be used to select another candidate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.options import inject_insufficient
from src.util import write_json


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _state(row: Dict[str, Any]) -> str:
    if int(row.get("answer_index", -1)) < 0:
        return "abstain"
    return "correct" if bool(row.get("correct")) else "wrong"


def _answer_label(answer_index: int, options: List[str]) -> str:
    if answer_index < 0:
        return "abstain"
    if answer_index >= len(options):
        raise ValueError(f"answer index {answer_index} outside {len(options)} options")
    return str(options[answer_index])


def _assistant_output(path: Path) -> str:
    last = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        for entry in value if isinstance(value, list) else [value]:
            if (isinstance(entry, dict) and entry.get("type") == "message"
                    and entry.get("message", {}).get("role") == "assistant"):
                content = entry["message"].get("content", [])
                last = "".join(item.get("text", "") for item in content
                               if isinstance(item, dict))
    return last


def analyze(report_path: Path, round_path: Path, dataset_path: Path,
            output_path: Path) -> Dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    round_report = json.loads(round_path.read_text(encoding="utf-8"))
    confirmation = report.get("internal_confirmation", {})
    if report.get("search_protocol_version") not in {
            "task-demo-search-v3-nested-confirmation",
            "task-demo-search-v4-class-nonregression",
            "task-demo-search-v5-class-hardcase-nonregression",
            "task-demo-search-v6-cyclic-demo-order-nonregression"}:
        raise ValueError("expected a supported nested-confirmation search report")
    if confirmation.get("status") != "passed":
        raise ValueError("confirmation must have passed before it can be diagnosed")
    if round_report.get("round_id") is None:
        raise ValueError("round report identity missing")

    train_ids = set(map(str, report.get("train_ids", [])))
    confirm_ids = set(map(str, report.get("confirmation_ids", [])))
    demo_ids = set(map(str, confirmation.get("candidate_demo_ids", [])))
    if train_ids & confirm_ids or demo_ids & confirm_ids:
        raise ValueError("confirmation IDs overlap train/demo IDs")

    tasks = {str(row["task_id"]): row for row in _read_jsonl(dataset_path)}
    baseline_run_ids = report.get("baseline_seed_run_ids", [])
    seeds = [int(row["seed"]) for row in confirmation.get("baseline_metrics_by_seed", [])]
    if len(seeds) != len(baseline_run_ids):
        raise ValueError("baseline seed/run mapping is incomplete")
    baselines: Dict[tuple[int, str], Dict[str, Any]] = {}
    for seed, run_id in zip(seeds, baseline_run_ids):
        path = report_path.parent.parent.parent / "runs" / str(run_id) / "per_task.jsonl"
        # report_path is .../search/task_demos/<search-id>/report.json
        if not path.exists():
            path = report_path.parents[3] / "runs" / str(run_id) / "per_task.jsonl"
        for row in _read_jsonl(path):
            baselines[(seed, str(row["task_id"]))] = row

    transitions: Dict[int, Counter[str]] = defaultdict(Counter)
    per_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen = set()
    for candidate in confirmation.get("rows", []):
        seed = int(candidate["seed"])
        task_id = str(candidate["task_id"])
        key = (seed, task_id)
        if key in seen:
            raise ValueError(f"duplicate confirmation row: {key}")
        seen.add(key)
        if task_id not in confirm_ids or task_id not in tasks:
            raise ValueError(f"unexpected or missing dataset confirmation ID: {task_id}")
        baseline = baselines.get(key)
        if baseline is None:
            raise ValueError(f"baseline row missing for {key}")
        task = inject_insufficient(tasks[task_id])
        if int(baseline.get("gold_index", -99)) != int(task["correct_index"]):
            raise ValueError(f"baseline gold index mismatch for {key}")
        options = list(task["options"])
        baseline_label = _answer_label(int(baseline.get("answer_index", -1)), options)
        candidate_label = _answer_label(int(candidate.get("answer_index", -1)), options)
        transition = f"{_state(baseline)} -> {_state(candidate)}"
        transitions[seed][transition] += 1
        if transition not in ("correct -> correct", "wrong -> wrong",
                              "abstain -> abstain"):
            candidate_output = _assistant_output(Path(candidate["pi_session_path"]))
            raw_match = re.search(r"(?im)^\s*ANSWER:\s*(\d+)\s*$", candidate_output)
            raw_label = (_answer_label(int(raw_match.group(1)), options)
                         if raw_match else "unparsed")
            per_task[task_id].append({
                "seed": seed,
                "transition": transition,
                "baseline_answer": baseline_label,
                "candidate_answer": candidate_label,
                "candidate_raw_output_answer": raw_label,
                "baseline_correct": bool(baseline.get("correct")),
                "candidate_correct": bool(candidate.get("correct")),
                "question": task.get("question", ""),
                "options": options,
                "gold_answer": options[int(task["correct_index"])],
                "evidence": task.get("evidence", ""),
            })

    expected = len(confirm_ids) * len(seeds)
    if len(seen) != expected:
        raise ValueError(f"incomplete confirmation pairs: {len(seen)} != {expected}")

    demo_labels = []
    for task_id in sorted(demo_ids):
        task = inject_insufficient(tasks[task_id])
        demo_labels.append({"task_id": task_id,
                            "answer": task["options"][int(task["correct_index"])],
                            "question": task.get("question", "")})

    result = {
        "analysis_type": "previously_used_nested_confirmation_error_diagnosis",
        "selection_or_confirmation_evidence": False,
        "search_protocol_version": report["search_protocol_version"],
        "round_id": round_report["round_id"],
        "candidate_id": round_report["candidates"][0].get("candidate_id"),
        "seeds": seeds,
        "confirmation_ids": sorted(confirm_ids),
        "train_ids_excluded": True,
        "selection_gate_read": False,
        "paired_row_count": len(seen),
        "transitions_by_seed": {
            str(seed): dict(sorted(counts.items()))
            for seed, counts in sorted(transitions.items())},
        "changed_tasks": {task_id: sorted(rows, key=lambda row: row["seed"])
                          for task_id, rows in sorted(per_task.items())},
        "selected_demo_gold_answers": demo_labels,
        "interpretation_limits": [
            "These IDs already participated in winner confirmation; diagnosis only, not new efficacy evidence.",
            "PubMedQA gold correctness is not an independent evidence-grounding judgment.",
            "Do not use this report to rank or select another candidate.",
        ],
    }
    write_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-report", type=Path, required=True)
    parser.add_argument("--round-report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/normalized/pubmedqa.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.search_report, args.round_report, args.dataset, args.output)
    print(json.dumps({"output": str(args.output),
                      "paired_rows": result["paired_row_count"],
                      "transitions_by_seed": result["transitions_by_seed"],
                      "changed_task_ids": list(result["changed_tasks"])},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
