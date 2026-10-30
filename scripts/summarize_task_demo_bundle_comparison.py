"""Summarize a completed fixed task-demo bundle comparison report."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    answered = sum(int(row["answer_index"]) >= 0 for row in rows)
    return {"n_calls": n, "correct": correct, "answered": answered,
            "accuracy": correct / n if n else 0.0,
            "coverage": answered / n if n else 0.0,
            "answered_precision": correct / answered if answered else 0.0}


def _direct_cluster_ci(pairs: List[Dict[str, Any]], first: str, second: str,
                       seed: int = 20261004, samples: int = 20000
                       ) -> Dict[str, List[float]]:
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for pair in pairs:
        by_id.setdefault(str(pair["task_id"]), []).append(pair)
    ids = sorted(by_id)
    rng = random.Random(seed)
    deltas = {metric: [] for metric in ("accuracy", "coverage", "answered_precision")}
    for _ in range(samples):
        sampled = [ids[rng.randrange(len(ids))] for __ in ids]
        a = [pair["candidates"][first] for task_id in sampled
             for pair in by_id[task_id]]
        b = [pair["candidates"][second] for task_id in sampled
             for pair in by_id[task_id]]
        ma, mb = _metrics(a), _metrics(b)
        for metric in deltas:
            deltas[metric].append(mb[metric] - ma[metric])
    result = {}
    for metric, values in deltas.items():
        values.sort()
        result[metric] = [values[int(0.025 * samples)],
                          values[min(samples - 1, int(0.975 * samples))]]
    return result


def summarize(report: Dict[str, Any]) -> Dict[str, Any]:
    pairs = report["pairs"]
    candidates = list(report["candidate_ids"])
    if len(candidates) != 2:
        raise ValueError("comparison report must contain exactly two candidates")
    first, second = candidates
    task_ids = sorted({str(pair["task_id"]) for pair in pairs})
    seeds = sorted({int(pair["seed"]) for pair in pairs})
    per_seed = {}
    for seed in seeds:
        rows = [pair for pair in pairs if int(pair["seed"]) == seed]
        per_seed[str(seed)] = {"baseline": _metrics([p["baseline"] for p in rows]),
                               first: _metrics([p["candidates"][first] for p in rows]),
                               second: _metrics([p["candidates"][second] for p in rows])}
    pooled = {arm: _metrics([pair[arm] for pair in pairs]) for arm in ["baseline"]}
    for candidate in candidates:
        pooled[candidate] = _metrics([pair["candidates"][candidate] for pair in pairs])
        pooled[candidate]["delta_vs_baseline"] = {
            metric: pooled[candidate][metric] - pooled["baseline"][metric]
            for metric in ("accuracy", "coverage", "answered_precision")}
    direct = {metric: pooled[second][metric] - pooled[first][metric]
              for metric in ("accuracy", "coverage", "answered_precision")}
    changed_answers = set()
    transition_counts = Counter()
    second_correct_first_wrong = 0
    first_correct_second_wrong = 0
    for pair in pairs:
        a, b = pair["candidates"][first], pair["candidates"][second]
        if int(a["answer_index"]) != int(b["answer_index"]):
            changed_answers.add(str(pair["task_id"]))
            def state(row: Dict[str, Any]) -> str:
                if int(row["answer_index"]) < 0:
                    return "abstain"
                return "correct" if row["correct"] else "wrong"
            transition_counts[f"{state(a)} -> {state(b)}"] += 1
        if b["correct"] and not a["correct"]:
            second_correct_first_wrong += 1
        if a["correct"] and not b["correct"]:
            first_correct_second_wrong += 1
    return {
        "protocol": "task-demo-bundle-comparison-summary-v1",
        "source_report": report.get("output_path"),
        "candidate_ids": candidates,
        "unique_tasks": len(task_ids),
        "repeats": seeds,
        "per_seed": per_seed,
        "pooled_descriptive": pooled,
        "direct_second_minus_first": direct,
        "direct_task_cluster_bootstrap_95ci": _direct_cluster_ci(
            pairs, first, second),
        "direct_correctness_discordant_calls": {
            "second_correct_first_wrong": second_correct_first_wrong,
            "first_correct_second_wrong": first_correct_second_wrong},
        "direct_answer_transition_counts": dict(sorted(transition_counts.items())),
        "unique_ids_with_answer_changes": len(changed_answers),
        "interpretation_limits": [
            "Repeated requests share task IDs; the bootstrap resamples task clusters.",
            "All rows are exploratory dev_optimize evidence, not sealed confirmation.",
            "Different bundles change example content as well as label composition.",
            "A static source-integrity gate does not certify answer entailment.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    summary = summarize(report)
    summary["source_report"] = str(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "direct_second_minus_first": summary["direct_second_minus_first"],
                      "direct_ci": summary["direct_task_cluster_bootstrap_95ci"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
