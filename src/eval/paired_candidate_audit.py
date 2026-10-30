"""Item-level audit of paired candidate outputs, including non-scoring changes."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable


def _session_sha256(row: dict) -> str | None:
    raw_path = row.get("pi_session_path")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_paired_rows(baseline_rows: Iterable[dict], candidate_rows: Iterable[dict]) -> tuple[dict, list[dict]]:
    """Return accurate answer-change and correctness-transition counts and rows.

    A row is paired by task ID and execution label (``seed``); repeated labels
    remain clustered at task level and are not treated as independent tasks.
    """
    def indexed(rows: Iterable[dict], arm: str) -> dict[tuple[str, str], dict]:
        result = {}
        for row in rows:
            task_id = str(row.get("task_id") or "")
            seed = str(row.get("seed") or "")
            if not task_id or not seed:
                raise ValueError(f"{arm} row is missing task_id or execution label")
            key = (task_id, seed)
            if key in result:
                raise ValueError(f"duplicate {arm} pair key: {key}")
            result[key] = row
        return result

    baseline = indexed(baseline_rows, "baseline")
    candidate = indexed(candidate_rows, "candidate")
    if baseline.keys() != candidate.keys():
        raise ValueError("baseline and candidate pair keys do not match")

    # Derive option text from the per-row gold index/label pair, checking that
    # the index-to-label mapping is stable throughout the compared cohort.
    option_labels: dict[int, str] = {}
    for row in (*baseline.values(), *candidate.values()):
        index = row.get("gold_index")
        label = str(row.get("gold_label") or "").strip()
        if type(index) is not int or not label:
            raise ValueError("row is missing a valid gold_index or gold_label")
        prior = option_labels.setdefault(index, label)
        if prior.casefold() != label.casefold():
            raise ValueError(f"gold label mapping is inconsistent for option {index}")

    changed_rows: list[dict] = []
    answer_change_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    changed_tasks: set[str] = set()
    transition_tasks: dict[str, set[str]] = {"improved": set(), "regressed": set()}
    for key in sorted(baseline, key=lambda item: (item[0], item[1])):
        before, after = baseline[key], candidate[key]
        before_index, after_index = before.get("answer_index"), after.get("answer_index")
        before_correct, after_correct = before.get("correct"), after.get("correct")
        if (before.get("gold_index") != after.get("gold_index")
                or str(before.get("gold_label") or "").casefold()
                != str(after.get("gold_label") or "").casefold()):
            raise ValueError(f"paired gold labels differ for pair {key}")
        if type(before_index) is not int or type(after_index) is not int:
            raise ValueError(f"invalid answer index for pair {key}")
        if type(before_correct) is not bool or type(after_correct) is not bool:
            raise ValueError(f"invalid correctness flag for pair {key}")
        if before_correct != (before_index == before.get("gold_index")):
            raise ValueError(f"baseline correctness does not match indices for pair {key}")
        if after_correct != (after_index == after.get("gold_index")):
            raise ValueError(f"candidate correctness does not match indices for pair {key}")

        if before_index != after_index:
            category = ("wrong_to_correct" if not before_correct and after_correct else
                        "correct_to_wrong" if before_correct and not after_correct else
                        "wrong_to_wrong" if not before_correct and not after_correct else
                        "correct_to_correct")
            answer_change_counts[category] += 1
            changed_tasks.add(key[0])
            changed_rows.append({
                "task_id": key[0], "execution_label": key[1],
                "gold_index": before["gold_index"], "gold_label": before["gold_label"],
                "baseline_answer_index": before_index,
                "baseline_answer_label": option_labels.get(before_index, "abstain" if before_index == -1 else None),
                "baseline_correct": before_correct,
                "candidate_answer_index": after_index,
                "candidate_answer_label": option_labels.get(after_index, "abstain" if after_index == -1 else None),
                "candidate_correct": after_correct,
                "transition": category,
                "baseline_skill": before.get("skill_recalled"),
                "candidate_skill": after.get("skill_recalled"),
                "baseline_session_path": before.get("pi_session_path"),
                "baseline_session_sha256": _session_sha256(before),
                "candidate_session_path": after.get("pi_session_path"),
                "candidate_session_sha256": _session_sha256(after),
            })
        if before_correct != after_correct:
            category = "improved" if after_correct else "regressed"
            transition_counts[category] += 1
            transition_tasks[category].add(key[0])

    summary = {
        "paired_row_count": len(baseline),
        "unique_task_count": len({task_id for task_id, _ in baseline}),
        "answer_index_change_count": sum(answer_change_counts.values()),
        "answer_index_change_by_outcome": dict(sorted(answer_change_counts.items())),
        "answer_index_changed_unique_task_count": len(changed_tasks),
        "correctness_transition_count": sum(transition_counts.values()),
        "correctness_transitions": dict(sorted(transition_counts.items())),
        "correctness_transition_unique_task_count": len(transition_tasks["improved"] | transition_tasks["regressed"]),
        "improved_unique_tasks": len(transition_tasks["improved"]),
        "regressed_unique_tasks": len(transition_tasks["regressed"]),
        "unit_of_analysis": "task_id; execution labels are repeated observations clustered within task",
    }
    return summary, changed_rows
