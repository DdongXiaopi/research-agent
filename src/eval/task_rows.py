"""Validation helpers for persisted per-task evaluator rows."""

from __future__ import annotations


def validate_task_answer_row(row: dict, task: dict, *, context: str,
                             require_reported_gold_index: bool = False) -> dict:
    """Validate status, answer and correctness against the task's frozen gold.

    ``TaskRunner`` uses ``status='abstain'`` for a valid no-answer result, so
    abstention is a scored outcome rather than an execution failure. Scoring
    boundaries that consume a serialized ``TaskOutcome`` can also require its
    reported gold index to match the frozen task exactly.
    """
    task_id = str(task.get("task_id") or "")
    if str(row.get("task_id") or "") != task_id:
        raise ValueError(f"{context} task ID differs from the frozen task")
    status = str(row.get("status") or "")
    raw_answer_index = row.get("answer_index")
    raw_gold_index = task.get("correct_index")
    if (type(raw_answer_index) is not int or type(raw_gold_index) is not int
            or not isinstance(row.get("correct"), bool)):
        raise ValueError(f"{context} has malformed answer/gold index or correct flag")
    if require_reported_gold_index:
        reported_gold_index = row.get("gold_index")
        if (type(reported_gold_index) is not int
                or reported_gold_index != raw_gold_index):
            raise ValueError(
                f"{context} reported gold index differs from the frozen task")
    answer_index = raw_answer_index
    gold_index = raw_gold_index
    options = task.get("options") or []
    if not 0 <= gold_index < len(options):
        raise ValueError(f"{context} task has malformed gold option")
    if answer_index < -1 or answer_index >= len(options):
        raise ValueError(f"{context} answer index is outside the task options")
    expected_status = "abstain" if answer_index == -1 else "ok"
    if status != expected_status:
        raise ValueError(
            f"{context} status {status!r} disagrees with answer index {answer_index}")
    recomputed_correct = answer_index >= 0 and answer_index == gold_index
    if row["correct"] != recomputed_correct:
        raise ValueError(f"{context} correct flag disagrees with answer and gold indices")
    gold_label = str(options[gold_index]).strip().casefold()
    if (row.get("gold_label") is not None
            and str(row["gold_label"]).strip().casefold() != gold_label):
        raise ValueError(f"{context} gold_label disagrees with the frozen task")
    return {**row, "task_id": task_id, "answer_index": answer_index,
            "correct": recomputed_correct, "gold_label": gold_label}
