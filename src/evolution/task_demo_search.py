"""Visible-dev search over labeled task-model demonstrations.

This is a bounded, MIPROv2-inspired demo subset search. It does not claim to be
the DSPy MIPROv2 optimizer; the outer round still performs full optimize/gate
evaluation and review for the selected demo bundle.
"""

from __future__ import annotations

import re
import random
import uuid
from collections import Counter
from typing import Any, Dict, List, Sequence

from ..config import config_fingerprint
from ..contracts.types import Candidate, content_id, content_sha256
from ..eval.metrics import total_cost
from ..eval.task_rows import validate_task_answer_row
from ..evolution.fewshot import parse_demos_bundle, render_demos_block, serialize_demos_bundle
from ..evolution.task_program import parse_task_program, serialize_task_program
from ..governance.policy import static_gate
from ..runtime.runner import ArtifactState, TaskRunner
from ..util import artifacts_path, read_jsonl, sha256_obj, write_json
from .gepa_search import visible_examples


SEARCH_PROTOCOL_VERSION = "task-demo-search-v6-cyclic-demo-order-nonregression"
DEFAULT_MAX_DEMO_CANDIDATE_SETS = 6


def _candidate_static_gate(candidate: Candidate, cfg: Dict[str, Any], state: Any) -> Dict[str, Any]:
    """Gate both the serialized artifact and its exact Pi demo rendering."""
    result = static_gate(candidate, cfg, state)
    if result.get("passed") is not True:
        return result
    try:
        demos = (parse_task_program(candidate.new_content)["task_demos"]
                 if candidate.artifact_type == "task_program"
                 else parse_demos_bundle(candidate.new_content))
    except (TypeError, ValueError, KeyError) as exc:
        check = {"check": "task_demo_rendering", "passed": False,
                 "detail": {"reason": "invalid_demo_artifact", "error": str(exc)}}
        return {**result, "passed": False,
                "results": [*(result.get("results") or []), check],
                "task_demo_rendering": check["detail"]}
    from ..evolution.task_input_gates import validate_task_demo_rendering

    rendering = validate_task_demo_rendering(demos, cfg)
    check = {"check": "task_demo_rendering",
             "passed": rendering.get("passed") is True,
             "detail": rendering}
    checks = [*(result.get("results") or []), check]
    return {**result, "passed": bool(result.get("passed") and check["passed"]),
            "results": checks, "task_demo_rendering": rendering}


def _search_protocol_sha256(cfg: Dict[str, Any]) -> str:
    return sha256_obj({"config_sha256": config_fingerprint(cfg),
                       "search_protocol_version": SEARCH_PROTOCOL_VERSION})


def _metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    correct = sum(bool(row.get("correct")) for row in rows)
    answered = sum(int(row.get("answer_index", -1)) >= 0 for row in rows)
    labels = sorted({str(row.get("gold_label") or "").strip().casefold()
                     for row in rows if str(row.get("gold_label") or "").strip()})
    per_label = {}
    for label in labels:
        subset = [row for row in rows
                  if str(row.get("gold_label") or "").strip().casefold() == label]
        label_answered = sum(int(row.get("answer_index", -1)) >= 0
                             for row in subset)
        label_correct = sum(bool(row.get("correct")) for row in subset)
        per_label[label] = {
            "n": len(subset),
            "accuracy": label_correct / len(subset),
            "coverage": label_answered / len(subset),
            "answered_precision": (label_correct / label_answered
                                   if label_answered else 0.0),
        }
    return {"n": n, "accuracy": correct / n if n else 0.0,
            "coverage": answered / n if n else 0.0,
            "answered_precision": correct / answered if answered else 0.0,
            "per_gold_label": per_label}


def _gold_label(task: Dict[str, Any]) -> str:
    options = task.get("options") or []
    try:
        index = int(task.get("correct_index", -1))
    except (TypeError, ValueError):
        return ""
    if 0 <= index < len(options):
        return str(options[index]).strip().casefold()
    return ""


def _nonregresses(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> bool:
    """Require aggregate and each represented gold class to avoid regression."""
    for metric in ("accuracy", "coverage", "answered_precision"):
        if float(candidate.get(metric, 0.0)) + 1e-9 < float(baseline.get(metric, 0.0)):
            return False
    candidate_labels = candidate.get("per_gold_label") or {}
    for label, baseline_metrics in (baseline.get("per_gold_label") or {}).items():
        candidate_metrics = candidate_labels.get(label)
        if candidate_metrics is None:
            return False
        for metric in ("accuracy", "coverage"):
            if (float(candidate_metrics.get(metric, 0.0)) + 1e-9
                    < float(baseline_metrics.get(metric, 0.0))):
                return False
    return True


def _paired_answer_transitions(
        candidate_rows: Sequence[Dict[str, Any]],
        baseline_rows_by_seed: Sequence[Sequence[Dict[str, Any]]],
        baseline_seed_numbers: Sequence[int],
        ) -> Dict[str, Any]:
    """Keep per-task candidate answers auditable against their paired baseline."""
    if len(baseline_rows_by_seed) != len(baseline_seed_numbers):
        raise RuntimeError("baseline rows and seed labels have different lengths")
    baseline_by_key = {
        (int(seed), str(row["task_id"])): row
        for seed, seed_rows in zip(baseline_seed_numbers, baseline_rows_by_seed)
        for row in seed_rows
    }
    transitions = []
    for candidate in candidate_rows:
        key = (int(candidate["seed"]), str(candidate["task_id"]))
        baseline = baseline_by_key.get(key)
        if baseline is None:
            raise RuntimeError(
                f"candidate task has no paired discovery baseline: seed={key[0]}, task={key[1]}")
        baseline_answer = int(baseline.get("answer_index", -1))
        candidate_answer = int(candidate.get("answer_index", -1))
        baseline_correct = bool(baseline.get("correct"))
        candidate_correct = bool(candidate.get("correct"))
        transitions.append({
            "seed": key[0], "task_id": key[1],
            "baseline_answer_index": baseline_answer,
            "candidate_answer_index": candidate_answer,
            "baseline_status": str(baseline.get("status") or ""),
            "candidate_status": str(candidate.get("status") or ""),
            "baseline_correct": baseline_correct,
            "candidate_correct": candidate_correct,
            "answer_changed": baseline_answer != candidate_answer,
            "correctness_transition": (
                "improved" if candidate_correct and not baseline_correct else
                "regressed" if baseline_correct and not candidate_correct else
                "unchanged_correct" if candidate_correct else "unchanged_incorrect"),
        })
    counts = Counter(item["correctness_transition"] for item in transitions)
    changed = sum(item["answer_changed"] for item in transitions)
    return {
        "paired_task_count": len(transitions),
        "answer_changed_count": changed,
        "answer_unchanged_count": len(transitions) - changed,
        "correctness_transitions": {
            name: counts.get(name, 0) for name in (
                "improved", "regressed", "unchanged_correct", "unchanged_incorrect")},
        "rows": transitions,
    }


def _validated_scored_outcome(outcome: Any, task: Dict[str, Any], *,
                              context: str) -> Dict[str, Any]:
    """Require the serialized TaskOutcome to agree with its frozen task."""
    to_dict = getattr(outcome, "to_dict", None)
    row = to_dict() if callable(to_dict) else vars(outcome)
    return validate_task_answer_row(
        row, task, context=context, require_reported_gold_index=True)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _candidate_sets(pool: Sequence[Dict[str, Any]],
                    max_sets: int = DEFAULT_MAX_DEMO_CANDIDATE_SETS,
                    k: int = 3, seed: int = 0,
                    hard_pool: Sequence[Dict[str, Any]] = (),
                    labeled_pool: Sequence[Dict[str, Any]] = (),
                    include_families: bool = False
                    ) -> List[List[Dict[str, Any]]] | tuple[
                        List[List[Dict[str, Any]]], List[str]]:
    """Generate bounded, deterministic class-balanced, bootstrapped and hard demos.

    ``pool`` contains examples the baseline answered correctly on a majority of
    repeated runs. ``hard_pool`` contains gold-labeled training failures.
    ``labeled_pool`` may contain any training examples with trusted labels.
    This follows MIPROv2's mix of bootstrapped and labeled demonstrations while
    keeping the project's exhaustive, budgeted search (not its Bayesian TPE).
    """
    ordered = sorted(pool, key=lambda item: str(item["task_id"]))
    hard = sorted(hard_pool, key=lambda item: str(item["task_id"]))
    labeled = sorted(labeled_pool, key=lambda item: str(item["task_id"]))
    sets: List[List[Dict[str, Any]]] = []
    families: List[str] = []

    def add(items: Sequence[Dict[str, Any]], family: str) -> None:
        chosen: List[Dict[str, Any]] = []
        seen = set()
        for item in items:
            task_id = str(item["task_id"])
            if task_id in seen:
                continue
            chosen.append(item)
            seen.add(task_id)
            if len(chosen) >= k:
                break
        ids = [str(item["task_id"]) for item in chosen]
        if chosen and len(ids) == len(set(ids)) and ids not in [
                [str(row["task_id"]) for row in existing] for existing in sets]:
            block = render_demos_block(chosen)
            if block:
                sets.append(chosen)
                families.append(family)

    def add_cyclic_orders(items: Sequence[Dict[str, Any]], family: str) -> None:
        """Search order effects without changing a bundle's demo membership."""
        if len(items) < 2:
            return
        for offset in range(1, len(items)):
            add([*items[offset:], *items[:offset]],
                f"{family}_rotation_{offset}")

    # Reserve the first candidate for class coverage from the gold-labeled
    # train pool; a small max_sets budget must not silently drop this family.
    labeled_by_label: Dict[str, List[Dict[str, Any]]] = {}
    for item in labeled:
        label = _gold_label(item.get("task") or {})
        if label:
            labeled_by_label.setdefault(label, []).append(item)
    hard_by_label: Dict[str, List[Dict[str, Any]]] = {}
    for item in hard:
        label = _gold_label(item.get("task") or {})
        if label:
            hard_by_label.setdefault(label, []).append(item)
    hard_ids = {str(item["task_id"]) for item in hard}
    balanced_labeled: List[Dict[str, Any]] = []
    labeled_labels = sorted(labeled_by_label)
    for index in range(k):
        if not labeled_labels:
            break
        label = labeled_labels[index % len(labeled_labels)]
        position = index // len(labeled_labels)
        nonhard = [item for item in labeled_by_label[label]
                   if str(item["task_id"]) not in hard_ids]
        candidates = nonhard or labeled_by_label[label]
        if position < len(candidates):
            balanced_labeled.append(candidates[position])
    add(balanced_labeled, "class_balanced_labeled")

    # Put a training-error-informed family in the early search budget. A small
    # max_sets value must not spend every slot on baseline-correct examples.
    if hard:
        balanced_hard: List[Dict[str, Any]] = []
        for index in range(k):
            if not labeled_labels:
                break
            label = labeled_labels[index % len(labeled_labels)]
            position = index // len(labeled_labels)
            candidates = hard_by_label.get(label) or labeled_by_label.get(label, [])
            if position < len(candidates):
                balanced_hard.append(candidates[position])
        add(balanced_hard, "class_balanced_hard_cases")

    add(ordered, "baseline_correct")
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for item in ordered:
        task = item["task"]
        label = str(task["options"][int(item["correct_index"])]).casefold()
        by_label.setdefault(label, []).append(item)
    balanced: List[Dict[str, Any]] = []
    labels = sorted(by_label)
    for index in range(k):
        label = labels[index % len(labels)] if labels else ""
        if not label:
            break
        candidates = by_label[label]
        if index // len(labels) < len(candidates):
            balanced.append(candidates[index // len(labels)])
    add(balanced, "class_balanced_baseline_correct")

    # The task model sees demonstrations in sequence. Preserve the same
    # class-balanced bundle but rotate which example appears first; otherwise
    # a small candidate budget confounds demo identity with prompt position.
    # These variants are distinct programs and are covered by the same hard
    # call budget as every other candidate set.
    add_cyclic_orders(balanced_labeled, "class_balanced_labeled_order")
    if hard:
        add_cyclic_orders(balanced_hard, "class_balanced_hard_order")
    add_cyclic_orders(balanced, "class_balanced_baseline_correct_order")

    diverse: List[Dict[str, Any]] = []
    remaining = list(ordered)
    while remaining and len(diverse) < k:
        if not diverse:
            selected = remaining[0]
        else:
            current = [_tokens(str(x["task"].get("question", "")) + " "
                               + str(x["task"].get("evidence", ""))) for x in diverse]
            selected = max(remaining, key=lambda item: min(
                1.0 - len(tok & old) / max(1, len(tok | old))
                for tok in [_tokens(str(item["task"].get("question", "")) + " "
                                    + str(item["task"].get("evidence", "")))]
                for old in current))
        diverse.append(selected)
        remaining.remove(selected)
    # Reserve early slots for diagnostic examples that the baseline missed.
    # These still use the source dataset's gold labels and remain training-only.
    if hard:
        add([*hard, *diverse], "hard_cases_plus_diverse")
        add(hard, "hard_cases_only")
    else:
        add(diverse, "diverse_baseline_correct")

    rng = random.Random(int(seed))
    # MIPROv2 also evaluates labeled-only demonstrations. These examples are
    # not filtered by baseline correctness: their labels come from train data.
    shuffled_labeled = list(labeled)
    rng.shuffle(shuffled_labeled)
    if shuffled_labeled:
        labeled_size = rng.randint(1, min(k, len(shuffled_labeled)))
        add(shuffled_labeled[:labeled_size], "random_labeled_subset")

    # Mix hard cases with correctly answered demonstrations before falling
    # back to random subsets from the complete gold-labeled training pool.
    if hard and ordered:
        mixed: List[Dict[str, Any]] = [hard[0]]
        mixed.extend(item for item in diverse
                     if str(item["task_id"]) != str(hard[0]["task_id"]))
        add(mixed, "mixed_hard_and_correct")

    attempts = 0
    max_attempts = max(100, max_sets * 30)
    random_pool = sorted({str(item["task_id"]): item
                          for item in [*ordered, *hard, *labeled]}.values(),
                         key=lambda item: str(item["task_id"]))
    while random_pool and len(sets) < max_sets and attempts < max_attempts:
        attempts += 1
        shuffled = list(random_pool)
        rng.shuffle(shuffled)
        add(shuffled[:rng.randint(1, min(k, len(shuffled)))], "random_labeled_subset")
    selected_sets = sets[:max_sets]
    selected_families = families[:max_sets]
    return (selected_sets, selected_families) if include_families else selected_sets


def _majority_correct_task_ids(rows_by_seed: Sequence[Sequence[Dict[str, Any]]]
                               ) -> set[str]:
    """Keep demos answered correctly in a strict majority of baseline seeds."""
    if not rows_by_seed:
        return set()
    votes: Dict[str, int] = {}
    for rows in rows_by_seed:
        for row in rows:
            task_id = str(row.get("task_id") or "")
            if task_id and bool(row.get("correct")):
                votes[task_id] = votes.get(task_id, 0) + 1
    return {task_id for task_id, count in votes.items()
            if count > len(rows_by_seed) / 2}


def _worst_seed_gain(item: Dict[str, Any], metric: str,
                     baseline_metrics_by_seed: Sequence[Dict[str, Any]]) -> float:
    gains = []
    for index, result in enumerate(item.get("seed_metrics") or []):
        candidate_value = float(result["metrics"][metric])
        baseline_value = float(baseline_metrics_by_seed[index][metric])
        gains.append(candidate_value - baseline_value)
    return min(gains) if gains else float("-inf")


def _worst_seed_class_gain(item: Dict[str, Any], metric: str,
                           baseline_metrics_by_seed: Sequence[Dict[str, Any]]) -> float:
    gains = []
    for index, result in enumerate(item.get("seed_metrics") or []):
        candidate_labels = result["metrics"].get("per_gold_label") or {}
        baseline_labels = baseline_metrics_by_seed[index].get("per_gold_label") or {}
        for label, baseline in baseline_labels.items():
            candidate = candidate_labels.get(label)
            if candidate is None:
                return float("-inf")
            gains.append(float(candidate[metric]) - float(baseline[metric]))
    return min(gains) if gains else 0.0


def _candidate_rank_key(item: Dict[str, Any],
                        baseline_metrics_by_seed: Sequence[Dict[str, Any]]) -> tuple:
    """Rank viable bundles by worst-repeat gains before aggregate tie-breaks."""
    return (_worst_seed_class_gain(item, "accuracy", baseline_metrics_by_seed),
            _worst_seed_class_gain(item, "coverage", baseline_metrics_by_seed),
            _worst_seed_gain(item, "accuracy", baseline_metrics_by_seed),
            _worst_seed_gain(item, "answered_precision", baseline_metrics_by_seed),
            _worst_seed_gain(item, "coverage", baseline_metrics_by_seed),
            item["metrics"]["accuracy"], item["metrics"]["coverage"],
            item["metrics"]["answered_precision"], -item["index"])


def split_demo_validation(rows: Sequence[Dict[str, Any]], *,
                          require_class_coverage: bool = True
                          ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministically split dev validation into search and one-shot confirm.

    Candidate ranking sees only ``search``. Exactly one winner can be measured
    on ``confirm``; a failed confirmation rejects the search without trying the
    runner-up. The outer selection gate remains separate and unread here.

    Search/replay callers require at least two examples per gold label so each
    label appears in both cohorts. The factorial planner may disable that
    check only while estimating call counts; the real search always uses it.
    """
    ordered = sorted(rows, key=lambda row: (
        str((row.get("options") or [])[int(row.get("correct_index", 0))])
        .strip().casefold() if row.get("options") else "",
        str(row.get("task_id") or "")))
    labels = [_gold_label(row) for row in ordered]
    if require_class_coverage and any(not label for label in labels):
        raise ValueError("few-shot class-aware validation found a task without a valid gold label")
    counts = Counter(labels)
    singleton_labels = sorted(label for label, count in counts.items()
                              if label and count < 2)
    if require_class_coverage and singleton_labels:
        raise ValueError(
            "few-shot class-aware validation needs at least two rows per gold label; "
            f"singleton_labels={singleton_labels}")
    # Split within each gold-label bucket. A global alternating split can
    # leave a label at 3/2 on one bucket and 2/3 on the next solely because
    # the previous bucket had odd cardinality.
    search, confirm = [], []
    offsets: Counter[str] = Counter()
    for row in ordered:
        label = _gold_label(row)
        if offsets[label] % 2 == 0:
            search.append(row)
        else:
            confirm.append(row)
        offsets[label] += 1
    if not search or not confirm:
        raise ValueError("few-shot nested validation requires at least two validation tasks")
    if {str(row["task_id"]) for row in search} & {
            str(row["task_id"]) for row in confirm}:
        raise RuntimeError("few-shot search and confirmation tasks overlap")
    if require_class_coverage and (
            {_gold_label(row) for row in search}
            != {_gold_label(row) for row in confirm}):
        raise RuntimeError("few-shot search and confirmation do not cover the same gold labels")
    return search, confirm


def search_task_demo_candidate(cfg: Dict[str, Any], state: ArtifactState,
                               datasets: Sequence[str], limit: int | None,
                               seed: int, failure_ids: Sequence[str],
                               baseline_run_ids: Dict[str, str | Sequence[str] | Dict[str, str]],
                               instruction_candidate: Candidate | None = None
                               ) -> Candidate | None:
    """Search demo sets, optionally crossed with a GEPA instruction candidate.

    With ``instruction_candidate`` this evaluates a bounded instruction × demo
    grid and emits one atomic task_program artifact. It remains a local bounded
    search; it does not claim to implement DSPy's Bayesian MIPROv2 optimizer.
    """
    if (cfg.get("runtime") or {}).get("input_mode") != "task_input":
        raise ValueError("task-demo search requires task_input mode")
    if str((cfg.get("runtime") or {}).get("provider") or "").lower() != "pi":
        raise ValueError("task-demo search requires the Pi task runtime")
    if (cfg.get("evaluation") or {}).get("main_metric") != "accuracy":
        raise ValueError("task-demo search requires accuracy selection")
    train, val = visible_examples(cfg, datasets, limit)
    declared_visible_ids = {str(row.get("task_id")) for row in [*train, *val]}
    excluded_ids = {str(task_id) for task_id in
                    (((cfg.get("evolution") or {}).get("fewshot") or {}).get(
                        "excluded_task_ids") or [])}
    if excluded_ids:
        train = [row for row in train if str(row.get("task_id")) not in excluded_ids]
        val = [row for row in val if str(row.get("task_id")) not in excluded_ids]
    if len(datasets) != 1:
        raise ValueError("task-demo search currently requires one dataset per round")
    dataset = str(datasets[0])
    run_ids_value = baseline_run_ids[dataset]
    if isinstance(run_ids_value, dict):
        paired_runs = sorted(((int(run_seed), str(run_id))
                              for run_seed, run_id in run_ids_value.items()),
                             key=lambda item: item[0])
    elif isinstance(run_ids_value, str):
        paired_runs = [(int(seed), run_ids_value)]
    else:
        paired_runs = [(int(seed) + index, str(run_id))
                       for index, run_id in enumerate(run_ids_value)]
    baseline_seed_numbers = [item[0] for item in paired_runs]
    baseline_ids = [item[1] for item in paired_runs]
    if not baseline_ids:
        raise ValueError("task-demo search requires baseline run IDs for every seed")
    baseline_rows_by_seed = [
        read_jsonl(artifacts_path("runs", run_id, "per_task.jsonl"))
        for run_id in baseline_ids]
    task_by_id = {str(row["task_id"]): row for row in [*train, *val]}
    visible_ids = set(task_by_id)
    configured_scope = (((cfg.get("evolution") or {}).get("fewshot") or {})
                        .get("baseline_allowed_task_ids"))
    allowed_ids = ({str(task_id) for task_id in configured_scope}
                   if configured_scope is not None else declared_visible_ids)
    if not visible_ids <= allowed_ids or not declared_visible_ids <= allowed_ids:
        raise RuntimeError("few-shot baseline scope omits visible train/validation IDs")
    validated_baseline_rows = []
    for seed, rows in zip(baseline_seed_numbers, baseline_rows_by_seed):
        row_ids = [str(row.get("task_id") or "") for row in rows]
        present = set(row_ids)
        if len(row_ids) != len(present):
            raise RuntimeError(f"few-shot baseline seed {seed} contains duplicate task IDs")
        if not visible_ids <= present:
            raise RuntimeError(
                "few-shot baseline per-task rows do not cover the full visible cohort")
        if present != allowed_ids:
            raise RuntimeError(
                f"few-shot baseline seed {seed} task IDs differ from declared scope")
        normalized = []
        for row in rows:
            task_id = str(row["task_id"])
            # Joint baselines can carry already-measured discovery rows for the
            # paired evaluator. They are not optimizer inputs: validate their
            # declared scope above, then retain only frozen train IDs here.
            if task_id not in visible_ids:
                continue
            normalized.append(validate_task_answer_row(
                row, task_by_id[task_id], context=f"fewshot-baseline:{seed}:{task_id}",
                require_reported_gold_index=True))
        validated_baseline_rows.append(normalized)
    baseline_rows_by_seed = validated_baseline_rows
    baseline_id_sets = [{str(row["task_id"]) for row in rows}
                        for rows in baseline_rows_by_seed]
    if any(ids != baseline_id_sets[0] for ids in baseline_id_sets[1:]):
        raise RuntimeError("few-shot baseline seeds do not contain the same task IDs")
    baseline_rows = baseline_rows_by_seed[0]
    baseline_by_id = {str(row["task_id"]): row for row in baseline_rows}
    baseline_by_seed = [
        {str(row["task_id"]): row for row in rows}
        for rows in baseline_rows_by_seed]
    train_ids = {str(row["task_id"]) for row in train}
    val_ids = {str(row["task_id"]) for row in val}
    if train_ids & val_ids:
        raise RuntimeError("few-shot train and internal validation overlap")
    missing_baseline_ids = sorted((train_ids | val_ids) - set(baseline_by_id))
    if missing_baseline_ids:
        raise RuntimeError("baseline per-task rows do not cover the full visible cohort; "
                           f"missing={missing_baseline_ids[:10]}")
    stable_correct_ids = _majority_correct_task_ids(baseline_rows_by_seed)
    failure_votes: Counter[str] = Counter()
    for rows in baseline_rows_by_seed:
        for row in rows:
            task_id = str(row.get("task_id") or "")
            if task_id and not bool(row.get("correct")):
                failure_votes[task_id] += 1
    explicit_failure_ids = {str(task_id) for task_id in failure_ids}
    failed_train_ids = {
        task_id for task_id in train_ids
        if failure_votes[task_id] > len(baseline_rows_by_seed) / 2
    }
    selected_failure_ids = (failed_train_ids | (explicit_failure_ids & train_ids))
    demos = []
    labeled_demos = []
    hard_demos = []
    by_task = {str(row["task_id"]): row for row in train}
    for task_id in sorted(train_ids):
        row = baseline_by_id.get(task_id)
        task = by_task[task_id]
        demo = {"task_id": task_id, "task": task,
                "correct_index": int(task["correct_index"])}
        labeled_demos.append(demo)
        if row and task_id in stable_correct_ids:
            demos.append(demo)
        if task_id in selected_failure_ids:
            hard_demos.append(demo)
    # Put repeatedly failed or explicitly attributed training examples first;
    # ties are deterministic, and validation IDs were filtered above.
    hard_demos.sort(key=lambda item: (
        -failure_votes[str(item["task_id"])],
        str(item["task_id"]) not in explicit_failure_ids,
        str(item["task_id"])))
    candidate_sets, candidate_families = _candidate_sets(demos,
        max_sets=int(((cfg.get("evolution") or {}).get("fewshot") or {}).get(
            "max_candidate_sets", DEFAULT_MAX_DEMO_CANDIDATE_SETS)),
        k=int(((cfg.get("evolution") or {}).get("fewshot") or {}).get("k", 3)),
        seed=seed, hard_pool=hard_demos, labeled_pool=labeled_demos,
        include_families=True)
    if not candidate_sets:
        return None

    # The joint arm must still search task-visible demos when GEPA finds no
    # instruction challenger. In that case keep the baseline instruction fixed
    # and emit the same atomic task_program artifact; the report records that
    # this measured a demo-only fallback, not instruction × demo synergy.
    allowed_artifacts = set(((cfg.get("evolution") or {}).get(
        "allowed_artifacts") or []))
    joint = (instruction_candidate is not None or
             "task_program" in allowed_artifacts)
    if (joint and instruction_candidate is not None
            and instruction_candidate.artifact_type != "system_prompt"):
        raise ValueError("joint task-program search requires a system_prompt candidate")
    instruction_variants = [state.system_prompt]
    if (joint and instruction_candidate is not None
            and instruction_candidate.new_content != state.system_prompt):
        instruction_variants.append(instruction_candidate.new_content)

    baseline_val_by_seed = [
        [rows[task_id] for task_id in sorted(val_ids) if task_id in rows]
        for rows in baseline_by_seed]
    baseline_metrics_by_seed = [_metrics(rows) for rows in baseline_val_by_seed]
    baseline_metrics = baseline_metrics_by_seed[0]
    baseline_aggregate_metrics = _metrics(
        [row for rows in baseline_val_by_seed for row in rows])
    search_val, confirm_val = split_demo_validation(val)
    search_ids = {str(row["task_id"]) for row in search_val}
    confirm_ids = {str(row["task_id"]) for row in confirm_val}
    if search_ids | confirm_ids != val_ids or search_ids & confirm_ids:
        raise RuntimeError("few-shot nested validation does not partition validation IDs")
    search_baseline_by_seed = [
        [rows[task_id] for task_id in sorted(search_ids)]
        for rows in baseline_by_seed]
    confirm_baseline_by_seed = [
        [rows[task_id] for task_id in sorted(confirm_ids)]
        for rows in baseline_by_seed]
    search_baseline_metrics_by_seed = [_metrics(rows)
                                       for rows in search_baseline_by_seed]
    confirm_baseline_metrics_by_seed = [_metrics(rows)
                                        for rows in confirm_baseline_by_seed]
    search_baseline_aggregate_metrics = _metrics(
        [row for rows in search_baseline_by_seed for row in rows])
    confirm_baseline_aggregate_metrics = _metrics(
        [row for rows in confirm_baseline_by_seed for row in rows])
    max_calls = int(((cfg.get("evolution") or {}).get("fewshot") or {}).get(
        "max_metric_calls", 100))
    if max_calls <= 0:
        raise ValueError("few-shot metric budget must be positive")
    out_dir = artifacts_path("search", "task_demos", "demo-search-" + uuid.uuid4().hex[:12])
    out_dir.mkdir(parents=True, exist_ok=False)
    report_path = out_dir / "report.json"
    # Evaluate every candidate under every baseline seed. A one-seed inner
    # search can select an answer-more demo bundle whose precision regression
    # only becomes visible in the outer selection gate.
    planned_discovery_calls = (len(candidate_sets) * len(instruction_variants)
                               * len(search_val) * len(baseline_ids))
    planned_confirmation_calls = len(confirm_val) * len(baseline_ids)
    planned_metric_calls = planned_discovery_calls + planned_confirmation_calls
    if planned_metric_calls > max_calls:
        write_json(report_path, {
            "status": "budget_insufficient_no_selection",
            "optimizer": ("bounded GEPA-instruction × task-demo grid" if joint
                          else "bounded task-demo subset search"),
            "dataset": dataset, "split": "dev_optimize",
            "protocol_sha256": config_fingerprint(cfg),
            "search_protocol_version": SEARCH_PROTOCOL_VERSION,
            "search_protocol_sha256": _search_protocol_sha256(cfg),
            "release_id": state.release_id,
            "train_ids": sorted(train_ids), "val_ids": sorted(val_ids),
            "discovery_ids": sorted(search_ids),
            "confirmation_ids": sorted(confirm_ids),
            "instruction_count": len(instruction_variants),
            "demo_set_count": len(candidate_sets),
            "planned_discovery_calls": planned_discovery_calls,
            "planned_confirmation_calls": planned_confirmation_calls,
            "planned_metric_calls": planned_metric_calls,
            "max_metric_calls": max_calls,
            "evaluator_attempts": 0,
            "selection_gate_read": False,
            "reason": "budget cannot evaluate the full candidate grid; no partial search was run",
        })
        return None
    task_providers = {}
    attempts = 0
    results = []
    task_outcomes = []
    candidate_index = 0
    for instruction_index, instruction in enumerate(instruction_variants):
        for demo_index, subset in enumerate(candidate_sets):
            bundle = serialize_demos_bundle(subset)
            parsed = parse_demos_bundle(bundle)
            trial_state = ArtifactState(state.release_id, instruction,
                                        state.registry, task_demos=parsed)
            candidate_type = "task_program" if joint else "task_demos"
            content = serialize_task_program(instruction, bundle) if joint else bundle
            provisional = Candidate(
                candidate_id=content_id("cand", state.release_id, candidate_type, content),
                parent_release_id=state.release_id, artifact_type=candidate_type,
                artifact_id=candidate_type,
                diff=("jointly replace system prompt and task demos" if joint
                      else "replace task demo set"),
                new_content=content, content_sha256=content_sha256(content))
            gate = _candidate_static_gate(provisional, cfg, state)
            if not gate["passed"]:
                results.append({"index": candidate_index,
                                "instruction_index": instruction_index,
                                "demo_index": demo_index,
                                "demo_family": candidate_families[demo_index],
                                "status": "static_reject",
                                "static_gate": gate,
                                "demo_ids": [d["task_id"] for d in subset]})
                candidate_index += 1
                continue
            seed_rows = []
            from ..runtime.provider import build_provider
            for candidate_seed_int in baseline_seed_numbers:
                if candidate_seed_int not in task_providers:
                    task_providers[candidate_seed_int] = build_provider(
                        cfg, seed=candidate_seed_int)
                task_rows = []
                for example in search_val:
                    if attempts >= max_calls:
                        write_json(report_path, {"status": "budget_exhausted_no_selection",
                                                 "optimizer": ("bounded GEPA-instruction × task-demo grid"
                                                               if joint else "bounded task-demo subset search"),
                                                 "baseline_metrics": baseline_metrics,
                                                 "baseline_metrics_by_seed": baseline_metrics_by_seed,
                                                 "attempts": attempts,
                                                 "max_metric_calls": max_calls,
                                                 "task_evaluation_calls": len(task_outcomes),
                                                 "task_evaluation_cost": total_cost(task_outcomes),
                                                 "candidate_results": results})
                        return None
                    attempts += 1
                    outcome = TaskRunner(cfg, trial_state,
                                         provider=task_providers[candidate_seed_int],
                                         seed=candidate_seed_int).run(example)
                    task_outcomes.append(outcome)
                    checked = _validated_scored_outcome(
                        outcome, example,
                        context=f"fewshot-candidate:{candidate_seed_int}:{example['task_id']}")
                    task_rows.append({"task_id": example["task_id"],
                                      "seed": candidate_seed_int,
                                      "gold_label": _gold_label(example),
                                      "gold_index": checked["gold_index"],
                                      "answer_index": checked["answer_index"],
                                      "correct": checked["correct"],
                                      "status": checked["status"],
                                      "pi_session_path": outcome.pi_session_path})
                seed_rows.append({"seed": candidate_seed_int,
                                  "metrics": _metrics(task_rows), "rows": task_rows})
            all_rows = [row for item in seed_rows for row in item["rows"]]
            metrics = _metrics(all_rows)
            paired_transitions = _paired_answer_transitions(
                all_rows, search_baseline_by_seed, baseline_seed_numbers)
            results.append({"index": candidate_index,
                            "instruction_index": instruction_index,
                            "instruction_sha256": content_sha256(instruction),
                            "demo_index": demo_index,
                            "demo_family": candidate_families[demo_index],
                            "status": "evaluated",
                            "demo_ids": [str(d["task_id"]) for d in subset],
                            "bundle_sha256": content_sha256(bundle),
                            "program_sha256": content_sha256(content),
                            "metrics": metrics,
                            "seed_metrics": [{"seed": item["seed"],
                                              "metrics": item["metrics"]}
                                             for item in seed_rows],
                            "paired_answer_transitions": paired_transitions,
                            "rows": all_rows})
            candidate_index += 1

    joint_with_instruction_challenger = joint and instruction_candidate is not None
    report = {"status": "completed",
              "optimizer": ("bounded GEPA-instruction × task-demo grid"
                            if joint_with_instruction_challenger else
                            "bounded baseline-instruction × task-demo grid"
                            if joint else "bounded task-demo subset search"),
              "method": ("GEPA instruction challenger crossed with visible-dev demo subsets; exhaustive bounded grid, not DSPy MIPROv2"
                         if joint_with_instruction_challenger else
                         "GEPA returned no instruction challenger; baseline instruction crossed with visible-dev demo subsets, so this measures demo effects only"
                         if joint else
                         "visible-dev demo subset search; not DSPy MIPROv2"),
              "joint_search_mode": (
                  "instruction_x_demos" if joint_with_instruction_challenger else
                  "baseline_instruction_x_demos_no_gepa_challenger"
                  if joint else "demos_only"),
              "dataset": dataset, "split": "dev_optimize",
              "protocol_sha256": config_fingerprint(cfg),
              "search_protocol_sha256": _search_protocol_sha256(cfg),
              "release_id": state.release_id,
              "train_ids": sorted(train_ids), "val_ids": sorted(val_ids),
              "discovery_ids": sorted(search_ids),
              "confirmation_ids": sorted(confirm_ids),
              "baseline_seed_run_ids": baseline_ids,
              "search_protocol_version": SEARCH_PROTOCOL_VERSION,
              "demo_correctness_rule": (
                  "gold labels from training examples; bootstrap pool uses strict-majority baseline-correct, "
                  "hard-case pool uses strict-majority baseline failures plus attributed failures restricted to train"),
              "demo_candidate_generation": (
                  "class-balanced labeled and class-balanced hard-case bundles first, followed by baseline-correct, "
                  "diverse, hard-case-only, mixed, and random labeled subsets; "
                  "class-balanced labeled bundle plus seeded labeled-only and mixed subsets with variable size; exhaustive bounded evaluation, "
                  "MIPROv2-inspired candidate families, not Bayesian optimization"),
              "demo_order_search": (
                  "cyclic rotations of each class-balanced bundle are evaluated as distinct programs"),
              "demo_generation_seed": int(seed),
              "stable_correct_demo_pool_ids": sorted(
                  train_ids & stable_correct_ids),
              "gold_labeled_train_pool_ids": sorted(train_ids),
              "baseline_failed_train_pool_ids": sorted(failed_train_ids),
              "attributed_failure_train_ids": sorted(explicit_failure_ids & train_ids),
              "hard_case_demo_pool_ids": [str(item["task_id"])
                                          for item in hard_demos],
              "instructions": [{"index": i, "sha256": content_sha256(text),
                                "source": "baseline" if i == 0 else "pinned-gepa"}
                               for i, text in enumerate(instruction_variants)],
              "gepa_candidate": (instruction_candidate.generator
                                 if instruction_candidate else None),
              "baseline_metrics": baseline_metrics, "candidate_results": results,
              "baseline_aggregate_metrics": baseline_aggregate_metrics,
              "discovery_baseline_metrics_by_seed": [
                  {"seed": baseline_seed_numbers[index], "metrics": metrics}
                  for index, metrics in enumerate(search_baseline_metrics_by_seed)],
              "discovery_baseline_aggregate_metrics": search_baseline_aggregate_metrics,
              "confirmation_baseline_metrics_by_seed": [
                  {"seed": baseline_seed_numbers[index], "metrics": metrics}
                  for index, metrics in enumerate(confirm_baseline_metrics_by_seed)],
              "confirmation_baseline_aggregate_metrics": confirm_baseline_aggregate_metrics,
              "baseline_metrics_by_seed": [
                  {"seed": baseline_seed_numbers[index], "metrics": metrics}
                  for index, metrics in enumerate(baseline_metrics_by_seed)],
              "selection_protocol": {
                  "type": "nested_dev_class_aware_confirmation_v3",
                  "candidate_ranking_ids": sorted(search_ids),
                  "single_winner_confirmation_ids": sorted(confirm_ids),
                  "confirmation_runner_up_fallback": False,
                  "selection_gate_read": False},
              "planned_discovery_calls": planned_discovery_calls,
              "planned_confirmation_calls": planned_confirmation_calls,
              "planned_metric_calls": planned_metric_calls,
              "evaluator_attempts": attempts, "actual_metric_calls": attempts,
              "task_evaluation_calls": len(task_outcomes),
              "task_evaluation_cost": total_cost(task_outcomes),
              "max_metric_calls": max_calls,
              "selection_gate_read": False}
    def nonregressing_on_every_seed(
            item: Dict[str, Any],
            reference_metrics_by_seed: Sequence[Dict[str, Any]]) -> bool:
        if item["status"] != "evaluated":
            return False
        seed_metrics = item.get("seed_metrics") or []
        if len(seed_metrics) != len(reference_metrics_by_seed):
            return False
        for index, result in enumerate(seed_metrics):
            candidate_metrics = result["metrics"]
            baseline_metrics_for_seed = reference_metrics_by_seed[index]
            if not _nonregresses(candidate_metrics, baseline_metrics_for_seed):
                return False
        return True

    nonregressing = [item for item in results
                     if nonregressing_on_every_seed(
                         item, search_baseline_metrics_by_seed)]
    # A candidate that merely ties every metric has no demonstrated benefit.
    # Require a positive gain on the configured task-input primary axis before
    # spending outer optimize/gate calls on it.
    viable = [item for item in nonregressing
              if item["metrics"]["accuracy"]
              > search_baseline_aggregate_metrics["accuracy"] + 1e-9]
    # Among candidates that satisfy the per-seed nonregression gate, prefer
    # the one with the strongest worst-seed movement on all three metrics.
    viable.sort(key=lambda item: _candidate_rank_key(
        item, search_baseline_metrics_by_seed), reverse=True)
    if not viable:
        report["status"] = ("no_candidate_with_primary_gain" if nonregressing
                             else "no_candidate_nonregressing")
        report["selection_criteria"] = {
            "all_seed_nonregression": ["accuracy", "coverage", "answered_precision",
                                       "per_gold_label.accuracy",
                                       "per_gold_label.coverage"],
            "aggregate_primary_metric": "accuracy",
            "requires_positive_primary_gain": True,
            "nonregressing_candidate_count": len(nonregressing),
        }
        write_json(report_path, report)
        return None
    selected = viable[0]
    selected_demos = candidate_sets[int(selected["demo_index"])]
    instruction = instruction_variants[int(selected["instruction_index"])]
    selected_bundle = serialize_demos_bundle(selected_demos)
    candidate_type = "task_program" if joint else "task_demos"
    content = serialize_task_program(instruction, selected_bundle) if joint else selected_bundle
    static_candidate = Candidate(
        candidate_id=content_id("cand", state.release_id, candidate_type, content),
        parent_release_id=state.release_id, artifact_type=candidate_type,
        artifact_id=candidate_type,
        diff=("jointly replace system prompt and task demos" if joint
              else "replace task demo set"),
        new_content=content, content_sha256=content_sha256(content))
    # Rebuild and gate the exact serialized winner at the final selection
    # boundary. Candidate-grid gates protect discovery scoring; this gate
    # protects the single-winner confirmation runner from serialization drift
    # or later changes to the selected artifact/config.
    final_gate = _candidate_static_gate(static_candidate, cfg, state)
    if not final_gate["passed"]:
        report.update({"status": "selected_candidate_static_gate_rejected",
                       "selected_idx": selected["index"],
                       "selected_candidate_id": static_candidate.candidate_id,
                       "selected_candidate_sha256": static_candidate.content_sha256,
                       "selected_static_gate": final_gate,
                       "selection_gate_read": False,
                       "evaluator_attempts": attempts,
                       "actual_metric_calls": attempts,
                       "task_evaluation_calls": len(task_outcomes),
                       "task_evaluation_cost": total_cost(task_outcomes)})
        write_json(report_path, report)
        return None
    selected_trial_state = ArtifactState(
        state.release_id, instruction, state.registry,
        task_demos=parse_demos_bundle(selected_bundle))
    selected_confirmation_rows = []
    for candidate_seed_int in baseline_seed_numbers:
        if candidate_seed_int not in task_providers:
            from ..runtime.provider import build_provider
            task_providers[candidate_seed_int] = build_provider(
                cfg, seed=candidate_seed_int)
        task_rows = []
        for example in confirm_val:
            attempts += 1
            outcome = TaskRunner(
                cfg, selected_trial_state,
                provider=task_providers[candidate_seed_int],
                seed=candidate_seed_int).run(example)
            task_outcomes.append(outcome)
            checked = _validated_scored_outcome(
                outcome, example,
                context=f"fewshot-confirmation:{candidate_seed_int}:{example['task_id']}")
            task_rows.append({"task_id": str(example["task_id"]),
                              "seed": candidate_seed_int,
                              "gold_label": _gold_label(example),
                              "gold_index": checked["gold_index"],
                              "answer_index": checked["answer_index"],
                              "correct": checked["correct"],
                              "status": checked["status"],
                              "pi_session_path": outcome.pi_session_path})
        selected_confirmation_rows.append({
            "seed": candidate_seed_int, "metrics": _metrics(task_rows),
            "rows": task_rows})
    selected_confirmation_metrics = _metrics([
        row for seed_result in selected_confirmation_rows
        for row in seed_result["rows"]])
    confirmation_pass_by_seed = []
    for index, seed_result in enumerate(selected_confirmation_rows):
        candidate_metrics = seed_result["metrics"]
        baseline_metrics_for_seed = confirm_baseline_metrics_by_seed[index]
        confirmation_pass_by_seed.append(
            _nonregresses(candidate_metrics, baseline_metrics_for_seed))
    confirmation_passed = (
        all(confirmation_pass_by_seed)
        and selected_confirmation_metrics["accuracy"]
        > confirm_baseline_aggregate_metrics["accuracy"] + 1e-9)
    confirmation_result = {
        "status": "passed" if confirmation_passed else "rejected",
                   "selected_candidate_index": selected["index"],
                   "selected_demo_family": selected.get("demo_family"),
                   "candidate_demo_ids": selected["demo_ids"],
        "metrics": selected_confirmation_metrics,
        "metrics_by_seed": [{"seed": result["seed"],
                             "metrics": result["metrics"]}
                            for result in selected_confirmation_rows],
        "baseline_metrics": confirm_baseline_aggregate_metrics,
        "baseline_metrics_by_seed": [
            {"seed": baseline_seed_numbers[index], "metrics": metrics}
            for index, metrics in enumerate(confirm_baseline_metrics_by_seed)],
        "seed_nonregression": confirmation_pass_by_seed,
        "class_metrics": "per_gold_label accuracy and coverage must not regress",
        "requires_positive_accuracy_gain": True,
        "rows": [row for result in selected_confirmation_rows
                 for row in result["rows"]],
    }
    selected["internal_confirmation"] = {
        key: value for key, value in confirmation_result.items() if key != "rows"}
    report["internal_confirmation"] = confirmation_result
    report["evaluator_attempts"] = attempts
    report["actual_metric_calls"] = attempts
    report["task_evaluation_calls"] = len(task_outcomes)
    report["task_evaluation_cost"] = total_cost(task_outcomes)
    if not confirmation_passed:
        report["status"] = "no_candidate_internal_confirmation"
        report["selection_criteria"] = {
            "candidate_ranking_ids": sorted(search_ids),
            "single_winner_confirmation_ids": sorted(confirm_ids),
            "confirmation_runner_up_fallback": False,
            "all_seed_nonregression": ["accuracy", "coverage",
                                       "answered_precision",
                                       "per_gold_label.accuracy",
                                       "per_gold_label.coverage"],
            "confirmation_requires_positive_accuracy_gain": True,
            "confirmation_seed_nonregression": confirmation_pass_by_seed,
            "nonregressing_candidate_count": len(nonregressing),
        }
        write_json(report_path, report)
        return None

    report.update({"status": "selected", "selected_idx": selected["index"],
                   "selected_instruction_index": selected["instruction_index"],
                   "selected_instruction_sha256": content_sha256(instruction),
                   "selected_demo_ids": selected["demo_ids"],
                   "selected_metrics": selected["metrics"],
                   # `baseline_metrics` is retained as the legacy single-seed
                   # full-validation summary. Pair the selected discovery
                   # score with the same discovery IDs across all baseline
                   # seeds so report readers cannot compare n=75 against n=50.
                   "selected_metrics_scope": "discovery_ids_across_baseline_seeds",
                   "selected_baseline_metrics": search_baseline_aggregate_metrics,
                   "selected_metrics_task_ids": sorted(search_ids),
                   "selected_confirmation_metrics": selected_confirmation_metrics,
                   "selection_criteria": {
                       "candidate_ranking_ids": sorted(search_ids),
                       "single_winner_confirmation_ids": sorted(confirm_ids),
                       "confirmation_runner_up_fallback": False,
                       "all_seed_nonregression": ["accuracy", "coverage",
                                                   "answered_precision",
                                                   "per_gold_label.accuracy",
                                                   "per_gold_label.coverage"],
                       "aggregate_primary_metric": "accuracy",
                       "tie_break": "maximize worst-seed gold-class accuracy, gold-class coverage, accuracy, answered_precision, then coverage",
                       "requires_positive_primary_gain": True,
                       "nonregressing_candidate_count": len(nonregressing),
                   },
                   "selected_sha256": sha256_obj(content),
                   "selected_static_gate": final_gate})
    write_json(report_path, report)
    static_candidate.rationale = (
        "Visible-dev GEPA instruction/demo search with aggregate and per-gold-class nonregression across every paired optimize seed"
        if joint_with_instruction_challenger else
        "Visible-dev demo search with baseline instruction after GEPA found no challenger; aggregate and per-gold-class nonregression across every paired optimize seed"
        if joint else
        "Visible-dev task-demo subset search with aggregate and per-gold-class nonregression across every paired optimize seed")
    static_candidate.attributed_class = (
        "experiment.task_program" if joint else "experiment.task_demos")
    static_candidate.source_failure_ids = list(failure_ids)
    static_candidate.generator = {"backend": (
                                      "gepa-task-demo-grid"
                                      if joint_with_instruction_challenger else
                                      "baseline-instruction-demo-grid"
                                      if joint else "task-demo-subset-search"),
                                  "search_report": str(report_path),
                                  "selected_idx": selected["index"],
                                  "selected_instruction_index": selected["instruction_index"],
                                  "instruction_candidate": (instruction_candidate.to_dict()
                                                            if instruction_candidate else None),
                                  "train_ids": sorted(train_ids),
                                  "val_ids": sorted(val_ids),
                                  "confirmation_ids": sorted(confirm_ids),
                                  "confirmation_metrics": selected_confirmation_metrics,
                                  "metrics": selected["metrics"]}
    return static_candidate

