"""Validated reconstruction of a selected visible-dev task-demo artifact.

This module is kept outside the hashed task-input execution protocol: replay
glue does not alter candidate evaluation semantics. The reconstructed content
is still checked by the normal static gate and then measured by ``run_round``.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from ..config import config_fingerprint
from ..contracts.types import Candidate, content_id, content_sha256
from ..evolution.fewshot import serialize_demos_bundle
from ..governance.policy import static_gate
from ..runtime.runner import ArtifactState
from ..util import read_json
from .gepa_search import visible_examples
from .task_demo_search import (SEARCH_PROTOCOL_VERSION, _search_protocol_sha256,
                               split_demo_validation)


def replay_task_demo_candidate(cfg: Dict[str, Any], state: ArtifactState,
                               datasets: Sequence[str], limit: int | None,
                               seed: int, failure_ids: Sequence[str],
                               report_path: str) -> Candidate:
    """Rebuild an already selected demo candidate after strict provenance checks."""
    report = read_json(report_path)
    if report.get("status") != "selected":
        raise ValueError("few-shot replay report must have status=selected")
    if report.get("search_protocol_version") != SEARCH_PROTOCOL_VERSION:
        raise ValueError("few-shot replay report uses an obsolete search protocol")
    if report.get("split") != "dev_optimize" or report.get("selection_gate_read") is not False:
        raise ValueError("few-shot replay requires a dev_optimize report with gate unread")
    search_mode = report.get("joint_search_mode")
    if search_mode not in {"demos_only",
                           "baseline_instruction_x_demos_no_gepa_challenger"}:
        raise ValueError("few-shot replay requires a demos-only or baseline-instruction search")
    if search_mode == "baseline_instruction_x_demos_no_gepa_challenger":
        instructions = list(report.get("instructions") or [])
        if (len(instructions) != 1
                or instructions[0].get("source") != "baseline"
                or instructions[0].get("sha256") != content_sha256(state.system_prompt)
                or report.get("selected_instruction_index") != 0
                or report.get("selected_instruction_sha256") !=
                content_sha256(state.system_prompt)):
            raise ValueError("joint fallback replay does not match the baseline instruction")
    if report.get("release_id") != state.release_id:
        raise ValueError("few-shot replay report parent release does not match ACTIVE")
    current_protocol = config_fingerprint(cfg)
    current_search_protocol = _search_protocol_sha256(cfg)
    if report.get("protocol_sha256") != current_protocol:
        raise ValueError("few-shot replay config fingerprint does not match search")
    if report.get("search_protocol_sha256") != current_search_protocol:
        raise ValueError("few-shot replay search-protocol fingerprint does not match")
    if list(datasets) != [str(report.get("dataset") or "")]:
        raise ValueError("few-shot replay dataset does not match the search report")
    if int(report.get("demo_generation_seed", -1)) != int(seed):
        raise ValueError("few-shot replay seed does not match the search report")
    confirmation = report.get("internal_confirmation") or {}
    if (confirmation.get("status") != "passed"
            or confirmation.get("selected_candidate_index") !=
            report.get("selected_idx")
            or confirmation.get("candidate_demo_ids") !=
            report.get("selected_demo_ids")):
        raise ValueError("class-aware nested confirmation does not approve the selected demos")
    criteria = report.get("selection_criteria") or {}
    required_axes = {"per_gold_label.accuracy", "per_gold_label.coverage"}
    if not required_axes.issubset(set(criteria.get("all_seed_nonregression") or [])):
        raise ValueError("few-shot replay report lacks per-class nonregression criteria")

    train, val = visible_examples(cfg, datasets, limit)
    train_ids = {str(row["task_id"]) for row in train}
    val_ids = {str(row["task_id"]) for row in val}
    if sorted(train_ids) != report.get("train_ids"):
        raise ValueError("few-shot replay train IDs do not match the search report")
    if sorted(val_ids) != report.get("val_ids"):
        raise ValueError("few-shot replay validation IDs do not match the search report")
    search_val, confirm_val = split_demo_validation(val)
    if ([str(row["task_id"]) for row in sorted(
            search_val, key=lambda row: str(row["task_id"]))]
            != report.get("discovery_ids")):
        raise ValueError("few-shot replay discovery IDs do not match the search report")
    if ([str(row["task_id"]) for row in sorted(
            confirm_val, key=lambda row: str(row["task_id"]))]
            != report.get("confirmation_ids")):
        raise ValueError("few-shot replay confirmation IDs do not match the search report")

    selected_ids = [str(value) for value in report.get("selected_demo_ids") or []]
    stable_pool = set(report.get("stable_correct_demo_pool_ids") or [])
    # v2 searches any trusted gold-labeled training example, with baseline
    # failures explicitly included as hard cases. Older reports only searched
    # the stable-correct bootstrap pool, so retain their narrower validation.
    eligible_pool = set(report.get("gold_labeled_train_pool_ids") or stable_pool)
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("few-shot replay report has an empty or duplicate demo selection")
    if not set(selected_ids).issubset(train_ids & eligible_pool):
        raise ValueError("few-shot replay demos are outside the eligible gold-labeled training pool")
    task_by_id = {str(row["task_id"]): row for row in train}
    demos = [{"task_id": task_id, "task": task_by_id[task_id],
              "correct_index": int(task_by_id[task_id]["correct_index"])}
             for task_id in selected_ids]
    bundle = serialize_demos_bundle(demos)
    selected_result = next((row for row in report.get("candidate_results", [])
                            if int(row.get("index", -1)) ==
                            int(report.get("selected_idx", -2))), None)
    if not selected_result or selected_result.get("demo_ids") != selected_ids:
        raise ValueError("few-shot replay selection does not match a candidate result")
    if (search_mode == "baseline_instruction_x_demos_no_gepa_challenger"
            and selected_result.get("instruction_index") != 0):
        raise ValueError("joint fallback replay selected a nonbaseline instruction")
    if content_sha256(bundle) != selected_result.get("bundle_sha256"):
        raise ValueError("few-shot replay reconstructed bundle hash does not match report")

    candidate = Candidate(
        candidate_id=content_id("cand", state.release_id, "task_demos", bundle),
        parent_release_id=state.release_id, artifact_type="task_demos",
        artifact_id="task_demos", diff="replay selected task demo bundle",
        new_content=bundle, content_sha256=content_sha256(bundle))
    candidate.rationale = "Replay selected visible-dev task demos for outer evaluation"
    candidate.attributed_class = "experiment.task_demos"
    candidate.source_failure_ids = list(failure_ids)
    candidate.generator = {"backend": "task-demo-search-replay",
                          "search_report": str(report_path),
                          "selected_idx": report["selected_idx"],
                          "selected_demo_ids": selected_ids,
                          "selection_metrics": selected_result.get("metrics"),
                          "confirmation_metrics": report.get(
                              "selected_confirmation_metrics"),
                          "confirmation_ids": report.get("confirmation_ids"),
                          "search_protocol_sha256": report.get("protocol_sha256"),
                          "search_algorithm_sha256": report.get("search_protocol_sha256"),
                          "search_protocol_version": report.get("search_protocol_version"),
                          "current_protocol_sha256": current_protocol,
                          "selection_protocol_match": (
                              report.get("protocol_sha256") == current_protocol),
                          "source_joint_search_mode": search_mode,
                          "train_ids": sorted(train_ids),
                          "val_ids": sorted(val_ids)}
    gate = static_gate(candidate, cfg, state)
    if not gate.get("passed"):
        raise ValueError("replayed task-demo candidate failed static gate")
    return candidate
