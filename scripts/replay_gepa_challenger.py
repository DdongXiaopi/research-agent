"""Replay a preselected GEPA dev-validation challenger through aggregate gate."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.contracts.types import Candidate, content_id  # noqa: E402
from src.eval.harness import evaluate  # noqa: E402
from src.evolution.gepa_search import (training_question_overlap,  # noqa: E402
                                       visible_examples)
from src.evolution.prompt_rewrite import validate_rewrite  # noqa: E402
from src.evolution.reflection import unified_diff  # noqa: E402
from src.evolution.round import task_input_secondary_gate  # noqa: E402
from src.governance.policy import static_gate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.runtime.runner import ArtifactState  # noqa: E402
from src.util import artifacts_path, read_json, sha256_obj, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-report", type=Path, required=True)
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--dataset", default="pubmedqa")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--reflection-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--gepa-metric-calls", type=int, default=60)
    args = parser.parse_args()
    raise SystemExit(
        "Historical GEPA replay is disabled: it can launch task-model calls and read "
        "selection_gate without a current hash-bound approved execution plan. "
        "Preserve existing reports for audit and use the controlled fixed-candidate "
        "preflight/execution path. No config, release, task, or provider is loaded here."
    )

    cfg = copy.deepcopy(load_config(args.config))
    cfg.setdefault("runtime", {})["input_mode"] = "task_input"
    cfg["runtime"].setdefault("task_input_max_chars", 30000)
    cfg.setdefault("evaluation", {})["sealed_pairing"] = False
    cfg["evaluation"]["main_metric"] = "accuracy"
    cfg["evaluation"]["primary_axis"] = "accuracy"
    cfg.setdefault("retrieval", {}).pop("query_control", None)
    cfg["retrieval"].pop("pubmed_source_queries", None)
    cfg.setdefault("tools", {})["active"] = [
        name for name in cfg["tools"].get("active", []) if name != "academic.search"]
    cfg.setdefault("evolution", {}).setdefault("gepa", {})["enabled"] = True
    cfg["evolution"]["gepa"]["max_metric_calls"] = args.gepa_metric_calls
    cfg.setdefault("models", {}).setdefault("reflection", {})["model_id"] = (
        args.reflection_model)

    search = read_json(args.search_report)
    state = load_state()
    train, val = visible_examples(cfg, [args.dataset], args.limit)
    if (search.get("release_id") != state.release_id or
            search.get("protocol_sha256") != sha256_obj(cfg) or
            search.get("train_ids") != [x["task_id"] for x in train] or
            search.get("val_ids") != [x["task_id"] for x in val]):
        raise ValueError("search report is not from this release, config, or visible-dev cohort")
    scores = search["val_aggregate_scores"]
    texts = search["candidate_texts"]
    if not 0 < args.index < len(texts) or float(scores[args.index]) < float(scores[0]):
        raise ValueError("index is not a nonregressing preselected GEPA challenger")
    proposed = str(texts[args.index]).strip()
    if training_question_overlap(proposed, train + val):
        raise ValueError("candidate memorizes a visible training question")
    reason = validate_rewrite(proposed, state.system_prompt, cfg)
    if reason:
        raise ValueError("rewrite invalid: " + reason)
    candidate = Candidate(
        candidate_id=content_id("cand", state.release_id, "system_prompt", proposed),
        parent_release_id=state.release_id, artifact_type="system_prompt",
        artifact_id="system_prompt", diff=unified_diff(state.system_prompt, proposed),
        new_content=proposed)
    if not static_gate(candidate, cfg, state)["passed"]:
        raise ValueError("candidate fails static gate")
    challenger = ArtifactState(state.release_id, proposed, state.registry)

    baseline, trial = {}, {}
    for split in ("dev_optimize", "selection_gate"):
        b = evaluate(cfg, state, args.dataset, split, seed=args.seed,
                     limit=args.limit, run_tag="baseline-replay",
                     write_per_task=split == "dev_optimize")
        c = evaluate(cfg, challenger, args.dataset, split, seed=args.seed,
                     limit=args.limit, run_tag="gepa-tie-replay",
                     write_per_task=split == "dev_optimize")
        if split == "selection_gate" and (b.per_task or c.per_task):
            raise RuntimeError("selection_gate returned per-task results")
        baseline[split] = {"run_id": b.run_id, "metrics": b.metric_values}
        trial[split] = {"run_id": c.run_id, "metrics": c.metric_values}
    secondary = task_input_secondary_gate(
        {args.dataset: {"optimize": baseline["dev_optimize"]["metrics"],
                        "gate": baseline["selection_gate"]["metrics"]}},
        {args.dataset: {"optimize": trial["dev_optimize"]["metrics"],
                        "gate": {"metrics": trial["selection_gate"]["metrics"]}}},
        [args.dataset])
    report = {
        "status": "exploratory_replay_only", "source_search_report": str(args.search_report),
        "selected_index": args.index, "candidate_id": candidate.candidate_id,
        "candidate_sha256": sha256_obj(proposed),
        "protocol_sha256": sha256_obj(cfg), "release_id": state.release_id,
        "dataset": args.dataset, "limit": args.limit,
        "baseline": baseline, "challenger": trial,
        "secondary_gate": secondary, "selection_gate_per_task_read": False,
    }
    out = artifacts_path("probe", "gepa_tie_replay",
                         "replay-" + uuid.uuid4().hex[:12] + ".json")
    write_json(out, report)
    summary = {"path": str(out),
               "optimize_accuracy": [baseline["dev_optimize"]["metrics"]["accuracy"],
                                     trial["dev_optimize"]["metrics"]["accuracy"]],
               "gate_accuracy": [baseline["selection_gate"]["metrics"]["accuracy"],
                                 trial["selection_gate"]["metrics"]["accuracy"]],
               "secondary_passed": secondary["passed"]}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
