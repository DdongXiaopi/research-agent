"""Actual pinned-GEPA Skill-body search on public dev, then gated exploration.

No selection_gate per-task feedback and no final_holdout access. The BioASQ
source is intentionally never imported here; it remains a possible new cohort.
Run with Python 3.10+ (this repository's .venv uses 3.12).
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rsi import load_config
from src.contracts.types import Candidate
from src.data.split import load_tasks
from src.eval.harness import evaluate
from src.eval.options import inject_all
from src.eval.task_rows import validate_task_answer_row
from src.evolution.gepa_adapter import (TaskExecutionFailure, build_metric, build_optimizer,
                                         run_reflection)
from src.governance.policy import static_gate
from src.governance.release import load_state
from src.retrieval.intent import query_for
from src.runtime.provider import HttpProvider, build_provider
from src.runtime.runner import ArtifactState, TaskRunner
from src.runtime.skills import load_body, with_body
from src.util import sha256_obj

SKILL = "litqa-literature-synthesis"
QUERY = "RETRIEVAL_QUERY: {question} {options}"


def _save(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _rows(run_id: str) -> list[dict]:
    p = ROOT / "artifacts/runs" / run_id / "per_task.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]


def _summary(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(bool(r["correct"]) for r in rows)
    answered = sum(r["status"] == "ok" for r in rows)
    return {"n": n, "correct": correct, "answered": answered,
            "accuracy": correct / n, "coverage": answered / n,
            "answered_precision": correct / answered if answered else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-evals", type=int, default=64)
    ap.add_argument("--train", type=int, default=18)
    ap.add_argument("--val", type=int, default=12)
    ap.add_argument("--smoke", type=int, default=10)
    ap.add_argument("--full-limit", type=int, default=230)
    ap.add_argument("--cohort", choices=["first30", "targeted_top3"], default="first30")
    args = ap.parse_args()
    raise SystemExit(
        "This legacy standalone GEPA Skill-search entrypoint is disabled: it can "
        "construct live providers and score candidates without a frozen plan-hash "
        "challenge or request ledger. No config, tasks, release state, or provider "
        "will be loaded. Use a controlled search entrypoint with a reviewed plan."
    )
    if args.cohort == "first30" and args.train + args.val > 30:
        raise ValueError("GEPA proposal/selection samples must stay within first 30 dev tasks")
    cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
    cfg["retrieval"]["query_control"] = {"enabled": True}
    base = load_state("baseline-v1-cited")
    skill = base.registry.get(SKILL)
    seed_body = load_body(skill).rstrip() + "\n\n" + QUERY + "\n"
    tasks = sorted(load_tasks("pubmedqa", splits=["dev_optimize"]),
                   key=lambda t: t["task_id"])
    if cfg.get("evaluation", {}).get("insufficient_option", True):
        tasks = inject_all(tasks)
    if args.cohort == "targeted_top3":
        if (args.train, args.val) != (18, 12):
            raise ValueError("targeted_top3 uses fixed 18/12 composition")
        baseline_rows = _rows("run-e7646b425e840c10")[30:]
        by_id = {t["task_id"]: t for t in tasks}
        classes = {
            "wrong": sorted(r["task_id"] for r in baseline_rows
                            if r["gold_in_topk"] and not r["correct"]
                            and r["status"] == "ok"),
            "abstain": sorted(r["task_id"] for r in baseline_rows
                              if r["gold_in_topk"] and r["status"] == "abstain"),
            "correct": sorted(r["task_id"] for r in baseline_rows
                              if r["gold_in_topk"] and r["correct"]),
        }
        if any(len(x) < 10 for x in classes.values()):
            raise RuntimeError("targeted_top3 class has fewer than 10 tasks")
        train_ids = [i for cls in classes.values() for i in cls[:6]]
        val_ids = [i for cls in classes.values() for i in cls[6:10]]
        train = [by_id[i] for i in train_ids]
        val = [by_id[i] for i in val_ids]
    else:
        train = tasks[:args.train]
        val = tasks[args.train:args.train + args.val]
    if len(train) != args.train or len(val) != args.val:
        raise RuntimeError("insufficient visible dev tasks")

    run_id = "gepa-skill-" + uuid.uuid4().hex[:12]
    out_dir = ROOT / "artifacts/probe/real_gepa" / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    out = out_dir / "report.json"
    task_provider = build_provider(cfg)
    task_provider.cache = False
    reflect_cfg = dict(cfg["models"]["reflection"])
    reflect_cfg.pop("provider", None)
    reflect_cfg["max_tokens"] = max(1800, int(reflect_cfg.get("max_tokens", 0)))
    reflection_provider = HttpProvider(cache=False, **reflect_cfg)
    metric = build_metric()
    evaluated: dict[str, dict] = {}

    def reflection_lm(prompt):
        if isinstance(prompt, list):
            user = "\n\n".join(str(x.get("content", "")) for x in prompt)
        else:
            user = str(prompt)
        return reflection_provider.chat(
            "You are GEPA's reflective mutation model. Return the complete replacement "
            "Skill procedure inside one fenced code block.", user, max_tokens=1800)

    def evaluator(candidate: str, example: dict):
        text = str(candidate).strip()
        candidate_id = run_id + ":" + sha256_obj(text)[:12]
        cand = Candidate(candidate_id, base.release_id, "skill_body", SKILL,
                         diff="GEPA full Skill-body replacement", new_content=text)
        checks = static_gate(cand, cfg, base)
        if not checks["passed"]:
            failed = [x["check"] for x in checks["results"] if not x["passed"]]
            return 0.0, {"Feedback": "Static gate rejected: " + ", ".join(failed)}
        key = sha256_obj({"text": text, "task": example["task_id"]})
        if key in evaluated:
            return evaluated[key]["score"], evaluated[key]["side_info"]
        state = ArtifactState(candidate_id, base.system_prompt,
                              with_body(base.registry, SKILL, text))
        runner = TaskRunner(cfg, state, provider=task_provider, seed=20260929)
        outcome = runner.run(example)
        if str(getattr(outcome, "status", "")) in {"ok", "abstain"}:
            try:
                outcome_row = (outcome.to_dict() if callable(
                    getattr(outcome, "to_dict", None)) else vars(outcome))
                validate_task_answer_row(
                    outcome_row, example, context="legacy GEPA Skill task evaluation",
                    require_reported_gold_index=True)
            except (AttributeError, TypeError, ValueError) as exc:
                raise TaskExecutionFailure(
                    str(getattr(outcome, "task_id", example.get("task_id", "unknown"))),
                    "invalid_outcome") from exc
        score, side = metric(outcome)
        corpus = runner.gateway.corpus(example)
        by_id = {d["doc_id"]: d for d in corpus}
        evidence = [str(by_id[d].get("text", ""))[:650]
                    for d in outcome.retrieved_doc_ids if d in by_id]
        side.update({
            "Question": example["question"],
            "Options": list(example["options"]),
            "Expected": example["options"][example["correct_index"]],
            "Selected": (example["options"][outcome.answer_index]
                         if outcome.answer_index >= 0 else "ABSTAIN"),
            "Evidence": evidence,
            "SkillRecalled": outcome.skill_recalled,
        })
        evaluated[key] = {"score": score, "side_info": side}
        return score, side

    config = build_optimizer(
        {"seed": 20260929, "max_candidate_proposals": 5,
         "reflection_minibatch_size": 3}, reflection_lm,
        str(out_dir / "gepa_state"), max_metric_calls=args.max_evals)
    report = {"run_id": run_id, "status": "searching",
              "optimizer": "vendor/gepa-main/gepa-main (actual GEPA optimize_anything)",
              "dataset": "pubmedqa", "split": "dev_optimize",
              "cohort": args.cohort,
              "proposal_n": len(train), "selection_n": len(val),
              "proposal_ids": [x["task_id"] for x in train],
              "selection_ids": [x["task_id"] for x in val],
              "seed_body_sha256": sha256_obj(seed_body),
              "config_sha256": sha256_obj(cfg), "new_protocol": "real-v3 + query-template-v1"}
    _save(out, report)
    try:
        result = run_reflection(
            seed_candidate=seed_body, evaluator=evaluator,
            trainset=train, valset=val, config=config,
            objective=("Improve grounded biomedical yes/no/maybe answering accuracy "
                       "by rewriting this Skill procedure. Keep citation and abstention rules."),
            background=("You can change the Skill procedure and optionally include a single "
                        "RETRIEVAL_QUERY template with {question} and {options}. "
                        "Evidence is fixed per task; do not use gold metadata at runtime. "
                        "Preserve numbered procedural steps and avoid answering from prior knowledge."))
    except Exception as exc:
        report.update({
            "status": "optimizer_error_no_selection",
            "optimizer_error_type": type(exc).__name__,
            # Keep a typed, fail-closed report without persisting arbitrary
            # provider exception text or echoed request material.
            "optimizer_error_detail_omitted": True,
            "optimizer_run_aborted": True,
            "val_aggregate_scores": None,
            "selected_idx": None,
            "selected_sha256": None,
        })
        if isinstance(exc, TaskExecutionFailure):
            report["task_execution_failure"] = dict(exc.failure)
        _save(out, report)
        raise
    best = result.best_candidate
    if isinstance(best, dict):
        best = next(iter(best.values()))
    best = str(best)
    best_path = out_dir / "best_skill.txt"
    best_path.write_text(best, encoding="utf-8")
    report.update({"status": "candidate", "best_sha256": sha256_obj(best),
                   "best_val_score": result.val_aggregate_scores[result.best_idx],
                   "n_candidates": len(result.candidates),
                   "total_metric_calls": result.total_metric_calls})
    _save(out, report)
    cand = Candidate(run_id + ":best", base.release_id, "skill_body", SKILL,
                     diff="GEPA selected full Skill body", new_content=best)
    static = static_gate(cand, cfg, base)
    report["static_gate"] = static
    if not static["passed"]:
        report["status"] = "rejected_static"
        _save(out, report)
        print(out)
        return 0
    state = ArtifactState(run_id + ":best", base.system_prompt,
                          with_body(base.registry, SKILL, best))
    smoke = evaluate(cfg, state, "pubmedqa", "dev_optimize", limit=args.smoke,
                     seed=20260929, run_tag=run_id + ":smoke", write_per_task=True,
                     cfg_fp=run_id + ":smoke")
    report["smoke"] = {"run_id": smoke.run_id, "metrics": smoke.metric_values}
    if smoke.metric_values["unsupported_answer_rate"] > 0.05:
        report["status"] = "rejected_grounding_smoke"
        _save(out, report)
        print(out)
        return 0
    _save(out, report)
    full = evaluate(cfg, state, "pubmedqa", "dev_optimize", limit=args.full_limit,
                    seed=20260924, run_tag=run_id + ":full", write_per_task=True,
                    cfg_fp=run_id + ":full")
    rows = _rows(full.run_id)
    excluded = {x["task_id"] for x in train + val}
    exploratory = [r for r in rows[30:] if r["task_id"] not in excluded]
    report["full"] = {"run_id": full.run_id, "metrics": full.metric_values,
                      "exploratory_after_30_excluding_search": _summary(exploratory),
                      "excluded_search_ids": sorted(excluded)}
    report["status"] = "complete"
    _save(out, report)
    print(json.dumps(report["full"]["exploratory_after_30_excluding_search"], ensure_ascii=False))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
