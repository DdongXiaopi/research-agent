"""Paired train-only mechanism pilot on a frozen PubMed article snapshot."""

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
from src.data.bioasq_unitor import load_tasks
from src.eval.metrics import aggregate
from src.governance.budget import check_tolerances
from src.governance.policy import static_gate
from src.governance.release import load_state
from src.retrieval.pubmed_snapshot import PubMedSnapshotGateway
from src.runtime.provider import build_provider
from src.runtime.runner import ArtifactState, TaskRunner
from src.runtime.skills import load_body, with_body
from src.util import sha256_obj

SKILL = "litqa-literature-synthesis"
QUERY_TEMPLATE = "RETRIEVAL_QUERY: {question} study results outcomes"


def _summary(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(x["correct"] for x in rows)
    answered = sum(x["status"] == "ok" for x in rows)
    return {"n": n, "correct": correct, "answered": answered,
            "accuracy": correct / n if n else 0,
            "coverage": answered / n if n else 0,
            "answered_precision": correct / answered if answered else 0,
            "answers_without_valid_citation": sum(
                x["status"] == "ok" and not x["cited"] for x in rows)}


def main() -> int:
    raise SystemExit(
        "unplanned historical snapshot scoring is retired: direct paired "
        "TaskRunner calls have no frozen execution-plan hash, request caps, "
        "or shared call ledger"
    )

    # Retained as historical implementation only pending governed replay.
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    gateway = PubMedSnapshotGateway(snapshot, top_k=3)
    task_ids = [r["task_id"] for r in snapshot["tasks"]]
    if args.offset < 0:
        parser.error("offset must be non-negative")
    task_ids = task_ids[args.offset:]
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("limit must be positive")
        task_ids = task_ids[:args.limit]
    tasks_by_id = {t["task_id"]: t for t in load_tasks("train", task_ids=set(task_ids))}
    tasks = [tasks_by_id[i] for i in task_ids]
    cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
    cfg["retrieval"]["query_control"] = {"enabled": True}
    cfg["skills"]["pinned_skill"] = SKILL
    base = load_state("baseline-v1-cited")
    body = load_body(base.registry.get(SKILL)).rstrip() + "\n\n" + QUERY_TEMPLATE + "\n"
    cand = Candidate("pubmed-query-pilot", base.release_id, "skill_body", SKILL,
                     diff="append retrieval query template", new_content=body)
    gate = static_gate(cand, cfg, base)
    if not gate["passed"]:
        raise RuntimeError("candidate static gate failed: " + str(gate))
    changed = ArtifactState("pubmed-query-pilot", base.system_prompt,
                            with_body(base.registry, SKILL, body))
    provider = build_provider(cfg)
    provider.cache = False
    arms = {"baseline": base, "query_variant": changed}
    results: dict[str, list[dict]] = {name: [] for name in arms}
    outcomes_by_arm: dict[str, list] = {name: [] for name in arms}
    for task in tasks:
        for name, state in arms.items():
            outcome = TaskRunner(cfg, state, provider=provider, gateway=gateway,
                                 seed=20260929).run(task)
            outcomes_by_arm[name].append(outcome)
            results[name].append({
                "task_id": task["task_id"], "correct": outcome.correct,
                "status": outcome.status, "answer_index": outcome.answer_index,
                "cited": outcome.cited,
                "query": outcome.retrieval_query,
                "top3": outcome.retrieved_doc_ids,
                "skill_recalled": outcome.skill_recalled,
            })
    before = results["baseline"]
    after = results["query_variant"]
    metrics = {name: aggregate(values) for name, values in outcomes_by_arm.items()}
    gate_cfg = {**cfg, "gates": {**cfg.get("gates", {}),
                "accept": {**cfg.get("gates", {}).get("accept", {}),
                           "tolerance": {**cfg.get("gates", {}).get("accept", {}).get(
                               "tolerance", {}),
                               "required": ["citation_precision",
                                            "unsupported_answer_rate"]}}}}
    grounding_gate = check_tolerances(metrics["baseline"],
                                      metrics["query_variant"], gate_cfg,
                                      kind="http")
    report = {
        "run_id": "pubmed-train-pilot-" + uuid.uuid4().hex[:12],
        "status": "train_only_mechanism_pilot",
        "snapshot_content_sha256": snapshot["content_sha256"],
        "evaluated_task_ids": task_ids,
        "config_sha256": sha256_obj(cfg),
        "candidate_body_sha256": sha256_obj(body),
        "candidate_static_gate": gate,
        "n_tasks_with_docs": sum(bool(gateway.corpus(t)) for t in tasks),
        "top3_order_changed": sum(a["top3"] != b["top3"]
                                  for a, b in zip(before, after)),
        "top3_membership_changed": sum(set(a["top3"]) != set(b["top3"])
                                       for a, b in zip(before, after)),
        "baseline": _summary(before), "query_variant": _summary(after),
        "grounding_gate": grounding_gate,
        "grounding_gate_passed": grounding_gate["passed"],
        "per_task": {"baseline": before, "query_variant": after},
        "metric_limit": "No gold labels on PubMed articles; citation_precision, "
                        "gold_in_topk, and unsupported_answer_rate are unavailable. "
                        "answers_without_valid_citation tests ID validity only, "
                        "not semantic entailment.",
    }
    path = ROOT / "artifacts/probe/pubmed_snapshot" / (report["run_id"] + ".json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "run_id", "n_tasks_with_docs", "top3_order_changed",
        "top3_membership_changed", "baseline", "query_variant",
        "grounding_gate_passed", "metric_limit")},
                     ensure_ascii=False, indent=2))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
