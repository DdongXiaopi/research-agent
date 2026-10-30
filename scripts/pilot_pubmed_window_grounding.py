"""Paired BioASQ train pilot with query-dependent evidence windows and citation judge."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rsi import load_config
from src.contracts.types import Candidate
from src.data.bioasq_unitor import load_tasks
from src.eval.final_eval import protocol_fingerprint
from src.eval.metrics import aggregate
from src.governance.budget import check_tolerances
from src.governance.policy import static_gate
from src.governance.release import load_state
from src.retrieval.pubmed_snapshot import PubMedSnapshotGateway
from src.runtime.provider import build_provider
from src.runtime.runner import ArtifactState, TaskOutcome, TaskRunner
from src.runtime.skills import load_body, with_body
from src.util import sha256_obj

SKILL = "litqa-literature-synthesis"
INSTRUCTION = ("RETRIEVAL_INSTRUCTION: Build a concise PubMed search query using the "
               "central biomedical entity and likely technical synonyms, plus the measured "
               "outcome. Remove generic yes/no phrasing.")
ANSWER_STRATEGY = (
    "Evidence decision rule\n"
    "1. Evaluate the exact proposition in the question, including population, "
    "outcome, direction, and qualifiers. Topic overlap alone is not support.\n"
    "2. For clinical prevention or treatment effects, prioritize directly measured "
    "clinical outcomes over mechanistic hypotheses or observational associations. "
    "Words such as may, possible, and trials under way do not establish efficacy.\n"
    "3. Choose yes only if a cited passage establishes that proposition; choose no "
    "only if a cited passage directly refutes it. If relevant evidence is missing "
    "or conflicting, choose Insufficient information. Cite the decisive passage."
)
EVIDENCE_COMPARISON = (
    "Evidence comparison rule\n"
    "1. Before choosing yes or no, compare each retrieved article with the "
    "exact question: entity, population, setting, intervention, outcome and "
    "direction. An article about a related marker or mechanism is not a "
    "direct answer.\n"
    "2. For an efficacy or prognosis question, weigh measured patient "
    "outcomes and study design. A relevant systematic review or controlled "
    "trial can outweigh an early feasibility study, mechanism paper, or "
    "association. Do not count papers as votes.\n"
    "3. Distinguish disagreement about the same population and outcome from "
    "results in different settings. Prefer the most direct and reliable "
    "evidence; if the balance cannot be resolved, abstain. Absence of a "
    "positive study alone does not establish no.\n"
    "4. Cite the passage that establishes the selected answer. If no "
    "retrieved passage directly supports either answer, abstain."
)
SOURCE_FOCUS = (
    "RETRIEVAL_INSTRUCTION: Choose saved PubMed sources matching the exact "
    "entities and outcome. If two molecular entities are named, include "
    "entity_pair; omit broad/fallback unless precise sources are sparse. "
    "Otherwise prefer raw and keywords. Preserve molecular punctuation."
)
SNAPSHOT = ROOT / "artifacts/probe/pubmed_snapshot/no-doc-four-v3-2026-09-29.json"
OUTPUT = ROOT / "artifacts/probe/pubmed_snapshot/window-grounding-v2-train4.json"


def compact(out):
    return {"task_id": out.task_id, "status": out.status, "correct": out.correct,
            "answer_index": out.answer_index, "query": out.retrieval_query,
            "query_source": out.retrieval_query_source,
            "source_names": out.retrieval_source_names,
            "top3": out.retrieved_doc_ids,
            "passage_hashes": out.retrieved_passage_hashes,
            "cited": out.cited, "citation_supported": out.citation_supported,
            "support_reason": out.citation_support_detail,
            "answer_evidence_consistent": out.answer_evidence_consistent,
            "answer_evidence_reason": out.answer_evidence_detail,
            "verdict": [e["payload"] for e in out.events
                        if e["event_type"] == "citation.support"],
            "cost": out.cost, "error": out.error}


def load_checkpoint(path: Path, pilot_id: str) -> dict:
    saved = {}
    if not path.exists():
        return saved
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry["pilot_id"] != pilot_id:
            raise ValueError("checkpoint belongs to a different pilot protocol")
        key = (entry["task_id"], entry["arm"])
        if key in saved:
            raise ValueError("duplicate checkpointed task arm")
        saved[key] = TaskOutcome(**entry["outcome"])
    return saved


def append_checkpoint(path: Path, pilot_id: str, task_id: str,
                      arm: str, outcome: TaskOutcome) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"pilot_id": pilot_id, "task_id": task_id,
                                 "arm": arm, "outcome": outcome.to_dict()},
                                ensure_ascii=False) + "\n")


def main():
    raise SystemExit(
        "unplanned historical window/grounding scoring is retired: direct "
        "model calls have no frozen execution-plan hash, request caps, or "
        "shared call ledger"
    )

    # Retained as historical implementation only pending governed replay.
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--task-id", action="append",
                        help="Choose named train tasks for a targeted diagnostic")
    parser.add_argument("--variant", choices=("query_synonyms", "answer_strategy",
                                              "evidence_comparison", "source_focus"),
                        default="query_synonyms")
    parser.add_argument("--model-sources", action="store_true",
                        help="Allow Prompt/Skill to select captured PubMed ESearch subsets")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--balanced", action="store_true",
                        help="Exploratory 1:1 yes/no subset from train labels")
    parser.add_argument("--exclude-ids-from", type=Path,
                        help="Run snapshot tasks absent from a previous pilot report")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--contrastive", action="store_true")
    parser.add_argument("--judge-mode", choices=("answer", "proposition"),
                        default="answer")
    parser.add_argument("--judge-model-id", default="")
    parser.add_argument("--check-top3", action="store_true")
    parser.add_argument("--freeze-query-from", type=Path, action="append",
                        help="Paired pilot report whose baseline queries are fixed for both arms")
    parser.add_argument("--anchor-question", action="store_true")
    args = parser.parse_args()
    if args.model_sources and args.freeze_query_from:
        parser.error("model-sources cannot be paired with frozen-query manifests")
    if args.variant == "source_focus" and not args.model_sources:
        parser.error("source_focus needs --model-sources")
    if args.check_top3 and (args.no_judge or args.judge_mode != "proposition"):
        parser.error("check-top3 requires enabled proposition judge")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    ids = [r["task_id"] for r in snapshot["tasks"]]
    if args.task_id:
        wanted = set(args.task_id)
        if not wanted <= set(ids):
            parser.error("some named train tasks are absent from snapshot")
        ids = [i for i in ids if i in wanted]
    if args.balanced and (args.skip or args.limit is not None):
        parser.error("balanced cannot be combined with skip or limit")
    if args.skip < 0:
        parser.error("skip must be nonnegative")
    ids = ids[args.skip:]
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("limit must be positive")
        ids = ids[:args.limit]
    if args.exclude_ids_from:
        excluded = set(json.loads(args.exclude_ids_from.read_text(
            encoding="utf-8"))["task_ids"])
        ids = [i for i in ids if i not in excluded]
    if not ids:
        parser.error("no train tasks selected")
    tasks_by_id = {t["task_id"]: t for t in load_tasks("train", task_ids=set(ids))}
    if args.balanced:
        no_ids = [i for i in ids if tasks_by_id[i]["correct_index"] == 1]
        yes_ids = [i for i in ids if tasks_by_id[i]["correct_index"] == 0]
        yes_ids = sorted(yes_ids, key=lambda x: hashlib.sha256(
            x.encode()).hexdigest())[:len(no_ids)]
        ids = [i for i in ids if i in set(no_ids + yes_ids)]
    tasks = [tasks_by_id[i] for i in ids]
    gateway = PubMedSnapshotGateway(snapshot, top_k=3, passage_mode="query_window")
    cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
    cfg["retrieval"]["query_control"] = {
        "enabled": True, "mode": "model_sources" if args.model_sources else "model"}
    if args.freeze_query_from:
        frozen = {}
        for path in args.freeze_query_from:
            prior = json.loads(path.read_text(encoding="utf-8"))
            if prior.get("snapshot_content_sha256") != snapshot["content_sha256"]:
                raise ValueError("frozen query report uses a different snapshot")
            frozen.update({row["task_id"]: row["query"]
                           for row in prior["per_task"]["baseline"]})
        if not set(ids) <= set(frozen):
            raise ValueError("frozen query reports omit selected tasks")
        cfg["retrieval"]["query_control"] = {
            "enabled": True, "mode": "frozen",
            "queries": {i: frozen[i] for i in ids}}
    cfg["retrieval"]["pubmed_passage_mode"] = "query_window"
    cfg["retrieval"]["anchor_question_terms"] = args.anchor_question
    cfg["retrieval"]["snapshot_content_sha256"] = snapshot["content_sha256"]
    cfg["skills"]["pinned_skill"] = SKILL
    cfg["evaluation"]["semantic_citation"] = {
        "enabled": not args.no_judge, "contrastive": args.contrastive,
        "mode": args.judge_mode, "check_top3": args.check_top3}
    if args.judge_model_id:
        cfg["evaluation"]["semantic_citation"]["model_id"] = args.judge_model_id
    base = load_state("baseline-v1-cited")
    addition = {"query_synonyms": INSTRUCTION,
                "answer_strategy": ANSWER_STRATEGY,
                "evidence_comparison": EVIDENCE_COMPARISON,
                "source_focus": SOURCE_FOCUS}[args.variant]
    body = load_body(base.registry.get(SKILL)).rstrip() + "\n\n" + addition + "\n"
    cand = Candidate("window-" + args.variant, base.release_id, "skill_body", SKILL,
                     diff="train-only " + args.variant, new_content=body)
    gate = static_gate(cand, cfg, base)
    if not gate["passed"]:
        raise RuntimeError(f"static gate failed: {gate}")
    changed = ArtifactState(cand.candidate_id, base.system_prompt,
                            with_body(base.registry, SKILL, body))
    provider = build_provider(cfg)
    provider.cache = False
    outcomes = {"baseline": [], "instruction": []}
    protocol_fp = protocol_fingerprint(cfg)
    pilot_id = sha256_obj({"snapshot": snapshot["content_sha256"],
                           "config": cfg, "protocol_fingerprint": protocol_fp,
                           "body": body, "task_ids": ids})
    journal = args.output.with_name(args.output.stem + ".partial.jsonl")
    saved = load_checkpoint(journal, pilot_id)
    for task in tasks:
        for arm, state in (("baseline", base), ("instruction", changed)):
            key = (task["task_id"], arm)
            if key in saved:
                outcome = saved[key]
            else:
                outcome = TaskRunner(cfg, state, provider=provider,
                                     gateway=gateway).run(task)
                append_checkpoint(journal, pilot_id, task["task_id"], arm,
                                  outcome)
            outcomes[arm].append(outcome)
    b, c = outcomes["baseline"], outcomes["instruction"]
    metrics = {name: aggregate(rows) for name, rows in outcomes.items()}
    labels = {task["task_id"]: task["options"][task["correct_index"]]
              for task in tasks}
    by_label = {arm: {label: {
        "n": sum(labels[row.task_id] == label for row in rows),
        "correct": sum(labels[row.task_id] == label and row.correct
                       for row in rows),
        "answered": sum(labels[row.task_id] == label and row.status == "ok"
                        for row in rows)}
        for label in ("yes", "no")}
        for arm, rows in outcomes.items()}
    proxy_cfg = {"runtime": {"provider": "http"},
                 "gates": {"accept": {"minimum": {
                     "citation_support_rate": 0.9,
                     "answer_evidence_consistency_rate": 0.9},
                     "tolerance": {
                     "citation_support_rate_pp": 0,
                     "answer_evidence_consistency_rate_pp": 0,
                     "required": ["citation_support_rate",
                                  "answer_evidence_consistency_rate"]}}}}
    proxy_gate = check_tolerances(metrics["baseline"], metrics["instruction"],
                                  proxy_cfg)
    report = {"status": "train_only_paired_pilot",
              "variant": args.variant,
              "snapshot_content_sha256": snapshot["content_sha256"],
              "candidate_body_sha256": sha256_obj(body),
              "pilot_id": pilot_id,
              "protocol_fingerprint": protocol_fp,
              "candidate_static_gate": gate,
              "task_ids": ids, "metrics": metrics,
              "by_label": by_label,
              "selection": "balanced train diagnostic" if args.balanced else
              "snapshot order excluding prior" if args.exclude_ids_from else
              "snapshot order",
              "snapshot_protocol": snapshot["protocol"],
              "query_protocol": cfg["retrieval"]["query_control"]["mode"],
              "question_anchor": args.anchor_question,
              "semantic_judge_enabled": not args.no_judge,
              "semantic_judge_contrastive": args.contrastive,
              "semantic_judge_check_top3": args.check_top3,
              "semantic_judge_mode": args.judge_mode,
              "semantic_judge_model_id": args.judge_model_id or
              cfg["models"]["task"]["model_id"],
              "support_proxy_gate": proxy_gate,
              "candidate_advancement_eligible": False,
              "advancement_blockers": (["no positive accuracy gain on the paired train pilot"]
                                       if metrics["instruction"]["accuracy"] <=
                                       metrics["baseline"]["accuracy"] else [])
                                      + ["citation grounding is not yet independently calibrated",
                                         "train exploration is not confirmation"],
              "query_changed": sum(x.retrieval_query != y.retrieval_query
                                   for x, y in zip(b, c)),
              "source_selection_changed": sum(
                  x.retrieval_source_names != y.retrieval_source_names
                  for x, y in zip(b, c)),
              "pool_changed": sum(
                  {d["doc_id"] for d in gateway.corpus_for_query_names(
                      task, x.retrieval_source_names)} !=
                  {d["doc_id"] for d in gateway.corpus_for_query_names(
                      task, y.retrieval_source_names)}
                  for task, x, y in zip(tasks, b, c)) if args.model_sources else 0,
              "top3_changed": sum(x.retrieved_doc_ids != y.retrieved_doc_ids
                                  for x, y in zip(b, c)),
              "passage_changed": sum(x.retrieved_passage_hashes != y.retrieved_passage_hashes
                                     for x, y in zip(b, c)),
              "per_task": {name: [compact(o) for o in rows]
                           for name, rows in outcomes.items()},
              "limit": "Explored train tasks only. Model citation and top-3 "
                       "judgments are calibrated proxies, not gold entailment. "
                       "No candidate advancement or dev access."}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "query_changed", "source_selection_changed", "pool_changed", "top3_changed",
        "passage_changed")},
        ensure_ascii=False, indent=2))
    print(json.dumps({arm: {key: metrics[arm][key] for key in (
        "accuracy", "coverage", "answered_precision", "citation_support_rate",
        "answer_evidence_consistency_rate")}
        for arm in metrics}, ensure_ascii=False, indent=2))
    print("support_proxy_gate_passed", proxy_gate["passed"])
    print(args.output)


if __name__ == "__main__":
    main()
