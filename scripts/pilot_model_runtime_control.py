"""Train-only pilot: model Skill routing and per-question query generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rsi import load_config
from src.contracts.types import Candidate
from src.data.bioasq_unitor import load_tasks
from src.governance.policy import static_gate
from src.governance.release import load_state
from src.retrieval.pubmed_snapshot import PubMedSnapshotGateway
from src.runtime.provider import build_provider
from src.runtime.runner import ArtifactState, TaskRunner
from src.runtime.skills import load_body, with_body
from src.util import sha256_obj

SNAPSHOT = ROOT / "artifacts/probe/pubmed_snapshot/no-doc-four-v3-2026-09-29.json"
OUTPUT = ROOT / "artifacts/probe/pubmed_snapshot/model-runtime-control-v2-grounding.json"
SKILL = "litqa-literature-synthesis"
INSTRUCTION = ("RETRIEVAL_INSTRUCTION: Extract disease, intervention and measured "
               "outcome; retain exact entity names and useful synonyms; omit yes/no words.")


def row(outcome):
    return {"task_id": outcome.task_id, "status": outcome.status,
            "correct": outcome.correct, "answer_index": outcome.answer_index,
            "skill": outcome.skill_recalled, "query": outcome.retrieval_query,
            "query_source": outcome.retrieval_query_source,
            "top3": outcome.retrieved_doc_ids, "cited": outcome.cited,
            "citation_supported": outcome.citation_supported,
            "citation_support_detail": outcome.citation_support_detail,
            "citation_support_trace": [e["payload"] for e in outcome.events
                                       if e["event_type"] == "citation.support"],
            "error": outcome.error, "cost": outcome.cost,
            "model_events": [{"event": e["event_type"], "latency_ms": e["latency_ms"],
                              "token_usage": e["token_usage"]}
                             for e in outcome.events
                             if e["event_type"].startswith("model.call")]}


def main():
    raise SystemExit(
        "unplanned historical pilot is retired: direct model-runtime scoring "
        "has no frozen execution-plan hash, request caps, or shared call ledger"
    )

    # Retained as historical implementation only pending governed replay.
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    if snapshot["protocol"] != "pubmed-esearch-efetch-snapshot-v3":
        raise RuntimeError("question-only PubMed v3 snapshot required")
    ids = [r["task_id"] for r in snapshot["tasks"]]
    tasks_by_id = {t["task_id"]: t for t in load_tasks("train", task_ids=set(ids))}
    tasks = [tasks_by_id[i] for i in ids]
    base = load_state("baseline-v1-cited")
    body = load_body(base.registry.get(SKILL)).rstrip() + "\n\n" + INSTRUCTION + "\n"
    changed = ArtifactState("query-instruction-pilot", base.system_prompt,
                            with_body(base.registry, SKILL, body))
    cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
    cfg["retrieval"]["query_control"] = {"enabled": True, "mode": "model"}
    cfg["skills"]["pinned_skill"] = SKILL
    cfg["evaluation"]["semantic_citation"] = {"enabled": True}
    candidate = Candidate("query-instruction-pilot", base.release_id,
                          "skill_body", SKILL, diff="add dynamic query instruction",
                          new_content=body)
    gate = static_gate(candidate, cfg, base)
    if not gate["passed"]:
        raise RuntimeError(f"candidate static gate failed: {gate}")
    provider = build_provider(cfg)
    provider.cache = False
    gateway = PubMedSnapshotGateway(snapshot, top_k=3)
    paired = {"baseline": [], "instruction": []}
    for task in tasks:
        paired["baseline"].append(row(TaskRunner(
            cfg, base, gateway=gateway, provider=provider).run(task)))
        paired["instruction"].append(row(TaskRunner(
            cfg, changed, gateway=gateway, provider=provider).run(task)))

    # Separate routing mechanism: same state and question query, only router changes.
    route_cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
    route_cfg["retrieval"]["query_control"] = {"enabled": False}
    route_cfg["skills"].pop("pinned_skill", None)
    routed = {"lexical": [], "model": []}
    for task in tasks:
        for mode in routed:
            route_cfg["skills"]["routing_mode"] = mode
            routed[mode].append(row(TaskRunner(
                route_cfg, base, gateway=gateway, provider=provider).run(task)))

    result = {"status": "train_only_mechanism_pilot",
              "snapshot_content_sha256": snapshot["content_sha256"],
              "candidate_body_sha256": sha256_obj(body),
              "candidate_static_gate": gate,
              "task_ids": ids, "paired_query": paired, "routing": routed,
              "query_changed": sum(a["query"] != b["query"] for a, b in zip(
                  paired["baseline"], paired["instruction"])),
              "query_top3_order_changed": sum(a["top3"] != b["top3"] for a, b in zip(
                  paired["baseline"], paired["instruction"])),
              "baseline_supported": sum(x["citation_supported"] is True
                                        for x in paired["baseline"]),
              "instruction_supported": sum(x["citation_supported"] is True
                                           for x in paired["instruction"]),
              "route_changed": sum(a["skill"] != b["skill"] for a, b in zip(
                  routed["lexical"], routed["model"])),
              "limit": "Mechanism pilot on four train questions; citation support is "
                       "a same-model quote-checked proxy, not gold entailment. "
                       "BioASQ dev was not accessed."}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "status", "query_changed", "query_top3_order_changed", "route_changed",
        "baseline_supported", "instruction_supported")},
        ensure_ascii=False, indent=2))
    print(OUTPUT)


if __name__ == "__main__":
    main()
