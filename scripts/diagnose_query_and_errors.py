"""探索性诊断：固定语料 query 干预及历史 23+23 失败样本归因。

不调用模型，不访问 selection_gate/final_holdout。gold 标记只用于离线诊断。
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rsi import load_config
from src.data.split import load_tasks
from src.eval.options import inject_all
from src.governance.release import load_state
from src.retrieval.gateway import RetrievalGateway, carries_conclusion
from src.retrieval.intent import query_for
from src.runtime.channels import effective_modes, load_channels
from src.runtime.runner import chunking_config
from src.runtime.skills import load_body, recall
from src.util import sha256_obj


def main() -> int:
    cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
    state = load_state("baseline-v1-cited")
    tasks = sorted(load_tasks("pubmedqa", ["dev_optimize"]),
                   key=lambda t: t["task_id"])[30:230]
    if bool(cfg.get("evaluation", {}).get("insufficient_option", True)):
        tasks = inject_all(tasks)
    old_path = ROOT / "artifacts/runs/run-4eb02d4643130dc3/per_task.jsonl"
    old = {str(r["task_id"]): r for r in
           (json.loads(x) for x in old_path.read_text(encoding="utf-8").splitlines())}
    gateway = RetrievalGateway(
        backend=cfg["retrieval"]["backend"], top_k=int(cfg["retrieval"]["top_k"]),
        corpus_version=cfg["retrieval"]["corpus_version"],
        doc_id_scheme=cfg["retrieval"]["doc_id_scheme"],
        chunking=chunking_config(cfg))
    channels = load_channels(cfg)
    templates = {
        "prompt_options": "RETRIEVAL_QUERY: {question} {options}",
        "prompt_result": "RETRIEVAL_QUERY: {question} study result outcome",
        "skill_options": "RETRIEVAL_QUERY: {question} {options}",
    }
    changes = []
    error_rows = []
    for task in tasks:
        tid = str(task["task_id"])
        hit = recall(task["question"] + " " + " ".join(task["options"]),
                     state.registry,
                     min_score=int(cfg.get("skills", {}).get("recall_min_score", 2)))
        body = load_body(state.registry.get(hit["name"])) if hit else ""
        modes = effective_modes(state.system_prompt, body, channels)
        corpus = gateway.corpus(task)
        base = gateway.search(task, skill_recalled=bool(hit),
                              rerank=bool(modes.get("rerank")), corpus=corpus)
        base_ids = [d["doc_id"] for d in base]
        variants = {}
        for name, directive in templates.items():
            if name.startswith("skill") and not hit:
                continue
            prompt = directive if name.startswith("prompt") else ""
            skill_body = directive if name.startswith("skill") else ""
            query, source = query_for(task, prompt, skill_body, enabled=True)
            docs = gateway.search(task, skill_recalled=bool(hit),
                                  rerank=bool(modes.get("rerank")),
                                  query=query, corpus=corpus)
            ids = [d["doc_id"] for d in docs]
            variants[name] = {
                "query": query, "source": source, "doc_ids": ids,
                "order_changed": ids != base_ids,
                "gold_rank": next((i for i, d in enumerate(docs)
                                   if d.get("is_gold")), -1),
            }
        changes.append({"task_id": tid,
                        "corpus_sha256": sha256_obj([(d["doc_id"], d["text"])
                                                       for d in corpus]),
                        "base_doc_ids": base_ids,
                        "base_gold_rank": next((i for i, d in enumerate(base)
                                                if d.get("is_gold")), -1),
                        "variants": variants})

        row = old.get(tid)
        if not row or not row.get("gold_in_topk") or row.get("correct"):
            continue
        bucket = "abstain" if row.get("status") == "abstain" else "wrong"
        if bucket == "abstain":
            if not carries_conclusion(task["options"], " ".join(
                    d["text"][:1500] for d in base)):
                cause = "gold_doc_retrieved_but_no_visible_option_signal"
            else:
                cause = "option_signal_visible_but_model_abstained"
        else:
            cited = set(row.get("cited") or [])
            if not cited:
                cause = "wrong_without_citation"
            elif any(d["doc_id"] in cited and d.get("is_gold") for d in base):
                cause = "wrong_despite_citing_gold"
            else:
                cause = "wrong_citing_only_other_docs"
        error_rows.append({"task_id": tid, "bucket": bucket, "cause": cause,
                           "question": task["question"],
                           "options": task["options"],
                           "answer_index": row.get("answer_index"),
                           "gold_index": row.get("gold_index"),
                           "cited": row.get("cited"),
                           "retrieved_doc_ids": base_ids,
                           "gold_rank": next((i for i, d in enumerate(base)
                                              if d.get("is_gold")), -1),
                           "gold_visible_excerpt": next((d["text"][:400]
                                                         for d in base if d.get("is_gold")), "")})
    counts = {name: sum(int(row["variants"].get(name, {}).get("order_changed", False))
                        for row in changes) for name in templates}
    report = {
        "status": "exploratory_reused_200",
        "protocol": "offline-fixture real-v3; fixed per-task corpus; query ranking only",
        "historical_run": "run-4eb02d4643130dc3",
        "n_tasks": len(tasks), "changed_order_counts": counts,
        "error_counts": dict(collections.Counter(
            f"{r['bucket']}:{r['cause']}" for r in error_rows)),
        "errors": error_rows, "query_changes": changes,
    }
    out = ROOT / "artifacts/probe/query_mechanism_and_errors_2026-09-29.json"
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_tasks", "changed_order_counts", "error_counts")},
                     ensure_ascii=False, indent=2))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
