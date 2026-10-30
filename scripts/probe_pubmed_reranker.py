"""Train-only, frozen-corpus reranker comparison; labels never reach the API."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnose_pubmed_snapshot import shingles
from src.data.bioasq_unitor import load_tasks
from src.retrieval.pubmed_snapshot import PubMedSnapshotGateway
from src.runtime.provider import load_project_dotenv
from src.runtime.call_ledger import (end_http_attempt, record_operation_end,
                                    start_http_attempt, start_operation)
from src.runtime.call_ledger import summarize_ledger
from src.util import (artifacts_path, artifacts_root, sha256_file, sha256_obj,
                      write_json)

API_URL = "https://api.siliconflow.cn/v1/rerank"
INSTRUCTION = ("Rank biomedical articles by direct evidence for the exact "
               "entities, population, intervention, outcome and qualifiers "
               "in the query. General topic overlap is insufficient.")
PROTOCOL = "pubmed-reranker-frozen-train-probe-v2"
MAX_HTTP_ATTEMPTS = 4
MAX_DOCUMENT_CHARS = 6000
MAX_CANDIDATE_DOCS = 100
MAX_REQUEST_TEXT_CHARS = 600000


def build_plan(*, snapshot_path: Path, query_reports: list[Path],
               snapshot: dict, queries: dict[str, str], tasks: dict,
               task_ids: list[str], model_id: str, prefilter_k: int,
               output_path: Path, request_material: dict[str, dict]) -> dict:
    """Freeze exact train-only rerank inputs, source bytes, and request bounds."""
    sources = (Path(__file__).resolve(),
               (ROOT / "src/data/bioasq_unitor.py").resolve(),
               (ROOT / "src/retrieval/pubmed_snapshot.py").resolve(),
               (ROOT / "src/runtime/call_ledger.py").resolve(),
               (ROOT / "src/runtime/provider.py").resolve())
    ordered_ids = list(task_ids)
    if not ordered_ids or len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("reranker plan requires non-empty unique train task IDs")
    if set(ordered_ids) - set(queries) or set(ordered_ids) - set(tasks):
        raise ValueError("reranker plan inputs omit a selected task")
    if set(ordered_ids) != set(request_material):
        raise ValueError("reranker plan request material differs from selected tasks")
    if any(str((tasks[task_id] or {}).get("split", "train")) != "train"
           for task_id in ordered_ids):
        raise ValueError("reranker probe is restricted to the train split")
    unsigned = {
        "schema": PROTOCOL,
        "protocol": PROTOCOL,
        "snapshot_path": str(snapshot_path.resolve()),
        "snapshot_file_sha256": sha256_file(snapshot_path),
        "snapshot_content_sha256": str(snapshot.get("content_sha256") or ""),
        "query_reports": [{"path": str(path.resolve()),
                            "sha256": sha256_file(path)} for path in query_reports],
        "selected_queries_sha256": sha256_obj(
            {task_id: queries[task_id] for task_id in ordered_ids}),
        "selected_train_tasks_sha256": sha256_obj(
            {task_id: tasks[task_id] for task_id in ordered_ids}),
        "request_material_sha256": sha256_obj(request_material),
        "per_task_request_bounds": {
            task_id: {key: request_material[task_id][key]
                      for key in ("candidate_count", "request_text_chars")}
            for task_id in ordered_ids},
        "task_ids": ordered_ids,
        "model_id": str(model_id),
        "endpoint_host": "api.siliconflow.cn",
        "instruction_sha256": sha256_obj(INSTRUCTION),
        "maximum_document_chars": MAX_DOCUMENT_CHARS,
        "maximum_candidate_documents": MAX_CANDIDATE_DOCS,
        "maximum_request_text_chars": MAX_REQUEST_TEXT_CHARS,
        "prefilter_k": int(prefilter_k),
        "logical_request_cap": len(ordered_ids),
        "physical_request_cap": len(ordered_ids) * MAX_HTTP_ATTEMPTS,
        "maximum_attempts_per_operation": MAX_HTTP_ATTEMPTS,
        "output_path": str(output_path),
        "source_sha256s": {path.relative_to(ROOT).as_posix(): sha256_file(path)
                            for path in sources},
        "split": "train",
        "experiment_role": "retrieval_runtime_diagnostic",
        "candidate_selection": False,
        "candidate_promotion": False,
        "independent_confirmation_eligible": False,
        "selection_gate_read": False,
        "final_holdout_read": False,
        "model_calls_during_preflight": 0,
        "live_authorized": False,
        "explicit_user_approval_required_before_execution": True,
    }
    return {**unsigned, "plan_sha256": sha256_obj(unsigned)}


def verify_plan(plan: dict, expected_sha256: str) -> None:
    actual = sha256_obj({key: value for key, value in plan.items()
                         if key != "plan_sha256"})
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or ""):
        raise ValueError("execution requires a 64-character --expected-plan-sha256")
    if plan.get("plan_sha256") != actual or expected_sha256 != actual:
        raise ValueError("expected plan SHA-256 does not match the current reranker plan")


def validate_checkpoint_row(row: dict, *, probe_id: str, task_id: str,
                            query: str, candidate_docs: list[dict],
                            bm25_top3: list[str]) -> None:
    """Reject stale, incomplete, or tampered rerank checkpoint rows."""
    top_n = min(3, len(candidate_docs))
    indices = row.get("rerank_indices")
    scores = row.get("relevance_scores")
    if (row.get("probe_id") != probe_id or row.get("task_id") != task_id
            or row.get("query") != query
            or row.get("candidate_count") != len(candidate_docs)
            or row.get("bm25_top3") != bm25_top3
            or not isinstance(indices, list) or len(indices) != top_n
            or len(set(indices)) != len(indices)
            or any(type(index) is not int or not 0 <= index < len(candidate_docs)
                   for index in indices)
            or row.get("rerank_top3") != [candidate_docs[index]["doc_id"]
                                          for index in indices]
            or not isinstance(scores, list) or len(scores) != top_n):
        raise ValueError("rerank checkpoint row differs from its frozen task inputs")
    try:
        values = [float(score) for score in scores]
    except (TypeError, ValueError) as exc:
        raise ValueError("rerank checkpoint scores are malformed") from exc
    if any(not math.isfinite(score) for score in values):
        raise ValueError("rerank checkpoint scores are non-finite")


def rerank(session: requests.Session, api_key: str, model_id: str,
           query: str, texts: list[str]) -> dict:
    payload = {"model": model_id, "query": query, "documents": texts,
               "top_n": min(3, len(texts)), "return_documents": False}
    if model_id.startswith("Qwen/Qwen3-Reranker-"):
        payload["instruction"] = INSTRUCTION
    last_error = None
    operation_id = start_operation(None, "http_rerank", model_id)
    terminal_status = "failed"
    error_type = "RuntimeError"
    try:
        for attempt in range(MAX_HTTP_ATTEMPTS):
            start = time.monotonic()
            start_http_attempt(operation_id, attempt=attempt + 1)
            try:
                response = session.post(
                    API_URL, json=payload,
                    headers={"Authorization": "Bearer " + api_key,
                             "Content-Type": "application/json"},
                    timeout=75)
            except requests.RequestException as exc:
                end_http_attempt(operation_id, attempt=attempt + 1,
                                 status=None, error_type=type(exc).__name__)
                last_error = type(exc).__name__
            else:
                end_http_attempt(operation_id, attempt=attempt + 1,
                                 status=int(response.status_code))
                if response.status_code in {429, 500, 502, 503, 504}:
                    last_error = f"HTTP {response.status_code}"
                else:
                    try:
                        response.raise_for_status()
                        data = response.json()
                        results = data.get("results", [])
                        indices = [int(item["index"]) for item in results]
                        scores = [float(item["relevance_score"]) for item in results]
                        if (len(indices) != payload["top_n"]
                                or len(set(indices)) != len(indices)
                                or any(index < 0 or index >= len(texts)
                                       for index in indices)
                                or any(not math.isfinite(score)
                                       for score in scores)):
                            raise ValueError("invalid rerank result indices or scores")
                        terminal_status = "success"
                        error_type = ""
                        return {"indices": indices, "scores": scores,
                                "meta": data.get("meta", {}),
                                "latency_ms": round((time.monotonic() - start) * 1000)}
                    except (requests.RequestException, ValueError, KeyError,
                            TypeError) as exc:
                        last_error = type(exc).__name__
            if attempt + 1 < MAX_HTTP_ATTEMPTS:
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("rerank service failed after bounded retries: "
                           + str(last_error or "unknown failure"))
    except Exception as exc:
        error_type = type(exc).__name__
        raise
    finally:
        record_operation_end(operation_id, status=terminal_status,
                             error_type=error_type)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--query-report", type=Path, action="append", required=True)
    parser.add_argument("--model-id", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--prefilter-k", type=int, default=0,
                        help="Send only BM25 top-k to reranker; 0 sends full pool")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true",
                      help="freeze exact train-only inputs and request bounds without calls")
    mode.add_argument("--execute", action="store_true",
                      help="execute only a previously reviewed exact plan")
    parser.add_argument("--expected-plan-sha256", default="",
                        help="required with --execute")
    args = parser.parse_args(argv)
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256):
        parser.error("--execute requires a 64-character --expected-plan-sha256")
    if args.validate_only and args.expected_plan_sha256:
        parser.error("--expected-plan-sha256 is only valid with --execute")
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("reranker probe requires an isolated RSI_HOME")
    output_path = args.output
    if not output_path.is_absolute():
        output_path = artifacts_root() / output_path
    output_path = output_path.resolve()
    if not output_path.is_relative_to(artifacts_root().resolve()):
        raise RuntimeError("reranker output must be inside the isolated RSI_HOME")
    if output_path.exists():
        raise FileExistsError("reranker report output already exists")

    snapshot_path = args.snapshot.resolve()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    ids = [r["task_id"] for r in snapshot["tasks"]]
    if len(ids) != len(set(ids)):
        raise ValueError("reranker snapshot contains duplicate train task IDs")
    if args.task_id:
        if not set(args.task_id) <= set(ids):
            parser.error("named task absent from snapshot")
        ids = [i for i in ids if i in set(args.task_id)]
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("limit must be positive")
        ids = ids[:args.limit]
    if args.prefilter_k < 0 or args.prefilter_k in {1, 2}:
        parser.error("prefilter-k must be 0 or at least 3")
    queries = {}
    query_report_paths = [path.resolve() for path in args.query_report]
    for path in query_report_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["snapshot_content_sha256"] != snapshot["content_sha256"]:
            raise ValueError("query report and snapshot differ")
        baseline_rows = report["per_task"].get("baseline")
        if not isinstance(baseline_rows, list):
            raise ValueError("query report lacks baseline query rows")
        for row in baseline_rows:
            task_id, query = str(row.get("task_id") or ""), row.get("query")
            if not task_id or not isinstance(query, str) or not query.strip():
                raise ValueError("query report contains an invalid task/query row")
            if task_id in queries and queries[task_id] != query:
                raise ValueError("query reports contain conflicting queries for one task")
            queries[task_id] = query
    if not set(ids) <= set(queries):
        raise ValueError("query reports omit selected tasks")
    task_rows = load_tasks("train", task_ids=set(ids))
    tasks = {t["task_id"]: t for t in task_rows}
    if (len(tasks) != len(task_rows) or set(tasks) != set(ids)
            or any(str(task.get("split", "train")) != "train"
                   for task in task_rows)):
        raise ValueError("loaded reranker task rows are duplicated or outside train scope")
    gateway = PubMedSnapshotGateway(snapshot)
    prefilter_gateway = (PubMedSnapshotGateway(snapshot, top_k=args.prefilter_k)
                         if args.prefilter_k else None)
    prepared = {}
    request_material = {}
    for task_id in ids:
        task = tasks[task_id]
        query = queries[task_id] + " " + gateway.anchor_query(
            task["question"], max_terms=20)
        docs = gateway.corpus(task)
        bm25 = [doc["doc_id"] for doc in gateway.search(
            task, query=query, corpus=docs)]
        rerank_docs = (prefilter_gateway.search(task, query=query, corpus=docs)
                       if prefilter_gateway else docs)
        request_text_chars = len(query) + sum(
            len(str(doc.get("text") or "")[:MAX_DOCUMENT_CHARS])
            for doc in rerank_docs)
        if len(rerank_docs) > MAX_CANDIDATE_DOCS:
            raise ValueError("reranker candidate pool exceeds the frozen document-count limit")
        if request_text_chars > MAX_REQUEST_TEXT_CHARS:
            raise ValueError("reranker request exceeds the frozen text-size limit")
        if len(rerank_docs) < 3:
            raise ValueError("reranker task needs at least three candidate documents")
        prepared[task_id] = {"query": query, "docs": docs,
                             "bm25_top3": bm25, "rerank_docs": rerank_docs}
        request_material[task_id] = {
            "query_sha256": sha256_obj(query),
            "bm25_top3": bm25,
            "candidate_doc_ids": [doc["doc_id"] for doc in rerank_docs],
            "candidate_text_sha256s": [sha256_obj(doc["text"][:MAX_DOCUMENT_CHARS])
                                       for doc in rerank_docs],
            "candidate_count": len(rerank_docs),
            "request_text_chars": request_text_chars,
        }
    plan = build_plan(
        snapshot_path=snapshot_path, query_reports=query_report_paths,
        snapshot=snapshot, queries=queries, tasks=tasks, task_ids=ids,
        model_id=args.model_id, prefilter_k=args.prefilter_k,
        output_path=output_path, request_material=request_material)
    plan_path = artifacts_path("probe", "pubmed_reranker", "plans",
                               "plan-" + plan["plan_sha256"] + ".json")
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise RuntimeError("saved reranker plan differs from recomputed inputs")
    else:
        write_json(plan_path, plan)
    if args.validate_only:
        print(json.dumps({
            "status": "validated_not_run", "plan": str(plan_path),
            "plan_sha256": plan["plan_sha256"], "task_count": len(ids),
            "split": "train", "logical_request_cap": plan["logical_request_cap"],
            "physical_request_cap": plan["physical_request_cap"],
            "model_calls": 0, "selection_gate_read": False,
            "final_holdout_read": False,
        }, ensure_ascii=False))
        return 0
    verify_plan(plan, args.expected_plan_sha256)

    probe_id = sha256_obj({key: plan[key] for key in (
        "snapshot_file_sha256", "snapshot_content_sha256", "query_reports",
        "selected_queries_sha256", "selected_train_tasks_sha256",
        "request_material_sha256", "task_ids", "model_id", "instruction_sha256",
        "maximum_document_chars", "prefilter_k", "source_sha256s")})
    journal = output_path.with_name(output_path.stem + ".partial.jsonl")
    ledger_path = artifacts_path(
        "probe", "pubmed_reranker", "ledgers",
        "ledger-" + plan["plan_sha256"] + ".jsonl")
    saved = {}
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            task_id = str(entry.get("task_id") or "")
            if task_id not in prepared:
                raise ValueError("rerank checkpoint task is outside the frozen plan")
            data = prepared[task_id]
            validate_checkpoint_row(
                entry, probe_id=probe_id, task_id=task_id,
                query=data["query"], candidate_docs=data["rerank_docs"],
                bm25_top3=data["bm25_top3"])
            if task_id in saved:
                raise ValueError("duplicate rerank checkpoint")
            saved[task_id] = entry
    execution_env = {
        "RSI_ACTIVE_EXECUTION_PLAN_SHA256": str(plan["plan_sha256"]),
        "RSI_CALL_LEDGER_PATH": str(ledger_path),
        "RSI_CALL_LEDGER_MAX_LOGICAL": str(plan["logical_request_cap"]),
        "RSI_CALL_LEDGER_MAX_PHYSICAL": str(plan["physical_request_cap"]),
    }
    previous_env = {key: os.environ.get(key) for key in execution_env}
    os.environ.update(execution_env)
    try:
        load_project_dotenv()
        api_key = os.environ.get("SILICONFLOW_API_KEY", "")
        if not api_key:
            raise RuntimeError("configured reranker credential is unavailable")
        session = requests.Session()
        rows = []
        for task_id in ids:
            task = tasks[task_id]
            data = prepared[task_id]
            query, docs = data["query"], data["docs"]
            bm25, rerank_docs = data["bm25_top3"], data["rerank_docs"]
            if task_id in saved:
                row = saved[task_id]
            else:
                response = rerank(session, api_key, args.model_id, query,
                                  [d["text"][:MAX_DOCUMENT_CHARS]
                                   for d in rerank_docs])
                ranked = [rerank_docs[i]["doc_id"] for i in response["indices"]]
                row = {"probe_id": probe_id, "task_id": task_id,
                       "query": query, "bm25_top3": bm25,
                       "rerank_indices": response["indices"],
                       "rerank_top3": ranked,
                       "candidate_count": len(rerank_docs),
                       "relevance_scores": response["scores"],
                       "meta": response["meta"],
                       "latency_ms": response["latency_ms"]}
                with journal.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            # Gold/source matching stays local and happens only after reranking.
            source = set().union(*(shingles(item["text"]) for item in
                                   task["candidate_docs"] if item["is_gold"]))
            matched = {doc["doc_id"] for doc in docs
                       if source & shingles(doc["text"])}
            row = dict(row)
            row["source_bm25"] = bool(matched & set(row["bm25_top3"]))
            row["source_rerank"] = bool(matched & set(row["rerank_top3"]))
            rows.append(row)
        summary = {"n": len(rows),
                   "top3_changed": sum(r["bm25_top3"] != r["rerank_top3"]
                                       for r in rows),
                   "source_bm25_top3": sum(r["source_bm25"] for r in rows),
                   "source_rerank_top3": sum(r["source_rerank"] for r in rows),
                   "source_gained": sum(not r["source_bm25"] and r["source_rerank"]
                                        for r in rows),
                   "source_lost": sum(r["source_bm25"] and not r["source_rerank"]
                                      for r in rows)}
        report = {"status": "train_only_frozen_corpus_rerank_probe",
                  "protocol": PROTOCOL, "execution_plan_sha256": plan["plan_sha256"],
                  "execution_plan_path": str(plan_path),
                  "request_ledger_path": str(ledger_path),
                  "request_ledger_summary": summarize_ledger(ledger_path),
                  "snapshot_content_sha256": snapshot["content_sha256"],
                  "probe_id": probe_id, "model_id": args.model_id,
                  "prefilter_k": args.prefilter_k,
                  "summary": summary, "rows": rows,
                  "limit": "Exact source 8-gram is high precision and low recall; "
                           "labels and supplied passages were never sent to rerank API."}
        with output_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
