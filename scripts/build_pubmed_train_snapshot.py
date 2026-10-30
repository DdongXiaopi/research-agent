"""Create a frozen, label-free PubMed article pool for BioASQ train only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.bioasq_unitor import load_train_question_manifest
from src.retrieval.pubmed_snapshot import PubMedClient, build_train_snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--task-id", action="append",
                        help="Use specified BioASQ train task IDs for a diagnostic snapshot")
    parser.add_argument("--per-query", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit <= 0 or args.skip < 0 or not 1 <= args.per_query <= 100:
        parser.error("limit must be positive, skip nonnegative, per-query 1..100")
    if args.task_id and args.skip:
        parser.error("skip cannot be combined with task-id")
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    tasks = (load_train_question_manifest(task_ids=set(args.task_id))
             if args.task_id else load_train_question_manifest(
                 limit=args.skip + args.limit)[args.skip:])
    snapshot = build_train_snapshot(tasks, PubMedClient(), args.per_query)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({"path": str(destination),
                      "protocol": snapshot["protocol"],
                      "n_tasks": len(snapshot["tasks"]),
                      "n_with_docs": sum(bool(t["docs"]) for t in snapshot["tasks"]),
                      "n_docs": sum(len(t["docs"]) for t in snapshot["tasks"]),
                      "content_sha256": snapshot["content_sha256"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
