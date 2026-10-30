"""Freeze a hash-bound manual attribution review packet from completed runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evolution.attribution_review import (  # noqa: E402
    build_attribution_review_packet,
    apply_attribution_corrections,
)
from src.util import artifacts_path, read_json, read_jsonl, sha256_obj, write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", nargs="+", required=True,
                        help="dev_optimize run IDs whose failure rows are being reviewed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate", action="store_true",
                        help="validate a completed correction document instead of creating a template")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.validate:
        raise FileExistsError(f"refusing to overwrite existing file: {args.output}")
    rows = []
    for run_id in args.run_id:
        run = read_json(artifacts_path("runs", run_id, "summary.json"))
        if run.get("split") != "dev_optimize":
            raise ValueError("manual attribution review only accepts dev_optimize runs")
        for row in read_jsonl(artifacts_path("runs", run_id, "failures.jsonl")):
            row = dict(row)
            row["source_run_id"] = run_id
            rows.append(row)
    if args.validate:
        document = read_json(args.output)
        applied = apply_attribution_corrections(rows, document)
        print(json.dumps({
            "valid": True,
            "review_packet_id": document["review_packet_id"],
            "review_document_sha256": sha256_obj(document),
            "source_run_ids": args.run_id,
            "failure_count": len(applied),
            "applied_count": sum(bool(row.get("human_correction", {}).get("applied"))
                                  for row in applied),
            "is_causal_evidence": False,
        }, ensure_ascii=False))
        return 0
    packet = build_attribution_review_packet(rows, source_run_ids=args.run_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, packet)
    print(json.dumps({
        "output": str(args.output), "review_packet_id": packet["review_packet_id"],
        "source_failures_sha256": packet["source_failures_sha256"],
        "source_run_ids": packet["source_run_ids"],
        "failure_count": packet["failure_count"],
        "pending_count": packet["failure_count"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
