"""Build an offline high-recall candidate ledger from a closed G09 capture.

This command reads persisted response bytes only.  It performs no HTTP or
model calls and refuses to treat staged rows as accepted source material.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.open_research_recall import build_recall_ledger, validate_recall_ledger


def _load(path: Path) -> dict:
    return json.loads(path.read_bytes().decode("utf-8"))


def build(*, plan_path: Path, topic_manifest_path: Path, responses_dir: Path,
          output_path: Path, min_matched_dimensions: int = 2) -> dict:
    plan = _load(plan_path)
    if plan.get("schema") != "g09-v23-offline-recall-expanded-plan-v1":
        raise ValueError("only a bound v23 recall plan is supported")
    manifest = _load(topic_manifest_path)
    specs = {row["topic_id"]: row for row in manifest["topic_specs"]}
    schedule = plan["protocol"]["planned_schedule"]
    topic_records = {topic_id: [] for topic_id in specs}
    for slot in schedule:
        response = responses_dir / f"response-{int(slot['slot']):02d}.bin"
        if not response.is_file():
            continue
        page = _load(response)
        rows = (page.get("resultList") or {}).get("result") or []
        if not isinstance(rows, list):
            raise ValueError(f"response result list is malformed: {response}")
        topic_records.setdefault(slot["topic_id"], []).extend(row for row in rows if isinstance(row, dict))
    protocol = plan["protocol"]
    ledger = build_recall_ledger(topic_records, topic_specs=specs,
        known_dois=protocol["known_identity_exclusion"]["dois"],
        known_pmcids=protocol["known_identity_exclusion"]["pmcids"],
        min_matched_dimensions=min_matched_dimensions,
        parent_artifact_sha256=plan["plan_sha256"])
    validate_recall_ledger(ledger)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ledger, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"status": "written", "output": str(output_path),
        "candidate_count": ledger["candidate_count"], "network_calls": 0, "model_calls": 0,
        "ledger_sha256": ledger["ledger_sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--topic-manifest", type=Path, required=True)
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-matched-dimensions", type=int, default=2)
    args = parser.parse_args(argv)
    print(json.dumps(build(plan_path=args.plan, topic_manifest_path=args.topic_manifest,
        responses_dir=args.responses_dir, output_path=args.output,
        min_matched_dimensions=args.min_matched_dimensions), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

