#!/usr/bin/env python3
"""Re-run only abstaining G09 adjudications with a separately configured model."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_key(env_name: str) -> str:
    value = os.environ.get(env_name, "")
    if value:
        return value
    dot_env = ROOT / ".env"
    if dot_env.is_file():
        for line in dot_env.read_text(encoding="utf-8").splitlines():
            if line.startswith(env_name + "="):
                return line.split("=", 1)[1].strip().strip('"\'')
    raise RuntimeError(f"missing {env_name}")


def _sha(value):
    from src.util import sha256_obj
    return sha256_obj(value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime-dir", type=Path, required=True)
    ap.add_argument("--evaluation-id", required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-id", default="deepseek-chat")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--base-url", default="https://api.deepseek.com/v1")
    args = ap.parse_args()
    os.environ["RSI_HOME"] = str(args.runtime_dir.resolve())
    from src.eval import ai_annotation, open_benchmark
    packet = open_benchmark.annotation_packet(args.evaluation_id)
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    packet_by_id = {row["case_id"]: row for row in packet["cases"]}
    out = copy.deepcopy(annotations)
    key = _load_key(args.api_key_env)
    repaired = 0
    for case_row in out["cases"]:
        nulls = [row for row in case_row.get("adjudications", []) if row.get("supported") is None]
        if not nulls:
            continue
        case = packet_by_id[case_row["case_id"]]
        rows = case_row["annotations"]
        by_item = {}
        for row in rows:
            by_item.setdefault((row["kind"], row["item_id"]), []).append(row)
        disagreements = {item for item, values in by_item.items()
                         if len(values) == 2 and values[0]["supported"] != values[1]["supported"]}
        payload = {"request": case["request"], "rubric": case["task_rubric"],
                   "answer": case["response"],
                   "expected_items": [{"kind": k, "item_id": i} for k, i in sorted(disagreements)],
                   "disagreements": [{"kind": k, "item_id": i, "first": by_item[(k, i)][0],
                                      "second": by_item[(k, i)][1]} for k, i in sorted(disagreements)]}
        user = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        system = ai_annotation.ADJUDICATE_PROMPT
        response = requests.post(args.base_url.rstrip("/") + "/chat/completions", headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json"}, json={
                "model": args.model_id,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0, "max_tokens": 2048,
                "response_format": {"type": "json_object"}}, timeout=180)
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        parsed = ai_annotation._parse_items(content, case, disagreements, allow_null=True, conservative=False)
        run_id = "provider-adjudicator-" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]
        provenance = {"blind": True, "model_id": args.model_id,
                      "other_review_outputs_consulted": True, "review_run_id": run_id,
                      "rubric_sha256": case["rubric_sha256"], "test_only": False,
                      "provider": args.base_url}
        case_row["adjudications"] = [{**row, "reviewer": "ai-adjudicator-" + run_id,
            "source": "independent_ai", "reviewer_provenance": provenance,
            "original_annotations_sha256": _sha(rows), "response_sha256": case["response_sha256"],
            "rubric_sha256": case["rubric_sha256"]} for row in parsed.values()]
        repaired += 1
    if repaired == 0:
        raise RuntimeError("no abstaining adjudications found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"repaired_cases": repaired, "annotations_sha256": _sha(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
