"""Create an offline-only, parent-bound G09 v9 field-scoped search plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.open_research_v9_field_plan import build_v9_field_search_plan  # noqa: E402


def _load(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-plan", required=True, type=Path)
    parser.add_argument("--v7-capture", required=True, type=Path)
    parser.add_argument("--v7-root-audit", required=True, type=Path)
    parser.add_argument("--topic-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-requests", type=int, default=40)
    args = parser.parse_args()

    plan, _ = _load(args.v7_plan)
    capture, capture_file_sha = _load(args.v7_capture)
    audit, audit_file_sha = _load(args.v7_root_audit)
    topic_manifest, topic_manifest_file_sha = _load(args.topic_manifest)
    result = build_v9_field_search_plan(v7_plan=plan, v7_capture=capture,
        v7_capture_file_sha256=capture_file_sha, v7_root_audit=audit,
        v7_root_audit_file_sha256=audit_file_sha, topic_manifest=topic_manifest,
        topic_manifest_file_sha256=topic_manifest_file_sha,
        max_requests=args.max_requests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
    print(json.dumps({"plan_sha256": result["plan_sha256"],
        "plan_file_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "request_budget": result["request_budget"],
        "selected_groups": result["selected_groups"],
        "network_calls": 0, "model_calls": 0, "cohort_created": False},
        sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
