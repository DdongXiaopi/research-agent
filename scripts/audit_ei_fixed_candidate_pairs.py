"""Create a reproducible item-level diff for an EI fixed-candidate probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.paired_candidate_audit import audit_paired_rows


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="directory containing baseline.jsonl and candidate.jsonl")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    summary, changes = audit_paired_rows(
        read_jsonl(run_dir / "baseline.jsonl"), read_jsonl(run_dir / "candidate.jsonl"))
    summary_path = run_dir / "item-audit-summary.json"
    changes_path = run_dir / "item-audit-answer-changes.jsonl"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with changes_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in changes:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({**summary, "summary_path": str(summary_path),
                      "answer_changes_path": str(changes_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
