"""Audit recorded task workflow offline, without model calls or hidden reasoning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.runtime.trace import audit_task_execution, read_trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit_task_execution(read_trace(args.trace), args.task_id)
    # An existing receipt may be historical evidence: never replace it.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"report_sha256": report["report_sha256"], "status": report["status"]}))


if __name__ == "__main__":
    main()
