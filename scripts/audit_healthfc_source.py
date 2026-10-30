from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.healthfc_source_audit import audit_healthfc_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit HealthFC source metadata without emitting claim or evidence text.")
    parser.add_argument("source", type=Path, help="Path to the HealthFC CSV release")
    parser.add_argument("--output", type=Path, help="Optional JSON report output path")
    args = parser.parse_args()
    report = audit_healthfc_csv(args.source)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
