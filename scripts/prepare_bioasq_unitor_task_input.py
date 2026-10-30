"""Build a train-only BioASQ yes/no task-input dataset and its split manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.bioasq_unitor_task_input import write_dataset  # noqa: E402


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    manifest = write_dataset()
    split_summary = {
        name: {"n": value["n"], "ids_sha256": value["ids_sha256"]}
        for name, value in manifest["splits"].items()
    }
    print(json.dumps({key: manifest[key] for key in (
        "dataset", "source_sha256", "normalized_sha256", "n_questions",
        "confirmation_status")} | {"splits": split_summary},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
