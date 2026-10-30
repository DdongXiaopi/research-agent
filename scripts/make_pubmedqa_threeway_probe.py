"""Prepare an exploratory native yes/no/maybe PubMedQA development view.

This uses PQA-L's labeled training parquet, limits yes/no rows to the legacy
development IDs, adds only the legacy adapter's previously dropped maybe rows,
and creates no selection gate or final holdout. It cannot support confirmation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.pubmedqa_threeway_probe import (  # noqa: E402
    DATASET, LEGACY_MANIFEST, SOURCE_PARQUET, write_dataset,
)
from src.util import read_json  # noqa: E402


def iter_eligible_source_rows():
    """Scan only eligible IDs and columns, with the holdout filter at Parquet scan time."""
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover - depends on data-prep runtime
        raise SystemExit(
            "Preparing the parquet view requires pyarrow; use the configured data-prep Python.") from exc

    legacy = read_json(LEGACY_MANIFEST)
    dev_ids = legacy.get("splits", {}).get("dev_optimize", {}).get("ids", [])
    numeric_dev_ids = []
    for value in dev_ids:
        try:
            numeric_dev_ids.append(int(str(value).strip()))
        except (TypeError, ValueError):
            continue
    if not numeric_dev_ids:
        raise ValueError("legacy PubMedQA dev manifest has no numeric IDs")

    dataset = ds.dataset(SOURCE_PARQUET, format="parquet")
    decision = ds.field("final_decision")
    pubid = ds.field("pubid")
    eligible = ((decision == "maybe") |
                (decision.isin(["yes", "no"]) & pubid.isin(numeric_dev_ids)))
    scanner = dataset.scanner(
        columns=["pubid", "final_decision", "question", "context"],
        filter=eligible,
        batch_size=256,
    )
    for batch in scanner.to_batches():
        yield from batch.to_pylist()


def main() -> int:
    if not SOURCE_PARQUET.is_file():
        raise FileNotFoundError(f"missing source training parquet: {SOURCE_PARQUET}")
    manifest = write_dataset(iter_eligible_source_rows())
    print({"dataset": DATASET,
           "n_questions": manifest["n_questions"],
           "label_counts": manifest["label_counts"],
           "eligible_for_confirmation": manifest["eligible_for_confirmation"],
           "normalized_file": manifest["normalized_file"],
           "manifest_file": "data/manifests/" + DATASET + ".json"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
