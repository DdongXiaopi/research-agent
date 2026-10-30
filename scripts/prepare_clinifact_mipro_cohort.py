#!/usr/bin/env python3
"""Create a zero-call, history-screened CliniFact MIPRO exploratory cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.clinifact_cohort import (  # noqa: E402
    SCOPE_TARGETS, build_clinifact_cohort,
)
from src.util import sha256_file, write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path,
                        default=ROOT / "data/raw/clinifact/train_set.csv")
    parser.add_argument("--validation", type=Path,
                        default=ROOT / "data/raw/clinifact/validation_set.csv")
    parser.add_argument("--audit", type=Path,
                        default=ROOT / "artifacts/probe/clinifact_source_audit_20261008.json")
    parser.add_argument("--history", type=Path,
                        default=ROOT / "artifacts/probe/clinifact_history_scan_20261008.json")
    parser.add_argument("--salt", default="clinifact-mipro-exploration-20261008-v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/probe/clinifact_mipro_exploratory_cohort_20261008.json")
    parser.add_argument("--mipro-train-per-class", type=int,
                        default=SCOPE_TARGETS["mipro_train"])
    parser.add_argument("--search-validation-per-class", type=int,
                        default=SCOPE_TARGETS["search_validation"])
    parser.add_argument("--internal-candidate-per-class", type=int,
                        default=SCOPE_TARGETS["internal_candidate"])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    try:
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
        history = json.loads(args.history.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read CliniFact audit/history manifest: {exc}") from exc
    cohort = build_clinifact_cohort(
        args.train, args.validation, audit, history, salt=args.salt,
        scope_targets_per_class={
            "mipro_train": args.mipro_train_per_class,
            "search_validation": args.search_validation_per_class,
            "internal_candidate": args.internal_candidate_per_class,
        },
        audit_manifest_sha256=sha256_file(args.audit))
    if args.write:
        write_json(args.output, cohort)
    print(json.dumps({
        "status": "written" if args.write else "validated_not_written",
        "path": str(args.output),
        "cohort_sha256": cohort["cohort_sha256"],
        "history_rows_excluded": cohort["history_rows_excluded"],
        "history_components_excluded": cohort["history_components_excluded"],
        "source_component_count": cohort["source_component_count"],
        "salt_attempts": cohort["salt_attempts"],
        "scopes": {name: {"n": entry["task_count"],
                           "labels": entry["class_counts"],
                           "components": entry["component_count"]}
                   for name, entry in cohort["scopes"].items()},
        "official_test_downloaded_or_read": cohort["official_test_downloaded_or_read"],
        "eligible_for_confirmation": cohort["eligible_for_confirmation"],
        "model_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
