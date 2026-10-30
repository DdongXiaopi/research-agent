#!/usr/bin/env python3
"""Prepare a history-excluded SciFact MIPRO cohort (zero model calls)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.scifact_cohort import (  # noqa: E402
    SCOPE_TARGETS, build_nested_cohort,
)
from src.data.scifact_task_input import (  # noqa: E402
    SOURCE_SHA256, load_tasks,
)
from src.util import sha256_file, sha256_obj, write_json  # noqa: E402


DEFAULT_OUTPUT = ROOT / "artifacts/probe/scifact_mipro_exploratory_cohort_20261008.json"
DEFAULT_HISTORY_ROOT = ROOT / "artifacts/probe/scifact"


def scan_historical_claim_ids(root: Path) -> tuple[set[int], int]:
    """Collect claim IDs explicitly recorded by prior SciFact probe outputs."""
    claim_ids: set[int] = set()
    scanned = 0

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"claim_id", "claimId"}:
                    try:
                        claim_ids.add(int(item))
                    except (TypeError, ValueError):
                        pass
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        if path.name == "source.json":
            continue
        try:
            content = path.read_text(encoding="utf-8")
            if path.suffix == ".json":
                visit(json.loads(content))
            else:
                for line in content.splitlines():
                    if line.strip():
                        visit(json.loads(line))
        except (OSError, UnicodeError, json.JSONDecodeError):
            # A malformed historical output cannot contribute trustworthy IDs.
            # Fail closed rather than silently skipping it.
            raise ValueError(f"cannot audit historical SciFact artifact: {path}")
        scanned += 1
    return claim_ids, scanned


def prepare_cohort(
    *,
    history_root: Path = DEFAULT_HISTORY_ROOT,
    salt: str = "scifact-mipro-exploration-20261008-v1",
    exclude_cohort_paths: list[Path] | None = None,
    scope_targets_per_class: dict[str, int] | None = None,
) -> dict[str, Any]:
    rows = load_tasks(["dev_optimize"])
    history_ids, file_count = scan_historical_claim_ids(history_root)
    by_claim_id = {
        int(row["metadata"]["claim_id"]): row for row in rows
    }
    mapped_ids = {claim_id for claim_id in history_ids if claim_id in by_claim_id}
    excluded_task_ids = {str(by_claim_id[claim_id]["task_id"]) for claim_id in mapped_ids}
    excluded_component_ids = {
        str(by_claim_id[claim_id]["group_id"]) for claim_id in mapped_ids
    }
    previous_cohort_hashes = []
    previous_cohort_components: set[str] = set()
    for cohort_path in exclude_cohort_paths or []:
        try:
            prior = json.loads(Path(cohort_path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read prior SciFact cohort for exclusion: {cohort_path}") from exc
        prior_hash = str(prior.get("cohort_sha256") or "")
        unhashed_prior = dict(prior)
        unhashed_prior.pop("cohort_sha256", None)
        if (prior.get("dataset") != "scifact_claim_verification"
                or not prior_hash or sha256_obj(unhashed_prior) != prior_hash):
            raise ValueError(f"invalid or mismatched prior SciFact cohort: {cohort_path}")
        previous_cohort_hashes.append(prior_hash)
        for scope in (prior.get("scopes") or {}).values():
            previous_cohort_components.update(
                str(component) for component in (scope.get("component_ids") or []))
    excluded_component_ids.update(previous_cohort_components)
    targets = (SCOPE_TARGETS if scope_targets_per_class is None
               else scope_targets_per_class)
    required_per_label = sum(targets.values())
    remaining_components_by_label = {
        label: len({str(row["group_id"]) for row in rows
                    if row["gold_label"] == label
                    and str(row["group_id"]) not in excluded_component_ids
                    and str(row["task_id"]) not in excluded_task_ids})
        for label in ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")
    }
    insufficient = {label: count for label, count in remaining_components_by_label.items()
                    if count < required_per_label}
    if insufficient:
        detail = ", ".join(
            f"{label} {count} < {required_per_label}"
            for label, count in insufficient.items())
        raise ValueError(
            "SciFact remaining independent components cannot satisfy requested "
            f"per-class scope totals ({detail})")
    cohort = None
    attempts = 0
    for attempt in range(1000):
        current_salt = salt if attempt == 0 else f"{salt}:{attempt}"
        try:
            cohort = build_nested_cohort(
                rows,
                salt=current_salt,
                excluded_component_ids=excluded_component_ids,
                excluded_task_ids=excluded_task_ids,
                scope_targets_per_class=scope_targets_per_class,
            )
            attempts = attempt + 1
            break
        except ValueError as exc:
            if "too few independent components" not in str(exc):
                raise
    if cohort is None:
        raise ValueError("could not stratify SciFact components after 1000 deterministic salts")
    try:
        history_root_label = history_root.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        history_root_label = str(history_root.resolve())
    audit_manifest_path = ROOT / "artifacts/probe/scifact_self_evolution_source_audit_20261008.json"
    cohort.update({
        "salt_attempts": attempts,
        "historical_claim_id_count": len(history_ids),
        "historical_claim_ids_sha256": sha256_obj(sorted(history_ids)),
        "historical_claim_ids_mapped_count": len(mapped_ids),
        "historical_artifact_file_count": file_count,
        "historical_artifact_root": history_root_label,
        "source_manifest_sha256": sha256_file(audit_manifest_path),
        "historical_exclusion_scope": (
            "claim IDs explicitly present in prior SciFact JSON/JSONL probe outputs; "
            "all matching connected cited-document components excluded from all scopes"),
        "prior_cohort_exclusion_sha256s": sorted(previous_cohort_hashes),
        "prior_cohort_excluded_component_count": len(previous_cohort_components),
        "source_archive_sha256": SOURCE_SHA256,
    })
    cohort.pop("cohort_sha256", None)
    cohort["cohort_sha256"] = sha256_obj(cohort)
    return cohort


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salt", default="scifact-mipro-exploration-20261008-v1")
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY_ROOT)
    parser.add_argument("--exclude-cohort", action="append", type=Path, default=[],
                        help="exclude all connected components in a previously used cohort")
    parser.add_argument("--train-per-label", type=int, default=10)
    parser.add_argument("--search-per-label", type=int, default=6)
    parser.add_argument("--internal-per-label", type=int, default=6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    default_targets = (args.train_per_label, args.search_per_label,
                       args.internal_per_label) == (10, 6, 6)
    targets = None if default_targets else {
        "mipro_train": args.train_per_label,
        "search_validation": args.search_per_label,
        "internal_candidate": args.internal_per_label,
    }
    cohort = prepare_cohort(history_root=args.history_root, salt=args.salt,
                            exclude_cohort_paths=args.exclude_cohort,
                            scope_targets_per_class=targets)
    if args.write:
        write_json(args.output, cohort)
    print(json.dumps({
        "status": "written" if args.write else "validated_not_written",
        "path": str(args.output),
        "cohort_sha256": cohort["cohort_sha256"],
        "history_files_scanned": cohort["historical_artifact_file_count"],
        "historical_claim_ids": cohort["historical_claim_id_count"],
        "historical_claim_ids_mapped": cohort["historical_claim_ids_mapped_count"],
        "excluded_components": cohort["excluded_component_count"],
        "salt_attempts": cohort["salt_attempts"],
        "scopes": {name: {"n": entry["task_count"],
                          "labels": entry["class_counts"],
                          "components": entry["component_count"]}
                   for name, entry in cohort["scopes"].items()},
        "eligible_for_confirmation": cohort["eligible_for_confirmation"],
        "official_test_read": cohort["official_test_read"],
        "model_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
