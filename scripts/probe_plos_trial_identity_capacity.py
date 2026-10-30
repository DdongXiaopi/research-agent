"""Count explicit trial/statistical markers in the pending PLOS source set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.plos_capacity_probe import (  # noqa: E402
    CI_PATTERN, NCT_PATTERN, PRIMARY_PATTERN, P_VALUE_PATTERN, summarize_plos_capacity,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--crossref-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source_rows = json.loads(args.source_index.read_text(encoding="utf-8"))
    manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    updates = json.loads(args.crossref_audit.read_text(encoding="utf-8"))
    eligible_dois = {
        row["doi"].casefold() for row in manifest["candidate_entries"]
        if row["disposition"] != "exclude"
    }
    # Exclude items with known Crossref updates from this capacity estimate until
    # the correction/retraction/new-version record is resolved.
    eligible_dois -= {
        row["doi"].casefold() for row in updates["candidate_entries"]
        if row["update_disposition"] != "no_crossref_update_found_not_status_proof"
    }
    result = summarize_plos_capacity(source_rows, eligible_dois)
    report = {
        "schema": "plos-trial-identity-capacity-probe-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_index_sha256": _sha(args.source_index),
        "candidate_manifest_sha256": _sha(args.candidate_manifest),
        "crossref_audit_sha256": _sha(args.crossref_audit),
        "eligible_doi_count_after_known_updates": len(eligible_dois),
        "regex_contracts": {
            "nct": NCT_PATTERN.pattern,
            "p_value": P_VALUE_PATTERN.pattern,
            "confidence_interval": CI_PATTERN.pattern,
            "primary_outcome_wording": PRIMARY_PATTERN.pattern,
        },
        **result,
        "cohort_or_label_created": False,
    }
    output = args.output.resolve()
    try:
        output.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("capacity report must stay under project .workbuddy") from exc
    if output.exists():
        raise FileExistsError(f"refusing to overwrite capacity report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "status": "written",
        "output": str(output),
        "eligible_doi_count_after_known_updates": len(eligible_dois),
        "indicator_counts": result["indicator_counts"],
        "unique_nct_count": result["unique_nct_count"],
        "labels_generated": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
