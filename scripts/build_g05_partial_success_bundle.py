"""Freeze a non-promotable bundle from independently valid G05 receipts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.g05_partial_success_bundle import build_g05_partial_success_bundle  # noqa: E402


def _write_new(path: Path, value: dict) -> None:
    path = path.resolve()
    try:
        path.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("partial-success evidence bundles must stay under .workbuddy") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--topup-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    parent = args.parent_run if args.parent_run.is_absolute() else ROOT / args.parent_run
    child = args.topup_run if args.topup_run.is_absolute() else ROOT / args.topup_run
    output = args.output if args.output.is_absolute() else ROOT / args.output
    bundle = build_g05_partial_success_bundle(parent, child, workspace=ROOT)
    _write_new(output, bundle)
    print(json.dumps({"status": bundle["status"], "bundle_sha256": bundle["bundle_sha256"],
        "success_count": bundle["success_count"], "distinct_doi_groups": bundle["distinct_doi_groups"],
        "excluded_count": len(bundle["excluded_receipts"]),
        "complete_120": bundle["complete_120"],
        "calibration_eligible": bundle["calibration_eligible"],
        "scorer_calls_authorized": bundle["scorer_calls_authorized"],
        "model_calls": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
