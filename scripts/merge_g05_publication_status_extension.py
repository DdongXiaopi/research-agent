"""Merge immutable G05 publication-status components into a new provenance report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.g05_publication_status import merge_status_extension_reports  # noqa: E402


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-pool", "parent-plan", "parent-report", "parent-sample-plan",
                 "extension-plan", "extension-report", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {name.replace("-", "_"): getattr(args, name.replace("-", "_"))
             for name in ("source-pool", "parent-plan", "parent-report", "parent-sample-plan",
                          "extension-plan", "extension-report", "output-dir")}
    output_dir = paths["output_dir"].resolve()
    try:
        output_dir.relative_to((ROOT / ".workbuddy").resolve())
    except ValueError as exc:
        raise ValueError("composite G05 output must stay under project .workbuddy") from exc
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite composite status artifacts: {output_dir}")
    loaded = {key: json.loads(paths[key].read_text(encoding="utf-8")) for key in (
        "source_pool", "parent_plan", "parent_report", "parent_sample_plan",
        "extension_plan", "extension_report")}
    composite_plan, report = merge_status_extension_reports(
        loaded["source_pool"], loaded["parent_plan"], loaded["parent_report"],
        loaded["parent_sample_plan"], loaded["extension_plan"], loaded["extension_report"],
        source_pool_file_sha256=_sha(paths["source_pool"].read_bytes()),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    plan_bytes = (json.dumps(composite_plan, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    plan_path = output_dir / "composite-plan.json"
    plan_path.write_bytes(plan_bytes)
    report["status_plan_file_sha256"] = _sha(plan_bytes)
    report["report_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in report.items() if key != "report_sha256"},
        sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": "composite_status_report_written",
        "output_dir": str(output_dir),
        "doi_groups": report["planned_count"],
        "verified_current_as_of_capture": report["verified_current_count"],
        "unverified_excluded": report["unverified_excluded_count"],
        "composite_plan_sha256": composite_plan["plan_sha256"],
        "report_sha256": report["report_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
