#!/usr/bin/env python3
"""Preflight the v16 builder, rubric-v7 references and target-v10 schema chain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import instruction_reviewer_cohort_v16 as cohort  # noqa: E402
from src.evolution.instruction_review_v10 import prepare_prompt_grounding_v10  # noqa: E402
from src.evolution.task_contracts import CLINIFACT_CONTRACT  # noqa: E402
from src.util import sha256_obj  # noqa: E402


def run_preflight() -> dict:
    packet = cohort.build_packet(ROOT)
    cohort.validate_packet_integrity(packet, ROOT)
    manifest = packet["reference_packet_manifest"]
    if len(manifest) != packet["case_count"]:
        raise ValueError("reference manifest does not cover each case")
    by_id = {row["case_id"]: row for row in manifest}
    checked = 0
    for case in packet["cases"]:
        row = by_id.get(case["case_id"])
        if row is None:
            raise ValueError("case is absent from reference/target manifest")
        reference = cohort.reference_payload(case)["reference_spans"]
        target = prepare_prompt_grounding_v10(case["candidate_text"], CLINIFACT_CONTRACT)
        if (row["reference_format_sha256"] != reference["response_format_sha256"]
                or row["reference_format_sha256"] != sha256_obj(reference["response_format"])
                or row["target_dynamic_schema_sha256"] != sha256_obj(target["response_format"])
                or target["target_protocol_version"] != packet["target_protocol"]["version"]):
            raise ValueError("reference packet → target dynamic schema binding mismatch")
        checked += 1
    audit = packet["exposure_audit"]
    expected_receipt = "code:instruction_reviewer_cohort_v15._build_cases"
    if (audit.get("overlap_count") != 0 or audit.get("overlap_verified") is not True
            or audit.get("prior_packet_count", 0) < 15
            or not any(row.get("source") == expected_receipt
                       for row in audit.get("prior_packet_receipts", []))):
        raise ValueError("v1-v15 exposure audit is incomplete or reports overlap")
    return {
        "status": "passed",
        "packet_sha256": packet["packet_sha256"],
        "case_count": packet["case_count"],
        "reference_manifest_rows": len(manifest),
        "reference_dynamic_schemas_checked": checked,
        "reference_rubric": packet["reference_protocol"]["rubric_version"],
        "target_protocol": packet["target_protocol"]["version"],
        "historical_exposure_receipts": audit["prior_packet_count"],
        "exposed_candidate_text_count": audit["exposed_candidate_text_count"],
        "overlap_count": audit["overlap_count"],
        "model_calls": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        help="optional path for a preflight record; use a new file")
    args = parser.parse_args(argv)
    result = run_preflight()
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
