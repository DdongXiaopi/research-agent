"""Score a frozen synthetic instruction-review response bundle offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.instruction_review_calibration import (  # noqa: E402
    analyze_calibration_responses,
)
from src.util import sha256_file, write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {name: value.resolve() for name, value in vars(args).items()}
    inputs = [paths[name] for name in ("packet", "annotations", "responses")]
    output = paths["report"]
    if len(inputs) != len(set(inputs)) or output in inputs:
        raise ValueError("score output must not overwrite an input document")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing score report: {output}")

    def frozen_document(path: Path):
        raw = path.read_bytes()
        return json.loads(raw.decode("utf-8-sig")), hashlib.sha256(raw).hexdigest()

    packet, packet_file_hash = frozen_document(paths["packet"])
    annotations, annotation_file_hash = frozen_document(paths["annotations"])
    responses, response_file_hash = frozen_document(paths["responses"])
    report = analyze_calibration_responses(
        packet, annotations, responses,
        responses_file_sha256=response_file_hash)
    report["input_documents_sha256"] = {
        "packet": packet_file_hash,
        "annotations": annotation_file_hash,
        "responses": response_file_hash,
    }
    report["scoring_cli_source_sha256"] = sha256_file(Path(__file__))
    write_json(output, report)
    print(json.dumps({
        "scorer_model_calls": 0,
        "response_case_count": report["response_case_count"],
        "completed_response_case_count": report["completed_response_case_count"],
        "runtime_error_response_case_count": report["runtime_error_response_case_count"],
        "physical_provider_attempt_count": None,
        "plan_sha256": report["plan_sha256"],
        "live_authorization_verified": False,
        "independent_calibration_eligible": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
