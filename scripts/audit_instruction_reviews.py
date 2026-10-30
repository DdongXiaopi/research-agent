"""Create a zero-call instruction audit and blinded development annotation packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.instruction_review_audit import (  # noqa: E402
    analyze_development_packet, blind_annotation_template,
)
from src.util import write_json  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--annotation-template")
    parser.add_argument("--annotations")
    parser.add_argument("--responses", help="Frozen raw response document; never calls a model")
    args = parser.parse_args(argv)
    packet_path = Path(args.packet).resolve()
    inputs = [packet_path] + [Path(p).resolve() for p in (args.annotations, args.responses) if p]
    outputs = [Path(args.report).resolve()]
    if args.annotation_template:
        outputs.append(Path(args.annotation_template).resolve())
    if len(outputs) != len(set(outputs)) or set(inputs) & set(outputs):
        raise ValueError("audit outputs must not overwrite input or one another")
    def frozen_document(p):
        raw = Path(p).read_bytes()
        return json.loads(raw.decode("utf-8-sig")), hashlib.sha256(raw).hexdigest()

    packet, digest = frozen_document(packet_path)
    frozen = [frozen_document(p) if p else (None, None)
              for p in (args.annotations, args.responses)]
    report = analyze_development_packet(
        packet, digest, annotations=frozen[0][0], responses=frozen[1][0])
    report["input_documents_sha256"] = {
        name: value for name, value in (
            ("packet", digest), ("annotations", frozen[0][1]),
            ("responses", frozen[1][1])) if value is not None
    }
    # Finish all validation before writing either output.
    template = (blind_annotation_template(packet, digest) if args.annotation_template else None)
    write_json(outputs[0], report)
    if template is not None:
        write_json(outputs[1], template)
    summary = report["historical_processed_reviews"]["provisional_contract_comparison"]
    print(json.dumps({"model_calls": 0, "cases": len(packet["cases"]),
                      "provisional_contract_disagreements": summary["false_approval"]
                      + summary["false_rejection"],
                      "independent_calibration_eligible": False,
                      "review_protocol_sha256": report["offline_plan"]["review_protocol_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
