#!/usr/bin/env python3
"""Recover provenance-bearing model proposals omitted from completed trials.

This creates unannotated, per-case blinded inputs. Old gate verdicts are not
reference labels. The old natural census and its frozen hashes stay unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.natural_instruction_calibration import CONTRACTS, _domain_for_path
from src.evolution.task_contracts import compose_task_instruction, task_instruction_strategy
from src.util import sha256_file, sha256_obj


def build(reports: list[Path], prior_reference: dict) -> dict:
    excluded = {row["candidate_sha256"] for row in prior_reference["cases"]}
    collected = {}
    source_receipts = []
    for path in sorted(reports, key=lambda p: str(p)):
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        domain = _domain_for_path(path)
        if domain is None:
            raise ValueError(f"report requires an explicitly supported domain path: {path}")
        source_receipts.append({"path": str(path.resolve()), "file_sha256": sha256_file(path)})
        contract = CONTRACTS[domain]
        for index, gate in enumerate(report.get("candidate_instruction_gates", [])):
            # The first gate can be the handwritten incumbent. Only proposals
            # carrying persisted optimizer-instruction provenance are eligible.
            proposal_hash = gate.get("proposal_instruction_sha256")
            if not isinstance(proposal_hash, str) or len(proposal_hash) != 64:
                continue
            raw = gate.get("candidate_text")
            if not isinstance(raw, str) or not raw.strip():
                continue
            strategy = task_instruction_strategy(raw, contract)
            text = compose_task_instruction(strategy, contract)
            identity = sha256_obj(text)
            if identity in excluded:
                continue
            origin = {"report": str(path.resolve()), "file_sha256": sha256_file(path),
                      "record_index": index, "proposal_instruction_sha256": proposal_hash,
                      "prior_reviewer_model_id": gate.get("grounding_review", {}).get("model_id")}
            row = collected.setdefault(identity, {
                "case_id": "proposal-" + identity[:16], "candidate_sha256": identity,
                "domain": domain,
                "blind_payload": {"task_contract": {"id": contract.contract_id,
                                                     "text": contract.text,
                                                     "sha256": contract.sha256},
                                  "candidate_strategy": strategy},
                "origins": [],
            })
            if origin not in row["origins"]:
                row["origins"].append(origin)
    body = {
        "schema": "g06-model-proposal-extension-packet-v1",
        "annotation_origin": None, "expected_labels_present": False,
        "old_gate_verdicts_imported": False,
        "prior_reference_sha256": prior_reference["reference_sha256"],
        "cases": sorted(collected.values(), key=lambda row: row["case_id"]),
        "case_count": len(collected), "source_receipts": source_receipts,
        "production_gate_eligible": False, "target_evaluation_authorized": False,
        "model_calls": 0, "network_calls": 0,
        "exposure_boundary": "absent from supplied frozen reference; historically seen by optimizer and prior gate",
        "reference_requirement": "independent AI double review and third-context arbitration; preserve uncertain",
    }
    return {**body, "packet_sha256": sha256_obj(body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--prior-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads(args.prior_reference.read_text(encoding="utf-8-sig"))
    if prior.get("reference_sha256") != sha256_obj({k: v for k, v in prior.items()
                                                  if k != "reference_sha256"}):
        raise ValueError("prior reference hash mismatch")
    result = build(args.report, prior)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != text:
            raise ValueError("refusing to overwrite packet")
    else:
        args.output.write_text(text, encoding="utf-8")
    print(json.dumps({"packet": str(args.output), "case_count": result["case_count"],
                      "packet_sha256": result["packet_sha256"], "model_calls": 0}))


if __name__ == "__main__":
    main()
