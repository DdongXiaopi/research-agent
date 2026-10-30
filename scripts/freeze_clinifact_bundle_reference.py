#!/usr/bin/env python3
"""Freeze blind AI reference packets for a specific G03 candidate bundle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.clinifact_ai_reference import build_packets  # noqa: E402
from src.util import read_json, sha256_file, sha256_obj, write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-cohort", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026101801)
    parser.add_argument("--max-cases", type=int, default=None,
                        help="optionally retain a deterministic hidden-label-balanced subset")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"reference directory is not empty: {out}")
    bundle = read_json(args.bundle.resolve())
    bundle_sha = bundle.get("candidate_bundle_sha256")
    if not isinstance(bundle_sha, str) or len(bundle_sha) != 64:
        raise ValueError("candidate bundle hash is missing")
    source = read_json(args.source_cohort.resolve())
    binding = {
        "candidate_bundle_id": bundle["candidate_bundle_id"],
        "candidate_bundle_sha256": bundle_sha,
        "parent_release_id": bundle["parent_release_id"],
        "source_cohort_sha256": source["cohort_sha256"],
        "created_after_candidate_preselection": True,
        "source_candidate_members": bundle["members"],
    }
    packets = build_packets(
        ROOT / "data/raw/clinifact/train_set.csv",
        ROOT / "data/raw/clinifact/validation_set.csv",
        read_json(args.audit.resolve()), read_json(args.history.resolve()), source,
        source_report_sha256=bundle_sha, seed=args.seed,
        candidate_binding=binding, expected_eligible_components=None,
    )
    if args.max_cases is not None:
        if args.max_cases < 90 or args.max_cases % 3:
            raise ValueError("--max-cases must be a multiple of three and at least 90")
        private = packets["private_key"]["cases"]
        per_class = args.max_cases // 3
        selected = []
        pools = {}
        for label in ("EVIDENCE", "INCONCLUSIVE", "NEI"):
            rows = [row for row in private if row.get("source_label") == label]
            rows.sort(key=lambda row: sha256_obj([args.seed, "bundle-reference-subset", label, row["case_id"]]))
            pools[label] = rows
            selected.extend(rows[:min(per_class, len(rows))])
        # If the audited source has fewer than the nominal per-class count,
        # retain every row in that source class and fill the remaining slots
        # from the other hidden pools. Reviewers still receive no source label;
        # the resulting AI consensus remains the sole acceptance reference.
        remaining = args.max_cases - len(selected)
        if remaining > 0:
            used = {row["case_id"] for row in selected}
            extras = [row for label in ("EVIDENCE", "NEI", "INCONCLUSIVE")
                      for row in pools[label] if row["case_id"] not in used]
            extras.sort(key=lambda row: sha256_obj([args.seed, "bundle-reference-extra", row["case_id"]]))
            selected.extend(extras[:remaining])
        if len(selected) != args.max_cases:
            raise ValueError("hidden source-label stratification lacks requested capacity")
        ids = {row["case_id"] for row in selected}
        packets["private_key"]["cases"] = selected
        packets["private_key"]["subset_policy"] = "deterministic hidden source-label capacity stratification; labels never shown to reviewers"
        protocol = sha256_obj({"parent_protocol": packets["protocol_sha256"],
                               "selected_case_ids": sorted(ids),
                               "subset_policy": packets["private_key"]["subset_policy"]})
        packets["protocol_sha256"] = protocol
        for slot in ("reviewer_a", "reviewer_b"):
            packet = packets["packets"][slot]
            packet_cases = [row for row in packet["cases"] if row["case_id"] in ids]
            unsigned = {"schema": packet["schema"], "protocol_sha256": protocol,
                        "reviewer_slot": packet["reviewer_slot"],
                        "instructions": packet["instructions"], "cases": packet_cases}
            packets["packets"][slot] = {**unsigned, "packet_sha256": sha256_obj(unsigned)}
        packets["response_templates"] = {
            slot: {"schema": "clinifact-blind-ai-review-v1", "protocol_sha256": protocol,
                   "packet_sha256": packets["packets"][slot]["packet_sha256"],
                   "reviewer_slot": slot, "model_id": None, "run_id": None, "annotations": []}
            for slot in ("reviewer_a", "reviewer_b")}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "reference_packets.json", packets)
    for slot, suffix in (("reviewer_a", "a"), ("reviewer_b", "b")):
        write_json(out / f"reviewer_{suffix}_packet.json", packets["packets"][slot])
    write_json(out / "private_source_key.json", packets["private_key"])
    print(json.dumps({"status": "bundle_reference_packets_frozen",
                      "candidate_bundle_id": bundle["candidate_bundle_id"],
                      "candidate_bundle_sha256": bundle_sha,
                      "protocol_sha256": packets["protocol_sha256"],
                      "case_count": len(packets["packets"]["reviewer_a"]["cases"]),
                      "out_dir": str(out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
