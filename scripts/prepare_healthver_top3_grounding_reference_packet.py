"""Build a blinded HealthVer three-document grounding reference packet."""
from __future__ import annotations
import hashlib
import inspect
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_healthver_grounding_reference_packet import (
    _case_id, build_reference_packets,
)
from src.util import sha256_file, sha256_obj, write_json

OUTPUT = ROOT / ".workbuddy/healthver-grounding-top3-reference-v1"


def _source_hashes() -> dict[str, str]:
    from scripts import prepare_healthver_grounding_reference_packet as base
    return {
        "calibration_selection_source_sha256": base._source_hashes()["calibration_selection_source_sha256"],
        "task_input_evidence_source_sha256": base._source_hashes()["task_input_evidence_source_sha256"],
        "reference_packet_builder_source_sha256": hashlib.sha256(
            inspect.getsource(build_top3_reference_packets).encode("utf-8")).hexdigest(),
        "human_reference_scorer_source_sha256": sha256_file(
            ROOT / "scripts/score_task_input_grounding_human_reference.py"),
    }


def build_top3_reference_packets(per_label: int = 20, *, seed: int = 20261015):
    """Attach one paired abstract and two distinct-component distractors per claim."""
    base_bundle = build_reference_packets(per_label, seed=seed)
    base_packet = base_bundle["packets"]["reviewer_a"]
    key_by_input = {row["input_sha256"]: row for row in base_bundle["researcher_key"]["cases"]}
    primary_cases = base_packet["cases"]
    shared = []
    private_key = []
    for base_case in sorted(base_packet["cases"], key=lambda row: row["case_id"]):
        primary = key_by_input[base_case["input_sha256"]]
        ranked = sorted((row for row in primary_cases
                         if key_by_input[row["input_sha256"]]["component_id"] != primary["component_id"]),
                        key=lambda row: sha256_obj(
                            [seed, "top3-distractor", base_case["case_id"], row["case_id"]]))
        chosen = []
        for row in ranked:
            chosen.append(row["evidence"][0])
            if len(chosen) == 2:
                break
        if len(chosen) != 2:
            raise ValueError("not enough component-disjoint HealthVer distractors")
        evidence = [base_case["evidence"][0]] + [
            {"doc_id": str(doc["doc_id"]), "text": str(doc["text"])} for doc in chosen]
        evidence.sort(key=lambda doc: sha256_obj(
            [seed, "top3-document-order", base_case["case_id"], doc["doc_id"]]))
        claim = base_case["question"].split("Claim:", 1)[-1].strip()
        question = "Do the supplied scientific documents collectively establish this biomedical claim? Claim: " + claim
        input_hash = sha256_obj({"question": question, "evidence": evidence})
        shared.append({"task_id": primary["task_id"], "question": question,
                       "evidence": evidence, "input_sha256": input_hash})
        private_key.append({**primary, "input_sha256": input_hash,
                            "source_label_semantics": "primary-single-abstract-only; not top3 gold"})

    context = dict(base_packet["protocol_context"])
    context.update({
        "sampling": "one-primary-and-two-unique-component-disjoint-distractors-per-case",
        "rubric_version": "scientific-grounding-ai-reference-top3-v1",
        "case_input_sha256s": sorted(row["input_sha256"] for row in shared),
        "case_count": len(shared),
        "source_hashes": _source_hashes(),
    })
    cohort_hash = sha256_obj({
        "dataset": "healthver", "seed": seed, "documents_per_case": 3,
        "base_cohort_sha256": base_bundle["cohort_sha256"],
        "case_input_sha256s": context["case_input_sha256s"],
        "source_manifest_sha256": sha256_file(ROOT / "data/manifests/healthver_task_input.json"),
    })
    context["cohort_sha256"] = cohort_hash
    protocol_hash = sha256_obj(context)
    instructions = (
        "Independently assess whether the supplied set of scientific abstracts collectively establishes the claim. "
        "Use only the claim and supplied documents. Supported means at least one document directly establishes the "
        "claim and no document directly establishes its opposite. Contradicted means at least one document directly "
        "establishes the opposite for the same population, intervention/exposure, comparator, outcome and direction, "
        "and no document directly supports the claim. If decisive documents conflict, or neither side is established, "
        "use insufficient. For a decisive verdict, return sentence IDs and doc_id for one decisive document; sentence IDs "
        "are zero-based and local to each document, and local code reconstructs the exact quote. For insufficient or "
        "unable_to_judge, return empty sentence IDs and doc_id. Do not consult another reviewer or dataset labels. "
        "Give a reason of at most 30 words and confidence 1, 2, or 3. Return one JSON object with an annotations array; "
        "each item has case_id, input_sha256, verdict, supporting_sentence_ids, quote_doc_id, reason, confidence_1_to_3.")
    packets, responses = {}, {}
    for slot, shuffle_seed in (("reviewer_a", 0), ("reviewer_b", 1)):
        visible = list(shared)
        random.Random(int(protocol_hash[:16], 16) + shuffle_seed).shuffle(visible)
        reviewer_cases = [{"case_id": _case_id(protocol_hash, row["task_id"]),
                           "question": row["question"], "evidence": row["evidence"],
                           "input_sha256": row["input_sha256"]} for row in visible]
        unsigned = {
            "schema": "task-input-grounding-human-reference-v2",
            "protocol_sha256": protocol_hash, "protocol_context": context,
            "reviewer_slot": slot,
            "scope": {"dataset": "healthver", "split": "dev_optimize",
                      "case_count": len(reviewer_cases),
                      "use": "AI-only noncommercial top-3 aggregation calibration",
                      "limitations": ["primary benchmark labels are not top3 gold",
                                      "AI agreement is not a human-gold guarantee",
                                      "distractor abstracts are not independent confirmation",
                                      "task records have prior non-grounding project exposure"]},
            "instructions": instructions, "cases": reviewer_cases,
        }
        packet = {**unsigned, "packet_sha256": sha256_obj(unsigned)}
        packets[slot] = packet
        responses[slot] = {
            "schema": "task-input-grounding-ai-review-v1", "reviewer_type": "ai",
            "reviewer_id": "", "independent_human": False,
            "eligible_as_human_reference": False,
            "protocol_sha256": protocol_hash, "packet_sha256": packet["packet_sha256"],
            "rubric_file_sha256": sha256_obj(instructions),
            "provenance": {"review_run_id": "", "model_id": None,
                           "rubric_sha256": sha256_obj(instructions), "blind": True,
                           "other_review_outputs_consulted": False},
            "annotations": [{"case_id": row["case_id"], "input_sha256": row["input_sha256"],
                             "verdict": "", "supporting_quote": "", "quote_doc_id": "",
                             "reason": "", "confidence_1_to_3": None}
                            for row in reviewer_cases],
        }
    key = {**base_bundle["researcher_key"], "protocol_sha256": protocol_hash,
           "cohort_sha256": cohort_hash, "cases": sorted(private_key, key=lambda row: row["task_id"]),
           "source_label_semantics": "primary-single-abstract-only; not top3 gold"}
    return {"protocol_sha256": protocol_hash, "cohort_sha256": cohort_hash,
            "packets": packets, "responses": responses, "researcher_key": key}


def write_reference_packets(output_dir: Path = OUTPUT, per_label: int = 20, *, seed: int = 20261015):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing top-3 packet artifacts: {output_dir}")
    bundle = build_top3_reference_packets(per_label, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for slot, suffix in (("reviewer_a", "a"), ("reviewer_b", "b")):
        packet_path = output_dir / f"reviewer_{suffix}_packet.json"
        write_json(packet_path, bundle["packets"][slot])
        paths["packet_" + suffix] = str(packet_path)
    key_path = output_dir / "researcher_key_do_not_share.json"
    write_json(key_path, bundle["researcher_key"])
    paths["researcher_key"] = str(key_path)
    return {"paths": paths, "protocol_sha256": bundle["protocol_sha256"],
            "cohort_sha256": bundle["cohort_sha256"],
            "case_count": len(bundle["packets"]["reviewer_a"]["cases"])}


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--per-label", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    print(json.dumps(write_reference_packets(args.output_dir, args.per_label, seed=args.seed), indent=2))
