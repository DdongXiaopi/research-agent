"""Freeze a blinded, component-balanced HealthVer grounding reference packet."""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data.healthver_task_input import LABELS, load_tasks
from src.util import sha256_file, sha256_obj, write_json

OUTPUT = ROOT / ".workbuddy/healthver-grounding-reference-v1"
SOURCE_LABELS = {
    "SUPPORT": "supported",
    "CONTRADICT": "contradicted",
    "NOT_ENOUGH_INFO": "insufficient",
}


def _source_hashes() -> dict[str, str]:
    from scripts import score_task_input_grounding_human_reference as scorer
    from src.data import healthver_task_input

    return {
        "calibration_selection_source_sha256": hashlib.sha256(
            inspect.getsource(select_cases).encode("utf-8")).hexdigest(),
        "task_input_evidence_source_sha256": sha256_file(
            Path(inspect.getsourcefile(healthver_task_input)).resolve()),
        "reference_packet_builder_source_sha256": hashlib.sha256(
            inspect.getsource(build_reference_packets).encode("utf-8")).hexdigest(),
        "human_reference_scorer_source_sha256": sha256_file(
            Path(inspect.getsourcefile(scorer)).resolve()),
    }


def select_cases(per_label: int = 20, *, seed: int = 20261015) -> list[dict[str, Any]]:
    """Select one case per connected component with balanced source labels."""
    if not 1 <= per_label <= 100 or type(seed) is not int:
        raise ValueError("per_label must be 1..100 and seed must be an integer")
    tasks = load_tasks(["dev_optimize"])
    by_component: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list))
    for task in tasks:
        source_label = str(task["options"][int(task["correct_index"])])
        if source_label not in SOURCE_LABELS:
            raise ValueError("HealthVer source label is outside the frozen mapping")
        by_component[str(task["group_id"])][source_label].append(task)
    components = sorted(by_component, key=lambda value: sha256_obj([seed, value]))
    labels = list(LABELS)
    # A deterministic bipartite matching assigns each selected component to one
    # label. This keeps every case in the cohort on a distinct claim/abstract
    # connected component, while preserving equal class counts.
    slots = [label for label in labels for _ in range(per_label)]
    slots.sort(key=lambda label: (sum(label in by_component[c] for c in components),
                                  sha256_obj([seed, "slot", label])))
    component_owner: dict[str, str] = {}

    def assign(label: str, visited: set[str]) -> bool:
        candidates = [c for c in components if label in by_component[c]]
        candidates.sort(key=lambda c: (len(by_component[c]), sha256_obj([seed, label, c])))
        for component in candidates:
            if component in visited:
                continue
            visited.add(component)
            prior = component_owner.get(component)
            if prior is None or assign(prior, visited):
                component_owner[component] = label
                return True
        return False

    for label in slots:
        if not assign(label, set()):
            raise ValueError("HealthVer components cannot provide the requested balanced cohort")

    chosen: list[dict[str, Any]] = []
    for component, source_label in sorted(component_owner.items()):
        eligible = by_component[component][source_label]
        eligible.sort(key=lambda row: sha256_obj([seed, row["task_id"]]))
        chosen.append(eligible[0])
    counts = Counter(str(row["options"][int(row["correct_index"])]) for row in chosen)
    if len(chosen) != per_label * len(LABELS) or any(
            counts[label] != per_label for label in LABELS):
        raise ValueError("HealthVer grounding cohort is not exactly class balanced")
    return sorted(chosen, key=lambda row: sha256_obj([seed, "case", row["task_id"]]))


def _case_id(protocol_sha256: str, task_id: str) -> str:
    return "HEALTHVER-GND-" + hashlib.sha256(
        f"{protocol_sha256}\0{task_id}".encode("utf-8")).hexdigest()[:14]


def build_reference_packets(per_label: int = 20, *, seed: int = 20261015):
    tasks = select_cases(per_label, seed=seed)
    cases = []
    key_rows = []
    for task in tasks:
        task_id = str(task["task_id"])
        evidence = [{"doc_id": task_id, "text": str(task["evidence"])}]
        question = (
            "Does the supplied scientific evidence establish this biomedical claim? "
            "Claim: " + str(task["question"]).strip())
        input_hash = sha256_obj({"question": question, "evidence": evidence})
        source_label = str(task["options"][int(task["correct_index"])])
        cases.append({"task_id": task_id, "input_sha256": input_hash,
                      "question": question, "evidence": evidence,
                      "source_label_for_audit_only": source_label})
        key_rows.append({"task_id": task_id, "source_label_for_post_review_only": source_label,
                         "input_sha256": input_hash, "component_id": task["group_id"]})
    source_hashes = _source_hashes()
    cohort_sha256 = sha256_obj({
        "dataset": "healthver", "seed": seed,
        "task_ids": sorted(row["task_id"] for row in tasks),
        "input_sha256s": sorted(row["input_sha256"] for row in cases),
        "source_manifest_sha256": sha256_file(ROOT / "data/manifests/healthver_task_input.json"),
    })
    protocol_context = {
        "schema": "task-input-grounding-reference-protocol-v3",
        "dataset": "healthver", "split": "dev_optimize",
        "sampling": "deterministic-label-balanced-one-case-per-connected-component",
        "per_label": per_label, "offset_per_label": 0,
        "judge_evidence_character_limit_per_document": 6000,
        "rubric_version": "scientific-grounding-ai-reference-v2",
        "case_input_sha256s": sorted(row["input_sha256"] for row in cases),
        "source_hashes": source_hashes, "case_count": len(cases),
        "labels": ["supported", "contradicted", "insufficient"],
        "cohort_sha256": cohort_sha256,
    }
    protocol_sha256 = sha256_obj(protocol_context)
    instructions = (
        "Independently assess whether the supplied abstract establishes the claim. "
        "Use only the claim and supplied text. Verdicts: supported (the evidence "
        "directly establishes the claim), contradicted (it directly establishes "
        "the opposite for the same population, intervention/exposure, comparator, "
        "outcome and direction), insufficient (neither side is established, a "
        "material qualifier is missing, or the evidence is conflicting), or "
        "unable_to_judge. Evidence sentence IDs are zero-based and local to the "
        "supplied abstract. For supported/contradicted, return one or more "
        "the inclusive sentence-ID range covering the supporting passage and the "
        "supplied doc_id; never copy sentence text. If evidence spans multiple "
        "sentences, include every ID from the first through the last. Local code "
        "reconstructs the exact quote from these IDs. "
        "For insufficient/unable_to_judge, use an empty sentence-ID list and "
        "empty doc_id. Do not consult another reviewer or any dataset labels. "
        "Give a reason of at most 30 words and confidence 1 (low), 2 "
        "(moderate), or 3 (high). Return one JSON object with an annotations "
        "array; each item has case_id, input_sha256, verdict, "
        "supporting_sentence_ids, quote_doc_id, reason, confidence_1_to_3."
    )
    packets, responses = {}, {}
    for slot, shuffle_seed in (("reviewer_a", 0), ("reviewer_b", 1)):
        visible = [{k: v for k, v in case.items()
                    if k != "source_label_for_audit_only"} for case in cases]
        random.Random(int(protocol_sha256[:16], 16) + shuffle_seed).shuffle(visible)
        reviewer_cases = [{"case_id": _case_id(protocol_sha256, case["task_id"]),
                           "question": case["question"], "evidence": case["evidence"],
                           "input_sha256": case["input_sha256"]}
                          for case in visible]
        unsigned = {
            "schema": "task-input-grounding-human-reference-v2",
            "protocol_sha256": protocol_sha256,
            "protocol_context": protocol_context,
            "reviewer_slot": slot,
            "scope": {"dataset": "healthver", "split": "dev_optimize",
                      "case_count": len(reviewer_cases),
                      "use": "AI-only semantic reference for exploratory noncommercial calibration",
                      "limitations": [
                          "public source data; not blind confirmation",
                          "source benchmark labels withheld from reviewers",
                          "AI agreement is not a human-gold guarantee",
                          "task records have prior non-grounding project exposure" ]},
            "instructions": instructions, "cases": reviewer_cases,
        }
        packet = {**unsigned, "packet_sha256": sha256_obj(unsigned)}
        packets[slot] = packet
        responses[slot] = {
            "schema": "task-input-grounding-ai-review-v1", "reviewer_type": "ai",
            "reviewer_id": "", "independent_human": False,
            "eligible_as_human_reference": False,
            "protocol_sha256": protocol_sha256,
            "packet_sha256": packet["packet_sha256"],
            "rubric_file_sha256": sha256_obj(instructions),
            "provenance": {"review_run_id": "", "model_id": None,
                           "rubric_sha256": sha256_obj(instructions), "blind": True,
                           "other_review_outputs_consulted": False},
            "annotations": [{"case_id": case["case_id"],
                             "input_sha256": case["input_sha256"],
                             "verdict": "", "supporting_quote": "",
                             "quote_doc_id": "", "reason": "",
                             "confidence_1_to_3": None}
                            for case in reviewer_cases],
        }
    key = {
        "schema": "healthver-grounding-reference-key-v1",
        "protocol_sha256": protocol_sha256, "cohort_sha256": cohort_sha256,
        "source_manifest_sha256": sha256_file(ROOT / "data/manifests/healthver_task_input.json"),
        "case_count": len(key_rows),
        "source_label_counts": dict(Counter(row["source_label_for_post_review_only"]
                                             for row in key_rows)),
        "cases": sorted(key_rows, key=lambda row: row["task_id"]),
        "license_scope": "user-authorized local noncommercial research; do not redistribute source data",
    }
    return {"protocol_sha256": protocol_sha256, "cohort_sha256": cohort_sha256,
            "packets": packets, "responses": responses, "researcher_key": key}


def write_reference_packets(output_dir: Path = OUTPUT, per_label: int = 20,
                            *, seed: int = 20261015):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing reference artifacts: {output_dir}")
    bundle = build_reference_packets(per_label, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for slot, suffix in (("reviewer_a", "a"), ("reviewer_b", "b")):
        packet_path = output_dir / f"reviewer_{suffix}_packet.json"
        response_path = output_dir / f"reviewer_{suffix}_responses.json"
        write_json(packet_path, bundle["packets"][slot])
        write_json(response_path, bundle["responses"][slot])
        paths["packet_" + suffix] = str(packet_path)
        paths["response_" + suffix] = str(response_path)
    key_path = output_dir / "researcher_key_do_not_share.json"
    write_json(key_path, bundle["researcher_key"])
    paths["researcher_key"] = str(key_path)
    return {"paths": paths, "protocol_sha256": bundle["protocol_sha256"],
            "cohort_sha256": bundle["cohort_sha256"],
            "case_count": len(bundle["packets"]["reviewer_a"]["cases"])}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--per-label", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    print(json.dumps(write_reference_packets(args.output_dir, args.per_label,
                                             seed=args.seed),
                     ensure_ascii=False, indent=2))
