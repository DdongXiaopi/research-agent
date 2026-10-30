"""Freeze a fresh, source-balanced HealthVer top-3 grounding packet.

Each case contains three distinct abstracts for the same claim and the same
upstream relation. Claims and abstract IDs already present in prior grounding
packets are excluded before deterministic class-balanced sampling. Reviewers
label each claim-document pair and cite sentence rationales separately; local
code aggregates those relations deterministically.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_healthver_grounding_reference_packet import SOURCE_LABELS, _case_id
from scripts.score_task_input_grounding_human_reference import _validate_packet
from src.data.healthver_task_input import LABELS, load_tasks
from src.util import sha256_file, sha256_obj, write_json

OUTPUT = ROOT / ".workbuddy/healthver-grounding-same-claim-top3-v1"
CLAIM_ID = re.compile(r"healthver-claim-(\d+)-abstract-(\d+)")


def _normalize_claim(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().rstrip(".")).casefold()


def _previously_exposed() -> tuple[set[str], set[str], list[str]]:
    claims: set[str] = set()
    documents: set[str] = set()
    packet_hashes: list[str] = []
    for base in (ROOT / ".workbuddy", ROOT / "artifacts/probe"):
        for path in sorted(base.rglob("*.json")):
            if "grounding" not in path.name.casefold() and "healthver" not in str(path).casefold():
                continue
            if "packet" not in path.name.casefold():
                continue
            try:
                packet = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(packet, dict) or not isinstance(packet.get("cases"), list):
                continue
            packet_hashes.append(sha256_file(path))
            for case in packet["cases"]:
                if not isinstance(case, dict):
                    continue
                question = str(case.get("question") or "")
                if "claim:" in question.casefold():
                    claims.add(_normalize_claim(re.split(r"claim:", question, maxsplit=1,
                                                         flags=re.IGNORECASE)[1]))
                for evidence in case.get("evidence") or []:
                    if isinstance(evidence, dict) and evidence.get("doc_id") is not None:
                        documents.add(str(evidence["doc_id"]))
    return claims, documents, packet_hashes


def select_same_claim_groups(per_label: int = 20, *, seed: int = 20261016):
    if not 1 <= per_label <= 100 or type(seed) is not int:
        raise ValueError("per_label must be 1..100 and seed must be an integer")
    exposed_claims, exposed_docs, packet_hashes = _previously_exposed()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    claim_texts: dict[str, str] = {}
    excluded_claims = set()
    excluded_documents = 0
    for task in load_tasks(["dev_optimize"]):
        match = CLAIM_ID.fullmatch(str(task["task_id"]))
        if not match:
            raise ValueError("HealthVer task identity does not match the audited source contract")
        label = str(task["options"][int(task["correct_index"])])
        if label not in SOURCE_LABELS:
            raise ValueError("HealthVer source label is outside the frozen mapping")
        claim = _normalize_claim(str(task["question"]).split("Claim:", 1)[-1])
        doc_id = str(task["task_id"])
        if claim in exposed_claims:
            excluded_claims.add(claim)
            continue
        if doc_id in exposed_docs:
            excluded_documents += 1
            continue
        claim_id = match.group(1)
        claim_texts[claim_id] = str(task["question"]).split("Claim:", 1)[-1].strip()
        grouped[(claim_id, label)].append(task)

    eligible: dict[str, list[tuple[str, str, list[dict[str, Any]]]]] = defaultdict(list)
    for (claim_id, label), rows in grouped.items():
        unique = {str(row["task_id"]): row for row in rows}
        if len(unique) >= 3:
            eligible[label].append((claim_id, label, list(unique.values())))
    selected = []
    for label in LABELS:
        options = sorted(eligible[label], key=lambda row: sha256_obj([seed, "claim-group", label, row[0]]))
        if len(options) < per_label:
            raise ValueError(f"only {len(options)} fresh same-claim groups for {label}; need {per_label}")
        for claim_id, group_label, rows in options[:per_label]:
            chosen = sorted(rows, key=lambda row: sha256_obj([seed, "abstract", row["task_id"]]))[:3]
            if len({row["task_id"] for row in chosen}) != 3:
                raise ValueError("a same-claim case must contain three distinct abstracts")
            selected.append({"claim_id": claim_id, "label": group_label,
                             "claim": claim_texts[claim_id], "tasks": chosen})
    if len({row["claim_id"] for row in selected}) != len(selected):
        raise ValueError("claim groups must be disjoint across the reference packet")
    audit = {
        "previous_packet_file_count": len(packet_hashes),
        "previous_packet_sha256s": packet_hashes,
        "previously_exposed_claim_count": len(exposed_claims),
        "previously_exposed_document_count": len(exposed_docs),
        "excluded_exposed_claim_count": len(excluded_claims),
        "excluded_exposed_document_rows": excluded_documents,
        "eligible_groups_by_source_label": {label: len(eligible[label]) for label in LABELS},
    }
    return selected, audit


def _source_hashes() -> dict[str, str]:
    from src.data import healthver_task_input
    return {
        "calibration_selection_source_sha256": hashlib.sha256(
            inspect.getsource(select_same_claim_groups).encode()).hexdigest(),
        "task_input_evidence_source_sha256": sha256_file(
            Path(inspect.getsourcefile(healthver_task_input)).resolve()),
        "reference_packet_builder_source_sha256": hashlib.sha256(
            inspect.getsource(build_reference_packets).encode()).hexdigest(),
        "human_reference_scorer_source_sha256": sha256_file(
            ROOT / "scripts/score_task_input_grounding_human_reference.py"),
    }


def build_reference_packets(per_label: int = 20, *, seed: int = 20261016):
    selected, exposure_audit = select_same_claim_groups(per_label, seed=seed)
    cases, key_rows = [], []
    for group in selected:
        evidence = [{"doc_id": str(task["task_id"]), "text": str(task["evidence"])}
                    for task in group["tasks"]]
        evidence.sort(key=lambda row: sha256_obj([seed, "document-order", row["doc_id"]]))
        question = ("Do these supplied scientific abstracts collectively establish this "
                    "biomedical claim? Claim: " + group["claim"])
        input_sha = sha256_obj({"question": question, "evidence": evidence})
        group_id = sha256_obj(["healthver-claim", group["claim_id"]])
        task_id = f"healthver-claim-{group['claim_id']}-same-label-top3"
        cases.append({"task_id": task_id, "input_sha256": input_sha, "question": question,
                      "evidence": evidence, "source_label_for_audit_only": group["label"]})
        key_rows.append({"task_id": task_id, "claim_id": group["claim_id"],
                         "source_label_for_post_review_only": group["label"],
                         "input_sha256": input_sha,
                         "source_task_ids": sorted(row["task_id"] for row in group["tasks"]),
                         "component_id": group_id})
    counts = Counter(row["source_label_for_audit_only"] for row in cases)
    if any(counts[label] != per_label for label in LABELS):
        raise ValueError("same-claim top-3 packet is not class balanced")
    source_hashes = _source_hashes()
    cohort_sha = sha256_obj({"dataset": "healthver", "seed": seed,
                             "construction": "same-claim-same-source-label-three-abstracts",
                             "task_ids": sorted(row["task_id"] for row in cases),
                             "input_sha256s": sorted(row["input_sha256"] for row in cases),
                             "source_manifest_sha256": sha256_file(ROOT / "data/manifests/healthver_task_input.json"),
                             "exposure_audit": exposure_audit})
    context = {
        "schema": "task-input-grounding-reference-protocol-v6",
        "dataset": "healthver", "split": "dev_optimize",
        "sampling": "fresh-claim-disjoint; three-distinct-abstracts-per-claim; all-three-share-upstream-relation",
        "per_label": per_label, "offset_per_label": 0,
        "judge_evidence_character_limit_per_document": 6000,
        "rubric_version": "scientific-grounding-ai-reference-same-claim-top3-v2",
        "case_input_sha256s": sorted(row["input_sha256"] for row in cases),
        "case_count": len(cases), "documents_per_case": 3,
        "labels": ["supported", "contradicted", "insufficient"],
        "source_hashes": source_hashes, "cohort_sha256": cohort_sha,
        "exposure_audit": exposure_audit,
        "source_label_semantics": "all three abstracts share the same upstream claim-abstract relation; aggregate verdict remains subject to independent AI review",
        "annotation_granularity": "independent-relation-and-rationale-per-document",
        "aggregation_version": "one-sided-decisive-relation-wins; both-sides-or-no-decisive-relation=insufficient",
    }
    protocol_sha = sha256_obj(context)
    instructions = (
        "For each supplied abstract independently, classify the relation between the claim and that document as "
        "supported, contradicted, or insufficient. Preserve material population, intervention/exposure, comparator, "
        "outcome, direction, distance, setting, and time qualifiers; a related topic or broad background statement "
        "is insufficient. A decisive label requires one or more exact zero-based sentence IDs from that same document "
        "that directly establish the relation; use an empty sentence_ids list for insufficient. Return exactly one "
        "document_relations entry for every supplied doc_id, with no missing or duplicate IDs. Do not assign an "
        "overall case verdict: local code aggregates per-document labels, treating support plus contradiction as "
        "insufficient. Do not consult another reviewer or dataset labels. Include a reason of at most 30 words and "
        "confidence 1, 2, or 3. Return one JSON object with an annotations array; each item has case_id, input_sha256, "
        "document_relations (each has only doc_id, verdict, sentence_ids), reason, confidence_1_to_3.")
    packets, key = {}, {"schema": "healthver-grounding-same-claim-top3-key-v1",
                        "protocol_sha256": protocol_sha, "cases": key_rows,
                        "promotion_authorized": False}
    key["key_sha256"] = sha256_obj(key)
    for slot, slot_seed in (("reviewer_a", 0), ("reviewer_b", 1)):
        visible = [{k: value for k, value in case.items() if k != "source_label_for_audit_only"}
                   for case in cases]
        random.Random(int(protocol_sha[:16], 16) + slot_seed).shuffle(visible)
        reviewer_cases = [{"case_id": _case_id(protocol_sha, case["task_id"]),
                           "question": case["question"], "evidence": case["evidence"],
                           "input_sha256": case["input_sha256"]} for case in visible]
        packet = {"schema": "task-input-grounding-human-reference-v2",
                  "protocol_sha256": protocol_sha, "protocol_context": context,
                  "reviewer_slot": slot, "scope": {"dataset": "healthver",
                  "split": "dev_optimize", "case_count": len(reviewer_cases),
                  "use": "AI-only top-3 semantic calibration; no candidate confirmation"},
                  "instructions": instructions, "cases": reviewer_cases}
        packet["packet_sha256"] = sha256_obj(packet)
        _validate_packet(packet, slot)
        packets[slot] = packet
    return {"schema": "healthver-grounding-same-claim-top3-bundle-v1",
            "protocol_sha256": protocol_sha, "packets": packets,
            "researcher_key": key, "selection_audit": exposure_audit,
            "source_reference_counts": dict(counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--per-label", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen reference packet: {args.output_dir}")
    bundle = build_reference_packets(args.per_label, seed=args.seed)
    args.output_dir.mkdir(parents=True)
    for slot, packet in bundle["packets"].items():
        write_json(args.output_dir / f"{slot}_packet.json", packet)
    write_json(args.output_dir / "researcher_key_do_not_share.json", bundle["researcher_key"])
    write_json(args.output_dir / "selection_audit.json", bundle["selection_audit"])
    write_json(args.output_dir / "manifest.json", {k: v for k, v in bundle.items()
                                                      if k not in {"packets", "researcher_key"}})
    print(json.dumps({"output_dir": str(args.output_dir), "protocol_sha256": bundle["protocol_sha256"],
                      "case_count": sum(bundle["source_reference_counts"].values()),
                      "source_reference_counts": bundle["source_reference_counts"],
                      "selection_audit": bundle["selection_audit"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
