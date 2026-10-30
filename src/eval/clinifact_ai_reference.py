"""Blind, source-grounded AI reference labels for CliniFact confirmation.

The source's single-vote label is kept in a private audit key and never shown
to reviewers. Two model-distinct runs label each frozen task independently;
only unanimous labels or a third-context majority are usable references.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Any

from ..data.clinifact_cohort import LABELS_ORDERED, component_ids, read_source_rows
from ..util import sha256_file, sha256_obj

LABELS = tuple(LABELS_ORDERED)
REVIEW_LABELS = (*LABELS, "unable_to_judge")
PACKET_SCHEMA = "clinifact-blind-ai-reference-packet-v1"
REFERENCE_SCHEMA = "clinifact-blind-ai-reference-v1"

LABEL_PROMPT = (
    "You are an independent clinical-trial abstract claim verifier. Treat the supplied claim and abstract as the only evidence. "
    "Classify the claim into exactly one label: EVIDENCE means the abstract's reported results directly support the claim; "
    "INCONCLUSIVE means the abstract directly reports an inconclusive, non-significant, mixed, or explicitly uncertain result "
    "for the same outcome and comparison; NEI means the abstract does not provide enough directly relevant information to decide. "
    "Do not infer from background statements, plausibility, or outside knowledge. Do not use any dataset label. "
    "For EVIDENCE or INCONCLUSIVE, give one exact contiguous quote from the supplied abstract that supports the label. "
    "For NEI, give an empty quote if no decisive result passage exists. Return JSON only with exactly label, evidence_quote, and rationale."
)

ADJUDICATE_PROMPT = (
    "Independently resolve this disagreement about a clinical-trial abstract and claim. The two prior AI labels and reasons are fallible; "
    "do not vote mechanically or defer to either reviewer. Apply the frozen definitions: EVIDENCE means reported results directly support "
    "the claim; INCONCLUSIVE means the abstract directly reports an inconclusive/non-significant/mixed/uncertain result for the same "
    "outcome and comparison; NEI means the abstract lacks enough directly relevant information. Use only the supplied claim and abstract; "
    "ignore outside knowledge and any source dataset label. For EVIDENCE or INCONCLUSIVE, return one exact contiguous quote from the abstract; "
    "for NEI the quote may be empty. Return JSON only with exactly label, evidence_quote, and rationale."
)


def build_packets(train_path, validation_path, audit: dict[str, Any],
                  history: dict[str, Any], source_cohort: dict[str, Any],
                  *, source_report_sha256: str, seed: int,
                  candidate_binding: dict[str, Any] | None = None,
                  expected_eligible_components: int | None = 177) -> dict[str, Any]:
    """Freeze all still-eligible components after v18 and prior-history exclusion."""
    paths = {"train": train_path, "validation": validation_path}
    hashes = {split: sha256_file(path) for split, path in paths.items()}
    audited = {split: audit["splits"][split]["sha256"] for split in ("train", "validation")}
    if hashes != audited or history.get("source_sha256") != hashes or history.get("complete") is not True:
        raise ValueError("CliniFact audit/history source identity is incomplete or stale")
    cohort_unsigned = dict(source_cohort)
    cohort_hash = cohort_unsigned.pop("cohort_sha256", None)
    if not cohort_hash or sha256_obj(cohort_unsigned) != cohort_hash:
        raise ValueError("candidate source cohort self-hash is invalid")
    rows = read_source_rows(paths)
    component_values = component_ids(rows, sha256_obj(hashes))
    prior_task_ids = {
        str(task_id)
        for scope in (source_cohort.get("scopes") or {}).values()
        for task_id in scope.get("task_ids", [])
    }
    history_indexes = history.get("matched_row_indexes") or {}
    excluded_indexes = set()
    for split in ("train", "validation"):
        for index in history_indexes.get(split, []):
            excluded_indexes.add((split, int(index)))
    excluded_components = {
        component_values[i] for i, row in enumerate(rows)
        if (row["source_split"], row["source_row_index"]) in excluded_indexes
        or row["task_id"] in prior_task_ids
    }
    by_component: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for i, row in enumerate(rows):
        by_component.setdefault(component_values[i], []).append((i, row))
    eligible = {key: values for key, values in by_component.items()
                if key not in excluded_components}
    if expected_eligible_components is not None and len(eligible) != expected_eligible_components:
        raise ValueError(f"eligible component count changed: expected {expected_eligible_components}, got {len(eligible)}")

    selected = []
    private = []
    for component, members in sorted(eligible.items()):
        index, row = min(members, key=lambda item: sha256_obj([seed, "representative", item[1]["task_id"]]))
        task_input = {
            "task_id": row["task_id"],
            "claim": row["claim"],
            "title": row["title"],
            "abstract": row["abstract"],
        }
        input_sha = sha256_obj(task_input)
        case_id = sha256_obj(["clinifact-ai-reference-case-v1", seed, component, input_sha])[:24]
        selected.append({"case_id": case_id, "component_id": component,
                         "task_id": row["task_id"], "input": task_input,
                         "input_sha256": input_sha})
        private.append({"case_id": case_id, "component_id": component,
                        "task_id": row["task_id"], "source_label": row["label"],
                        "source_split": row["source_split"], "source_row_index": index,
                        "input_sha256": input_sha})

    body = {
        "schema": PACKET_SCHEMA,
        "protocol": "clinifact-abstract-claim-threeway-ai-reference-v1",
        "candidate_binding": candidate_binding or {
            "candidate_id": "cand-962fae44e191",
            "candidate_report_sha256": source_report_sha256,
            "source_cohort_sha256": cohort_hash,
            "created_after_candidate_preselection": True,
        },
        "dataset": "clinifact_fixed_abstract",
        "source_hashes": hashes,
        "history_scan_sha256": sha256_obj(history),
        "seed": seed,
        "component_exclusion_policy": "exclude every connected component matching historical rows or any v18 cohort task",
        "eligible_component_count": len(selected),
        "source_label_shown_to_reviewers": False,
        "labels": list(LABELS),
        "instructions": LABEL_PROMPT,
        "cases": selected,
    }
    protocol_sha = sha256_obj(body)
    packets, response_templates = {}, {}
    for slot, offset in (("reviewer_a", 0), ("reviewer_b", 1)):
        visible = list(selected)
        random.Random(seed + offset).shuffle(visible)
        unsigned = {"schema": PACKET_SCHEMA, "protocol_sha256": protocol_sha,
                    "reviewer_slot": slot, "instructions": LABEL_PROMPT,
                    "cases": visible}
        packets[slot] = {**unsigned, "packet_sha256": sha256_obj(unsigned)}
        response_templates[slot] = {
            "schema": "clinifact-blind-ai-review-v1", "protocol_sha256": protocol_sha,
            "packet_sha256": packets[slot]["packet_sha256"], "reviewer_slot": slot,
            "model_id": None, "run_id": None, "annotations": [],
        }
    return {
        "protocol_sha256": protocol_sha,
        "candidate_binding": body["candidate_binding"],
        "source_hashes": hashes,
        "history_scan_sha256": body["history_scan_sha256"],
        "source_cohort_sha256": cohort_hash,
        "packets": packets,
        "response_templates": response_templates,
        "private_key": {"schema": "clinifact-ai-reference-private-key-v1",
                         "protocol_sha256": protocol_sha, "cases": private},
    }


def build_packets_from_source_bundle(source_bundle: dict[str, Any], *,
                                    source_bundle_sha256: str,
                                    seed: int = 2026101701) -> dict[str, Any]:
    """Create blind reference packets from an unlabeled CT.gov source pool.

    The CT.gov expansion already performs identity and component deduplication
    before writing its source bundle.  This adapter deliberately keeps one
    deterministic representative per component, so duplicate registry rows do
    not inflate the reference cohort.  Registry provenance remains private and
    is carried only in ``private_key``; reviewers receive claim/title/abstract.
    """
    if source_bundle.get("schema") != "clinifact-ctgov-pubmed-source-bundle-v1":
        raise ValueError("source bundle schema is not the CT.gov source-pool schema")
    if source_bundle.get("source_label_shown_to_reviewers") is not False:
        raise ValueError("source labels/provenance must be hidden from reviewers")
    if source_bundle.get("eligible_for_confirmation") is not False:
        raise ValueError("unlabeled source pool must not claim confirmation eligibility")
    cases = source_bundle.get("cases")
    packets = source_bundle.get("packets") or {}
    if not isinstance(cases, list) or not cases or set(packets) != {"reviewer_a", "reviewer_b"}:
        raise ValueError("source bundle has no complete blind reviewer packets")
    by_case = {case.get("case_id"): case for case in cases}
    if len(by_case) != len(cases) or any(not isinstance(case, dict) for case in cases):
        raise ValueError("source bundle contains duplicate or malformed case IDs")
    # The capture adapter provides the component identity. Select a stable
    # representative per component using the frozen seed and task identity.
    by_component: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        component = str(case.get("component_id") or "")
        if not component or not case.get("input_sha256") or not case.get("input"):
            raise ValueError("source case lacks component, input hash, or reviewer input")
        by_component.setdefault(component, []).append(case)
    selected = []
    for component, members in sorted(by_component.items()):
        selected.append(min(members, key=lambda case: sha256_obj([
            seed, "representative", case["task_id"]])))
    if len(selected) < 144:
        raise ValueError(f"source pool has only {len(selected)} independent components; 144 required")
    selected_ids = {case["case_id"] for case in selected}
    source_protocol = source_bundle.get("protocol_sha256")
    if not isinstance(source_protocol, str) or len(source_protocol) != 64:
        raise ValueError("source bundle protocol hash is missing")
    body = {
        "schema": PACKET_SCHEMA,
        "protocol": "clinifact-ctgov-abstract-claim-threeway-ai-reference-v1",
        "source_bundle_sha256": source_bundle_sha256,
        "source_protocol_sha256": source_protocol,
        "source_hashes": source_bundle.get("source_hashes", {}),
        "history_exclusion_sha256": source_bundle.get("history_exclusion_sha256"),
        "seed": seed,
        "selection_policy": "one deterministic representative per connected component",
        "independent_component_count": len(selected),
        "source_label_shown_to_reviewers": False,
        "labels": list(LABELS),
        "instructions": LABEL_PROMPT,
        "case_ids": sorted(selected_ids),
    }
    protocol_sha = sha256_obj(body)
    out_packets: dict[str, Any] = {}
    for slot, offset in (("reviewer_a", 0), ("reviewer_b", 1)):
        visible = list(selected)
        random.Random(seed + offset).shuffle(visible)
        cases_for_packet = [{
            "case_id": case["case_id"], "component_id": case["component_id"],
            "task_id": case["task_id"], "input": case["input"],
            "input_sha256": case["input_sha256"],
        } for case in visible]
        unsigned = {"schema": PACKET_SCHEMA, "protocol_sha256": protocol_sha,
                    "reviewer_slot": slot, "instructions": LABEL_PROMPT,
                    "cases": cases_for_packet}
        out_packets[slot] = {**unsigned, "packet_sha256": sha256_obj(unsigned)}
    return {
        "schema": "clinifact-ctgov-ai-reference-bundle-v1",
        "protocol_sha256": protocol_sha,
        "candidate_binding": {
            "candidate_id": None,
            "source_bundle_sha256": source_bundle_sha256,
            "source_protocol_sha256": source_protocol,
            "created_before_candidate_preselection": True,
        },
        "source_hashes": source_bundle.get("source_hashes", {}),
        "packets": out_packets,
        "response_templates": {
            slot: {"schema": "clinifact-blind-ai-review-v1",
                   "protocol_sha256": protocol_sha,
                   "packet_sha256": packet["packet_sha256"],
                   "reviewer_slot": slot, "model_id": None,
                   "run_id": None, "annotations": []}
            for slot, packet in out_packets.items()
        },
        "private_key": {
            "schema": "clinifact-ctgov-ai-reference-private-key-v1",
            "protocol_sha256": protocol_sha,
            "source_bundle_sha256": source_bundle_sha256,
            "provenance": source_bundle.get("private_provenance", []),
        },
    }


def validate_annotation(case: dict[str, Any], raw: Any) -> dict[str, Any]:
    required = {"label", "evidence_quote", "rationale"}
    allowed = required | {"case_id", "input_sha256"}
    metadata = {"reviewer_run_id", "reviewer_slot", "model_id", "unusable_reason"}
    if (not isinstance(raw, dict) or not required.issubset(raw)
            or not set(raw).issubset(allowed | metadata)):
        raise ValueError("AI label output must contain exactly label, evidence_quote, rationale")
    if (raw.get("case_id", case.get("case_id")) != case.get("case_id")
            or raw.get("input_sha256", case.get("input_sha256")) != case.get("input_sha256")):
        raise ValueError("AI label output does not bind the frozen case input")
    label, quote, rationale = raw["label"], raw["evidence_quote"], raw["rationale"]
    if label not in REVIEW_LABELS or not isinstance(quote, str) or not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("AI label output has an invalid label, quote, or rationale")
    abstract = case["input"]["abstract"]
    if quote and quote not in abstract:
        raise ValueError("AI label quote is not an exact contiguous source passage")
    if label in {"EVIDENCE", "INCONCLUSIVE"} and not quote:
        raise ValueError("decisive AI label requires an exact source quote")
    return {"case_id": case["case_id"], "input_sha256": case["input_sha256"],
            "label": label, "evidence_quote": quote, "rationale": rationale.strip()}


def merge_annotations(packet_a: dict[str, Any], response_a: dict[str, Any],
                      packet_b: dict[str, Any], response_b: dict[str, Any]) -> dict[str, Any]:
    if (packet_a.get("protocol_sha256") != packet_b.get("protocol_sha256")
            or packet_a.get("reviewer_slot") != "reviewer_a"
            or packet_b.get("reviewer_slot") != "reviewer_b"
            or response_a.get("status") != "completed"
            or response_b.get("status") != "completed"
            or response_a.get("model_id") == response_b.get("model_id")):
        raise ValueError("AI reviewers must be distinct model/run contexts on the same frozen protocol")
    expected = {case["case_id"]: case for case in packet_a["cases"]}
    votes = []
    for packet, response in ((packet_a, response_a), (packet_b, response_b)):
        if response.get("packet_sha256") != packet.get("packet_sha256") or response.get("protocol_sha256") != packet.get("protocol_sha256"):
            raise ValueError("AI response does not bind its blinded packet")
        by_id = {row.get("case_id"): row for row in response.get("annotations", [])}
        if set(by_id) != set(expected):
            raise ValueError("AI reference response does not cover the exact cohort")
        validated = {}
        for case_id, case in expected.items():
            validated[case_id] = validate_annotation(case, by_id[case_id])
        votes.append(validated)
    reviewer_runs = [
        {str(row.get("reviewer_run_id") or "") for row in response.get("annotations", [])}
        for response in (response_a, response_b)
    ]
    if (any(not run for run in reviewer_runs[0] | reviewer_runs[1])
            or reviewer_runs[0] & reviewer_runs[1]):
        raise ValueError("each reference row must bind to a separate reviewer call context")
    agreements = []
    disagreements = []
    for case_id, case in sorted(expected.items()):
        left, right = votes[0][case_id], votes[1][case_id]
        row = {"case_id": case_id, "task_id": case["task_id"],
               "component_id": case["component_id"], "input": case["input"],
               "input_sha256": case["input_sha256"],
               "reviewer_a": left, "reviewer_b": right}
        (agreements if left["label"] == right["label"] else disagreements).append(row)
    return {"schema": REFERENCE_SCHEMA, "protocol_sha256": packet_a["protocol_sha256"],
            "reviewer_a": {"model_id": response_a["model_id"], "run_id": response_a["run_id"]},
            "reviewer_b": {"model_id": response_b["model_id"], "run_id": response_b["run_id"]},
            "agreement_count": len(agreements), "disagreement_count": len(disagreements),
            "agreements": agreements, "disagreements_for_adjudication": disagreements}


def apply_adjudications(merged: dict[str, Any], adjudications: list[dict[str, Any]], *,
                        model_id: str, run_id: str) -> dict[str, Any]:
    if not model_id or run_id in {merged["reviewer_a"]["run_id"], merged["reviewer_b"]["run_id"]}:
        raise ValueError("adjudicator must use a separate AI run context")
    disputes = {row["case_id"]: row for row in merged["disagreements_for_adjudication"]}
    by_id = {row.get("case_id"): row for row in adjudications}
    if set(by_id) != set(disputes):
        raise ValueError("adjudication must cover every and only disputed case")
    resolved = []
    for case_id, row in sorted(disputes.items()):
        judge = validate_annotation(row, by_id[case_id])
        labels = [row["reviewer_a"]["label"], row["reviewer_b"]["label"], judge["label"]]
        counts = Counter(labels)
        top, count = counts.most_common(1)[0]
        resolved_label = top if count >= 2 else None
        resolved.append({**row, "adjudicator": judge, "three_labels": labels,
                         "resolved_label": resolved_label,
                         "adjudicator_model_id": model_id, "adjudicator_run_id": run_id})
    consensus = [
        {**row, "resolved_label": row["reviewer_a"]["label"], "resolution": "independent_agreement"}
        for row in merged["agreements"]
    ] + [
        {**row, "resolution": "third_ai_majority"}
        for row in resolved if row["resolved_label"] is not None
    ]
    usable = [row for row in consensus if row["resolved_label"] in LABELS]
    counts = Counter(row["resolved_label"] for row in usable)
    return {"schema": REFERENCE_SCHEMA, "protocol_sha256": merged["protocol_sha256"],
            "reviewers": {"reviewer_a": merged["reviewer_a"], "reviewer_b": merged["reviewer_b"],
                          "adjudicator": {"model_id": model_id, "run_id": run_id}},
            "case_count": merged["agreement_count"] + merged["disagreement_count"],
            "agreement_count": merged["agreement_count"],
            "disagreement_count": merged["disagreement_count"],
            "adjudicated_count": len(resolved), "unresolved_count": sum(r["resolved_label"] is None for r in resolved),
            "usable_count": len(usable),
            "class_counts": {label: counts[label] for label in LABELS},
            "consensus": sorted(usable, key=lambda row: row["case_id"]),
            "unresolved": [row for row in resolved if row["resolved_label"] is None]}


def confirmation_capacity(reference: dict[str, Any], minimum_per_class: int = 30) -> dict[str, Any]:
    counts = reference.get("class_counts", {})
    return {"minimum_per_class": minimum_per_class, "class_counts": counts,
            "eligible": reference.get("usable_count", 0) >= minimum_per_class * len(LABELS)
                       and all(int(counts.get(label, 0)) >= minimum_per_class for label in LABELS)}
