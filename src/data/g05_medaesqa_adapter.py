"""Deterministic, source-native adapter for MedAESQA G05 calibration.

The adapter consumes a caller-supplied frozen JSON snapshot and a matching
exact-history scan report. It makes no network or model calls. MedAESQA's
``supporting``/``contradicting``/``neutral`` citation labels map directly to
the selected-answer support task. ``not relevant`` and ``invalid citation``
are excluded because neither is a relation label for this task.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

from ..eval.grounding_selected_answer_calibration import (
    PACKET_SCHEMA,
    REFERENCE_SCHEMA,
    SCOPE,
    TASK_CONTRACT,
    seal_selected_answer_packet,
    seal_selected_answer_reference,
    validate_selected_answer_packet,
    validate_selected_answer_reference,
    build_selected_answer_plan,
)

DATASET_ID = "g05-medaesqa-selected-answer-support"
HISTORY_SCAN_SCHEMA = "g05-medaesqa-history-scan-v1"
MEDAESQA_LABEL_MAP = {
    "supporting": "supported",
    "contradicting": "contradicted",
    "neutral": "insufficient",
}
EXCLUDED_LABELS = {"not relevant", "invalid citation"}
_CLASS_PRIORITY = ("contradicting", "neutral", "supporting")
_SOURCE_REFERENCE_SCHEMA = "g05-selected-answer-support-source-native-reference-v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _raw_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _annotation_binding(*, question_id: str, model_id: str,
                        answer_sentence_id: str, pmid: str,
                        source_label: str, answer: str, evidence: str) -> dict[str, str]:
    return {
        "question_id": question_id,
        "model_id": model_id,
        "answer_sentence_id": answer_sentence_id,
        "pmid": pmid,
        "source_label": source_label,
        "answer_sha256": _raw_sha(answer.encode("utf-8")),
        "evidence_sha256": _raw_sha(evidence.encode("utf-8")),
    }


def _collect_candidates(rows: list[dict[str, Any]], excluded_question_ids: set[str]):
    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    raw_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    malformed_counts: Counter[str] = Counter()
    for source_row in rows:
        question_id = str(source_row.get("question_id", "")).strip()
        question = source_row.get("question")
        if not question_id or not isinstance(question, str) or not question.strip():
            malformed_counts["missing_question_identity_or_text"] += 1
            continue
        if question_id in excluded_question_ids:
            excluded_counts["history_question_group"] += sum(
                len((answer or {}).get("answer_sentences") or [])
                for answer in (source_row.get("machine_generated_answers") or {}).values()
                if isinstance(answer, dict)
            )
            continue
        answers = source_row.get("machine_generated_answers")
        if not isinstance(answers, dict):
            malformed_counts["missing_machine_generated_answers"] += 1
            continue
        for model_id, answer in answers.items():
            if not isinstance(answer, dict):
                malformed_counts["malformed_answer_record"] += 1
                continue
            for sentence in answer.get("answer_sentences") or []:
                if not isinstance(sentence, dict):
                    malformed_counts["malformed_answer_sentence"] += 1
                    continue
                answer_text = sentence.get("answer_sentence")
                if not isinstance(answer_text, str) or not answer_text.strip():
                    malformed_counts["missing_answer_sentence"] += 1
                    continue
                citations = sentence.get("citation_assessment") or []
                for citation in citations:
                    if not isinstance(citation, dict):
                        malformed_counts["malformed_citation_assessment"] += 1
                        continue
                    source_label = citation.get("evidence_relation")
                    raw_counts[str(source_label)] += 1
                    if source_label in EXCLUDED_LABELS:
                        excluded_counts[source_label.replace(" ", "_")] += 1
                        continue
                    if source_label not in MEDAESQA_LABEL_MAP:
                        malformed_counts["unknown_relation_label"] += 1
                        continue
                    pmid = str(citation.get("cited_pmid", "")).strip()
                    evidence = citation.get("evidence_support")
                    if (not pmid.isdigit() or not isinstance(evidence, str)
                            or not evidence.strip()):
                        malformed_counts["missing_pmid_or_evidence"] += 1
                        continue
                    candidates[source_label].append({
                        "question_id": question_id,
                        "question": question.strip(),
                        "model_id": str(model_id),
                        "answer_sentence_id": str(sentence.get("answer_sentence_id", "")),
                        "answer": answer_text.strip(),
                        "pmid": pmid,
                        "evidence": evidence.strip(),
                        "source_label": source_label,
                    })
    return candidates, raw_counts, excluded_counts, malformed_counts


def build_medaesqa_calibration_bundle(
    source_bytes: bytes,
    *,
    history_report: dict[str, Any],
    protocol_sha256: str,
    sample_seed: str,
    minimum_per_class: int = 30,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build a hash-bound packet, native-label reference and zero-call audit.

    Source groups are PMIDs. Every selected PMID is assigned to at most one
    class, resolving cross-label PMID conflicts by scarce-first allocation
    (contradicted, insufficient, supported). History-matched question groups
    are excluded before allocation. No references are generated by a model.
    """
    if not isinstance(source_bytes, bytes) or not source_bytes:
        raise ValueError("MedAESQA source snapshot must be non-empty bytes")
    if not isinstance(history_report, dict) or history_report.get("schema") != HISTORY_SCAN_SCHEMA:
        raise ValueError("a matching MedAESQA exact-history report is required")
    source_sha256 = _raw_sha(source_bytes)
    if history_report.get("source_sha256") != source_sha256:
        raise ValueError("history report source hash mismatch")
    history_sha256 = _sha(history_report)
    if not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64:
        raise ValueError("frozen protocol SHA-256 is required")
    if not isinstance(sample_seed, str) or not sample_seed.strip():
        raise ValueError("deterministic sample seed is required")
    if type(minimum_per_class) is not int or minimum_per_class < 30:
        raise ValueError("minimum per class cannot be lower than the frozen 30-case gate")
    try:
        decoded = json.loads(source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("MedAESQA source snapshot is not valid UTF-8 JSON") from exc
    if not isinstance(decoded, list) or not decoded:
        raise ValueError("MedAESQA source snapshot must contain question records")
    excluded_question_ids = history_report.get("matched_question_ids")
    if (not isinstance(excluded_question_ids, list)
            or any(not isinstance(item, str) for item in excluded_question_ids)):
        raise ValueError("history report question exclusions are invalid")
    candidates, raw_counts, excluded_counts, malformed_counts = _collect_candidates(
        decoded, set(excluded_question_ids))

    selected: dict[str, list[dict[str, str]]] = {
        "supported": [], "contradicted": [], "insufficient": [],
    }
    used_pmids: set[str] = set()
    for source_label in _CLASS_PRIORITY:
        verdict = MEDAESQA_LABEL_MAP[source_label]
        order = sorted(
            candidates[source_label],
            key=lambda row: hashlib.sha256((
                f"{sample_seed}\0{source_label}\0{row['pmid']}\0{row['question_id']}\0"
                f"{row['model_id']}\0{row['answer_sentence_id']}"
            ).encode("utf-8")
            ).hexdigest(),
        )
        seen_here: set[str] = set()
        for row in order:
            if row["pmid"] in used_pmids or row["pmid"] in seen_here:
                continue
            seen_here.add(row["pmid"])
            selected[verdict].append(row)
            if len(selected[verdict]) == minimum_per_class:
                break
        if len(selected[verdict]) < minimum_per_class:
            raise ValueError(
                f"MedAESQA has fewer than {minimum_per_class} independent source groups "
                f"for {verdict} after history and cross-class PMID exclusions"
            )
        used_pmids.update(row["pmid"] for row in selected[verdict])

    cases: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    source_label_by_verdict = {value: key for key, value in MEDAESQA_LABEL_MAP.items()}
    for verdict in ("supported", "contradicted", "insufficient"):
        for index, row in enumerate(selected[verdict]):
            case_id = f"medaesqa-{verdict}-{index:03d}-{row['pmid']}"
            task_id = f"{row['question_id']}:{row['model_id']}:{row['answer_sentence_id']}:{row['pmid']}"
            doc_id = f"pmid:{row['pmid']}"
            case = {
                "case_id": case_id,
                "task_id": task_id,
                "source_group_id": row["pmid"],
                "source_id": row["pmid"],
                "question": row["question"],
                "selected_answer": row["answer"],
                "evidence": [{"doc_id": doc_id, "text": row["evidence"]}],
            }
            binding = _annotation_binding(
                question_id=row["question_id"], model_id=row["model_id"],
                answer_sentence_id=row["answer_sentence_id"], pmid=row["pmid"],
                source_label=source_label_by_verdict[verdict],
                answer=row["answer"], evidence=row["evidence"],
            )
            reference_rows.append({
                "case_id": case_id,
                "source_group_id": row["pmid"],
                "input_sha256": "",  # filled after packet sealing
                "source_label": source_label_by_verdict[verdict],
                "resolved_verdict": verdict,
                "resolved_sentence_ids": [0],
                "source_annotation": binding,
                "source_annotation_sha256": _sha(binding),
            })
            cases.append(case)

    packet = seal_selected_answer_packet({
        "schema": PACKET_SCHEMA,
        "dataset": DATASET_ID,
        "split": "calibration",
        "task_contract": TASK_CONTRACT,
        "scope": SCOPE,
        "source_pool_sha256": source_sha256,
        "publication_status_report_sha256": history_sha256,
        "sample_plan_sha256": _sha({"seed": sample_seed, "minimum_per_class": minimum_per_class,
                                     "source_sha256": source_sha256,
                                     "history_scan_sha256": history_sha256}),
        "protocol_sha256": protocol_sha256,
        "source_id_type": "pmid",
        "source_data_sha256": source_sha256,
        "history_scan_sha256": history_sha256,
        "cases": cases,
    })
    input_by_id = {row["case_id"]: row["input_sha256"] for row in packet["cases"]}
    for row in reference_rows:
        row["input_sha256"] = input_by_id[row["case_id"]]
    reference = seal_selected_answer_reference({
        "schema": _SOURCE_REFERENCE_SCHEMA,
        "annotation_origin": "source_native_human",
        "task_contract": TASK_CONTRACT,
        "scope": SCOPE,
        "packet_sha256": packet["packet_sha256"],
        "protocol_sha256": protocol_sha256,
        "source_data_sha256": source_sha256,
        "history_scan_sha256": history_sha256,
        "label_mapping_sha256": _sha(MEDAESQA_LABEL_MAP),
        "per_case": reference_rows,
    })
    validation = validate_selected_answer_packet(packet)
    reference_validation = validate_selected_answer_reference(packet, reference)
    audit = {
        "schema": "g05-medaesqa-adapter-audit-v1",
        "dataset": DATASET_ID,
        "source_paper": "https://doi.org/10.1038/s41597-025-05233-z",
        "source_osf_doi": "https://doi.org/10.17605/OSF.IO/YDBZQ",
        "source_code_repository": "https://github.com/deepaknlp/MedAESQA",
        "source_hf_mirror_revision": "d8b626a5af08b07205eb538a1cea6447977b96bf",
        "source_hf_mirror_card_license": "CC0-1.0 (mirror metadata only)",
        "source_paper_license": "CC-BY-4.0 (article only)",
        "source_code_license": "Apache-2.0 (repository code only)",
        "source_osf_original_license": "unknown_unverified",
        "source_data_sha256": source_sha256,
        "source_data_bytes": len(source_bytes),
        "history_scan_sha256": history_sha256,
        "excluded_history_question_ids": sorted(excluded_question_ids),
        "source_relation_counts": dict(sorted(raw_counts.items())),
        "excluded_not_relevant": excluded_counts["not_relevant"],
        "excluded_invalid_citation": excluded_counts["invalid_citation"],
        "excluded_history_question_records": excluded_counts["history_question_group"],
        "malformed_or_unknown_exclusions": dict(sorted(malformed_counts.items())),
        "selected_case_count": validation["case_count"],
        "selected_independent_pmid_groups": validation["source_group_count"],
        "selected_class_case_counts": reference_validation["class_case_counts"],
        "selected_class_independent_groups": reference_validation["class_independent_groups"],
        "source_label_mapping": MEDAESQA_LABEL_MAP,
        "excluded_source_labels": sorted(EXCLUDED_LABELS),
        "class_allocation_priority": list(_CLASS_PRIORITY),
        "sample_seed": sample_seed,
        "labels_generated": False,
        "model_calls": 0,
        "target_scorer_calls": 0,
        "personal_internal_calibration_authorization": True,
        "personal_authorization_scope": "user-authorized noncommercial internal project calibration only",
        "upstream_data_license_status": "unknown_unverified_osf_original",
        "upstream_license_claim": False,
        "redistribution_authorized_by_upstream": False,
    }
    return packet, reference, audit


def build_medaesqa_zero_call_plan(
    packet: dict[str, Any],
    reference: dict[str, Any],
    *,
    model_id: str,
    created_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze target prompts and a strict $0/no-dispatch preflight record."""
    allowed_free_models = {
        "Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct",
        "THUDM/GLM-4-9B-0414", "XingChenAGI/Xing4.0-29B",
    }
    if model_id not in allowed_free_models:
        raise ValueError("G05 zero-call plan only accepts the approved free model IDs")
    model_protocol = {
        "provider": "siliconflow",
        "base_url": "https://api.siliconflow.cn/v1",
        "model_id": model_id,
        "price_policy": "zero_price_only",
        "paid_fallback": False,
        "max_physical_attempts_per_case": 1,
        "retries_per_case": 0,
    }
    model_protocol_sha256 = _sha(model_protocol)
    plan = build_selected_answer_plan(
        packet, reference, model_id=model_id,
        model_protocol_sha256=model_protocol_sha256,
        created_at=created_at,
    )
    preflight = {
        "schema": "g05-medaesqa-zero-call-preflight-v1",
        "status": "passed_zero_call_plan_only",
        "dataset": DATASET_ID,
        "packet_sha256": packet["packet_sha256"],
        "reference_sha256": reference["reference_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "model_protocol": model_protocol,
        "model_protocol_sha256": model_protocol_sha256,
        "request_count": plan["request_count"],
        "max_physical_attempts": plan["max_physical_attempts"],
        "model_calls": 0,
        "target_scorer_calls": 0,
        "cost_usd": "0.00",
        "labels_generated": False,
        "dispatch_authorized": False,
        "zero_price_only": True,
        "paid_fallback": False,
        "account_price_verified": False,
        "upstream_data_license_status": "unknown_unverified_osf_original",
        "personal_internal_calibration_authorization": True,
    }
    return plan, preflight


__all__ = ["DATASET_ID", "MEDAESQA_LABEL_MAP", "EXCLUDED_LABELS",
           "build_medaesqa_calibration_bundle", "build_medaesqa_zero_call_plan"]
