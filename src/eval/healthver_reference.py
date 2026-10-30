"""Build a source-native HealthVer reference from the archived upstream labels."""

from __future__ import annotations

import re
from typing import Any

from ..data.healthver_task_input import SOURCE_SHA256, load_tasks, verify_source_archive
from ..util import sha256_obj

SOURCE_TO_GROUNDING = {
    "SUPPORT": "supported",
    "CONTRADICT": "contradicted",
    "NOT_ENOUGH_INFO": "insufficient",
}
QUESTION_PREFIX = (
    "Does the supplied scientific evidence establish this biomedical claim? Claim: "
)


def build_reference(packet: dict[str, Any]) -> dict[str, Any]:
    """Bind official per-claim/abstract labels to every case in a frozen packet."""
    from scripts.score_task_input_grounding_human_reference import _validate_packet

    if packet.get("protocol_context", {}).get("dataset") != "healthver":
        raise ValueError("source-native reference builder only accepts HealthVer packets")
    if packet.get("protocol_context", {}).get("split") != "dev_optimize":
        raise ValueError("HealthVer source-native calibration is limited to dev_optimize")
    cases = _validate_packet(packet, packet.get("reviewer_slot", "reviewer_a"))
    verify_source_archive()
    tasks = {row["task_id"]: row for row in load_tasks(["dev_optimize"])}
    top3 = packet.get("protocol_context", {}).get("rubric_version") == (
        "scientific-grounding-ai-reference-same-claim-top3-v2")
    per_case = []
    for case_id, case in sorted(cases.items()):
        expected_docs = 3 if top3 else 1
        if len(case["evidence"]) != expected_docs:
            raise ValueError(f"source-native reference requires {expected_docs} paired abstracts: {case_id}")
        if "Claim:" not in case["question"]:
            raise ValueError(f"unexpected packet question format: {case_id}")
        claim = re.split(r"Claim:", case["question"], maxsplit=1, flags=re.IGNORECASE)[1].strip().rstrip(".")
        relations = []
        for evidence in case["evidence"]:
            doc_id = evidence["doc_id"]
            match = re.fullmatch(r"healthver-claim-(\d+)-abstract-(\d+)", doc_id)
            if not match:
                raise ValueError(f"malformed HealthVer task identity: {doc_id}")
            task_id = f"healthver-claim-{match.group(1)}-abstract-{match.group(2)}"
            task = tasks.get(task_id)
            if task is None:
                raise ValueError(f"task is absent from the source-native development pool: {task_id}")
            if claim.casefold() != str(task["question"]).strip().rstrip(".").casefold():
                raise ValueError(f"claim differs from original source task: {case_id}")
            if evidence["text"] != task["evidence"]:
                raise ValueError(f"abstract differs from original source task: {case_id}")
            source_label = str(task["options"][task["correct_index"]])
            if source_label not in SOURCE_TO_GROUNDING:
                raise ValueError(f"unknown upstream label {source_label!r}: {case_id}")
            relations.append({"doc_id": doc_id, "task_id": task_id,
                              "source_label": source_label,
                              "resolved_verdict": SOURCE_TO_GROUNDING[source_label]})
        labels = {row["resolved_verdict"] for row in relations}
        decisive = labels & {"supported", "contradicted"}
        aggregate = ("insufficient" if decisive == {"supported", "contradicted"}
                     else next(iter(decisive)) if decisive else "insufficient")
        per_case.append({"case_id": case_id, "task_id": relations[0]["task_id"],
                         "task_ids": [row["task_id"] for row in relations],
                         "document_relations": relations,
                         "input_sha256": case["input_sha256"],
                         "resolved_verdict": aggregate})
    reference: dict[str, Any] = {
        "schema": ("healthver-source-top3-grounding-reference-v1" if top3
                   else "healthver-source-grounding-reference-v1"),
        "annotation_origin": "upstream_manual",
        "dataset": "healthver",
        "split": "dev_optimize",
        "interpretation": ("each original upstream manual claim-abstract relation is retained separately; case label uses the frozen local one-sided-decisive aggregation rule; labels are evidence relations, not biomedical truth"
                           if top3 else
                           "original upstream manual evidence-claim relation labels; label means relation to the paired abstract, not independent biomedical truth"),
        "source_archive_sha256": SOURCE_SHA256,
        "packet_sha256": packet["packet_sha256"],
        "protocol_sha256": packet["protocol_sha256"],
        "per_case": per_case,
        "promotion_authorized": False,
    }
    reference["reference_sha256"] = sha256_obj(reference)
    return reference
