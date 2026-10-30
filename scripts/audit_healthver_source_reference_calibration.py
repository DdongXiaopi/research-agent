"""Score frozen HealthVer grounding predictions against upstream labels.

The report is post-hoc diagnostic evidence on an exposed development cohort;
it cannot qualify a judge for release or independent confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.healthver_task_input import SOURCE_SHA256, load_tasks, verify_source_archive

LABELS = ("supported", "contradicted", "insufficient")
SOURCE_LABELS = {
    "SUPPORT": "supported",
    "CONTRADICT": "contradicted",
    "NOT_ENOUGH_INFO": "insufficient",
}
DEFAULT_ROOT = Path(".workbuddy/healthver-grounding-reference-v5")


def _read(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _sha_obj(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    matrix = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}
    for row in rows:
        matrix[row["reference"]][row["prediction"]] += 1
    per_class_f1: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    for label in LABELS:
        tp = matrix[label][label]
        actual = sum(matrix[label].values())
        predicted = sum(matrix[other][label] for other in LABELS)
        per_class_recall[label] = tp / actual if actual else 0.0
        denom = 2 * tp + predicted - tp + actual - tp
        per_class_f1[label] = 2 * tp / denom if denom else 0.0
    negatives = sum(sum(matrix[label].values()) for label in LABELS if label != "supported")
    false_support = sum(matrix[label]["supported"] for label in LABELS if label != "supported")
    return {
        "n": len(rows),
        "reference_label_counts": dict(Counter(row["reference"] for row in rows)),
        "confusion_matrix_actual_by_prediction": matrix,
        "macro_f1": sum(per_class_f1.values()) / len(LABELS),
        "per_class_f1": per_class_f1,
        "per_class_recall": per_class_recall,
        "false_support_rate_among_non_supported": false_support / negatives if negatives else None,
        "invalid_outputs": 0,
        "invalid_output_rate": 0.0,
    }


def _task_id_from_case(case: dict[str, Any]) -> str:
    if len(case.get("evidence", [])) != 1:
        raise ValueError(f"expected one paired evidence document: {case.get('case_id')}")
    doc_id = case["evidence"][0]["doc_id"]
    match = re.fullmatch(r"healthver-claim-(\d+)-abstract-(\d+)", doc_id)
    if not match:
        raise ValueError(f"unexpected HealthVer document identity: {doc_id}")
    return f"healthver-claim-{match.group(1)}-abstract-{match.group(2)}"


def build_report(root: Path, plan_path: Path) -> dict[str, Any]:
    plan, plan_file_sha = _read(plan_path)
    plan_identity = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != _sha_obj(plan_identity):
        raise ValueError("frozen plan hash mismatch")
    if plan.get("status") != "frozen_before_scoring_source_labels":
        raise ValueError("plan must be frozen before scoring source labels")
    if plan.get("source_archive_sha256") != SOURCE_SHA256:
        raise ValueError("plan is bound to a different HealthVer source archive")
    verify_source_archive()

    packet, packet_sha = _read(root / "reviewer_a_packet.json")
    if packet_sha != plan.get("reviewer_input_packet_sha256"):
        raise ValueError("reviewer packet hash differs from frozen plan")
    cases = {case["case_id"]: case for case in packet["cases"]}
    expected_inputs = plan.get("case_input_sha256", {})
    if set(cases) != set(expected_inputs):
        raise ValueError("frozen plan case set differs from reviewer packet")
    for case_id, case in cases.items():
        if case.get("input_sha256") != expected_inputs[case_id]:
            raise ValueError(f"frozen input hash mismatch: {case_id}")

    task_rows = {task["task_id"]: task for task in load_tasks(["dev_optimize"])}
    native_labels: dict[str, str] = {}
    source_tasks: dict[str, str] = {}
    for case_id, case in cases.items():
        task_id = _task_id_from_case(case)
        task = task_rows.get(task_id)
        if task is None:
            raise ValueError(f"case is absent from source task pool: {case_id}/{task_id}")
        claim_prefix = "Does the supplied scientific evidence establish this biomedical claim? Claim: "
        if not case.get("question", "").startswith(claim_prefix):
            raise ValueError(f"unexpected task wording: {case_id}")
        packet_claim = case["question"][len(claim_prefix):].strip().rstrip(".")
        if packet_claim != str(task["question"]).strip().rstrip("."):
            raise ValueError(f"claim mismatch between packet and upstream row: {case_id}")
        if case["evidence"][0]["text"] != task["evidence"]:
            raise ValueError(f"abstract mismatch between packet and upstream row: {case_id}")
        source_label = str(task["options"][task["correct_index"]])
        if source_label not in SOURCE_LABELS:
            raise ValueError(f"unknown source label {source_label!r} for {case_id}")
        native_labels[case_id] = SOURCE_LABELS[source_label]
        source_tasks[case_id] = task_id

    prediction_files = {
        "judge_8b": "v13_calibration_result.json",
        "judge_32b": "v13_calibration_result-32b.json",
        "judge_122b": "v13-calibration-result-122b-diagnostic.json",
    }
    prediction_results: dict[str, dict[str, Any]] = {}
    source_hashes = {"upstream_source_archive": SOURCE_SHA256, "reviewer_packet": packet_sha}
    for role, filename in prediction_files.items():
        expected = plan["prediction_sources"][role]
        if expected["path"] != filename:
            raise ValueError(f"unexpected prediction path in plan for {role}")
        result_path = root / filename
        result, digest = _read(result_path)
        if digest != expected["sha256"]:
            raise ValueError(f"prediction result hash differs from frozen plan: {role}")
        predictions = {row["case_id"]: row for row in result["calibration"]["per_case"]}
        if set(predictions) != set(cases):
            raise ValueError(f"prediction coverage differs from frozen cohort: {role}")
        scored: list[dict[str, str]] = []
        per_case: list[dict[str, str]] = []
        for case_id, prediction in predictions.items():
            if prediction.get("input_sha256") != expected_inputs[case_id]:
                raise ValueError(f"prediction input hash mismatch: {role}/{case_id}")
            label = prediction.get("prediction")
            if label not in LABELS:
                raise ValueError(f"invalid prediction label: {role}/{case_id}/{label}")
            scored.append({"reference": native_labels[case_id], "prediction": label})
            per_case.append({
                "case_id": case_id,
                "task_id": source_tasks[case_id],
                "input_sha256": expected_inputs[case_id],
                "upstream_reference": native_labels[case_id],
                "prediction": label,
            })
        metrics = _metrics(scored)
        metrics["frozen_gate_pass"] = (
            metrics["macro_f1"] >= plan["frozen_gate"]["macro_f1_min"]
            and all(value >= plan["frozen_gate"]["each_class_recall_min"]
                    for value in metrics["per_class_recall"].values())
        )
        prediction_results[role.removeprefix("judge_")] = {"metrics": metrics, "per_case": per_case}
        source_hashes[role] = digest

    reviewer_files = {
        "reviewer_a": "reviewer_a_qwen32_executed_responses.json",
        "reviewer_b": "reviewer_b_deepseek_v32_responses.json",
    }
    reviewer_results: dict[str, Any] = {}
    for role, filename in reviewer_files.items():
        data, digest = _read(root / filename)
        source_hashes[role] = digest
        annotations = {row["case_id"]: row for row in data["annotations"]}
        if set(annotations) != set(cases):
            raise ValueError(f"reviewer coverage differs from source-label cohort: {role}")
        scored = []
        for case_id, annotation in annotations.items():
            if annotation.get("input_sha256") != expected_inputs[case_id]:
                raise ValueError(f"reviewer input hash mismatch: {role}/{case_id}")
            verdict = annotation.get("verdict")
            if verdict in LABELS:
                scored.append({"reference": native_labels[case_id], "prediction": verdict})
        reviewer_results[role] = {
            "reviewer_id": data.get("reviewer_id"),
            "metrics_against_upstream_labels": _metrics(scored),
        }

    return {
        "schema": "healthver-source-reference-calibration-audit-v1",
        "annotation_origin": "original_healthver_manual_evidence_claim_labels",
        "interpretation": "post-hoc diagnostic rescoring against upstream manually annotated evidence labels; exposed development cohort; not independent confirmation or promotion authorization",
        "use_scope": "user-authorized non-commercial internal research only; upstream rights remain unverified; no source redistribution",
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": plan_file_sha,
        "scoring_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_archive_sha256": SOURCE_SHA256,
        "source_hashes": source_hashes,
        "cohort": {
            "dataset": plan["dataset"],
            "split": plan["split"],
            "n_cases": len(cases),
            "upstream_label_distribution": dict(Counter(native_labels.values())),
            "source_task_ids": source_tasks,
        },
        "frozen_gate": plan["frozen_gate"],
        "judge_results": prediction_results,
        "ai_reviewer_results": reviewer_results,
        "promotion_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_ROOT / "source-reference-recalibration-plan-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.root, args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
