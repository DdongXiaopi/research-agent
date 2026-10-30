"""Compare a frozen grounding judge run with two blind human references.

This is an offline, post-review analysis. It never constructs a provider and
does not treat PubMedQA's benchmark answer as semantic-entailment gold.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import calibrate_task_input_grounding as calibration
from scripts.prepare_task_input_grounding_reference_packet import (
    SCHEMA as PACKET_SCHEMA,
    _case_id,
    build_reference_packets,
)
from scripts.score_task_input_grounding_human_reference import (
    score_human_reference,
)
from src.eval.grounding import parse_proposition_verdict
from src.runtime.input_evidence import task_input_docs
from src.util import sha256_obj


HUMAN_REPORT_SCHEMA = "task-input-grounding-human-reference-agreement-v1"
JUDGE_REPORT_STATUS = "exploratory_task_input_grounding_calibration"
JUDGE_ROW_FIELDS = {
    "protocol", "task_id", "gold_answer", "gold_supported", "gold_reason",
    "opposite_answer", "opposite_supported", "opposite_reason", "raw_verdict",
    "prompt_tokens", "completion_tokens", "evidence_chars",
}
HUMAN_LABELS = ("supported", "contradicted", "insufficient")
JUDGE_LABELS = (*HUMAN_LABELS, "invalid_output")


def _current_judge_source_hashes() -> dict[str, str]:
    grounding_source = Path(inspect.getsourcefile(parse_proposition_verdict)).resolve()
    evidence_source = Path(inspect.getsourcefile(task_input_docs)).resolve()
    split_source = Path(inspect.getsourcefile(calibration.load_tasks)).resolve()
    return {
        "calibration_script_sha256": hashlib.sha256(
            Path(calibration.__file__).resolve().read_bytes()).hexdigest(),
        "grounding_module_sha256": hashlib.sha256(
            grounding_source.read_bytes()).hexdigest(),
        "task_input_evidence_module_sha256": hashlib.sha256(
            evidence_source.read_bytes()).hexdigest(),
        "split_loader_sha256": hashlib.sha256(split_source.read_bytes()).hexdigest(),
    }


def _comparison_source_hashes() -> dict[str, str]:
    from scripts import score_task_input_grounding_human_reference as human_scorer

    return {
        "comparison_script_sha256": hashlib.sha256(
            Path(__file__).resolve().read_bytes()).hexdigest(),
        "human_reference_scorer_sha256": hashlib.sha256(
            Path(human_scorer.__file__).resolve().read_bytes()).hexdigest(),
    }


def _verify_plan(plan: dict[str, Any], expected_plan_sha256: str,
                 cases: list[dict[str, Any]]) -> None:
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("schema") != "task-input-grounding-calibration-execution-plan-v1":
        raise ValueError("unsupported judge execution-plan schema")
    if plan.get("plan_sha256") != sha256_obj(unsigned):
        raise ValueError("judge execution-plan self-hash is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_plan_sha256 or ""):
        raise ValueError("expected judge execution-plan hash is required")
    if plan["plan_sha256"] != expected_plan_sha256:
        raise ValueError("judge execution-plan hash does not match the expected value")
    if (plan.get("dataset") != "pubmedqa" or plan.get("split") != "dev_optimize"
            or plan.get("selection") != "task-id SHA-256 hash, balanced yes/no, supplied evidence only"
            or plan.get("selection_gate_read") is not False
            or plan.get("official_validation_read") is not False
            or plan.get("official_test_read") is not False):
        raise ValueError("judge execution plan is outside the permitted calibration scope")
    if plan.get("source_hashes") != _current_judge_source_hashes():
        raise ValueError("judge execution plan is bound to stale implementation sources")

    tasks_by_id = {str(task["task_id"]): task for task in cases}
    task_ids = set(tasks_by_id)
    if (len(tasks_by_id) != len(cases)
            or set(plan.get("task_ids", [])) != task_ids
            or plan.get("task_count") != len(task_ids)):
        raise ValueError("judge execution plan does not cover the human-reference cohort")
    pending = set(plan.get("pending_task_ids", []))
    cached = set(plan.get("cached_task_ids", []))
    if pending & cached or pending | cached != task_ids:
        raise ValueError("judge execution plan has malformed pending/cached task scopes")

    model_id = plan.get("model_id")
    prompt_version = plan.get("judge_prompt_version")
    provider_config_sha256 = plan.get("provider_config_sha256")
    if (not isinstance(model_id, str) or not model_id
            or not isinstance(prompt_version, str) or not prompt_version
            or not isinstance(provider_config_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", provider_config_sha256)):
        raise ValueError("judge execution plan lacks frozen model/config identity")

    requests: dict[str, tuple[str, str]] = {}
    protocol_rows = []
    for task in cases:
        task_id = str(task["task_id"])
        docs = task_input_docs(task)
        system, user = calibration.proposition_judge_messages(
            task["question"], docs, prompt_version=prompt_version)
        requests[task_id] = (system, user)
        protocol_rows.append({
            "task_id": task_id,
            "gold_answer": str(task["options"][int(task["correct_index"])])
            .strip().casefold(),
            "system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
            "user_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(),
        })
    expected_protocol = sha256_obj({
        "schema": "task-input-grounding-calibration-protocol-v2",
        "dataset": "pubmedqa",
        "split": "dev_optimize",
        "task_inputs": protocol_rows,
        "model_id": model_id,
        "prompt_version": prompt_version,
        "provider_config_sha256": provider_config_sha256,
        "calibration_script_sha256": _current_judge_source_hashes()[
            "calibration_script_sha256"],
        "grounding_module_sha256": _current_judge_source_hashes()[
            "grounding_module_sha256"],
        "max_tokens": 450,
    })
    if plan.get("protocol_sha256") != expected_protocol:
        raise ValueError("judge execution plan does not match current task requests")
    expected_request_rows = [
        {"task_id": task_id,
         "system_sha256": hashlib.sha256(requests[task_id][0].encode("utf-8")).hexdigest(),
         "user_sha256": hashlib.sha256(requests[task_id][1].encode("utf-8")).hexdigest()}
        for task_id in sorted(pending)
    ]
    if plan.get("task_input_request_hashes") != expected_request_rows:
        raise ValueError("judge plan request hashes differ from the frozen task inputs")


def _validate_human_reference(packet_a: dict[str, Any], packet_b: dict[str, Any],
                              response_a: dict[str, Any], response_b: dict[str, Any],
                              human_report: dict[str, Any],
                              researcher_key: dict[str, Any]
                              ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    reproduced = score_human_reference(packet_a, packet_b, response_a, response_b)
    if (human_report.get("schema") != HUMAN_REPORT_SCHEMA
            or human_report.get("report_sha256") != reproduced["report_sha256"]):
        raise ValueError("human agreement report does not match the completed blind reviews")
    if researcher_key.get("schema") != "task-input-grounding-human-reference-researcher-key-v1":
        raise ValueError("researcher key schema is unsupported")
    if (researcher_key.get("protocol_sha256") != reproduced["protocol_sha256"]
            or researcher_key.get("reviewer_packet_sha256")
            != reproduced["reviewer_packet_sha256"]):
        raise ValueError("researcher key does not match the scored reviewer packets")
    if (packet_a.get("schema") != PACKET_SCHEMA
            or packet_a["protocol_context"].get("dataset") != "pubmedqa"
            or packet_a["protocol_context"].get("split") != "dev_optimize"):
        raise ValueError("human-reference packet is outside PubMedQA dev_optimize")

    context = packet_a["protocol_context"]
    bundle = build_reference_packets(
        per_label=context["per_label"],
        offset_per_label=context["offset_per_label"])
    if bundle["protocol_sha256"] != reproduced["protocol_sha256"]:
        raise ValueError("human-reference packet inputs or sources have changed")
    if researcher_key != bundle["researcher_key"]:
        raise ValueError("researcher key does not match the frozen visible cohort")
    cases_a = {case["case_id"]: case for case in packet_a["cases"]}
    cases_b = {case["case_id"]: case for case in packet_b["cases"]}
    if set(cases_a) != set(cases_b):
        raise ValueError("reviewer packets contain different case sets")
    for case_id, case in cases_a.items():
        if case != cases_b[case_id]:
            raise ValueError("reviewer packets differ in case inputs")
    key_rows = researcher_key.get("cases")
    if (not isinstance(key_rows, list)
            or len(key_rows) != context["per_label"] * 2
            or researcher_key.get("case_count") != len(key_rows)):
        raise ValueError("researcher key has an invalid case count")
    by_case_id: dict[str, dict[str, Any]] = {}
    seen_task_ids = set()
    for row in key_rows:
        if not isinstance(row, dict) or set(row) != {
                "case_id", "task_id", "benchmark_answer_label_for_post_review_only",
                "input_sha256"}:
            raise ValueError("researcher key has malformed case fields")
        case_id = row["case_id"]
        task_id = row["task_id"]
        if (not isinstance(case_id, str) or case_id in by_case_id
                or not isinstance(task_id, str) or not task_id or task_id in seen_task_ids
                or row["benchmark_answer_label_for_post_review_only"] not in {"yes", "no"}
                or not re.fullmatch(r"[0-9a-f]{64}", str(row["input_sha256"]))):
            raise ValueError("researcher key contains invalid or duplicate identities")
        if case_id != _case_id(reproduced["protocol_sha256"], task_id):
            raise ValueError("researcher key case ID does not match its task identity")
        if case_id not in cases_a or cases_a[case_id]["input_sha256"] != row["input_sha256"]:
            raise ValueError("researcher key input hash does not match reviewer packet")
        by_case_id[case_id] = row
        seen_task_ids.add(task_id)
    if set(by_case_id) != set(cases_a):
        raise ValueError("researcher key does not cover the reviewer case set")
    return reproduced, by_case_id


def _prediction_from_row(row: dict[str, Any], task: dict[str, Any],
                         docs: list[dict[str, Any]]) -> str:
    if set(row) != JUDGE_ROW_FIELDS:
        raise ValueError("judge calibration row has unexpected or missing fields")
    gold = str(task["options"][int(task["correct_index"])])
    gold = gold.strip().casefold()
    opposite = "no" if gold == "yes" else "yes"
    if row.get("gold_answer") != gold or row.get("opposite_answer") != opposite:
        raise ValueError("judge row labels differ from the current visible task")
    if row.get("evidence_chars") != sum(len(doc["text"]) for doc in docs):
        raise ValueError("judge row evidence size differs from the current task")
    for field in ("prompt_tokens", "completion_tokens", "evidence_chars"):
        if type(row.get(field)) is not int or row[field] < 0:
            raise ValueError("judge row has invalid usage metadata")
    if (not isinstance(row.get("raw_verdict"), str)
            or not isinstance(row.get("gold_reason"), str)
            or not isinstance(row.get("opposite_reason"), str)
            or type(row.get("gold_supported")) is not bool
            or type(row.get("opposite_supported")) is not bool):
        raise ValueError("judge row has malformed verdict fields")
    expected_gold = parse_proposition_verdict(row["raw_verdict"], docs, gold)
    expected_opposite = parse_proposition_verdict(row["raw_verdict"], docs, opposite)
    if (row["gold_supported"], row["gold_reason"]) != expected_gold:
        raise ValueError("judge gold-answer parse does not match the raw verdict")
    if (row["opposite_supported"], row["opposite_reason"]) != expected_opposite:
        raise ValueError("judge opposite-answer parse does not match the raw verdict")
    if row["gold_supported"] and row["opposite_supported"]:
        return "invalid_output"
    if row["gold_supported"]:
        return "supported" if gold == "yes" else "contradicted"
    if row["opposite_supported"]:
        return "contradicted" if gold == "yes" else "supported"
    if (row["gold_reason"] == "proposition_uncertain"
            and row["opposite_reason"] == "proposition_uncertain"):
        return "insufficient"
    return "invalid_output"


def score_human_reference_against_judge(
        packet_a: dict[str, Any], packet_b: dict[str, Any],
        response_a: dict[str, Any], response_b: dict[str, Any],
        human_report: dict[str, Any], researcher_key: dict[str, Any],
        judge_plan: dict[str, Any], judge_report: dict[str, Any],
        expected_judge_plan_sha256: str) -> dict[str, Any]:
    """Validate identities, then compare current judge output with human consensus."""
    human, key_by_case_id = _validate_human_reference(
        packet_a, packet_b, response_a, response_b, human_report, researcher_key)
    context = packet_a["protocol_context"]
    tasks = calibration.select_cases(context["per_label"], context["offset_per_label"])
    _verify_plan(judge_plan, expected_judge_plan_sha256, tasks)
    if (judge_report.get("status") != JUDGE_REPORT_STATUS
            or judge_report.get("dataset") != "pubmedqa"
            or judge_report.get("split") != "dev_optimize"
            or judge_report.get("plan_sha256") != judge_plan["plan_sha256"]
            or judge_report.get("protocol") != judge_plan["protocol_sha256"]
            or judge_report.get("model_id") != judge_plan["model_id"]
            or judge_report.get("judge_prompt_version")
            != judge_plan["judge_prompt_version"]):
        raise ValueError("judge report does not match the frozen execution plan")
    rows = judge_report.get("rows")
    task_ids = {row["task_id"] for row in key_by_case_id.values()}
    if (not isinstance(rows, list) or len(rows) != len(task_ids)
            or judge_report.get("n_tasks") != len(task_ids)):
        raise ValueError("judge report does not contain the complete reference cohort")
    judge_by_task_id = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("task_id") not in task_ids:
            raise ValueError("judge report contains an unknown task")
        if row["task_id"] in judge_by_task_id:
            raise ValueError("judge report contains a duplicate task")
        if row.get("protocol") != judge_plan["protocol_sha256"]:
            raise ValueError("judge row protocol does not match the frozen plan")
        judge_by_task_id[row["task_id"]] = row
    if set(judge_by_task_id) != task_ids:
        raise ValueError("judge report is missing tasks from the human-reference cohort")

    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    human_by_case_id = {row["case_id"]: row for row in human["per_case"]}
    confusion = {human_label: {judge_label: 0 for judge_label in JUDGE_LABELS}
                 for human_label in HUMAN_LABELS}
    per_case = []
    invalid_consensus = 0
    excluded_unable = 0
    excluded_disagreement = 0
    for case_id in sorted(human_by_case_id):
        human_row = human_by_case_id[case_id]
        key_row = key_by_case_id[case_id]
        consensus = human_row["consensus_verdict"]
        if consensus is None:
            excluded_disagreement += 1
            continue
        if consensus == "unable_to_judge":
            excluded_unable += 1
            continue
        if consensus not in HUMAN_LABELS:
            raise ValueError("human scorer emitted an unsupported consensus label")
        task_id = key_row["task_id"]
        task = tasks_by_id[task_id]
        docs = task_input_docs(task)
        judge_row = judge_by_task_id[task_id]
        prediction = _prediction_from_row(judge_row, task, docs)
        confusion[consensus][prediction] += 1
        per_case.append({
            "case_id": case_id,
            "task_id": task_id,
            "human_consensus": consensus,
            "judge_relation": prediction,
            "exact_match": prediction == consensus,
        })
        if prediction == "invalid_output":
            invalid_consensus += 1

    n = len(per_case)
    if not n:
        raise ValueError("no usable human consensus cases are available")
    correct = sum(row["exact_match"] for row in per_case)
    valid_predictions = n - invalid_consensus
    class_metrics = {}
    recalls = []
    for label in HUMAN_LABELS:
        support = sum(confusion[label].values())
        matched = confusion[label][label]
        recall = matched / support if support else None
        class_metrics[label] = {"n": support, "correct": matched, "recall": recall}
        if recall is not None:
            recalls.append(recall)
    unsigned = {
        "schema": "task-input-grounding-human-judge-comparison-v1",
        "status": "exploratory_semantic_calibration_comparison",
        "human_reference_protocol_sha256": human["protocol_sha256"],
        "human_agreement_report_sha256": human["report_sha256"],
        "reviewer_packet_sha256": human["reviewer_packet_sha256"],
        "judge_plan_sha256": judge_plan["plan_sha256"],
        "judge_protocol_sha256": judge_plan["protocol_sha256"],
        "judge_model_id": judge_plan["model_id"],
        "judge_prompt_version": judge_plan["judge_prompt_version"],
        "analysis_source_hashes": _comparison_source_hashes(),
        "dataset": "pubmedqa",
        "split": "dev_optimize",
        "n_human_agreement_cases": human["n_cases"] - human["n_disagreement_unresolved"],
        "n_human_disagreement_excluded": excluded_disagreement,
        "n_unable_to_judge_consensus_excluded": excluded_unable,
        "n_usable_consensus": n,
        "n_judge_invalid_output": invalid_consensus,
        "judge_valid_relation_coverage": valid_predictions / n,
        "accuracy_all_usable_consensus": correct / n,
        "accuracy_valid_predictions_only": (
            sum(row["exact_match"] for row in per_case
                if row["judge_relation"] != "invalid_output") / valid_predictions
            if valid_predictions else None),
        "balanced_accuracy_over_observed_human_classes": (
            sum(recalls) / len(recalls) if recalls else None),
        "human_class_metrics": class_metrics,
        "confusion_matrix_human_rows_judge_columns": confusion,
        "decision_policy": (
            "matching supported/contradicted/insufficient human consensus only; "
            "disagreement and unable_to_judge are excluded; invalid judge outputs "
            "count wrong in all-consensus accuracy; no threshold is tuned here"),
        "benchmark_labels_are_semantic_gold": False,
        "calibration_pass": None,
        "per_case": per_case,
    }
    return {**unsigned, "report_sha256": sha256_obj(unsigned)}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet-a", "packet-b", "responses-a", "responses-b",
                 "human-report", "researcher-key", "judge-plan", "judge-report",
                 "expected-judge-plan-sha256", "output"):
        parser.add_argument("--" + name, type=Path if name != "expected-judge-plan-sha256"
                            else str, required=True)
    args = parser.parse_args()
    report = score_human_reference_against_judge(
        _load(args.packet_a), _load(args.packet_b),
        _load(args.responses_a), _load(args.responses_b),
        _load(args.human_report), _load(args.researcher_key),
        _load(args.judge_plan), _load(args.judge_report),
        args.expected_judge_plan_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "n_usable_consensus", "accuracy_all_usable_consensus",
        "judge_valid_relation_coverage", "balanced_accuracy_over_observed_human_classes",
        "n_human_disagreement_excluded", "n_unable_to_judge_consensus_excluded",
        "report_sha256")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
