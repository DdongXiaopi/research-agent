"""Audit a blinded third-AI adjudication against two HealthVer AI references.

This produces an AI-reference sensitivity report only. It does not establish a
human gold standard, independent generalization, or permission to promote a
grounding judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

LABELS = ("supported", "contradicted", "insufficient")
DEFAULT_ROOT = Path(".workbuddy/healthver-grounding-reference-v5")


def _read(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _canonical_sha(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "report_sha256"}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    matrix = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}
    for row in rows:
        matrix[row["reference"]][row["prediction"]] += 1
    f1: dict[str, float] = {}
    recall: dict[str, float] = {}
    for label in LABELS:
        tp = matrix[label][label]
        actual = sum(matrix[label].values())
        predicted = sum(matrix[other][label] for other in LABELS)
        recall[label] = tp / actual if actual else 0.0
        denom = 2 * tp + (predicted - tp) + (actual - tp)
        f1[label] = 2 * tp / denom if denom else 0.0
    non_supported = sum(sum(matrix[label].values()) for label in LABELS if label != "supported")
    false_support = sum(matrix[label]["supported"] for label in LABELS if label != "supported")
    return {
        "n": len(rows),
        "label_counts": dict(Counter(row["reference"] for row in rows)),
        "confusion_matrix_actual_by_prediction": matrix,
        "macro_f1": sum(f1.values()) / len(LABELS),
        "per_class_f1": f1,
        "per_class_recall": recall,
        "false_support_rate_among_non_supported": false_support / non_supported if non_supported else None,
    }


def build_report(root: Path) -> dict[str, Any]:
    files = {
        "packet": "blind-cross-model-adjudication-packet-v1.json",
        "adjudication": "blind-cross-model-adjudication-agent-v1.json",
        "pairwise_agreement": "cross_model_agreement_qwen32_deepseek_v32.json",
        "reviewer_a": "reviewer_a_qwen32_executed_responses.json",
        "reviewer_b": "reviewer_b_deepseek_v32_responses.json",
        "judge_8b": "v13_calibration_result.json",
        "judge_32b": "v13_calibration_result-32b.json",
        "judge_122b": "v13-calibration-result-122b-diagnostic.json",
    }
    loaded = {key: _read(root / name) for key, name in files.items()}
    data = {key: item[0] for key, item in loaded.items()}
    hashes = {key: item[1] for key, item in loaded.items()}

    packet = data["packet"]
    adjudication = data["adjudication"]
    if packet.get("schema") != "blind-cross-model-grounding-adjudication-v1":
        raise ValueError("unexpected packet schema")
    packet_hash = hashes["packet"]
    if adjudication.get("packet_sha256", "").lower() != packet_hash:
        raise ValueError("adjudication packet SHA-256 mismatch")
    if adjudication.get("annotation_origin") != "ai":
        raise ValueError("adjudication origin must be recorded as AI")

    packet_cases = {item["case_id"]: item for item in packet["cases"]}
    adjudications = {item["case_id"]: item for item in adjudication["annotations"]}
    pair_rows = {item["case_id"]: item for item in data["pairwise_agreement"]["per_case"]}
    dispute_ids = {
        case_id
        for case_id, item in pair_rows.items()
        if item["reviewer_a_verdict"] != item["reviewer_b_verdict"]
    }
    if set(adjudications) != dispute_ids:
        raise ValueError("adjudication coverage does not exactly match pairwise disagreements")
    if set(packet_cases) != set(adjudications):
        raise ValueError("blind packet coverage does not exactly match disagreements")

    vote_rows: list[dict[str, Any]] = []
    resolved: dict[str, str] = {}
    for case_id in sorted(packet_cases):
        case = packet_cases[case_id]
        annotation = adjudications[case_id]
        pair = pair_rows[case_id]
        if not (case["input_sha256"] == annotation["input_sha256"] == pair["input_sha256"]):
            raise ValueError(f"input hash mismatch for {case_id}")
        if annotation.get("verdict") not in (*LABELS, "unable_to_judge"):
            raise ValueError(f"invalid adjudication label for {case_id}")
        quote = annotation.get("supporting_quote", "")
        quote_doc_id = annotation.get("quote_doc_id", "")
        evidence = {doc["doc_id"]: doc["text"] for doc in case["evidence"]}
        if quote:
            if quote_doc_id not in evidence or quote not in evidence[quote_doc_id]:
                raise ValueError(f"quote does not match supplied evidence for {case_id}")
        elif quote_doc_id:
            raise ValueError(f"quote_doc_id supplied without quote for {case_id}")
        if not annotation.get("reason", "").strip():
            raise ValueError(f"missing adjudication rationale for {case_id}")

        votes = [pair["reviewer_a_verdict"], pair["reviewer_b_verdict"], annotation["verdict"]]
        usable_votes = [vote for vote in votes if vote in LABELS]
        counts = Counter(usable_votes)
        winning = [label for label, count in counts.items() if count >= 2]
        verdict = winning[0] if len(winning) == 1 else None
        if verdict:
            resolved[case_id] = verdict
        vote_rows.append({
            "case_id": case_id,
            "input_sha256": case["input_sha256"],
            "reviewer_a": votes[0],
            "reviewer_b": votes[1],
            "blind_adjudicator": votes[2],
            "resolved_by_three_ai_votes": verdict,
            "adjudicator_quote_verified": bool(quote),
            "adjudicator_reason": annotation["reason"],
        })

    consensus_rows = []
    for case_id, pair in pair_rows.items():
        a, b = pair["reviewer_a_verdict"], pair["reviewer_b_verdict"]
        if a == b and a in LABELS:
            consensus_rows.append({"case_id": case_id, "reference": a, "source": "two_model_agreement"})
        elif case_id in resolved:
            consensus_rows.append({"case_id": case_id, "reference": resolved[case_id], "source": "three_ai_majority"})

    judge_results: dict[str, Any] = {}
    for name in ("judge_8b", "judge_32b", "judge_122b"):
        calibration = data[name]["calibration"]
        predictions = {row["case_id"]: row for row in calibration["per_case"]}
        for case_id, prediction in predictions.items():
            if case_id in pair_rows and prediction.get("input_sha256") != pair_rows[case_id]["input_sha256"]:
                raise ValueError(f"judge input hash mismatch for {name}/{case_id}")
        evaluated = [
            {"reference": row["reference"], "prediction": predictions[row["case_id"]]["prediction"]}
            for row in consensus_rows
            if row["case_id"] in predictions and predictions[row["case_id"]]["prediction"] in LABELS
        ]
        judge_results[name.removeprefix("judge_")] = _metrics(evaluated)

    pair_agreement = data["pairwise_agreement"]
    report: dict[str, Any] = {
        "schema": "healthver-three-ai-reference-audit-v1",
        "annotation_origin": "ai",
        "interpretation": "internal non-commercial AI-reference sensitivity analysis; not human gold, not an independent cohort, and not a promotion/calibration pass",
        "upstream_use_scope": "user-authorized non-commercial internal research only; source rights remain unverified; do not redistribute source data or use commercially",
        "reviewers": {
            "reviewer_a": data["reviewer_a"]["reviewer_id"],
            "reviewer_b": data["reviewer_b"]["reviewer_id"],
            "blind_adjudicator": adjudication["reviewer"]["identity"],
            "blind_adjudicator_context": adjudication["reviewer"]["context"],
        },
        "source_sha256": hashes,
        "packet_sha256": packet_hash,
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "adjudication_sha256": hashes["adjudication"],
        "pairwise_summary": {
            "n_cases": pair_agreement["n_cases"],
            "exact_agreement": pair_agreement["exact_agreement_rate"],
            "cohen_kappa": pair_agreement["cohen_kappa_nominal"],
            "n_pair_agreement_with_label": sum(
                row["reviewer_a_verdict"] == row["reviewer_b_verdict"]
                and row["reviewer_a_verdict"] in LABELS
                for row in pair_rows.values()
            ),
            "n_disagreements_adjudicated": len(vote_rows),
        },
        "adjudication_summary": {
            "n_three_ai_majorities_on_disagreements": len(resolved),
            "n_unresolved_disagreements": len(vote_rows) - len(resolved),
            "label_distribution_all_resolved_cases": dict(Counter(row["reference"] for row in consensus_rows)),
            "n_all_resolved_cases": len(consensus_rows),
        },
        "judge_sensitivity_on_ai_consensus_subset": judge_results,
        "promotion_authorized": False,
        "per_disagreement": vote_rows,
    }
    report["report_sha256"] = _canonical_sha(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
