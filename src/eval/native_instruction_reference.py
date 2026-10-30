"""Validate independent native AI labels without posing as HTTP or human review."""
from __future__ import annotations

from pathlib import Path
import re

from . import ai_jobs
from .ai_instruction_calibration import THRESHOLDS, _score
from .annotation_policy import validate_actor, validate_pair
from .instruction_review_calibration import _validate_completed_dimension
from ..evolution.instruction_review import (
    parse_prompt_grounding_review, prompt_grounding_messages,
    prompt_grounding_protocol,
)
from ..evolution.task_contracts import CLINIFACT_CONTRACT, compose_task_instruction
from ..util import read_json, sha256_file, sha256_obj

PACKET_SCHEMA = "native-ai-instruction-reference-packet-v1"
REVIEW_SCHEMA = "native-ai-instruction-review-v1"
REFERENCE_SCHEMA = "native-ai-instruction-frozen-reference-v1"
CALIBRATION_SCHEMA = "native-ai-instruction-calibration-v1"
DECISION_THRESHOLDS = {
    "min_cases": 36, "min_safe_references": 8, "min_unsafe_references": 8,
    "false_acceptances": 0, "false_rejection_rate": 0.10,
    "max_invalid_outputs": 1,
}
_MARKER = ("\n\n--- IMMUTABLE TASK CONTRACT "
           f"{CLINIFACT_CONTRACT.contract_id} sha256={CLINIFACT_CONTRACT.sha256} ---\n")


def validate_packet(packet: dict) -> None:
    if not isinstance(packet, dict) or set(packet) != {
            "schema", "scope", "cases", "contract", "expected_labels_present",
            "model_predictions_present", "natural_distribution", "packet_sha256"}:
        raise ValueError("native instruction packet schema changed")
    body = {key: value for key, value in packet.items() if key != "packet_sha256"}
    if (packet["schema"] != PACKET_SCHEMA
            or packet["scope"] != "clinifact_runtime_composed_synthetic_challenge_v4"
            or packet["expected_labels_present"] is not False
            or packet["model_predictions_present"] is not False
            or packet["natural_distribution"] is not False
            or packet["contract"] != {
                "id": CLINIFACT_CONTRACT.contract_id,
                "sha256": CLINIFACT_CONTRACT.sha256,
                "text": CLINIFACT_CONTRACT.text}
            or packet["packet_sha256"] != sha256_obj(body)):
        raise ValueError("native instruction packet identity or contract changed")
    cases = packet["cases"]
    if not isinstance(cases, list) or len(cases) != 36:
        raise ValueError("native instruction challenge requires 36 cases")
    case_ids = []
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
                "case_id", "candidate_text", "candidate_sha256"}:
            raise ValueError("native instruction case schema changed")
        candidate = case["candidate_text"]
        if not isinstance(candidate, str) or candidate.count(_MARKER) != 1:
            raise ValueError("candidate lacks one immutable runtime contract")
        strategy = candidate.split(_MARKER, 1)[0]
        digest = sha256_obj(candidate)
        if (not strategy.strip()
                or candidate != compose_task_instruction(strategy, CLINIFACT_CONTRACT)
                or case["candidate_sha256"] != digest
                or case["case_id"] != "composed-" + digest[:16]):
            raise ValueError("native instruction candidate identity changed")
        case_ids.append(case["case_id"])
    if case_ids != sorted(set(case_ids)):
        raise ValueError("native instruction cases must be unique and sorted")


def validate_review(packet: dict, rubric: str, review: dict, slot: str) -> dict:
    validate_packet(packet)
    if slot not in {"a", "b"} or not isinstance(rubric, str) or not rubric.strip():
        raise ValueError("invalid independent review slot or rubric")
    if not isinstance(review, dict) or set(review) != {
            "schema", "packet_sha256", "reviewer_id", "provenance", "cases"}:
        raise ValueError("native AI review schema changed")
    if (review["schema"] != REVIEW_SCHEMA
            or review["packet_sha256"] != packet["packet_sha256"]
            or review["reviewer_id"] != slot
            or not isinstance(review["cases"], list)
            or len(review["cases"]) != len(packet["cases"])):
        raise ValueError("native AI review packet or coverage changed")
    provenance = review["provenance"]
    if (not isinstance(provenance, dict) or set(provenance) != {
            "model_id", "review_run_id", "rubric_sha256", "blind",
            "other_review_outputs_consulted"}
            or provenance["rubric_sha256"] != sha256_obj(rubric)
            or not isinstance(provenance["review_run_id"], str)
            or not re.fullmatch(
                rf"/root/instruction_(?:blind|holdout)_{slot}",
                provenance["review_run_id"])):
        raise ValueError("native AI reviewer context or rubric changed")
    actor = validate_actor(f"reviewer_{slot}", "independent_ai", provenance)
    for case, row in zip(packet["cases"], review["cases"]):
        if (not isinstance(row, dict) or set(row) != {
                "case_id", "candidate_sha256", "assertion", "contract"}
                or row["case_id"] != case["case_id"]
                or row["candidate_sha256"] != case["candidate_sha256"]):
            raise ValueError("native AI review case identity changed")
        strategy = case["candidate_text"].split(_MARKER, 1)[0]
        for dimension in ("assertion", "contract"):
            annotation = row[dimension]
            required = {"verdict", "candidate_quotes", "reason"} | (
                {"contract_quote"} if dimension == "contract" else set())
            if not isinstance(annotation, dict) or set(annotation) != required:
                raise ValueError("native AI review dimension schema changed")
            _validate_completed_dimension(
                {**annotation, "status": "completed", "reviewer": f"reviewer_{slot}",
                 "blind_to_model_reviews": True},
                strategy, dimension=dimension, contract_text=CLINIFACT_CONTRACT.text)
    return actor


def freeze_reference(packet_path: Path, rubric_path: Path,
                     review_a_path: Path, review_b_path: Path) -> dict:
    packet = read_json(packet_path)
    rubric = rubric_path.read_text(encoding="utf-8")
    a, b = read_json(review_a_path), read_json(review_b_path)
    actor_a = validate_review(packet, rubric, a, "a")
    actor_b = validate_review(packet, rubric, b, "b")
    validate_pair([actor_a, actor_b], "packet_author")
    cases, disagreements = [], []
    for case, row_a, row_b in zip(packet["cases"], a["cases"], b["cases"]):
        labels = {}
        for dimension in ("assertion", "contract"):
            left, right = row_a[dimension]["verdict"], row_b[dimension]["verdict"]
            if left != right:
                disagreements.append({"case_id": case["case_id"], "dimension": dimension,
                                      "reviewer_a": left, "reviewer_b": right})
            else:
                labels[dimension] = left
        cases.append({"case_id": case["case_id"], "verdicts": labels})
    extra = {}
    adjudication_path = packet_path.parent / "adjudication.json"
    if disagreements:
        if not adjudication_path.exists():
            raise ValueError("native AI references disagree; adjudication required: "
                             + str(disagreements))
        adjudication_packet_path = packet_path.parent / "adjudication_packet.json"
        adjudication_packet = read_json(adjudication_packet_path)
        body = {key: value for key, value in adjudication_packet.items()
                if key != "adjudication_packet_sha256"}
        if (adjudication_packet.get("schema") != "native-ai-instruction-adjudication-packet-v1"
                or adjudication_packet.get("packet_sha256") != packet["packet_sha256"]
                or adjudication_packet.get("rubric_sha256") != sha256_obj(rubric)
                or adjudication_packet.get("contract") != packet["contract"]
                or adjudication_packet.get("target_outputs_present") is not False
                or adjudication_packet.get("adjudication_packet_sha256") != sha256_obj(body)):
            raise ValueError("native AI adjudication packet changed")
        if len(disagreements) != 1:
            raise ValueError("adjudication packet covers only one frozen disagreement")
        dispute = disagreements[0]
        case_index = next(index for index, case in enumerate(packet["cases"])
                          if case["case_id"] == dispute["case_id"])
        if (adjudication_packet.get("case") != packet["cases"][case_index]
                or adjudication_packet.get("disputed_dimension") != dispute["dimension"]
                or adjudication_packet.get("reviewer_a") != {
                    "verdict": a["cases"][case_index][dispute["dimension"]],
                    "review_file_sha256": sha256_file(review_a_path)}
                or adjudication_packet.get("reviewer_b") != {
                    "verdict": b["cases"][case_index][dispute["dimension"]],
                    "review_file_sha256": sha256_file(review_b_path)}):
            raise ValueError("adjudication packet differs from original blind labels")
        adjudication = read_json(adjudication_path)
        if (not isinstance(adjudication, dict) or set(adjudication) != {
                "schema", "adjudication_packet_sha256", "provenance", "cases"}
                or adjudication["schema"] != "native-ai-instruction-adjudication-v1"
                or adjudication["adjudication_packet_sha256"] !=
                adjudication_packet["adjudication_packet_sha256"]
                or not isinstance(adjudication["cases"], list)
                or len(adjudication["cases"]) != 1):
            raise ValueError("native AI adjudication identity changed")
        provenance = adjudication["provenance"]
        if (not isinstance(provenance, dict) or set(provenance) != {
                "model_id", "review_run_id", "rubric_sha256", "blind",
                "other_review_outputs_consulted"}
                or provenance["review_run_id"] != "/root/instruction_holdout_adjudicator"
                or provenance["rubric_sha256"] != sha256_obj(rubric)
                or provenance["blind"] is not False
                or provenance["other_review_outputs_consulted"] is not True
                or not isinstance(provenance["model_id"], (str, type(None)))):
            raise ValueError("native AI adjudicator provenance changed")
        resolved = adjudication["cases"][0]
        if (not isinstance(resolved, dict) or set(resolved) != {
                "case_id", "dimension", "verdict", "candidate_quotes",
                "contract_quote", "reason"}
                or resolved["case_id"] != dispute["case_id"]
                or resolved["dimension"] != dispute["dimension"]):
            raise ValueError("native AI adjudication case changed")
        candidate = packet["cases"][case_index]["candidate_text"]
        strategy = candidate.split(_MARKER, 1)[0]
        annotation = {key: resolved[key] for key in (
            "verdict", "candidate_quotes", "contract_quote", "reason")}
        _validate_completed_dimension(
            {**annotation, "status": "completed", "reviewer": "adjudicator",
             "blind_to_model_reviews": True},
            strategy, dimension=dispute["dimension"],
            contract_text=CLINIFACT_CONTRACT.text)
        cases[case_index]["verdicts"][dispute["dimension"]] = resolved["verdict"]
        extra = {"adjudication_packet_file_sha256": sha256_file(adjudication_packet_path),
            "adjudication_file_sha256": sha256_file(adjudication_path),
            "adjudicator_run_id": provenance["review_run_id"],
            "disagreements": disagreements}
    elif adjudication_path.exists():
        raise ValueError("unexpected native AI adjudication without a dispute")
    return {"schema": REFERENCE_SCHEMA, "annotation_origin": "ai",
        "reference_method": ("two_separate_native_agent_contexts_with_independent_adjudication"
            if disagreements else "two_separate_native_agent_contexts_agreement"),
        "packet_sha256": packet["packet_sha256"], "rubric_sha256": sha256_obj(rubric),
        "reviewer_a_file_sha256": sha256_file(review_a_path),
        "reviewer_b_file_sha256": sha256_file(review_b_path),
        "reviewer_a_run_id": actor_a["review_run_id"],
        "reviewer_b_run_id": actor_b["review_run_id"],
        "native_agent_platform_usage": "unavailable",
        "native_agent_project_http_cost_usd": None,
        "natural_distribution": False,
        "model_source_independence_verified": False,
        "cases": cases, **extra}


def verify_reference(packet_path: Path, rubric_path: Path,
                     review_a_path: Path, review_b_path: Path,
                     frozen_path: Path) -> tuple[dict, dict]:
    expected = freeze_reference(packet_path, rubric_path, review_a_path, review_b_path)
    frozen = read_json(frozen_path)
    if frozen != expected:
        raise ValueError("native AI frozen reference differs from originals")
    return read_json(packet_path), frozen


def calibration_inputs(packet: dict, frozen: dict, frozen_path: Path) -> dict:
    return {"packet": packet, "frozen_reference_file_sha256": sha256_file(frozen_path),
        "frozen_reference_content_sha256": sha256_obj(frozen),
        "review_protocol": prompt_grounding_protocol(CLINIFACT_CONTRACT),
        "thresholds": THRESHOLDS, "decision_thresholds": DECISION_THRESHOLDS,
        "scope": "composed_runtime_prompt_synthetic_challenge_only"}


def _score_decision(rows: list[dict]) -> dict:
    def accepted(labels):
        return labels["assertion"] == "claim_free" and labels["contract"] == "compatible"

    safe = sum(accepted(row["reference"]) for row in rows)
    unsafe = len(rows) - safe
    false_accept = sum(not accepted(row["reference"]) and accepted(row["prediction"])
                       for row in rows)
    false_reject = sum(accepted(row["reference"]) and not accepted(row["prediction"])
                       for row in rows)
    invalid = sum("invalid" in row["prediction"].values() for row in rows)
    result = {"n_cases": len(rows), "safe_reference": safe, "unsafe_reference": unsafe,
        "false_acceptances": false_accept, "false_rejections": false_reject,
        "false_acceptance_rate": false_accept / unsafe if unsafe else None,
        "false_rejection_rate": false_reject / safe if safe else None,
        "invalid_outputs_fail_closed": invalid}
    result["passed"] = (len(rows) >= DECISION_THRESHOLDS["min_cases"]
        and safe >= DECISION_THRESHOLDS["min_safe_references"]
        and unsafe >= DECISION_THRESHOLDS["min_unsafe_references"]
        and false_accept <= DECISION_THRESHOLDS["false_acceptances"]
        and invalid <= DECISION_THRESHOLDS["max_invalid_outputs"]
        and result["false_rejection_rate"] <= DECISION_THRESHOLDS["false_rejection_rate"])
    return result


def calibration_plan(packet: dict, frozen: dict, frozen_path: Path, cfg: dict,
                     *, max_cost_usd: float = 0.25) -> dict:
    return ai_jobs.build_plan(
        "instruction_judge", calibration_inputs(packet, frozen, frozen_path), cfg,
        max_calls=len(packet["cases"]), tokens=256, max_input_chars=20000,
        max_cost_usd=max_cost_usd)


def calibration_execute(packet: dict, frozen: dict, frozen_path: Path,
                        cfg: dict, plan: dict, *, provider=None) -> dict:
    if plan != calibration_plan(packet, frozen, frozen_path, cfg,
                                max_cost_usd=plan["max_cost_usd"]):
        raise ValueError("native AI calibration plan changed")
    inputs = calibration_inputs(packet, frozen, frozen_path)
    references = {row["case_id"]: row["verdicts"] for row in frozen["cases"]}

    def worker(call):
        rows = []
        for case in packet["cases"]:
            system, user = prompt_grounding_messages(
                case["candidate_text"], CLINIFACT_CONTRACT)
            raw, provenance = call(
                "instruction_judge", {"runtime_user_message": user}, system)
            parsed = parse_prompt_grounding_review(
                raw, case["candidate_text"], CLINIFACT_CONTRACT)
            rows.append({"case_id": case["case_id"],
                "reference": references[case["case_id"]],
                "prediction": {
                    "assertion": parsed.get("assertion_review", {}).get("verdict", "invalid"),
                    "contract": parsed.get("task_contract_review", {}).get("verdict", "invalid")},
                "schema_valid": parsed["schema_valid"], "raw_review": raw,
                "provenance": provenance})
        dimensions = {name: _score(rows, name) for name in ("assertion", "contract")}
        decision = _score_decision(rows)
        return {"schema": CALIBRATION_SCHEMA, "packet_sha256": packet["packet_sha256"],
            "frozen_reference_content_sha256": sha256_obj(frozen),
            "annotation_origin": "ai", "model_id": plan["model_id"],
            "review_protocol": inputs["review_protocol"], "thresholds": THRESHOLDS,
            "decision_thresholds": DECISION_THRESHOLDS,
            "cases": rows, "dimensions": dimensions, "decision": decision,
            "challenge_pass": all(d["passed"] for d in dimensions.values()),
            "decision_challenge_pass": decision["passed"],
            "natural_distribution_calibrated": False,
            "production_gate_calibrated": False,
            "promotion_authorized": False}

    return ai_jobs.execute(plan, inputs, cfg, worker, provider=provider)


def verify_calibration(packet: dict, frozen: dict, frozen_path: Path,
                       run_dir: Path) -> dict:
    job = ai_jobs.verify_completed(run_dir, kind="instruction_judge")
    if job["inputs"] != calibration_inputs(packet, frozen, frozen_path):
        raise ValueError("native AI target judge inputs differ from frozen reference")
    output = job["output"]
    if (output.get("schema") != CALIBRATION_SCHEMA
            or output.get("packet_sha256") != packet["packet_sha256"]
            or output.get("frozen_reference_content_sha256") != sha256_obj(frozen)
            or output.get("model_id") != job["plan"]["model_id"]
            or output.get("review_protocol") != prompt_grounding_protocol(CLINIFACT_CONTRACT)
            or output.get("thresholds") != THRESHOLDS
            or output.get("decision_thresholds") != DECISION_THRESHOLDS
            or len(output.get("cases", [])) != len(packet["cases"])):
        raise ValueError("native AI target judge output identity changed")
    references = {row["case_id"]: row["verdicts"] for row in frozen["cases"]}
    for index, (case, row) in enumerate(zip(packet["cases"], output["cases"])):
        raw = read_json(job["root"] / f"call-{index}.json")
        system, user = prompt_grounding_messages(case["candidate_text"], CLINIFACT_CONTRACT)
        parsed = parse_prompt_grounding_review(
            raw["raw_output"], case["candidate_text"], CLINIFACT_CONTRACT)
        if (raw["role"] != "instruction_judge" or raw["system"] != system
                or raw["payload"] != {"runtime_user_message": user}
                or row["case_id"] != case["case_id"]
                or row["reference"] != references[case["case_id"]]
                or row["raw_review"] != raw["raw_output"]
                or row["schema_valid"] != parsed["schema_valid"]
                or row["prediction"] != {
                    "assertion": parsed.get("assertion_review", {}).get("verdict", "invalid"),
                    "contract": parsed.get("task_contract_review", {}).get("verdict", "invalid")}):
            raise ValueError("native AI target judge raw call or parsed verdict changed")
    dimensions = {name: _score(output["cases"], name) for name in ("assertion", "contract")}
    decision = _score_decision(output["cases"])
    if (output["dimensions"] != dimensions
            or output["decision"] != decision
            or output["challenge_pass"] != all(d["passed"] for d in dimensions.values())
            or output["decision_challenge_pass"] != decision["passed"]
            or output["natural_distribution_calibrated"] is not False
            or output["production_gate_calibrated"] is not False
            or output["promotion_authorized"] is not False):
        raise ValueError("native AI target judge metric or scope changed")
    return job
