from __future__ import annotations

import json

import pytest

from src.eval import natural_instruction_adjudication as adjudication
from src.eval import natural_instruction_calibration as natural
from src.util import sha256_obj, write_json


def _case(cid: str) -> dict:
    return {"case_id": cid, "contract": {"id": "c", "text": "contract", "sha256": "x"},
            "strategy_text": f"strategy {cid}", "candidate_text": f"strategy {cid}",
            "candidate_sha256": "y", "domain": "clinifact", "origins": [{
                "path": "x", "file_sha256": "z", "location": "x",
                "raw_candidate_sha256": "q"}]}


def _jobs(packet):
    rows_a, rows_b = [], []
    for i, case in enumerate(packet["cases"]):
        if i % 2 == 0:
            a, b = ("claim_free", "compatible"), ("contains_claims", "incompatible")
        else:
            a, b = ("contains_claims", "incompatible"), ("claim_free", "compatible")
        rows_a.append({"case_id": case["case_id"], "dimensions": {
            "assertion": {"verdict": a[0]}, "contract": {"verdict": a[1]}}})
        rows_b.append({"case_id": case["case_id"], "dimensions": {
            "assertion": {"verdict": b[0]}, "contract": {"verdict": b[1]}}})
    base = {"plan": {"model_id": "THUDM/GLM-4-9B-0414", "plan_sha256": "p"},
            "receipt_file_sha256": "a", "receipt": {"test_only": False},
            "output": {"packet_sha256": packet["packet_sha256"], "cases": rows_a},
            "root": object()}
    other = {"plan": {"model_id": "Qwen/Qwen2.5-7B-Instruct", "plan_sha256": "q"},
             "receipt_file_sha256": "b", "receipt": {"test_only": False},
             "output": {"packet_sha256": packet["packet_sha256"], "cases": rows_b},
             "root": object()}
    return base, other, {r["case_id"]: r for r in rows_a}, {r["case_id"]: r for r in rows_b}


@pytest.fixture
def packet():
    body = {"schema": natural.PACKET_SCHEMA, "packet_sha256": "packet", "cases":
            [_case(f"c{i}") for i in range(6)]}
    return body


def test_dispute_index_is_case_balanced(packet):
    a, b, by_a, by_b = _jobs(packet)
    rows = adjudication._dispute_cases(packet, by_a, by_b)
    assert len(rows) == 6
    assert sum(len(row["disputed_dimensions"]) for row in rows) == 12


def test_plan_uses_only_disputed_cases_and_zero_price(monkeypatch, packet):
    a, b, by_a, by_b = _jobs(packet)
    monkeypatch.setattr(natural, "validate_packet", lambda _: None)
    monkeypatch.setattr(adjudication, "_pair", lambda *args, **kwargs: (a, b, by_a, by_b))

    def fake_build(kind, inputs, cfg, **kwargs):
        assert kind == "instruction_reference"
        assert kwargs["max_calls"] == 6
        assert kwargs["zero_price_only"] is True
        body = {"schema": "ai-evaluation-plan-v1", "kind": kind,
                "inputs_sha256": sha256_obj(inputs), "max_physical_attempts": 6,
                "max_cost_usd": 0.0, "model_id": "XingChenAGI/Xing4.0-29B",
                "zero_price_only": True,
                "plan_sha256": "job"}
        return body

    monkeypatch.setattr(adjudication.ai_jobs, "build_plan", fake_build)
    cfg = {"models": {"task": {"model_id": "XingChenAGI/Xing4.0-29B"}}}
    plan = adjudication.build_adjudication_plan(packet, None, None, cfg)
    assert plan["disputed_case_count"] == 6
    assert plan["disputed_dimension_count"] == 12
    assert plan["zero_price_only"] is True
    assert plan["source_runs_immutable"] is True


def test_plan_rejects_reviewer_family_reuse(monkeypatch, packet):
    a, b, by_a, by_b = _jobs(packet)
    monkeypatch.setattr(natural, "validate_packet", lambda _: None)
    monkeypatch.setattr(adjudication, "_pair", lambda *args, **kwargs: (a, b, by_a, by_b))
    with pytest.raises(ValueError, match="third model ID and family"):
        adjudication.build_adjudication_plan(
            packet, None, None,
            {"models": {"task": {"model_id": "Qwen/Qwen3-8B"}}})


def test_freeze_v2_uses_adjudicator_for_disputed_dimensions(monkeypatch, packet):
    a, b, by_a, by_b = _jobs(packet)
    monkeypatch.setattr(adjudication, "_pair", lambda *args, **kwargs: (a, b, by_a, by_b))
    adjudicator_rows = []
    labels = [("claim_free", "compatible"), ("contains_claims", "incompatible"),
              ("claim_free", "compatible"), ("contains_claims", "incompatible"),
              ("claim_free", "compatible"), ("contains_claims", "incompatible")]
    for case, (assertion, contract) in zip(packet["cases"], labels):
        adjudicator_rows.append({"case_id": case["case_id"],
            "disputed_dimensions": ["assertion", "contract"],
            "dimensions": {"assertion": {"verdict": assertion},
                            "contract": {"verdict": contract}},
            "normalization_issues": []})
    monkeypatch.setattr(adjudication, "verify_adjudication_run", lambda *args, **kwargs: {
        "output": {"cases": adjudicator_rows},
        "receipt_file_sha256": "r", "receipt": {"test_only": False}})
    plan = {"schema": adjudication.PLAN_SCHEMA, "adjudicator": {
        "model_id": "XingChenAGI/Xing4.0-29B", "model_family": "xing"},
        "ai_job_plan": {"max_cost_usd": 0.0}, "plan_sha256": "plan"}
    # The freeze function validates the plan hash before consuming rows.
    plan["plan_sha256"] = sha256_obj({k: v for k, v in plan.items() if k != "plan_sha256"})
    frozen = adjudication.freeze_reference_v2(packet, None, None, None, plan)
    assert frozen["schema"] == adjudication.FROZEN_SCHEMA
    assert frozen["natural_distribution"] is True
    assert frozen["reference_class_coverage_sufficient"] is True
    assert frozen["reference_class_coverage"]["assertion"]["approval_reference"] == 3
    assert frozen["reference_class_coverage"]["assertion"]["rejection_reference"] == 3
    assert frozen["limitations"]


def test_adjudication_verifier_binds_actor_to_job_and_raw_call(monkeypatch, packet, tmp_path):
    packet = {**packet, "cases": packet["cases"][:1]}
    a, b, by_a, by_b = _jobs(packet)
    monkeypatch.setattr(adjudication, "_pair", lambda *args, **kwargs: (a, b, by_a, by_b))
    disputes = adjudication._dispute_cases(packet, by_a, by_b)
    model = "XingChenAGI/Xing4.0-29B"
    inputs = adjudication._inputs(packet, a, b, disputes, model)
    job_plan = {"model_id": model}
    plan = {"schema": adjudication.PLAN_SCHEMA, "adjudicator": {"model_id": model}, "ai_job_plan": job_plan}
    plan["plan_sha256"] = sha256_obj(plan)
    dims = {"assertion": {"verdict": "claim_free"}, "contract": {"verdict": "compatible"}}
    monkeypatch.setattr(natural, "_reference_dimensions", lambda raw, case: (dims, []))
    case, dispute = packet["cases"][0], disputes[0]
    write_json(tmp_path / "call-0.json", {"role": "instruction_adjudicator", "system": adjudication.ADJUDICATION_RUBRIC,
               "payload": adjudication._request(case, dispute), "raw_output": "fixture"})
    actor = {"model_id": model, "review_run_id": tmp_path.name + "-0",
             "rubric_sha256": sha256_obj(adjudication.ADJUDICATION_RUBRIC), "blind": False,
             "other_review_outputs_consulted": True, "test_only": True,
             "origin": "independent_ai", "identity": "adjudicator"}
    output = {"schema": adjudication.RUN_SCHEMA, "packet_sha256": packet["packet_sha256"],
              "inputs_sha256": sha256_obj(inputs), "adjudication_rubric_sha256": sha256_obj(adjudication.ADJUDICATION_RUBRIC),
              "cases": [{"case_id": case["case_id"], "dimensions": dims, "normalization_issues": [],
                         "disputed_dimensions": dispute["disputed_dimensions"], "actor": actor}]}
    job = {"root": tmp_path, "plan": job_plan, "receipt": {"test_only": True}, "output": output}
    monkeypatch.setattr(adjudication.ai_jobs, "verify_completed", lambda *args, **kwargs: job)
    assert adjudication.verify_adjudication_run(packet, None, None, plan, tmp_path) is job
    output["cases"][0]["actor"] = {**actor, "model_id": "other-model"}
    with pytest.raises(ValueError, match="identity"):
        adjudication.verify_adjudication_run(packet, None, None, plan, tmp_path)


def test_natural_target_inputs_accept_v2_reference(monkeypatch, packet):
    frozen = {"schema": adjudication.FROZEN_SCHEMA,
              "packet_sha256": packet["packet_sha256"]}
    monkeypatch.setattr(natural, "validate_packet", lambda _: None)
    monkeypatch.setattr(natural, "prompt_grounding_protocol", lambda contract: {"version": "test"})
    inputs = natural.calibration_inputs(packet, frozen)
    assert inputs["packet"] is packet
    assert inputs["frozen_reference_sha256"] == sha256_obj(frozen)

