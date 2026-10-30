"""Contract-only top-up for the natural instruction reference.

The top-up is a fresh, immutable lineage over exactly the three historical
contract disagreements where reviewer A said ``compatible`` and reviewer B
said ``incompatible``.  It does not copy either label; a third Xing call must
produce the replacement verdict.  A failed top-up remains diagnostic and is
never converted into a class label.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import ai_jobs
from . import natural_instruction_adjudication as base
from . import natural_instruction_calibration as natural
from .instruction_reviewer_pair_gate import model_family
from ..util import sha256_obj


PLAN_SCHEMA = "natural-instruction-contract-topup-plan-v1"
RUN_SCHEMA = "natural-instruction-contract-topup-run-v1"
FROZEN_SCHEMA = "natural-instruction-frozen-reference-v3"
TARGET_CASE_COUNT = 3

CONTRACT_RUBRIC = (
    "Judge only whether the untrusted candidate strategy is compatible with the "
    "supplied task contract. Do not judge assertion presence. Mark incompatible "
    "when it changes the contract's label meanings, imports an external standard "
    "or evidence requirement, or adds a decision criterion not in the contract. "
    "Otherwise mark compatible; preserve uncertain if the text is insufficient. "
    "Return exactly one JSON object with verdict (compatible, incompatible, or "
    "uncertain), candidate_quotes (exact contiguous excerpts), contract_quote "
    "(exact contract excerpt for incompatible, otherwise null), and a nonempty "
    "reason. Incompatible requires at least one exact candidate quote and one "
    "exact contract quote. Do not copy either prior verdict."
)

CONTRACT_FORMAT = {"type": "json_schema", "json_schema": {
    "name": "natural_instruction_contract_adjudication", "strict": True,
    "schema": {"type": "object", "additionalProperties": False,
        "required": ["verdict", "candidate_quotes", "contract_quote", "reason"],
        "properties": {
            "verdict": {"type": "string",
                        "enum": ["compatible", "incompatible", "uncertain"]},
            "candidate_quotes": {"type": "array", "items": {"type": "string"}},
            "contract_quote": {"type": ["string", "null"]},
            "reason": {"type": "string", "minLength": 1},
        }}}}


def _pair(packet: dict, run_a: Path, run_b: Path):
    a, b, by_a, by_b = base._pair(packet, run_a, run_b)
    selected = []
    for case in packet["cases"]:
        cid = case["case_id"]
        av = by_a[cid]["dimensions"]["contract"]["verdict"]
        bv = by_b[cid]["dimensions"]["contract"]["verdict"]
        if av == "compatible" and bv == "incompatible":
            selected.append({"case_id": cid, "reviewer_a": by_a[cid]["dimensions"]["contract"],
                             "reviewer_b": by_b[cid]["dimensions"]["contract"]})
    if len(selected) != TARGET_CASE_COUNT:
        raise ValueError("contract top-up requires exactly three A-compatible/B-incompatible disputes")
    return a, b, selected


def _inputs(packet: dict, a: dict, b: dict, selected: list[dict], model_id: str) -> dict:
    return {"schema": "natural-instruction-contract-topup-inputs-v1",
        "packet_sha256": packet["packet_sha256"],
        "reviewer_a": {"model_id": a["plan"]["model_id"],
            "plan_sha256": a["plan"]["plan_sha256"], "receipt_sha256": a["receipt_file_sha256"],
            "output_sha256": sha256_obj(a["output"])},
        "reviewer_b": {"model_id": b["plan"]["model_id"],
            "plan_sha256": b["plan"]["plan_sha256"], "receipt_sha256": b["receipt_file_sha256"],
            "output_sha256": sha256_obj(b["output"])},
        "topup_model": {"model_id": model_id, "model_family": model_family(model_id)},
        "rubric_sha256": sha256_obj(CONTRACT_RUBRIC),
        "selected_case_ids": [row["case_id"] for row in selected]}


def build_plan(packet: dict, run_a: Path, run_b: Path, cfg: dict,
               *, model_id: str = "XingChenAGI/Xing4.0-29B") -> dict:
    natural.validate_packet(packet)
    a, b, selected = _pair(packet, run_a, run_b)
    if model_id in {a["plan"]["model_id"], b["plan"]["model_id"]}:
        raise ValueError("top-up model ID must differ from both original reviewers")
    inputs = _inputs(packet, a, b, selected, model_id)
    job_plan = ai_jobs.build_plan("instruction_reference", inputs, cfg,
        max_calls=TARGET_CASE_COUNT, tokens=640, max_input_chars=16000,
        max_cost_usd=0.0, response_format=CONTRACT_FORMAT, zero_price_only=True)
    body = {"schema": PLAN_SCHEMA, "packet_sha256": packet["packet_sha256"],
        "inputs_sha256": sha256_obj(inputs), "selected_case_ids": inputs["selected_case_ids"],
        "topup_model": inputs["topup_model"], "planned_calls": TARGET_CASE_COUNT,
        "zero_price_only": True, "paid_fallback": False,
        "source_runs_immutable": True, "job_plan": job_plan}
    return {**body, "plan_sha256": sha256_obj(body)}


def _payload(case: dict, dispute: dict) -> dict:
    return {"task_contract": case["contract"], "candidate_strategy": case["strategy_text"],
        "reviewer_a_contract": dispute["reviewer_a"],
        "reviewer_b_contract": dispute["reviewer_b"],
        "focus": "contract_only"}


def _parse(raw: str, case: dict) -> dict:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid contract top-up JSON") from exc
    expected = {"verdict", "candidate_quotes", "contract_quote", "reason"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("contract top-up schema changed")
    # Quotes are required evidence only for an ``incompatible`` decision.  A
    # free model often emits a paraphrase quote even for an approval; retaining
    # that non-verbatim text would make an otherwise valid approval fail closed
    # for a formatting defect.  Normalize approval/uncertain annotations to the
    # contract's empty-quote form while keeping the unmodified response in the
    # immutable call receipt for audit.
    if value["verdict"] != "incompatible":
        value = {**value, "candidate_quotes": [], "contract_quote": None}
    natural._validate_completed_dimension({**value, "status": "completed",
        # The structural validator's blind flag records that labels are not
        # host-injected; prior votes are explicitly carried in the payload and
        # provenance below records this as a third-context adjudication.
        "reviewer": "adjudicator", "blind_to_model_reviews": True},
        case["strategy_text"], dimension="contract",
        contract_text=case["contract"]["text"])
    return value


def _normalization_issue(raw: str) -> str | None:
    """Return an auditable marker when approval-only quote fields were dropped."""
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if (isinstance(value, dict) and value.get("verdict") != "incompatible"
            and (value.get("candidate_quotes") or value.get("contract_quote") is not None)):
        return "approval_or_uncertain_quotes_normalized"
    return None


def execute(packet: dict, run_a: Path, run_b: Path, cfg: dict, plan: dict,
            *, provider=None) -> dict:
    a, b, selected = _pair(packet, run_a, run_b)
    expected = build_plan(packet, run_a, run_b, cfg, model_id=plan["topup_model"]["model_id"])
    if plan != expected:
        raise ValueError("contract top-up plan or source binding changed")
    by_id = {case["case_id"]: case for case in packet["cases"]}
    inputs = _inputs(packet, a, b, selected, plan["topup_model"]["model_id"])

    def worker(call):
        rows = []
        for dispute in selected:
            case = by_id[dispute["case_id"]]
            payload = _payload(case, dispute)
            raw, provenance = call("instruction_adjudicator", payload, CONTRACT_RUBRIC)
            parsed = _parse(raw, case)
            rows.append({"case_id": case["case_id"], "dimension": "contract",
                         "verdict": parsed["verdict"], "annotation": parsed,
                         "normalization_issue": _normalization_issue(raw),
                         "provenance": {**provenance, "blind": False,
                             "other_review_outputs_consulted": True,
                             "origin": "independent_ai", "identity": "contract_topup"}})
        return {"schema": RUN_SCHEMA, "packet_sha256": packet["packet_sha256"],
            "inputs_sha256": sha256_obj(inputs), "annotation_origin": "ai",
            "natural_distribution": True, "cases": rows}

    return ai_jobs.execute(plan["job_plan"], inputs, cfg, worker, provider=provider)


def verify(packet: dict, run_a: Path, run_b: Path, plan: dict, run_dir: Path) -> dict:
    a, b, selected = _pair(packet, run_a, run_b)
    job = ai_jobs.verify_completed(run_dir, kind="instruction_reference")
    output = job["output"]
    inputs = _inputs(packet, a, b, selected, plan["topup_model"]["model_id"])
    if (plan.get("schema") != PLAN_SCHEMA
            or plan.get("plan_sha256") != sha256_obj({k: v for k, v in plan.items()
                                                       if k != "plan_sha256"})
            or job["plan"] != plan.get("job_plan")
            or output.get("schema") != RUN_SCHEMA
            or output.get("packet_sha256") != packet["packet_sha256"]
            or output.get("inputs_sha256") != sha256_obj(inputs)
            or [row.get("case_id") for row in output.get("cases", [])]
                != [row["case_id"] for row in selected]):
        raise ValueError("contract top-up run identity changed")
    for index, dispute in enumerate(selected):
        case = next(c for c in packet["cases"] if c["case_id"] == dispute["case_id"])
        raw = json.loads((Path(job["root"]) / f"call-{index}.json").read_text(encoding="utf-8"))
        payload = _payload(case, dispute)
        parsed = _parse(raw["raw_output"], case)
        row = output["cases"][index]
        if (raw.get("role") != "instruction_adjudicator" or raw.get("system") != CONTRACT_RUBRIC
                or raw.get("payload") != payload or row.get("annotation") != parsed
                or row.get("verdict") != parsed["verdict"]
                or row.get("normalization_issue") != _normalization_issue(raw["raw_output"])):
            raise ValueError("contract top-up receipt or parser output changed")
    return job


def merge_v3(frozen_v2: dict, topup_job: dict) -> dict:
    if frozen_v2.get("schema") != "natural-instruction-frozen-reference-v2":
        raise ValueError("contract top-up merge requires frozen v2 reference")
    if frozen_v2.get("reference_sha256") != sha256_obj({k: v for k, v in frozen_v2.items()
                                                          if k != "reference_sha256"}):
        raise ValueError("frozen v2 reference hash mismatch")
    output = topup_job.get("output", {})
    if output.get("schema") != RUN_SCHEMA or len(output.get("cases", [])) != TARGET_CASE_COUNT:
        raise ValueError("contract top-up output is incomplete")
    replacements = {row["case_id"]: row for row in output["cases"]}
    rows = []
    for row in frozen_v2["cases"]:
        replacement = replacements.get(row["case_id"])
        verdicts = dict(row["verdicts"])
        if replacement is not None:
            verdicts["contract"] = replacement["verdict"]
        rows.append({**row, "verdicts": verdicts,
                     **({"contract_topup": replacement} if replacement is not None else {})})
    coverage = natural.reference_class_coverage(rows)
    sufficient = all(coverage[d][key] >= natural.THRESHOLDS[t]
                     for d in ("assertion", "contract")
                     for key, t in (("approval_reference", "min_approval_reference"),
                                    ("rejection_reference", "min_rejection_reference")))
    body = {**frozen_v2, "schema": FROZEN_SCHEMA,
        "reference_method": "v2_reference_plus_contract_only_third_ai_topup",
        "contract_topup": {"run_output_sha256": sha256_obj(output),
                           "case_count": len(replacements),
                           "model_id": output["cases"][0]["provenance"]["model_id"]},
        "cases": rows, "reference_class_coverage": coverage,
        "reference_class_coverage_sufficient": sufficient,
        "prior_reference_v2_sha256": frozen_v2["reference_sha256"]}
    body.pop("reference_sha256", None)
    return {**body, "reference_sha256": sha256_obj(body)}


__all__ = ["PLAN_SCHEMA", "RUN_SCHEMA", "FROZEN_SCHEMA", "CONTRACT_RUBRIC",
           "build_plan", "execute", "verify", "merge_v3"]
