"""Plan or execute one blinded AI review of a HealthVer grounding packet."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.eval import ai_jobs  # noqa: E402
from src.eval.ai_grounding_reference import validate_review  # noqa: E402
from src.eval.grounding import evidence_sentence_spans  # noqa: E402
from src.util import read_json, sha256_obj, write_json  # noqa: E402


def _config(path: Path, model_id: str) -> dict:
    cfg = load_config(path)
    model = cfg.setdefault("models", {}).setdefault("task", {})
    if model.get("provider") != "http":
        raise ValueError("HealthVer AI review requires an HTTP task model")
    model["model_id"] = model_id
    return cfg


def _inputs(packet: dict, slot: str, batch_size: int) -> dict:
    if packet.get("reviewer_slot") != slot or packet.get("scope", {}).get("dataset") != "healthver":
        raise ValueError("HealthVer review packet/slot mismatch")
    if not 1 <= batch_size <= 8:
        raise ValueError("batch size must be between 1 and 8")
    if len(packet.get("cases", [])) % batch_size:
        raise ValueError("batch size must divide the frozen reviewer case count")
    return {"packet": packet, "reviewer_slot": slot, "batch_size": batch_size}


def _response_format(batch_size: int, mode: str = "schema", *, per_document: bool = False) -> dict:
    if mode not in {"schema", "json_object"}:
        raise ValueError("grounding AI review output mode must be schema or json_object")
    if mode == "json_object":
        # Use this for providers that expose JSON mode but not strict JSON Schema.
        # validate_review and _worker still enforce the complete local contract.
        return {"type": "json_object"}
    if per_document:
        relation = {"type": "object", "additionalProperties": False,
                    "properties": {
                        "doc_id": {"type": "string"},
                        "verdict": {"type": "string", "enum": ["supported", "contradicted", "insufficient"]},
                        "sentence_ids": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000}, "maxItems": 5},
                    }, "required": ["doc_id", "verdict", "sentence_ids"]}
        annotation = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "case_id": {"type": "string"}, "input_sha256": {"type": "string"},
                "document_relations": {"type": "array", "items": relation, "minItems": 3, "maxItems": 3},
                "reason": {"type": "string", "maxLength": 300},
                "confidence_1_to_3": {"type": "integer", "enum": [1, 2, 3]},
            }, "required": ["case_id", "input_sha256", "document_relations", "reason", "confidence_1_to_3"],
        }
    else:
        annotation = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "case_id": {"type": "string"},
            "input_sha256": {"type": "string"},
            "verdict": {"type": "string", "enum": [
                "supported", "contradicted", "insufficient", "unable_to_judge"]},
            "supporting_sentence_ids": {"type": "array", "items": {
                "type": "integer", "minimum": 0, "maximum": 1000},
                "maxItems": 5},
            "quote_doc_id": {"type": "string"},
            "reason": {"type": "string", "maxLength": 300},
            "confidence_1_to_3": {"type": "integer", "enum": [1, 2, 3]},
        },
        "required": ["case_id", "input_sha256", "verdict", "supporting_sentence_ids",
                     "quote_doc_id", "reason", "confidence_1_to_3"],
        }
    return {"type": "json_schema", "json_schema": {
        "name": "healthver_grounding_ai_reference",
        "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "properties": {"annotations": {
                       "type": "array", "items": annotation,
                       "minItems": batch_size, "maxItems": batch_size}},
                   "required": ["annotations"]},
    }}


def _decode_annotation_rows(output, model_fields: set[str]):
    """Decode complete per-case rows, including a narrow repeated-wrapper repair.

    Some JSON-mode endpoints return ``{"annotations":[row1]},
    "annotations":[row2]`` instead of one JSON object. We recover this only
    when every extracted row is a complete JSON object and removing those rows
    leaves exactly the repeated annotations-array envelope. Other malformed
    text remains invalid. A missing/duplicate row is handled per case below,
    so one bad row cannot erase valid siblings from the same batch.
    """
    if isinstance(output, str):
        def reject_duplicate_keys(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate JSON key")
                value[key] = item
            return value
        try:
            parsed = json.loads(output, object_pairs_hook=reject_duplicate_keys)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if parsed is not None:
            return _decode_annotation_rows(parsed, model_fields)
        decoder = json.JSONDecoder()
        rows, spans = [], []
        for match in re.finditer(r"\{", output):
            try:
                value, end = decoder.raw_decode(output, match.start())
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and set(value) == model_fields:
                rows.append(value)
                spans.append((match.start(), end))
        if not rows:
            return [], set(), False
        residual = output
        for start, end in reversed(spans):
            residual = residual[:start] + "<ROW>" + residual[end:]
        residual = re.sub(r"\s+", "", residual).replace("<ROW>", "")
        repeated_envelope = re.fullmatch(
            r'\{"annotations":\[\](?:,"annotations":\[\])*\}', residual)
        if not repeated_envelope or output.count('"annotations"') != len(rows):
            return [], set(), False
        output = {"annotations": rows}

    if isinstance(output, list):
        output = {"annotations": output}
    if not isinstance(output, dict) or set(output) != {"annotations"} or not isinstance(output["annotations"], list):
        return [], set(), False
    rows_by_id, duplicate_ids = {}, set()
    for row in output["annotations"]:
        if not isinstance(row, dict) or set(row) != model_fields or not isinstance(row.get("case_id"), str):
            continue
        case_id = row["case_id"]
        if case_id in rows_by_id:
            duplicate_ids.add(case_id)
        else:
            rows_by_id[case_id] = row
    return list(rows_by_id.values()), duplicate_ids, True


def _worker(call, inputs: dict, model_id: str) -> dict:
    packet = inputs["packet"]
    cases = sorted(packet["cases"], key=lambda row: row["case_id"])
    batch_size = inputs["batch_size"]
    annotations = []
    provenance_rows = []
    per_document = packet.get("protocol_context", {}).get("schema", "").endswith("v6")
    model_fields = ({"case_id", "input_sha256", "document_relations", "reason", "confidence_1_to_3"}
                    if per_document else
                    {"case_id", "input_sha256", "verdict", "supporting_sentence_ids",
                     "quote_doc_id", "reason", "confidence_1_to_3"})
    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]
        payload = {"reviewer_slot": inputs["reviewer_slot"], "cases": batch}
        output, provenance = call("grounding_reference", payload, packet["instructions"])
        rows, duplicate_ids, envelope_valid = _decode_annotation_rows(output, model_fields)
        rows_by_id = {row["case_id"]: row for row in rows}
        expected = {case["case_id"]: case for case in batch}
        for case_id, case in expected.items():
            raw = rows_by_id.get(case_id)
            if per_document and isinstance(raw, dict):
                raw = _aggregate_document_relations(raw, case)
            usable = (envelope_valid and case_id not in duplicate_ids and isinstance(raw, dict)
                      and raw.get("input_sha256") == case["input_sha256"]
                      and raw.get("verdict") in {"supported", "contradicted", "insufficient", "unable_to_judge"}
                      and type(raw.get("confidence_1_to_3")) is int
                      and raw.get("confidence_1_to_3") in (1, 2, 3)
                      and isinstance(raw.get("reason"), str) and bool(raw["reason"].strip())
                      and isinstance(raw.get("supporting_sentence_ids"), list)
                      and isinstance(raw.get("quote_doc_id"), str))
            quote = ""
            doc_id = ""
            if usable:
                sentence_ids = raw["supporting_sentence_ids"]
                if raw["verdict"] in {"supported", "contradicted"}:
                    usable = (all(type(value) is int for value in sentence_ids)
                              and len(sentence_ids) == len(set(sentence_ids)))
                if raw["verdict"] in {"supported", "contradicted"} and usable:
                    evidence_docs = case["evidence"]
                    doc_id = raw["quote_doc_id"]
                    matching = [doc for doc in evidence_docs if doc["doc_id"] == doc_id]
                    if len(matching) != 1 or not sentence_ids:
                        usable = False
                    else:
                        source = matching[0]["text"]
                        spans = evidence_sentence_spans(source)
                        if any(value < 0 or value >= len(spans) for value in sentence_ids):
                            usable = False
                        else:
                            ordered_ids = sorted(sentence_ids)
                            quote = source[spans[ordered_ids[0]]["start"]:spans[ordered_ids[-1]]["end"]]
                            usable = len(quote.strip()) >= 20
                # Evidence pointers are meaningful only for decisive labels.
                # Models sometimes return a stray document ID or sentence list
                # with an insufficient/abstain verdict; canonicalize those
                # unused fields instead of discarding an otherwise valid label.
            if usable:
                annotation = {key: value for key, value in raw.items()
                              if key != "supporting_sentence_ids"}
                annotation["supporting_quote"] = quote
                annotation["quote_doc_id"] = doc_id
            else:
                # Invalid model output is preserved in the private call record,
                # but can only become an excluded abstention in the reference.
                annotation = {"case_id": case_id, "input_sha256": case["input_sha256"],
                              "verdict": "unable_to_judge", "supporting_quote": "",
                              "quote_doc_id": "",
                              "reason": "The reviewer output failed local schema or evidence-index validation.",
                              "confidence_1_to_3": 1}
            annotations.append(annotation)
        provenance_rows.append(provenance)
    first = provenance_rows[0]
    response = {
        "schema": "task-input-grounding-ai-review-v1", "reviewer_type": "ai",
        "reviewer_id": inputs["reviewer_slot"] + "@" + model_id,
        "independent_human": False, "eligible_as_human_reference": False,
        "protocol_sha256": packet["protocol_sha256"],
        "packet_sha256": packet["packet_sha256"],
        "rubric_file_sha256": sha256_obj(packet["instructions"]),
        "provenance": {"review_run_id": first["review_run_id"], "model_id": model_id,
                       "rubric_sha256": sha256_obj(packet["instructions"]),
                       "blind": True, "other_review_outputs_consulted": False,
                       "batch_run_ids": [row["review_run_id"] for row in provenance_rows]},
        "annotations": sorted(annotations, key=lambda row: row["case_id"]),
    }
    validate_review(packet, response)
    return response


def _aggregate_document_relations(raw: dict, case: dict) -> dict:
    """Validate document-level labels/citations, then apply the frozen local rule."""
    relations = raw.get("document_relations")
    docs = case.get("evidence", [])
    expected_ids = [doc.get("doc_id") for doc in docs]
    valid = (isinstance(relations, list) and len(relations) == len(expected_ids)
             and all(isinstance(row, dict) and set(row) == {"doc_id", "verdict", "sentence_ids"}
                     for row in relations))
    by_doc = {}
    if valid:
        for row in relations:
            doc_id, verdict, sentence_ids = row["doc_id"], row["verdict"], row["sentence_ids"]
            if doc_id in by_doc or verdict not in {"supported", "contradicted", "insufficient"} or not isinstance(sentence_ids, list):
                valid = False
                break
            if verdict in {"supported", "contradicted"}:
                if (not sentence_ids or any(type(value) is not int for value in sentence_ids)
                        or len(sentence_ids) != len(set(sentence_ids))):
                    valid = False
                    break
                doc = next((item for item in docs if item["doc_id"] == doc_id), None)
                if doc is None:
                    valid = False
                    break
                spans = evidence_sentence_spans(doc["text"])
                if any(value < 0 or value >= len(spans) for value in sentence_ids):
                    valid = False
                    break
            elif sentence_ids:
                valid = False
                break
            by_doc[doc_id] = row
        if set(by_doc) != set(expected_ids):
            valid = False
    if not valid:
        return {"case_id": raw.get("case_id"), "input_sha256": raw.get("input_sha256"),
                "verdict": "unable_to_judge", "supporting_sentence_ids": [], "quote_doc_id": "",
                "reason": "Per-document relation rows failed identity or citation validation.",
                "confidence_1_to_3": 1}
    labels = {row["verdict"] for row in by_doc.values()}
    decisive = labels & {"supported", "contradicted"}
    verdict = ("insufficient" if decisive == {"supported", "contradicted"}
               else next(iter(decisive)) if decisive else "insufficient")
    quote_doc_id, sentence_ids = "", []
    if verdict in {"supported", "contradicted"}:
        chosen = next(doc_id for doc_id in expected_ids if by_doc[doc_id]["verdict"] == verdict)
        quote_doc_id = chosen
        sentence_ids = by_doc[chosen]["sentence_ids"]
    return {"case_id": raw["case_id"], "input_sha256": raw["input_sha256"],
            "verdict": verdict, "supporting_sentence_ids": sentence_ids,
            "quote_doc_id": quote_doc_id,
            "reason": raw["reason"], "confidence_1_to_3": raw["confidence_1_to_3"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--slot", choices=("reviewer_a", "reviewer_b"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.http.pubmedqa-final.yaml")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--output-mode", choices=("schema", "json_object"), default="schema")
    parser.add_argument("--max-cost-usd", type=float, default=0.25)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args()

    packet = read_json(args.packet)
    inputs = _inputs(packet, args.slot, args.batch_size)
    cfg = _config(args.config, args.model_id)
    cases = len(packet["cases"])
    calls = math.ceil(cases / args.batch_size)
    plan = ai_jobs.build_plan("grounding_reference", inputs, cfg, max_calls=calls,
                              tokens=3000, max_input_chars=60000,
                              max_cost_usd=args.max_cost_usd,
                              response_format=_response_format(
                                  args.batch_size, args.output_mode,
                                  per_document=packet.get("protocol_context", {}).get("schema", "").endswith("v6")))
    if not args.execute:
        write_json(args.plan, plan)
        print(json.dumps({"plan_path": str(args.plan), "plan_sha256": plan["plan_sha256"],
                          "model_id": args.model_id, "reviewer_slot": args.slot,
                          "case_count": cases, "call_count": calls,
                          "max_cost_usd": args.max_cost_usd}, indent=2))
        return 0
    if args.expected_plan_sha256 != plan["plan_sha256"]:
        raise ValueError("execution requires the exact reviewed plan SHA256")
    frozen = read_json(args.plan)
    if frozen != plan:
        raise ValueError("saved plan differs from the current packet/model/budget")
    job = ai_jobs.execute(
        plan, inputs, cfg,
        lambda call: _worker(call, inputs, args.model_id))
    if args.output:
        write_json(args.output, job["output"])
    print(json.dumps({"run_dir": job["run_dir"], "receipt": job["receipt"],
                      "reviewer_id": job["output"]["reviewer_id"],
                      "annotations": len(job["output"]["annotations"]),
                      "response_sha256": sha256_obj(job["output"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
