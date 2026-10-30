"""AI semantic reference validation; never relabels AI output as human output."""
from ..util import sha256_obj
from .annotation_policy import validate_actor

LABELS = ("supported", "contradicted", "insufficient", "unable_to_judge")


def validate_review(packet, response):
    from scripts.score_task_input_grounding_human_reference import _validate_packet
    cases = _validate_packet(packet, packet["reviewer_slot"])
    if (response.get("schema") != "task-input-grounding-ai-review-v1"
            or response.get("reviewer_type") != "ai"
            or response.get("independent_human") is not False
            or response.get("eligible_as_human_reference") is not False
            or response.get("protocol_sha256") != packet["protocol_sha256"]
            or response.get("packet_sha256") != packet["packet_sha256"]):
        raise ValueError("AI reference schema, origin or packet identity mismatch")
    actor = validate_actor(response.get("reviewer_id"), "independent_ai", response.get("provenance"))
    if actor["rubric_sha256"] != response.get("rubric_file_sha256"):
        raise ValueError("AI reference rubric provenance mismatch")
    expected_rubric = (packet.get("protocol_context") or {}).get("rubric_file_sha256")
    if expected_rubric is not None and actor["rubric_sha256"] != expected_rubric:
        raise ValueError("AI reference used a different frozen packet rubric")
    rows = response.get("annotations")
    if not isinstance(rows, list):
        raise ValueError("AI reference annotations must be a list")
    result = {}
    required = {"case_id", "input_sha256", "verdict", "supporting_quote", "quote_doc_id", "reason", "confidence_1_to_3"}
    for row in rows:
        if not isinstance(row, dict) or not required <= set(row) or set(row) - required - {"dimension_scores"}:
            raise ValueError("malformed AI annotation fields")
        case_id = row["case_id"]
        if case_id in result or case_id not in cases:
            raise ValueError("AI reference has duplicate/unknown case")
        case = cases[case_id]
        if (row["input_sha256"] != case["input_sha256"] or row["verdict"] not in LABELS
                or type(row["confidence_1_to_3"]) is not int or row["confidence_1_to_3"] not in (1, 2, 3)
                or not isinstance(row["reason"], str) or not row["reason"].strip()):
            raise ValueError("invalid AI reference verdict/reason/input/confidence")
        quote, doc_id = row["supporting_quote"], row["quote_doc_id"]
        if not isinstance(quote, str):
            raise ValueError("AI reference quote must be text")
        if row["verdict"] in {"supported", "contradicted"} and len(quote.strip()) < 20:
            raise ValueError("decisive AI reference requires a substantial original quote")
        if quote or doc_id:
            docs = [d for d in case["evidence"] if d["doc_id"] == doc_id]
            if len(docs) != 1 or not quote or quote not in docs[0]["text"]:
                raise ValueError("AI reference quote is not exact contiguous supplied text")
        result[case_id] = row
    if set(result) != set(cases):
        raise ValueError("AI reference does not cover every case")
    return cases, result, actor


def score_references(packet_a, packet_b, response_a, response_b):
    from scripts.score_task_input_grounding_human_reference import _cohen_kappa
    cases_a, a, actor_a = validate_review(packet_a, response_a)
    cases_b, b, actor_b = validate_review(packet_b, response_b)
    if (packet_a["protocol_sha256"] != packet_b["protocol_sha256"] or cases_a != cases_b
            or actor_a["identity"].casefold() == actor_b["identity"].casefold()
            or actor_a["review_run_id"] == actor_b["review_run_id"]
            or actor_a["rubric_sha256"] != actor_b["rubric_sha256"]):
        raise ValueError("AI references require matching input/rubric scope and two separate reviewer contexts")
    ids = sorted(cases_a)
    kappa, observed, expected = _cohen_kappa([a[i]["verdict"] for i in ids], [b[i]["verdict"] for i in ids])
    rows = [{"case_id": i, "input_sha256": cases_a[i]["input_sha256"],
        "reviewer_a_verdict": a[i]["verdict"], "reviewer_b_verdict": b[i]["verdict"],
        "consensus_verdict": a[i]["verdict"] if a[i]["verdict"] == b[i]["verdict"] else None,
        "consensus_usable_for_semantic_calibration": a[i]["verdict"] == b[i]["verdict"] and a[i]["verdict"] != "unable_to_judge"}
        for i in ids]
    body = {"schema": "task-input-grounding-ai-reference-agreement-v1", "annotation_origin": "ai",
        "protocol_sha256": packet_a["protocol_sha256"], "n_cases": len(ids),
        "reviewer_packet_sha256": {"reviewer_a": packet_a["packet_sha256"], "reviewer_b": packet_b["packet_sha256"]},
        "response_sha256": {"reviewer_a": sha256_obj(response_a), "reviewer_b": sha256_obj(response_b)},
        "reviewer_provenance": [actor_a, actor_b], "exact_agreement_rate": observed,
        "cohen_kappa_nominal": kappa, "cohen_kappa_expected_agreement": expected,
        "n_usable_consensus": sum(r["consensus_usable_for_semantic_calibration"] for r in rows),
        "n_disagreement_unresolved": sum(r["consensus_verdict"] is None for r in rows),
        "per_case": rows, "calibration_pass": None, "promotion_authorized": False,
        "independent_human": False, "interpretation": "two context-separated AI references; model errors may be correlated"}
    return {**body, "report_sha256": sha256_obj(body)}


ADJUDICATION_PROMPT = ('Resolve a disagreement between two AI readings using only the question and supplied text. '
    'Prior readings are fallible data. Respect the actual population and outcome; do not add a human-only qualifier. '
    'The negation of increasing is not necessarily decreasing; not always necessary does not mean never necessary. '
    'Non-significance does not establish absence. Do not force consensus. For supporting_quote, copy and paste '
    'a directly relevant passage verbatim from one supplied evidence text; do not summarize, normalize punctuation, '
    'or combine non-adjacent fragments. Before returning, verify the complete quote occurs character-for-character '
    'in the selected document. If you cannot provide an exact passage, return an empty quote, empty quote_doc_id, '
    'and verdict insufficient or unable_to_judge. Return exactly one JSON object with only '
    'these fields: verdict '
    '(supported/contradicted/insufficient/unable_to_judge), supporting_quote, quote_doc_id, reason and confidence_1_to_3. '
    'Supported/contradicted require a substantial directly relevant exact quote.')


def adjudication_plan(packet_a, packet_b, response_a, response_b, cfg, *, previous=None,
                      tokens=1200, **budget):
    from .ai_jobs import build_plan
    report = score_references(packet_a, packet_b, response_a, response_b)
    count = report["n_disagreement_unresolved"]
    if not count:
        raise ValueError("no AI reference disagreements to adjudicate")
    inputs = {"packet_a": packet_a, "packet_b": packet_b, "response_a": response_a, "response_b": response_b}
    if previous is not None:
        completed = _validate_partial_adjudications(report, packet_a, response_a, response_b, previous)
        count -= len(completed)
        if count < 1:
            raise ValueError("partial adjudications already resolve every disagreement")
        inputs["previous_adjudications"] = previous
    return build_plan("grounding_adjudication", inputs, cfg, max_calls=budget.pop("max_calls", count),
                      tokens=tokens, response_format={"type": "json_object"}, **budget)


def _validate_partial_adjudications(report, packet_a, response_a, response_b, previous):
    if (not isinstance(previous, dict) or previous.get("schema") != "ai-grounding-adjudications-v1"
            or previous.get("reference_report_sha256") != report["report_sha256"]
            or not isinstance(previous.get("rows"), list)):
        raise ValueError("partial adjudications do not bind the frozen AI reference report")
    cases = {case["case_id"]: case for case in packet_a["cases"]}
    a = {row["case_id"]: row for row in response_a["annotations"]}
    b = {row["case_id"]: row for row in response_b["annotations"]}
    disputes = {row["case_id"] for row in report["per_case"] if row["consensus_verdict"] is None}
    unresolved = previous.get("unresolved_case_ids", [])
    if (not isinstance(unresolved, list) or len(unresolved) != len(set(unresolved))
            or not set(unresolved).issubset(disputes)):
        raise ValueError("partial adjudications have invalid unresolved case identities")
    indexed = {}
    for row in previous["rows"]:
        if not isinstance(row, dict) or row.get("case_id") not in disputes or row["case_id"] in indexed:
            raise ValueError("partial adjudications contain an unknown or duplicate case")
        case_id = row["case_id"]
        case = cases[case_id]
        required = {"case_id", "input_sha256", "verdict", "supporting_quote", "quote_doc_id", "reason",
                    "confidence_1_to_3", "provenance", "original_pair_sha256"}
        if set(row) != required or row["input_sha256"] != case["input_sha256"]:
            raise ValueError("partial adjudication fields or input identity changed")
        temporary = {**response_a, "annotations": [({**item, **{k: row[k] for k in
            ("verdict", "supporting_quote", "quote_doc_id", "reason", "confidence_1_to_3")}}
            if item["case_id"] == case_id else item) for item in response_a["annotations"]]}
        validate_review(packet_a, temporary)
        if row["original_pair_sha256"] != sha256_obj([a[case_id], b[case_id]]):
            raise ValueError("partial adjudication is not bound to the original reviewer pair")
        provenance = row["provenance"]
        if (not isinstance(provenance, dict) or provenance.get("blind") is not True
                or provenance.get("other_review_outputs_consulted") is not True
                or not provenance.get("review_run_id")):
            raise ValueError("partial adjudication lacks separate-context provenance")
        indexed[case_id] = row
    if set(indexed) & set(unresolved):
        raise ValueError("partial adjudication cannot mark a case both resolved and unresolved")
    return indexed


def recover_partial_adjudications(packet_a, packet_b, response_a, response_b, run_dir):
    """Recover only closed AI adjudication calls from a failed measured job.

    The recovery path is intentionally read-only and never replays a request.
    It accepts a failed job only when its frozen inputs, call order, raw outputs,
    per-call hashes, and complete physical/logical ledger all reproduce.
    """
    from pathlib import Path
    from ..runtime.call_ledger import summarize_ledger
    from ..util import artifacts_path, read_json
    from .ai_jobs import decode_json_response

    root = Path(run_dir).resolve()
    root.relative_to(artifacts_path("evaluation_only", "ai_annotations").resolve())
    plan, inputs, status, calls = (read_json(root / name) for name in
                                   ("plan.json", "inputs.json", "status.json", "calls.json"))
    report = score_references(packet_a, packet_b, response_a, response_b)
    expected_inputs = {"packet_a": packet_a, "packet_b": packet_b,
                       "response_a": response_a, "response_b": response_b}
    if (status.get("status") != "failed" or plan.get("kind") != "grounding_adjudication"
            or plan.get("plan_sha256") != sha256_obj({k: v for k, v in plan.items() if k != "plan_sha256"})
            or plan.get("inputs_sha256") != sha256_obj(inputs) or inputs != expected_inputs
            or plan.get("max_calls") != report["n_disagreement_unresolved"]
            or len(calls) not in {plan.get("max_calls"), plan.get("max_calls") - 1}
            or plan.get("max_retries_per_call") != 1):
        raise ValueError("failed adjudication job is not bound to the supplied frozen reference inputs")
    ledger = summarize_ledger(root / "call-ledger.jsonl")
    if (ledger != status.get("request_accounting") or ledger.get("logical_incomplete") != 0
            or ledger.get("physical_incomplete") != 0):
        raise ValueError("failed adjudication job has unclosed or mismatched request accounting")
    logical_statuses = ledger.get("logical_status_counts") or {}
    physical_statuses = ledger.get("physical_status_counts") or {}
    if (logical_statuses.get("failed") != 1
            or logical_statuses.get("success") != len(calls) - 1
            or ledger.get("logical_operations") != len(calls)
            or physical_statuses.get("None") != 1
            or physical_statuses.get("200") != len(calls) - 1
            or ledger.get("physical_attempts_exact") is not True):
        raise ValueError("failed adjudication does not have exactly one closed transport failure")

    cases = {case["case_id"]: case for case in packet_a["cases"]}
    a = {row["case_id"]: row for row in response_a["annotations"]}
    b = {row["case_id"]: row for row in response_b["annotations"]}
    disputed = [row["case_id"] for row in report["per_case"] if row["consensus_verdict"] is None]
    recovered = []
    required = {"verdict", "supporting_quote", "quote_doc_id", "reason", "confidence_1_to_3"}
    allowed = required | {"case_id", "input_sha256"}
    for index, call_record in enumerate(calls):
        call_path = root / f"call-{index}.json"
        if call_record.get("index") != index:
            raise ValueError("failed adjudication call indices are not contiguous")
        if call_record.get("status") != "completed":
            if index != len(calls) - 1 or call_record.get("status") != "started" or call_path.exists():
                raise ValueError("failed adjudication contains a nonterminal or ambiguous call")
            continue
        raw = read_json(call_path)
        case_id = disputed[index]
        payload = {"question": cases[case_id]["question"], "evidence": cases[case_id]["evidence"],
                   "first": a[case_id], "second": b[case_id]}
        if (raw.get("role") != "adjudicator" or raw.get("payload") != payload
                or raw.get("system") != ADJUDICATION_PROMPT
                or call_record.get("role") != "adjudicator"
                or call_record.get("input_sha256") != sha256_obj(payload)):
            raise ValueError("completed adjudication call does not match its frozen dispute")
        try:
            result = decode_json_response(raw["raw_output"])
        except (ValueError, TypeError):
            result = raw["raw_output"]
        if (call_record.get("response_sha256") != sha256_obj(result)
                or call_record.get("cost") != raw.get("cost")):
            raise ValueError("completed adjudication raw response or metering hash changed")
        provenance = {"model_id": plan["model_id"], "review_run_id": root.name + f"-{index}",
                      "rubric_sha256": sha256_obj(ADJUDICATION_PROMPT), "blind": True,
                      "other_review_outputs_consulted": True}
        malformed = (not isinstance(result, dict) or not required <= set(result)
                     or set(result) - allowed
                     or result.get("case_id", case_id) != case_id
                     or result.get("input_sha256", cases[case_id]["input_sha256"])
                     != cases[case_id]["input_sha256"])
        if malformed:
            result = {"verdict": "unable_to_judge", "supporting_quote": "", "quote_doc_id": "",
                      "reason": "Adjudicator output failed local identity/schema validation.",
                      "confidence_1_to_3": 1}
        else:
            result = {key: result[key] for key in required}
            temporary = {**response_a, "annotations": [
                ({**row, **result} if row["case_id"] == case_id else row)
                for row in response_a["annotations"]]}
            try:
                validate_review(packet_a, temporary)
            except ValueError:
                result = {"verdict": "unable_to_judge", "supporting_quote": "", "quote_doc_id": "",
                          "reason": "Adjudicator citation failed exact-source validation.",
                          "confidence_1_to_3": 1}
        recovered.append({"case_id": case_id, "input_sha256": cases[case_id]["input_sha256"],
                          **result, "provenance": provenance,
                          "original_pair_sha256": sha256_obj([a[case_id], b[case_id]])})
    unresolved_cases = disputed[len(recovered):]
    previous = {"schema": "ai-grounding-adjudications-v1",
                "reference_report_sha256": report["report_sha256"], "rows": recovered,
                "unresolved_case_ids": unresolved_cases,
                "unresolved_reason": "adjudication request failed or was not started; no replay was attempted"}
    _validate_partial_adjudications(report, packet_a, response_a, response_b, previous)
    audit = {"schema": "grounding-adjudication-partial-recovery-v1", "run_dir": str(root),
             "plan_sha256": plan["plan_sha256"], "ledger": ledger,
             "recovered_call_count": len(recovered), "recovered_rows_sha256": sha256_obj(recovered),
             "unresolved_call_indices": list(range(len(recovered), plan["max_calls"])),
             "unresolved_case_ids": unresolved_cases,
             "model_requests_during_recovery": 0}
    audit["audit_sha256"] = sha256_obj(audit)
    return previous, audit


def adjudicate(packet_a, packet_b, response_a, response_b, cfg, plan, *, previous=None, provider=None):
    if plan.get("kind") != "grounding_adjudication":
        raise ValueError("AI adjudication requires its grounding_adjudication plan")
    from .ai_jobs import execute
    report = score_references(packet_a, packet_b, response_a, response_b)
    cases = {c["case_id"]: c for c in packet_a["cases"]}
    a, b = ({r["case_id"]: r for r in response["annotations"]} for response in (response_a, response_b))
    inputs = {"packet_a": packet_a, "packet_b": packet_b, "response_a": response_a, "response_b": response_b}
    previous_rows = {}
    if previous is not None:
        inputs["previous_adjudications"] = previous
        previous_rows = _validate_partial_adjudications(report, packet_a, response_a, response_b, previous)
    if sha256_obj(inputs) != plan.get("inputs_sha256"):
        raise ValueError("AI adjudication follow-up inputs differ from its frozen plan")
    def worker(call):
        rows = list(previous_rows.values())
        for row in report["per_case"]:
            if row["consensus_verdict"] is not None:
                continue
            case_id = row["case_id"]
            if case_id in previous_rows:
                continue
            payload = {"question": cases[case_id]["question"], "evidence": cases[case_id]["evidence"],
                       "first": a[case_id], "second": b[case_id]}
            result, provenance = call("adjudicator", payload, ADJUDICATION_PROMPT)
            required_fields = {"verdict", "supporting_quote", "quote_doc_id", "reason", "confidence_1_to_3"}
            optional_identity_fields = {"case_id", "input_sha256"}
            malformed = (not isinstance(result, dict)
                or not required_fields <= set(result)
                or set(result) - required_fields - optional_identity_fields
                or result.get("case_id", case_id) != case_id
                or result.get("input_sha256", cases[case_id]["input_sha256"])
                != cases[case_id]["input_sha256"])
            if malformed:
                # The raw response and metering are already persisted by ai_jobs.
                # Keep the disagreement unresolved instead of losing a closed
                # HTTP operation or inventing a verdict from malformed output.
                result = {"verdict": "unable_to_judge", "supporting_quote": "",
                    "quote_doc_id": "", "reason": "Adjudicator output failed local identity/schema validation.",
                    "confidence_1_to_3": 1}
            else:
                # Some models echo the case identity despite the exact-field
                # instruction. Verify those optional echoes, then normalize to
                # the canonical result schema before validation and persistence.
                result = {key: result[key] for key in required_fields}
                # Reuse the strict evidence checker with a fully covered temporary
                # view; malformed quotes fail closed to unresolved.
                temporary = {**response_a, "annotations": [({**r, **result} if r["case_id"] == case_id else r) for r in response_a["annotations"]]}
                try:
                    validate_review(packet_a, temporary)
                except ValueError:
                    result = {"verdict": "unable_to_judge", "supporting_quote": "",
                        "quote_doc_id": "", "reason": "Adjudicator citation failed exact-source validation.",
                        "confidence_1_to_3": 1}
            rows.append({"case_id": case_id, "input_sha256": cases[case_id]["input_sha256"], **result,
                "provenance": provenance, "original_pair_sha256": sha256_obj([a[case_id], b[case_id]])})
        return {"schema": "ai-grounding-adjudications-v1", "reference_report_sha256": report["report_sha256"], "rows": rows}
    return execute(plan, inputs, cfg, worker, provider=provider)


def resolve(report, packet_a, packet_b, response_a, response_b, adjudications):
    reproduced = score_references(packet_a, packet_b, response_a, response_b)
    if (report != reproduced or adjudications.get("schema") != "ai-grounding-adjudications-v1"
            or adjudications.get("reference_report_sha256") != report["report_sha256"]):
        raise ValueError("AI grounding adjudication does not bind original reference report")
    original_a, original_b = ({r["case_id"]: r for r in response["annotations"]} for response in (response_a, response_b))
    expected = {r["case_id"] for r in report["per_case"] if r["consensus_verdict"] is None}
    rows = adjudications.get("rows", [])
    indexed = {r["case_id"]: r for r in rows}
    unresolved_cases = adjudications.get("unresolved_case_ids", [])
    if (len(indexed) != len(rows) or not isinstance(unresolved_cases, list)
            or len(unresolved_cases) != len(set(unresolved_cases))
            or set(indexed) & set(unresolved_cases)
            or set(indexed) | set(unresolved_cases) != expected):
        raise ValueError("AI adjudication must resolve or explicitly preserve every disagreement")
    original_runs = {a["review_run_id"] for a in report["reviewer_provenance"]}
    for case_id, row in indexed.items():
        provenance = row.get("provenance") or {}
        if (provenance.get("review_run_id") in original_runs | {None, ""}
                or provenance.get("other_review_outputs_consulted") is not True or provenance.get("blind") is not True
                or row.get("original_pair_sha256") != sha256_obj([original_a[case_id], original_b[case_id]])):
            raise ValueError("AI grounding adjudication lacks a separate context bound to original judgments")
        selected = {k: row[k] for k in ("verdict", "supporting_quote", "quote_doc_id", "reason", "confidence_1_to_3", "input_sha256")}
        temporary = {**response_a, "annotations": [({**r, **selected} if r["case_id"] == case_id else r) for r in response_a["annotations"]]}
        validate_review(packet_a, temporary)
    revised = []
    for row in report["per_case"]:
        if row["case_id"] in indexed:
            verdict = indexed[row["case_id"]]["verdict"]
            resolution_source = "third_ai_context"
        elif row["consensus_verdict"] is not None:
            verdict = row["consensus_verdict"]
            resolution_source = "two_ai_agreement"
        else:
            verdict = "unable_to_judge"
            resolution_source = "transport_unresolved"
        revised.append({**row, "resolved_verdict": verdict, "resolved_usable": verdict in LABELS[:-1],
                        "resolution_source": resolution_source,
                        "resolution_note": adjudications.get("unresolved_reason", "")
                        if resolution_source == "transport_unresolved" else ""})
    body = {**report, "schema": "ai-grounding-resolved-reference-v1", "original_agreement_report_sha256": report["report_sha256"],
        "adjudications_sha256": sha256_obj(adjudications), "per_case": revised,
        "n_resolved_usable": sum(r["resolved_usable"] for r in revised),
        "n_adjudicated": len(indexed), "n_unresolved": sum(not r["resolved_usable"] for r in revised)}
    body.pop("report_sha256")
    return {**body, "report_sha256": sha256_obj(body)}
