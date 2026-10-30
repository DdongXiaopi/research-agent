"""AI-reference calibration is measured separately from annotation creation."""
import copy
import json
from pathlib import Path

from ..util import sha256_file, sha256_obj, read_json, write_json
from .grounding import (proposition_judge_messages, parse_proposition_verdict,
                        proposition_judge_token_budget,
                        proposition_judge_response_format)
from .grounding_v10 import (evidence_selection_messages,
                            scope_classification_messages, parse_v10_verdict)
from .grounding_v11 import (evidence_selection_messages_v11,
                            scope_classification_messages_v11, parse_v11_verdict)
from .grounding_v12 import claim_relation_messages_v12, parse_v12_verdict, _direct_relations
from .grounding_v13 import claim_relation_messages_v13, parse_v13_verdict
from .grounding_v14 import claim_relation_messages_v14, parse_v14_verdict
from .grounding_v15 import claim_relation_messages_v15, parse_v15_verdict
from .grounding_v16 import claim_relation_messages_v16, prediction_v16
from . import ai_jobs

LABELS = ("supported", "contradicted", "insufficient")
THRESHOLDS = {"minimum_usable": 30, "minimum_per_class": 3, "macro_f1": 0.80,
              "minimum_class_recall": 0.75, "false_support_rate": 0.10}
# Exact v14 bundle used by the completed 2026-10-16 job. Prompts are unchanged;
# v14 now ignores two known locator metadata keys, allowing a safe offline rescore
# after exact prompt and response identity checks.
LEGACY_V14_GROUNDING_SOURCE_SHA256_20261016 = (
    "db68cc0246940d08b037be12fe4bc2474859152c5bf4771c0307c7f1c80155a0"
)


def runtime_model_config(cfg):
    model = copy.deepcopy((cfg.get("models") or {}).get("task") or {})
    semantic = (cfg.get("evaluation") or {}).get("semantic_citation") or {}
    if semantic.get("model_id"):
        model["model_id"] = semantic["model_id"]
    return model


def model_protocol(cfg):
    model = runtime_model_config(cfg)
    return sha256_obj({k: model.get(k) for k in ("model_id", "base_url", "temperature", "enable_thinking", "extra_body")})


def _prompt_version(cfg):
    semantic = (cfg.get("evaluation") or {}).get("semantic_citation") or {}
    return semantic.get("prompt_version", "v4")


def _judge_config(cfg):
    value = copy.deepcopy(cfg)
    value.setdefault("models", {})["judge"] = runtime_model_config(cfg)
    semantic = value.setdefault("evaluation", {}).setdefault("semantic_citation", {})
    semantic.setdefault("enabled", True)
    semantic.setdefault("mode", "proposition")
    semantic.setdefault("max_doc_chars", 6000)
    semantic["prompt_version"] = _prompt_version(cfg)
    return value


def _grounding_source_sha256(prompt_version):
    root = Path(__file__).resolve().parent
    files = {"grounding.py": sha256_file(root / "grounding.py")}
    if prompt_version == "v10":
        files["grounding_v10.py"] = sha256_file(root / "grounding_v10.py")
    if prompt_version == "v11":
        files["grounding_v11.py"] = sha256_file(root / "grounding_v11.py")
    if prompt_version == "v12":
        files["grounding_v11.py"] = sha256_file(root / "grounding_v11.py")
        files["grounding_v12.py"] = sha256_file(root / "grounding_v12.py")
    if prompt_version == "v13":
        files["grounding_v11.py"] = sha256_file(root / "grounding_v11.py")
        files["grounding_v12.py"] = sha256_file(root / "grounding_v12.py")
        files["grounding_v13.py"] = sha256_file(root / "grounding_v13.py")
    if prompt_version == "v14":
        files["grounding_v11.py"] = sha256_file(root / "grounding_v11.py")
        files["grounding_v12.py"] = sha256_file(root / "grounding_v12.py")
        files["grounding_v13.py"] = sha256_file(root / "grounding_v13.py")
        files["grounding_v14.py"] = sha256_file(root / "grounding_v14.py")
    if prompt_version == "v15":
        for name in ("grounding_v11.py", "grounding_v12.py", "grounding_v13.py",
                     "grounding_v14.py", "grounding_v15.py"):
            files[name] = sha256_file(root / name)
    if prompt_version == "v16":
        for name in ("grounding_v11.py", "grounding_v12.py", "grounding_v13.py",
                     "grounding_v14.py", "grounding_v15.py", "grounding_v16.py"):
            files[name] = sha256_file(root / name)
    return sha256_obj(files)


def _scoring_version(prompt_version):
    if prompt_version == "v10":
        return "runtime-equivalent-v3-grounding-v10"
    if prompt_version == "v11":
        return "runtime-equivalent-v4-grounding-v11"
    if prompt_version == "v12":
        return "runtime-equivalent-v5-grounding-v12"
    if prompt_version == "v13":
        return "runtime-equivalent-v6-grounding-v13"
    if prompt_version == "v14":
        return "runtime-equivalent-v9-grounding-v14-document-relation"
    if prompt_version == "v15":
        return "runtime-equivalent-v10-grounding-v15-explicit-evidence-basis"
    if prompt_version == "v16":
        return "calibration-only-v12-grounding-v16-healthfc-native-direction-v2"
    return "runtime-equivalent-v2"


def calibration_task_contract(packet):
    """Describe the proposition target used by grounding calibration inputs.

    Existing claim/evidence packets assess the claim carried in ``question``;
    they do not provide the answer selected by the production task model.
    Future answer-support calibration must opt into a separately implemented
    packet/prompt contract instead of inheriting this default.
    """
    # The current calibration executor only passes case.question + evidence
    # to the judge. It has no answer field, so packet metadata cannot relabel
    # that measurement as selected-answer support.
    return "claim_veracity_v1"


def runtime_scorer_sha256():
    """Bind calibration to the production outcome/scoring implementation."""
    from ..runtime import runner
    from . import metrics
    return sha256_obj({
        "runner.py": sha256_file(Path(runner.__file__).resolve()),
        "metrics.py": sha256_file(Path(metrics.__file__).resolve()),
    })


def calibration_evidence_scope(packet):
    counts = [len(case.get("evidence") or []) for case in packet.get("cases") or []]
    if not counts:
        raise ValueError("grounding calibration has no evidence cases")
    if all(count == 1 for count in counts):
        return "single_doc"
    if all(1 <= count <= 3 for count in counts) and max(counts) == 3:
        return "top3"
    raise ValueError("grounding calibration evidence scope is neither single_doc nor top3")


def _grounding_source_compatible(actual_sha256, prompt_version):
    allowed = {_grounding_source_sha256(prompt_version)}
    if prompt_version == "v14":
        allowed.add(LEGACY_V14_GROUNDING_SOURCE_SHA256_20261016)
    return actual_sha256 in allowed


def _two_stage(prompt_version):
    return prompt_version in {"v10", "v11", "v12", "v13", "v14", "v15", "v16"}


def _calls_per_case(prompt_version):
    return 3 if prompt_version == "v12" else 2 if _two_stage(prompt_version) else 1


def build_plan(packet, reference, cfg, *, output_mode="schema", **budget):
    _validate_inputs(packet, reference)
    if output_mode not in {"schema", "json_object"}:
        raise ValueError("grounding output mode must be schema or json_object")
    prompt_version = _prompt_version(cfg)
    response_format = ({"type": "json_object"} if _two_stage(prompt_version) else
                       proposition_judge_response_format(prompt_version)
                       if output_mode == "schema" else {"type": "json_object"})
    calls_per_case = _calls_per_case(prompt_version)
    required_calls = len(packet["cases"]) * calls_per_case
    requested_calls = budget.pop("max_calls", required_calls)
    if _two_stage(prompt_version) and requested_calls != required_calls:
        raise ValueError("multi-stage grounding calibration requires its exact planned call count")
    return ai_jobs.build_plan("grounding_judge", {"packet": packet, "reference": reference}, _judge_config(cfg),
        max_calls=requested_calls,
        tokens=proposition_judge_token_budget(_prompt_version(cfg)),
        response_format=response_format,
        **budget)


def _validate_inputs(packet, reference):
    from scripts.score_task_input_grounding_human_reference import _validate_packet
    cases = _validate_packet(packet, packet["reviewer_slot"])
    source_native = reference.get("schema") in {
        "healthver-source-grounding-reference-v1",
        "healthver-source-top3-grounding-reference-v1",
    }
    if source_native:
        from .healthver_reference import SOURCE_TO_GROUNDING, build_reference
        from ..data.healthver_task_input import SOURCE_SHA256

        expected_schema = ("healthver-source-top3-grounding-reference-v1"
                           if len(next(iter(cases.values()))["evidence"]) == 3
                           else "healthver-source-grounding-reference-v1")
        if (reference.get("schema") != expected_schema
                or packet.get("protocol_context", {}).get("dataset") != "healthver"
                or reference.get("annotation_origin") != "upstream_manual"
                or reference.get("dataset") != "healthver"
                or reference.get("split") != "dev_optimize"
                or reference.get("source_archive_sha256") != SOURCE_SHA256
                or reference.get("packet_sha256") != packet.get("packet_sha256")
                or reference.get("protocol_sha256") != packet.get("protocol_sha256")
                or reference.get("reference_sha256") != sha256_obj(
                    {k: v for k, v in reference.items() if k != "reference_sha256"})):
            raise ValueError("HealthVer source-native reference identity or provenance mismatch")
        for row in reference.get("per_case", []):
            if row.get("resolved_verdict") not in set(SOURCE_TO_GROUNDING.values()):
                raise ValueError("HealthVer source-native reference has an invalid label")
        if reference != build_reference(packet):
            raise ValueError("HealthVer source-native labels do not reproduce from the audited source archive")
    else:
        if (reference.get("schema") != "ai-grounding-resolved-reference-v1"
                or reference.get("annotation_origin") != "ai"
                or reference.get("protocol_sha256") != packet["protocol_sha256"]
                or reference.get("report_sha256") != sha256_obj(
                    {k: v for k, v in reference.items() if k != "report_sha256"})):
            raise ValueError("grounding requires a resolved, bound AI or source-native reference")
        provenance = reference.get("reviewer_provenance")
        rubric_hashes = ([row.get("rubric_sha256") for row in provenance]
                         if isinstance(provenance, list) and len(provenance) == 2
                         and all(isinstance(row, dict) for row in provenance) else [])
        expected_rubric = (packet.get("protocol_context") or {}).get("rubric_file_sha256")
        if (len(rubric_hashes) != 2 or any(not isinstance(value, str) or len(value) != 64
                                          for value in rubric_hashes)
                or len(set(rubric_hashes)) != 1
                or expected_rubric is not None and rubric_hashes[0] != expected_rubric):
            raise ValueError("grounding AI reference reviewers used different or unbound rubrics")
    rows = reference.get("per_case", [])
    if len(rows) != len(cases) or {r["case_id"] for r in rows} != set(cases):
        raise ValueError("grounding reference calibration input coverage changed")
    for row in rows:
        if row["input_sha256"] != cases[row["case_id"]]["input_sha256"]:
            raise ValueError("AI calibration reference evidence changed")


def summarize(rows):
    matrix = {a: {b: 0 for b in (*LABELS, "invalid_output")} for a in LABELS}
    excluded = 0
    for row in rows:
        if row["reference"] not in LABELS:
            excluded += 1
            continue
        if row["prediction"] not in (*LABELS, "invalid_output"):
            raise ValueError("invalid grounding prediction category")
        matrix[row["reference"]][row["prediction"]] += 1
    per_class = {}
    for label in LABELS:
        support = sum(matrix[label].values())
        predicted = sum(r[label] for r in matrix.values())
        true = matrix[label][label]
        recall = true/support if support else 0
        precision = true/predicted if predicted else 0
        per_class[label] = {"support": support, "precision": precision, "recall": recall,
                            "f1": 2*precision*recall/(precision+recall) if precision+recall else 0}
    n = sum(sum(r.values()) for r in matrix.values())
    negatives = sum(sum(matrix[label].values()) for label in LABELS if label != "supported")
    false_support = sum(matrix[label]["supported"] for label in LABELS if label != "supported")
    invalid = sum(matrix[label]["invalid_output"] for label in LABELS)
    macro = sum(v["f1"] for v in per_class.values())/3
    rate = false_support/negatives if negatives else 1
    passed = (n >= THRESHOLDS["minimum_usable"] and invalid == 0
        and all(v["support"] >= THRESHOLDS["minimum_per_class"] for v in per_class.values())
        and macro >= THRESHOLDS["macro_f1"]
        and min(v["recall"] for v in per_class.values()) >= THRESHOLDS["minimum_class_recall"]
        and rate <= THRESHOLDS["false_support_rate"])
    return {"matrix": matrix, "per_class": per_class, "n_usable": n, "n_excluded": excluded,
            "macro_f1": macro, "false_support_rate": rate, "invalid_outputs": invalid, "calibration_pass": passed}


def prediction_for(raw, evidence, prompt_version="v4"):
    if prompt_version in {"v10", "v11", "v12", "v13", "v14", "v15", "v16"}:
        expected = ({"evidence_selection", "claim_relation"} if prompt_version in {"v13", "v14", "v15", "v16"} else
                    {"evidence_selection", "scope_classification", "claim_relation"}
                    if prompt_version == "v12" else {"evidence_selection", "scope_classification"})
        if not isinstance(raw, dict) or set(raw) != expected:
            return "invalid_output"
        selection = raw["evidence_selection"]
        if prompt_version in {"v13", "v14", "v15", "v16"}:
            if prompt_version == "v16":
                return prediction_v16(raw["claim_relation"], evidence)
            parser = (parse_v15_verdict if prompt_version == "v15" else
                      parse_v14_verdict if prompt_version == "v14" else parse_v13_verdict)
            positive, positive_reason = parser(
                selection, raw["claim_relation"], evidence, "yes")
            negative, negative_reason = parser(
                selection, raw["claim_relation"], evidence, "no")
            return ("supported" if positive and not negative else
                    "contradicted" if negative and not positive else
                    "insufficient" if positive_reason == negative_reason == "proposition_uncertain"
                    else "invalid_output")
        if prompt_version == "v12":
            positive, positive_reason = parse_v12_verdict(
                raw["evidence_selection"], raw["scope_classification"], raw["claim_relation"], evidence, "yes")
            negative, negative_reason = parse_v12_verdict(
                raw["evidence_selection"], raw["scope_classification"], raw["claim_relation"], evidence, "no")
            return ("supported" if positive and not negative else
                    "contradicted" if negative and not positive else
                    "insufficient" if positive_reason == negative_reason == "proposition_uncertain"
                    else "invalid_output")
        parser = parse_v11_verdict if prompt_version == "v11" else parse_v10_verdict
        positive, positive_reason = parser(raw["evidence_selection"], raw["scope_classification"], evidence, "yes")
        negative, negative_reason = parser(raw["evidence_selection"], raw["scope_classification"], evidence, "no")
        return ("supported" if positive and not negative else
                "contradicted" if negative and not positive else
                "insufficient" if positive_reason == negative_reason == "proposition_uncertain"
                else "invalid_output")
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    positive, positive_reason = parse_proposition_verdict(
        text, evidence, "yes", prompt_version=prompt_version)
    negative, negative_reason = parse_proposition_verdict(
        text, evidence, "no", prompt_version=prompt_version)
    return ("supported" if positive and not negative else "contradicted" if negative and not positive
        else "insufficient" if positive_reason == negative_reason == "proposition_uncertain" else "invalid_output")


def document_relation_rows(per_case_rows, packet, reference, *, max_doc_chars=6000):
    """Score HealthVer's native claim-abstract relation unit from measured outputs."""
    packet_cases = {case["case_id"]: case for case in packet["cases"]}
    reference_cases = {row["case_id"]: row for row in reference["per_case"]}
    scored = []
    for row in per_case_rows:
        case = packet_cases[row["case_id"]]
        docs = [{**doc, "text": str(doc.get("text", ""))[:max_doc_chars]}
                for doc in case["evidence"]]
        ref_case = reference_cases[row["case_id"]]
        doc_refs = {item["doc_id"]: item["resolved_verdict"]
                    for item in ref_case.get("document_relations", [])}
        raw = row.get("raw_verdict")
        relations, reason = _direct_relations(
            raw.get("claim_relation") if isinstance(raw, dict) else None,
            docs, max_doc_chars)
        valid = reason in {"valid", "conflicting_document_relations"}
        raw_relation = raw.get("claim_relation") if isinstance(raw, dict) else None
        raw_relation = ai_jobs.decode_json_response(raw_relation) if isinstance(raw_relation, str) else raw_relation
        predictions = ({item["doc_id"]: item["verdict"]
                       for item in raw_relation["results"]}
                      if valid and isinstance(raw_relation, dict)
                      and isinstance(raw_relation.get("results"), list) else {})
        for doc in docs:
            doc_id = str(doc["doc_id"])
            if doc_id not in doc_refs:
                raise ValueError("source reference is missing a document relation")
            scored.append({
                "case_id": row["case_id"],
                "doc_id": doc_id,
                "input_sha256": row["input_sha256"],
                "reference": doc_refs[doc_id],
                "prediction": predictions.get(doc_id, "invalid_output"),
            })
    return scored


def _calibration_metrics(per_case_rows, packet, reference, *, max_doc_chars=6000):
    collection_summary = summarize(per_case_rows)
    if reference.get("schema") == "healthver-source-top3-grounding-reference-v1":
        documents = document_relation_rows(per_case_rows, packet, reference,
                                           max_doc_chars=max_doc_chars)
        summary = summarize(documents)
        return summary, {"per_document": documents, **summary}, collection_summary
    return collection_summary, None, collection_summary


def execute(packet, reference, cfg, plan, *, provider=None):
    if plan.get("kind") != "grounding_judge":
        raise ValueError("AI calibration requires its grounding_judge plan")
    _validate_inputs(packet, reference)
    prompt_version = _prompt_version(cfg)
    references = {r["case_id"]: r["resolved_verdict"] for r in reference["per_case"]}
    def worker(call):
        rows = []
        for case in packet["cases"]:
            presented = [{**d, "text": str(d.get("text", ""))[:6000]} for d in case["evidence"]]
            if _two_stage(prompt_version):
                select_messages = (evidence_selection_messages_v11 if prompt_version in {"v11", "v12", "v13", "v14", "v15", "v16"}
                                   else evidence_selection_messages)
                classify_messages = (scope_classification_messages_v11 if prompt_version in {"v11", "v12"}
                                     else scope_classification_messages)
                system, user = select_messages(case["question"], presented)
                evidence_selection, provenance_a = call(
                    "grounding_judge", {"runtime_user_message": user}, system)
                if prompt_version in {"v13", "v14", "v15", "v16"}:
                    relation_messages = (claim_relation_messages_v16 if prompt_version == "v16" else
                                         claim_relation_messages_v15 if prompt_version == "v15" else
                                         claim_relation_messages_v14 if prompt_version == "v14"
                                         else claim_relation_messages_v13)
                    system2, user2 = relation_messages(
                        case["question"], presented, evidence_selection)
                    claim_relation, provenance_b = call(
                        "grounding_judge", {"runtime_user_message": user2}, system2)
                    raw = {"evidence_selection": evidence_selection,
                           "claim_relation": claim_relation}
                    provenance = {"stages": [provenance_a, provenance_b]}
                    prediction = prediction_for(raw, presented, prompt_version)
                    rows.append({"case_id": case["case_id"], "input_sha256": case["input_sha256"],
                                 "reference": references[case["case_id"]], "prediction": prediction,
                                 "raw_verdict": raw, "provenance": provenance})
                    continue
                system2, user2 = classify_messages(
                    case["question"], presented, evidence_selection)
                scope_classification, provenance_b = call(
                    "grounding_judge", {"runtime_user_message": user2}, system2)
                raw = {"evidence_selection": evidence_selection,
                       "scope_classification": scope_classification}
                stage_provenance = [provenance_a, provenance_b]
                if prompt_version == "v12":
                    system3, user3 = claim_relation_messages_v12(case["question"], presented)
                    claim_relation, provenance_c = call(
                        "grounding_judge", {"runtime_user_message": user3}, system3)
                    raw["claim_relation"] = claim_relation
                    stage_provenance.append(provenance_c)
                provenance = {"stages": stage_provenance}
            else:
                system, user = proposition_judge_messages(case["question"], case["evidence"], prompt_version=prompt_version)
                raw, provenance = call("grounding_judge", {"runtime_user_message": user}, system)
            prediction = prediction_for(raw, presented, prompt_version)
            rows.append({"case_id": case["case_id"], "input_sha256": case["input_sha256"],
                         "reference": references[case["case_id"]], "prediction": prediction,
                         "raw_verdict": raw, "provenance": provenance})
        primary_metrics, document_metrics, collection_metrics = _calibration_metrics(
            rows, packet, reference)
        result = {"schema": "ai-grounding-calibration-v1",
            "dataset": packet["protocol_context"]["dataset"], "split": "dev_optimize",
            "task_contract": calibration_task_contract(packet),
            "evidence_scope": calibration_evidence_scope(packet),
            "runtime_scorer_sha256": runtime_scorer_sha256(),
            "annotation_origin": reference["annotation_origin"],
            "reference_sha256": reference.get("report_sha256", reference.get("reference_sha256")),
            "packet_sha256": packet["packet_sha256"], "model_id": plan["model_id"],
            "model_protocol_sha256": model_protocol(cfg), "prompt_version": prompt_version, "max_doc_chars": 6000,
            "grounding_source_sha256": _grounding_source_sha256(prompt_version),
            "thresholds": THRESHOLDS, "per_case": rows, **primary_metrics,
            "collection_level_metrics": collection_metrics,
            "document_relation_calibration": document_metrics,
            "promotion_authorized": False}
        return {**result, "report_sha256": sha256_obj(result)}
    result = ai_jobs.execute(plan, {"packet": packet, "reference": reference}, _judge_config(cfg), worker, provider=provider)
    # The public report binds the private job receipt, rather than mutating
    # the completed private output and invalidating its content identity.
    public = {**result["output"], "scoring_version": _scoring_version(prompt_version),
        "scoring_source_sha256": sha256_file(__file__), "test_only": result["receipt"]["test_only"],
        "job_run_dir": result["run_dir"], "job_receipt_sha256": sha256_obj(result["receipt"])}
    public["report_sha256"] = sha256_obj({k: v for k, v in public.items() if k != "report_sha256"})
    write_json(Path(result["run_dir"]) / "calibration.json", public)
    return {**result, "calibration": public}


def verify_report(path, cfg, dataset):
    report = read_json(path)
    semantic = (cfg.get("evaluation") or {}).get("semantic_citation") or {}
    prompt_version = _prompt_version(cfg)
    if (report.get("schema") != "ai-grounding-calibration-v1" or report.get("dataset") != dataset
            or report.get("task_contract") != "claim_veracity_v1"
            or report.get("runtime_scorer_sha256") != runtime_scorer_sha256()
            or report.get("scoring_version") != _scoring_version(prompt_version)
            or report.get("scoring_source_sha256") != sha256_file(__file__)
            or semantic.get("enabled") is not True or semantic.get("mode", "answer") != "proposition"
            or semantic.get("max_doc_chars", 6000) != 6000
            or report.get("split") != "dev_optimize"
            or report.get("annotation_origin") not in {"ai", "upstream_manual"}
            or report.get("test_only") is not False or report.get("thresholds") != THRESHOLDS
            or report.get("model_protocol_sha256") != model_protocol(cfg)
            or report.get("prompt_version") != prompt_version or report.get("max_doc_chars") != 6000
            or report.get("grounding_source_sha256") != _grounding_source_sha256(prompt_version)
            or report.get("report_sha256") != sha256_obj({k: v for k, v in report.items() if k != "report_sha256"})):
        raise ValueError("AI grounding calibration identity/scope/model/source mismatch")
    root = Path(report["job_run_dir"])
    from ..util import artifacts_path
    if not root.resolve().is_relative_to(artifacts_path("evaluation_only", "ai_annotations").resolve()):
        raise ValueError("calibration receipt must be a private evaluation job")
    ai_jobs.verify_completed(root, kind="grounding_judge")
    receipt = read_json(root / "receipt.json")
    actual = read_json(root / "output.json")
    if (len(actual.get("per_case", [])) != len(report["per_case"])
            or (actual.get("per_case") != report["per_case"] and
                (report.get("recomputed_from_completed_job") is not True
                 or report.get("original_scored_output_sha256") != receipt["output_sha256"]))):
        raise ValueError("corrected calibration must declare its bound recomputation")
    inputs = read_json(root / "inputs.json")
    plan = read_json(root / "plan.json")
    if (report.get("task_contract") != calibration_task_contract(inputs["packet"])
            or report.get("evidence_scope") != calibration_evidence_scope(inputs["packet"])):
        raise ValueError("AI grounding calibration task target/evidence scope mismatch")
    _validate_inputs(inputs["packet"], inputs["reference"])
    primary_summary, document_metrics, collection_summary = _calibration_metrics(
        report["per_case"], inputs["packet"], inputs["reference"])
    if (any(report.get(k) != v for k, v in primary_summary.items())
            or report.get("collection_level_metrics") != collection_summary
            or report.get("document_relation_calibration") != document_metrics
            or not primary_summary["calibration_pass"]):
        raise ValueError("AI grounding calibration did not pass frozen measurements")
    if (sha256_obj(inputs) != receipt["inputs_sha256"] or plan["inputs_sha256"] != receipt["inputs_sha256"]
            or plan["kind"] != "grounding_judge" or plan["model_id"] != runtime_model_config(cfg)["model_id"]
            or plan["max_tokens"] != proposition_judge_token_budget(prompt_version)
            or plan["max_calls"] != len(inputs["packet"]["cases"]) * _calls_per_case(prompt_version)
            or plan["plan_sha256"] != receipt["plan_sha256"]
            or plan["plan_sha256"] != sha256_obj({k: v for k, v in plan.items() if k != "plan_sha256"})
            or report["reference_sha256"] != inputs["reference"].get(
                "report_sha256", inputs["reference"].get("reference_sha256"))
            or report["packet_sha256"] != inputs["packet"]["packet_sha256"]):
        raise ValueError("calibration input/plan identity changed")
    refs = {r["case_id"]: r["resolved_verdict"] for r in inputs["reference"]["per_case"]}
    cases = inputs["packet"]["cases"]
    if len(cases) != len(report["per_case"]):
        raise ValueError("calibration measured coverage changed")
    for index, (case, row) in enumerate(zip(cases, report["per_case"])):
        presented = [{**d, "text": str(d.get("text", ""))[:6000]} for d in case["evidence"]]
        if _two_stage(prompt_version):
            offset = _calls_per_case(prompt_version) * index
            calls_for_case = [read_json(root / f"call-{offset}.json"),
                              read_json(root / f"call-{offset+1}.json")]
            select_messages = (evidence_selection_messages_v11 if prompt_version in {"v11", "v12", "v13", "v14", "v15", "v16"}
                               else evidence_selection_messages)
            system, user = select_messages(case["question"], presented)
            if prompt_version in {"v13", "v14", "v15", "v16"}:
                try:
                    selected_for_prompt = ai_jobs.decode_json_response(calls_for_case[0]["raw_output"])
                except (ValueError, TypeError):
                    selected_for_prompt = calls_for_case[0]["raw_output"]
                relation_messages = (claim_relation_messages_v16 if prompt_version == "v16" else
                                     claim_relation_messages_v15 if prompt_version == "v15" else
                                     claim_relation_messages_v14 if prompt_version == "v14"
                                     else claim_relation_messages_v13)
                system2, user2 = relation_messages(
                    case["question"], presented, selected_for_prompt)
            else:
                classify_messages = (scope_classification_messages_v11 if prompt_version in {"v11", "v12"}
                                     else scope_classification_messages)
                system2, user2 = classify_messages(
                    case["question"], presented, row["raw_verdict"]["evidence_selection"])
            prompts = [(system, user), (system2, user2)]
            if prompt_version == "v12":
                system3, user3 = claim_relation_messages_v12(case["question"], presented)
                calls_for_case.append(read_json(root / f"call-{offset+2}.json"))
                prompts.append((system3, user3))
            parsed = []
            valid = True
            for call_record in calls_for_case:
                try:
                    parsed.append(ai_jobs.decode_json_response(call_record["raw_output"]))
                except (ValueError, TypeError):
                    parsed.append(call_record["raw_output"])
            valid = all(c["role"] == "grounding_judge" and
                        (c["system"], c["user"]) == prompt and
                        c["payload"] == {"runtime_user_message": prompt[1]}
                        for c, prompt in zip(calls_for_case, prompts))
            parsed_raw = {"evidence_selection": parsed[0]}
            if prompt_version in {"v13", "v14", "v15", "v16"}:
                parsed_raw["claim_relation"] = parsed[1]
            else:
                parsed_raw["scope_classification"] = parsed[1]
            if prompt_version == "v12":
                parsed_raw["claim_relation"] = parsed[2]
        else:
            raw = read_json(root / f"call-{index}.json")
            system, user = proposition_judge_messages(case["question"], case["evidence"], prompt_version=prompt_version)
            try:
                # Match ai_jobs' exact response decoder so complete JSON fences
                # remain reproducible instead of being counted as invalid output.
                parsed_raw = ai_jobs.decode_json_response(raw["raw_output"])
            except (ValueError, TypeError):
                parsed_raw = raw["raw_output"]
            valid = (raw["role"] == "grounding_judge" and raw["system"] == system
                     and raw["user"] == user)
        if (row["case_id"] != case["case_id"] or row["input_sha256"] != case["input_sha256"]
                or row["reference"] != refs[case["case_id"]] or not valid
                or parsed_raw != row["raw_verdict"]
                or prediction_for(row["raw_verdict"], presented, prompt_version) != row["prediction"]):
            raise ValueError("calibration prediction no longer reproduces measured runtime messages")
    if (read_json(root / "status.json").get("status") != "completed" or receipt.get("test_only") is not False
            or sha256_obj(receipt) != report["job_receipt_sha256"] or sha256_obj(actual) != receipt["output_sha256"]
            or actual.get("per_case") != [{**r, "prediction": old["prediction"]} for r, old in zip(report["per_case"], actual.get("per_case", []))]
            or receipt["call_count"] != len(report["per_case"]) * _calls_per_case(prompt_version)
            or receipt["request_accounting"]["physical_incomplete"] != 0):
        raise ValueError("AI grounding calibration lacks exact completed measured receipts")
    return report


def recompute_completed_job(run_dir, cfg):
    """Correct scoring from immutable measured responses; never calls a model."""
    from ..util import artifacts_path
    from ..runtime.call_ledger import summarize_ledger
    root = Path(run_dir).resolve()
    if not root.is_relative_to(artifacts_path("evaluation_only", "ai_annotations").resolve()):
        raise ValueError("recomputation requires a private completed evaluation job")
    ai_jobs.verify_completed(root, kind="grounding_judge")
    actual, receipt, plan, inputs = (read_json(root/name) for name in ("output.json", "receipt.json", "plan.json", "inputs.json"))
    _validate_inputs(inputs["packet"], inputs["reference"])
    prompt_version = _prompt_version(cfg)
    n = len(inputs["packet"]["cases"])
    if (read_json(root/"status.json").get("status") != "completed" or receipt.get("test_only") is not False
            or plan.get("kind") != "grounding_judge" or plan.get("max_tokens") != proposition_judge_token_budget(prompt_version)
            or actual.get("model_protocol_sha256") != model_protocol(cfg)
            or not _grounding_source_compatible(
                actual.get("grounding_source_sha256"), prompt_version)
            or actual.get("thresholds") != THRESHOLDS or actual.get("prompt_version") != prompt_version
            or actual.get("max_doc_chars") != 6000 or sha256_obj(actual) != receipt["output_sha256"]
            or sha256_obj(inputs) != receipt["inputs_sha256"] or plan["inputs_sha256"] != receipt["inputs_sha256"]
            or plan["plan_sha256"] != receipt["plan_sha256"]
            or plan["plan_sha256"] != sha256_obj({k:v for k,v in plan.items() if k != "plan_sha256"})
            or receipt["call_count"] != n * _calls_per_case(prompt_version)
            or len(actual["per_case"]) != n
            or receipt["request_accounting"] != summarize_ledger(root/"call-ledger.jsonl")
            or receipt["request_accounting"]["physical_incomplete"] != 0
            or receipt["request_accounting"]["logical_incomplete"] != 0):
        raise ValueError("recomputation requires exact completed, bound measured responses")
    refs = {r["case_id"]:r["resolved_verdict"] for r in inputs["reference"]["per_case"]}
    rows = []
    for index, (case, row) in enumerate(zip(inputs["packet"]["cases"], actual["per_case"])):
        presented = [{**d, "text": str(d.get("text", ""))[:6000]} for d in case["evidence"]]
        if _two_stage(prompt_version):
            offset = _calls_per_case(prompt_version) * index
            call_a = read_json(root/f"call-{offset}.json")
            call_b = read_json(root/f"call-{offset+1}.json")
            select_messages = (evidence_selection_messages_v11 if prompt_version in {"v11", "v12", "v13", "v14", "v15", "v16"}
                               else evidence_selection_messages)
            system, user = select_messages(case["question"], presented)
            if prompt_version in {"v13", "v14", "v15", "v16"}:
                try:
                    selected_for_prompt = ai_jobs.decode_json_response(call_a["raw_output"])
                except (ValueError, TypeError):
                    selected_for_prompt = call_a["raw_output"]
                relation_messages = (claim_relation_messages_v16 if prompt_version == "v16" else
                                     claim_relation_messages_v15 if prompt_version == "v15" else
                                     claim_relation_messages_v14 if prompt_version == "v14"
                                     else claim_relation_messages_v13)
                system2, user2 = relation_messages(
                    case["question"], presented, selected_for_prompt)
            else:
                classify_messages = (scope_classification_messages_v11 if prompt_version in {"v11", "v12"}
                                     else scope_classification_messages)
                system2, user2 = classify_messages(
                    case["question"], presented, row["raw_verdict"]["evidence_selection"])
            calls_for_case = [call_a, call_b]
            prompts = [(system, user), (system2, user2)]
            if prompt_version == "v12":
                call_c = read_json(root/f"call-{offset+2}.json")
                system3, user3 = claim_relation_messages_v12(case["question"], presented)
                calls_for_case.append(call_c)
                prompts.append((system3, user3))
            try:
                parsed_a = ai_jobs.decode_json_response(call_a["raw_output"])
            except (ValueError, TypeError):
                parsed_a = call_a["raw_output"]
            try:
                parsed_b = ai_jobs.decode_json_response(call_b["raw_output"])
            except (ValueError, TypeError):
                parsed_b = call_b["raw_output"]
            parsed = {"evidence_selection": parsed_a}
            if prompt_version in {"v13", "v14", "v15", "v16"}:
                parsed["claim_relation"] = parsed_b
            else:
                parsed["scope_classification"] = parsed_b
            if prompt_version == "v12":
                try:
                    parsed["claim_relation"] = ai_jobs.decode_json_response(call_c["raw_output"])
                except (ValueError, TypeError):
                    parsed["claim_relation"] = call_c["raw_output"]
            valid = all(c["system"] == prompt[0] and c["user"] == prompt[1]
                        and c["role"] == "grounding_judge"
                        for c, prompt in zip(calls_for_case, prompts))
        else:
            call = read_json(root/f"call-{index}.json")
            system, user = proposition_judge_messages(case["question"], case["evidence"], prompt_version=prompt_version)
            try:
                parsed = ai_jobs.decode_json_response(call["raw_output"])
            except (ValueError, TypeError):
                parsed = call["raw_output"]
            valid = (call["system"] == system and call["user"] == user
                     and call["role"] == "grounding_judge")
        if (not valid or parsed != row["raw_verdict"] or row["case_id"] != case["case_id"]
                or row["input_sha256"] != case["input_sha256"] or row["reference"] != refs[case["case_id"]]):
            raise ValueError("recomputation input/raw output identity changed")
        rows.append({**row,"prediction":prediction_for(row["raw_verdict"], presented, prompt_version)})
    primary_summary, document_metrics, collection_summary = _calibration_metrics(
        rows, inputs["packet"], inputs["reference"])
    result = {**actual, **primary_summary, "per_case": rows,
        "collection_level_metrics": collection_summary,
        "document_relation_calibration": document_metrics,
        "task_contract": calibration_task_contract(inputs["packet"]),
        "evidence_scope": calibration_evidence_scope(inputs["packet"]),
        "runtime_scorer_sha256": runtime_scorer_sha256(),
        "scoring_version": _scoring_version(prompt_version),
        "scoring_source_sha256": sha256_file(__file__), "recomputed_from_completed_job": True,
        "original_scored_output_sha256": receipt["output_sha256"], "test_only": False,
        "job_run_dir": str(root), "job_receipt_sha256": sha256_obj(receipt), "model_calls_during_recomputation": 0}
    result["report_sha256"] = sha256_obj({k:v for k,v in result.items() if k != "report_sha256"})
    destination = root/("calibration-recomputed-" + result["report_sha256"][:16] + ".json")
    if destination.exists() and read_json(destination) != result:
        raise ValueError("immutable recomputed report already differs")
    write_json(destination,result)
    return {"report_path":str(destination),"calibration":result}
