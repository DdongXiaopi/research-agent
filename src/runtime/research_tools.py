"""Reviewable lightweight research tools and source-identity verification."""
from pathlib import Path
from uuid import uuid4

from ..config import config_fingerprint
from ..eval.run_manifest import source_tree_fingerprint
from ..util import artifacts_path, read_json, read_jsonl, sha256_file, sha256_obj, sha256_text, write_json
from .execution import ExecutionGateway


def tool_plan(value, cfg):
    if not isinstance(value, dict) or set(value) != {"kind", "payload"} or not isinstance(value["payload"], dict):
        raise ValueError("tool request requires kind and payload")
    allowed = {"calc": {"expr", "variables"}, "latex": {"template", "values"}, "python_static": {"source"},
        "latex_compile":{"source"}, "plot":{"series","title","xlabel","ylabel","style","formats","dpi"}}
    if value["kind"] not in allowed or set(value["payload"]) - allowed[value["kind"]]:
        raise ValueError("unsupported research tool or payload; arbitrary execution is unavailable")
    primary = {"calc": "expr", "latex": "template", "python_static": "source", "latex_compile":"source"}.get(value["kind"])
    if primary and not isinstance(value["payload"].get(primary), str):
        raise ValueError("research tool source/expression/template must be text")
    for key in ("variables", "values"):
        if key in value["payload"] and not isinstance(value["payload"][key], dict):
            raise ValueError("research tool values must be an object")
    rendering = value["kind"] in {"latex_compile","plot"}
    worker_identity = None
    if rendering:
        from .worker_contract import validate_request
        from .docker_worker import settings, source_identity
        validate_request(value["kind"],value["payload"])
        options = settings(cfg)
        worker_identity = {"image":options["image"],"receipt_sha256":options["receipt_sha256"],"sources":source_identity()}
    if len(str(value["payload"])) > (300000 if rendering else 25000):
        raise ValueError("research tool input exceeds budget")
    body = {"schema": "research-tool-plan-v1", "kind": value["kind"],
        "payload_sha256": sha256_obj(value["payload"]), "config_sha256": config_fingerprint(cfg),
        "source_tree_sha256": source_tree_fingerprint()["sha256"],
        "model_calls": 0, "network_requests": 0, "arbitrary_code_execution": False}
    if rendering:
        body.update(isolated_rendering=True,host_code_execution=False,worker_identity=worker_identity)
    return {**body, "plan_sha256": sha256_obj(body)}


def execute_tool(value, cfg, expected_plan_sha256):
    plan = tool_plan(value, cfg)
    if plan["plan_sha256"] != expected_plan_sha256:
        raise ValueError("research tool differs from frozen plan")
    root = artifacts_path("research_tools", "tool-" + uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "plan.json", plan)
    write_json(root / "request.json", value)
    write_json(root / "status.json", {"status": "started"})
    try:
        result = ExecutionGateway(cfg).run(value["kind"], value["payload"])
        write_json(root / "result.json", result)
        receipt = {"schema": "research-tool-receipt-v1", "plan_sha256": plan["plan_sha256"],
            "request_sha256": sha256_file(root / "request.json"), "result_sha256": sha256_file(root / "result.json"),
            "model_calls": 0, "network_requests": 0, "arbitrary_code_execution": False}
        if plan.get("isolated_rendering"):
            worker_root = Path(result["run_dir"])
            receipt.update(worker_identity=plan["worker_identity"],host_code_execution=False,
                worker_receipt_sha256=sha256_file(worker_root/"receipt.json") if (worker_root/"receipt.json").exists() else None,
                worker_status_sha256=sha256_file(worker_root/"status.json"),worker_cleanup_sha256=sha256_file(worker_root/"cleanup.json"))
        write_json(root / "receipt.json", receipt)
        write_json(root / "status.json", {"status": "completed" if result.get("ok") else "rejected"})
        return {"run_dir": str(root), "result": result, "receipt": receipt}
    except BaseException as exc:
        write_json(root / "status.json", {"status": "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                                         "error_type": type(exc).__name__})
        raise


def verify_citations(run_dir):
    from ..evolution.session_feedback import verify_research_run
    from .research_assistant import request_from_dict, _limits, _visible_sources
    root = Path(run_dir).resolve()
    request, receipt = verify_research_run(root)
    request = request_from_dict(request)
    plan = read_json(root / "plan.json")
    if (plan.get("plan_sha256") != sha256_obj({k: v for k, v in plan.items() if k != "plan_sha256"})
            or plan.get("request_sha256") != sha256_obj(request.to_dict())):
        raise ValueError("research plan identity changed")
    visible = {r["source_id"]: r for r in _visible_sources(request, _limits(request))}
    response = read_json(root / "response.json")
    results = []
    for index, citation in enumerate(response.get("citations", [])):
        source = visible.get(citation.get("source_id"))
        quote, claim = citation.get("evidence_excerpt"), citation.get("claim")
        valid = bool(source and isinstance(quote, str) and quote.strip()
            and quote in source["text"] and quote in source["visible_text"]
            and citation.get("content_sha256") == sha256_text(source["text"])
            and all(citation.get(k, "") == source.get(k, "") for k in ("url", "locator", "retrieved_at"))
            and isinstance(claim, str) and claim.strip() and claim in response.get("answer", ""))
        results.append({"citation_index": index, "source_id": citation.get("source_id"), "identity_verified": valid})
    return {"schema": "research-citation-verification-v1", "run_receipt_sha256": sha256_obj(receipt),
        "status": "identity_verified" if all(r["identity_verified"] for r in results) else "identity_mismatch",
        "citation_count": len(results), "citations": results, "semantic_support_verified": False,
        "scientific_utility_verified": False, "network_requests": 0, "model_calls": 0}


def audit_open_research_trace(run_dir):
    """Verify the visible workflow and artifact bindings for one open-research run.

    This is consistency evidence only; it does not authenticate a fully rewritten
    run directory or record hidden model reasoning.
    """
    root = Path(run_dir).resolve()
    trace_path = root / "workflow-trace.jsonl"
    plan = read_json(root / "plan.json")
    request = read_json(root / "request.json")
    status_artifact = read_json(root / "status.json")
    terminal_status = status_artifact.get("status")
    if terminal_status in {"completed", "completed_with_parse_failure"}:
        receipt = read_json(root / "receipt.json")
        response = read_json(root / "response.json")
        raw = read_json(root / "raw-model-output.json")
        expected_trace_sha = receipt.get("workflow_trace_sha256")
        artifact_identity_ok = (
            receipt.get("schema") == "open-research-receipt-v1"
            and receipt.get("response_sha256") == sha256_file(root / "response.json")
            and receipt.get("raw_model_output_sha256") == sha256_file(root / "raw-model-output.json"))
    elif terminal_status in {"failed", "cancelled"}:
        receipt = read_json(root / "failure-receipt.json")
        response = {}
        raw = (read_json(root / "raw-model-output.json")
               if (root / "raw-model-output.json").is_file() else {})
        expected_trace_sha = receipt.get("workflow_trace_sha256")
        artifact_identity_ok = (
            receipt.get("schema") == "open-research-failure-receipt-v1"
            and receipt.get("status_sha256") == sha256_file(root / "status.json")
            and receipt.get("raw_model_output_sha256") == status_artifact.get("raw_model_output_sha256")
            and (not receipt.get("raw_model_output_sha256")
                 or receipt.get("raw_model_output_sha256") == sha256_file(root / "raw-model-output.json")))
    else:
        raise ValueError("open research run has no auditable terminal status")
    rows = read_jsonl(trace_path)
    if (not artifact_identity_ok
            or expected_trace_sha != sha256_file(trace_path)
            or receipt.get("plan_sha256") != plan.get("plan_sha256")
            or plan.get("plan_sha256") != sha256_obj({k: v for k, v in plan.items()
                                                        if k != "plan_sha256"})
            or plan.get("request_sha256") != sha256_obj(request)
            or receipt.get("request_sha256") != sha256_file(root / "request.json")):
        raise ValueError("open research workflow artifact identity changed")
    if not rows:
        raise ValueError("open research workflow trace is empty")
    run_ids = {row.get("run_id") for row in rows}
    task_ids = {row.get("task_id") for row in rows}
    if (len(run_ids) != 1 or not next(iter(run_ids))
            or task_ids != {request.get("task_id")}
            or [row.get("event_id") for row in rows]
            != [f"e{index:04d}" for index in range(1, len(rows) + 1)]):
        raise ValueError("open research workflow event identity is invalid")
    by_id = {row["event_id"]: row for row in rows}
    if len(by_id) != len(rows) or any(row.get("parent_event_id")
            and row["parent_event_id"] not in by_id for row in rows):
        raise ValueError("open research workflow parent link is invalid")
    kinds = [row.get("event_type") for row in rows]
    if len(kinds) < 4 or kinds[0:2] != ["task.start", "task.plan"] or kinds[-1] != "task.end":
        raise ValueError("open research workflow boundaries are invalid")
    starts = [row for row in rows if row.get("event_type") == "task.start"]
    plans = [row for row in rows if row.get("event_type") == "task.plan"]
    ends = [row for row in rows if row.get("event_type") == "task.end"]
    requests = [row for row in rows if row.get("event_type") == "task.answer.requested"]
    calls = [row for row in rows if row.get("tool_name") == "model.call.task"
             and row.get("event_type") in {"model.call", "model.call.failed"}]
    selected = [row for row in rows if row.get("event_type") == "task.answer.selected"]
    if not (len(starts) == len(plans) == len(ends) == 1):
        raise ValueError("open research workflow has ambiguous boundaries")
    start, plan_event, end = starts[0], plans[0], ends[0]
    if (plan_event.get("parent_event_id") != start["event_id"]
            or end.get("parent_event_id") != start["event_id"]
            or start["payload"].get("task_input_sha256") != sha256_obj(request)
            or plan_event["payload"].get("task_input_sha256") != sha256_obj(request)
            or plan_event["payload"].get("plan_sha256") != plan["plan_sha256"]
            or plan_event["payload"].get("causal_evidence") is not False
            or plan_event["payload"].get("model_reasoning_recorded") is not False):
        raise ValueError("open research workflow plan binding changed")
    if terminal_status == "completed":
        if (end["payload"].get("status") not in {"ok", "abstain"}
                or len(selected) != 1 or selected[0].get("parent_event_id") not in {
                    plan_event["event_id"], *(row["event_id"] for row in requests)}):
            raise ValueError("open research selected answer is missing or unbound")
    elif terminal_status == "completed_with_parse_failure":
        failures = [row for row in rows if row.get("event_type") == "task.answer.parse_failed"]
        if (end["payload"].get("status") != "error" or end["payload"].get("parse_failure") is not True
                or response.get("status") != "error" or response.get("execution_summary", {}).get("parse_failure") is not True
                or selected or len(failures) != 1
                or failures[0].get("payload", {}).get("raw_model_output_sha256") != sha256_file(root / "raw-model-output.json")):
            raise ValueError("open research parse failure trace is incomplete")
    elif end["payload"].get("status") != "error" or selected:
        raise ValueError("open research failure trace has an invalid terminal event")
    if len(requests) != len(calls) or len(requests) > 1:
        raise ValueError("open research answer dispatch is not closed")
    if requests:
        request_event, call = requests[0], calls[0]
        binding = request_event["payload"].get("binding")
        if (not isinstance(binding, dict)
                or request_event["payload"].get("binding_sha256") != sha256_obj(binding)
                or binding.get("scope") != "provider_complete_arguments"
                or binding.get("task_input_sha256") != sha256_obj(request)
                or any(not isinstance(binding.get(key), str)
                       or len(binding[key]) != 64
                       for key in ("system_prompt_sha256", "skill_body_sha256",
                                   "documents_sha256", "context_sha256", "channels_sha256"))
                or request_event.get("parent_event_id") != plan_event["event_id"]
                or call.get("parent_event_id") != request_event["event_id"]
                or call["payload"].get("request_event_id") != request_event["event_id"]
                or call["payload"].get("request_binding_sha256") != request_event["payload"]["binding_sha256"]
                or call.get("event_type") not in {"model.call", "model.call.failed"}
                or call.get("event_type") == "model.call"
                   and call["payload"].get("response_content_sha256") != sha256_text(raw.get("content", ""))):
            raise ValueError("open research provider call is not bound to its recorded inputs")
    if raw.get("mode") == "pi_open_research" and raw.get("pi_trace"):
        pi = raw["pi_trace"]
        if not pi.get("ephemeral"):
            from .pi_trace import read_pi_session
            session_path = Path(pi.get("session_path") or "").resolve()
            try:
                session_path.relative_to((root / "pi-sessions").resolve())
                if sha256_file(session_path) != pi.get("session_sha256"):
                    raise ValueError("Pi session hash changed")
                session = read_pi_session(session_path)
                if (session["session_id"] != pi.get("session_id")
                        or not any(row.get("role") == "assistant" and row.get("text") == raw.get("content")
                                   for row in session["messages"])):
                    raise ValueError("Pi session differs from recorded answer")
            except (OSError, ValueError) as exc:
                raise ValueError("open research Pi session verification failed") from exc
    if terminal_status == "completed":
        if (selected[0]["payload"].get("answer_sha256") != sha256_text(response.get("answer", ""))
                or selected[0]["payload"].get("status") != response.get("status")
                or selected[0]["payload"].get("citation_count") != len(response.get("citations", []))
                or end["payload"].get("response_sha256") != sha256_file(root / "response.json")
                or end["payload"].get("status") != response.get("status")):
            raise ValueError("open research answer selection differs from response artifact")
    elif terminal_status == "completed_with_parse_failure":
        if (receipt.get("response_sha256") != sha256_file(root / "response.json")
                or end["payload"].get("status") != "error"):
            raise ValueError("open research parse failure response differs from receipt")
    report = {"schema": "open-research-workflow-audit-v1", "run_id": next(iter(run_ids)),
        "task_id": request["task_id"], "trace_sha256": sha256_file(trace_path),
        "plan_sha256": plan["plan_sha256"],
        "status": end["payload"]["status"],
        "provider_calls": len(calls), "answer_selected": bool(selected),
        "model_reasoning_recorded": False, "causal_evidence": False,
        "scope": "visible open-research workflow and artifact consistency"}
    report["report_sha256"] = sha256_obj(report)
    return report


def verify_tool_run(run_dir):
    root=Path(run_dir).resolve()
    root.relative_to(artifacts_path("research_tools").resolve())
    receipt=read_json(root/"receipt.json")
    plan=read_json(root/"plan.json")
    request=read_json(root/"request.json")
    result=read_json(root/"result.json")
    if (receipt.get("schema")!="research-tool-receipt-v1"
            or receipt["plan_sha256"]!=plan["plan_sha256"]
            or plan["plan_sha256"]!=sha256_obj({k:v for k,v in plan.items() if k!="plan_sha256"})
            or plan["payload_sha256"]!=sha256_obj(request["payload"])
            or receipt["request_sha256"]!=sha256_file(root/"request.json")
            or receipt["result_sha256"]!=sha256_file(root/"result.json")):
        raise ValueError("research tool receipt/input/result identity changed")
    checked=[]
    if plan.get("isolated_rendering"):
        child=Path(result["run_dir"]).resolve()
        child.relative_to(artifacts_path("research_tools").resolve())
        if (receipt.get("worker_identity")!=plan["worker_identity"]
                or receipt.get("worker_receipt_sha256")!=sha256_file(child/"receipt.json")
                or receipt.get("worker_status_sha256")!=sha256_file(child/"status.json")
                or receipt.get("worker_cleanup_sha256")!=sha256_file(child/"cleanup.json")
                or read_json(child/"cleanup.json").get("status")!="completed"):
            raise ValueError("rendering worker receipt or cleanup identity changed")
        worker=read_json(child/"receipt.json")
        if (worker["image"]!=plan["worker_identity"]["image"]
                or worker["sources"]!=plan["worker_identity"]["sources"]
                or worker["request_sha256"]!=sha256_file(child/"request.json")
                or worker["result_sha256"]!=sha256_file(child/"result.json")
                or worker["policy_sha256"]!=sha256_file(child/"container-policy.json")):
            raise ValueError("rendering worker content identity changed")
        for item in result.get("artifacts",[]):
            file=Path(item["path"]).resolve()
            file.relative_to(child)
            if not file.is_file() or file.is_symlink() or file.stat().st_size!=item["size"] or sha256_file(file)!=item["sha256"]:
                raise ValueError("rendered artifact identity changed")
            checked.append({"path":str(file),"sha256":item["sha256"]})
    return {"status":"verified","result_ok":bool(result.get("ok")),"artifacts":checked,"model_calls":0,"network_requests":0}
