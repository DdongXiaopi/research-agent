"""One bounded research task from question to verified citations."""
from __future__ import annotations

from uuid import uuid4

from ..config import config_fingerprint
from ..eval.run_manifest import source_tree_fingerprint
from ..governance import release
from ..retrieval import literature
from ..util import artifacts_path, now_iso, sha256_obj, sha256_text, write_json
from . import research_assistant
from .research_tools import audit_open_research_trace, verify_citations
from .skills import select_skill

STAGES = ("parse", "route_skill", "retrieve", "organize_evidence", "answer", "verify")


def build_plan(value, cfg, *, providers=None, limit=5, max_requests=8, retries=1, timeout=15, state=None):
    request = research_assistant.request_from_dict(value)
    state = state or release.load_state()
    providers = list(providers or [])
    if providers and request.evidence:
        raise ValueError("choose supplied evidence or literature discovery, not both")
    retrieval_plan = (literature.build_plan("discover", query=request.query, providers=providers,
        limit=limit, max_requests=max_requests, retries=retries, timeout=timeout) if providers else None)
    # Validate answer budgets and provider before any network requests.
    limits = research_assistant._limits(request)
    effective = research_assistant.effective_config(cfg, limits)
    selected = select_skill(request.query, state.registry,
                            min_score=int((cfg.get("skills") or {}).get("recall_min_score", 2)))
    body = {"schema": "research-six-stage-plan-v1", "request_sha256": sha256_obj(request.to_dict()),
        "config_sha256": sha256_obj(cfg), "protocol_sha256": config_fingerprint(effective),
        "source_tree_sha256": source_tree_fingerprint()["sha256"], "release_id": state.release_id,
        "system_prompt_sha256": sha256_text(state.system_prompt),
        "registry_fingerprint": state.registry.fingerprint(), "selected_skill": selected["name"] if selected else None,
        "retrieval_plan": retrieval_plan, "stages": list(STAGES),
        "model_logical_request_cap": 0 if (cfg.get("runtime") or {}).get("provider", "fake") == "fake" else 1,
        "model_physical_request_cap": limits["max_physical_requests"], "publication_authorized": False}
    return {**body, "plan_sha256": sha256_obj(body)}


def execute(value, cfg, expected_plan_sha256, *, providers=None, limit=5, max_requests=8,
            retries=1, timeout=15, state=None, session=None, provider=None, on_stage=None):
    state = state or release.load_state()
    plan = build_plan(value, cfg, providers=providers, limit=limit, max_requests=max_requests,
                      retries=retries, timeout=timeout, state=state)
    if expected_plan_sha256 != plan["plan_sha256"]:
        raise ValueError("research workflow differs from its frozen plan")
    root = artifacts_path("research_workflows", "workflow-" + uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "plan.json", plan)
    write_json(root / "request.json", value)
    stages = []
    current = None

    def begin(name):
        nonlocal current
        current = name
        write_json(root / "status.json", {"status": "running", "current_stage": name, "stages_completed": len(stages)})
        if on_stage:
            on_stage({"stage": name, "status": "running", "stages_completed": len(stages), "workflow_dir": str(root)})

    def complete(name, details):
        nonlocal current
        current = name
        stages.append({"stage": name, "status": "completed", "completed_at": now_iso(), **details})
        write_json(root / "stages.json", stages)
        write_json(root / "status.json", {"status": "running", "current_stage": name, "stages_completed": len(stages)})
        if on_stage:
            on_stage({"stage": name, "status": "completed", "stages_completed": len(stages), "workflow_dir": str(root)})

    try:
        begin("parse")
        request = research_assistant.request_from_dict(value)
        complete("parse", {"task_id": request.task_id, "query_sha256": sha256_obj(request.query)})
        begin("route_skill")
        complete("route_skill", {"selected_skill": plan["selected_skill"], "registry_fingerprint": plan["registry_fingerprint"]})
        begin("retrieve")
        snapshot = literature.execute(plan["retrieval_plan"], session=session) if plan["retrieval_plan"] else None
        complete("retrieve", {"mode": "literature" if snapshot else "supplied",
            "snapshot_sha256": snapshot["snapshot_sha256"] if snapshot else None,
            "physical_requests": snapshot["physical_requests"] if snapshot else 0,
            "source_status": snapshot["status"] if snapshot else "supplied"})
        begin("organize_evidence")
        if snapshot:
            request.evidence = literature.evidence_from_snapshot(snapshot)
        request = research_assistant.request_from_dict(request.to_dict())
        complete("organize_evidence", {"source_count": len(request.evidence),
                                      "source_ids": [row["source_id"] for row in request.evidence]})
        begin("answer")
        answer_plan = research_assistant.build_plan(request, cfg, state, retrieval_snapshot=snapshot)
        if answer_plan["selected_skill"] != plan["selected_skill"]:
            raise ValueError("Skill route changed during workflow")
        answer = research_assistant.execute(request, cfg, answer_plan["plan_sha256"], state=state,
                                            retrieval_snapshot=snapshot, provider=provider)
        complete("answer", {"run_dir": answer["run_dir"], "response_status": answer["response"]["status"],
                            "answer_plan_sha256": answer_plan["plan_sha256"]})
        begin("verify")
        citations = verify_citations(answer["run_dir"])
        trace = audit_open_research_trace(answer["run_dir"])
        if citations["status"] != "identity_verified" or trace["status"] not in {"ok", "abstain"}:
            raise ValueError("answer or citation verification failed")
        complete("verify", {"citation_status": citations["status"], "citation_count": citations["citation_count"],
                            "trace_report_sha256": trace["report_sha256"]})
        result = {"schema": "research-six-stage-result-v1", "workflow_dir": str(root),
                  "status": "completed", "plan_sha256": plan["plan_sha256"],
                  "run_dir": answer["run_dir"], "response": answer["response"],
                  "stages": stages, "citation_verification": citations, "trace_audit": trace}
        result["result_sha256"] = sha256_obj(result)
        write_json(root / "result.json", result)
        write_json(root / "status.json", {"status": "completed", "result_sha256": result["result_sha256"]})
        return result
    except BaseException as exc:
        write_json(root / "status.json", {"status": "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
            "current_stage": current, "stages_completed": len(stages), "error_type": type(exc).__name__})
        raise
