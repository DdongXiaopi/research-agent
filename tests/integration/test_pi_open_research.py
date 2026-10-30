"""Exercise open answers through the actual Pi harness with a faux model."""
import json
from pathlib import Path

import pytest

from src.runtime import research_assistant as research
from src.runtime import research_workflow as workflow
from src.runtime.runner import ArtifactState
from src.runtime.skills import Registry
from src.runtime.research_tools import audit_open_research_trace, verify_citations
from src.util import read_json, sha256_obj

pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "tools/pi-bridge/node_modules").is_dir(),
    reason="Pi bridge dependencies are optional; install with npm ci --prefix tools/pi-bridge")


def test_pi_open_answer_uses_real_harness_and_checks_quotes():
    state = ArtifactState("pi-open-fixture", "Answer from evidence.", Registry("pi-open-fixture", []))
    request = research.request_from_dict({"task_id": "pi-open", "task_type": "open_research",
        "query": "What was observed?", "project_id": "pi-open-tests", "session_id": "one",
        "evidence": [{"source_id": "paper", "text": "Two controls were used.",
                      "url": "https://example.org/article", "locator": "abstract"}]})
    answer = {"status": "ok", "claims": [{"text": "Two controls were used.",
              "citations": [{"source_id": "paper", "quote": "Two controls were used."}]}],
              "limitations": []}
    cfg = {"runtime": {"provider": "pi", "input_mode": "task_input", "protocol_epoch": "public-task-input-v1",
                       "pi_fake_response": json.dumps(answer)}, "models": {"task": {"model_id": "fixture"}}}
    plan = research.build_plan(request, cfg, state)
    result = research.execute(request, cfg, plan["plan_sha256"], state=state)
    assert result["response"]["status"] == "ok"
    assert result["response"]["execution_summary"]["mode"] == "pi_open_research"
    raw = read_json(Path(result["run_dir"]) / "raw-model-output.json")
    assert Path(raw["pi_trace"]["session_path"]).is_file()
    assert raw["pi_trace"]["provider_http_requests"] == 0
    assert audit_open_research_trace(result["run_dir"])["provider_calls"] == 1
    assert verify_citations(result["run_dir"])["status"] == "identity_verified"
    session = Path(raw["pi_trace"]["session_path"])
    session.write_bytes(session.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Pi session verification failed"):
        audit_open_research_trace(result["run_dir"])
    answer["claims"][0]["citations"][0]["quote"] = "fabricated"
    cfg["runtime"]["pi_fake_response"] = json.dumps(answer)
    plan = research.build_plan(request, cfg, state)
    with pytest.raises(ValueError, match="citation does not match"):
        research.execute(request, cfg, plan["plan_sha256"], state=state)


def test_six_stage_workflow_calls_pi_and_saves_result():
    state = ArtifactState("pi-open-fixture", "Answer from evidence.", Registry("pi-open-fixture", []))
    value = {"task_id": "six-stages", "task_type": "open_research", "query": "What was observed?",
             "project_id": "pi-open-tests", "session_id": "six",
             "evidence": [{"source_id": "paper", "text": "Two controls were used."}]}
    answer = {"status": "ok", "claims": [{"text": "Two controls were used.",
              "citations": [{"source_id": "paper", "quote": "Two controls were used."}]}], "limitations": []}
    cfg = {"runtime": {"provider": "pi", "input_mode": "task_input", "protocol_epoch": "public-task-input-v1",
                       "pi_fake_response": json.dumps(answer)}, "models": {"task": {"model_id": "fixture"}}}
    plan = workflow.build_plan(value, cfg, state=state)
    result = workflow.execute(value, cfg, plan["plan_sha256"], state=state)
    assert [row["stage"] for row in result["stages"]] == list(workflow.STAGES)
    assert all(row["status"] == "completed" for row in result["stages"])
    assert result["citation_verification"]["status"] == "identity_verified"
    assert result["response"]["execution_summary"]["mode"] == "pi_open_research"
    assert read_json(Path(result["workflow_dir"]) / "result.json") == result
    with pytest.raises(ValueError, match="frozen plan"):
        workflow.execute({**value, "query": "Different question"}, cfg, plan["plan_sha256"], state=state)


def test_empty_discovery_abstains_without_a_model_request():
    state = ArtifactState("pi-open-fixture", "Answer from evidence.", Registry("pi-open-fixture", []))
    value = {"task_id": "no-sources", "task_type": "open_research", "query": "What was observed?",
             "project_id": "pi-open-tests", "session_id": "empty", "evidence": []}
    cfg = {"runtime": {"provider": "pi", "input_mode": "task_input", "protocol_epoch": "public-task-input-v1",
                       "pi_fake_response": "should not be called"}, "models": {"task": {"model_id": "fixture"}}}
    plan = workflow.build_plan(value, cfg, state=state)
    result = workflow.execute(value, cfg, plan["plan_sha256"], state=state)
    assert result["response"]["status"] == "abstain"
    assert result["trace_audit"]["provider_calls"] == 0
    assert result["citation_verification"]["citation_count"] == 0
    assert len(result["stages"]) == 6


def test_pi_open_benchmark_uses_ephemeral_sessions():
    from src.config import load_config
    from src.eval import open_benchmark as benchmark
    from src.governance import release
    release.init_baseline()
    cfg = load_config("configs/experiment.public-task-input.yaml")
    answer = {"status": "ok", "claims": [{"text": "Two controls were used.",
              "citations": [{"source_id": "paper", "quote": "Two controls were used."}]}], "limitations": []}
    cfg["runtime"].update(provider="pi", pi_fake_response=json.dumps(answer))
    cfg["models"]["task"].update(provider="pi", model_id="fixture")
    partitions = {}
    for split in benchmark.SPLITS:
        request = research.request_from_dict({"task_id": "case-" + split, "task_type": "open_research",
            "query": "What was used?", "project_id": "benchmark-pi", "session_id": split,
            "evidence": [{"source_id": "paper", "text": "Two controls were used. " + split}]})
        rubric = {"schema": "open-research-task-rubric-v1", "author": "fixture-author",
            "authorship": "engineering_fixture", "request_sha256": sha256_obj(request.to_dict()),
            "items": [{"item_id": "control", "criterion": "Identify controls.",
                       "evidence_quotes": [{"source_id": "paper", "quote": "Two controls were used."}]}]}
        partitions[split] = [{"request": request.to_dict(), "group_id": "group-" + split, "rubric": rubric}]
    dataset = benchmark.freeze({"schema": "open-research-dataset-v1", "partitions": partitions,
        "provenance": {"source": "synthetic", "revision": "1", "license": "SELF-AUTHORED", "reference_origin": "engineering_fixture"}})
    plan = benchmark.build_plan(dataset, cfg)
    assert plan["logical_request_cap"] == 1
    result = benchmark.run(dataset, cfg, plan)
    packet = benchmark.annotation_packet(result["evaluation_id"])
    assert packet["cases"][0]["response"]["status"] == "ok"
    case = read_json(benchmark._evaluation(result["evaluation_id"]) / "cases.json")[0]
    raw = read_json(Path(case["run_dir"]) / "raw-model-output.json")
    assert raw["pi_trace"]["ephemeral"] is True and not raw["pi_trace"]["session_path"]
    assert audit_open_research_trace(case["run_dir"])["provider_calls"] == 1
