"""Consented feedback produces bounded, independently reviewable candidates."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import load_config
from src.evolution import feedback_candidate, session_feedback
from src.eval import open_benchmark
from src.governance import release
from src.runtime import research_assistant as research
from src.runtime.skills import load_body
from src.util import read_json


@pytest.mark.parametrize("kind", ["system_prompt", "skill_description", "skill_body"])
def test_feedback_proposes_each_existing_artifact_with_consent_and_gate(tmp_path, kind):
    release.init_baseline()
    state = release.load_state()
    cfg = load_config("configs/experiment.yaml")
    value = {"task_id": "feedback-task", "task_type": "open_research", "query": "What was observed?",
             "project_id": "feedback-candidates", "session_id": kind,
             "evidence": [{"source_id": "paper", "text": "Two controls were used."}]}
    request = research.request_from_dict(value)
    answer_plan = research.build_plan(request, cfg, state)
    answer = research.execute(request, cfg, answer_plan["plan_sha256"], state=state)
    session_feedback.record_feedback(Path(answer["run_dir"]), category="procedure", reviewer="researcher",
                                     correction="Check the study population.", use_for_evolution=True)
    dataset = session_feedback.export_development("feedback-candidates", tmp_path / "feedback.json")
    target = "system_prompt" if kind == "system_prompt" else "protocol-troubleshooting"
    skill = state.registry.get(target)
    current = state.system_prompt if kind == "system_prompt" else skill.description if kind == "skill_description" else load_body(skill)
    new = current + (" Check the study population." if kind != "skill_body" else "\n4. Check the study population.\n")
    if kind == "skill_description":
        new = " ".join(new.split())
    fixture = SimpleNamespace(chat_measured=lambda *args, **kwargs:
                              SimpleNamespace(content=json.dumps({"content": new}), cost={}))
    plan = feedback_candidate.build_plan(dataset, cfg, kind=kind, artifact_id=target, state=state)
    result = feedback_candidate.execute(dataset, cfg, plan["plan_sha256"], kind=kind, artifact_id=target,
                                        state=state, provider=fixture)
    candidate = read_json(Path(result["proposal_dir"]) / "candidate.json")
    assert candidate["artifact_type"] == kind
    assert candidate["extra"]["feedback_ids"] == plan["selected_feedback_ids"]
    assert result["status"] == "proposed_pending_independent_evaluation", [r for r in result["static_gate"]["results"] if not r["passed"]]
    candidate_state = open_benchmark._candidate_state(state, cfg, candidate)
    if kind == "system_prompt":
        assert candidate_state.system_prompt == new
    elif kind == "skill_description":
        assert candidate_state.registry.get(target).description == new
    else:
        assert load_body(candidate_state.registry.get(target)) == new.strip()
    assert not result["publication_authorized"]
    dataset["examples"][0]["expected_behavior"] = "Changed without consent"
    with pytest.raises(ValueError, match="development set is invalid"):
        feedback_candidate.build_plan(dataset, cfg, kind=kind, artifact_id=target, state=state)
