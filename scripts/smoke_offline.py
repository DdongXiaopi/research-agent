"""Run a no-network, synthetic TaskRunner smoke check in an isolated RSI_HOME."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runtime.provider import FakeProvider, ModelOutput  # noqa: E402
from src.runtime.runner import ArtifactState, TaskRunner  # noqa: E402
from src.runtime.skills import Registry, Skill  # noqa: E402
from src.runtime.trace import Trace, new_run_id  # noqa: E402
from src.util import artifacts_path, artifacts_root, repo_root  # noqa: E402


class _SmokeProvider(FakeProvider):
    """Return one fixed answer so the smoke check never constructs a live provider."""

    def complete(self, **_kwargs):
        return ModelOutput(answer_index=0, answer_text="The result is supported.",
                           mode="offline_smoke")


def main() -> int:
    rsi_home = os.environ.get("RSI_HOME")
    if not rsi_home or artifacts_root().resolve() == (repo_root() / "artifacts").resolve():
        raise SystemExit("Set RSI_HOME to an isolated directory before running this smoke check.")

    run_id = new_run_id("offline-smoke")
    trace_path = artifacts_path("runs", run_id, "trace.jsonl")
    trace = Trace(run_id, out_path=trace_path)
    skill_name = "offline-smoke-evidence"
    state = ArtifactState(
        "offline-smoke-release",
        "Answer only from the evidence supplied with the task.",
        Registry("offline-smoke-release", [Skill(
            name=skill_name,
            group="G-smoke",
            description="Check whether the supplied evidence states the result.",
            triggers=["smoke evidence"],
            body="Procedure\n1. Compare each option against the supplied evidence.",
            license="SELF-AUTHORED",
        )]),
    )
    config = {
        "runtime": {
            "provider": "fake",
            "input_mode": "task_input",
            "max_model_calls": 1,
            "max_tool_calls": 1,
            "task_input_max_chars": 2000,
        },
        "skills": {"routing_mode": "lexical"},
        "tools": {"active": ["citation.verify"]},
    }
    task = {
        "task_id": "offline-smoke-synthetic-1",
        "dataset": "synthetic-smoke",
        "split": "local",
        "question": "Does the smoke evidence support the result?",
        "options": ["The result is supported.", "The result is contradicted."],
        "correct_index": 0,
        "evidence": "The smoke evidence states that the result is supported.",
    }

    outcome = TaskRunner(config, state, provider=_SmokeProvider(), trace=trace).run(task)
    expected_events = {
        "task.start", "skill.candidates", "skill.loaded",
        "model.call", "task.answer.selected", "task.end",
    }
    observed_events = {str(event.get("event_type")) for event in outcome.events}
    if (outcome.status != "ok" or not outcome.correct
            or outcome.answer_index != 0 or outcome.skill_recalled != skill_name
            or not expected_events.issubset(observed_events)
            or not trace_path.is_file()):
        raise SystemExit("Offline smoke check failed; inspect its isolated trace.")

    print(json.dumps({
        "status": "ok",
        "mode": "offline_fake_provider",
        "task_id": task["task_id"],
        "skill_recalled": outcome.skill_recalled,
        "trace_event_count": len(outcome.events),
        "trace_path": str(trace_path),
        "data_splits_read": [],
        "provider_requests": 0,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
