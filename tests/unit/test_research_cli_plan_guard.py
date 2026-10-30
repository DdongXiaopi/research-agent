"""Caller plan review is mandatory before research dispatch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_answer_execute_requires_matching_caller_hash(tmp_path):
    env = {**os.environ, "RSI_HOME": str(tmp_path / "run-home"), "PYTHONIOENCODING": "cp1252"}

    def run(*args):
        return subprocess.run([sys.executable, str(ROOT / "scripts" / "research_assistant.py"),
                               "answer", "--request", str(ROOT / "configs/research/request.example.json"),
                               *args], cwd=ROOT, env=env, capture_output=True, text=True)

    init = subprocess.run([sys.executable, str(ROOT / "scripts" / "rsi.py"), "init"],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    assert init.returncode == 0, init.stderr
    planned = run()
    assert planned.returncode == 0, planned.stderr
    plan_hash = json.loads(planned.stdout)["plan_sha256"]
    missing = run("--execute")
    wrong = run("--execute", "--expected-plan-sha256", "0" * 64)
    assert missing.returncode != 0 and "requires an explicit frozen plan hash" in missing.stderr
    assert wrong.returncode != 0 and "caller plan differs" in wrong.stderr
    assert not (tmp_path / "run-home/research").exists()
    executed = run("--execute", "--expected-plan-sha256", plan_hash)
    assert executed.returncode == 0, executed.stderr
    assert json.loads(executed.stdout)["response"]["status"] == "ok"


def test_rsi_json_output_survives_limited_console_encoding():
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    code = "from scripts.rsi import _print; _print('status', {'detail': chr(27979)+chr(35797)+chr(0x1f642)})"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                            capture_output=True, text=True, encoding="cp1252")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.split("=== status ===", 1)[1])["detail"] == "测试🙂"


def test_discover_rejects_missing_hash_before_network_dispatch(tmp_path, monkeypatch):
    from scripts import research_assistant as cli
    from src.retrieval import literature

    monkeypatch.setenv("RSI_HOME", str(tmp_path))
    with patch.object(literature, "build_plan", return_value={"plan_sha256": "a" * 64}), \
         patch.object(literature, "execute") as dispatch:
        with pytest.raises(ValueError, match="explicit frozen plan hash"):
            cli.main(["discover", "--query", "synthetic", "--execute"])
        dispatch.assert_not_called()


def test_feedback_skill_rejects_missing_hash_before_model_dispatch(tmp_path, monkeypatch):
    from scripts import research_assistant as cli
    from src.evolution import feedback_skill

    monkeypatch.setenv("RSI_HOME", str(tmp_path))
    with patch.object(cli, "read_json", return_value={"schema": "synthetic"}), \
         patch.object(cli, "load_config", return_value={}), \
         patch.object(feedback_skill, "build_feedback_skill_plan",
                      return_value={"plan_sha256": "b" * 64}), \
         patch.object(feedback_skill, "execute_feedback_skill") as dispatch:
        with pytest.raises(ValueError, match="explicit frozen plan hash"):
            cli.main(["propose-skill", "--development-set", "fixture.json",
                      "--config", "fixture.yaml", "--execute"])
        dispatch.assert_not_called()
