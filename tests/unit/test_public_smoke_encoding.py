"""Subprocess output must match the UTF-8 decoding used by public launchers."""
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("module_name", ["smoke_open_promotion", "smoke_public_task_input"])
def test_public_launcher_requests_utf8_from_python_child(monkeypatch, module_name):
    from importlib import import_module
    from scripts import build_public_snapshot
    module = import_module(f"scripts.{module_name}")
    monkeypatch.setattr(sys, "argv", [module_name])
    monkeypatch.setattr(build_public_snapshot, "source_files", lambda: [])
    observed = {}

    def run(*args, **kwargs):
        observed.update(kwargs["env"])
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.main() == 0
    assert observed["PYTHONIOENCODING"] == "utf-8"
