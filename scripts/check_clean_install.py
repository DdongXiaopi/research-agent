"""Reproduce runtime installation and offline smoke in a fresh workspace/venv."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimizers", action="store_true")
    parser.add_argument("--pi", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.relative_to((ROOT / ".workbuddy").resolve())
    if output.exists():
        raise ValueError("clean-install destination must not exist")
    output.mkdir(parents=True)
    checkout = output / "checkout"
    checkout.mkdir()
    for name in ("src", "configs", "skills", "patches"):
        shutil.copytree(ROOT / name, checkout / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (checkout / "scripts").mkdir()
    shutil.copytree(ROOT / "tools/docker-worker", checkout / "tools/docker-worker",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("smoke_offline.py", "bootstrap_project.py", "deploy_docker_worker.py"):
        shutil.copy2(ROOT / "scripts" / name, checkout / "scripts" / name)
    for name in ("requirements.txt", "requirements-mipro.txt", "requirements-optimizers.constraints.txt", "UPSTREAM_LOCK.yml"):
        shutil.copy2(ROOT / name, checkout / name)
    if args.pi:
        shutil.copytree(ROOT / "tools/pi-bridge", checkout / "tools/pi-bridge",
                        ignore=shutil.ignore_patterns("node_modules", "__pycache__"))
    # Only OS/tool discovery environment enters the fresh process, not .env,
    # model credentials, Python paths, existing venv or historical RSI state.
    keys = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "USERPROFILE",
            "APPDATA", "LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PATHEXT"}
    env = {key: value for key, value in os.environ.items() if key.upper() in keys}
    env.update({"PYTHONUTF8": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_CONFIG_FILE": os.devnull, "PIP_INDEX_URL": "https://pypi.org/simple",
                "RSI_HOME": str(output / "artifacts")})
    report = {"schema": "clean-install-receipt-v1", "status": "started",
              "steps": [], "model_requests": 0, "historical_artifacts_copied": False,
              "credentials_copied": False, "data_assets_required": False}

    def save():
        temporary = output / "receipt.tmp"
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(output / "receipt.json")

    def step(name, command, timeout=1200):
        started = time.monotonic()
        with (output / (name + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=checkout, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        report["steps"].append({"name": name, "returncode": result.returncode,
                                "seconds": round(time.monotonic() - started, 2)})
        save()
        print(name + ": " + str(result.returncode), flush=True)
        if result.returncode:
            raise RuntimeError("clean installation step failed: " + name)

    save()
    try:
        step("create-venv", [sys.executable, "-m", "venv", str(output / "venv")])
        python = output / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        step("install-runtime", [str(python), "-m", "pip", "install", "-r", "requirements.txt"])
        step("offline-smoke", [str(python), "scripts/smoke_offline.py"])
        renderer_smoke = """from src.runtime.docker_worker import source_identity
from src.runtime.execution import ExecutionGateway, ExecutionError
from unittest.mock import patch
assert len(source_identity()) == 3
with patch("src.runtime.docker_worker._cli") as cli:
    try:
        ExecutionGateway({"execution":{"enabled":True}}).run("latex_compile",{"source":"x"})
    except ExecutionError:
        pass
    else:
        raise AssertionError("renderer enabled without deployment")
    cli.assert_not_called()
print("renderer sources present; disabled renderer made zero Docker calls")
"""
        step("renderer-default-denial", [str(python), "-c", renderer_smoke])
        if args.optimizers:
            step("restore-optimizers", [str(python), "scripts/bootstrap_project.py", "--restore-vendor", "--install-optimizers"])
            step("optimizer-import", [str(python), "-c", "from pathlib import Path; import dspy, gepa, optuna; assert Path(dspy.__file__).resolve().is_relative_to(Path('vendor/dspy-main/dspy-main').resolve()); assert Path(gepa.__file__).resolve().is_relative_to(Path('vendor/gepa-main/gepa-main').resolve()); assert optuna.__version__ == '5.0.0'; print(dspy.__file__); print(gepa.__file__); print(optuna.__version__)"])
            step("dependency-check", [str(python), "-m", "pip", "check"])
            step("installed-dependencies", [str(python), "-m", "pip", "freeze", "--all"])
        if args.pi:
            npm = shutil.which("npm")
            if not npm:
                raise RuntimeError("npm is unavailable")
            step("install-pi", [npm, "--prefix", "tools/pi-bridge", "ci"])
            step("pi-contract", [npm, "--prefix", "tools/pi-bridge", "test"])
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error_type=type(exc).__name__)
        save()
        raise
    save()
    print(str(output / "receipt.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
