#!/usr/bin/env python3
"""Launch a frozen EI MIPRO plan with a credential supplied outside chat.

Uses only the standard library and never loads project dotenv settings. The
research runner remains responsible for current-source plan verification,
credential checks, one-shot execution and request accounting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
KEY_ENV = "SILICONFLOW_API_KEY"


class LaunchError(RuntimeError):
    pass


@dataclass(frozen=True)
class LaunchSpec:
    command: tuple[str, ...]
    root: Path
    sandbox: Path
    summary: dict


def prepare_launch(plan_path: Path, cohort_path: Path, *,
                   config_path: Path = Path("configs/experiment.http.pubmedqa-pilot.yaml"),
                   root: Path = ROOT) -> LaunchSpec:
    root = root.resolve()
    plan_path = (root / plan_path).resolve()
    cohort_path = (root / cohort_path).resolve()
    config_path = (root / config_path).resolve()
    if not plan_path.is_relative_to(root / "artifacts/sandboxes"):
        raise LaunchError("Choose a frozen preflight inside artifacts/sandboxes.")
    if plan_path.parent.name != "evidence_inference_mipro" or plan_path.parent.parent.name != "probe":
        raise LaunchError("The preflight must belong to an EI MIPRO sandbox.")
    sandbox = plan_path.parents[2]
    for value in (cohort_path, config_path):
        if not value.is_relative_to(root) or not value.is_file():
            raise LaunchError("The cohort and configuration must be existing project files.")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise LaunchError("Cannot read the frozen preflight.") from None
    if not isinstance(plan, dict):
        raise LaunchError("The preflight must be a JSON object.")
    sha = plan.get("plan_sha256", "")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise LaunchError("The preflight has no valid plan SHA-256.")
    # Preflight v4 canonical exclusions, also enforced by the authoritative
    # runner. This is a file-integrity check, not current-source validation.
    canonical = {k: v for k, v in plan.items()
                 if k not in {"plan_sha256", "api_key_explicit_at_launch", "status"}}
    actual = hashlib.sha256(json.dumps(canonical, sort_keys=True,
                                      ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    if actual != sha or plan_path.name != f"preflight-{sha[:16]}.json":
        raise LaunchError("The preflight content or filename does not match its hash.")
    if (plan.get("plan_schema_version") != 4
            or plan.get("protocol") != "evidence-inference-v2-abstract-mipro-task-input-v6"
            or plan.get("dataset") != "evidence_inference_v2_abstract"
            or plan.get("request_budget_version") != "bounded-physical-attempt-budget-v1"):
        raise LaunchError("This launcher supports only bounded EI MIPRO v4 preflights.")
    if any(plan.get(k) is not False for k in (
            "selection_gate_read", "official_validation_read", "official_test_read",
            "project_selection_gate_or_confirmation_read")):
        raise LaunchError("The plan must exclude protected splits and confirmation.")
    if any(type(plan.get(k)) is not int or plan[k] <= 0 for k in (
            "maximum_configured_model_calls", "maximum_physical_http_requests")):
        raise LaunchError("The preflight must declare positive logical and physical ceilings.")
    numeric_flags = {
        "mipro_num_candidates": "--num-candidates",
        "mipro_num_trials": "--num-trials",
        "minimum_generated_instructions_evaluated": "--minimum-generated-instructions-evaluated",
        "mipro_max_labeled_demos": "--max-labeled-demos",
        "mipro_task_evaluation_budget": "--mipro-task-evaluation-budget",
        "proposer_call_budget": "--proposer-call-budget",
        "grounding_call_budget": "--grounding-call-budget",
        "physical_request_budget": "--physical-request-budget",
        "mipro_execution_seed": "--seed",
    }
    command = [sys.executable, str(root / "scripts/run_evidence_inference_mipro_search.py"),
               "--dataset-adapter", "evidence_inference", "--config", str(config_path),
               "--cohort", str(cohort_path)]
    for key, flag in numeric_flags.items():
        value = plan.get(key)
        minimum = 0 if key in {"minimum_generated_instructions_evaluated", "mipro_execution_seed"} else 1
        if type(value) is not int or value < minimum:
            raise LaunchError("The preflight contains invalid numeric execution settings.")
        command.extend((flag, str(value)))
    labels = plan.get("execution_labels")
    if (not isinstance(labels, list) or not labels
            or any(type(v) is not int for v in labels) or len(set(labels)) != len(labels)):
        raise LaunchError("The preflight must have unique integer execution labels.")
    task_model = plan.get("task_model")
    routing = plan.get("routing_mode")
    if task_model not in {"Qwen/Qwen3-32B", "Qwen/Qwen3-8B"} or routing not in {"model", "lexical"}:
        raise LaunchError("Unsupported task model or routing mode.")
    if plan.get("mipro_require_balanced_demo_bundle") is True:
        command.append("--require-balanced-demo-bundle")
    command.extend(("--execution-labels", *(str(v) for v in labels),
                    "--routing-mode", routing, "--task-model-id", task_model,
                    "--execute", "--expected-plan-sha256", sha))
    return LaunchSpec(tuple(command), root, sandbox, {
        "plan_sha256": sha, "scopes": plan.get("scopes"),
        "logical_request_ceiling": plan.get("maximum_configured_model_calls"),
        "physical_request_ceiling": plan.get("maximum_physical_http_requests"),
        "sandbox": str(sandbox), "credential_env": KEY_ENV,
        "current_source_verification": "performed by the runner before dispatch",
    })


def has_interactive_console() -> bool:
    if not sys.stdin.isatty():
        return False
    if os.name != "nt":
        return True
    # Windows isatty() also accepts NUL character devices. getpass uses
    # msvcrt console input and can hang on those handles even with stdin closed.
    try:
        import ctypes
        import msvcrt

        mode = ctypes.c_ulong()
        get_mode = ctypes.WinDLL("kernel32", use_last_error=True).GetConsoleMode
        get_mode.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        get_mode.restype = ctypes.c_int
        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
        return bool(get_mode(handle, ctypes.byref(mode)))
    except (OSError, ValueError, AttributeError):
        return False


def prompt_local_credential() -> str:
    if not has_interactive_console():
        raise LaunchError("Run in a local interactive terminal to enter the rotated credential.")
    with warnings.catch_warnings():
        # Refuse getpass's plaintext-input fallback if terminal echo control fails.
        warnings.simplefilter("error", getpass.GetPassWarning)
        try:
            return getpass.getpass("Rotated SILICONFLOW_API_KEY (hidden, local terminal): ")
        except (getpass.GetPassWarning, EOFError):
            raise LaunchError("Secure terminal input is unavailable; no child was started.") from None


def launch(spec: LaunchSpec, *, base_env=None, get_secret=None, run_child=None) -> int:
    environment = dict(os.environ if base_env is None else base_env)
    if not environment.get(KEY_ENV):
        credential = (get_secret or prompt_local_credential)()
        if not isinstance(credential, str) or not credential.strip():
            raise LaunchError("No credential supplied; no child was started.")
        environment[KEY_ENV] = credential
    environment["RSI_HOME"] = str(spec.sandbox)
    # No shell interpolation, credential CLI argument, file or parent-env write.
    result = (run_child or subprocess.run)(list(spec.command), cwd=str(spec.root), env=environment)
    return int(result.returncode)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/experiment.http.pubmedqa-pilot.yaml"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true", help="read metadata only; no prompt, process or writes")
    mode.add_argument("--execute", action="store_true", help="use a launch credential or prompt locally, then run")
    args = parser.parse_args(argv)
    try:
        spec = prepare_launch(args.plan, args.cohort, config_path=args.config)
        print(json.dumps(spec.summary, ensure_ascii=False))
        return 0 if args.inspect else launch(spec)
    except KeyboardInterrupt:
        print("Launch interrupted.", file=sys.stderr)
        return 130
    except LaunchError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception:
        # Never echo arbitrary exception messages that could contain input.
        print("Launcher failed; no credential values are included in this message.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
