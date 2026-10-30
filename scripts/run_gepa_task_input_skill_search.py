"""Run bounded Prompt-independent GEPA search on one task-input Skill body.

This command creates only a draft candidate from visible `dev_optimize`
examples. It never reads selection_gate/final_holdout and never publishes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.evolution.gepa_skill_search import search_skill_body_candidate  # noqa: E402
from src.governance.release import load_state  # noqa: E402
from src.util import artifacts_path  # noqa: E402
from scripts.run_gepa_prompt_search import build_search_config  # noqa: E402


def main() -> int:
    raise SystemExit(
        "retired_unplanned_entrypoint: this legacy GEPA Skill-search CLI has no "
        "caller-challenged execution plan or request ledger; use a controlled, "
        "source-bound runner after its exact plan and request cap are approved")
    if os.name == "nt" and not sys.flags.utf8_mode:
        return subprocess.run(
            [sys.executable, "-X", "utf8", *sys.argv],
            env=os.environ.copy(), check=False).returncode
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--release", default="task-input-baseline-v1")
    parser.add_argument("--datasets", nargs="+", default=["bioasq_unitor_task_input"])
    parser.add_argument("--skill", default="litqa-literature-synthesis")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--task-provider", choices=["pi", "http"], default="pi")
    parser.add_argument("--metric-calls", type=int, default=90)
    parser.add_argument("--proposals", type=int, default=3)
    parser.add_argument("--minibatch", type=int, default=5)
    parser.add_argument("--acceptance", choices=["strict_improvement",
                                                   "improvement_or_equal",
                                                   "any_objective_improved"],
                        default="improvement_or_equal")
    parser.add_argument("--reflection-model", default=None)
    parser.add_argument("--single-objective", action="store_true",
                        help="use accuracy only instead of the default three objectives")
    parser.add_argument("--no-feedback-signals", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("RSI_HOME"):
        raise RuntimeError("GEPA Skill search requires an explicit isolated RSI_HOME")
    if args.limit < 3:
        raise ValueError("GEPA visible task limit must be at least 3")
    cfg = build_search_config(
        load_config(args.config), task_provider=args.task_provider,
        task_offset=args.offset, metric_calls=args.metric_calls,
        proposals=args.proposals, minibatch=args.minibatch,
        acceptance=args.acceptance, reflection_model=args.reflection_model,
        feedback_signals=not args.no_feedback_signals,
        multiobjective=not args.single_objective)
    gepa = cfg.setdefault("evolution", {}).setdefault("gepa", {})
    gepa["target_artifact"] = "skill_body"
    gepa["target_skill"] = args.skill
    state = load_state(args.release)
    before = {p.resolve() for p in artifacts_path("search", "gepa_skill").glob("*/report.json")}
    candidate = None
    error = None
    try:
        candidate = search_skill_body_candidate(
            cfg, state, args.datasets, args.limit, args.seed,
            failure_ids=[], skill_name=args.skill)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    reports = [p for p in artifacts_path("search", "gepa_skill").glob("*/report.json")
               if p.resolve() not in before]
    report_path = max(reports, key=lambda p: p.stat().st_mtime) if reports else None
    print(json.dumps({
        "candidate_id": candidate.candidate_id if candidate else None,
        "candidate_status": candidate.status if candidate else "no_change",
        "report": str(report_path.resolve()) if report_path else None,
        "search_error": error,
        "selection_gate_read": False,
        "release_published": False,
    }, ensure_ascii=False))
    return 2 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
