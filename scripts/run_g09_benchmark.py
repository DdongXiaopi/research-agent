#!/usr/bin/env python3
"""Reproducible, isolated CLI for the required G09 benchmark stages.

The CLI is intentionally task-first: it can freeze an already authored
dataset, run only development/selection answers, and then perform the
separate AI annotation and assessment stages.  It never exposes the
``final_holdout`` split.  Every mutating action requires a dedicated
``RSI_HOME`` and writes a small hash-bound pointer into ``--work-dir``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _sha(value) -> str:
    from src.util import sha256_obj
    return sha256_obj(value)


def _guard_runtime(path: Path) -> Path:
    """Set an isolated artifact root before importing modules that cache paths."""
    path = path.resolve()
    if path == (ROOT / "artifacts").resolve() or path == ROOT:
        raise ValueError("G09 CLI requires an isolated RSI_HOME")
    path.mkdir(parents=True, exist_ok=True)
    # The normal CLI is a fresh process and binds all module-level artifact
    # paths to this root.  If an embedding caller has already imported the
    # governance modules, changing RSI_HOME here would split one release
    # object across two roots (registry from the cached root, body paths from
    # the new environment).  Keep the already-bound root in that case; the
    # caller can start a fresh process for a hard runtime switch.
    if "src.governance.release" not in sys.modules:
        os.environ["RSI_HOME"] = str(path)
    return path


def _ensure_release_snapshot(runtime: Path) -> None:
    """Make baseline release state available without mutating the repository."""
    target = runtime / "releases"
    if target.exists():
        if not (target / "ACTIVE").is_file():
            raise ValueError("isolated RSI_HOME releases is incomplete")
        return
    source = ROOT / "artifacts" / "releases"
    if not (source / "ACTIVE").is_file():
        raise ValueError("repository baseline release is missing")
    shutil.copytree(source, target)


def _cfg(path: str, model_id: str | None = None) -> dict:
    from src.config import load_config
    cfg = load_config(path)
    if model_id:
        cfg.setdefault("models", {}).setdefault("task", {})["model_id"] = model_id
        cfg["models"]["task"]["provider"] = "http"
    return cfg


def _plan_hash(plan: dict) -> str:
    actual = plan.get("plan_sha256")
    if not isinstance(actual, str) or actual != _sha({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError("plan hash is invalid")
    return actual


def _require_plan(path: Path, expected: str | None) -> dict:
    plan = _json(path)
    digest = _plan_hash(plan)
    if expected and expected != digest:
        raise ValueError("expected plan hash differs")
    return plan


def _reject_final(split: str) -> None:
    if split == "final_holdout":
        raise ValueError("G09 CLI does not expose final_holdout")
    if split not in {"dev_optimize", "selection_gate"}:
        raise ValueError("G09 CLI requires dev_optimize or selection_gate")


def _evaluation_plan(runtime: Path, evaluation_id: str) -> dict:
    """Load a private evaluation plan and enforce the no-final CLI boundary."""
    private = (runtime / "evaluation_only" / "open_benchmarks" / evaluation_id).resolve()
    private.relative_to(runtime.resolve())
    plan_path = private / "plan.json"
    if not plan_path.is_file():
        raise ValueError("evaluation plan is missing")
    plan = _require_plan(plan_path, None)
    _reject_final(plan.get("split"))
    return plan


def _freeze(args, runtime: Path) -> dict:
    from src.eval import open_benchmark
    dataset = _json(args.dataset.resolve())
    root = open_benchmark.freeze(dataset)
    manifest = _json(root / "manifest.json")
    pointer = {"schema": "g09-freeze-pointer-v1", "root": str(root),
               "dataset_sha256": root.name, "manifest_sha256": manifest["manifest_sha256"]}
    _write(args.work_dir.resolve() / "freeze-report.json", pointer)
    return pointer


def _answer_plan(args, runtime: Path) -> dict:
    from src.eval import open_benchmark
    _reject_final(args.split)
    cfg = _cfg(args.config, args.model_id)
    root = args.dataset_root.resolve()
    candidate = _json(args.candidate.resolve()) if args.candidate else None
    plan = open_benchmark.build_plan(
        root,
        cfg,
        split=args.split,
        candidate_release=args.candidate_release,
        candidate=candidate,
    )
    _write(args.work_dir.resolve() / "answer-plan.json", plan)
    return {"plan_path": str(args.work_dir.resolve() / "answer-plan.json"),
            "plan_sha256": plan["plan_sha256"], "model_calls": 0,
            "split": args.split, "final_holdout_read": False}


def _answer_execute(args, runtime: Path) -> dict:
    from src.eval import open_benchmark
    plan = _require_plan(args.plan.resolve(), args.expected_plan_sha256)
    _reject_final(plan.get("split"))
    cfg = _cfg(args.config, args.model_id)
    result = open_benchmark.run(args.dataset_root.resolve(), cfg, plan)
    _write(args.work_dir.resolve() / "answer-report.json", result)
    return result


def _annotation_plan(args, runtime: Path) -> dict:
    from src.eval import ai_annotation, open_benchmark
    _evaluation_plan(runtime, args.evaluation_id)
    packet = open_benchmark.annotation_packet(args.evaluation_id)
    kwargs = {}
    if args.max_calls is not None:
        kwargs["max_calls"] = args.max_calls
    if args.max_cost_usd is not None:
        kwargs["max_cost_usd"] = args.max_cost_usd
    plan = ai_annotation.annotation_plan(packet, _cfg(args.config, args.model_id), **kwargs)
    _write(args.work_dir.resolve() / "annotation-plan.json", plan)
    return {"plan_path": str(args.work_dir.resolve() / "annotation-plan.json"),
            "plan_sha256": plan["plan_sha256"], "model_calls": 0,
            "evaluation_id": args.evaluation_id, "final_holdout_read": False}


def _annotation_execute(args, runtime: Path) -> dict:
    from src.eval import ai_annotation, ai_jobs, open_benchmark
    _evaluation_plan(runtime, args.evaluation_id)
    plan = _require_plan(args.plan.resolve(), args.expected_plan_sha256)
    packet = open_benchmark.annotation_packet(args.evaluation_id)
    if plan.get("kind") != "open_annotations":
        raise ValueError("annotation plan kind mismatch")
    result = ai_annotation.annotate(packet, _cfg(args.config, args.model_id), plan)
    verified = ai_jobs.verify_completed(result["run_dir"], kind="open_annotations")
    # ``verify_completed`` returns the resolved private root as a Path for
    # callers that need filesystem access; the CLI report is JSON and must
    # carry the same identity as a string.
    verified = {**verified, "root": str(verified["root"])}
    annotations = result["output"]
    _write(args.work_dir.resolve() / "annotations.json", annotations)
    report = {"schema": "g09-annotation-report-v1", "evaluation_id": args.evaluation_id,
              "plan_sha256": plan["plan_sha256"], "run_dir": result["run_dir"],
              "annotations_sha256": _sha(annotations), "verified": verified,
              "final_holdout_read": False}
    _write(args.work_dir.resolve() / "annotation-report.json", report)
    return report


def _assess(args, runtime: Path) -> dict:
    from src.eval import open_benchmark
    _evaluation_plan(runtime, args.evaluation_id)
    annotations = _json(args.annotations.resolve())
    report = open_benchmark.assess(args.evaluation_id, annotations)
    report["final_holdout_read"] = False
    _write(args.work_dir.resolve() / "assessment-report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    # JSON reports contain source text and reviewer rationale; Windows' GBK
    # console must not turn a completed job into a false CLI failure.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "answer-plan", "answer-execute",
                                             "annotation-plan", "annotation-execute", "assess"))
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--evaluation-id")
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--config", default="configs/experiment.http.pubmedqa-pilot.yaml")
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--split", default="selection_gate")
    parser.add_argument("--candidate-release",
                        help="published release id for the candidate arm")
    parser.add_argument("--candidate", type=Path,
                        help="JSON file for an unpublished static-gated candidate arm")
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--max-cost-usd", type=float, default=None,
                        help="frozen AI annotation spend ceiling; required for non-zero priced models when the default is too low")
    args = parser.parse_args(argv)
    runtime = _guard_runtime(args.runtime_dir)
    if args.action in {"answer-plan", "answer-execute", "annotation-plan", "annotation-execute", "assess"}:
        _ensure_release_snapshot(runtime)
    if args.action == "freeze":
        if not args.dataset:
            parser.error("freeze requires --dataset")
        result = _freeze(args, runtime)
    elif args.action == "answer-plan":
        if not args.dataset_root:
            parser.error("answer-plan requires --dataset-root")
        result = _answer_plan(args, runtime)
    elif args.action == "answer-execute":
        if not args.dataset_root or not args.plan or not args.expected_plan_sha256:
            parser.error("answer-execute requires --dataset-root, --plan and --expected-plan-sha256")
        result = _answer_execute(args, runtime)
    elif args.action == "annotation-plan":
        if not args.evaluation_id:
            parser.error("annotation-plan requires --evaluation-id")
        result = _annotation_plan(args, runtime)
    elif args.action == "annotation-execute":
        if not args.evaluation_id or not args.plan or not args.expected_plan_sha256:
            parser.error("annotation-execute requires --evaluation-id, --plan and --expected-plan-sha256")
        result = _annotation_execute(args, runtime)
    else:
        if not args.evaluation_id or not args.annotations:
            parser.error("assess requires --evaluation-id and --annotations")
        result = _assess(args, runtime)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
