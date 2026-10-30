"""RSI 命令行入口：一条命令跑完 M1 主线闭环。

    python scripts/rsi.py init                     按 experiment.yaml 建立不可变基线
    python scripts/rsi.py status                   当前生效 release / 指纹
    python scripts/rsi.py freeze                   M2：冻结三层 split（此后改 split 会被拒）
    python scripts/rsi.py baseline                 A0 基线评测（optimize + gate）
    python scripts/rsi.py round                    一轮进化：归因 → 候选 → 门禁 → 判定
    python scripts/rsi.py campaign --rounds 3      连续多轮进化（Spec 工程成功标准）
    python scripts/rsi.py multiseed                已停用（实时基线请用计划绑定的 baseline）
    python scripts/rsi.py ablate                   已停用（缺少当前协议的冻结计划）
    python scripts/rsi.py final-baseline          final_holdout 基线终评（只跑一次，随后封存）
    python scripts/rsi.py review --candidate-id X --decision approve|reject
    python scripts/rsi.py rollback --to baseline-v0
    python scripts/rsi.py verify                   泄漏检查 + split 冻结校验（--strict 含故障注入自检）
    python scripts/rsi.py e2e                      仅独立 RSI_HOME：门禁 / review / 回滚
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import config_fingerprint, load_config  # noqa: E402
from src.contracts.types import Candidate  # noqa: E402
from src.data import freeze as frz  # noqa: E402
from src.eval import ablation, leakage, multiseed  # noqa: E402
from src.eval.ablation import factorial_self_test  # noqa: E402
from src.runtime.channels import self_test as channels_self_test  # noqa: E402
from src.eval.final_eval import (  # noqa: E402
    ASSESSMENT_FILE,
    audit_log,
    baseline_exists,
    load_baseline,
    run_final_assessment,
    run_final_baseline,
)
from src.eval.harness import evaluate  # noqa: E402
from src.eval.stats import self_test as stats_self_test  # noqa: E402
from src.evolution.campaign import run_campaign  # noqa: E402
from src.evolution.round import run_round  # noqa: E402
from src.evolution.task_demo_search import (  # noqa: E402
    DEFAULT_MAX_DEMO_CANDIDATE_SETS,
)
from src.governance import release as rel  # noqa: E402
from src.governance import review as rv  # noqa: E402
from src.governance.budget import self_test as budget_self_test  # noqa: E402
from src.util import write_json  # noqa: E402
from src.eval.unblind_budget import self_test as unblind_budget_self_test  # noqa: E402
from src.eval.verdict import self_test as verdict_self_test  # noqa: E402
from src.retrieval.corpus import self_test as corpus_self_test  # noqa: E402
from src.runtime.execution import build_gateway  # noqa: E402
from src.util import artifacts_path, artifacts_root, repo_root, path, read_json, sha256_obj  # noqa: E402


def _print(title: str, obj: Any) -> None:
    print(f"\n=== {title} ===")
    # ASCII escapes preserve JSON values on legacy Windows console code pages.
    print(json.dumps(obj, ensure_ascii=True, indent=2, sort_keys=True))


def cmd_init(args: argparse.Namespace) -> int:
    task_input_version = int(getattr(args, "task_input_baseline_version", 0) or 0)
    if task_input_version:
        if (not os.environ.get("RSI_HOME") or
                artifacts_root().resolve() == (repo_root() / "artifacts").resolve()):
            raise ValueError("task_input 基线只允许在独立 RSI_HOME 中初始化")
        if args.git:
            raise ValueError("独立 task_input 基线不应提交当前仓库")
        registry_name = "registry.yaml" if task_input_version == 1 else "registry.v2.yaml"
        release_ids = {
            1: "task-input-baseline-v1",
            2: "task-input-baseline-v2",
            3: "task-input-baseline-v2-label-compatible",
            4: "task-input-baseline-v2-minimal-label-compatible",
            5: "task-input-evidence-inference-v1",
            6: "task-input-healthver-v1",
            7: "task-input-clinifact-v1",
            8: "task-input-clinifact-v2",
            9: "task-input-scifact-v1",
            10: "task-input-clinifact-v3",
        }
        prompt_files = {
            1: "system_prompt.v1.txt",
            2: "system_prompt.v1.txt",
            3: "system_prompt.pubmedqa-threeway-v3.txt",
            4: "system_prompt.pubmedqa-threeway-v4.txt",
            5: "system_prompt.evidence-inference-v1.txt",
            6: "system_prompt.healthver-v1.txt",
            7: "system_prompt.clinifact.v0.txt",
            8: "system_prompt.clinifact.v1.txt",
            9: "system_prompt.scifact.v0.txt",
            10: "system_prompt.clinifact.v2.txt",
        }
        release_id = release_ids[task_input_version]
        prompt_path = (path("configs", "prompts", prompt_files[task_input_version])
                       if task_input_version in {7, 8, 9, 10} else
                       path("configs", "task_input", prompt_files[task_input_version]))
        release_note = {
            3: "PubMedQA yes/no/maybe 标签兼容 task_input v3；增加maybe判据说明；无检索；Prompt与Skill快照",
            4: "PubMedQA yes/no/maybe 最小输出标签契约 task_input v4；不增加类别判据；无检索；Prompt与Skill快照",
            5: "Evidence Inference abstract task_input v1；RCT干预效果方向三分类契约；固定任务证据、无检索；独立协议快照",
            6: "HealthVer claim-abstract task_input v1；SUPPORT/CONTRADICT/NOT_ENOUGH_INFO标签契约；固定任务证据、无检索；探索专用独立协议快照",
            7: "CliniFact clinical-research task_input v1；EVIDENCE/INCONCLUSIVE/NEI三分类契约；固定PubMed摘要、无检索；探索专用独立协议快照",
            8: "CliniFact clinical-research task_input v2；显著支持/非显著结果/背景或无结果三分类契约；固定PubMed摘要、无检索；探索专用独立协议快照",
            9: "SciFact claim-verification task_input v1；SUPPORT/CONTRADICT/NOT_ENOUGH_INFO 标签契约；使用随题给定的引用摘要、无检索；探索专用独立协议快照",
            10: "CliniFact clinical-research task_input v3；保持v2三类语义，输出与运行时选项编号契约一致；固定PubMed摘要、无检索；探索专用独立协议快照",
        }.get(task_input_version,
              f"task_input v{task_input_version} 无检索协议基线；Prompt 与 Skill 正文已快照")
        rid = rel.init_baseline(
            prompt_path=prompt_path,
            registry_path=path("configs", "task_input", registry_name),
            release_id=release_id,
            note=release_note)
    else:
        cfg = load_config(getattr(args, "config", None))
        skills_cfg = cfg.get("skills") or {}
        registry_ref = Path(str(skills_cfg.get("registry") or
                                "skills/registry.yaml"))
        rid = rel.init_baseline(
            registry_path=path(*registry_ref.parts),
            release_id=str(skills_cfg.get("release_id") or "baseline-v0"))
    if args.git:
        _git_init_and_commit("chore(rsi): baseline release")
    _print("init", {"release_id": rid, "releases": rel.list_releases()})
    return 0


def _git_identity_args() -> List[str]:
    """未配置 git 身份时给出默认身份，避免 commit 静默失败。"""
    out = subprocess.run(["git", "config", "user.email"], cwd=str(path(".")),
                         capture_output=True, text=True, check=False)
    if out.stdout.strip():
        return []
    return ["-c", "user.name=rsi", "-c", "user.email=rsi@local"]


def _git_init_and_commit(message: str) -> Optional[str]:
    repo = path(".")
    ident = _git_identity_args()
    import tempfile
    created_commit = None
    try:
        if not (repo / ".git").exists():
            subprocess.run(["git", "init"], cwd=str(repo), check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        release_id = rel.current_release_id()
        targets = [rel.RELEASES_ROOT / release_id, rel.ACTIVE_FILE]
        if rel.ALIASES_FILE.exists():
            targets.append(rel.ALIASES_FILE)
        relative = [target.resolve().relative_to(repo.resolve()).as_posix() for target in targets]
        def git(*args, env=None):
            return subprocess.run(["git", *args], cwd=str(repo), env=env,
                capture_output=True, text=True, check=True)
        head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=str(repo),
                              capture_output=True, text=True, check=False)
        before = head.stdout.strip() if head.returncode == 0 else None
        staged = git("diff", "--cached", "--name-only").stdout.splitlines()
        if any(filename == target or filename.startswith(target + "/")
               for filename in staged for target in relative):
            raise RuntimeError("release paths contain user-staged changes")
        # Prepare just the explicit release snapshot using a separate index;
        # never absorb unrelated data/config/skill edits or the user's staging.
        with tempfile.TemporaryDirectory(prefix="rsi-git-index-") as directory:
            env = {**os.environ, "GIT_INDEX_FILE": str(Path(directory) / "index")}
            git("read-tree", "HEAD" if before else "--empty", env=env)
            git("add", "--", *relative, env=env)
            diff = git("diff", "--cached", "--name-only", env=env).stdout.strip()
            if not diff:
                return None
            git(*ident, "commit", "-m", message, env=env)
            after = git("rev-parse", "HEAD").stdout.strip()
            created_commit = after
            if after == before:
                raise RuntimeError("release commit did not create a new revision")
            git("reset", "-q", "HEAD", "--", *relative)
            return after
    except Exception as exc:
        if created_commit:
            raise RuntimeError(f"release commit {created_commit} exists; index reconciliation failed: "
                               f"{type(exc).__name__}") from exc
        raise RuntimeError(f"release Git receipt not created: {type(exc).__name__}") from exc


def cmd_status(args: argparse.Namespace) -> int:
    state = rel.load_state()
    info = rel.release_info()
    _print("status", {
        "active_release": state.release_id,
        "system_prompt_sha256": sha256_obj(state.system_prompt),
        "skills_fingerprint": state.registry.fingerprint(),
        "n_skills": len(state.registry.skills),
        "parent": info.get("parent"),
        "created_at": info.get("created_at"),
    })
    return 0


def _apply_noise(cfg: Dict[str, Any], noise: Optional[float]) -> Dict[str, Any]:
    """命令行覆盖题级噪声（默认 0 = 完全确定性；多 seed 统计验证时才开）。"""
    if noise is not None:
        cfg = dict(cfg)
        runtime = dict(cfg.get("runtime", {}))
        runtime["fake_noise"] = float(noise)
        cfg["runtime"] = runtime
    return cfg


def _seeds_of(cfg: Dict[str, Any], args: Any) -> List[int]:
    if getattr(args, "seeds", None):
        return [int(s) for s in args.seeds]
    return [int(s) for s in cfg.get("evaluation", {}).get("seeds", [cfg.get("evaluation", {}).get("seed", 0)])]


def cmd_freeze(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    ds = args.datasets or list(cfg.get("evaluation", {}).get("datasets", []))
    rep = (frz.upgrade_content_freeze(operator=args.operator)
           if getattr(args, "upgrade_content", False) else
           frz.freeze(ds, operator=args.operator, note=args.note or ""))
    brief = {
        "frozen_at": rep["frozen_at"],
        "freeze_sha256": rep["freeze_sha256"][:16],
        "datasets": {
            d: {s: {"n": v["n"], "ids_sha256": v["recomputed_ids_sha256"][:12]}
                for s, v in rep["datasets"][d]["splits"].items()}
            for d in rep["datasets"]
        },
        "policy": rep["policy"],
    }
    _print("freeze", brief)
    return 0


def cmd_multiseed(args: argparse.Namespace) -> int:
    if getattr(args, "validate_only", False) or getattr(args, "execute", False):
        return _controlled_experiment(args, "multiseed")
    raise SystemExit(
        "rsi multiseed execution is disabled: both baseline and challenger modes "
        "score dev_optimize/selection_gate without a current hash-bound execution "
        "plan and bounded request ledger. No config, release, task, or evaluator "
        "is loaded here. Use `rsi baseline --validate-only` to freeze a bounded "
        "baseline plan, then execute only its reviewed hash; analyze existing "
        "multiseed artifacts offline."
    )


def cmd_ablate(args: argparse.Namespace) -> int:
    if getattr(args, "validate_only", False) or getattr(args, "execute", False):
        return _controlled_experiment(args, "ablate")
    _print("ablate", {
        "ok": False,
        "status": "retired_unplanned_entrypoint",
        "error": (
            "rsi ablate is disabled until it has a native frozen execution plan "
            "that binds its source releases, conditions, splits, cohorts, request "
            "caps, and one-shot execution identity; use a reviewed plan-bound "
            "experiment entrypoint instead"
        ),
        "model_calls": 0,
        "evaluation_started": False,
    })
    return 2


def _controlled_experiment(args, kind):
    from src.eval.controlled_experiment import build_plan, execute
    cfg = load_config(args.config)
    if args.noise is not None:
        cfg["runtime"]["fake_noise"] = args.noise
    if getattr(args, "legacy_union", False):
        cfg["evolution"]["behavior_channels"]["enabled"] = False
    conditions = {}
    for entry in getattr(args, "condition", None) or []:
        name, separator, rid = entry.partition("=")
        if not separator or not rid or name in conditions:
            raise ValueError("conditions must be unique LABEL=RELEASE entries")
        conditions[name] = rid
    spec = {"base": args.base_release or rel.current_release_id()}
    if kind == "multiseed":
        spec["compare"] = args.compare
        if args.candidate:
            spec["candidate"] = read_json(Path(args.candidate))
    else:
        spec.update(conditions=conditions, prompt=args.prompt_release,
                    description=args.desc_release, body=args.body_release)
        if conditions and any((args.prompt_release, args.desc_release, args.body_release)):
            raise ValueError("select explicit conditions or factor releases, not both")
    plan = build_plan(cfg, kind=kind, spec=spec,
        datasets=args.datasets or cfg["evaluation"]["datasets"],
        splits=getattr(args, "splits", None) or ["dev_optimize", "selection_gate"],
        seeds=args.seeds or cfg["evaluation"].get("seeds", []), limit=args.limit,
        logical_request_cap=args.logical_request_cap, physical_request_cap=args.physical_request_cap)
    write_json(artifacts_path("experiment_plans", plan["plan_sha256"] + ".json"), plan)
    if args.execute:
        result = execute(cfg, plan, args.expected_plan_sha256)
    else:
        result = {"status": "validated_not_run", **plan}
    _print(kind, result)
    return 0


def _planned_final_evaluation(args: argparse.Namespace, purpose: str) -> int:
    """Run a final evaluator only after a current plan hash is caller-challenged."""
    import re
    import uuid

    from src.eval import final_eval
    from src.eval.final_execution_plan import (build_final_execution_plan,
                                               verify_final_execution_plan)
    from src.runtime.call_ledger import summarize_ledger
    from src.util import ensure_dir

    title = "final-baseline" if purpose == "final-baseline" else "final-eval"
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256 or ""):
        _print(title, {"ok": False,
                       "error": "--execute requires a 64-character --expected-plan-sha256"})
        return 2
    if args.validate_only and args.expected_plan_sha256:
        _print(title, {"ok": False,
                       "error": "--expected-plan-sha256 is only valid with --execute"})
        return 2
    if args.limit is not None and args.limit < 0:
        _print(title, {"ok": False, "error": "--limit cannot be negative"})
        return 2
    if not os.environ.get("RSI_HOME"):
        _print(title, {"ok": False,
                       "error": "set RSI_HOME to an isolated directory before planning final evaluation"})
        return 1

    plan_path = None
    guard_path = None
    execution_path = None
    ledger_path = None
    try:
        cfg = load_config(args.config)
        state = rel.load_state(args.release) if args.release else rel.load_state()
        evaluation = cfg.get("evaluation", {})
        datasets = list(args.datasets or evaluation.get("primary_final_datasets")
                        or evaluation.get("datasets", []))
        if purpose == "final-eval" and args.transfer:
            datasets.extend(ds for ds in (evaluation.get("transfer_datasets") or [])
                            if ds not in datasets)
        if purpose == "final-baseline":
            seeds = list(args.seeds or [args.seed])
            reason = args.reason or "baseline 封存前跑一次 final（Spec §6.1 铁律 3）"
        else:
            seeds = list(args.seeds)
            reason = args.reason or "M4 冻结终评"
        override = ({"actor": args.budget_override_actor or args.actor,
                     "reason": args.budget_override_reason}
                    if args.budget_override_reason else None)
        plan = build_final_execution_plan(
            cfg, purpose=purpose, release_id=state.release_id,
            datasets=datasets, seeds=seeds, limit=args.limit,
            force=bool(getattr(args, "force", False)), actor=args.actor,
            reason=reason, strict_protocol=not bool(
                getattr(args, "no_strict_protocol", False)),
            budget_override=override)
        plan_root = artifacts_path("final_holdout", "plans", purpose)
        ensure_dir(plan_root)
        plan_path = plan_root / ("plan-" + plan["plan_sha256"] + ".json")
        if plan_path.exists():
            if read_json(plan_path) != plan:
                raise RuntimeError("saved final evaluation plan differs from recomputed plan")
        else:
            write_json(plan_path, plan)
        if args.validate_only:
            _print(title, {
                "status": "validated_not_run", "plan": str(plan_path),
                "plan_sha256": plan["plan_sha256"], "datasets": datasets,
                "split": "final_holdout", "seeds": seeds,
                "logical_request_cap": plan["logical_request_cap"],
                "physical_request_cap": plan["physical_request_cap"],
                "model_calls": 0, "final_holdout_task_rows_read": False,
                "selection_gate_read": False,
                "explicit_user_approval_required": True,
            })
            return 0

        verify_final_execution_plan(plan, args.expected_plan_sha256)
        task_model = (cfg.get("models") or {}).get("task") or {}
        key_env = str(task_model.get("api_key_env") or "").strip()
        if plan["provider"] in {"http", "pi"} and (
                not key_env or not os.environ.get(key_env)):
            raise RuntimeError("final evaluation is missing the configured task-model credential")

        plan_run_root = artifacts_path("final_holdout", "runs", purpose,
                                       plan["plan_sha256"])
        ensure_dir(plan_run_root)
        run_id = uuid.uuid4().hex
        guard_path = plan_run_root / "execution-started.json"
        with guard_path.open("x", encoding="utf-8") as handle:
            json.dump({"plan_sha256": plan["plan_sha256"], "run_id": run_id,
                       "status": "started"}, handle, sort_keys=True)
            handle.write("\n")
        run_dir = plan_run_root / run_id
        ensure_dir(run_dir)
        ledger_path = run_dir / "call-ledger.jsonl"
        execution_path = run_dir / "execution.json"
        with execution_path.open("x", encoding="utf-8") as handle:
            json.dump({"plan_sha256": plan["plan_sha256"], "status": "started",
                       "ledger": str(ledger_path)}, handle, sort_keys=True)
            handle.write("\n")

        execution_env = {
            "RSI_ACTIVE_EXECUTION_PLAN_SHA256": plan["plan_sha256"],
            "RSI_CALL_LEDGER_PATH": str(ledger_path),
            "RSI_CALL_LEDGER_MAX_LOGICAL": str(plan["logical_request_cap"]),
            "RSI_CALL_LEDGER_MAX_PHYSICAL": str(plan["physical_request_cap"]),
            "RSI_CALL_LEDGER_RESERVE_PI_PHYSICAL": "1",
        }
        previous_env = {key: os.environ.get(key) for key in execution_env}
        os.environ.update(execution_env)
        try:
            if purpose == "final-baseline":
                rep = run_final_baseline(
                    cfg, state, datasets, seed=args.seed, seeds=seeds,
                    limit=args.limit, force=args.force, actor=args.actor,
                    reason=reason, budget_override=override)
                brief = {
                    "ok": True, "release_id": rep["release_id"], "seeds": rep["seeds"],
                    "datasets": {d: {"n_tasks": v["n_tasks"],
                                     "accuracy_by_seed": {
                                         seed: result["metrics"].get("accuracy")
                                         for seed, result in v["runs"].items()}}
                                 for d, v in rep["datasets"].items()},
                    "protocol": rep["protocol"], "policy": rep["policy"],
                    "audit_entries": len(audit_log()),
                }
            else:
                rep = run_final_assessment(
                    cfg, state, datasets, seeds=seeds, limit=args.limit,
                    actor=args.actor, reason=reason,
                    strict_protocol=not args.no_strict_protocol,
                    budget_override=override)
                brief = {
                    "ok": True, "baseline_release": rep["baseline_release_id"],
                    "final_release": rep["release_id"],
                    "protocol_ok": rep["protocol_check"]["ok"],
                    "datasets": {
                        dataset: {
                            "role": result["role"], "n_tasks": result["n_tasks"],
                            "primary_seed": result["primary_seed"],
                            "baseline_acc": result["per_seed"][
                                str(result["primary_seed"])]["baseline_accuracy"],
                            "final_acc": result["per_seed"][
                                str(result["primary_seed"])]["final_accuracy"],
                            "delta_pp": result["per_seed"][
                                str(result["primary_seed"])]["delta_pp"],
                            "ci95": result["per_seed"][
                                str(result["primary_seed"])]["ci95"],
                            "p_delta_gt_0": result["per_seed"][
                                str(result["primary_seed"])]["p_delta_gt_0"],
                            "mean_delta_pp": result["mean_delta_pp"],
                            "triple": result["per_seed"][
                                str(result["primary_seed"])]["triple"],
                            "regressed": result["per_seed"][
                                str(result["primary_seed"])]["negative_results"]["n_regressed"],
                            "improved": result["per_seed"][
                                str(result["primary_seed"])]["negative_results"]["n_improved"],
                            "cost_usd": result["per_seed"][
                                str(result["primary_seed"])]["cost_usd"],
                        } for dataset, result in rep["datasets"].items()
                    },
                    "report": str(ASSESSMENT_FILE),
                }
            summary = summarize_ledger(ledger_path)
            execution_path.write_text(json.dumps({
                "plan_sha256": plan["plan_sha256"], "status": "completed",
                "ledger": str(ledger_path), "ledger_summary": summary,
                "report_path": str(final_eval.BASELINE_FILE if purpose == "final-baseline"
                                    else ASSESSMENT_FILE),
            }, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            guard_path.write_text(json.dumps({
                "plan_sha256": plan["plan_sha256"], "run_id": run_id,
                "status": "completed", "execution": str(execution_path),
            }, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            brief.update({"plan_sha256": plan["plan_sha256"],
                          "request_ledger_summary": summary})
            _print(title, brief)
            return 0
        except Exception as exc:
            summary = summarize_ledger(ledger_path)
            execution_path.write_text(json.dumps({
                "plan_sha256": plan["plan_sha256"], "status": "failed",
                "error_type": type(exc).__name__, "ledger": str(ledger_path),
                "ledger_summary": summary,
            }, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            guard_path.write_text(json.dumps({
                "plan_sha256": plan["plan_sha256"], "run_id": run_id,
                "status": "failed", "execution": str(execution_path),
            }, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise
        finally:
            for key, previous in previous_env.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous
    except Exception as exc:
        _print(title, {"ok": False, "error": str(exc),
                       "plan": str(plan_path) if plan_path else None})
        return 1


def cmd_final_baseline(args: argparse.Namespace) -> int:
    return _planned_final_evaluation(args, "final-baseline")


def cmd_gate_verdict(args: argparse.Namespace) -> int:
    """决策 1（R1）：判定下沉到 selection_gate —— 不消耗封存揭盲预算。"""
    import json
    import re
    import uuid

    from src.eval.gate_plan import (build_gate_execution_plan,
                                    verify_gate_execution_plan)
    from src.eval.gate_verdict import run_gate_verdict
    from src.util import artifacts_path, ensure_dir, read_json, write_json

    cfg = load_config(args.config)
    if args.validate_only and args.expected_plan_sha256:
        _print("gate-verdict", {"ok": False,
                                "error": "--expected-plan-sha256 仅可用于 --execute"})
        return 2
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}", args.expected_plan_sha256 or ""):
        _print("gate-verdict", {"ok": False,
                                "error": "--execute 必须提供 64 位小写十六进制计划哈希"})
        return 2
    if args.limit is not None and args.limit < 0:
        _print("gate-verdict", {"ok": False, "error": "--limit 不能为负数"})
        return 2
    datasets = args.datasets or list(cfg.get("evaluation", {}).get("datasets", []))
    seeds = args.seeds or [int(cfg.get("evaluation", {}).get("seed", 0))]
    # `--limit 0` = 不限（跑满该 split 的全部题）。直接把 0 传给 evaluate 会变成"0 题"。
    limit = args.limit if (args.limit and args.limit > 0) else None
    if not __import__("os").environ.get("RSI_HOME"):
        _print("gate-verdict", {"ok": False,
                                "error": "请先把 RSI_HOME 指向隔离的本次运行目录"})
        return 1
    try:
        champion_id = args.champion
        if not champion_id:
            from src.governance.release import current_release_id
            champion_id = current_release_id()
        plan = build_gate_execution_plan(
            cfg, challenger_id=args.challenger, champion_id=champion_id,
            datasets=datasets, seeds=seeds, limit=limit,
            protocol_check=not args.no_protocol_check)
        plan_path = artifacts_path("gate_verdicts", "plans",
                                   "plan-" + plan["plan_sha256"] + ".json")
        ensure_dir(plan_path.parent)
        if plan_path.exists():
            if read_json(plan_path) != plan:
                raise RuntimeError("saved gate plan differs from recomputed plan")
        else:
            write_json(plan_path, plan)
        if args.validate_only:
            _print("gate-verdict", {
                "status": "validated_not_run", "plan": str(plan_path),
                "plan_sha256": plan["plan_sha256"], "datasets": datasets,
                "split": plan["split"], "seeds": seeds,
                "logical_request_cap": plan["logical_request_cap"],
                "physical_request_cap": plan["physical_request_cap"],
                "model_calls": 0, "selection_gate_task_rows_read": False,
                "final_holdout_read": False,
            })
            return 0
        verify_gate_execution_plan(plan, args.expected_plan_sha256)
        task_model = (cfg.get("models") or {}).get("task") or {}
        key_env = str(task_model.get("api_key_env") or "").strip()
        if plan["provider"] in {"http", "pi"} and (not key_env or not __import__("os").environ.get(key_env)):
            raise RuntimeError("gate execution is missing the configured task-model credential")
        run_id = uuid.uuid4().hex
        plan_run_root = artifacts_path("gate_verdicts", "runs", plan["plan_sha256"])
        ensure_dir(plan_run_root)
        execution_guard = plan_run_root / "execution-started.json"
        # One frozen plan can consume selection_gate only once. A crash leaves
        # the guard in place for audit instead of making an implicit replay easy.
        with execution_guard.open("x", encoding="utf-8") as handle:
            json.dump({"plan_sha256": plan["plan_sha256"], "run_id": run_id,
                       "status": "started"}, handle, sort_keys=True)
            handle.write("\n")
        run_dir = plan_run_root / run_id
        ensure_dir(run_dir)
        ledger_path = run_dir / "call-ledger.jsonl"
        import os
        os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(plan["plan_sha256"])
        os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path)
        os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(plan["logical_request_cap"])
        os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(plan["physical_request_cap"])
        os.environ["RSI_CALL_LEDGER_RESERVE_PI_PHYSICAL"] = "1"
        run_manifest = run_dir / "execution.json"
        # Exclusive create prevents accidental reuse of this execution identity.
        with run_manifest.open("x", encoding="utf-8") as handle:
            json.dump({"plan_sha256": plan["plan_sha256"], "status": "started",
                       "ledger": str(ledger_path)}, handle, sort_keys=True)
            handle.write("\n")
        rep = run_gate_verdict(cfg, challenger_id=args.challenger,
                               champion_id=champion_id, datasets=datasets,
                               split=args.split, seeds=args.seeds or None,
                               limit=limit,
                               check_protocol=not args.no_protocol_check,
                               execution_plan_sha256=plan["plan_sha256"])
        from src.runtime.call_ledger import summarize_ledger
        run_manifest.write_text(json.dumps({
            "plan_sha256": plan["plan_sha256"], "status": "completed",
            "ledger": str(ledger_path), "ledger_summary": summarize_ledger(ledger_path),
            "report_path": rep.get("report_path"),
        }, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        execution_guard.write_text(json.dumps({
            "plan_sha256": plan["plan_sha256"], "run_id": run_id,
            "status": "completed", "report_path": rep.get("report_path"),
        }, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        _print("gate-verdict", {"ok": False, "error": str(exc)})
        return 1

    brief_ds = {}
    for ds, blk in rep["per_dataset"].items():
        sd = str(blk["primary_seed"])
        row = blk["per_seed"][sd]
        brief_ds[ds] = {
            "role": blk["role"],
            "n_tasks": row.get("n_tasks"),
            "baseline": row["baseline"],
            "candidate": row["candidate"],
            "deltas_pp": row["deltas_pp"],
            "primary_axis": row["primary_axis"],
            "primary_delta_pp": row["primary_delta_pp"],
            "verdict": row["verdict"],
            "floor_effect": row["floor_effect"]["triggered"],
            "caveats": row["caveats"],
        }
    _print("gate-verdict", {
        "ok": True,
        "split": rep["split"],
        "champion": rep["champion_release"],
        "challenger": rep["challenger_release"],
        "datasets": brief_ds,
        "promotion": rep["promotion"],
        "budget_note": rep["budget_note"],
        "significance_note": rep["significance_note"],
        "report": rep.get("report_path"),
    })
    return 0


def cmd_alias(args: argparse.Namespace) -> int:
    """决策 4：champion / challenger 治理层（对齐 MLflow alias）。"""
    if args.set is None and not args.caveat and not args.resolve_caveat:
        data = rel.ensure_aliases(actor=args.actor) if args.adopt else rel.load_aliases()
        _print("alias", {"aliases": data, "active": rel.current_release_id()
                         if (artifacts_path("releases", "ACTIVE")).exists() else None})
        return 0
    if args.caveat:
        out = rel.note_champion(args.caveat, actor=args.actor)
        out["ok"] = True
        _print("alias", out)
        return 0
    if args.resolve_caveat:
        try:
            out = rel.resolve_caveat(args.resolve_caveat, actor=args.actor,
                                     reason=args.reason or "")
        except Exception as exc:
            _print("alias", {"ok": False, "error": str(exc)})
            return 1
        out["ok"] = True
        _print("alias", out)
        return 0
    try:
        out = rel.set_alias(args.set, args.release, actor=args.actor, reason=args.reason)
    except Exception as exc:
        _print("alias", {"ok": False, "error": str(exc)})
        return 1
    out["active_release"] = rel.current_release_id()
    _print("alias", {**out, "ok": True,
                     "note": ("challenger 只是登记，不生效；达标后走 `rsi.py promote`"
                              if args.set == "challenger" else "champion 已变更")})
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    """决策 4：challenger → champion（必须过发布门 + 人工 sign-off）。"""
    if not args.gate_report:
        _print("promote", {
            "ok": False,
            "error": ("晋升必须基于发布门结论：先跑 `rsi.py gate-verdict --challenger ...` "
                      "再用 --gate-report <该文件>。"),
        })
        return 1
    promotion = read_json(Path(args.gate_report))
    promotion = promotion.get("promotion", promotion)
    try:
        out = rel.promote_challenger(reviewer=args.reviewer, reason=args.reason,
                                     promotion=promotion)
    except Exception as exc:
        _print("promote", {"ok": False, "error": str(exc)})
        return 1
    out["ok"] = True
    out["active_release"] = rel.current_release_id()
    out["gate"] = "已用发布门结论"
    _print("promote", out)
    return 0


def cmd_reseal_plan(args: argparse.Namespace) -> int:
    """决策 3：待重封清单 —— 必须合并为一次重封，不单独花揭盲预算。"""
    from src.governance.reseal import plan

    p = plan()
    _print("reseal-plan", p)
    # 退出码只反映"命令本身是否跑通"；要不要重封是人的决策，不在这里卡 CI
    return 0


def cmd_final_budget(args: argparse.Namespace) -> int:
    """P8 揭盲预算账本：各封存集已用 / 限额 / 剩余 / 是否退役。

    与 `preregister` 的分工：后者报"揭盲了几次 + 哪套协议"（描述过去），
    这里报"还能不能再揭盲"（约束未来），超限时命令会直接拒绝访问。
    """
    from src.eval import final_eval
    from src.eval import unblind_budget as ub

    cfg = load_config(args.config)
    fp = final_eval.protocol_fingerprint(cfg)
    datasets = args.datasets or list(cfg.get("evaluation", {}).get("primary_final_datasets")
                                     or []) or ["suppqa", "scholarqa_cs"]
    ledger = ub.load_ledger()
    if ledger is None and args.adopt:
        ledger = ub.ensure_ledger(actor=args.actor, note=args.reason or "CLI 启用预算账本")

    states = {}
    for ds in datasets:
        st = ub.budget_state(ds, fp, ledger)
        states[ds] = st

    out: Dict[str, Any] = {
        "protocol_fingerprint": fp,
        "budget_file": str(ub.BUDGET_FILE),
        "ledger": ("adopted" if ledger else "not-adopted（加 --adopt 建立账本）"),
        "adopted_at": (ledger or {}).get("created_at"),
        "limits": (ledger or {}).get("limits") or ub.DEFAULT_LIMITS,
        "datasets": states,
        "overrides": (ledger or {}).get("overrides", []),
        "policy": ub.POLICY_NOTE,
    }
    retired = [d for d, s in states.items() if s["retired"]]
    out["retired_datasets"] = retired
    out["note"] = ("下列数据集揭盲额度已用尽，任何新的非同协议访问都会被拒绝："
                   + ", ".join(retired) if retired else "额度内，可正常访问")
    ok = not (args.strict and retired)
    out["ok"] = ok
    _print("final-budget", out)
    return 0 if ok else 1


def cmd_final_eval(args: argparse.Namespace) -> int:
    """M4 final assessment, gated by a frozen plan and caller hash challenge."""
    return _planned_final_evaluation(args, "final-eval")


def _cmd_campaign_impl(args: argparse.Namespace) -> int:
    cfg = _apply_noise(load_config(args.config), args.noise)
    seeds = _seeds_of(cfg, args) if args.seeds else None
    selected_datasets = args.datasets or cfg.get("evaluation", {}).get("datasets")
    plan_action = _prepare_live_round_execution(
        args, cfg, command="campaign", datasets=selected_datasets,
        seeds=seeds or [args.seed], rounds=args.rounds,
        max_candidates=int(cfg.get("evolution", {}).get("max_candidates_per_round", 5)))
    if plan_action == "validated":
        return 0
    try:
        rep = run_campaign(cfg, datasets=args.datasets, rounds=args.rounds, seed=args.seed,
                           seeds=seeds, limit=args.limit, approve=args.approve_simulated,
                           resume_campaign=getattr(args, "resume_campaign", None))
    except BaseException as exc:
        _complete_live_round_execution(plan_action, {},
            status="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_type=type(exc).__name__)
        raise
    if args.git:
        rep["git_commit"] = _git_init_and_commit(
            f"feat(rsi): campaign 发布 {rep['rounds_published']}/{rep['rounds_executed']} 轮"
            f" -> {rep['final_release']}")
    _complete_live_round_execution(plan_action, rep)
    brief = {
        "report_path": rep["report_path"],
        "start_release": rep["start_release"],
        "final_release": rep["final_release"],
        "rounds_executed": rep["rounds_executed"],
        "rounds_published": rep["rounds_published"],
        "approve_mode": rep["approve_mode"],
        "steps": [{k: s.get(k) for k in ("round", "from_release", "published", "artifact_type",
                                         "attributed_class", "delta_pp", "ok")}
                  for s in rep["steps"]],
    }
    _print("campaign", brief)
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    import re
    import uuid

    from src.eval.baseline_plan import (build_baseline_execution_plan,
                                        verify_baseline_execution_plan)
    from src.runtime.call_ledger import summarize_ledger
    from src.util import artifacts_path, ensure_dir, read_json, write_json

    if args.validate_only and args.expected_plan_sha256:
        _print("baseline", {"ok": False,
                            "error": "--expected-plan-sha256 仅可用于 --execute"})
        return 2
    if args.execute and not re.fullmatch(r"[0-9a-f]{64}",
                                         args.expected_plan_sha256 or ""):
        _print("baseline", {"ok": False,
                            "error": "--execute 必须提供 64 位小写十六进制计划哈希"})
        return 2
    if args.limit is not None and args.limit < 0:
        _print("baseline", {"ok": False, "error": "--limit 不能为负数"})
        return 2
    if not os.environ.get("RSI_HOME"):
        _print("baseline", {"ok": False,
                            "error": "请先把 RSI_HOME 指向隔离的本次运行目录"})
        return 1
    try:
        cfg = load_config(args.config)
        seeds = args.seeds or [args.seed]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("baseline seeds must be non-empty and unique")
        release_id = rel.current_release_id()
        plan = build_baseline_execution_plan(
            cfg, release_id=release_id, datasets=args.datasets,
            seeds=seeds, limit=args.limit)
        plan_root = artifacts_path("baselines", "plans")
        plan_path = plan_root / ("plan-" + plan["plan_sha256"] + ".json")
        ensure_dir(plan_root)
        if plan_path.exists():
            if read_json(plan_path) != plan:
                raise RuntimeError("saved baseline plan differs from recomputed plan")
        else:
            write_json(plan_path, plan)
        if args.validate_only:
            _print("baseline", {
                "status": "validated_not_run", "plan": str(plan_path),
                "plan_sha256": plan["plan_sha256"], "datasets": args.datasets,
                "splits": plan["split_policy"], "seeds": seeds,
                "logical_request_cap": plan["logical_request_cap"],
                "physical_request_cap": plan["physical_request_cap"],
                "model_calls": 0, "selection_gate_task_rows_read": False,
                "final_holdout_read": False,
            })
            return 0

        verify_baseline_execution_plan(plan, args.expected_plan_sha256)
        task_model = (cfg.get("models") or {}).get("task") or {}
        key_env = str(task_model.get("api_key_env") or "").strip()
        if plan["provider"] == "http" and (not key_env or not os.environ.get(key_env)):
            raise RuntimeError("baseline execution is missing the configured task-model credential")
        plan_run_root = artifacts_path("baselines", "runs", plan["plan_sha256"])
        ensure_dir(plan_run_root)
        run_id = uuid.uuid4().hex
        guard_path = plan_run_root / "execution-started.json"
        with guard_path.open("x", encoding="utf-8") as handle:
            json.dump({"plan_sha256": plan["plan_sha256"], "run_id": run_id,
                       "status": "started"}, handle, sort_keys=True)
            handle.write("\n")
        run_dir = plan_run_root / run_id
        ensure_dir(run_dir)
        ledger_path = run_dir / "call-ledger.jsonl"
        os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = str(plan["plan_sha256"])
        os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path)
        os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(plan["logical_request_cap"])
        os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(plan["physical_request_cap"])
        os.environ["RSI_CALL_LEDGER_RESERVE_PI_PHYSICAL"] = "1"

        state = rel.load_state(release_id)
        protocol_sha = config_fingerprint(cfg)
        results: Dict[str, Any] = {}
        for dataset in args.datasets:
            results[dataset] = {}
            for split in plan["split_policy"]:
                results[dataset][split] = {}
                for seed in seeds:
                    result = evaluate(
                        cfg, state, dataset, split, seed=seed, limit=args.limit,
                        run_tag="baseline", write_per_task=(
                            split == "dev_optimize"
                            or bool((cfg.get("evaluation") or {}).get("persist_selection_per_task"))
                        ),
                        cfg_fp=protocol_sha)
                    results[dataset][split][str(seed)] = {
                        "run_id": result.run_id, "n_tasks": result.n_tasks,
                        "metrics": result.metric_values,
                    }
        report_path = run_dir / "report.json"
        report = {
            "schema": "rsi-baseline-execution-report-v1",
            "execution_plan_sha256": plan["plan_sha256"],
            "release_id": release_id, "config_sha256": plan["configuration_sha256"],
            "protocol_fingerprint": plan["protocol_fingerprint"],
            "results": results, "request_ledger": str(ledger_path),
            "request_ledger_summary": summarize_ledger(ledger_path),
            "selection_gate_per_task_persistence": bool(
                (cfg.get("evaluation") or {}).get("persist_selection_per_task")),
            "final_holdout_read": False,
        }
        write_json(report_path, report)
        guard_path.write_text(json.dumps({
            "plan_sha256": plan["plan_sha256"], "run_id": run_id,
            "status": "completed", "report_path": str(report_path),
        }, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        _print("baseline", {"status": "completed", "report": str(report_path),
                            "plan_sha256": plan["plan_sha256"], "results": results})
        return 0
    except Exception as exc:
        _print("baseline", {"ok": False, "error": str(exc)})
        return 1


def _cmd_round_impl(args: argparse.Namespace) -> int:
    cfg = _apply_noise(load_config(args.config), args.noise)
    if getattr(args, "pi", False):
        if not getattr(args, "task_input", False):
            raise ValueError("--pi currently requires --task-input")
        cfg.setdefault("runtime", {})["provider"] = "pi"
    if getattr(args, "task_input", False):
        # A self-evolution cohort can consume task-supplied evidence directly.
        # It is a new protocol and cannot pair with a retrieval-based sealed baseline.
        cfg.setdefault("runtime", {})["input_mode"] = "task_input"
        cfg["runtime"].setdefault("task_input_max_chars", 30000)
        # A task-input cohort may be paired when its baseline and candidate
        # are both evaluated under the same task-input/model protocol.  Older
        # retrieval experiments set this false because they were being paired
        # against a different sealed retrieval baseline; preserve an explicit
        # config value for a fresh task-input protocol instead of silently
        # switching model/measurement semantics.
        cfg.setdefault("evaluation", {}).setdefault("sealed_pairing", False)
        cfg.setdefault("retrieval", {}).pop("query_control", None)
        cfg["retrieval"].pop("pubmed_source_queries", None)
        cfg.setdefault("tools", {})["active"] = [
            name for name in cfg.get("tools", {}).get("active", [])
            if name != "academic.search"]
    if getattr(args, "gepa", False):
        if (cfg.get("runtime") or {}).get("input_mode") != "task_input":
            raise ValueError("--gepa currently requires --task-input")
        # GEPA's per-task score is correctness (accuracy). This new task-input
        # protocol must use the same axis at the outer selection gate.
        cfg.setdefault("evaluation", {})["main_metric"] = "accuracy"
        cfg["evaluation"]["primary_axis"] = "accuracy"
        gp = cfg.setdefault("evolution", {}).setdefault("gepa", {})
        gp["enabled"] = True
        if getattr(args, "gepa_task_offset", None) is not None:
            if args.gepa_task_offset < 0:
                raise ValueError("--gepa-task-offset must be non-negative")
            gp["task_offset"] = args.gepa_task_offset
        if getattr(args, "gepa_target", "prompt") == "skill-body":
            if not getattr(args, "gepa_skill", None):
                raise ValueError("--gepa-target skill-body requires --gepa-skill")
            gp["target_artifact"] = "skill_body"
            gp["target_skill"] = args.gepa_skill
        if getattr(args, "gepa_reflection_model", None):
            cfg.setdefault("models", {}).setdefault("reflection", {})["model_id"] = (
                args.gepa_reflection_model)
        if args.gepa_metric_calls is not None:
            gp["max_metric_calls"] = args.gepa_metric_calls
        if getattr(args, "gepa_reflection_minibatch_size", None) is not None:
            gp["reflection_minibatch_size"] = args.gepa_reflection_minibatch_size
        if getattr(args, "gepa_acceptance_criterion", None) is not None:
            gp["acceptance_criterion"] = args.gepa_acceptance_criterion
        if getattr(args, "gepa_multiobjective", False):
            gp["multi_objective"] = True
            if getattr(args, "gepa_acceptance_criterion", None) is None:
                gp["acceptance_criterion"] = "any_objective_improved"
        if getattr(args, "gepa_use_merge", False):
            gp["use_merge"] = True
        if getattr(args, "gepa_merge_max_invocations", None) is not None:
            gp["merge_max_invocations"] = args.gepa_merge_max_invocations
        if getattr(args, "gepa_merge_val_overlap_floor", None) is not None:
            gp["merge_val_overlap_floor"] = args.gepa_merge_val_overlap_floor
        if getattr(args, "gepa_feedback_signals", False):
            gp["feedback_evidence_signals"] = True
    fewshot_search = bool(getattr(args, "fewshot_search", False))
    fewshot_replay_report = getattr(args, "fewshot_replay_report", None)
    configured_fewshot_budget = ((cfg.get("evolution", {}).get("fewshot") or {})
                                 .get("max_metric_calls"))
    if (str((cfg.get("runtime") or {}).get("provider") or "fake").lower() != "fake"
            and fewshot_search and args.fewshot_metric_calls is None
            and configured_fewshot_budget is None):
        raise RuntimeError(
            "live few-shot preflight requires an explicit metric-call budget; "
            "automatic sizing would materialize task rows before plan review")
    if fewshot_replay_report and not fewshot_search:
        raise ValueError("--fewshot-replay-report requires --fewshot-search")
    if fewshot_replay_report and getattr(args, "gepa", False):
        raise ValueError("--fewshot-replay-report currently supports demos-only rounds")
    if fewshot_search:
        if not getattr(args, "pi", False) or not getattr(args, "task_input", False):
            raise ValueError("--fewshot-search requires both --pi and --task-input")
        if getattr(args, "gepa", False) and getattr(args, "gepa_target", "prompt") != "prompt":
            raise ValueError("joint --gepa --fewshot-search requires --gepa-target prompt")
        cfg.setdefault("evaluation", {})["main_metric"] = "accuracy"
        cfg["evaluation"]["primary_axis"] = "accuracy"
        cfg.setdefault("evolution", {}).setdefault("fewshot", {})["enabled"] = True
        if getattr(args, "fewshot_candidate_sets", None) is not None:
            cfg["evolution"]["fewshot"]["max_candidate_sets"] = int(
                args.fewshot_candidate_sets)
        allowed = list(cfg.setdefault("evolution", {}).get("allowed_artifacts", []))
        needed_artifacts = (["task_program", "system_prompt"]
                            if getattr(args, "gepa", False) else ["task_demos"])
        for artifact in needed_artifacts:
            if artifact not in allowed:
                allowed.append(artifact)
        if fewshot_replay_report:
            # The replay report's fingerprint was produced by the explicit
            # task-demo-only search protocol. Recreate that exact allowed set.
            allowed = ["task_demos"]
        cfg["evolution"]["allowed_artifacts"] = allowed
        if getattr(args, "fewshot_metric_calls", None) is not None:
            cfg["evolution"]["fewshot"]["max_metric_calls"] = int(
                args.fewshot_metric_calls)
        elif "max_metric_calls" not in cfg["evolution"]["fewshot"]:
            from src.evolution.gepa_search import visible_examples
            selected_datasets = list(args.datasets or
                                     cfg.get("evaluation", {}).get("datasets", []))
            _, internal_validation = visible_examples(
                cfg, selected_datasets, args.limit)
            cfg["evolution"]["fewshot"]["max_metric_calls"] = (
                fewshot_metric_call_budget(
                    len(internal_validation),
                    len(args.seeds) if args.seeds else 1,
                    bool(getattr(args, "gepa", False)),
                    int(cfg["evolution"]["fewshot"].get(
                        "max_candidate_sets",
                        DEFAULT_MAX_DEMO_CANDIDATE_SETS))))
    if (getattr(args, "gepa_reflection_model", None) and
            not getattr(args, "gepa", False)):
        raise ValueError("--gepa-reflection-model requires --gepa")
    if (getattr(args, "gepa_task_offset", None) is not None and
            not getattr(args, "gepa", False)):
        raise ValueError("--gepa-task-offset requires --gepa")
    if getattr(args, "gepa_replay_report", None) and not getattr(args, "gepa", False):
        raise ValueError("--gepa-replay-report requires --gepa")
    if getattr(args, "gepa_replay_report", None) and getattr(args, "gepa_target", "prompt") != "prompt":
        raise ValueError("--gepa-replay-report currently supports Prompt only")
    selected_datasets = (args.datasets or
                         cfg.get("evaluation", {}).get("datasets"))
    plan_action = _prepare_live_round_execution(
        args, cfg, command="round", datasets=selected_datasets,
        seeds=list(args.seeds) if args.seeds else [args.seed], rounds=1,
        max_candidates=int(args.max_candidates or cfg.get("evolution", {}).get(
            "max_candidates_per_round", 5)))
    if plan_action == "validated":
        return 0
    demo_search_module = None
    original_demo_search = None
    if fewshot_replay_report:
        from src.evolution import task_demo_search as demo_search_module
        from src.evolution.task_demo_replay import replay_task_demo_candidate

        # Fail before the round spends any baseline model calls.
        replay_task_demo_candidate(cfg, rel.load_state(), selected_datasets,
                                   args.limit, args.seed, [], fewshot_replay_report)
        original_demo_search = demo_search_module.search_task_demo_candidate

        def replay_selected_demo(cfg_arg, state_arg, datasets_arg, limit_arg,
                                 seed_arg, failure_ids_arg, baseline_run_ids_arg,
                                 instruction_candidate=None):
            if instruction_candidate is not None:
                raise ValueError("demo-only replay cannot accept a GEPA instruction candidate")
            return replay_task_demo_candidate(
                cfg_arg, state_arg, datasets_arg, limit_arg, seed_arg,
                failure_ids_arg, fewshot_replay_report)

        demo_search_module.search_task_demo_candidate = replay_selected_demo
    try:
        report = run_round(cfg, datasets=selected_datasets, seed=args.seed,
                           seeds=list(args.seeds) if args.seeds else None,
                           limit=args.limit, max_candidates=args.max_candidates,
                           gepa_replay_report=getattr(args, "gepa_replay_report", None),
                           gepa_replay_index=getattr(args, "gepa_replay_index", 1),
                           fewshot_search=fewshot_search,
                           attribution_corrections=(
                               read_json(args.attribution_corrections)
                               if getattr(args, "attribution_corrections", None) else None))
    except BaseException as exc:
        _complete_live_round_execution(plan_action, {},
            status="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_type=type(exc).__name__)
        raise
    finally:
        if demo_search_module is not None and original_demo_search is not None:
            demo_search_module.search_task_demo_candidate = original_demo_search
    if fewshot_replay_report:
        report["fewshot_replay"] = {"search_report": fewshot_replay_report,
                                    "exploratory": True}
        write_json(report["report_path"], report)
    _complete_live_round_execution(plan_action, report)
    brief = {
        "round_id": report["round_id"],
        "release_id": report["release_id"],
        "n_failures": report["n_failures"],
        "failure_class_counts": report["failure_class_counts"],
        "candidates": [
            {k: c[k] for k in ("candidate_id", "attributed_class", "target_artifact",
                               "status", "n_failures", "delta_pp", "review_request")
             if k in c}
            for c in report["candidates"]
        ],
        "report_path": report["report_path"],
    }
    _print("round", brief)
    return 0


def _prepare_live_round_execution(args: argparse.Namespace, cfg: Dict[str, Any], *,
                                  command: str, datasets: List[str],
                                  seeds: List[int], rounds: int,
                                  max_candidates: int) -> Optional[Dict[str, Any] | str]:
    """Preflight or arm a one-shot bounded execution for a live round/campaign."""
    import re
    import uuid
    from src.evolution.round_plan import (build_round_execution_plan,
                                          verify_round_execution_plan)
    from src.util import ensure_dir, sha256_file, write_json

    runtime = cfg.get("runtime") or {}
    if str(runtime.get("provider") or "fake").lower() == "fake":
        if getattr(args, "validate_only", False) or getattr(args, "execute", False):
            raise ValueError("round execution plans are required only for live providers")
        return None
    validate_only = bool(getattr(args, "validate_only", False))
    execute = bool(getattr(args, "execute", False))
    if validate_only == execute:
        raise RuntimeError("live round/campaign requires exactly one of --validate-only or --execute")
    rsi_home = os.environ.get("RSI_HOME", "").strip()
    if not rsi_home:
        raise RuntimeError("live round/campaign requires an isolated RSI_HOME")
    home = Path(rsi_home).resolve()
    if home == repo_root().resolve():
        raise RuntimeError("live round/campaign RSI_HOME cannot be the repository root")
    if command == "campaign" and getattr(args, "resume_campaign", None):
        campaign_id = args.resume_campaign
        if not re.fullmatch(r"[a-f0-9]{32}", campaign_id):
            raise ValueError("invalid campaign checkpoint ID")
        checkpoint = read_json(artifacts_path("campaigns", f"checkpoint-{campaign_id}.json"))
        if (checkpoint.get("contract", {}).get("config_fingerprint") != config_fingerprint(cfg)
                or checkpoint.get("status") not in {"awaiting_review", "round_measured", "completed"}):
            raise ValueError("campaign checkpoint is stale or its round cannot be resumed")
        binding = checkpoint.get("live_budget") or {}
        plan_sha = binding.get("RSI_ACTIVE_EXECUTION_PLAN_SHA256", "")
        plan = read_json(artifacts_path("rounds", "plans", f"plan-{plan_sha}.json"))
        verify_round_execution_plan(plan, plan_sha)
        if plan.get("protocol_fingerprint") != config_fingerprint(cfg):
            raise ValueError("campaign source/protocol drift")
        from src.runtime.call_ledger import summarize_ledger
        ledger_path = Path(binding["RSI_CALL_LEDGER_PATH"])
        ledger_path.resolve().relative_to(home)
        accounting = summarize_ledger(ledger_path)
        if accounting.get("logical_incomplete") or accounting.get("physical_incomplete"):
            raise ValueError("campaign resume blocked: outstanding request outcome unknown")
        if validate_only:
            _print("campaign-resume-plan", {"campaign_id": campaign_id,
                "plan_sha256": plan_sha, "model_calls": 0, "request_accounting": accounting})
            return "validated"
        if getattr(args, "expected_plan_sha256", "") != plan_sha:
            raise ValueError("campaign resume requires its original frozen plan SHA-256")
        run_root = ledger_path.parent
        guard_path = run_root / "campaign-resume-active.json"
        run_id = uuid.uuid4().hex
        with guard_path.open("x", encoding="utf-8") as stream:
            json.dump({"status": "started", "run_id": run_id, "plan_sha256": plan_sha}, stream)
        names = (*binding, "RSI_EXECUTION_GUARD_PATH")
        before = {name: os.environ.get(name) for name in names}
        os.environ.update(binding)
        os.environ["RSI_EXECUTION_GUARD_PATH"] = str(guard_path)
        action = {"plan": plan, "run_id": run_id, "run_dir": run_root / ("resume-" + run_id),
                "guard_path": guard_path, "ledger_path": ledger_path,
                "environment_before": before, "resume_lock": guard_path}
        args._armed_execution_action = action
        return action
    logical_cap = getattr(args, "logical_request_cap", None)
    physical_cap = getattr(args, "physical_request_cap", None)
    if logical_cap is None or physical_cap is None:
        raise RuntimeError("live round/campaign requires explicit logical and physical request caps")
    options = {key: value for key, value in vars(args).items()
               if not key.startswith("_") and key not in {"func", "validate_only", "execute",
                              "expected_plan_sha256"}}
    bound_files = {}
    for name in ("gepa_replay_report", "fewshot_replay_report", "attribution_corrections"):
        value = getattr(args, name, None)
        if value:
            file_path = Path(value).resolve()
            if not file_path.is_file():
                raise ValueError(f"execution-plan input does not exist: {name}")
            bound_files[name] = {"path": str(file_path), "sha256": sha256_file(file_path)}
    options["bound_input_files"] = bound_files
    from src.governance.release import current_release_id
    plan = build_round_execution_plan(
        cfg, command=command, release_id=current_release_id(), datasets=datasets,
        seeds=seeds, limit=args.limit, rounds=rounds, max_candidates=max_candidates,
        logical_cap=int(logical_cap), physical_cap=int(physical_cap),
        execution_options=options)
    plan_root = artifacts_path("rounds", "plans")
    ensure_dir(plan_root)
    plan_path = plan_root / ("plan-" + plan["plan_sha256"] + ".json")
    if plan_path.exists():
        existing = read_json(plan_path)
        if existing != plan:
            raise RuntimeError("saved round execution plan differs from recomputed plan")
    else:
        write_json(plan_path, plan)
    if validate_only:
        _print("round-plan", {
            "status": "validated_not_run", "plan": str(plan_path),
            "plan_sha256": plan["plan_sha256"], "command": command,
            "datasets": datasets, "seeds": seeds, "rounds": rounds,
            "logical_request_cap": logical_cap,
            "physical_request_cap": physical_cap,
            "model_calls": 0, "selection_gate_rows_materialized": False,
            "final_holdout_read": False,
        })
        return "validated"
    expected = getattr(args, "expected_plan_sha256", "") or ""
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError("--execute requires --expected-plan-sha256")
    verify_round_execution_plan(plan, expected)
    task_cfg = (cfg.get("models") or {}).get("task") or {}
    key_env = str(task_cfg.get("api_key_env") or "").strip()
    if plan["provider"] == "http" and (not key_env or not os.environ.get(key_env)):
        raise RuntimeError("live round/campaign is missing the configured task-model credential")
    run_root = artifacts_path("rounds", "runs", plan["plan_sha256"])
    ensure_dir(run_root)
    run_id = uuid.uuid4().hex
    guard_path = run_root / "execution-started.json"
    with guard_path.open("x", encoding="utf-8") as handle:
        json.dump({"plan_sha256": plan["plan_sha256"], "run_id": run_id,
                   "status": "started"}, handle, sort_keys=True)
        handle.write("\n")
    run_dir = run_root / run_id
    ensure_dir(run_dir)
    ledger_path = run_dir / "call-ledger.jsonl"
    environment_before = {name: os.environ.get(name) for name in (
        "RSI_ACTIVE_EXECUTION_PLAN_SHA256", "RSI_EXECUTION_GUARD_PATH",
        "RSI_CALL_LEDGER_PATH", "RSI_CALL_LEDGER_MAX_LOGICAL",
        "RSI_CALL_LEDGER_MAX_PHYSICAL", "RSI_CALL_LEDGER_RESERVE_PI_PHYSICAL")}
    os.environ["RSI_ACTIVE_EXECUTION_PLAN_SHA256"] = plan["plan_sha256"]
    os.environ["RSI_EXECUTION_GUARD_PATH"] = str(guard_path)
    os.environ["RSI_CALL_LEDGER_PATH"] = str(ledger_path)
    os.environ["RSI_CALL_LEDGER_MAX_LOGICAL"] = str(logical_cap)
    os.environ["RSI_CALL_LEDGER_MAX_PHYSICAL"] = str(physical_cap)
    os.environ["RSI_CALL_LEDGER_RESERVE_PI_PHYSICAL"] = "1"
    action = {"plan": plan, "run_id": run_id, "run_dir": run_dir,
            "environment_before": environment_before,
            "guard_path": guard_path, "ledger_path": ledger_path}
    args._armed_execution_action = action
    return action


def _record_live_round_execution(action: Optional[Dict[str, Any] | str],
                                   report: Dict[str, Any], *, status: str = "completed",
                                   error_type: str | None = None) -> None:
    if not isinstance(action, dict):
        return
    from src.runtime.call_ledger import summarize_ledger
    from src.util import write_json
    path = action["run_dir"] / "execution-report.json"
    record = {
        "schema": "rsi-round-execution-report-v1",
        "execution_plan_sha256": action["plan"]["plan_sha256"],
        "status": status,
        "error_type": error_type,
        "report_path": report.get("report_path"),
        "request_ledger": str(action["ledger_path"]),
        "request_ledger_summary": summarize_ledger(action["ledger_path"]),
    }
    write_json(path, record)
    action["guard_path"].write_text(json.dumps({
        "plan_sha256": action["plan"]["plan_sha256"],
        "run_id": action["run_id"], "status": status,
        "report_path": str(path),
    }, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    for name, previous in action.get("environment_before", {}).items():
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _complete_live_round_execution(action: Optional[Dict[str, Any] | str],
                                   report: Dict[str, Any], *, status: str = "completed",
                                   error_type: str | None = None) -> None:
    try:
        _record_live_round_execution(action, report, status=status, error_type=error_type)
    finally:
        if isinstance(action, dict):
            action["terminal_attempted"] = True
            for name, previous in action.get("environment_before", {}).items():
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous
            if action.get("resume_lock"):
                Path(action["resume_lock"]).unlink(missing_ok=True)


def _run_execution_command(args: argparse.Namespace, implementation) -> int:
    # Includes replay setup and Git receipt failures outside the optimizer itself.
    args._armed_execution_action = None
    try:
        return implementation(args)
    except BaseException as exc:
        action = args._armed_execution_action
        if isinstance(action, dict) and not action.get("terminal_attempted"):
            _complete_live_round_execution(action, {},
                status="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                error_type=type(exc).__name__)
        raise
    finally:
        args._armed_execution_action = None


def cmd_round(args: argparse.Namespace) -> int:
    return _run_execution_command(args, _cmd_round_impl)


def cmd_campaign(args: argparse.Namespace) -> int:
    return _run_execution_command(args, _cmd_campaign_impl)


def fewshot_metric_call_budget(validation_tasks: int, seed_count: int,
                               has_gepa_instruction: bool,
                               demo_set_count: int) -> int:
    """Budget candidate discovery plus one nested-confirmation winner."""
    if min(validation_tasks, seed_count, demo_set_count) <= 0:
        raise ValueError("few-shot budget dimensions must be positive")
    instruction_variants = 2 if has_gepa_instruction else 1
    discovery_tasks = (validation_tasks + 1) // 2
    confirmation_tasks = validation_tasks // 2
    return (discovery_tasks * seed_count * demo_set_count * instruction_variants
            + confirmation_tasks * seed_count)


def cmd_review(args: argparse.Namespace) -> int:
    res = rv.decide(args.candidate_id, args.decision, reason=args.reason,
                    reviewer=args.reviewer,
                    accept_cost_overrun=bool(args.accept_cost_overrun))
    out: Dict[str, Any] = {"decision": args.decision, "review_path": res["path"]}
    if args.decision == "approve":
        cand = Candidate(**read_json(artifacts_path("candidates", f"{args.candidate_id}.json")))
        req = res["request"]
        new_rel = rel.publish(req["parent_release_id"], cand,
                              {"optimize": req["optimize_metrics"], "gate": req["gate_metrics"]})
        out["published_release"] = new_rel
        out["active_release"] = rel.current_release_id()
        if args.git:
            sha = _git_init_and_commit(f"feat(rsi): release {new_rel} ({args.candidate_id})")
            out["git_commit"] = sha
    _print("review", out)
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    before = rel.current_release_id() if (artifacts_path("releases", "ACTIVE")).exists() else None
    info = rel.rollback(args.to)
    after = rel.current_release_id()
    sync = info.get("governance_sync") or {}
    ok = after == args.to and sync.get("status") != "failed"
    if args.git:
        info["git_commit"] = _git_init_and_commit(f"revert(rsi): rollback to {args.to}")
    _print("rollback", {"before": before, "requested": args.to, "after": after,
                         "active_pointer_rolled_back": after == args.to,
                         "ok": ok, **info})
    return 0 if ok else 1


def _latest_ablation_brief() -> Dict[str, Any]:
    """最新消融报告的结论口径摘要（帮助确认"结论没被越界解读"）。"""
    try:
        files = sorted((artifacts_path("ablations")).glob("ablation-*.json"),
                       key=lambda p: p.stat().st_mtime)
        if not files:
            return {"exists": False}
        d = json.loads(files[-1].read_text(encoding="utf-8"))
        return {
            "exists": True,
            "path": files[-1].name,
            "design": d.get("design"),
            "noise": (d.get("runtime_snapshot") or {}).get("fake_noise"),
            "factors_available": d.get("factors_available"),
            "overlap_verdict": (d.get("behavior_overlap") or {}).get("verdict"),
            "synergy_claim": (d.get("interpretation") or {}).get("synergy_claim"),
        }
    except Exception as exc:  # pragma: no cover - 报告损坏时不应让 verify 崩掉
        return {"exists": False, "error": str(exc)}


def cmd_preregister(args: argparse.Namespace) -> int:
    """P5 预注册：打印/登记当前协议指纹，并报告各封存集的揭盲状态。

    用法见 `docs/preregistration/README.md`。核心是让"揭盲了几次、用的哪套协议"
    成为可以自动读出来的数，而不是靠人回忆。
    """
    from src.eval.final_eval import protocol_fingerprint, unblind_state

    cfg = load_config()
    fp = protocol_fingerprint(cfg)
    datasets = list(cfg.get("evaluation", {}).get("manifests", [])) or []
    names = [str(d).replace(".json", "") for d in datasets] or ["suppqa", "scholarqa_cs"]

    state = {ds: unblind_state(ds, fp) for ds in names}
    out: Dict[str, Any] = {"protocol_fingerprint": fp, "unblind_state": state}

    if not args.file:
        _print("preregister", out)
        return 0

    p = Path(args.file)
    text = p.read_text(encoding="utf-8")
    marker = "**协议指纹**（`protocol_fingerprint`）："
    if marker not in text:
        out["error"] = f"文件里没找到指纹字段（期望包含：{marker}）"
        out["ok"] = False
        _print("preregister", out)
        return 1

    import re
    m = re.search(re.escape(marker) + r"`?([0-9a-f]{16}|待填写)`?", text)
    current = m.group(1) if m else None
    out["file"] = str(p)
    out["fingerprint_in_file"] = current

    if args.write:
        new_text = re.sub(re.escape(marker) + r"`?[0-9a-f]{16}|待填写`?",
                          marker + f"`{fp}`", text, count=1)
        p.write_text(new_text, encoding="utf-8")
        out["written"] = fp
        out["ok"] = True
    else:
        out["ok"] = (current == fp)
        if current != fp:
            out["hint"] = ("文件里的指纹与当前协议不一致 —— 要么协议已变更（须新开一份编号），"
                           "要么还没登记（加 --write）")
    _print("preregister", out)
    return 0 if out["ok"] else 1


def cmd_verify(args: argparse.Namespace) -> int:
    # 【真实跑测加固 2026-09-25】这两个目录是接真实 API 后新出现的产物落点，
    # 若不纳管就形成"泄漏检查扫不到"的暗门：
    #   real_probe/runs  真实跑测的逐题产物（per_task.jsonl / failures.jsonl）
    #   http_cache       真实 API 的原始响应缓存（含题干、段落、canary 文本）
    # 后者尤其关键：一旦有人用真实 API 跑 final_holdout，题干与 canary 会原样落盘。
    roots = [artifacts_path("runs"), artifacts_path("rounds"),
             artifacts_path("candidates"), artifacts_path("reviews"),
             artifacts_path("multiseed"), artifacts_path("ablations"),
             artifacts_path("campaigns"),
             artifacts_path("real_probe"), artifacts_path("http_cache"),
             artifacts_path("pi_sessions")]
    leak = leakage.check_all(roots)
    pi_gate = leakage.check_pi_gate_sessions(artifacts_path("pi_sessions"))
    out: Dict[str, Any] = {"leakage": leak, "pi_gate_sessions": pi_gate,
                           "split_freeze": frz.verify()}
    ok = bool(leak["ok"]) and bool(pi_gate["ok"]) and bool(out["split_freeze"]["ok"])

    if args.strict:
        # 故障注入：证明"泄漏检查"真的能抓到泄漏，而不是永远返回 ok
        out["leakage_self_test"] = leakage.self_test()
        out["stats_self_test"] = stats_self_test()
        out["channels_self_test"] = channels_self_test()
        out["factorial_math_self_test"] = factorial_self_test()
        # R1b：证明"测不了的量不会被写成通过"（对齐 gepa api.py:286 的 fail-loud）
        out["budget_gate_self_test"] = budget_self_test()
        # P8：证明揭盲预算真的会拦（不是只记账）
        out["unblind_budget_self_test"] = unblind_budget_self_test()
        # 决策 2：证明三指标口径不会因为换个说法就放过刷分
        out["verdict_self_test"] = verdict_self_test()
        # R0：真实语料自检 —— 模板假话/答案直给一旦回到语料里，这里会红
        out["corpus_self_test"] = corpus_self_test()
        out["final_baseline"] = {
            "exists": baseline_exists(),
            "audit_entries": len(audit_log()),
        }
        ok = (ok and out["leakage_self_test"]["ok"] and out["stats_self_test"]["ok"]
              and out["channels_self_test"]["ok"] and out["factorial_math_self_test"]["ok"]
              and out["budget_gate_self_test"]["ok"]
              and out["unblind_budget_self_test"]["ok"]
              and out["verdict_self_test"]["ok"]
              and out["corpus_self_test"]["ok"])
        # 最新消融报告的结论口径（只报，不判失败：数值本身不该进 verify）
        out["latest_ablation"] = _latest_ablation_brief()

    out["ok"] = ok
    _print("verify", out)
    return 0 if ok else 1


def cmd_exec(args: argparse.Namespace) -> int:
    """受控执行网关自检：白名单求值 + LaTeX 渲染 + 越界拒绝 + WSL2 显式失败。"""
    cfg = load_config(args.config)
    cfg.setdefault("execution", {})["enabled"] = args.enable
    gw = build_gateway(cfg)
    cases: List[Dict[str, Any]] = [
        {"name": "calc_ok", "kind": "calc", "payload": {"expr": "round(100/3, 2)"}},
        {"name": "calc_vars", "kind": "calc", "payload": {"expr": "log10(n) * 100",
                                                          "variables": {"n": 1000}}},
        {"name": "calc_blocked_import", "kind": "calc", "payload": {"expr": "__import__('os')"}},
        {"name": "calc_blocked_attr", "kind": "calc",
         "payload": {"expr": "(1).__class__"}},
        {"name": "calc_blocked_func", "kind": "calc", "payload": {"expr": "open('x')"}},
        {"name": "latex_ok", "kind": "latex",
         "payload": {"template": r"$\mu = {mu} \pm {sd}$", "values": {"mu": 12.3, "sd": 0.4}}},
        {"name": "latex_missing", "kind": "latex",
         "payload": {"template": "$p = {p}$", "values": {}}},
        {"name": "python_isolated_blocked", "kind": "python_isolated", "payload": {}},
    ]
    out: List[Dict[str, Any]] = []
    for c in cases:
        try:
            out.append({"name": c["name"], **gw.run(c["kind"], c["payload"])})
        except Exception as exc:  # 高风险路径显式失败也要记为"被拒绝"，而不是让命令崩掉
            out.append({"name": c["name"], "ok": False, "kind": c["kind"], "output": "",
                        "error": str(exc), "elapsed_ms": 0, "expr_sha256": ""})
    # 关闭时"全部被拒绝"才是正确行为；开启时才要求白名单用例通过
    expect_ok = {"calc_ok", "calc_vars", "latex_ok"} if args.enable else set()
    passed = all((r["ok"] if r["name"] in expect_ok else not r["ok"]) for r in out
                 if r["name"] != "python_isolated_blocked")
    report = {"enabled": args.enable, "cases": out, "ok": bool(passed)}
    _print("exec", report)
    return 0 if report["ok"] else 1


def _require_fake_e2e_providers(cfg: Dict[str, Any]) -> None:
    """Prevent the isolated lifecycle self-test from scoring with live providers."""
    runtime_provider = str((cfg.get("runtime") or {}).get("provider") or "fake").lower()
    configured = {"runtime.provider": runtime_provider}
    models = cfg.get("models") or {}
    if isinstance(models, dict):
        for name, model_cfg in models.items():
            if isinstance(model_cfg, dict):
                configured[f"models.{name}.provider"] = str(
                    model_cfg.get("provider") or runtime_provider).lower()
    skills = cfg.get("skills") or {}
    if isinstance(skills, dict) and skills.get("router_provider") is not None:
        configured["skills.router_provider"] = str(
            skills["router_provider"]).lower()
    live = {name: provider for name, provider in configured.items()
            if provider != "fake"}
    if live:
        raise RuntimeError(
            "rsi e2e is fake-provider-only; live provider configs are not accepted: "
            + ", ".join(sorted(live)))


def cmd_e2e(args: argparse.Namespace) -> int:
    """隔离产物下验证候选门禁、人工 review、可发布时的回滚。"""
    if (not os.environ.get("RSI_HOME") or
            artifacts_root().resolve() == (repo_root() / "artifacts").resolve()):
        raise ValueError("e2e 会初始化和回滚 ACTIVE，必须先指定独立 RSI_HOME")
    cfg = load_config(args.config)
    _require_fake_e2e_providers(cfg)
    steps: List[Dict[str, Any]] = []

    skills_cfg = cfg.get("skills") or {}
    registry_ref = Path(str(skills_cfg.get("registry") or
                            "skills/registry.yaml"))
    baseline_release_id = str(skills_cfg.get("release_id") or "baseline-v0")
    rel.init_baseline(registry_path=path(*registry_ref.parts),
                      release_id=baseline_release_id)
    state = rel.load_state(baseline_release_id)
    steps.append({"step": "init", "release": state.release_id, "ok": True})

    fp = config_fingerprint(cfg)
    base = evaluate(cfg, state, "litqa2", "dev_optimize", seed=args.seed, limit=args.limit,
                    run_tag="baseline", cfg_fp=fp)
    steps.append({"step": "baseline", "run_id": base.run_id,
                  "accuracy": base.metric_values["accuracy"], "ok": True})

    report = run_round(cfg, datasets=["litqa2"], seed=args.seed, limit=args.limit)
    evaluated = [c for c in report["candidates"]
                 if c.get("candidate_id") and c.get("status") in
                 ("proposed_accept", "proposed_reject")]
    if not evaluated:
        _print("e2e", {"error": "本轮没有产出可评测的候选", "steps": steps})
        return 1
    cand = next((c for c in evaluated if c["status"] == "proposed_accept"),
                evaluated[0])
    steps.append({"step": "round", "round_id": report["round_id"], "candidate": cand["candidate_id"],
                  "verdict": cand["status"], "delta_pp": cand.get("delta_pp"), "ok": True})

    # 拒绝一次（只写 review，不发布）
    rej = rv.decide(cand["candidate_id"], "reject", reason="e2e: 人工拒绝路径验证")
    steps.append({"step": "reject", "review": rej["path"],
                  "active_release": rel.current_release_id(), "ok": True})

    if cand["status"] == "proposed_accept":
        app = rv.decide(cand["candidate_id"], "approve", reason="e2e: 人工批准路径验证")
        cand_obj = Candidate(**read_json(artifacts_path("candidates", f"{cand['candidate_id']}.json")))
        new_rel = rel.publish(app["request"]["parent_release_id"], cand_obj,
                              {"optimize": app["request"].get("optimize_metrics", {}),
                               "gate": app["request"].get("gate_metrics", {})})
        published = rel.current_release_id()
        steps.append({"step": "approve", "published": new_rel, "active_release": published,
                      "ok": published == new_rel})
    else:
        blocked = False
        try:
            rv.decide(cand["candidate_id"], "approve",
                      reason="e2e: 验证未过门候选不能发布")
        except rv.ReviewError:
            blocked = True
        steps.append({"step": "gate_blocks_approval", "ok": blocked and
                      rel.current_release_id() == state.release_id})

    # 回滚一次
    rel.rollback(state.release_id)
    after = rel.current_release_id()
    steps.append({"step": "rollback", "after": after,
                  "ok": after == state.release_id})

    # 泄漏检查
    leak = leakage.check_all([artifacts_path("runs"), artifacts_path("rounds"),
                              artifacts_path("candidates"), artifacts_path("reviews")])
    steps.append({"step": "leakage", "ok": leak["ok"]})

    ok = all(s.get("ok") for s in steps)
    _print("e2e", {"ok": ok, "steps": steps})
    return 0 if ok else 1


def _add_controlled_experiment_args(parser):
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true", help="保存零调用冻结计划")
    mode.add_argument("--execute", action="store_true", help="一次性执行调用方确认的计划")
    parser.add_argument("--expected-plan-sha256", default="")
    parser.add_argument("--base-release", default=None)
    parser.add_argument("--logical-request-cap", type=int, default=None)
    parser.add_argument("--physical-request-cap", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rsi", description="RSI 科研助手自进化系统 CLI")
    p.add_argument("--config", default=None, help="配置文件路径（默认 configs/experiment.yaml）")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="按配置建立不可变基线；或在独立 RSI_HOME 中建立 task_input 基线")
    s.add_argument("--git", action="store_true", help="同时 git init 并提交")
    task_input_baseline = s.add_mutually_exclusive_group()
    task_input_baseline.add_argument("--task-input-baseline", dest="task_input_baseline_version",
                                     action="store_const", const=1,
                                     help="独立 RSI_HOME 下建立 v1 无检索基线并快照 Skill 正文")
    task_input_baseline.add_argument("--task-input-baseline-v2",
                                     dest="task_input_baseline_version",
                                     action="store_const", const=2,
                                     help="独立 RSI_HOME 下建立 v2 十个自撰 Skills 的无检索基线")
    task_input_baseline.add_argument("--task-input-baseline-v3-label-compatible",
                                     dest="task_input_baseline_version",
                                     action="store_const", const=3,
                                     help="独立 RSI_HOME 下建立三分类标签兼容的新实验基线")
    task_input_baseline.add_argument("--task-input-baseline-v4-minimal-label-compatible",
                                     dest="task_input_baseline_version",
                                     action="store_const", const=4,
                                     help="独立 RSI_HOME 下建立只约束输出标签、不改类别判据的v4基线")
    task_input_baseline.add_argument("--task-input-baseline-v5-evidence-inference",
                                     dest="task_input_baseline_version",
                                     action="store_const", const=5,
                                     help="独立 RSI_HOME 下建立 Evidence Inference RCT效果方向标签契约基线")
    task_input_baseline.add_argument("--task-input-baseline-v6-healthver",
                                     dest="task_input_baseline_version",
                                     action="store_const", const=6,
                                     help="独立 RSI_HOME 下建立 HealthVer claim-abstract 三分类标签契约基线")
    task_input_baseline.add_argument("--task-input-baseline-v7-clinifact",
                                     dest="task_input_baseline_version",
                                     action="store_const", const=7,
                                     help="独立 RSI_HOME 下建立 CliniFact clinical-research 三分类标签契约基线")
    task_input_baseline.add_argument("--task-input-baseline-v8-clinifact",
                                     dest="task_input_baseline_version",
                                     action="store_const", const=8,
                                     help="独立 RSI_HOME 下建立统计显著性契约已校准的 CliniFact v2 基线")
    task_input_baseline.add_argument("--task-input-baseline-v9-scifact",
                                     dest="task_input_baseline_version",
                                     action="store_const", const=9,
                                     help="独立 RSI_HOME 下建立 SciFact 三分类固定摘要 task_input 基线")
    task_input_baseline.add_argument("--task-input-baseline-v10-clinifact",
                                     dest="task_input_baseline_version",
                                     action="store_const", const=10,
                                     help="独立 RSI_HOME 下建立输出格式与运行时一致的 CliniFact v3 基线")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("status", help="当前生效 release 与指纹")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("freeze", help="M2：冻结三层 split（此后改动会被拒绝）")
    s.add_argument("--upgrade-content", action="store_true",
                   help="保留旧冻结记录并开始实际题目内容绑定的新协议期")
    s.add_argument("--datasets", nargs="*", default=None)
    s.add_argument("--operator", default="rsi-cli")
    s.add_argument("--note", default="")
    s.set_defaults(func=cmd_freeze)

    s = sub.add_parser("baseline", help="A0 基线评测")
    s.add_argument("--datasets", nargs="*", default=["litqa2", "protocolqa"])
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--seed", type=int, default=20260924)
    s.add_argument("--seeds", nargs="*", type=int, default=None)
    baseline_mode = s.add_mutually_exclusive_group(required=True)
    baseline_mode.add_argument("--validate-only", action="store_true",
                               help="只用冻结 manifest 元数据写零调用计划")
    baseline_mode.add_argument("--execute", action="store_true",
                               help="执行计划哈希匹配的一次基线评测")
    s.add_argument("--expected-plan-sha256", default="",
                   help="--execute 必填，须匹配当前冻结的基线计划")
    s.set_defaults(func=cmd_baseline)

    s = sub.add_parser("round", help="跑一轮进化")
    s.add_argument("--datasets", nargs="*", default=None)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--seed", type=int, default=20260924)
    s.add_argument("--seeds", nargs="*", type=int, default=None,
                   help="多 seed 判定（每个 seed 都跑基线+候选，并给配对 bootstrap CI）")
    s.add_argument("--noise", type=float, default=None, help="覆盖 runtime.fake_noise")
    s.add_argument("--max-candidates", type=int, default=None)
    s.add_argument("--task-input", action="store_true",
                   help="用题目自带 evidence/context 做自进化；不调用检索，且不与旧封存基线配对")
    s.add_argument("--pi", action="store_true",
                   help="在 task_input 轮次中用 Pi harness 执行任务模型并记录 JSONL；新实验协议")
    s.add_argument("--gepa", action="store_true",
                   help="在 task_input 模式下使用 vendor GEPA 的真实多候选 Prompt 搜索")
    s.add_argument("--fewshot-search", action="store_true",
                   help="在 Pi task_input 轮次中搜索并评测可发布的 task-demo 工件")
    s.add_argument("--fewshot-metric-calls", type=int, default=None,
                   help="few-shot 候选验证调用上限；默认按完整内部验证网格自动计算")
    s.add_argument("--fewshot-candidate-sets", type=int, default=None,
                   help="few-shot demo 候选组合数；未指定时使用配置值，默认 3")
    s.add_argument("--fewshot-replay-report", default=None,
                   help="从同协议的已选 task-demo 搜索报告重放候选，跳过重复的内层搜索")
    s.add_argument("--gepa-target", choices=["prompt", "skill-body"], default="prompt",
                   help="本轮只搜索 Prompt 或指定 Skill 正文；后者是显式工件实验")
    s.add_argument("--gepa-skill", default=None,
                   help="--gepa-target skill-body 时指定可变 Skill 名称")
    s.add_argument("--gepa-metric-calls", type=int, default=None,
                   help="GEPA 逐题评测调用上限")
    s.add_argument("--gepa-reflection-minibatch-size", type=int, default=None,
                   help="GEPA 每次反思提供的任务反馈数；写入协议指纹")
    s.add_argument("--gepa-acceptance-criterion",
                   choices=["strict_improvement", "improvement_or_equal",
                            "any_objective_improved"], default=None,
                   help="GEPA 局部接受条件；同分候选仍需通过完整 val 与外层门")
    s.add_argument("--gepa-multiobjective", action="store_true",
                   help="探索 accuracy/coverage/answer-quality 的 GEPA Pareto 搜索目标")
    s.add_argument("--gepa-use-merge", action="store_true",
                   help="启用 pinned GEPA 的 Pareto 候选合并；写入协议指纹")
    s.add_argument("--gepa-merge-max-invocations", type=int, default=None,
                   help="GEPA 合并提案上限；默认 5")
    s.add_argument("--gepa-merge-val-overlap-floor", type=int, default=None,
                   help="候选合并的最小验证题重叠数；默认 5")
    s.add_argument("--gepa-reflection-model", default=None,
                   help="显式指定 GEPA 反思模型；写入配置指纹与搜索报告")
    s.add_argument("--gepa-task-offset", type=int, default=None,
                   help="每个数据集内部 dev_optimize 排序偏移；写入协议指纹和候选重放配置")
    s.add_argument("--gepa-feedback-signals", action="store_true",
                   help="探索性加入材料中的否定结果句；改变优化器协议指纹")
    s.add_argument("--gepa-replay-report", default=None,
                   help="重放同协议已落盘 GEPA 搜索候选；仅供探索与流程验证")
    s.add_argument("--gepa-replay-index", type=int, default=1,
                   help="搜索报告中预选的候选序号，不能按 gate 结果选择")
    s.add_argument("--attribution-corrections", default=None,
                   help="应用经 failure/hash/trace 校验的人工归因修正 JSON")
    round_plan = s.add_mutually_exclusive_group()
    round_plan.add_argument("--validate-only", action="store_true",
                            help="为实时进化轮次写零调用冻结计划")
    round_plan.add_argument("--execute", action="store_true",
                            help="执行经哈希确认的一次实时进化轮次")
    s.add_argument("--expected-plan-sha256", default="",
                   help="--execute 必填，须匹配当前冻结计划")
    s.add_argument("--logical-request-cap", type=int, default=None,
                   help="实时执行必填：共享调用账本逻辑请求上限")
    s.add_argument("--physical-request-cap", type=int, default=None,
                   help="实时执行必填：共享调用账本物理 HTTP 上限")
    s.set_defaults(func=cmd_round)

    s = sub.add_parser("campaign", help="连续多轮进化（≥3 轮）")
    s.add_argument("--resume-campaign", help="续接指定 campaign 的人工评审后阶段")
    s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--datasets", nargs="*", default=None)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--seed", type=int, default=20260924)
    s.add_argument("--seeds", nargs="*", type=int, default=None)
    s.add_argument("--noise", type=float, default=None)
    approval = s.add_mutually_exclusive_group()
    approval.add_argument("--approve-simulated", action="store_true",
                          help="仅在 RSI_HOME 独立产物目录中模拟批准与发布，用于闭环测试")
    approval.add_argument("--no-approve", action="store_true",
                          help="兼容旧命令；默认即只产出 review 请求")
    s.add_argument("--git", action="store_true")
    campaign_plan = s.add_mutually_exclusive_group()
    campaign_plan.add_argument("--validate-only", action="store_true",
                               help="为实时 campaign 写零调用冻结计划")
    campaign_plan.add_argument("--execute", action="store_true",
                               help="执行经哈希确认的一次实时 campaign")
    s.add_argument("--expected-plan-sha256", default="",
                   help="--execute 必填，须匹配当前冻结计划")
    s.add_argument("--logical-request-cap", type=int, default=None,
                   help="实时执行必填：共享调用账本逻辑请求上限")
    s.add_argument("--physical-request-cap", type=int, default=None,
                   help="实时执行必填：共享调用账本物理 HTTP 上限")
    s.set_defaults(func=cmd_campaign)

    s = sub.add_parser(
        "multiseed", help="冻结计划绑定的可见分区多 seed 评分/对照")
    s.add_argument("--datasets", nargs="*", default=None)
    s.add_argument("--seeds", nargs="*", type=int, default=None)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--noise", type=float, default=None)
    s.add_argument("--compare", default=None,
                   help="与指定 release 对比（不填则只跑当前 release 的多 seed 稳定性）")
    s.add_argument("--candidate", default=None, help="与未发布 candidate JSON 对照；不能同时指定 --compare")
    _add_controlled_experiment_args(s)
    s.set_defaults(func=cmd_multiseed)

    s = sub.add_parser("ablate", help="冻结计划绑定的可见分区消融")
    s.add_argument("--datasets", nargs="*", default=None)
    s.add_argument("--splits", nargs="*", default=None)
    s.add_argument("--seeds", nargs="*", type=int, default=None)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--seed", type=int, default=20260924)
    s.add_argument("--noise", type=float, default=None)
    s.add_argument("--prompt-release", default=None, help="P 因子用该 release 的 prompt")
    s.add_argument("--desc-release", default=None, help="D 因子用该 release 的 skill descriptions")
    s.add_argument("--body-release", default=None, help="B 因子用该 release 的 skill 正文")
    s.add_argument("--condition", action="append", default=None,
                   help="明确条件 LABEL=RELEASE，可重复；支持 A0 到 A4，A0 必须为基线")
    s.add_argument("--legacy-union", action="store_true",
                   help="关闭行为通道（回到 M3 的并集口径），用于对照实验")
    _add_controlled_experiment_args(s)
    s.set_defaults(func=cmd_ablate)

    s = sub.add_parser("final-baseline", help="final_holdout 基线终评（只跑一次，随后封存至 M4）")
    final_mode = s.add_mutually_exclusive_group(required=True)
    final_mode.add_argument("--validate-only", action="store_true",
                            help="冻结零调用封存评测计划；不读取题目行或调用模型")
    final_mode.add_argument("--execute", action="store_true",
                            help="仅执行经明确批准并传入哈希的封存评测计划")
    s.add_argument("--expected-plan-sha256", default="",
                   help="--execute 必填，需与当前冻结计划完全一致")
    s.add_argument("--datasets", nargs="*", default=None)
    s.add_argument("--seed", type=int, default=20260924)
    s.add_argument("--seeds", nargs="*", type=int, default=None)
    s.add_argument("--release", default=None, help="基线 release（默认 ACTIVE；M4 应显式传 baseline-v0）")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--actor", default="evaluator")
    s.add_argument("--reason", default="")
    s.add_argument("--force", action="store_true", help="强制重跑（会写入审计日志）")
    s.add_argument("--budget-override-reason", default=None,
                   help="超出揭盲预算时显式授权的理由（必填才放行，写入账本可追责）")
    s.add_argument("--budget-override-actor", default=None, help="授权人（默认同 --actor）")
    s.set_defaults(func=cmd_final_baseline)

    s = sub.add_parser("final-eval", help="M4 冻结终评：当前 release vs 封存基线（配对 CI + 成本 + 负面结果）")
    final_mode = s.add_mutually_exclusive_group(required=True)
    final_mode.add_argument("--validate-only", action="store_true",
                            help="冻结零调用封存评测计划；不读取题目行或调用模型")
    final_mode.add_argument("--execute", action="store_true",
                            help="仅执行经明确批准并传入哈希的封存评测计划")
    s.add_argument("--expected-plan-sha256", default="",
                   help="--execute 必填，需与当前冻结计划完全一致")
    s.add_argument("--datasets", nargs="*", default=None)
    s.add_argument("--seeds", nargs="+", type=int, required=True,
                   help="计划绑定的终评 seed；每个 seed 必须在封存基线中有配对结果")
    s.add_argument("--release", default=None, help="终评 release（默认 ACTIVE）")
    s.add_argument("--transfer", action="store_true", help="同时跑迁移数据集（ScholarQABench CS）")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--actor", default="evaluator")
    s.add_argument("--reason", default="")
    s.add_argument("--no-strict-protocol", action="store_true",
                   help="放弃与基线的协议一致性校验（仅调试用，结果不可报告）")
    s.add_argument("--budget-override-reason", default=None,
                   help="超出揭盲预算时显式授权的理由（必填才放行，写入账本可追责）")
    s.add_argument("--budget-override-actor", default=None, help="授权人（默认同 --actor）")
    s.set_defaults(func=cmd_final_eval)

    s = sub.add_parser("final-budget", help="P8 揭盲预算账本：已用/限额/剩余/是否退役")
    s.add_argument("--datasets", nargs="*", default=None)
    s.add_argument("--actor", default="evaluator")
    s.add_argument("--reason", default="")
    s.add_argument("--adopt", action="store_true",
                   help="首次启用：快照当前已用额度并建立账本文件")
    s.add_argument("--strict", action="store_true",
                   help="任一数据集揭盲额度用尽时返回非 0（CI 守门用）")
    s.set_defaults(func=cmd_final_budget)

    s = sub.add_parser("gate-verdict",
                       help="决策1：在 selection_gate 上比较两个 release（不花封存揭盲预算）")
    s.add_argument("--challenger", required=True, help=" challenger release（待验证）")
    s.add_argument("--champion", default=None, help="当前 champion（默认 ACTIVE）")
    s.add_argument("--datasets", nargs="*", default=None)
    s.add_argument("--split", choices=["selection_gate"], default="selection_gate",
                   help="正式候选门禁固定使用 selection_gate；优化/终评走独立入口")
    s.add_argument("--seeds", nargs="*", type=int, default=None)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--no-protocol-check", action="store_true",
                   help="离线夹具等不与封存基线配对的场景：协议项记为『未检查』（非通过）")
    mode = s.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true",
                      help="只根据 manifest 元数据写零调用冻结计划")
    mode.add_argument("--execute", action="store_true",
                      help="执行一次计划绑定的 aggregate-only gate 比较")
    s.add_argument("--expected-plan-sha256", default="",
                   help="--execute 必填，必须与当前冻结计划一致")
    s.set_defaults(func=cmd_gate_verdict)

    s = sub.add_parser("alias", help="决策4：champion / challenger 治理层")
    s.add_argument("--set", choices=["champion", "challenger"], default=None)
    s.add_argument("release", nargs="?", default=None, help="--set 时的目标 release")
    s.add_argument("--actor", default="cli")
    s.add_argument("--reason", default="")
    s.add_argument("--adopt", action="store_true", help="建立 alias 文件（把当前 ACTIVE 记为 champion）")
    s.add_argument("--caveat", default=None,
                   help="给当前 champion 挂 caveat（换人/回滚时自动清空，避免旧警告跟着新版本）")
    s.add_argument("--resolve-caveat", default=None, metavar="ID_OR_SUBSTR",
                   help="把某条 caveat 标记为已解决（按 id 或子串匹配；留痕不删除）")
    s.set_defaults(func=cmd_alias)

    s = sub.add_parser("promote", help="决策4：challenger 升为 champion（过发布门 + sign-off）")
    s.add_argument("--reviewer", required=True)
    s.add_argument("--reason", required=True)
    s.add_argument("--gate-report", required=True, help="完整 gate-verdict 报告路径")
    s.set_defaults(func=cmd_promote)

    s = sub.add_parser("reseal-plan", help="决策3：待重封清单（须合并为一次重封）")
    s.set_defaults(func=cmd_reseal_plan)

    s = sub.add_parser("review", help="人工批准 / 拒绝候选")
    s.add_argument("--candidate-id", required=True)
    s.add_argument("--decision", required=True, choices=["approve", "reject"])
    s.add_argument("--reason", default="")
    s.add_argument("--reviewer", default="human")
    s.add_argument("--accept-cost-overrun", action="store_true",
                   help="明确批准该请求中已测量的成本超容差例外；仅限已提交人工评审的例外请求")
    s.add_argument("--git", action="store_true")
    s.set_defaults(func=cmd_review)

    s = sub.add_parser("rollback", help="单命令回滚到指定 release")
    s.add_argument("--to", required=True)
    s.add_argument("--git", action="store_true")
    s.set_defaults(func=cmd_rollback)

    s = sub.add_parser("exec", help="受控执行网关自检（轻量表达式 / LaTeX，默认关闭）")
    s.add_argument("--enable", action="store_true", help="本次运行时临时开启网关")
    s.set_defaults(func=cmd_exec)

    s = sub.add_parser("verify", help="泄漏检查 + split 冻结校验")
    s.add_argument("--strict", action="store_true",
                   help="额外跑故障注入自检（证明检查真的能抓到泄漏）与统计模块自检")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("e2e", help="端到端闭环自检")
    s.add_argument("--limit", type=int, default=60)
    s.add_argument("--seed", type=int, default=20260924)
    s.set_defaults(func=cmd_e2e)

    s = sub.add_parser("preregister", help="P5 预注册：登记/校验协议指纹与各集揭盲次数")
    s.add_argument("--file", default=None,
                   help="预注册文件路径（docs/preregistration/*.md）；"
                        "省略时只打印当前指纹与各集揭盲状态")
    s.add_argument("--write", action="store_true",
                   help="把当前协议指纹写进文件的『协议指纹』字段")
    s.set_defaults(func=cmd_preregister)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    # Pinned vendor GEPA writes candidate text through locale-default `open()`.
    # Windows' GBK default cannot persist ordinary Unicode prompt punctuation.
    # Re-run only the GEPA CLI with Python UTF-8 mode before any GEPA state is written.
    if (argv is None and os.name == "nt" and not sys.flags.utf8_mode
            and "--gepa" in sys.argv
            and os.environ.get("RSI_GEPA_UTF8_REEXEC") != "1"):
        env = os.environ.copy()
        env["RSI_GEPA_UTF8_REEXEC"] = "1"
        return subprocess.run(
            [sys.executable, "-X", "utf8", *sys.argv], env=env, check=False).returncode
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
