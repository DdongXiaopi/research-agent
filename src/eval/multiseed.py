"""多 seed 重复运行与配对比较（M2 / Spec §4-M2 第 3 条）。

设计要点（与 Spec §6.1 铁律一致）：
- `dev_optimize`：**逐题配对** bootstrap（同一批题、同一顺序，消掉题目难度方差）→ 主证据
- `selection_gate`：**只回传聚合**，因此只能报跨 seed 的聚合差值均值/范围，
  明确不给 CI（样本量=seed 数，做 CI 是自欺欺人）

每个 seed 的 run 都是独立的 `evaluate()` 调用，run_id 含 seed，产物可逐个复查。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..runtime.runner import ArtifactState
from ..util import now_iso, artifacts_path, path, sha256_obj, write_json
from .harness import evaluate
from .stats import paired_bootstrap_binary, summary

MULTISEED_ROOT = artifacts_path("multiseed")


def _validate_paired_task_scope(base, candidate, *, seed: int,
                                require_per_task: bool) -> List[str]:
    """Require the aggregate and bootstrap comparisons to use the same cohort."""
    if base.n_tasks != candidate.n_tasks:
        raise ValueError(
            f"paired comparison task cohort size differs for seed {seed}: "
            f"baseline={base.n_tasks}, candidate={candidate.n_tasks}")
    if base.task_cohort_sha256 != candidate.task_cohort_sha256:
        raise ValueError(f"paired comparison task cohort fingerprint differs for seed {seed}")
    if not require_per_task:
        return []

    base_ids = [str(row.get("task_id", "")) for row in base.per_task]
    candidate_ids = [str(row.get("task_id", "")) for row in candidate.per_task]
    if (len(base_ids) != base.n_tasks or len(candidate_ids) != candidate.n_tasks
            or any(not task_id for task_id in base_ids + candidate_ids)
            or len(set(base_ids)) != len(base_ids)
            or len(set(candidate_ids)) != len(candidate_ids)
            or set(base_ids) != set(candidate_ids)):
        raise ValueError(f"paired comparison task cohort rows differ for seed {seed}")
    return sorted(base_ids)


def _validated_seeds(seeds: Sequence[int]) -> List[int]:
    """Require the M2 minimum and prevent repeated seeds masquerading as runs."""
    normalized = [int(seed) for seed in seeds]
    if len(normalized) < 3:
        raise ValueError("M2 multiseed evaluation requires at least three distinct seeds")
    if len(set(normalized)) != len(normalized):
        raise ValueError("M2 multiseed evaluation does not allow duplicate seeds")
    return normalized


def _seed_sampling_metadata(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Describe whether seed labels control the provider's stochasticity."""
    runtime = cfg.get("runtime", {}) or {}
    provider = str(runtime.get("provider", "fake")).lower()
    if provider == "fake":
        noise = float(runtime.get("fake_noise", 0.0))
        return {
            "noise": noise,
            "deterministic_provider": noise <= 0.0,
            "seed_controls_provider_output": noise > 0.0,
            "seed_variability_mode": (
                "synthetic_fake_jitter" if noise > 0.0 else "deterministic_fake"),
            "note": (
                "fake_noise=0：FakeProvider 确定性输出，seed 不产生方差；"
                "跨 seed 方差不代表真实模型方差"
                if noise <= 0.0 else
                "fake_noise>0：seed 控制合成题级噪声，仅用于验证统计链路，"
                "不代表真实模型方差"),
        }
    return {
        "noise": None,
        "deterministic_provider": None,
        "seed_controls_provider_output": False,
        "seed_variability_mode": "uncontrolled_remote_sampling",
        "note": ("run seed 仅标识本地运行；HTTP provider 还用它隔离本地响应缓存，"
                 "但不会作为远端 RNG seed 发送。"
                 "不同 seed 是独立请求/缓存样本，但输出随机性由 provider 配置控制，"
                 "不构成 seed-controlled RNG reproducibility。"),
    }


def eval_seeds(cfg: Dict[str, Any], state: ArtifactState, dataset: str, split: str,
               seeds: Sequence[int], limit: Optional[int] = None,
               run_tag: str = "multiseed", metric: str = "accuracy") -> Dict[str, Any]:
    """在同一状态上跑多个 seed，返回逐 seed 指标与跨 seed 汇总。"""
    seed_values = _validated_seeds(seeds)
    write_per_task = split == "dev_optimize"     # gate 分区不落逐题结果（铁律 2）
    per_seed: List[Dict[str, Any]] = []
    for s in seed_values:
        res = evaluate(cfg, state, dataset, split, seed=int(s), limit=limit,
                       run_tag=run_tag, write_per_task=write_per_task,
                       cfg_fp=sha256_obj(cfg))
        row: Dict[str, Any] = {
            "seed": int(s),
            "run_id": res.run_id,
            "n_tasks": res.n_tasks,
            "metrics": res.metric_values,
            "metric": res.metric_values.get(metric, 0.0),
        }
        per_seed.append(row)
    vals = [float(r["metric"]) for r in per_seed]
    sampling = _seed_sampling_metadata(cfg)
    return {
        "dataset": dataset,
        "split": split,
        "metric": metric,
        "seeds": seed_values,
        **sampling,
        "per_seed": per_seed,
        "summary": summary(vals),
        "created_at": now_iso(),
    }


def compare(cfg: Dict[str, Any], base_state: ArtifactState, cand_state: ArtifactState,
            dataset: str, split: str, seeds: Sequence[int],
            limit: Optional[int] = None, metric: str = "accuracy",
            B: int = 10000, rng_seed: int = 20260924,
            base_tag: str = "base", cand_tag: str = "cand") -> Dict[str, Any]:
    """基线 vs 候选：逐 seed 配对比较 + bootstrap CI。"""
    seed_values = _validated_seeds(seeds)
    write_per_task = split == "dev_optimize"
    if write_per_task and metric != "accuracy":
        raise ValueError(
            "per-task bootstrap currently only supports accuracy; "
            f"got metric={metric!r}")
    if write_per_task:
        if isinstance(B, bool):
            raise ValueError("paired bootstrap requires a positive integer replicate count")
        try:
            bootstrap_replicates = int(B)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "paired bootstrap requires a positive integer replicate count") from exc
        if bootstrap_replicates < 1 or bootstrap_replicates != B:
            raise ValueError("paired bootstrap requires a positive integer replicate count")
        B = bootstrap_replicates
    per_seed: List[Dict[str, Any]] = []

    for s in seed_values:
        b = evaluate(cfg, base_state, dataset, split, seed=int(s), limit=limit,
                     run_tag=base_tag, write_per_task=write_per_task, cfg_fp=sha256_obj(cfg))
        c = evaluate(cfg, cand_state, dataset, split, seed=int(s), limit=limit,
                     run_tag=cand_tag, write_per_task=write_per_task, cfg_fp=sha256_obj(cfg))
        ids = _validate_paired_task_scope(
            b, c, seed=int(s), require_per_task=write_per_task)
        bv = float(b.metric_values.get(metric, 0.0))
        cv = float(c.metric_values.get(metric, 0.0))
        row: Dict[str, Any] = {
            "seed": int(s),
            "base_run_id": b.run_id,
            "cand_run_id": c.run_id,
            "n_tasks": b.n_tasks,
            "base": round(bv, 6),
            "cand": round(cv, 6),
            "delta_pp": round((cv - bv) * 100.0, 4),
        }
        if write_per_task and b.per_task and c.per_task:
            bb = {str(r["task_id"]): int(bool(r["correct"])) for r in b.per_task}
            cc = {str(r["task_id"]): int(bool(r["correct"])) for r in c.per_task}
            ci = paired_bootstrap_binary([bb[i] for i in ids], [cc[i] for i in ids],
                                         B=B, rng_seed=rng_seed)
            if round((cv - bv) * 100.0, 4) != ci["delta_pp"]:
                raise ValueError(
                    f"aggregate and paired task accuracy differ for seed {s}")
            row.update({
                "pairing": "per_task",
                "n_pairs": len(ids),
                "ci_low_pp": ci["ci_low_pp"],
                "ci_high_pp": ci["ci_high_pp"],
                "p_delta_gt_0": ci["p_delta_gt_0"],
                "bootstrap_B": ci["B"],
            })
        else:
            row.update({
                "pairing": "per_seed_aggregate",
                "n_pairs": 0,
                "ci_low_pp": None,
                "ci_high_pp": None,
                "p_delta_gt_0": None,
                "note": "selection_gate 只回传聚合指标，无法逐题配对；此处不给出 CI",
            })
        per_seed.append(row)

    deltas = [float(r["delta_pp"]) for r in per_seed]
    probs = [r["p_delta_gt_0"] for r in per_seed if r.get("p_delta_gt_0") is not None]
    return {
        "dataset": dataset,
        "split": split,
        "metric": metric,
        "seeds": seed_values,
        "base_release": base_state.release_id,
        "cand_release": cand_state.release_id,
        **_seed_sampling_metadata(cfg),
        "per_seed": per_seed,
        "summary": {
            **summary(deltas),
            "min_p_delta_gt_0": round(min(probs), 4) if probs else None,
            "all_seeds_positive": bool(deltas) and all(d > 0 for d in deltas),
        },
        "created_at": now_iso(),
    }


def save(report: Dict[str, Any], name: str) -> str:
    rid = sha256_obj({k: report.get(k) for k in ("dataset", "split", "seeds", "created_at")})[:12]
    p = MULTISEED_ROOT / f"{name}-{rid}.json"
    write_json(p, report)
    return p.as_posix()
