"""消融：2^3 全因子设计 + 交互项 + Shapley（M3.1 重写；保留 A0–A4 的 OFAT 视图）。

为什么从 A0–A4 换成全因子
--------------------------
A0/A1/A2/A3/A4 是**一次一因子**（OFAT）设计：每次只开一个因子，最后一格全开。
它在数学上估不出交互效应 —— Fisher 1935 年就证明了。也就是说
"A4 ≤ max(A1,A2,A3) ⇒ 不能声称协同"这个推理，用 OFAT 表**本来就没有资格做**，
无论开关重不重叠。上游也不这么做：GEPA 压根不做组件消融，而是靠
`proposer/merge.py` 的"合并前要求两个候选确实分歧"来做信用分配；
DSPy MIPROv2 把 instruction × few-shot 当联合搜索空间，用 TPE 直接建模交互。

2^3 全因子只要 8 个条件（含全关的 A0），一次跑完可同时得到：
  - 3 个主效应
  - 3 个二阶交互 + 1 个三阶交互
  - 加性预测的残差（观测值 − 加性叠加）
  - **Shapley 值** —— 它需要的 2^3 个子集价值函数正好就是这 8 个条件，
    因此归因不额外花一次实验

因子编码 `F{p}{d}{b}`：
  P = system_prompt（evolved prompt）
  D = skill_description（evolved descriptions）
  B = skill artifact（evolved 正文 + 结晶的新 Skill）

口径：
- `dev_optimize`：逐题配对 bootstrap → 给 Δ 与 95% CI、P(Δ>0)
- `selection_gate`：只回传聚合 → 只给跨 seed 的均值/范围，**不给 CI**
- 单次运行内 8 个条件同 seed、同噪声、同夹具 —— 不再出现"两张表噪声不同"的口径问题
"""

from __future__ import annotations

import itertools
import os
import re
from math import factorial
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..governance import release as rel
from ..runtime.channels import behavior_profile, load_channels, overlap_matrix
from ..runtime.runner import ArtifactState
from ..util import now_iso, artifacts_path, path, sha256_obj, write_json
from .harness import evaluate
from .stats import paired_bootstrap, summary

ABLATION_ROOT = artifacts_path("ablations")


def _require_live_execution_guard(cfg: Dict[str, Any],
                                  execution_plan_sha256: str | None) -> None:
    """Keep real ablation scoring behind a challenged plan and bounded ledger."""
    provider = str((cfg.get("runtime") or {}).get("provider") or "fake").lower()
    if provider == "fake":
        return
    if not re.fullmatch(r"[0-9a-f]{64}", execution_plan_sha256 or ""):
        raise PermissionError(
            "live ablation scoring requires a verified frozen execution-plan SHA-256")
    if os.environ.get("RSI_ACTIVE_EXECUTION_PLAN_SHA256") != execution_plan_sha256:
        raise PermissionError(
            "live ablation scoring requires the active verified execution plan")
    if not os.environ.get("RSI_CALL_LEDGER_PATH"):
        raise PermissionError(
            "live ablation scoring requires the bounded append-only request ledger")
    for name in ("RSI_CALL_LEDGER_MAX_LOGICAL", "RSI_CALL_LEDGER_MAX_PHYSICAL"):
        try:
            value = int(os.environ.get(name, "0"))
        except ValueError:
            value = 0
        if value < 1:
            raise PermissionError(
                f"live ablation scoring requires a positive {name} ceiling")

# --- OFAT（旧口径，保留兼容） -----------------------------------------------
CONDITIONS = ("A0", "A1", "A2", "A3", "A4")
CONDITION_LABELS = {
    "A0": "baseline（v0 prompt + v0 Skill）",
    "A1": "prompt-only（evolved prompt + v0 Skill）",
    "A2": "description-only（v0 prompt + evolved descriptions）",
    "A3": "Skill-artifact-only（v0 prompt + evolved body/新 Skill）",
    "A4": "combined（prompt + description + body/新 Skill）",
}

# --- 2^3 全因子 -------------------------------------------------------------
FACTORS: Tuple[str, ...] = ("P", "D", "B")
FACTOR_ARTIFACTS = {
    "P": "system_prompt",
    "D": "skill_description",
    "B": "skill_body + new_skill",
}
FACTOR_LABELS = {
    "P": "prompt（evolved system_prompt）",
    "D": "description（evolved skill descriptions）",
    "B": "skill 工件（evolved 正文 + 结晶的新 Skill）",
}
_CODE_RE = re.compile(r"^F([01])([01])([01])$")
# OFAT 条件 → 全因子条件码
OFAT_MAP = {"A0": "F000", "A1": "F100", "A2": "F010", "A3": "F001", "A4": "F111"}
SINGLETON_CODES = ("F100", "F010", "F001")
# 次级指标：主指标（accuracy）看不见的收益都写在这里，避免"某维度无效"的误读
SECONDARY_METRICS = ("skill_recall_rate", "abstain_rate", "unsupported_answer_rate",
                     "citation_precision")


def factorial_codes() -> List[str]:
    return [f"F{p}{d}{b}" for p in (0, 1) for d in (0, 1) for b in (0, 1)]


def decode(code: str) -> Dict[str, int]:
    m = _CODE_RE.match(code)
    if not m:
        raise ValueError(f"不是合法的全因子条件码: {code}")
    return {"P": int(m.group(1)), "D": int(m.group(2)), "B": int(m.group(3))}


def code_of(levels: Dict[str, int]) -> str:
    return f"F{int(levels.get('P', 0))}{int(levels.get('D', 0))}{int(levels.get('B', 0))}"


def condition_label(code: str) -> str:
    lv = decode(code)
    on = [FACTOR_LABELS[f] for f in FACTORS if lv[f]]
    return "全关（v0）" if not on else " + ".join(on)


def _terms() -> List[Tuple[str, ...]]:
    """主效应 + 二阶交互 + 三阶交互（顺序固定，保证报告可复现）。"""
    out: List[Tuple[str, ...]] = [(f,) for f in FACTORS]
    out += [t for t in itertools.combinations(FACTORS, 2)]
    out += [FACTORS]
    return out


def effect_weights(term: Tuple[str, ...]) -> Dict[str, float]:
    """标准 2^3 因子效应的权重：effect = Σ_c w_c · y_c（= contrast / 2^{k-1}）。

    用 -1/+1 编码（低水平 -1、高水平 +1），对比度除以 2^{k-1}=4。
    主效应因此等于"在其它因子的所有水平上取平均的简单效应"。
    """
    w: Dict[str, float] = {}
    for p in (0, 1):
        for d in (0, 1):
            for b in (0, 1):
                lv = {"P": p, "D": d, "B": b}
                sign = 1.0
                for f in term:
                    sign *= 1.0 if lv[f] else -1.0
                w[f"F{p}{d}{b}"] = sign / (2 ** (len(FACTORS) - 1))
    return w


def did_multiplier(term: Tuple[str, ...]) -> int:
    """标准效应 → "差分的差分"直观口径的换算系数。

    主效应两者一致（×1）；二阶交互的 DiD 是标准效应的 2 倍；三阶是 4 倍。
    """
    return 2 ** (len(term) - 1)


def shapley_weights() -> Dict[str, Dict[str, float]]:
    """每个因子的 Shapley 值写成"8 个条件值"的线性组合权重。

    φ_i = Σ_{S⊆N\\{i}} |S|!(n−|S|−1)!/n! · [v(S∪{i}) − v(S)]
    n=3 时权重为 1/3（|S|=0）、1/6（|S|=1）、1/3（|S|=2）。
    因为 φ 是 v 的线性函数，可以逐题套用同一组权重做 bootstrap。
    """
    n = len(FACTORS)
    out: Dict[str, Dict[str, float]] = {f: {c: 0.0 for c in factorial_codes()} for f in FACTORS}
    for f in FACTORS:
        others = [g for g in FACTORS if g != f]
        for r in range(len(others) + 1):
            for S in itertools.combinations(others, r):
                w = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                base = {g: (1 if g in S else 0) for g in FACTORS}
                out[f][code_of(base)] -= w
                with_f = dict(base)
                with_f[f] = 1
                out[f][code_of(with_f)] += w
    return out


def factorial_effects(y: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """由 8 个条件的指标值算出全部效应（标准口径 + DiD 口径）。"""
    out: Dict[str, Dict[str, float]] = {}
    for term in _terms():
        w = effect_weights(term)
        val = sum(w[c] * float(y[c]) for c in w)
        name = "×".join(term)
        out[name] = {
            "effect": round(val, 6),
            "effect_pp": round(val * 100.0, 4),
            "did_pp": round(val * 100.0 * did_multiplier(term), 4),
            "order": len(term),
        }
    return out


def shapley_values(y: Dict[str, float]) -> Dict[str, float]:
    """三个因子的 Shapley 归因（效率性：Σφ = v(N) − v(∅)）。"""
    w = shapley_weights()
    return {f: round(sum(w[f][c] * float(y[c]) for c in w[f]), 6) for f in FACTORS}


def additive_prediction(y: Dict[str, float]) -> Dict[str, float]:
    """以全关条件为基准的**加性叠加**预测，以及它与实测全开条件的残差。

    add = A0 + (P−A0) + (D−A0) + (B−A0)
    residual = 实测全开 − add  ← 这就是"交互"的总量（三阶 + 各二阶之和）
    residual > 0 才谈得上协同；≈0 说明三维只是叠加。
    """
    add = float(y["F100"]) + float(y["F010"]) + float(y["F001"]) - 2.0 * float(y["F000"])
    obs = float(y["F111"])
    return {
        "additive_prediction": round(add, 6),
        "observed_all_on": round(obs, 6),
        "residual_interaction": round(obs - add, 6),
        "residual_interaction_pp": round((obs - add) * 100.0, 4),
    }


def build_conditions(base: ArtifactState,
                     evolved_prompt: Optional[str] = None,
                     evolved_registry: Any = None,
                     body_registry: Any = None,
                     combined_registry: Any = None) -> Dict[str, ArtifactState]:
    """按 §6.4 组合出各条件；缺失的那一维退化为 v0（并在报告里标注）。"""
    return {
        "A0": ArtifactState(release_id=base.release_id,
                            system_prompt=base.system_prompt, registry=base.registry,
                            task_demos=base.task_demos),
        "A1": ArtifactState(release_id=base.release_id,
                            system_prompt=evolved_prompt or base.system_prompt,
                            registry=base.registry, task_demos=base.task_demos),
        "A2": ArtifactState(release_id=base.release_id,
                            system_prompt=base.system_prompt,
                            registry=evolved_registry or base.registry,
                            task_demos=base.task_demos),
        "A3": ArtifactState(release_id=base.release_id,
                            system_prompt=base.system_prompt,
                            registry=body_registry or base.registry,
                            task_demos=base.task_demos),
        "A4": ArtifactState(release_id=base.release_id,
                            system_prompt=evolved_prompt or base.system_prompt,
                            registry=combined_registry or evolved_registry
                            or body_registry or base.registry,
                            task_demos=base.task_demos),
    }


def build_factorial_conditions(base: ArtifactState,
                               evolved_prompt: Optional[str] = None,
                               desc_registry: Any = None,
                               body_registry: Any = None,
                               both_registry: Any = None) -> Dict[str, ArtifactState]:
    """组合出 2^3 = 8 个条件；缺失的那一维退化为 v0（报告里会标注）。"""
    out: Dict[str, ArtifactState] = {}
    for code in factorial_codes():
        lv = decode(code)
        reg = base.registry
        if lv["D"] and lv["B"]:
            reg = both_registry or desc_registry or body_registry or base.registry
        elif lv["D"]:
            reg = desc_registry or base.registry
        elif lv["B"]:
            reg = body_registry or base.registry
        out[code] = ArtifactState(
            release_id=base.release_id,
            system_prompt=(evolved_prompt if (lv["P"] and evolved_prompt) else base.system_prompt),
            registry=reg,
            task_demos=base.task_demos,
        )
    return out


def run_ablation(cfg: Dict[str, Any], conditions: Dict[str, ArtifactState],
                 datasets: Sequence[str], splits: Sequence[str] = ("dev_optimize", "selection_gate"),
                 seeds: Sequence[int] = (0,), limit: Optional[int] = None,
                 metric: str = "accuracy", B: int = 10000, rng_seed: int = 20260924,
                 available: Optional[Dict[str, bool]] = None,
                 factors_available: Optional[Dict[str, bool]] = None,
                 execution_plan_sha256: str | None = None) -> Dict[str, Any]:
    if not conditions:
        raise ValueError("ablation requires at least one condition")
    if not datasets or any(not isinstance(ds, str) or not ds for ds in datasets):
        raise ValueError("ablation requires non-empty dataset names")
    if not splits:
        raise ValueError("ablation requires at least one split")
    seed_values = list(seeds)
    if not seed_values:
        raise ValueError("ablation requires at least one evaluation seed")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seed_values):
        raise ValueError("ablation seeds must be integers")
    if len(seed_values) != len(set(seed_values)):
        raise ValueError("ablation seeds must be unique")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("ablation task limit must be a positive integer")
    if isinstance(B, bool) or not isinstance(B, int) or B < 1:
        raise ValueError("ablation bootstrap B must be a positive integer")

    _require_live_execution_guard(cfg, execution_plan_sha256)

    cfg_fp = sha256_obj(cfg)
    codes = list(conditions.keys())
    is_factorial = (set(codes) == set(factorial_codes()))
    # G03's minimum experiment is a fixed-candidate A0/A1/A2 comparison. The
    # evaluator historically only translated the legacy labels through
    # OFAT_MAP (A1 -> F100, etc.), which silently produced an empty comparison
    # when callers supplied the documented A0/A1/A2 states directly. Keep the
    # full-factorial path unchanged, while making the direct OFAT contract a
    # first-class, fail-closed input shape.
    uses_legacy_ofat_labels = any(str(code).startswith("A") for code in codes)
    if uses_legacy_ofat_labels:
        allowed_ofat = set(CONDITIONS)
        if not set(codes).issubset(allowed_ofat):
            raise ValueError("OFAT conditions must use only A0-A4 labels")
        if "A0" not in codes:
            raise ValueError("OFAT comparison requires the explicit A0 baseline condition")
    available = available or {c: True for c in codes}
    results: Dict[str, Any] = {}
    comparisons: Dict[str, Any] = {}
    factorial: Dict[str, Any] = {}

    for ds in datasets:
        results[ds] = {}
        comparisons[ds] = {}
        factorial[ds] = {}
        for split in splits:
            write_per_task = split == "dev_optimize"
            per_cond: Dict[str, Dict[str, Any]] = {}
            expected_ids_by_seed: Dict[int, set[str]] = {}
            expected_count_by_seed: Dict[int, int] = {}
            expected_cohort_sha_by_seed: Dict[int, str] = {}
            for cond in codes:
                runs: List[Dict[str, Any]] = []
                for s in seed_values:
                    res = evaluate(cfg, conditions[cond], ds, split, seed=int(s), limit=limit,
                                   run_tag=f"ablation-{cond}", write_per_task=write_per_task,
                                   cfg_fp=cfg_fp)
                    count = int(res.n_tasks)
                    if count < 1:
                        raise ValueError(
                            f"ablation returned no tasks for {ds}/{split}/{cond}/seed={s}")
                    cohort_sha = str(getattr(res, "task_cohort_sha256", ""))
                    if not cohort_sha:
                        raise ValueError(
                            f"ablation missing task-cohort fingerprint for "
                            f"{ds}/{split}/{cond}/seed={s}")
                    if s in expected_count_by_seed and count != expected_count_by_seed[s]:
                        raise ValueError(
                            f"ablation task-count mismatch for {ds}/{split}/seed={s}: "
                            f"{cond} has {count}, expected {expected_count_by_seed[s]}")
                    expected_count_by_seed.setdefault(s, count)
                    if write_per_task:
                        if len(res.per_task) != count:
                            raise ValueError(
                                f"ablation per-task row count mismatch for "
                                f"{ds}/{split}/{cond}/seed={s}: "
                                f"{len(res.per_task)} rows for {count} tasks")
                        task_ids = [str(row.get("task_id", "")) for row in res.per_task]
                        if any(not task_id for task_id in task_ids):
                            raise ValueError(
                                f"ablation missing task_id for {ds}/{split}/{cond}/seed={s}")
                        task_set = set(task_ids)
                        if len(task_set) != len(task_ids):
                            raise ValueError(
                                f"ablation duplicate task_id for {ds}/{split}/{cond}/seed={s}")
                        expected = expected_ids_by_seed.get(s)
                        if expected is not None and task_set != expected:
                            missing = sorted(expected - task_set)
                            extra = sorted(task_set - expected)
                            raise ValueError(
                                f"ablation paired cohort mismatch for {ds}/{split}/seed={s}/"
                                f"{cond}: missing={missing[:3]} ({len(missing)} total), "
                                f"extra={extra[:3]} ({len(extra)} total)")
                        expected_ids_by_seed.setdefault(s, task_set)
                    expected_sha = expected_cohort_sha_by_seed.get(s)
                    if expected_sha is not None and cohort_sha != expected_sha:
                        raise ValueError(
                            f"ablation task-cohort fingerprint mismatch for "
                            f"{ds}/{split}/seed={s}/{cond}")
                    expected_cohort_sha_by_seed.setdefault(s, cohort_sha)
                    runs.append({
                        "seed": int(s),
                        "run_id": res.run_id,
                        "metric": float(res.metric_values.get(metric, 0.0)),
                        "metrics": res.metric_values,
                        "correct": ({r["task_id"]: int(bool(r["correct"])) for r in res.per_task}
                                    if write_per_task else {}),
                    })
                per_cond[cond] = {
                    "available": bool(available.get(cond, True)),
                    "label": (condition_label(cond) if is_factorial
                              else CONDITION_LABELS.get(cond, cond)),
                    "factors": decode(cond) if is_factorial else {},
                    "per_seed": [{k: r[k] for k in ("seed", "run_id", "metric")} for r in runs],
                    "summary": summary([float(r["metric"]) for r in runs]),
                    # 次级指标：主指标看不见的收益（例如 abstain 换来的"少编答案"）都在这里
                    "secondary": {k: (round(sum(float(r["metrics"][k]) for r in runs) / len(runs), 6)
                                      if all(r["metrics"].get(k) is not None for r in runs) else None)
                                  for k in SECONDARY_METRICS},
                }
                if write_per_task:
                    per_cond[cond]["_correct_by_seed"] = {
                        str(r["seed"]): r["correct"] for r in runs
                    }
            results[ds][split] = per_cond

            # 与 A0 比较。固定候选的 SPEC M2/G03 最小设计传 A0/A1/A2；
            # 全因子仍通过 OFAT_MAP 使用 F000/F100/F010/F001/F111。
            comp: Dict[str, Any] = {}
            if uses_legacy_ofat_labels:
                contrast_specs = [(name, name) for name in CONDITIONS[1:]
                                  if name in per_cond]
                base_code = "A0"
            else:
                contrast_specs = [(name, code) for name, code in OFAT_MAP.items()
                                  if name != "A0" and code in per_cond]
                base_code = OFAT_MAP["A0"]
            for name, code in contrast_specs:
                if not available.get(code, available.get(name, True)):
                    comp[f"{name}_vs_A0"] = {"available": False, "condition": code,
                                             "note": "本轮没有产出该维度的候选，条件退化为 A0"}
                    continue
                if code == base_code:
                    continue
                if write_per_task:
                    cis = []
                    for s in seed_values:
                        b = per_cond[base_code]["_correct_by_seed"][str(s)]
                        c = per_cond[code]["_correct_by_seed"][str(s)]
                        # Earlier validation requires exact identity equality;
                        # retain the explicit check here so the estimator is
                        # visibly bound to the shared task sample.
                        if set(b) != set(c):
                            raise ValueError(
                                f"paired comparison task IDs differ for {code}/seed={s}")
                        ids = sorted(b)
                        cis.append({"seed": int(s),
                                    **paired_bootstrap([float(b[i]) for i in ids],
                                                       [float(c[i]) for i in ids],
                                                       B=B, rng_seed=rng_seed)})
                    comp[f"{name}_vs_A0"] = {
                        "available": True,
                        "condition": code,
                        "pairing": "per_task",
                        "per_seed": cis,
                        "mean_delta_pp": round(sum(c["delta_pp"] for c in cis) / len(cis), 4),
                        "min_p_delta_gt_0": round(min(c["p_delta_gt_0"] for c in cis), 4),
                    }
                else:
                    deltas = [
                        round((per_cond[code]["per_seed"][i]["metric"]
                               - per_cond[base_code]["per_seed"][i]["metric"]) * 100.0, 4)
                        for i in range(len(seeds))
                    ]
                    comp[f"{name}_vs_A0"] = {
                        "available": True,
                        "condition": code,
                        "pairing": "per_seed_aggregate",
                        "delta_pp_per_seed": deltas,
                        "mean_delta_pp": round(sum(deltas) / len(deltas), 4),
                        "note": "gate 只回传聚合，不做逐题配对，不给出 CI",
                    }
            comparisons[ds][split] = comp

            # 全因子分析：主效应 / 交互 / 残差 / Shapley
            if is_factorial:
                factorial[ds][split] = _factorial_analysis(per_cond, seed_values, write_per_task,
                                                           B=B, rng_seed=rng_seed, metric=metric)

    # 临时字段不进报告（全因子分析已在此之前算完）
    for ds in results:
        for split in results[ds]:
            for cond in results[ds][split]:
                results[ds][split][cond].pop("_correct_by_seed", None)

    # 行为通道画像：单维条件之间是否争夺同一批开关
    profiles = {c: behavior_profile(conditions[c], cfg) for c in conditions}
    singletons = {c: profiles[c] for c in SINGLETON_CODES if c in profiles}
    overlap = overlap_matrix(singletons)
    # 基线（全关）自带哪个越界不算候选的锅 —— 只报相对基线**新增**的越界
    base_viol = {(v.get("behavior"), v.get("appears_in"))
                 for v in (profiles.get("F000", {}).get("violations") or [])}

    def _prof(c: str) -> Dict[str, Any]:
        p = profiles[c]
        new_viol = [v for v in (p.get("violations") or [])
                    if (v.get("behavior"), v.get("appears_in")) not in base_viol]
        return {"effective": p.get("effective") or [], "by_channel": p.get("by_channel") or {},
                "violations": new_viol,
                "baseline_violations_inherited": sorted(
                    f"{b}@{a}" for b, a in base_viol)}

    overlap["per_condition"] = {c: _prof(c) for c in singletons}
    overlap["per_condition_all"] = {c: _prof(c) for c in sorted(profiles)}
    overlap["channel_config"] = {
        "enabled": bool(load_channels(cfg).get("enabled", True)),
        "owner": load_channels(cfg).get("owner", {}),
    }

    report: Dict[str, Any] = {
        "created_at": now_iso(),
        "execution_plan_sha256": execution_plan_sha256,
        "metric": metric,
        "design": ("factorial_2^3" if is_factorial else "ofat"),
        "datasets": list(datasets),
        "splits": list(splits),
        "seeds": [int(s) for s in seeds],
        # 口径快照：把"这次跑用的是什么噪声/什么通道"写进报告，
        # 避免把不同噪声下跑出来的两张表并排放（M3 报告踩过这个坑）
        "runtime_snapshot": {
            "provider": str(cfg.get("runtime", {}).get("provider", "fake")),
            "fake_noise": float(cfg.get("runtime", {}).get("fake_noise", 0.0)),
            "top_k": cfg.get("retrieval", {}).get("top_k"),
            "recall_min_score": cfg.get("skills", {}).get("recall_min_score"),
            "bootstrap_B": int(B),
            "rng_seed": int(rng_seed),
        },
        "conditions": {c: {"label": (condition_label(c) if is_factorial
                                     else CONDITION_LABELS.get(c, c)),
                           "factors": decode(c) if is_factorial else {},
                           "available": bool(available.get(c, True))} for c in codes},
        "factors": {f: {"artifact": FACTOR_ARTIFACTS[f], "label": FACTOR_LABELS[f]}
                    for f in FACTORS} if is_factorial else {},
        "definitions": _definitions() if is_factorial else {},
        "results": results,
        "comparisons": comparisons,
        "factorial": factorial,
        "behavior_overlap": overlap,
        "factors_available": dict(factors_available) if factors_available else None,
    }
    report["interpretation"] = _interpret(report, is_factorial)
    return report


def _definitions() -> Dict[str, Any]:
    """效应口径写进报告，避免读者把两种口径混着看。"""
    return {
        "effect_pp": "标准 2^k 因子效应（对比度 / 2^{k-1}）；主效应 = 在其它因子各水平上"
                     "取平均的简单效应",
        "did_pp": "差分的差分（直观口径）：二阶为 effect_pp×2、三阶为 ×4；"
                  "例如 P×D = 「D 开启时 P 的效应」−「D 关闭时 P 的效应」",
        "residual_interaction_pp": "实测全开 − 加性叠加（A0 + 各单维增量之和）；"
                                   ">0 才谈得上协同，≈0 说明只是叠加",
        "shapley": "三因子 Shapley 值（效率性：Σφ = 全开 − 全关）；"
                   "它用的 2^3 个子集价值函数正是这 8 个条件，不额外花实验",
        "ci": "dev_optimize 上逐题配对 bootstrap（百分位，B 见 runtime_snapshot）；"
              "把效应写成题级线性组合后重采样，因此交互项也有 CI",
    }


def factorial_self_test() -> Dict[str, Any]:
    """全因子数学自检（进 verify --strict）：口径错了后面全错，必须开机自检。"""
    checks: List[Dict[str, Any]] = []

    def _ck(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    for term in _terms():
        w = effect_weights(term)
        _ck(f"weights_sum_zero_{'×'.join(term)}", abs(sum(w.values())) < 1e-9,
            f"Σw={sum(w.values()):.2e}")

    # 纯加性：残差必须为 0
    y = {c: 0.0 for c in factorial_codes()}
    y.update({"F000": 0.40, "F100": 0.50, "F010": 0.45, "F001": 0.55,
              "F110": 0.55, "F101": 0.65, "F011": 0.60, "F111": 0.70})
    _ck("additive_residual_zero", abs(additive_prediction(y)["residual_interaction"]) < 1e-9,
        f"residual={additive_prediction(y)['residual_interaction']:.2e}")

    # Shapley 效率性
    y2 = dict(y)
    y2.update({"F110": 0.62, "F101": 0.80, "F011": 0.55, "F111": 0.90})
    sh = shapley_values(y2)
    _ck("shapley_efficiency",
        abs(sum(sh.values()) - (y2["F111"] - y2["F000"])) < 1e-9,
        f"Σφ={sum(sh.values()):.6f} vs v(N)-v(∅)={y2['F111'] - y2['F000']:.6f}")

    # 主效应 = 平均简单效应
    y3 = {c: 0.0 for c in factorial_codes()}
    y3.update({"F000": 0.4, "F001": 0.5, "F010": 0.6, "F011": 0.7,
               "F100": 0.6, "F101": 0.7, "F110": 0.8, "F111": 0.9})
    eff = factorial_effects(y3)
    simple = ((y3["F100"] - y3["F000"]) + (y3["F110"] - y3["F010"])
              + (y3["F101"] - y3["F001"]) + (y3["F111"] - y3["F011"])) / 4.0
    _ck("main_effect_equals_mean_simple_effect",
        abs(eff["P"]["effect"] - simple) < 1e-9, f"{eff['P']['effect']:.6f} vs {simple:.6f}")

    _ck("ofat_map_consistent",
        OFAT_MAP["A0"] == "F000" and OFAT_MAP["A4"] == "F111",
        "OFAT 视图与全因子条件一一对应")
    return {"ok": all(c["passed"] for c in checks), "checks": checks}


def _interpret(report: Dict[str, Any], is_factorial: bool) -> Dict[str, Any]:
    """给报告一个**结论口径**，避免再出现"A4 ≤ max ⇒ 协同无效"这种越界解读。"""
    overlap = report.get("behavior_overlap") or {}
    verdict = str(overlap.get("verdict") or "unknown")
    out: Dict[str, Any] = {
        "behavior_overlap_verdict": verdict,
        "note": overlap.get("verdict_note", ""),
    }
    if verdict == "not_detectable_by_design":
        out["synergy_claim"] = "不可判定"
        out["reason"] = ("单维条件之间存在共享行为开关：全开条件在行为层面退化为单维条件，"
                         "本设计无法区分'叠加'与'协同'。应表述为'本设计检不出协同'，"
                         "而不是'实证发现互不放大'。")
    elif not is_factorial:
        out["synergy_claim"] = "不可判定"
        out["reason"] = "OFAT（一次一因子）设计在数学上估不出交互效应（Fisher 1935）。"
    else:
        missing = [f for f, ok in (report.get("factors_available") or {}).items() if not ok]
        if missing:
            # 因子没产出可用候选 → 涉及它的格子退化为等价条件，交互恒为 0。
            # 这时"未检出交互"是**没做实验**，不是"做了实验没检出"。
            out["synergy_claim"] = "不可判定（因子缺失）"
            out["missing_factors"] = missing
            out["reason"] = (f"因子 {missing} 本轮没有产出可用候选，涉及它的条件退化为等价"
                             "条件，交互恒为 0 —— 这是'没做成实验'，不能写成'实测无交互'。")
            return out
        hits = []
        for ds, per_split in (report.get("factorial") or {}).items():
            for split, sec in per_split.items():
                agg = sec.get("aggregate") or {}
                for name, e in (agg.get("interactions") or {}).items():
                    if abs(float(e.get("mean_pp", 0.0))) >= 1.0:
                        hits.append({"dataset": ds, "split": split, "term": name,
                                     "mean_pp": e.get("mean_pp")})
                r = (agg.get("additivity") or {})
                if abs(float(r.get("mean_residual_interaction_pp", 0.0))) >= 1.0:
                    hits.append({"dataset": ds, "split": split, "term": "residual",
                                 "mean_pp": r.get("mean_residual_interaction_pp")})
        out["synergy_claim"] = "检出交互" if hits else "未检出交互（可判定）"
        out["evidence"] = hits
        if hits:
            out["reason"] = "存在 |效应| ≥ 1pp 的交互项或加性残差。"
        else:
            out["reason"] = ("维度间行为开关互斥，交互可被估计；实测未出现 |效应| ≥ 1pp 的"
                             "交互项或加性残差 —— 这是**可判定的否定结论**。")
    return out


def _factorial_analysis(per_cond: Dict[str, Any], seeds: Sequence[int], write_per_task: bool,
                        B: int = 10000, rng_seed: int = 20260924,
                        metric: str = "accuracy") -> Dict[str, Any]:
    """由 8 个条件的逐 seed 结果算效应、Shapley、加性残差与（可给时的）CI。"""
    codes = factorial_codes()
    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        y = {c: float(per_cond[c]["per_seed"][seeds.index(s)]["metric"]) for c in codes}
        eff = factorial_effects(y)
        add = additive_prediction(y)
        sh = shapley_values(y)
        row: Dict[str, Any] = {
            "seed": int(s),
            "conditions": {c: round(y[c], 6) for c in codes},
            "main_effects": {k: v for k, v in eff.items() if v["order"] == 1},
            "interactions": {k: v for k, v in eff.items() if v["order"] == 2},
            "interaction_3way": {k: v for k, v in eff.items() if v["order"] == 3},
            "additivity": add,
            "shapley": sh,
            "shapley_sum_check": round(sum(sh.values())
                                       - (y["F111"] - y["F000"]), 6),
        }
        per_seed.append(row)

    def _agg(key: str, field: str = "effect_pp") -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for row in per_seed:
            for name, e in (row.get(key) or {}).items():
                out.setdefault(name, []).append(float(e[field]))
        return {n: {"mean_pp": round(sum(v) / len(v), 4), "min_pp": round(min(v), 4),
                    "max_pp": round(max(v), 4)} for n, v in out.items()}

    aggregate: Dict[str, Any] = {
        "main_effects": _agg("main_effects"),
        "interactions": _agg("interactions"),
        "interaction_3way": _agg("interaction_3way"),
        "shapley": {
            f: {"mean_pp": round(sum(r["shapley"][f] for r in per_seed) / len(per_seed) * 100.0, 4),
                "min_pp": round(min(r["shapley"][f] for r in per_seed) * 100.0, 4),
                "max_pp": round(max(r["shapley"][f] for r in per_seed) * 100.0, 4)}
            for f in FACTORS},
        "additivity": {
            "mean_residual_interaction_pp": round(
                sum(r["additivity"]["residual_interaction_pp"] for r in per_seed) / len(per_seed), 4),
            "min_residual_interaction_pp": round(
                min(r["additivity"]["residual_interaction_pp"] for r in per_seed), 4),
            "max_residual_interaction_pp": round(
                max(r["additivity"]["residual_interaction_pp"] for r in per_seed), 4),
            "mean_additive_prediction": round(
                sum(r["additivity"]["additive_prediction"] for r in per_seed) / len(per_seed), 6),
            "mean_observed_all_on": round(
                sum(r["additivity"]["observed_all_on"] for r in per_seed) / len(per_seed), 6),
        },
    }

    ci: Dict[str, Any] = {}
    if write_per_task:
        ci = _contrast_cis(per_cond, seeds, B=B, rng_seed=rng_seed)

    return {
        "metric": metric,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "ci": ci,
        "factor_purity": _factor_purity(per_cond),
        "ci_note": ("题级配对 bootstrap；交互项把效应写成题级线性组合后重采样"
                    if write_per_task else
                    "该 split 只回传聚合指标，不给 CI"),
    }


def _factor_purity(per_cond: Dict[str, Any],
                   tol_pp: float = 0.5) -> Dict[str, Any]:
    """因子纯度：单维条件是否**只**动了自己那条通路。

    已知的口径污染：B 因子含"结晶的新 Skill"，而新 Skill 自带 description ——
    它会提升召回率，也就是顺带踩进了 D 的通路。这时 B 的主效应并非纯程序效应，
    报告里必须写出来，否则会把"召回提升"误读成"程序改进"。
    """
    def _recall(code: str) -> Optional[float]:
        v = (per_cond.get(code) or {}).get("secondary") or {}
        return float(v["skill_recall_rate"]) if v.get("skill_recall_rate") is not None else None

    base = _recall("F000")
    out: Dict[str, Any] = {"note": ("若某单维条件同时改变了召回率，说明它踩进了 description "
                                    "的通路，主效应不是'纯'的该维度效应"),
                           "tolerance_pp": tol_pp, "per_factor": {}}
    if base is None:
        return out
    for f, code in (("P", "F100"), ("D", "F010"), ("B", "F001")):
        r = _recall(code)
        if r is None:
            continue
        delta = round((r - base) * 100.0, 4)
        pure = (abs(delta) <= tol_pp) if f != "D" else True
        out["per_factor"][f] = {
            "condition": code,
            "skill_recall_rate": round(r, 6),
            "recall_delta_pp": delta,
            "pure": bool(pure),
            "note": ("仅动答题策略，未改召回 → 纯" if f == "P" and pure else
                     "改动召回正是本维度的作用通路 → 纯" if f == "D" else
                     "该维度改变了召回率：new_skill 自带 description，"
                     "主效应里混有 description 通道的贡献 → 非纯" if not pure else
                     "未改召回 → 纯"),
        }
    return out


def _contrast_cis(per_cond: Dict[str, Any], seeds: Sequence[int],
                  B: int = 10000, rng_seed: int = 20260924) -> Dict[str, Any]:
    """对每个效应项做逐题 bootstrap：把效应写成题级线性组合后重采样。"""
    out: Dict[str, Any] = {"main_effects": {}, "interactions": {},
                           "interaction_3way": {}, "shapley": {}, "residual": {}}
    sh_w = shapley_weights()
    for s in seeds:
        corr = {c: per_cond[c]["_correct_by_seed"][str(int(s))] for c in per_cond}
        task_sets = {code: set(rows) for code, rows in corr.items()}
        reference = task_sets[next(iter(corr))]
        mismatched = {code: (len(reference - ids), len(ids - reference))
                      for code, ids in task_sets.items() if ids != reference}
        if mismatched:
            raise ValueError(
                f"factorial CI requires identical task IDs for seed={s}; "
                f"mismatched conditions (missing, extra)={mismatched}")
        ids = sorted(reference)
        for term in _terms():
            w = effect_weights(term)
            name = "×".join(term)
            vals = [sum(w[c] * float(corr[c][i]) for c in w) for i in ids]
            key = ("main_effects" if len(term) == 1 else
                   "interactions" if len(term) == 2 else "interaction_3way")
            ci = paired_bootstrap([0.0] * len(vals), vals, B=B, rng_seed=rng_seed)
            out[key].setdefault(name, []).append({"seed": int(s), **ci})
        for f in FACTORS:
            w = sh_w[f]
            vals = [sum(w[c] * float(corr[c][i]) for c in w) for i in ids]
            out["shapley"].setdefault(f, []).append(
                {"seed": int(s), **paired_bootstrap([0.0] * len(vals), vals, B=B,
                                                    rng_seed=rng_seed)})
        # 加性残差：全开 − (F100+F010+F001−2·F000)
        w = {"F111": 1.0, "F100": -1.0, "F010": -1.0, "F001": -1.0, "F000": 2.0}
        vals = [sum(w[c] * float(corr[c][i]) for c in w) for i in ids]
        out["residual"].setdefault("residual_interaction", []).append(
            {"seed": int(s), **paired_bootstrap([0.0] * len(vals), vals, B=B, rng_seed=rng_seed)})

    # 多 seed 只保留逐 seed 结果 + 汇总（不做跨 seed 合并，避免假装独立）
    for key in out:
        for name, rows in out[key].items():
            out[key][name] = {
                "per_seed": rows,
                "mean_delta_pp": round(sum(r["delta_pp"] for r in rows) / len(rows), 4),
                "min_p_delta_gt_0": round(min(r["p_delta_gt_0"] for r in rows), 4),
                "max_p_delta_gt_0": round(max(r["p_delta_gt_0"] for r in rows), 4),
            }
    return out


def save(report: Dict[str, Any]) -> str:
    rid = sha256_obj({k: report.get(k) for k in ("created_at", "datasets", "seeds")})[:12]
    p = ABLATION_ROOT / f"ablation-{rid}.json"
    write_json(p, report)
    return p.as_posix()


def states_from_round(cfg: Dict[str, Any], datasets: Sequence[str], seed: int = 0,
                      limit: Optional[int] = None) -> Dict[str, Any]:
    """跑一轮进化（**不发布**），把候选拆成 P / D / B 三个因子，组合出 2^3 个条件。

    返回 {states, available, round_id, candidates_summary}
    """
    from ..evolution.apply import combine, registry_with
    from ..evolution.round import run_round

    base = rel.load_state()
    report = run_round(cfg, datasets=list(datasets), seed=seed, limit=limit)
    evolved_prompt: Optional[str] = None
    desc_registry = None
    body_registry = None
    body_cands: List[Any] = []
    desc_cands: List[Any] = []
    for c in report.get("candidates", []):
        cid = c.get("candidate_id")
        if not cid:
            continue
        cand = _load_candidate(cid)
        if cand is None:
            continue
        if cand.artifact_type == "system_prompt" and evolved_prompt is None:
            evolved_prompt = cand.new_content
        elif cand.artifact_type == "skill_description":
            desc_cands.append(cand)
        elif cand.artifact_type in ("skill_body", "new_skill"):
            body_cands.append(cand)

    # description 维度：可能有多个候选（每个数据集/每个归因一个类），逐个叠加
    if desc_cands:
        reg = base.registry
        for c in desc_cands:
            reg = registry_with(reg, c)
        desc_registry = reg
    if body_cands:
        reg = base.registry
        for c in body_cands:
            reg = registry_with(reg, c)
        body_registry = reg

    combined = None
    if desc_cands or body_cands:
        # D 与 B 同时开的那一格：两个维度都作用到 registry，按 desc → body 顺序叠加
        combined = combine(base, list(desc_cands) + list(body_cands)).registry

    # 因子可用性：某一维本轮没产出候选 → 涉及它的条件全部退化为等价条件，
    # 效应不可估（报告里会标 available=False，interpretation 也会降级）
    has_p = evolved_prompt is not None
    has_d = desc_registry is not None
    has_b = body_registry is not None
    available = {code: all((not lv["P"] or has_p) and (not lv["D"] or has_d)
                           and (not lv["B"] or has_b)
                           for lv in [decode(code)])
                 for code in factorial_codes()}
    return {
        "states": build_factorial_conditions(base, evolved_prompt, desc_registry,
                                             body_registry, combined),
        "available": available,
        "factors_available": {"P": has_p, "D": has_d, "B": has_b},
        "round_id": report.get("round_id"),
        "base_release": base.release_id,
        "candidates_summary": {c: len([x for x in report.get("candidates", [])
                                       if x.get("target_artifact") == c])
                               for c in ("system_prompt", "skill_description",
                                         "skill_body", "new_skill")},
    }


def _load_candidate(candidate_id: str):
    from ..contracts.types import Candidate
    from ..util import read_json

    p = artifacts_path("candidates", f"{candidate_id}.json")
    if not p.exists():
        return None
    return Candidate(**read_json(p))
