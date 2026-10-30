"""统计：配对 bootstrap 置信区间（M2 / Spec §4-M2 第 3 条）。

为什么用**配对**（paired）而不是两组独立比较：
    候选与基线跑的是**同一批题目、同一顺序**，逐题差值把"题目难度"这个
    巨大的方差来源消掉了。同样样本量下，配对检验的检验力高得多。

为什么用 bootstrap 而不是 t 检验：
    单次正确率是 0/1，分布远非正态；bootstrap 不依赖分布假设，
    且只需纯标准库即可确定性重放（固定 rng seed）。

边界（别在报告里夸大）：
    M2 阶段跑的是 FakeProvider + 受控题级噪声，因此这里的 CI
    证明的是**统计链路可用**，不是真实模型的方差估计。
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_B = 10000
DEFAULT_RNG_SEED = 20260924


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: Sequence[float], ddof: int = 1) -> float:
    n = len(values)
    if n <= ddof:
        return 0.0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (n - ddof)) ** 0.5


def paired_diffs(before: Sequence[float], after: Sequence[float]) -> List[float]:
    """逐题差值（after - before）。长度不等直接报错——配对必须严格对齐。"""
    if len(before) != len(after):
        raise ValueError(f"配对长度不一致: {len(before)} vs {len(after)}")
    pairs = [(float(b), float(a)) for b, a in zip(before, after)]
    if any(not math.isfinite(b) or not math.isfinite(a) for b, a in pairs):
        raise ValueError("paired measurements must be finite")
    diffs = [a - b for b, a in pairs]
    if any(not math.isfinite(value) for value in diffs):
        raise ValueError("paired differences must be finite")
    return diffs


def _binary_pair(before: Sequence[float], after: Sequence[float]) -> tuple[list[float], list[float]]:
    if len(before) != len(after):
        raise ValueError(f"paired lengths differ: {len(before)} vs {len(after)}")
    left, right = [float(value) for value in before], [float(value) for value in after]
    if any(value not in (0.0, 1.0) for value in (*left, *right)):
        raise ValueError("paired binary measurements must contain only 0 or 1")
    return left, right


def paired_bootstrap(before: Sequence[float], after: Sequence[float],
                     B: int = DEFAULT_B, rng_seed: int = DEFAULT_RNG_SEED,
                     alpha: float = 0.05) -> Dict[str, Any]:
    """对逐题差值做非参数 bootstrap，返回百分点差值及其置信区间。

    返回：
        delta_pp        点估计（百分点）
        ci_low/ci_high  百分位 bootstrap 置信区间（百分点）
        p_delta_gt_0    P(Δ>0)：bootstrap 样本中 Δ>0 的比例（单侧证据强度）
    """
    if type(B) is not int or B < 1:
        raise ValueError("bootstrap sample count B must be a positive integer")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0 < alpha < 1:
        raise ValueError("bootstrap alpha must be between 0 and 1")
    diffs = paired_diffs(before, after)
    n = len(diffs)
    if n == 0:
        return {"n": 0, "delta_pp": 0.0, "ci_low_pp": 0.0, "ci_high_pp": 0.0,
                "p_delta_gt_0": 0.0, "B": B, "rng_seed": rng_seed}

    rng = random.Random(rng_seed)
    point = sum(diffs) / n * 100.0

    samples: List[float] = []
    for _ in range(int(B)):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        samples.append(s / n * 100.0)

    samples.sort()
    lo_idx = max(0, int((alpha / 2) * len(samples)))
    hi_idx = min(len(samples) - 1, int((1 - alpha / 2) * len(samples)) - 1)
    p_gt = sum(1 for s in samples if s > 0) / len(samples)

    return {
        "n": n,
        "delta_pp": round(point, 4),
        "ci_low_pp": round(samples[lo_idx], 4),
        "ci_high_pp": round(samples[hi_idx], 4),
        "p_delta_gt_0": round(p_gt, 4),
        "B": int(B),
        "rng_seed": int(rng_seed),
        "alpha": alpha,
        "method": "paired bootstrap (percentile)",
    }


def mcnemar_exact(before: Sequence[float], after: Sequence[float]) -> Dict[str, Any]:
    """配对 0/1 数据的**精确 McNemar 检验**（双侧），小样本下取代 bootstrap 作主检验。

    为什么要在 bootstrap 之外再给一个精确检验：
        配对 bootstrap 在 n=18 / n=36 且数据是 0/1 时覆盖率很差 ——
        百分位 CI 只能取 1/n 的整数倍（n=18 时 CI 端点必然是 5.56 的倍数），
        且小样本下 bootstrap 分布离散、尾部估计不可靠。
        McNemar 只看**不一致对**（b: 基线对终评错；c: 基线错终评对），
        在 H0 下 b+c 个不一致对应当 Binomial(n, 0.5)，可以精确算 p 值，
        不需要任何分布近似 —— 这正是小样本配对设计的标准做法。

    与其他项目的关系：asta-bench / LitQA2 报的正是这类"翻转计数"，
    本项目过去只报 bootstrap 的 P(Δ>0)，把"证据强度"和"点估计"混在了一起。
    """
    before, after = _binary_pair(before, after)
    b = sum(1 for x, y in zip(before, after) if x == 1.0 and y == 0.0)
    c = sum(1 for x, y in zip(before, after) if x == 0.0 and y == 1.0)
    n = b + c
    if n == 0:
        p = 1.0
    else:
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(k + 1))   # P(X <= min(b,c)), X~Bin(n,.5)
        p = min(1.0, 2.0 * (tail / (1 << n)))                # 双侧；避免大整数转 float 溢出
    return {
        "n_regressed": b,
        "n_improved": c,
        "n_discordant": n,
        "p_value": round(float(p), 6),
        "method": "exact McNemar (two-sided binomial)",
        "note": "只看不一致对；一致对不提供方向信息（McNemar 的前提）",
    }


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> Dict[str, Any]:
    """Holm-Bonferroni 逐步降级校正（比 Bonferroni 强，仍控制 FWER）。

    为什么必须做：终评同时报 3 个数据集（+ 迁移域）× 多个指标。
    不做校正时，"某一个数据集显著"的家族错误率远高于 5% ——
    三个数据集里挑最好看的那个报，是最常见的选择性报告路径。
    """
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0 < alpha < 1:
        raise ValueError("Holm alpha must be between 0 and 1")
    ps = [float(p) for p in pvalues]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in ps):
        raise ValueError("Holm p-values must be finite probabilities")
    m = len(ps)
    if m == 0:
        return {"alpha": alpha, "m": 0, "p_values": [], "adjusted": [], "reject": [],
                "method": "Holm-Bonferroni (step-down)"}
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        a = min(1.0, (m - rank) * ps[i])
        a = max(a, prev)          # 单调化：校正后 p 值不得随排名下降
        prev = a
        adj[i] = round(a, 6)
    return {
        "alpha": alpha,
        "m": m,
        "p_values": [round(x, 6) for x in ps],
        "adjusted": adj,
        "reject": [bool(a < alpha) for a in adj],
        "method": "Holm-Bonferroni (step-down)",
    }


def paired_bootstrap_binary(before: Sequence[int], after: Sequence[int], **kw: Any) -> Dict[str, Any]:
    """0/1 正确率序列的便捷入口（语义同上）。"""
    left, right = _binary_pair(before, after)
    return paired_bootstrap(left, right, **kw)


def summary(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": round(mean(values), 6),
        "std": round(std(values), 6),
        "min": round(min(values), 6) if values else 0.0,
        "max": round(max(values), 6) if values else 0.0,
        "n": len(values),
    }


def meets_threshold(ci: Dict[str, Any], min_delta_pp: float = 1.0,
                    min_prob: float = 0.90) -> Dict[str, Any]:
    """Spec §6.3 增强版第 4 条：Δ ≥ 阈值 或 P(Δ>0) ≥ 0.90。"""
    delta = float(ci.get("delta_pp", 0.0))
    prob = float(ci.get("p_delta_gt_0", 0.0))
    ok = (delta >= min_delta_pp) or (prob >= min_prob)
    return {
        "passed": ok,
        "delta_pp": delta,
        "p_delta_gt_0": prob,
        "rule": f"Δ≥{min_delta_pp}pp 或 P(Δ>0)≥{min_prob}",
        "reason": ("Δ 达阈值" if delta >= min_delta_pp else "")
                  + ("；" if delta >= min_delta_pp and prob >= min_prob else "")
                  + (f"P(Δ>0)={prob} 达阈值" if prob >= min_prob else ""),
    }


def _check(name: str, passed: bool, detail: str = "") -> Dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail}


def self_test() -> Dict[str, Any]:
    """统计模块自检：用已知答案的小样本验证方向性与缩放是否正确。"""
    checks: List[Dict[str, Any]] = []
    # 1) 完全一致 → Δ=0
    ci = paired_bootstrap([1, 0, 1, 0, 1], [1, 0, 1, 0, 1], B=2000)
    checks.append(_check("identical_sequences_zero_delta",
                         abs(ci["delta_pp"]) < 1e-9, f"delta={ci['delta_pp']}"))
    # 2) 全部翻转 → Δ=100pp，CI 也应为 100
    ci = paired_bootstrap([0, 0, 0, 0, 0], [1, 1, 1, 1, 1], B=2000)
    checks.append(_check("full_flip_100pp",
                         abs(ci["delta_pp"] - 100.0) < 1e-9 and ci["ci_low_pp"] == 100.0,
                         f"delta={ci['delta_pp']} ci={ci['ci_low_pp']}"))
    # 3) 小幅提升 → P(Δ>0) 应 > 0.5
    before = [0] * 50 + [1] * 50
    after = before[:]
    for i in range(0, 20):
        after[i] = 1
    ci = paired_bootstrap(before, after, B=2000)
    checks.append(_check("positive_shift_p_gt_half",
                         ci["p_delta_gt_0"] > 0.5, f"p={ci['p_delta_gt_0']}"))
    # 4) 确定性：同参数两次结果一致
    a = paired_bootstrap(before, after, B=1000, rng_seed=7)
    b = paired_bootstrap(before, after, B=1000, rng_seed=7)
    checks.append(_check("deterministic_given_rng_seed", a == b, "same rng_seed → same result"))
    # 5) 精确 McNemar：完全一致 → 无不一致对 → p=1
    mc = mcnemar_exact([1, 0, 1, 0], [1, 0, 1, 0])
    checks.append(_check("mcnemar_no_discordant_p_one",
                         mc["n_discordant"] == 0 and mc["p_value"] == 1.0,
                         f"n={mc['n_discordant']} p={mc['p_value']}"))
    # 6) 精确 McNemar：5 题全部由错翻对 → b=0,c=5 → p = 2*C(5,0)/2^5 = 0.0625
    mc = mcnemar_exact([0] * 5, [1] * 5)
    checks.append(_check("mcnemar_full_flip_p_0625",
                         mc["n_improved"] == 5 and abs(mc["p_value"] - 0.0625) < 1e-9,
                         f"c={mc['n_improved']} p={mc['p_value']}"))
    # 7) 4 进 1 出（n=5, min=1）→ p = 2*(C(5,0)+C(5,1))/32 = 2*6/32 = 0.375
    mc = mcnemar_exact([0, 0, 0, 0, 1], [1, 1, 1, 1, 0])
    checks.append(_check("mcnemar_4up_1down_p_0375",
                         abs(mc["p_value"] - 0.375) < 1e-9, f"p={mc['p_value']}"))
    # 8) Holm 单调化：p=(0.01,0.02,0.03) → 逐步降级 3p/2p/1p = (0.03,0.04,0.03)
    #    再单调化取 running max → (0.03, 0.04, 0.04)。三者均 < 0.05 → 全部拒绝原假设。
    h = holm_bonferroni([0.01, 0.02, 0.03])
    checks.append(_check("holm_monotone_and_stepdown",
                         h["adjusted"] == [0.03, 0.04, 0.04] and all(h["reject"]),
                         f"adj={h['adjusted']}"))
    # 9) Holm 比 Bonferroni 不更保守（m=3 时首个应为 3*p 而非更宽松）
    h2 = holm_bonferroni([0.04, 0.9, 0.9])
    checks.append(_check("holm_not_weaker_than_bonferroni",
                         abs(h2["adjusted"][0] - 0.12) < 1e-9 and not h2["reject"][0],
                         f"adj={h2['adjusted']} reject={h2['reject']}"))
    return {"ok": all(c["passed"] for c in checks), "checks": checks}
