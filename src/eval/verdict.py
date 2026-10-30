"""决策 2：主结论口径 —— **三指标并列**，不承认单一 accuracy。

## 为什么改口径（项目内的硬事实）
新协议下 suppqa 基线 accuracy = 0.024、弃权率 0.890。基线几乎贴着地板，
任何"敢作答"的候选都会显得大涨（当前 ACTIVE +13.4pp 就是这么来的）。
accuracy 在这个位置是**低杆效应放大器**，不能单独当结论。

## 外部依据
- **asta-bench LitQA2**（`evals/labbench/litqa2/task.py:115-146`）：与本项目同构
  （同样注入 "Insufficient information"），它报三指标 ——
  `accuracy`（弃权记错）、`precision`（**sure 答案中的正确率**）、`coverage`（选 sure 的比例）。
- **squad_v2**（HuggingFace evaluate）：13 个值，核心是**分组报告** HasAns / NoAns，
  避免"全答 NoAns"刷分。
- **AbstentionBench**（facebookresearch, 2025）：主指标是 **Abstention F1**，
  把"该答答对"与"该弃权弃权"合成一个分，用于单一排序。

## 本模块的口径
1. `accuracy` 保留，但**降为参考项**；基线贴地时自动挂 caveat（禁止拿它下结论）；
2. `answered_precision`（敢答时的正确率）升为**主判据** —— 不受"敢不敢答"影响；
3. `coverage`（敢答率）**必须同报**，防止靠保守刷 precision；
4. `abstention_f1` 作为**单一排序分**，避免多指标各说各话。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .metrics import derive_triple

TRIPLE_KEYS = ("accuracy", "coverage", "answered_precision")

# 【D2 2026-09-27】主判据轴的单一真相源。
# 此前它只是 pair_triple 里的硬编码默认值、没有任何配置入口，于是出现
# 「选拔按 main_metric=accuracy、判定按 answered_precision」的双轨：同一份数据
# 能得出两套相反的启用结论（P8 即卡在此处）。现在配置可覆盖，缺省仍取此值，
# 且它已被 final_eval.check_protocol 纳入协议字段（历史基线按此值补全）。
DEFAULT_PRIMARY_AXIS = "answered_precision"
DEFAULT_PRIMARY_AXIS_TOLERANCE_PP = 0.0

# 基线 accuracy 低于此值 ⇒ 判定为"贴地"，accuracy 的 Δ 不可作为结论依据。
# 取值理由：suppqa 基线 0.024 远低于任何有意义的水平；0.10 是"随便答答也能到"的量级。
FLOOR_ACCURACY = 0.10

READING_RULE = (
    "三指标必须同报：accuracy 是参考项（基线贴地时不得据此下结论）；"
    "answered_precision 是主判据（敢答时的正确率，不受敢不敢答影响）；"
    "coverage 必报（防止靠保守刷 precision）；abstention_f1 是单一排序分。"
)


def triple_of(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """从一次运行的聚合指标里取出三指标。

    历史产物（例如已封存的 assessment）里只有 accuracy + abstain_rate，
    没有题级弃权信息 —— 这时用 `derive_triple` 反推 coverage × precision，
    **不重新跑任何题**（封存集已退役，不可能重跑）。
    """
    m = dict(metrics or {})
    acc = m.get("accuracy")
    if m.get("coverage") is not None and m.get("answered_precision") is not None:
        cov = float(m["coverage"])
        prec = float(m["answered_precision"])
        derived = False
    else:
        d = derive_triple(acc, m.get("abstain_rate"))
        cov, prec = d["coverage"], d["precision"]
        derived = bool(d.get("derivable"))
    return {
        "accuracy": None if acc is None else round(float(acc), 6),
        "coverage": round(cov, 6),
        "answered_precision": round(prec, 6),
        "abstain_rate": (None if m.get("abstain_rate") is None
                         else round(float(m["abstain_rate"]), 6)),
        "unsupported_answer_rate": (None if m.get("unsupported_answer_rate") is None
                                    else round(float(m["unsupported_answer_rate"]), 6)),
        "abstention_f1": (None if m.get("abstention_f1") is None
                          else round(float(m["abstention_f1"]), 6)),
        "derived_from_accuracy": derived,
    }


def pair_triple(base_metrics: Dict[str, Any], cand_metrics: Dict[str, Any],
                cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """基线 vs 候选的三指标并列结论 + 主判据判定。"""
    cfg = cfg or {}
    ev = cfg.get("evaluation", {}) or {}
    b = triple_of(base_metrics)
    c = triple_of(cand_metrics)
    deltas = {k: (None if (b[k] is None or c[k] is None)
                  else round((c[k] - b[k]) * 100.0, 4))
              for k in TRIPLE_KEYS}

    axis = str(ev.get("primary_axis") or DEFAULT_PRIMARY_AXIS)
    if axis not in TRIPLE_KEYS:
        axis = DEFAULT_PRIMARY_AXIS
    tol = float(ev.get("primary_axis_tolerance_pp",
                       DEFAULT_PRIMARY_AXIS_TOLERANCE_PP) or 0.0)

    p_delta = deltas[axis]
    non_regress = (p_delta is not None) and (p_delta >= -tol - 1e-9)
    cov_delta = deltas["coverage"]

    if p_delta is None:
        verdict = "inconclusive"
    elif not non_regress:
        verdict = "regressed"
    elif p_delta > 1e-9 and (cov_delta is not None and cov_delta < 0):
        verdict = "improved_with_coverage_tradeoff"
    elif p_delta > 1e-9:
        verdict = "improved"
    else:
        verdict = "flat"

    base_acc = b["accuracy"]
    floor = bool(base_acc is not None and base_acc < FLOOR_ACCURACY)
    caveat: list = []
    if floor:
        caveat.append(
            f"基线 accuracy={base_acc:.3f} 贴地（<{FLOOR_ACCURACY}）："
            "accuracy 的 Δ 是低杆效应产物，不得作为结论依据；结论按 answered_precision 下。"
        )
    if b.get("derived_from_accuracy") or c.get("derived_from_accuracy"):
        caveat.append("coverage/precision 由 accuracy 与弃权率反推（无题级弃权信息），精度受限。")

    f1_b, f1_c = b.get("abstention_f1"), c.get("abstention_f1")
    return {
        "baseline": b,
        "candidate": c,
        "deltas_pp": deltas,
        "primary_axis": axis,
        "primary_delta_pp": p_delta,
        "primary_non_regression": bool(non_regress),
        "primary_tolerance_pp": tol,
        "coverage_reported": True,
        "floor_effect": {
            "baseline_accuracy": base_acc,
            "threshold": FLOOR_ACCURACY,
            "triggered": floor,
        },
        "ranking_score": {
            "name": "abstention_f1",
            "baseline": f1_b,
            "candidate": f1_c,
            "delta": (None if (f1_b is None or f1_c is None)
                      else round((f1_c - f1_b) * 100.0, 4)),
            "note": ("题级 gold_in_corpus 可得时才有值；历史封存产物缺该字段 → None"
                     if f1_b is None else ""),
        },
        "verdict": verdict,
        "caveats": caveat,
        "reading_rule": READING_RULE,
    }


def from_assessment(report: Dict[str, Any]) -> Dict[str, Any]:
    """从已封存的 `assessment.json` 回填三指标（**不重新跑题**）。

    assessment 的 per_seed 里存了 baseline_accuracy / final_accuracy 与
    secondary.abstain_rate（基线与候选都有）→ 足以反推 coverage × precision。
    """
    out: Dict[str, Any] = {}
    for ds, blk in (report.get("datasets") or {}).items():
        per_seed = blk.get("per_seed") or {}
        rows = {}
        for sd, r in per_seed.items():
            base_m = {
                "accuracy": r.get("baseline_accuracy"),
                "abstain_rate": (r.get("secondary_baseline") or {}).get("abstain_rate"),
                "unsupported_answer_rate": (r.get("secondary_baseline") or {})
                .get("unsupported_answer_rate"),
            }
            cand_m = {
                "accuracy": r.get("final_accuracy"),
                "abstain_rate": (r.get("secondary") or {}).get("abstain_rate"),
                "unsupported_answer_rate": (r.get("secondary") or {})
                .get("unsupported_answer_rate"),
            }
            rows[sd] = {
                **pair_triple(base_m, cand_m),
                "role": blk.get("role"),
                "delta_accuracy_pp": r.get("delta_pp"),
                "ci95": r.get("ci95"),
                "p_delta_gt_0": r.get("p_delta_gt_0"),
            }
        out[ds] = {"role": blk.get("role"), "n_tasks": blk.get("n_tasks"),
                   "per_seed": rows,
                   "primary_seed": blk.get("primary_seed")}
    return out


def self_test() -> Dict[str, Any]:
    """自检：口径真的会拦住"刷分"，而不是换个说法继续放行。"""
    checks: list = []

    def _ck(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    # 1) suppqa 真实数字：基线 accuracy 0.024 / 弃权 0.89，ACTIVE 0.1585 / 0.585
    r = pair_triple({"accuracy": 0.024, "abstain_rate": 0.890},
                    {"accuracy": 0.1585, "abstain_rate": 0.585})
    _ck("suppqa_triple_matches_reported_numbers",
        abs(r["baseline"]["coverage"] - 0.11) < 0.01
        and abs(r["candidate"]["answered_precision"] - 0.382) < 0.01,
        f"base={r['baseline']} cand={r['candidate']}")
    _ck("floor_effect_flagged", r["floor_effect"]["triggered"] is True,
        "基线 0.024 必须触发贴地警示")
    _ck("accuracy_delta_is_not_the_verdict",
        r["primary_axis"] == "answered_precision" and r["verdict"] == "improved",
        f"verdict={r['verdict']} via {r['primary_axis']}")

    # 2) 全弃权刷 precision：precision 1.0 但 coverage 0 → 必须判回退（coverage 崩了）
    r2 = pair_triple({"accuracy": 0.5, "abstain_rate": 0.0, "coverage": 1.0,
                      "answered_precision": 0.5},
                     {"accuracy": 0.02, "abstain_rate": 0.98, "coverage": 0.02,
                      "answered_precision": 1.0})
    _ck("abstain_everything_is_not_a_win",
        r2["deltas_pp"]["coverage"] <= -50 and r2["verdict"] == "improved_with_coverage_tradeoff",
        f"coverage Δ={r2['deltas_pp']['coverage']}pp，verdict={r2['verdict']}")

    # 3) 主判据回退必须被判 regressed
    r3 = pair_triple({"accuracy": 0.5, "coverage": 0.8, "answered_precision": 0.62},
                     {"accuracy": 0.5, "coverage": 0.8, "answered_precision": 0.40})
    _ck("primary_regression_is_regressed",
        r3["verdict"] == "regressed" and r3["primary_non_regression"] is False,
        f"Δ={r3['primary_delta_pp']}pp")

    # 4) abstention_f1 排序分：题级信息齐全时能算出来
    class _O:
        def __init__(self, gold: bool, abstained: bool) -> None:
            self.gold_in_corpus = gold
            self.answer_index = -1 if abstained else 0
            self.correct = False

    from .metrics import abstention_f1

    outs = [_O(True, False), _O(True, True), _O(False, True), _O(False, False)]
    _ck("abstention_f1_in_range", 0.0 < abstention_f1(outs) < 1.0,
        f"F1={abstention_f1(outs):.3f}（TP=1,FP=1,FN=1 → 2*0.5*0.5/(1.0)=0.5）")

    return {"ok": all(c["passed"] for c in checks), "checks": checks}
