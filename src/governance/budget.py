"""次指标容差门禁 + **适用性判定**（R1b）。

这条模块的起因是一次真实跑测：配置里写着 `gates.accept.tolerance`
（`cost_pct: 15` / `p95_latency_pct: 20` / `citation_precision_pp: 1.0`），
但**全仓库没有任何一行代码读取过它们** —— 门禁配了，却从来没有接上。
更糟的是底层量本身：离线 FakeProvider 的 `latency_ms` 是写死的常数
（`90 + len(docs)*15`），token 数是 `len(text)/4` 估的。
即便把门禁接上，这两个量也是编的 —— 接了等于恒真通过。

开源怎么处理的 —— gepa 的答案非常直接
-----------------------------------
`vendor/gepa-main/gepa-main/src/gepa/api.py:280-300`：
调用方可以传 `max_reflection_cost` 作为成本预算，但 gepa 在**构建阶段**就校验
这个预算到底能不能被执行：

    "max_reflection_cost is set but reflection_strategy does not expose total_cost — "
    "the cost stopper would silently never fire (unbounded reflection spend)."

也就是说：**"配了门槛但底层量测不出来"在 gepa 眼里不是"通过"，是故障**，
它在配置期就报错，绝不让它悄悄空转。
`src/gepa/utils/stop_condition.py:176-190` 的 `MaxReflectionCostStopper`
docstring 更是把这件事写死："Custom callables wrapped in TrackingLM always
report 0.0 cost and **never trip this stopper**."

本模块照抄这个判据：
- 每个容差项先判**能不能测**（provider 是否真的产生该量）；
- 测不了 → 状态 `n_a`，**不进通过/失败判定**，并且理由写进报告；
- 能测 → 正常比；
- 若调用方显式要求 `strict`，测不了就直接抛错（gepa 口径）。

这样报告里永远不可能出现"成本门禁：通过"这种假话。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# 各次指标在哪些 provider 下是**真实可测**的。
#
# fake:  latency 是常数、token 是 len/4 估算 → 两者都不可测；
#        citation_precision 由 outcome（引用 vs 是否 gold）真实算出 → 可测；
#        unsupported_answer_rate 由 outcome 的 gold_in_corpus / answer_index 真值算出
#        → 也是可测的（与 provider 无关，纯离线判据）。
# http:  三个都由真实调用产生 → 全部可测。
MEASURABLE_BY_PROVIDER: Dict[str, Dict[str, bool]] = {
    "fake": {"cost": False, "p95_latency": False,
             "citation_precision": True, "unsupported_answer_rate": True,
             "citation_support_rate": False,
             "answer_evidence_consistency_rate": False},
    "http": {"cost": True, "p95_latency": True,
             "citation_precision": True, "unsupported_answer_rate": True,
             "citation_support_rate": True,
             "answer_evidence_consistency_rate": True},
}

# 配置项 → 内部量名 → 单位语义
#   pct     : 相对变化百分比（base=0 时不可判），用于成本/延迟
#   pp      : 百分点下行容差（越低越差，允许下降不超过 limit），用于引用精度
#   pp_up   : 百分点**上行**容差（越高越差，允许上升不超过 limit），用于无支撑主张率
_TOLERANCE_SPEC = {
    "cost_pct": ("cost", "pct"),
    "p95_latency_pct": ("p95_latency", "pct"),
    "citation_precision_pp": ("citation_precision", "pp"),
    # 【B / P3】无支撑主张率：越高越差（幻觉迹象），候选只允许比基线高不超过 limit 个百分点。
    "unsupported_answer_rate_pp": ("unsupported_answer_rate", "pp_up"),
    "citation_support_rate_pp": ("citation_support_rate", "pp"),
    "answer_evidence_consistency_rate_pp": ("answer_evidence_consistency_rate", "pp"),
}


class BudgetGateError(RuntimeError):
    pass


def provider_kind(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("runtime", {}).get("provider", "fake")).lower()


def measurable(quantity: str, kind: Optional[str] = None, cfg: Dict[str, Any] = None) -> bool:
    kind = kind or provider_kind(cfg or {})
    return bool(MEASURABLE_BY_PROVIDER.get(kind, {}).get(quantity, False))


def _value(metrics: Dict[str, Any], quantity: str) -> Optional[float]:
    """取次指标数值；取不到（None）也算测不了。

    成本特别处理：`cost_usd` 为 None 表示"没有真实计价"（P2），
    此时即便 provider 是 http，金额也不可比 —— 退到 `cost_usd_nominal` 同样
    不成立（那是估的），所以直接判不可测。
    """
    if quantity == "cost":
        v = metrics.get("cost_usd")
        return None if v is None else float(v)
    if quantity == "p95_latency":
        v = metrics.get("p95_latency_ms")
        return None if v is None else float(v)
    if quantity == "citation_precision":
        v = metrics.get("citation_precision")
        return None if v is None else float(v)
    if quantity == "unsupported_answer_rate":
        v = metrics.get("unsupported_answer_rate")
        return None if v is None else float(v)
    if quantity == "citation_support_rate":
        v = metrics.get("citation_support_rate")
        return None if v is None else float(v)
    if quantity == "answer_evidence_consistency_rate":
        v = metrics.get("answer_evidence_consistency_rate")
        return None if v is None else float(v)
    return None


def check_tolerances(base: Dict[str, Any], cand: Dict[str, Any],
                     cfg: Dict[str, Any], kind: Optional[str] = None,
                     base_effective: Optional[Any] = None,
                     cand_effective: Optional[Any] = None) -> Dict[str, Any]:
    """比对次指标容差，返回逐项状态 + 总体判定。

    返回：
        {
          "items": [{name, quantity, status: pass|fail|n_a, reason, base, cand, limit}],
          "passed": bool,          # 只对 status != n_a 的项做判定
          "not_measurable": [...], # 测不了的量（报告里必须点名）
          "provider_kind": str,
        }
    """
    kind = kind or provider_kind(cfg)
    tol = dict((cfg.get("gates", {}) or {}).get("accept", {}).get("tolerance", {}) or {})
    strict = bool(tol.pop("strict", False)) if "strict" in tol else False
    required = set(tol.pop("required", []) or [])

    items = []
    for name, (quantity, unit) in _TOLERANCE_SPEC.items():
        limit = tol.get(name)
        if limit is None:
            continue
        # 【问题3 修复】引用精度门禁：只有基线与候选**都实际启用引用行为**时才可比。
        # 否则（典型：baseline-v0 不开引用通道 → 引用恒 0）这门禁即使无限容差也永远
        # 判通过，却保护不了引用能力 —— 按推荐方案标 n_a（与 gepa 式 fail-loud 一致）。
        # 注意：配置键是 `citation_precision_pp`，内部量名是 `citation_precision`，
        # 这里必须用 quantity（内部量名）判定，而不是 name（配置键）。
        # 向后兼容：仅当调用方**显式**传入两个 effective 集合时才启用此规则；
        # 旧调用（不传 effective）继续走原 pp 口径，行为不变。
        if quantity == "citation_precision" and base_effective is not None and cand_effective is not None:
            beff = set(base_effective or [])
            ceff = set(cand_effective or [])
            if "citation" not in beff or "citation" not in ceff:
                needed = name in required or quantity in required
                items.append({"name": name, "quantity": quantity,
                              "status": "fail" if needed else "n_a",
                              "reason": "基线与候选未同时启用引用行为（effective 不含 citation），"
                                        "引用精度不可比；" +
                                        ("必需门禁失败" if needed else "标 n/a 不计入判定"),
                              "base": None, "cand": None, "limit": limit})
                continue
        can = measurable(quantity, kind, cfg)
        b, c = _value(base, quantity), _value(cand, quantity)
        if not can or b is None or c is None:
            why = (f"provider={kind} 下 {quantity} 不是真实观测量"
                   if not can else f"{quantity} 无真实数值（None）")
            if strict:
                raise BudgetGateError(
                    f"门禁 {name} 被显式要求强制执行（tolerance.strict=true），"
                    f"但 {why} —— 该门槛永远不会响。对齐 gepa "
                    f"（api.py:286 `the cost stopper would silently never fire`）："
                    f"要么换成可测的 provider，要么把这个门禁删掉。"
                )
            needed = name in required or quantity in required
            items.append({"name": name, "quantity": quantity,
                          "status": "fail" if needed else "n_a",
                          "reason": why + ("；必需门禁失败" if needed
                                           else "；不计入通过/失败判定"),
                          "base": None, "cand": None, "limit": limit})
            continue

        if unit == "pct":
            # 相对变化百分比：两臂都是真实零价时，成本变化就是精确的 0，
            # 不应把可观测的零价误报为不可判。只有一臂非零时才保持 n_a，
            # 因为此时相对变化没有有限的分母。
            if b == 0:
                if c == 0 and quantity == "cost" and kind == "http":
                    items.append({"name": name, "quantity": quantity, "status": "pass",
                                  "reason": "基线与候选均为已计量的精确零价，成本变化为 0%",
                                  "base": b, "cand": c, "limit": limit})
                else:
                    items.append({"name": name, "quantity": quantity, "status": "n_a",
                                  "reason": "基线值为 0，无法计算相对变化",
                                  "base": b, "cand": c, "limit": limit})
                continue
            change = (c - b) / abs(b) * 100.0
            ok = change <= float(limit) + 1e-9
            items.append({"name": name, "quantity": quantity,
                          "status": "pass" if ok else "fail",
                          "reason": f"相对变化 {change:+.2f}% ≤ 容差 {limit}%"
                                    if ok else f"相对变化 {change:+.2f}% 超出容差 {limit}%",
                          "base": b, "cand": c, "limit": limit})
        elif unit == "pp_up":
            # 上行容差（越高越差）：无支撑主张率这类指标，候选只允许比基线
            # **高**不超过 limit 个百分点，超过即判失败（接地性回退 / P3）。
            change = (c - b) * 100.0
            ok = change <= float(limit) + 1e-9
            items.append({"name": name, "quantity": quantity,
                          "status": "pass" if ok else "fail",
                          "reason": f"上升 {change:+.2f}pp ≤ 容差 {limit}pp"
                                    if ok else f"上升 {change:+.2f}pp 超过容差 {limit}pp（接地性回退）",
                          "base": b, "cand": c, "limit": limit})
        else:  # pp：下行容差（越低越差，允许下降不超过 limit 个百分点）
            change = (c - b) * 100.0
            ok = change >= -float(limit) - 1e-9
            items.append({"name": name, "quantity": quantity,
                          "status": "pass" if ok else "fail",
                          "reason": f"变化 {change:+.2f}pp ≥ -{limit}pp"
                                    if ok else f"下降 {abs(change):.2f}pp 超过容差 {limit}pp",
                          "base": b, "cand": c, "limit": limit})

    minimum = dict((cfg.get("gates", {}) or {}).get("accept", {}).get("minimum", {}) or {})
    allowed_minimums = {"citation_support_rate",
                        "answer_evidence_consistency_rate"}
    for quantity, floor in minimum.items():
        if quantity not in allowed_minimums:
            raise BudgetGateError(f"unknown absolute grounding minimum: {quantity}")
        if not isinstance(floor, (int, float)) or isinstance(floor, bool) or not 0 <= floor <= 1:
            raise BudgetGateError(f"invalid absolute grounding minimum: {quantity}")
        observed = _value(cand, quantity)
        can = measurable(quantity, kind, cfg)
        ok = can and observed is not None and observed >= float(floor) - 1e-9
        items.append({"name": "minimum." + quantity, "quantity": quantity,
                      "status": "pass" if ok else "fail",
                      "reason": (f"candidate {quantity}={observed} meets floor {floor}"
                                 if ok else f"candidate {quantity}={observed} is below floor {floor} or unmeasurable"),
                      "base": _value(base, quantity), "cand": observed,
                      "limit": floor})
    judged = [i for i in items if i["status"] != "n_a"]
    return {
        "items": items,
        "passed": all(i["status"] == "pass" for i in judged),
        "not_measurable": [i["quantity"] for i in items if i["status"] == "n_a"],
        "provider_kind": kind,
    }


def self_test() -> Dict[str, Any]:
    """自检：证明"测不了就不算通过"这条真的生效，而不是恒真。"""
    checks = []

    def _ck(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    cfg = {"runtime": {"provider": "fake"},
           "gates": {"accept": {"tolerance": {"cost_pct": 15, "p95_latency_pct": 20,
                                              "citation_precision_pp": 1.0}}}}
    base = {"cost_usd": None, "p95_latency_ms": 150.0, "citation_precision": 0.5}
    cand = {"cost_usd": None, "p95_latency_ms": 900.0, "citation_precision": 0.5}
    r = check_tolerances(base, cand, cfg)
    _ck("fake_provider_cost_latency_are_n_a",
        set(r["not_measurable"]) == {"cost", "p95_latency"},
        f"not_measurable={r['not_measurable']}")
    # 关键：延迟涨了 6 倍，但 fake 下**不能**据此判失败
    _ck("fake_provider_latency_spike_not_judged",
        all(i["status"] == "n_a" for i in r["items"] if i["quantity"] == "p95_latency"),
        "fake 下延迟暴涨 6 倍也只能记 n_a，绝不能记 pass")
    _ck("fake_overall_passed_is_true_but_only_from_measurable",
        r["passed"] is True and len(r["items"]) == 3, "只有可测项参与判定")

    # 真实 provider：同样的暴涨必须判失败
    cfg_h = {"runtime": {"provider": "http"},
             "gates": {"accept": {"tolerance": {"p95_latency_pct": 20}}}}
    base_h = {"cost_usd": 0.1, "p95_latency_ms": 959.0, "citation_precision": 0.0}
    cand_h = {"cost_usd": 0.1, "p95_latency_ms": 4160.0, "citation_precision": 0.0}
    r2 = check_tolerances(base_h, cand_h, cfg_h)
    lat_item = next(i for i in r2["items"] if i["quantity"] == "p95_latency")
    _ck("http_provider_latency_spike_is_fail",
        lat_item["status"] == "fail" and r2["passed"] is False,
        f"真实数据 959→4160ms 必须判 fail（{lat_item['reason']}）")

    # strict 口径：对齐 gepa，配了却测不了 → 直接抛错
    cfg_s = {"runtime": {"provider": "fake"},
             "gates": {"accept": {"tolerance": {"cost_pct": 15, "strict": True}}}}
    try:
        check_tolerances(base, cand, cfg_s)
        _ck("strict_mode_raises", False, "strict=true 时未抛错")
    except BudgetGateError as exc:
        _ck("strict_mode_raises", "永远不会响" in str(exc), str(exc)[:80])

    # 【问题3 修复】引用门：基线不开引用（effective 不含 citation）→ 标 n_a，
    # 即使 candidate 开了引用、且引用精度真实可测、数值非 0。
    cfg_c = {"runtime": {"provider": "http"},
             "gates": {"accept": {"tolerance": {"citation_precision_pp": 1.0}}}}
    # 基线不开引用（空 effective），候选开了引用 + 真实引用精度 0.7
    r3 = check_tolerances(
        {"citation_precision": 0.0}, {"citation_precision": 0.7}, cfg_c,
        base_effective=set(), cand_effective={"citation"},
    )
    cit3 = next(i for i in r3["items"] if i["quantity"] == "citation_precision")
    _ck("citation_gate_n_a_when_base_not_citing",
        cit3["status"] == "n_a" and r3["passed"] is True,
        f"基线不开引用时引用门必须 n_a（{cit3['reason']}）")
    # 双方都开引用 → 正常比（候选 0.7 vs 基线 0.5，下降 20pp 超容差 1 → fail）
    r4 = check_tolerances(
        {"citation_precision": 0.5}, {"citation_precision": 0.3}, cfg_c,
        base_effective={"citation"}, cand_effective={"citation"},
    )
    cit4 = next(i for i in r4["items"] if i["quantity"] == "citation_precision")
    _ck("citation_gate_compares_when_both_cite",
        cit4["status"] == "fail" and r4["passed"] is False,
        f"双方都引用时应正常判 fail（{cit4['reason']}）")

    # 【B / P3】无支撑主张率（unsupported_answer_rate）：越高越差，上行容差。
    # fake 下该量可测（纯离线判据），不会被记成 n_a。
    cfg_u = {"runtime": {"provider": "fake"},
             "gates": {"accept": {"tolerance": {"unsupported_answer_rate_pp": 5.0}}}}
    # 候选 0.10 vs 基线 0.07 → 上升 3pp ≤ 5pp → pass
    r5 = check_tolerances(
        {"unsupported_answer_rate": 0.07}, {"unsupported_answer_rate": 0.10}, cfg_u)
    u5 = next(i for i in r5["items"] if i["quantity"] == "unsupported_answer_rate")
    _ck("unsupported_small_increase_passes",
        u5["status"] == "pass" and r5["passed"] is True,
        f"上升 3pp 应在容差内（{u5['reason']}）")
    # 候选 0.30 vs 基线 0.07 → 上升 23pp > 5pp → fail（接地性回退被拦下）
    r6 = check_tolerances(
        {"unsupported_answer_rate": 0.07}, {"unsupported_answer_rate": 0.30}, cfg_u)
    u6 = next(i for i in r6["items"] if i["quantity"] == "unsupported_answer_rate")
    _ck("unsupported_large_increase_fails",
        u6["status"] == "fail" and r6["passed"] is False,
        f"上升 23pp 应超容差失败（{u6['reason']}）")

    return {"ok": all(c["passed"] for c in checks), "checks": checks}
