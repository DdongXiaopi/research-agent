"""指标计算（纯函数，便于单元测试，Spec §11 Unit）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def accuracy(outcomes: Sequence[Any]) -> float:
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if o.correct) / len(outcomes)


def citation_precision(outcomes: Sequence[Any]) -> float:
    """引用精度：被引文档中支撑答案（gold）的比例；未引用不计入分子也不惩罚。"""
    cited = [o for o in outcomes if getattr(o, "cited", None)]
    if not cited:
        return 0.0
    return sum(getattr(o, "citation_precision", 0.0) for o in cited) / len(cited)


def p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, int(round(0.95 * (len(s) - 1))))
    return float(s[idx])


def total_cost(outcomes: Sequence[Any]) -> Dict[str, object]:
    """成本汇总（P2）。

    `usd` 在任何一条样本**没有真实计价**时为 `None` —— 宁可空着，也不要把
    估算值混进一个看起来精确的字段里。估算态由 `cost_estimated` / `cost_basis`
    显式携带，`usd_nominal` 只作量级参考。
    """
    usd_values: List[float] = []
    nominal = 0.0
    nominal_complete = True
    pt = ct = 0
    estimated = False
    usage_complete = True
    pi_catalog_cost_total = 0.0
    pi_catalog_cost_seen = False
    pi_catalog_cost_complete = True
    bases: set = set()
    for o in outcomes:
        c = getattr(o, "cost", {}) or {}
        usage_available = bool(c.get("usage_available", True))
        usage_complete = usage_complete and usage_available and c.get("usage_complete", True)
        if not usage_available:
            nominal_complete = False
        u = c.get("usd", None)
        if u is None:
            estimated = True          # 有样本没真值 → 汇总就不能声称是真值
        else:
            usd_values.append(float(u))
        if c.get("usd_nominal") is None and "usage_available" in c:
            nominal_complete = False
        else:
            nominal += float(c.get("usd_nominal", 0.0))
        estimated = estimated or bool(c.get("cost_estimated", False))
        if c.get("cost_basis"):
            bases.add(str(c["cost_basis"]))
        pi_cost = c.get("pi_usage_cost_metadata")
        if (isinstance(pi_cost, dict) and
                isinstance(pi_cost.get("total"), (int, float)) and
                not isinstance(pi_cost.get("total"), bool)):
            pi_catalog_cost_seen = True
            pi_catalog_cost_total += float(pi_cost["total"])
        else:
            pi_catalog_cost_complete = False
        pt += int(c.get("prompt_tokens", 0))
        ct += int(c.get("completion_tokens", 0))
    return {
        "usd": round(sum(usd_values), 6) if usd_values and not estimated else None,
        "usd_nominal": round(nominal, 6) if nominal_complete else None,
        "cost_estimated": estimated,
        "cost_basis": "; ".join(sorted(bases)) if bases else "",
        # Pi's usage.cost is calculated from its model catalog metadata, not
        # returned as a provider bill. Keep it visible for diagnosis only.
        "pi_usage_cost_metadata_usd": (
            round(pi_catalog_cost_total, 9)
            if pi_catalog_cost_seen and pi_catalog_cost_complete else None),
        "pi_usage_cost_metadata_complete": (
            pi_catalog_cost_seen and pi_catalog_cost_complete),
        "pi_usage_cost_authoritative": False,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "usage_complete": usage_complete,
    }


def rate(outcomes: Sequence[Any], field: str) -> float:
    """某布尔字段为真的比例（用于过程指标）。"""
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if bool(getattr(o, field, False))) / len(outcomes)


def unsupported_answer_rate(outcomes: Sequence[Any]) -> float:
    """无证据仍给出确定答案的比例（幻觉率的离线近似）。

    分母 = 语料里根本没有支撑答案的题目（`gold_in_corpus=False`）。
    这类题无论怎么选都是错的，区别只在于**是硬猜还是声明无证据**：
    结晶出的 Skill 带来的正是这个差别，所以必须有一个指标看得到它。

    注意：分母依赖评测侧的 oracle（语料里有没有答案），线上无法直接测；
    真实环境对应的可观测量是"引用支撑率"与"弃权率"，本指标只用于离线夹具。
    """
    no_gold = [o for o in outcomes if not getattr(o, "gold_in_corpus", True)]
    if not no_gold:
        return 0.0
    return sum(1 for o in no_gold if int(getattr(o, "answer_index", -1)) >= 0) / len(no_gold)


def abstain_rate(outcomes: Sequence[Any]) -> float:
    """弃权率：明确声明"证据不足"而不作答的比例。"""
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if int(getattr(o, "answer_index", -1)) < 0) / len(outcomes)


def answered(outcomes: Sequence[Any]) -> List[Any]:
    """给出确定答案（未弃权）的题目子集。"""
    return [o for o in outcomes if int(getattr(o, "answer_index", -1)) >= 0]


def coverage(outcomes: Sequence[Any]) -> float:
    """coverage：给出确定答案的题目占比（= 1 − 弃权率）。"""
    if not outcomes:
        return 0.0
    return len(answered(outcomes)) / len(outcomes)


def answered_precision(outcomes: Sequence[Any]) -> float:
    """precision：**只在"有把握"（未弃权）子集上**的正确率。

    为什么要和 accuracy 分开（对齐 asta-bench LitQA2 `task.py:114-139`）：
        答错和弃权在 accuracy 上都是"不得分"，但对用户完全不是一回事 ——
        前者是编了一个答案，后者是老实说不知道。
        把两者混在 accuracy 里，"弃权率 0 / 幻觉率 1"这种最差组合是**隐形**的。
        拆成 coverage × precision 之后，这套组合会被 precision 直接暴露。
    """
    ans = answered(outcomes)
    if not ans:
        return 0.0
    return sum(1 for o in ans if o.correct) / len(ans)


def abstention_f1(outcomes: Sequence[Any]) -> float:
    """Abstention F1（对齐 AbstentionBench 的思路）：把"该弃权时弃权"当正类算 F1。

    为什么需要它：coverage 与 answered_precision 是**两轴**，单独看任何一个都能被刷
    —— 全弃权 ⇒ precision 无定义/虚高，全作答 ⇒ coverage=1 但幻觉满地。
    AbstentionBench（facebookresearch, 2025）的用法正是"把该答答对与该弃权弃掉
    合成一个分"，这样多指标就不用各说各话，可以拿它做**单一排序分**。

    口径（本项目的 oracle 是 `gold_in_corpus`）：
        TP = 语料里没有支撑答案 且 弃权了      （该弃权，弃对了）
        FP = 语料里有支撑答案  却 弃权了       （过度保守）
        FN = 语料里没有支撑答案 却 硬答了      （无支撑主张）
    与 `unsupported_answer_rate` 互补：后者只看 FN 那一格，这里三格一起看。
    """
    tp = fp = fn = 0
    for o in outcomes:
        has_gold = bool(getattr(o, "gold_in_corpus", True))
        abstained = int(getattr(o, "answer_index", -1)) < 0
        if not has_gold and abstained:
            tp += 1
        elif has_gold and abstained:
            fp += 1
        elif not has_gold and not abstained:
            fn += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def derive_triple(accuracy: Optional[float], abstain_rate: Optional[float]) -> Dict[str, float]:
    """从 accuracy 与弃权率**反推** coverage × precision（不重新跑任何题）。

    前提：弃权记错（"信息不足"选项不是正确答案）。此时
        accuracy = coverage × precision
    于是 coverage = 1 − abstain_rate、precision = accuracy / coverage。

    为什么这个函数很关键：封存集已经退役（揭盲预算耗尽），历史 assessment 里
    只存了 accuracy 与 abstain_rate —— 没有题级弃权信息。有了它，三指标口径
    可以**在不碰封存集的前提下**回填到已有数字上，而不是"等下次揭盲再说"。
    两者对不上（比如评分把弃权记对）时算出的 precision 会 >1，调用方应校验。
    """
    if accuracy is None or abstain_rate is None:
        return {"coverage": 0.0, "precision": 0.0, "derivable": False}
    cov = 1.0 - float(abstain_rate)
    if cov <= 0:
        return {"coverage": 0.0, "precision": 0.0, "derivable": False}
    return {"coverage": round(cov, 6), "precision": round(float(accuracy) / cov, 6),
            "derivable": True}


def aggregate(outcomes: Sequence[Any]) -> Dict[str, Any]:
    """Aggregate outcomes; provenance-dependent metrics are None without labels."""
    lat = [float(getattr(o, "latency_ms", 0)) for o in outcomes]
    cost = total_cost(outcomes)
    provenance_available = all(getattr(o, "provenance_labels_available", True)
                               for o in outcomes)
    answered = [o for o in outcomes if getattr(o, "status", "") == "ok"]
    citation_support_rate = (
        round(sum(bool(getattr(o, "citation_supported", False)) for o in answered)
              / len(answered), 6)
        if answered and all(getattr(o, "citation_supported", None) is not None
                            for o in answered) else None)
    answer_evidence_consistency_rate = (
        round(sum(bool(getattr(o, "answer_evidence_consistent", False))
                  for o in answered) / len(answered), 6)
        if answered and all(getattr(o, "answer_evidence_consistent", None)
                            is not None for o in answered) else None)
    return {
        "accuracy": round(accuracy(outcomes), 6),
        "citation_precision": (round(citation_precision(outcomes), 6)
                               if provenance_available else None),
        "citation_support_rate": citation_support_rate,
        "answer_evidence_consistency_rate": answer_evidence_consistency_rate,
        # 过程指标：主指标没动时，它们能证明"改 description 确实改变了行为"
        "skill_recall_rate": round(rate(outcomes, "skill_recalled"), 6),
        "gold_in_topk_rate": (round(rate(outcomes, "gold_in_topk"), 6)
                              if provenance_available else None),
        "gold_in_corpus_rate": (round(rate(outcomes, "gold_in_corpus"), 6)
                                if provenance_available else None),
        # 【M3】结晶 Skill 的收益体现在这里，而不是 accuracy
        "abstain_rate": round(abstain_rate(outcomes), 6),
        "unsupported_answer_rate": (round(unsupported_answer_rate(outcomes), 6)
                                    if provenance_available else None),
        # 【M4+】coverage × precision：把"编答案"和"说不知道"分开看（见函数文档）
        "coverage": round(coverage(outcomes), 6),
        "answered_precision": round(answered_precision(outcomes), 6),
        # 【决策 2】单一排序分：AbstentionBench 式 F1（该弃权就弃权的能力）
        "abstention_f1": (round(abstention_f1(outcomes), 6)
                          if provenance_available else None),
        "p95_latency_ms": p95(lat),
        "mean_latency_ms": round(sum(lat) / len(lat), 3) if lat else 0.0,
        # P2：成本三件套。usd 为 None 表示"无真实计价"，不要当成"花了 0 元"
        "cost_usd": cost["usd"],
        "cost_usd_nominal": cost["usd_nominal"],
        "cost_estimated": cost["cost_estimated"],
        "cost_basis": cost["cost_basis"],
        "pi_usage_cost_metadata_usd": cost["pi_usage_cost_metadata_usd"],
        "pi_usage_cost_metadata_complete": cost[
            "pi_usage_cost_metadata_complete"],
        "pi_usage_cost_authoritative": cost["pi_usage_cost_authoritative"],
        "prompt_tokens": cost["prompt_tokens"],
        "completion_tokens": cost["completion_tokens"],
        "pi_bridge_attempts": sum(int(getattr(o, "pi_bridge_attempts", 0))
                                  for o in outcomes),
        "pi_bridge_timeouts": sum(int(getattr(o, "pi_bridge_timeouts", 0))
                                  for o in outcomes),
        "n_tasks": len(outcomes),
    }


def aggregate_by_gold_label(outcomes: Sequence[Any],
                            tasks: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Return only class aggregates for MCQ evaluations, never per-task rows.

    This is particularly useful for a multi-class selection gate: overall
    accuracy can hide a collapse in one label, while the gate must not expose
    individual answers to candidate generation.
    """
    task_by_id = {str(task.get("task_id") or ""): task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("class-metric task map contains duplicate or empty task IDs")
    grouped: Dict[str, List[Any]] = {}
    labeled = 0
    for outcome in outcomes:
        task = task_by_id.get(str(getattr(outcome, "task_id", "")))
        if task is None:
            raise ValueError("class-metric outcome is outside the evaluated task scope")
        options = task.get("options")
        try:
            gold_index = int(task.get("correct_index", -1))
        except (TypeError, ValueError):
            gold_index = -1
        if not isinstance(options, list) or not 0 <= gold_index < len(options):
            continue
        label = str(options[gold_index]).strip()
        if not label:
            continue
        labeled += 1
        grouped.setdefault(label, []).append(outcome)

    if not labeled:
        return {}
    if labeled != len(outcomes):
        raise ValueError("class-metric scope mixes labeled and unlabeled tasks")

    result: Dict[str, Dict[str, float]] = {}
    for label, rows in sorted(grouped.items()):
        n = len(rows)
        answered = sum(int(getattr(row, "answer_index", -1)) >= 0 for row in rows)
        correct = sum(bool(getattr(row, "correct", False)) for row in rows)
        result[label] = {
            "n": n,
            "accuracy": round(correct / n, 6),
            "coverage": round(answered / n, 6),
            "answered_precision": round(correct / answered, 6) if answered else 0.0,
        }
    return result


def delta_pp(before: float, after: float) -> float:
    """百分点差值。"""
    return round((after - before) * 100.0, 4)
