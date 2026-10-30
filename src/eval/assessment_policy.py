"""Aggregate assessment rules shared by evolution and release consumers."""
from __future__ import annotations

import math
from typing import Any


def finite_rate(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)


def classwise_non_regression(before: dict, after: dict, min_support: int = 1) -> dict:
    checks = []
    if not before or set(before) != set(after):
        checks.append({"check": "label_scope", "passed": False})
    for label in sorted(set(before) | set(after)):
        b, c = before.get(label) or {}, after.get(label) or {}
        n = b.get("n")
        checks.append({"label": label, "check": "support", "passed": (
            isinstance(n, int) and not isinstance(n, bool)
            and n >= min_support and n == c.get("n"))})
        for metric in ("accuracy", "coverage", "answered_precision"):
            bv, cv = b.get(metric), c.get(metric)
            checks.append({"label": label, "check": metric,
                           "baseline": bv, "candidate": cv,
                           "passed": finite_rate(bv) and finite_rate(cv)
                           and cv + 1e-9 >= bv})
    return {"passed": bool(checks) and all(x["passed"] for x in checks), "checks": checks}


def seed_safety(row: dict, *, primary_tolerance_pp: float = 0,
                coverage_tolerance_pp: float = 0, accuracy_tolerance_pp: float = 0,
                unsupported_tolerance_pp: float = 5) -> dict:
    """Require useful coverage and utility on every registered execution."""
    before, after = row.get("baseline") or {}, row.get("candidate") or {}
    delta = row.get("primary_delta_pp")
    checks = {"primary": (isinstance(delta, (int, float)) and not isinstance(delta, bool)
        and math.isfinite(delta) and delta >= -primary_tolerance_pp - 1e-9
        and row.get("primary_non_regression") is True)}
    for metric, tolerance in (("coverage", coverage_tolerance_pp),
                              ("accuracy", accuracy_tolerance_pp)):
        b, c = before.get(metric), after.get(metric)
        checks[metric] = (finite_rate(b) and finite_rate(c)
                          and (c - b) * 100 >= -tolerance - 1e-9)
    b, c = before.get("unsupported_answer_rate"), after.get("unsupported_answer_rate")
    checks["unsupported"] = (finite_rate(b) and finite_rate(c)
                              and (c - b) * 100 <= unsupported_tolerance_pp + 1e-9)
    return {"passed": all(checks.values()), "checks": checks}
