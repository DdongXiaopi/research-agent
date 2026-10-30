"""待重封清单（决策 3）。

P6（doc_id 去 task_id）已实现但**不启用**：启用就要重封 final_holdout 基线，
而揭盲预算已耗尽。本模块把"哪些变更会花掉一次重封"登记成可机器读取的清单，
并强制**合并执行** —— 单独为一项花掉唯一那次授权，等于把预算浪费在防御深度上。

用法：`python scripts/rsi.py reseal-plan`
"""

from __future__ import annotations

from typing import Any, Dict, List

import yaml

from ..util import path

PLAN_FILE = path("configs", "pending_reseal.yaml")

RESEAL_RULE = (
    "清单里任何一项启用 ⇒ 协议指纹变化 ⇒ 必须重跑封存基线。"
    "重封必须与清单中其它待办项**合并为同一次**（共用一次揭盲授权）。"
)


def load_plan() -> Dict[str, Any]:
    if not PLAN_FILE.exists():
        return {"items": [], "policy": RESEAL_RULE}
    with open(PLAN_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"items": [], "policy": RESEAL_RULE}


def reseal_items(statuses: List[str] | None = None) -> List[Dict[str, Any]]:
    items = load_plan().get("items") or []
    if statuses:
        return [i for i in items if i.get("status") in statuses]
    return items


def plan() -> Dict[str, Any]:
    """给出"现在要不要重封 / 要重封就得一起做哪些"的判定。"""
    data = load_plan()
    items = data.get("items") or []
    pending = [i for i in items if i.get("status") != "done"]
    ready = [i for i in pending if i.get("id") in ("P4", "P6")]
    from ..eval.unblind_budget import budget_state

    # 【D1 2026-09-27】结论集已从 suppqa 换到 pubmedqa（ADR-0004），这里不能再
    # 硬编码 suppqa —— 否则 reseal-plan 会拿一个已退役数据集的额度来说事。
    dataset = None
    warnings = []
    try:
        from ..config import load_config

        prim = (load_config().get("evaluation", {})
                .get("primary_final_datasets") or [])
        if prim:
            dataset = str(prim[0])
        else:
            warnings.append("primary_final_datasets_missing")
    except Exception as exc:
        warnings.append("config_unavailable:" + type(exc).__name__)

    try:
        if dataset is None:
            raise ValueError("final dataset is unknown")
        st = budget_state(dataset)
        budget = {"dataset": dataset, "retired": st["retired"],
                  "remaining": st["remaining"], "used_total": st["used_total"]}
    except Exception:
        budget = {"dataset": dataset, "retired": None,
                  "remaining": None, "used_total": None}

    return {
        "plan_file": str(PLAN_FILE),
        "updated_at": data.get("updated_at"),
        "policy": data.get("policy") or RESEAL_RULE,
        "rule": RESEAL_RULE,
        "items": items,
        "pending_ids": [i.get("id") for i in pending],
        "ready_to_reseal_ids": [i.get("id") for i in ready],
        "decision": {
            # 决策 3 的当前结论：**暂不重封**。清单存在的意义是"一旦要做就必须合并"。
            "action": "hold",
            "if_triggered": ("必须合并为同一次重封：" + " + ".join(i["id"] for i in ready)
                             if ready else "无待重封项"),
            "reason": ("没有待重封项时维持现状（P6 保持 task_scoped，靠"
                       "封存期禁缓存 + 泄漏扫描 + 协议指纹三件套兜底）"
                       if not ready else
                       "待重封项已就位，但单独启用任何一项都只是防御深度，"
                       "不值得花掉唯一那次揭盲授权；要动就一起动"),
        },
        "budget": budget,
        "warnings": warnings,
        "how": ("python scripts/rsi.py --config configs/experiment.http.yaml final-baseline "
                "--force --budget-override-reason '<理由>' "
                "--budget-override-actor '<授权人>'"),
    }
