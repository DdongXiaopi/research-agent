"""B2 复核：历史 promote 的次指标"通过"是不是真的通过。

背景（一条已经确认的 bug）：`src/evolution/round.py` 里候选指标被二次索引
`.get("metrics", {})`，候选侧恒为空字典 ⇒ cost / latency / citation_precision /
unsupported_answer_rate 四条次指标**从未真正比较过候选值**，一律落到 `n/a`，
而 `n/a` 不参与通过/失败判定 —— 于是所有历史 promote 的次指标"通过"都是**假通过**。

本脚本用**修好之后的门禁**把历史轮次重判一遍，回答两个问题：
  1. 有多少次 promote 在真实比较下会被拦下？
  2. 当前 champion 是否属于这一类（决策 4：若属于，降为 challenger，不删除）。

用法：
    python scripts/audit_promotions.py [--config configs/experiment.http.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.eval.final_eval import load_baseline
from src.governance import release as rel
from src.governance.budget import check_tolerances
from src.util import artifacts_path, read_json, write_json

ROUNDS_ROOT = artifacts_path("rounds")
OUT = artifacts_path("audit", "promotion_audit.json")


def _candidate_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not ROUNDS_ROOT.exists():
        return rows
    for p in sorted(ROUNDS_ROOT.glob("round-*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot audit malformed round file {p}: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("candidates", []), list):
            raise ValueError(f"Invalid round or candidates in {p}")
        for c in data.get("candidates") or []:
            if not isinstance(c, dict):
                raise ValueError(f"Invalid candidate in {p}")
            if not c.get("candidate_id"):
                raise ValueError(f"Candidate without an ID in {p}")
            rows.append({
                "round_id": data.get("round_id"),
                "candidate_id": c["candidate_id"],
                "status": c.get("status"),
                "target": c.get("target_artifact"),
                "parent_release_id": c.get("parent_release_id"),
                "candidate_metrics": c.get("candidate_metrics") or {},
                "baseline_metrics": c.get("baseline_metrics") or {},
            })
    return rows


def audit(cfg: Dict[str, Any]) -> Dict[str, Any]:
    champion = None
    try:
        champion = rel.load_aliases().get("champion") or rel.current_release_id()
    except Exception:
        try:
            champion = rel.current_release_id()
        except Exception:
            champion = None

    findings: List[Dict[str, Any]] = []
    for row in _candidate_rows():
        per_ds: Dict[str, Any] = {}
        incomplete = not row["candidate_metrics"] or not row["baseline_metrics"]
        if set(row["candidate_metrics"]) != set(row["baseline_metrics"]):
            incomplete = True
        for ds, cm in row["candidate_metrics"].items():
            base = (row["baseline_metrics"].get(ds) or {}).get("optimize") or {}
            cand = (cm or {}).get("optimize") or {}
            if not base or not cand:
                incomplete = True
                continue
            rep = check_tolerances(base, cand, cfg)
            if not rep["items"] or any(i["status"] == "n_a" for i in rep["items"]):
                incomplete = True
            judged = [i for i in rep["items"] if i["status"] == "fail"]
            per_ds[ds] = {
                "passed": rep["passed"],
                "failed_items": judged,
                "not_measurable": rep["not_measurable"],
            }
        failed = {ds: v for ds, v in per_ds.items() if v["failed_items"]}
        # 候选被发布后的 release id（命名规则见 release.publish：`rel-<candidate_id>`）
        release_id = f"rel-{row['candidate_id']}"
        findings.append({
            "round_id": row["round_id"],
            "candidate_id": row["candidate_id"],
            "release_id": release_id,
            "status_at_the_time": row["status"],
            "is_current_champion": release_id == champion,
            "would_be_blocked": bool(failed),
            "audit_complete": not incomplete,
            "failed_by_dataset": failed,
        })

    blocked = [f for f in findings if f["would_be_blocked"]]
    champion_blocked = [f for f in blocked if f["is_current_champion"]]
    champion_findings = [f for f in findings if f["is_current_champion"]]
    champion_complete = bool(champion_findings) and all(f["audit_complete"] for f in champion_findings)
    return {
        "created_at": None,
        "config_tolerance": (cfg.get("gates", {}).get("accept", {}).get("tolerance") or {}),
        "champion": champion,
        "n_candidates": len(findings),
        "n_blocked": len(blocked),
        "blocked_releases": [f["release_id"] for f in blocked],
        "champion_would_be_blocked": True if champion_blocked else (False if champion_complete else None),
        "verdict": ("当前 champion 在修好的门禁下会被拦下 —— 按决策 4 应降为 challenger"
                    if champion_blocked else
                    "当前 champion 在修好的门禁下仍可放行" if champion_complete else
                    "证据不足，无法判断当前 champion 是否可以放行"),
        "findings": findings,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.http.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rep = audit(cfg)
    from src.util import now_iso

    rep["created_at"] = now_iso()
    write_json(OUT, rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "findings"},
                     ensure_ascii=False, indent=2))
    print(f"\n明细 {len(rep['findings'])} 条 → {OUT}")
    for f in rep["findings"]:
        if f["would_be_blocked"]:
            for ds, v in f["failed_by_dataset"].items():
                for i in v["failed_items"]:
                    print(f"  - {f['release_id']} ({ds}) {i['name']}: {i['reason']}"
                          f"{'  <== champion' if f['is_current_champion'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
