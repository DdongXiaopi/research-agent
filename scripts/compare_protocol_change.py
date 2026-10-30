"""协议变更前后基线对比（逐题），用于验证"改动只波及该波及的地方"。

用法：
    python scripts/compare_protocol_change.py \
        --new artifacts/gate_verdicts/n4_baseline.json \
        --old litqa2=run-295a6e6a089514ab protocolqa=run-cfe60506d962bf64 \
              pubmedqa=run-395ddd6830d9f22c

为什么单独一个脚本：改协议之后，"哪些集应该变、哪些集必须纹丝不动"是**事先
声明的可证伪预测**。protocolqa 修的是证据截断，那么 litqa2 / pubmedqa（从没被
截断过）就应当逐题零变化 —— 一旦它们也变了，说明改动波及了不该波及的地方，
这次协议变更就必须回退重做。

这不是事后找补，是变更前的预测登记。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.util import path  # noqa: E402


def per_task(run_id: str) -> dict:
    """从 trace.jsonl 抽每个 task 的 (correct, status)。"""
    f = path(f"artifacts/runs/{run_id}/trace.jsonl")
    out: dict = {}
    if not f.exists():
        return out
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event_type") != "task.end":
            continue
        p = rec.get("payload")
        if isinstance(p, str):
            try:
                p = json.loads(p)
            except Exception:
                p = {}
        out[rec.get("task_id")] = {
            "correct": bool((p or {}).get("correct")),
            "status": (p or {}).get("status"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="协议变更前后基线逐题对比")
    ap.add_argument("--new", default="artifacts/gate_verdicts/n4_baseline.json",
                    help="重跑脚本产出的 JSON（含 runs.<ds>.run_id）")
    ap.add_argument("--old", nargs="*", default=[],
                    help="旧协议基线，格式 dataset=run_id")
    ap.add_argument("--expect-unchanged", nargs="*", default=["litqa2", "pubmedqa"],
                    help="预测**必须零变化**的数据集（阴性对照）")
    ap.add_argument("--expect-changed", nargs="*", default=["protocolqa"],
                    help="预测**应该变化**的数据集")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    new = json.loads(path(args.new).read_text(encoding="utf-8"))
    old_map = dict(kv.split("=", 1) for kv in args.old)

    rows = []
    ok = True
    for ds, blk in new.get("runs", {}).items():
        new_id = blk["run_id"]
        old_id = old_map.get(ds)
        if not old_id:
            print(f"[{ds:11s}] 无旧基线可对比，跳过", flush=True)
            continue
        a, b = per_task(old_id), per_task(new_id)
        ids = sorted(set(a) & set(b))
        flipped = [t for t in ids if a[t]["correct"] != b[t]["correct"]]
        status_changed = [t for t in ids if a[t]["status"] != b[t]["status"]]
        n = len(ids)
        oa = sum(1 for t in ids if a[t]["correct"])
        ob = sum(1 for t in ids if b[t]["correct"])
        oab = sum(1 for t in ids if a[t]["status"] == "abstain")
        bab = sum(1 for t in ids if b[t]["status"] == "abstain")
        unchanged = not flipped and not status_changed
        expect = ("必须不变" if ds in args.expect_unchanged
                  else ("应该变化" if ds in args.expect_changed else "无预测"))
        verdict = "PASS" if ((ds in args.expect_unchanged and unchanged) or
                             (ds in args.expect_changed and not unchanged) or
                             ds not in args.expect_unchanged + args.expect_changed) else "FAIL"
        if verdict == "FAIL":
            ok = False
        rows.append({
            "dataset": ds, "n_paired": n, "old_run": old_id, "new_run": new_id,
            "old_correct": oa, "new_correct": ob,
            "old_acc": round(oa / n, 4) if n else None,
            "new_acc": round(ob / n, 4) if n else None,
            "delta_pp": round((ob - oa) / n * 100, 2) if n else None,
            "old_abstain": oab, "new_abstain": bab,
            "abstain_delta_pp": round((bab - oab) / n * 100, 2) if n else None,
            "n_flipped_correct": len(flipped),
            "n_status_changed": len(status_changed),
            "byte_identical_results": unchanged,
            "expectation": expect, "verdict": verdict,
        })
        print(f"[{ds:11s}] n={n:3d} acc {oa/n:.3f}->{ob/n:.3f} ({(ob-oa)/n*100:+.1f}pp) "
              f"弃权 {oab/n:.3f}->{bab/n:.3f} | 翻转 {len(flipped)} 题, "
              f"状态变化 {len(status_changed)} 题 | 预测={expect} → {verdict}", flush=True)

    out = {"new": args.new, "old": old_map, "rows": rows, "all_expectations_met": ok}
    p = path(args.out) if args.out else path(args.new).with_name(
        path(args.new).stem + "_compare.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {p}")
    print("全部预测成立" if ok else "⚠️ 有预测未成立，需回退或重新归因")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
