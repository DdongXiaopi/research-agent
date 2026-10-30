"""多轮进化 campaign（Spec §1 工程成功标准：完成 ≥3 个进化轮次）。

单轮 `round` 证明"链路通"，多轮 campaign 证明"能连续进化且不退化"：
每一轮都从**上一轮发布后的 release** 出发，
走完整的 归因 → 候选 → 静态门 → 评测 → 判定 → 人工 review → 发布。

关于自动批准：
    发布必须过人工 review（FR-205）。默认在首个待审候选处停止；
    仅在隔离的 RSI_HOME 下显式请求 `--approve-simulated` 时，
    才以 `auto-sim` 身份演练发布，并在报告中标明是链路压测。
"""

from __future__ import annotations

import os
import re
from uuid import uuid4
from typing import Any, Dict, List, Optional, Sequence

from ..contracts.types import Candidate
from ..governance import release as rel
from ..governance import review as rv
from ..util import now_iso, artifacts_path, artifacts_root, repo_root, read_json, sha256_obj, write_json
from .round import run_round
from ..config import config_fingerprint
from ..util import sha256_file
from ..contracts.types import content_sha256
from ..eval.run_manifest import source_tree_fingerprint
from pathlib import Path
from .candidate_memory import proposal_sha256

CAMPAIGN_ROOT = artifacts_path("campaigns")


def _load_candidate(candidate_id: str) -> Optional[Candidate]:
    p = artifacts_path("candidates", f"{candidate_id}.json")
    if not p.exists():
        return None
    return Candidate(**read_json(p))


def _bind_review(candidate: Candidate, request_path: str) -> Dict[str, Any]:
    request = read_json(request_path)
    if (request.get("candidate_id") != candidate.candidate_id
            or request.get("parent_release_id") != candidate.parent_release_id
            or request.get("candidate_artifact_sha256") != rv.candidate_artifact_sha256(candidate)
            or request.get("new_content_sha256") != content_sha256(candidate.new_content)):
        raise ValueError("campaign review request does not bind its exact candidate")
    return {"review_request_path": str(request_path),
            "review_request_sha256": rv.request_sha256(request),
            "candidate_artifact_sha256": rv.candidate_artifact_sha256(candidate)}


def _review_matches(record: Dict[str, Any], step: Dict[str, Any], candidate: Candidate) -> bool:
    return (record.get("action_type") == "release"
            and record.get("subject_id") == candidate.candidate_id
            and record.get("subject_sha256") == content_sha256(candidate.new_content)
            and record.get("parent_release_id") == step["from_release"]
            and record.get("diff") == candidate.diff
            and bool(str(record.get("reviewer") or "").strip())
            and (record.get("metrics") or {}).get("review_request_sha256") == step["review_request_sha256"])


def _mean_delta_pp(cand: Dict[str, Any]) -> float:
    """候选在**全部数据集**上的 dev_optimize 主指标平均增量（跨数据集权衡）。"""
    d = cand.get("delta_pp") or {}
    vals = [float(v.get("optimize", 0.0)) for v in d.values() if isinstance(v, dict)]
    return sum(vals) / len(vals) if vals else 0.0


def _task_support(cand: Dict[str, Any]) -> Dict[str, int]:
    ts = cand.get("task_support") or {}
    return {"improved": int(ts.get("improved", 0)),
            "regressed": int(ts.get("regressed", 0)),
            "net": int(ts.get("net", 0))}


def _pick_best(accepted: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """在拟接受候选里选**逐题覆盖最大、回退最小**的那个。

    演进三段，每段都是被实测推着改的：
      1. 最初取 `accepted[0]` —— 由"归因类别的枚举顺序"决定发布谁。
         M3.1 修正：实测 description 候选总排第一，而它在 ProtocolQA 上增益是 0
         （召回已饱和 96.7%），三轮下来 ProtocolQA 一分没涨。
      2. 改成 `max(mean_delta_pp)` —— 但仍然不够：**跨数据集平均是最会掩盖回退的
         聚合方式**。CS 转移集 6 个回归里 5 个，在 dev_optimize 上的平均增益为正，
         于是被放行。平均增益看不见"在某个子集上变差"。
      3. 【M4+】改成逐题口径：主排序键 = 修好的题数（覆盖），次级键 = 弄坏的题数
         （回退越小越好），再往后才是平均增量做平手裁决。
         对齐 GEPA instance-wise Pareto front（`core/state.py:246`）的思想 ——
         按**每个验证样本**组织，而不是一个全局标量。
         但**不引入 GEPA 的 rng 加权采样**（`gepa_utils.py:133` `rng.choice`）：
         那会伤掉本项目的离线可复现性，这里保持确定性。

    硬过滤：净覆盖 <= 0 的候选不参与（"没有净增益就不发布"，与提案侧同口径）。

    【M1/M2 回顾】平手裁决的最后一位是 `candidate_id` —— 这在当时是个**结构性风险**：
    `candidate_id` 原为 uuid，等于"两个候选完全打平时由随机数决定发布谁"。
    修法不是换成插入序号（那会退化成"列表顺序决定"，且与既有
    `test_deterministic_tie_break` 要求的顺序无关性冲突），而是**让 ID 本身变确定**：
    `candidate_id` 改为内容哈希（`contracts.types.content_id`，对齐 GEPA
    `core/state.py:37-39` 用内容 sha256 做候选身份）。于是末位键同时满足
    确定性 + 顺序无关 + 跨运行可复现 —— 比插入序号（GEPA `candidate_selector.py:68`
    用下标 i）更强一等。
    """
    usable = [c for c in accepted if _task_support(c)["net"] > 0]
    if not usable:
        return None
    return max(usable, key=lambda c: (_task_support(c)["improved"],
                                      -_task_support(c)["regressed"],
                                      round(_mean_delta_pp(c), 6),
                                      str(c.get("candidate_id") or "")))


def run_campaign(cfg: Dict[str, Any], datasets: Optional[List[str]] = None,
                 rounds: int = 3, seed: int = 0, seeds: Optional[Sequence[int]] = None,
                 limit: Optional[int] = None, approve: bool = False,
                 reviewer: str = "auto-sim", git: bool = False,
                 resume_campaign: Optional[str] = None) -> Dict[str, Any]:
    if approve and (not os.environ.get("RSI_HOME") or
                    artifacts_root().resolve() == (repo_root() / "artifacts").resolve()):
        raise ValueError("模拟批准只允许在 RSI_HOME 指定的独立产物目录运行")
    cfg_fp = sha256_obj(cfg)
    datasets = datasets or list(cfg.get("evaluation", {}).get("datasets", ["litqa2"]))
    steps: List[Dict[str, Any]] = []
    # 【M4+】跨轮候选池：GEPA 的 `state.program_candidates` 是跨轮永久保留的，
    # 本轮落选的好候选不该被丢掉 —— 下一轮可能没有更好的，就该轮到它。
    pool: List[Dict[str, Any]] = []

    contract = {"config_fingerprint": config_fingerprint(cfg), "datasets": list(datasets),
                "rounds": int(rounds), "seeds": list(seeds) if seeds else [seed],
                "limit": limit, "approve": approve,
                "source_sha256": source_tree_fingerprint()["sha256"]}
    campaign_id = resume_campaign or uuid4().hex
    if not re.fullmatch(r"[a-f0-9]{32}", campaign_id):
        raise ValueError("invalid campaign checkpoint ID")
    checkpoint_path = CAMPAIGN_ROOT / f"checkpoint-{campaign_id}.json"
    checkpoint = {}
    if resume_campaign:
        checkpoint = read_json(checkpoint_path)
        checksum = checkpoint.get("checkpoint_sha256")
        if checkpoint.get("schema") != "evolution-campaign-checkpoint-v2" or sha256_obj({k: v for k, v in checkpoint.items()
                       if k != "checkpoint_sha256"}) != checksum or checkpoint.get("contract") != contract:
            raise ValueError("campaign checkpoint identity/contract changed")
        if checkpoint.get("status") in {"round_in_progress", "failed"}:
            raise ValueError("interrupted round needs request reconciliation; finished calls cannot be replayed")
        steps, pool = checkpoint["steps"], checkpoint["pool"]
        if checkpoint.get("live_budget"):
            if any(os.environ.get(name) != value
                   for name, value in checkpoint["live_budget"].items()):
                raise ValueError("campaign resume must retain the original plan and cumulative request ledger")
        if checkpoint.get("status") in {"awaiting_review", "publication_pending"}:
            step = steps[-1]
            candidate = _load_candidate(step["candidate_id"])
            if candidate is None or _bind_review(candidate, step["review_request_path"]) != {
                    key: step[key] for key in ("review_request_path", "review_request_sha256", "candidate_artifact_sha256")}:
                raise ValueError("campaign awaited candidate or review request changed")
            current = rel.current_release_id()
            if current == step["from_release"]:
                review_path = Path(step["review_request_path"]).parent / f"{candidate.candidate_id}.review.json"
                record = read_json(review_path) if review_path.exists() else {}
                if record.get("decision") != "reject" or not _review_matches(record, step, candidate):
                    raise ValueError("campaign awaits exact human rejection or approved publication")
                step.update({"published": None, "active_release": current, "awaiting_review": None,
                             "human_decision": "reject", "review_path": str(review_path),
                             "review_sha256": sha256_file(review_path), "ok": True})
            else:
                info = rel.release_info(current)
                published = Candidate(**(info.get("candidate") or {}))
                record = info.get("review") or {}
                if (info.get("parent") != step["from_release"]
                        or rv.candidate_artifact_sha256(published) != step["candidate_artifact_sha256"]
                        or record.get("decision") != "approve"
                        or not _review_matches(record, step, candidate)):
                    raise ValueError("campaign published release does not match its frozen review")
                step.update({"published": current, "active_release": current,
                             "awaiting_review": None, "human_decision": "approve", "ok": True})
            pool = [p for p in pool if p.get("candidate_id") != step["candidate_id"]]
            checkpoint["expected_active"] = current
        if checkpoint.get("expected_active") != rel.current_release_id():
            raise ValueError("campaign release lineage changed")

    def save(status, **extra):
        nonlocal checkpoint
        checkpoint = {"schema": "evolution-campaign-checkpoint-v2", "campaign_id": campaign_id,
            "contract": contract, "status": status, "steps": steps, "pool": pool,
            "expected_active": rel.current_release_id(),
            "live_budget": {name: os.environ[name] for name in (
                "RSI_ACTIVE_EXECUTION_PLAN_SHA256", "RSI_CALL_LEDGER_PATH",
                "RSI_CALL_LEDGER_MAX_LOGICAL", "RSI_CALL_LEDGER_MAX_PHYSICAL",
                "RSI_CALL_LEDGER_RESERVE_PI_PHYSICAL")
                if name in os.environ}, **extra}
        checkpoint["checkpoint_sha256"] = sha256_obj(checkpoint)
        write_json(checkpoint_path, checkpoint)

    if resume_campaign and checkpoint.get("status") in {"awaiting_review", "publication_pending"}:
        save("round_completed")

    for i in range(len(steps), int(rounds)):
        before = rel.current_release_id()
        if checkpoint.get("status") == "round_measured" and checkpoint.get("next_round") == i:
            report_path = checkpoint["round_report_path"]
            if sha256_file(report_path) != checkpoint["round_report_sha256"]:
                raise ValueError("completed round receipt changed")
            rep = read_json(report_path)
        else:
            save("round_in_progress", next_round=i)
            rejected_steps = [s for s in steps if s.get("human_decision") == "reject"
                              and s.get("from_release") == before]
            for rejected in rejected_steps:
                if sha256_file(rejected["review_path"]) != rejected["review_sha256"]:
                    raise ValueError("campaign human rejection receipt changed")

            def rejection_gate(candidate, _cfg, _state):
                matching = next((s for s in rejected_steps
                                 if s.get("proposal_sha256") == proposal_sha256(candidate)), None)
                return {"passed": matching is None,
                        "reason": "previous_human_rejection" if matching else "new_proposal",
                        "review_sha256": matching["review_sha256"] if matching else None}
            try:
                extra = {"candidate_pre_evaluation_gate": rejection_gate} if rejected_steps else {}
                rep = run_round(cfg, datasets=datasets, seed=seed, limit=limit,
                                seeds=list(seeds) if seeds else None, **extra)
            except BaseException as exc:
                save("failed", next_round=i, error_type=type(exc).__name__)
                raise
            save("round_measured", next_round=i, round_report_path=rep["report_path"],
                 round_report_sha256=sha256_file(rep["report_path"]))
        accepted = [c for c in rep.get("candidates", [])
                    if c.get("status") == "proposed_accept" and c.get("candidate_id")]

        # 入池（同一 candidate_id 只留一份）
        for c in accepted:
            if not any(p.get("candidate_id") == c.get("candidate_id") for p in pool):
                pool.append(dict(c, pool_round=i + 1))

        step: Dict[str, Any] = {
            "round": i + 1,
            "round_id": rep.get("round_id"),
            "from_release": before,
            "n_failures": rep.get("n_failures"),
            "failure_class_counts": rep.get("failure_class_counts"),
            "n_accepted_candidates": len(accepted),
            # 被拒/延后的候选也要留痕：进化史里"试过什么、为什么没过"和"发布过什么"同样重要
            "candidates": [
                {k: c.get(k) for k in ("candidate_id", "attributed_class", "target_artifact",
                                       "status", "n_failures", "delta_pp", "reason")}
                for c in rep.get("candidates", [])
            ],
        }
        if not pool:
            step.update({"published": None, "ok": False,
                         "note": "本轮没有拟接受候选，且历史候选池为空"})
            steps.append(step)
            save("round_completed")
            continue

        # 优先用**本轮度量**的候选；本轮无新候选时才回退到历史池（陈旧度量要写明）
        cur = [p for p in pool if int(p.get("pool_round", 0)) == i + 1]
        # 【M1/M2 回顾，严重 bug 修复】跨轮复用必须过"父指针对齐"这道闸。
        #
        # 事故：第 3 轮本轮无新候选，池里回退到第 1 轮度量于 `baseline-v0` 的候选，
        # 而 `rel.publish` 用的是候选自己记的 `parent_release_id` —— 于是新 release
        # 的父被设成 baseline-v0，**整条发布链静默回退**：第 1 轮发布的正文补丁
        # （protocol-troubleshooting 的 evidence 步）和第 2 轮的提示词一并丢失，
        # registry.yaml 的 path 变回 skills/main/*.md。表现是终评 Δ 全为 0，
        # 而 campaign 报告里每一轮都写着 published=ok。
        #
        # 所以：跨轮复用**只在父指针与当前 release 一致时**才允许；
        # 否则度量的基线已经不对了，发布它就是拿旧分数掩盖新状态。
        stale = [p for p in pool if int(p.get("pool_round", 0)) != i + 1
                 and str(p.get("parent_release_id") or "") != str(before)]
        usable_pool = [p for p in pool if p not in stale]
        chosen_from = cur or usable_pool
        if stale:
            step["pool_stale_skipped"] = [
                {"candidate_id": p.get("candidate_id"),
                 "measured_in_round": p.get("pool_round"),
                 "measured_against": p.get("parent_release_id") or "<unknown>",
                 "current_release": before,
                 "reason": "该候选度量于其它 release，跨轮复用会让发布链静默回退，已跳过"}
                for p in stale]
        if not chosen_from:
            step.update({"published": None, "ok": False,
                         "note": "本轮没有可用候选（历史池候选均度量于其它 release，已按父指针对齐规则跳过）"})
            steps.append(step)
            save("round_completed")
            continue
        c = _pick_best(chosen_from)
        if c is None:
            step.update({"published": None, "ok": True,
                         "note": "no_improving_candidate：没有正净覆盖候选"})
            steps.append(step)
            save("round_completed")
            continue
        step["selection_policy"] = ("逐题覆盖最大 + 回退最小（跨轮候选池；"
                                    "优先本轮度量，本轮无候选时回退到历史池）")
        step["selection_scores"] = {
            x.get("candidate_id"): {
                "improved": _task_support(x)["improved"],
                "regressed": _task_support(x)["regressed"],
                "mean_delta_pp": round(_mean_delta_pp(x), 4),
                "measured_in_round": x.get("pool_round"),
            } for x in chosen_from}
        step["pool_size"] = len(pool)
        step["selected_measured_in_round"] = c.get("pool_round")
        if not cur:
            step["selection_note"] = (
                f"本轮无新拟接受候选，回退到历史池（度量来自第 {c.get('pool_round')} 轮；"
                "此后状态已变化，该度量为陈旧值，仅作兜底）")
        cand = _load_candidate(c["candidate_id"])
        if cand is None:
            step.update({"published": None, "ok": False, "note": "候选文件缺失"})
            steps.append(step)
            save("round_completed")
            continue

        if cand.parent_release_id != before:
            raise ValueError("campaign selected candidate has a different parent release")
        step.update(_bind_review(cand, c.get("review_request") or str(rv.REVIEWS_ROOT / f"{cand.candidate_id}.request.json")))
        step["proposal_sha256"] = proposal_sha256(cand)

        if not approve:
            req_path = step["review_request_path"]
            step.update({"published": None, "ok": True, "awaiting_review": req_path,
                         "candidate_id": c["candidate_id"],
                         "note": "等待精确人工 review；拒绝或批准并发布后可继续下一轮"})
            steps.append(step)
            save("awaiting_review")
            break

        step.update({"candidate_id": cand.candidate_id, "awaiting_review": step["review_request_path"]})
        steps.append(step)
        save("publication_pending")
        res = rv.decide(c["candidate_id"], "approve",
                        reason=f"campaign 第 {i+1} 轮：链路压测下的模拟人工批准",
                        reviewer=reviewer)
        req = res["request"]
        new_rel = rel.publish(req["parent_release_id"], cand,
                              {"optimize": req.get("optimize_metrics", {}),
                               "gate": req.get("gate_metrics", {})})
        after = rel.current_release_id()
        # 已发布的候选从池里取出（它已被消费，不该再被重复发布）
        pool = [p for p in pool if p.get("candidate_id") != c["candidate_id"]]
        step.update({
            "candidate_id": c["candidate_id"],
            "artifact_type": cand.artifact_type,
            "attributed_class": c.get("attributed_class"),
            "delta_pp": c.get("delta_pp"),
            "task_support": _task_support(c),
            "published": new_rel,
            "active_release": after,
            "review_path": res["path"],
            "awaiting_review": None,
            "ok": after == new_rel,
        })
        save("round_completed")

    report: Dict[str, Any] = {
        "campaign_id": campaign_id,
        "checkpoint_path": str(checkpoint_path),
        "created_at": now_iso(),
        "config_fingerprint": cfg_fp,
        "datasets": list(datasets),
        "rounds_requested": int(rounds),
        "rounds_executed": len(steps),
        "rounds_published": sum(1 for s in steps if s.get("published")),
        "seeds": [int(s) for s in seeds] if seeds else [int(seed)],
        "approve_mode": "auto-sim（链路压测，非生产发布）" if approve else "manual",
        "start_release": steps[0]["from_release"] if steps else None,
        "final_release": rel.current_release_id(),
        "steps": steps,
    }
    if not (steps and steps[-1].get("awaiting_review")):
        save("completed")
    out = CAMPAIGN_ROOT / f"campaign-{campaign_id}.json"
    write_json(out, report)
    report["report_path"] = out.as_posix()
    return report
