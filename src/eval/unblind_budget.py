"""final_holdout 揭盲**预算账本**（Spec §418 的可执行版）。

## 起因（本次踩到的坑）
Spec §418 写的是 "`final_holdout` 基线封存后 + 最终各一次"，
但代码里只有 `unblind_state()` —— 它**统计**揭盲次数，却从不拒绝。
实测 suppqa 的非 forced 访问已经有 2 个不同协议（旧协议基线 + 新协议终评），
预算 2/2 用满；再跑一次不会报错、只会静默变成第 3 次揭盲。

序贯窥视（反复看同一 holdout）会系统性抬高假阳性率 —— Dwork et al. 2015
《The Reusable Holdout》(Science) 与工程化的 peek accounting 都把"每次查看"
当作**必须记账的支出**，预算耗尽则该 holdout **退役**、不再查询。
gepa 的范式（`api.py:286`，"配了却测不到就报错、绝不静默空转"）进一步要求：
违规必须是 **fail-loud**，不是警告。

## 本模块做什么
1. 账本文件 `artifacts/final_holdout/unblind_budget.json`：
   记录**启用时点的已用额度快照**、限额、以及所有**显式授权**（override）。
2. `budget_state()`：随时可读出某集"已用 / 限额 / 剩余 / 是否退役"。
3. `assert_within_budget()`：在任何封存访问**之前**调用 —— 超限直接 raise，
   唯一放行方式是显式 override（actor + reason 必填，写进账本可事后审计）。

为什么用"快照 + 新增"而不是直接对历史设限：账本启用时 suppqa 已经用满，
若直接对历史设限，系统会一启用就红、且无法区分"历史既成事实"与"新的违规"。
快照把责任边界划清楚：**不追溯历史，但此后每一次新增都算数**。
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from uuid import uuid4
from typing import Any, Dict, List, Optional

from ..util import artifacts_path, now_iso, read_json, write_json, sha256_obj

FINAL_ROOT = artifacts_path("final_holdout")
BUDGET_FILE = FINAL_ROOT / "unblind_budget.json"

# Spec §418：final_holdout 的访问 = 基线封存时一次 + 最终一次。
#   final_unblind   非 forced 的**不同协议**访问次数（真正的揭盲）
#   baseline_reseal forced 重封基线次数（同协议回放不算；换协议重封算）
DEFAULT_LIMITS: Dict[str, int] = {"final_unblind": 2, "baseline_reseal": 1}

POLICY_NOTE = (
    "final_unblind = 不同协议的非 forced 访问次数（Spec §418：基线封存后 + 最终各一次）；"
    "baseline_reseal = forced 重封基线次数。同协议回放不消耗额度；"
    "超出限额的访问必须显式 override（actor + reason），否则直接 raise。"
)


class UnblindBudgetExceeded(RuntimeError):
    """揭盲预算耗尽（或未授权重封）—— 静默违规在此变成硬失败。"""


@dataclass(frozen=True)
class FinalAccessGrant:
    """In-process capability issued only after the unblind budget check passes."""

    dataset: str
    protocol_fingerprint: str
    forced: bool
    decision: str
    batch_id: str = ""
    artifact_sha256: str = ""
    seeds: tuple[int, ...] = ()
    purpose: str = ""
    limit: Optional[int] = None


def _access_grant(dataset: str, fingerprint: str, forced: bool,
                  decision: str) -> FinalAccessGrant:
    return FinalAccessGrant(dataset, fingerprint, bool(forced), decision)


def _audit_log() -> List[Dict[str, Any]]:
    # 延迟导入：`final_eval` 会导入本模块，模块级互相 import 会成环。
    from .final_eval import audit_log

    return audit_log()


def _count(rows: List[Dict[str, Any]], dataset: str) -> Dict[str, Any]:
    """从审计日志统计某数据集的已用额度。

    两路都按**协议指纹去重**：一次"重封基线 / 揭盲"可能跑多个 seed，
    每个 seed 写一条日志，但它们属于同一次访问 —— 按条目数记会把
    "一次重封 3 个 seed"算成 3 次，凭空吃光预算。

    【fail-closed】指纹缺失（老日志没有该字段）的条目按**一次独立揭盲**计：
    判定不了协议，就无法证明它是"同协议回放"，那就按最坏情况算。
    （`unblind_state()` 会把这些条目整条跳过 —— 那是**描述性**统计，
    本函数是**约束性**统计，宁可少给额度，不能多给。）
    """
    fps: List[str] = []
    reseal_fps: List[str] = []
    for index, r in enumerate(rows):
        if r.get("dataset") != dataset:
            continue
        fp = (r.get("access_batch_id") or r.get("protocol_fingerprint")
              or f"<unknown-protocol:{index}>")
        if r.get("forced"):
            if fp not in reseal_fps:
                reseal_fps.append(fp)
            continue
        if fp not in fps:
            fps.append(fp)
    return {"final_unblind": len(fps), "baseline_reseal": len(reseal_fps),
            "fingerprints": fps, "reseal_fingerprints": reseal_fps}


def load_ledger() -> Optional[Dict[str, Any]]:
    return read_json(BUDGET_FILE) if BUDGET_FILE.exists() else None


def ensure_ledger(actor: str = "system", note: str = "") -> Dict[str, Any]:
    """建立（或读取）预算账本。

    首次调用 = "预算账本启用"，此时把各集**已经用掉的额度**快照下来：
    历史既成事实不追溯、不掩盖，但此后每一笔新增都受限额约束。
    """
    led = load_ledger()
    if led:
        return led
    rows = _audit_log()
    datasets = sorted({str(r.get("dataset")) for r in rows if r.get("dataset")})
    led = {
        "created_at": now_iso(),
        "actor": actor,
        "note": note or "预算账本启用：快照当时已用额度，此后新增访问受限额约束（不追溯历史）",
        "limits": dict(DEFAULT_LIMITS),
        "policy": POLICY_NOTE,
        "snapshot": {ds: _count(rows, ds) for ds in datasets},
        "overrides": [],
        "budget_file": str(BUDGET_FILE),
    }
    write_json(BUDGET_FILE, led)
    return led


def budget_state(dataset: str, fingerprint: Optional[str] = None,
                 ledger: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """某数据集当前的预算状态（只读，不写文件）。"""
    led = ledger or load_ledger()
    rows = _audit_log()
    rows = rows + [dict(row, access_batch_id=row["batch_id"])
                   for row in ((led or {}).get("reservations") or [])]
    cur = _count(rows, dataset)
    snap = ((led or {}).get("snapshot") or {}).get(dataset) or {
        "final_unblind": 0, "baseline_reseal": 0}
    limits = dict((led or {}).get("limits") or DEFAULT_LIMITS)
    new_fu = max(0, cur["final_unblind"] - int(snap.get("final_unblind", 0)))
    new_br = max(0, cur["baseline_reseal"] - int(snap.get("baseline_reseal", 0)))
    remaining = {"final_unblind": max(0, int(limits["final_unblind"]) - new_fu),
                 "baseline_reseal": max(0, int(limits["baseline_reseal"]) - new_br)}
    is_repeat = (fingerprint in cur["fingerprints"]) if fingerprint else None
    # 剩余额度按**总用量**算（不是"启用以来新增"）：
    # 账本不追溯历史责任，但历史用掉的额度就是没了 —— 快照只用来展示"启用后又花了多少"。
    remaining = {k: max(0, int(limits[k]) - int(cur[k])) for k in limits}
    return {
        "dataset": dataset,
        "adopted_at": (led or {}).get("created_at"),
        "used_total": cur,
        "snapshot_at_adoption": snap,
        "new_since_adoption": {"final_unblind": new_fu, "baseline_reseal": new_br},
        "limits": limits,
        "remaining": remaining,
        # 退役 = 新的揭盲额度已经没有了。退役后只允许同协议回放 / 显式 override。
        "retired": remaining["final_unblind"] <= 0,
        "is_repeat": is_repeat,
        "policy": POLICY_NOTE,
    }


def assert_within_budget(dataset: str, fingerprint: str, forced: bool = False,
                         actor: str = "evaluator", reason: str = "",
                         override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """封存访问**之前**的硬校验：超限直接 raise。

    放行的三种情形：
      1. 同协议回放（`is_repeat=True`）—— 确定性重跑不产生新信息，不消耗额度；
      2. 额度内 —— 正常记账；
      3. 显式 override —— 必须给出 actor + reason，且**写进账本**供事后审计。

    其余一律 `UnblindBudgetExceeded`。
    """
    led = ensure_ledger()
    st = budget_state(dataset, fingerprint, led)
    kind = "baseline_reseal" if forced else "final_unblind"
    seen = (st["used_total"].get("reseal_fingerprints", []) if forced
            else st["used_total"].get("fingerprints", []))

    # 同一次访问的另一个 seed（同一协议指纹已经记过账）→ 不重复消耗额度
    if fingerprint and fingerprint in seen:
        return {**st, "decision": "repeat_allowed", "kind": kind,
                "access_grant": _access_grant(dataset, fingerprint, forced, "repeat_allowed"),
                "note": "同协议回放（本次访问的其它 seed）：不消耗揭盲额度"}

    projected = int(st["used_total"][kind]) + 1
    limit = int(st["limits"][kind])
    head = {**st, "kind": kind, "projected": projected, "limit": limit}

    if projected <= limit:
        return {**head, "decision": "within_budget",
                "access_grant": _access_grant(dataset, fingerprint, forced, "within_budget"),
                "note": f"额度内（{projected}/{limit}）"}

    ov = override or {}
    ov_actor = str(ov.get("actor", "") or actor or "").strip()
    ov_reason = str(ov.get("reason", "") or "").strip()
    if not ov_actor or not ov_reason:
        what = "重封基线（forced）" if forced else "新的协议揭盲"
        raise UnblindBudgetExceeded(
            f"【揭盲预算耗尽】{dataset}：本次是一次{what}，"
            f"累计将达 {projected} 次，限额 {limit} 次"
            f"（总用量：非 forced 不同协议 {st['used_total']['final_unblind']} 次、"
            f"forced 重封 {st['used_total']['baseline_reseal']} 次）。"
            "Spec §418 规定 final_holdout 只允许『基线封存后 + 最终各一次』；"
            "反复查看同一 holdout 会系统性抬高假阳性率"
            "（Dwork et al. 2015《The Reusable Holdout》）。"
            "若确需继续，必须显式授权：传 override={actor, reason} 或命令行 "
            "--budget-override-reason <理由>（会写入账本与审计日志，可事后追责）。"
        )

    entry = {
        "at": now_iso(),
        "dataset": dataset,
        "kind": kind,
        "actor": ov_actor,
        "reason": ov_reason,
        "protocol_fingerprint": fingerprint,
        "projected": projected,
        "limit": limit,
    }
    led.setdefault("overrides", []).append(entry)
    write_json(BUDGET_FILE, led)
    return {**head, "decision": "override_granted",
            "access_grant": _access_grant(dataset, fingerprint, forced, "override_granted"),
            "override": entry,
            "note": f"超限但已显式授权（{projected}/{limit}），理由已入账"}


@contextmanager
def _ledger_lock():
    """Serialize reserve/claim writes; a stale lock fails closed for reconciliation."""
    BUDGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock = BUDGET_FILE.with_suffix(".lock")
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(now_iso())
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def artifact_identity(state) -> str:
    return sha256_obj({"release": state.release_id, "prompt": state.system_prompt,
                       "skills": state.registry.fingerprint(), "demos": state.task_demos})


def reserve_final_batch(dataset: str, fingerprint: str, state, seeds,
                        purpose: str, limit: Optional[int], *, forced: bool = False,
                        actor: str = "evaluator", reason: str = "", override=None) -> dict:
    """Reserve before protected access; each invocation consumes one batch.

    Multiple declared seeds share a batch. A second physical batch, even for
    identical artifacts/protocol, consumes another exposure; it is not replay.
    """
    from ..data.split import load_manifest, SEALED_DIR
    from ..util import sha256_file
    seed_tuple = tuple(int(seed) for seed in seeds)
    if not seed_tuple or len(seed_tuple) != len(set(seed_tuple)):
        raise ValueError("final batch requires distinct declared seeds")
    artifact_sha = artifact_identity(state)
    identity = {"dataset": dataset, "protocol": fingerprint, "artifact": artifact_sha,
                "purpose": purpose, "seeds": seed_tuple, "limit": limit,
                "cohort": sha256_obj(load_manifest(dataset).get("splits", {}).get("final_holdout", {})),
                "sealed_sha256": sha256_file(SEALED_DIR / f"{dataset}.final_holdout.jsonl")}
    batch_id = sha256_obj(identity) + "-" + uuid4().hex
    with _ledger_lock():
        decision = assert_within_budget(dataset, batch_id, forced=forced,
            actor=actor, reason=reason, override=override)
        ledger = load_ledger()
        ledger.setdefault("reservations", []).append({
            "batch_id": batch_id, "dataset": dataset, "forced": forced,
            "identity": identity, "at": now_iso(), "status": "reserved",
            "per_seed": {str(seed): "reserved" for seed in seed_tuple}})
        write_json(BUDGET_FILE, ledger)
    grant = FinalAccessGrant(dataset, fingerprint, forced, decision["decision"],
        batch_id, artifact_sha, seed_tuple, purpose, limit)
    return {**decision, "access_grant": grant, "access_batch_id": batch_id}


def claim_final_seed(grant: FinalAccessGrant, state, seed: int, purpose: str,
                     limit: Optional[int]) -> None:
    if (not grant.batch_id or grant.artifact_sha256 != artifact_identity(state)
            or seed not in grant.seeds or purpose != grant.purpose or limit != grant.limit):
        raise PermissionError("final grant does not bind this artifact/seed/purpose/cohort scope")
    with _ledger_lock():
        ledger = load_ledger() or {}
        row = next((r for r in ledger.get("reservations", [])
                    if r.get("batch_id") == grant.batch_id), None)
        if (row is None or row.get("dataset") != grant.dataset
                or row.get("identity", {}).get("artifact") != grant.artifact_sha256
                or row.get("per_seed", {}).get(str(seed)) != "reserved"):
            raise PermissionError("final batch reservation missing or seed already consumed")
        row["per_seed"][str(seed)] = "started_outcome_unknown"
        row["status"] = "exposed_or_unknown"
        write_json(BUDGET_FILE, ledger)


def finish_final_seed(grant: FinalAccessGrant, seed: int, *, succeeded: bool) -> None:
    with _ledger_lock():
        ledger = load_ledger()
        row = next(r for r in ledger["reservations"] if r["batch_id"] == grant.batch_id)
        row["per_seed"][str(seed)] = "completed" if succeeded else "failed_exposure_unknown"
        row["status"] = ("completed" if all(v == "completed" for v in row["per_seed"].values())
                         else "exposed_or_unknown")
        write_json(BUDGET_FILE, ledger)


def self_test() -> Dict[str, Any]:
    """自检：证明限额真的会拦（而不是只记账）。"""
    from . import final_eval

    checks: List[Dict[str, Any]] = []

    def _ck(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    saved_log = final_eval.audit_log
    saved_file = BUDGET_FILE
    try:
        # 用临时账本文件，避免自检污染真实 artifacts
        import tempfile
        from pathlib import Path

        tmp = Path(tempfile.mkdtemp(prefix="rsi-budget-selftest-"))
        globals()["BUDGET_FILE"] = tmp / "unblind_budget.json"

        rows = [
            {"dataset": "ds", "forced": False, "protocol_fingerprint": "aaa"},
            {"dataset": "ds", "forced": False, "protocol_fingerprint": "bbb"},
            # 历史已经重封过一次基线（forced）：这才是"Spec 只封一次"的那一次
            {"dataset": "ds", "forced": True, "protocol_fingerprint": "aaa"},
        ]
        final_eval.audit_log = lambda: rows
        led = ensure_ledger(actor="selftest")
        _ck("snapshot_records_history",
            led["snapshot"]["ds"]["final_unblind"] == 2,
            f"snapshot={led['snapshot']['ds']}")

        st = budget_state("ds", "aaa", led)
        _ck("is_repeat_detected", st["is_repeat"] is True, "aaa 已出现过 → 回放")
        _ck("retired_when_budget_used_up", st["retired"] is True,
            f"remaining={st['remaining']}")

        # 同协议回放放行
        d = assert_within_budget("ds", "aaa", actor="t", reason="r")
        _ck("repeat_allowed", d["decision"] == "repeat_allowed", d.get("note", ""))

        # 新协议 → 超限 → raise
        try:
            assert_within_budget("ds", "ccc", actor="t", reason="r")
            _ck("new_protocol_raises", False, "超限时未抛错")
        except UnblindBudgetExceeded as exc:
            _ck("new_protocol_raises", "揭盲预算耗尽" in str(exc), str(exc)[:60])

        # 给授权 → 放行并记账
        d2 = assert_within_budget("ds", "ccc", actor="pi", reason="协议升级后重封基线",
                                  override={"actor": "pi", "reason": "协议升级后重封基线"})
        _ck("override_granted_and_recorded",
            d2["decision"] == "override_granted"
            and len(load_ledger()["overrides"]) == 1,
            "授权已写入账本 overrides")

        # 授权必须带理由：只给 actor 不给 reason → 仍然 raise
        try:
            assert_within_budget("ds", "ddd", actor="pi", reason="",
                                 override={"actor": "pi"})
            _ck("override_requires_reason", False, "缺理由仍被放行")
        except UnblindBudgetExceeded:
            _ck("override_requires_reason", True, "缺理由 → raise")

        # forced 重封也要授权（Spec 未定义 forced 豁免 → 口径统一为"也需授权"）
        try:
            assert_within_budget("ds", "eee", forced=True, actor="pi", reason="r")
            _ck("forced_reseal_needs_grant", False, "forced 未授权仍被放行")
        except UnblindBudgetExceeded as exc:
            _ck("forced_reseal_needs_grant", "重封基线" in str(exc), str(exc)[:60])
    finally:
        final_eval.audit_log = saved_log
        globals()["BUDGET_FILE"] = saved_file

    return {"ok": all(c["passed"] for c in checks), "checks": checks}
