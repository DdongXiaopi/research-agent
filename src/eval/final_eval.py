"""final_holdout 基线终评（Spec §6.1 铁律 3）。

铁律 3 原文：*"基线阶段也跑一次 final（记录基线值），此后封存至 M4。"*

因此本模块做三件事：
1. 只允许**跑一次**（重复调用直接拒绝，除非显式 --force 并留审计）
2. 全程走 evaluator 入口：`RSI_ALLOW_FINAL=1` 只在本函数内临时置位，退出即恢复
3. 产物落在 `artifacts/final_holdout/`（**不在** `artifacts/runs/` 下），
   这样 `rsi.py verify` 的泄漏扫描天然不会碰到它，也不需要给检查开例外

每次解锁都往 `artifacts/final_access.log` 追加一条审计记录：谁、何时、跑了哪些数据集。
M4 解封前的任何"多跑一次"都会被这条日志暴露出来。
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import config_fingerprint
from ..data.freeze import require_verified
from ..retrieval.gateway import DEFAULT_DOC_ID_SCHEME, RETRIEVAL_RULE_VERSION
from ..runtime.runner import ArtifactState
from ..runtime.trace import run_dir
from ..util import append_jsonl, now_iso, artifacts_path, path, read_json, read_jsonl, sha256_obj, write_json
from .harness import evaluate
from .metrics import derive_triple
from .options import INSUFFICIENT_OPTION_DEFAULT
from .verdict import DEFAULT_PRIMARY_AXIS, pair_triple
from .stats import holm_bonferroni
from .unblind_budget import assert_within_budget, budget_state, ensure_ledger

FINAL_ROOT = artifacts_path("final_holdout")
BASELINE_FILE = FINAL_ROOT / "baseline.json"
AUDIT_LOG = artifacts_path("final_access.log")


class FinalHoldoutError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 【N3】协议指纹：让"final 到底被揭盲了几次"这件事可以**自动判定**
# ---------------------------------------------------------------------------
#
# 背景（本次踩到的坑）：审计日志只记了 dataset / reason / forced，
# 于是"这次是同协议重复重跑，还是换了协议的新一次揭盲"只能靠人回忆 ——
# 本次就差点记错：suppqa 日志层面实际有 2 次非 forced 访问，口头报成了 1 次。
# 序贯窥视会系统性抬高假阳性率，揭盲次数是这个项目最不能含糊的数字。
#
# 做法：把"哪些判据构成了本轮评测协议"压成一个指纹写进日志。
# 指纹相同 ⇒ 同协议重复（确定性回放下结果必然一致，不计入揭盲）；
# 指纹不同 ⇒ 换协议重跑，是一次**新的**揭盲，必须计入。
#
# 构成协议的四个判据（任一变化都必须递增对应版本号）：
#   attribution  归因判据（N1：gold_* → evidence_*）
#   retrieval    检索排序判据（A1：is_gold 置顶 → 内容派生打分）
#   scoring      评分/弃权判据（M4+ #3：raw_scores==0 → 未被字面陈述）
#   evidence     证据呈现判据（N4：硬截断 → 段落边界截断 + 显式声明省略量）
#   config       configs/experiment.yaml 全量指纹
#   （P8 的分块判据并入 retrieval —— 见 `retrieval_rule_version`）
def retrieval_rule_version(cfg: Dict[str, Any]) -> str:
    """检索层判据版本 —— 【P8】启用证据分块时追加分块判据。

    **为什么用"追加"而不是直接改常量**：同一个数据集的揭盲次数是按协议指纹
    去重计数的（`unblind_state`）。若把常量从 v2 改成 v3，那么"分块关闭"的
    运行也会得到一个新指纹，明明是同一个协议却被算成新的一次揭盲。
    追加写法保证：**分块关闭时字符串逐字等于旧值**，历史指纹不受影响；
    只有真的启用分块才产生新指纹。
    """
    if (cfg.get("runtime") or {}).get("input_mode") == "task_input":
        version = "task-input-v3"
        citation_cfg = ((cfg.get("evaluation") or {}).get("semantic_citation") or {})
        if citation_cfg.get("enabled"):
            if citation_cfg.get("mode") == "proposition":
                version += "+task-input-proposition-grounding-v4"
            else:
                version += "+task-input-answer-grounding-v3"
            if citation_cfg.get("check_top3"):
                version += "+task-input-evidence-consistency-v2"
        return version

    from ..retrieval.chunking import CHUNK_RULE_VERSION, chunking_enabled
    from ..retrieval.gateway import RETRIEVAL_RULE_VERSION

    version = RETRIEVAL_RULE_VERSION
    if chunking_enabled(cfg):
        version += f"+{CHUNK_RULE_VERSION}"
    if bool((cfg.get("retrieval", {}).get("query_control") or {}).get("enabled")):
        mode = (cfg.get("retrieval", {}).get("query_control") or {}).get("mode", "template")
        version += {"model": "+query-model-v1",
                    "model_sources": "+query-model-sources-v1",
                    "frozen": "+query-frozen-v1"}.get(mode, "+query-template-v1")
    if (cfg.get("skills", {}) or {}).get("routing_mode") == "model":
        version += "+skill-model-route-v1"
    if (cfg.get("retrieval", {}) or {}).get("pubmed_passage_mode") == "query_window":
        version += "+pubmed-query-window-v2"
    if (cfg.get("retrieval", {}) or {}).get("anchor_question_terms"):
        version += "+question-term-anchor-v1"
    if (cfg.get("retrieval", {}) or {}).get("pubmed_source_queries"):
        version += "+pubmed-source-subset-v1"
    citation_cfg = ((cfg.get("evaluation") or {}).get("semantic_citation") or {})
    if citation_cfg.get("enabled"):
        version += ("+citation-proposition-v1" if citation_cfg.get("mode") == "proposition"
                    else "+citation-judge-v3")
        if citation_cfg.get("check_top3"):
            version += "+evidence-top3-consistency-v2"
    return version


def normalized_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """指纹用的配置副本：**未启用的分块配置会被剔除**。

    `chunking.enabled=false` 时里面写什么（`budget_chars: 9000` 之类）对运行
    **零影响**。若把它算进指纹，那么"把说明性的分块配置写进主配置"这件纯粹
    的文档动作，会让封存集看起来换了一套协议 —— 揭盲账本因此多出一次假的新揭盲。
    所以：没启用就当它不存在。
    """
    from ..retrieval.chunking import chunking_enabled

    out = copy.deepcopy(cfg)
    ret = out.get("retrieval")
    if isinstance(ret, dict) and not chunking_enabled(cfg) and "chunking" in ret:
        ret = dict(ret)
        ret.pop("chunking", None)
        out["retrieval"] = ret
    ret = out.get("retrieval")
    if isinstance(ret, dict) and not bool((ret.get("query_control") or {}).get("enabled")):
        ret.pop("query_control", None)
    return out


def protocol_fingerprint(cfg: Dict[str, Any]) -> str:
    """本轮评测协议的指纹（16 位短哈希）。"""
    from ..evolution.attribution import ATTRIBUTION_RULE_VERSION
    from ..runtime.provider import EVIDENCE_RULE_VERSION, SCORING_RULE_VERSION

    return sha256_obj({
        # Task-input/Pi source files are part of the frozen protocol. Reuse the
        # same fingerprint as experiment plans so source drift cannot masquerade
        # as an already-authorized repeat in the unblind ledger.
        "config": config_fingerprint(normalized_config(cfg)),
        "attribution": ATTRIBUTION_RULE_VERSION,
        "retrieval": retrieval_rule_version(cfg),
        "scoring": SCORING_RULE_VERSION,
        "evidence": EVIDENCE_RULE_VERSION,
    })[:16]


def unblind_state(dataset: str, fingerprint: str) -> Dict[str, Any]:
    """在**写入日志之前**判定：本次访问对该 dataset 属于第几次揭盲。

    `is_repeat=True` 表示同协议重复（不计入揭盲次数）；
    `unblind_index` 是"第几个不同协议"（即真正的揭盲序号）。
    """
    seen: List[str] = []
    for r in audit_log():
        if r.get("dataset") != dataset or r.get("forced"):
            continue
        fp = r.get("protocol_fingerprint")
        if fp and fp not in seen:
            seen.append(fp)
    if fingerprint in seen:
        return {"is_repeat": True, "unblind_index": seen.index(fingerprint) + 1,
                "distinct_protocols": len(seen)}
    return {"is_repeat": False, "unblind_index": len(seen) + 1,
            "distinct_protocols": len(seen) + 1}


def baseline_exists() -> bool:
    return BASELINE_FILE.exists()


def load_baseline() -> Dict[str, Any]:
    if not BASELINE_FILE.exists():
        raise FinalHoldoutError("final 基线尚未跑过")
    return read_json(BASELINE_FILE)


def audit_log() -> List[Dict[str, Any]]:
    return read_jsonl(AUDIT_LOG)


def _run_final_seeds(cfg: Dict[str, Any], state: ArtifactState, dataset: str,
                     seeds: Sequence[int], run_tag: str, limit: Optional[int],
                     actor: str, reason: str, forced: bool,
                     budget_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """在一个数据集上跑若干 seed 的 final_holdout，返回 {seed: {run_id, metrics, per_task}}。

    每次解锁都写审计日志。RSI_ALLOW_FINAL 只在函数内临时置位，退出即恢复。

    【P8 揭盲预算 2026-09-26】每次访问**之前**先过预算账本
    （`unblind_budget.assert_within_budget`）：超限直接 raise，
    唯一放行方式是显式 override（actor + reason 入账）。在此之前
    `unblind_state()` 只统计不拒绝 —— 超预算是静默违规。

    【P7 封存隔离 2026-09-26】同时临时置位 `RSI_HTTP_NO_CACHE=1`：
    封存题的 passage 标识符里含 task_id（如 `cand::suppqa-037::1`），模型回引时会把它
    写进响应正文；provider 默认把**原始响应**落盘到 `artifacts/http_cache`，
    于是封存 ID 原样出现在"进化路径可读"的共享缓存里（verify 的泄漏扫描已因此变红：
    命中 suppqa-037/038/039/046/058、cs-095）。
    封存评估本就该每次现跑（不复用、也不需要被 dev 侧命中），故禁用其缓存落盘。
    """
    prev_env = os.environ.get("RSI_ALLOW_FINAL")
    prev_cache = os.environ.get("RSI_HTTP_NO_CACHE")
    os.environ["RSI_ALLOW_FINAL"] = "1"
    os.environ["RSI_HTTP_NO_CACHE"] = "1"
    per_seed: Dict[str, Any] = {}
    try:
        from .unblind_budget import reserve_final_batch, finish_final_seed
        budget = reserve_final_batch(dataset, protocol_fingerprint(cfg), state,
            seeds, run_tag, limit, forced=bool(forced), actor=actor,
            reason=reason, override=budget_override)
        for sd in seeds:
            # 【P8】先过预算再跑：超预算的评估本身就不该发生（跑完再拒 = 白花钱 + 已窥视）
            try:
                res = evaluate(cfg, state, dataset, "final_holdout", seed=int(sd), limit=limit,
                               run_tag=run_tag, write_per_task=True,
                               allow_final=True, final_access_grant=budget["access_grant"],
                               cfg_fp=sha256_obj(cfg), out_root=Path(FINAL_ROOT))
            except BaseException:
                finish_final_seed(budget["access_grant"], int(sd), succeeded=False)
                raise
            finish_final_seed(budget["access_grant"], int(sd), succeeded=True)
            # EvaluationResult 不保留逐题结果（final 的产物按设计留在 final_holdout/ 下），
            # 这里直接从落盘的 per_task.jsonl 读回，用于与基线做题级配对。
            per_task_path = run_dir(Path(FINAL_ROOT), res.run_id) / "per_task.jsonl"
            # 【决策 2】三指标口径需要"是否弃权 / 语料里有没有答案 / 有没有引用"，
            # 只留 correct 的话 accuracy 之外的两个轴事后补不回来（封存集不能重跑）。
            per_task = [{
                "task_id": r["task_id"],
                "correct": bool(r["correct"]),
                "abstained": int(r.get("answer_index", -1)) < 0,
                "gold_in_corpus": bool(r.get("gold_in_corpus", True)),
                "cited": list(r.get("cited", []) or []),
            } for r in read_jsonl(per_task_path)]
            # 【N3】先判揭盲状态再写日志，否则本次会把自己算进去
            fp = protocol_fingerprint(cfg)
            ub = unblind_state(dataset, fp)
            append_jsonl(AUDIT_LOG, {
                "at": now_iso(),
                "actor": actor,
                "dataset": dataset,
                "seed": int(sd),
                "run_id": res.run_id,
                "n_tasks": res.n_tasks,
                "reason": reason,
                "forced": bool(forced),
                "protocol_fingerprint": fp,
                "access_batch_id": budget["access_batch_id"],
                "is_repeat": ub["is_repeat"],
                "unblind_index": ub["unblind_index"],
                # 【P8】本次访问的预算判定（额度内 / 同协议回放 / 显式授权）
                "budget_decision": budget.get("decision"),
                "budget_override": budget.get("override"),
            })
            per_seed[str(sd)] = {
                "run_id": res.run_id,
                "n_tasks": res.n_tasks,
                "metrics": res.metric_values,
                "per_task": per_task,
            }
    finally:
        if prev_env is None:
            os.environ.pop("RSI_ALLOW_FINAL", None)
        else:
            os.environ["RSI_ALLOW_FINAL"] = prev_env
        # 【P7】恢复缓存开关：只有封存评估期间禁用，退出后 dev/optimize 侧行为不变
        if prev_cache is None:
            os.environ.pop("RSI_HTTP_NO_CACHE", None)
        else:
            os.environ["RSI_HTTP_NO_CACHE"] = prev_cache
    return per_seed


def run_final_baseline(cfg: Dict[str, Any], state: ArtifactState, datasets: Sequence[str],
                       seed: int = 0, limit: Optional[int] = None,
                       force: bool = False, actor: str = "evaluator",
                       reason: str = "baseline 封存前跑一次 final（Spec §6.1 铁律 3）",
                       seeds: Optional[Sequence[int]] = None,
                       budget_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if baseline_exists() and not force:
        prev = load_baseline()
        raise FinalHoldoutError(
            "final_holdout 基线已跑过一次，按 Spec 铁律 3 此后封存至 M4。"
            f"上次：{prev.get('created_at')}（release {prev.get('release_id')}）。"
            "确需重跑请显式加 --force 并说明理由（会写入审计日志）。"
        )

    require_verified(list(datasets))     # 冻结校验不通过就不允许碰 final

    seed_list = [int(s) for s in (seeds or [seed])]
    results: Dict[str, Any] = {}
    for ds in datasets:
        results[ds] = {
            "n_tasks": None,
            "runs": _run_final_seeds(cfg, state, ds, seed_list, "final-baseline", limit,
                                     actor, reason, force, budget_override),
        }
        first = results[ds]["runs"][str(seed_list[0])]
        results[ds]["n_tasks"] = first["n_tasks"]

    report: Dict[str, Any] = {
        "created_at": now_iso(),
        "release_id": state.release_id,
        "system_prompt_sha256": sha256_obj(state.system_prompt),
        "skills_fingerprint": state.registry.fingerprint(),
        "task_demos_sha256": sha256_obj(state.task_demos),
        "seed": seed_list[0],
        "seeds": seed_list,
        "actor": actor,
        "reason": reason,
        # Keep baseline and final evaluation on one protocol definition.  In
        # particular, HTTP/task-input baselines must bind the runtime source
        # fingerprint; a hand-copied field list can silently omit it.
        "protocol": {**current_protocol(cfg), "config_sha256": sha256_obj(cfg)},
        "datasets": results,
        # 【P8】本次访问在预算账本里的位置（启用后每一次封存访问都必须留下这一笔）
        "unblind_budget": {ds: budget_state(ds) for ds in datasets},
        "policy": {
            "next_unlock_milestone": "M4",
            "note": "此后 final_holdout 封存，进化器任何路径不得读取",
        },
    }
    write_json(BASELINE_FILE, report)
    return report


# --------------------------------------------------------------------------- #
# M4：终评（与基线配对）
# --------------------------------------------------------------------------- #

ASSESSMENT_FILE = FINAL_ROOT / "assessment.json"


class ProtocolMismatch(RuntimeError):
    """终评与基线不在同一协议下 —— 数字不可比。"""


def current_protocol(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """当前配置对应的协议字段（单一真相源）。

    **为什么要抽成函数**：此前 `cur` 内联在 `check_protocol` 里，而测试
    （`tests/integration/test_final_assessment.py:_protocol_of`）手工抄了一份字段列表。
    每新增一个协议字段，抄的那份就会漏（2026-09-27 新增 doc_id_scheme / primary_axis
    时正是这样翻车的）⇒ 产生"同配置也判不一致"的**假失败**。
    现在代码与测试都调用本函数，字段不可能再漂移。
    """
    protocol = {
        "noise": cfg.get("runtime", {}).get("fake_noise", 0.0),
        "top_k": cfg.get("retrieval", {}).get("top_k"),
        "recall_min_score": cfg.get("skills", {}).get("recall_min_score"),
        "behavior_channels": (cfg.get("evolution", {}).get("behavior_channels", {})
                              .get("enabled", None)),
        # 【协议升级 2026-09-26】语料版本 / 注入选项属于协议，变化即不可比
        "corpus_version": cfg.get("retrieval", {}).get("corpus_version"),
        "insufficient_option": bool(cfg.get("evaluation", {}).get(
            "insufficient_option", INSUFFICIENT_OPTION_DEFAULT)),
        # 【P6/P7】passage 标识符方案属于协议：它决定模型看到的 passage 标签
        # （`cand::suppqa-037::1` vs `cand::1`）。换了却不重跑封存基线 → Δ 不可比。
        "doc_id_scheme": cfg.get("retrieval", {}).get("doc_id_scheme")
        or DEFAULT_DOC_ID_SCHEME,
        # 【P8】证据分块属于协议：关着跑出来的 Δ 与开着跑出来的不可比
        "chunking": retrieval_rule_version(cfg),
        # 【D2 判据轴 2026-09-27】判定轴属于协议：同一个 Δ 在 accuracy 轴和
        # answered_precision 轴下能得出相反的启用结论（P8 就卡在这里），
        # 换了轴却不重跑基线 ⇒ 结论不可比。
        # 注意 `main_metric`（选拔轴）**不在此列**：它只影响 round.py 选哪个候选，
        # 不改变单次评测产出的数字，换它不需要重跑基线。
        "primary_axis": (cfg.get("evaluation", {}).get("primary_axis")
                         or DEFAULT_PRIMARY_AXIS),
        "primary_axis_tolerance_pp": float(
            cfg.get("evaluation", {}).get("primary_axis_tolerance_pp", 0.0) or 0.0),
    }
    runtime = cfg.get("runtime") or {}
    if (str(runtime.get("provider", "fake")).lower() != "fake"
            or str(runtime.get("input_mode", "")).lower() == "task_input"):
        # No historical default is safe here: a baseline that did not record
        # its runtime/evaluator source identity cannot prove paired comparability.
        protocol["runtime_source_sha256"] = config_fingerprint(normalized_config(cfg))
    return protocol


def check_protocol(cfg: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """终评前校验：夹具/配置必须与基线一致，否则 Δ 不可解释。

    M3.1 的教训：夹具在基线跑完之后升级了两次（evidence 字面判据、rerank 答案承载），
    旧基线与新终评混在一起会得出无法解释的数字。Spec §14 要求"同一协议下比较"。
    """
    base_proto = baseline.get("protocol") or {}
    cur = current_protocol(cfg)
    # 基线是在某个字段**尚未引入**的年代跑的（M2 旧基线、2026-09-26 前的语料版本等），
    # 缺失字段按"当时生效值"补全再比 —— 否则每新增一个协议字段都会误报不一致。
    # 这里只对有明确历史默认值的字段补全，其余仍按 <missing> 处理（不一致即拦）。
    base_defaults = {"doc_id_scheme": DEFAULT_DOC_ID_SCHEME,
                     "chunking": RETRIEVAL_RULE_VERSION,
                     "primary_axis": DEFAULT_PRIMARY_AXIS,
                     "primary_axis_tolerance_pp": 0.0}
    diffs = {k: {"baseline": base_proto.get(k, base_defaults.get(k, "<missing>")),
                 "now": cur[k]}
             for k in cur
             if base_proto.get(k, base_defaults.get(k, "<missing>")) != cur[k]}
    return {"ok": not diffs, "diffs": diffs,
            "config_sha256": sha256_obj(cfg),
            "baseline_config_sha256": base_proto.get("config_sha256"),
            "note": ("基线协议字段缺失（M2 旧格式）时按不一致处理 —— 请先重跑基线"
                     if "config_sha256" not in base_proto else "")}


def pair_seed(base_per_task: Sequence[Dict[str, Any]], cur_per_task: Sequence[Dict[str, Any]],
              B: int = 10000, rng_seed: int = 20260924) -> Dict[str, Any]:
    """题级配对：两个 run 的逐题正确率 → Δ / 95% CI / P(Δ>0) / 回归-提升题。

    抽成纯函数是为了可单测：终评本身会写 `assessment.json`（不可变产物），
    测试不应该碰它。
    """
    from .stats import mcnemar_exact, paired_bootstrap_binary

    b_map = {r["task_id"]: bool(r["correct"]) for r in base_per_task}
    c_map = {r["task_id"]: bool(r["correct"]) for r in cur_per_task}
    if (len(b_map) != len(base_per_task) or len(c_map) != len(cur_per_task)
            or not all(b_map) or not all(c_map)):
        raise FinalHoldoutError("配对结果包含重复或空 task ID")
    if set(b_map) != set(c_map):
        raise FinalHoldoutError("配对结果的任务集合必须完全相同")
    ids = sorted(b_map)
    if not ids:
        raise FinalHoldoutError("两个 run 没有可配对的题目")
    ci = paired_bootstrap_binary([b_map[i] for i in ids], [c_map[i] for i in ids],
                                 B=B, rng_seed=rng_seed)
    regressed = sorted(i for i in ids if b_map[i] and not c_map[i])
    improved = sorted(i for i in ids if not b_map[i] and c_map[i])
    # 【B1】小样本下把**精确 McNemar** 作为主检验，bootstrap 降为辅证：
    # n=18 时百分位 bootstrap 的端点只能是 5.56pp 的整数倍，尾部不可靠。
    mc = mcnemar_exact([float(b_map[i]) for i in ids], [float(c_map[i]) for i in ids])
    return {
        "estimand": "accuracy",
        "n_paired": len(ids),
        "baseline_accuracy": round(sum(b_map[i] for i in ids) / len(ids), 6),
        "final_accuracy": round(sum(c_map[i] for i in ids) / len(ids), 6),
        "delta_pp": ci["delta_pp"],
        "ci95": [ci["ci_low_pp"], ci["ci_high_pp"]],
        "p_delta_gt_0": ci["p_delta_gt_0"],
        "primary_test": mc,
        "n_regressed": len(regressed),
        "n_improved": len(improved),
        "regressed_task_ids": regressed,
        "improved_task_ids": improved,
    }


def paired_estimands(base_rows, candidate_rows, B=10000, rng_seed=20260924):
    """Pair-resample whole tasks and recompute each numerator/denominator."""
    import random
    if not isinstance(B, int) or isinstance(B, bool) or not 1 <= B <= 100000:
        raise ValueError("bootstrap B must be an integer in [1, 100000]")
    before, after = ({row["task_id"]: row for row in rows}
                     for rows in (base_rows, candidate_rows))
    if (not before or set(before) != set(after) or len(before) != len(base_rows)
            or len(after) != len(candidate_rows)):
        raise FinalHoldoutError("estimand bootstrap requires identical unique task scope")
    pairs = []
    for task_id in sorted(before):
        b, c = before[task_id], after[task_id]
        if any(not isinstance(row.get("abstained"), bool) for row in (b, c)):
            raise FinalHoldoutError("precision inference requires explicit abstention outcomes")
        pairs.append((int(bool(b["correct"])), int(bool(c["correct"])),
                      int(not b["abstained"]), int(not c["abstained"])))
    def values(sample):
        bc, cc, ba, ca = map(sum, zip(*sample))
        n = len(sample)
        return {"accuracy": (bc / n, cc / n), "coverage": (ba / n, ca / n),
                "answered_precision": (bc / ba if ba else None, cc / ca if ca else None)}
    point = values(pairs)
    draws = {metric: [] for metric in point}
    rng = random.Random(rng_seed)
    for _ in range(B):
        sampled = values(rng.choices(pairs, k=len(pairs)))
        for metric, (b, c) in sampled.items():
            if b is not None and c is not None:
                draws[metric].append((c - b) * 100)
    result = {}
    for metric, (b, c) in point.items():
        ordered = sorted(draws[metric])
        undefined = B - len(ordered)
        # Undefined precision draws must be disclosed, not silently conditioned away.
        interval = None
        if ordered and undefined / B <= .025 and b is not None and c is not None:
            interval = [ordered[int(.025 * (len(ordered) - 1))],
                        ordered[int(.975 * (len(ordered) - 1))]]
        result[metric] = {"estimand": metric, "baseline": b, "candidate": c,
            "delta_pp": (c - b) * 100 if b is not None and c is not None else None,
            "ci95_pp": interval, "undefined_resamples": undefined, "resamples": B,
            "interpretation": "paired task sampling uncertainty; not a posterior or provider RNG variance"}
    return result


def _triple_of_run(rows: Sequence[Dict[str, Any]],
                   metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """一次运行的三指标（accuracy / coverage / answered_precision）。

    优先用**题级**信息算（本次运行新落盘的 per_task 带 `abstained`）；
    历史产物只存了 accuracy + 弃权率，则退回反推（`derive_triple`），
    并在 `source` 里标明 —— 反推值不能当成题级精度用。
    """
    m = dict(metrics or {})
    if rows and "abstained" in rows[0]:
        n = len(rows)
        ans = [r for r in rows if not r.get("abstained")]
        cov = len(ans) / n if n else 0.0
        prec = (sum(1 for r in ans if r.get("correct")) / len(ans)) if ans else 0.0
        return {"accuracy": sum(1 for r in rows if r.get("correct")) / n if n else 0.0,
                "coverage": cov, "answered_precision": prec,
                "abstain_rate": 1.0 - cov,
                "unsupported_answer_rate": m.get("unsupported_answer_rate"),
                "source": "per_task"}
    d = derive_triple(m.get("accuracy"), m.get("abstain_rate"))
    return {"accuracy": m.get("accuracy"), "coverage": d["coverage"],
            "answered_precision": d["precision"],
            "abstain_rate": m.get("abstain_rate"),
            "unsupported_answer_rate": m.get("unsupported_answer_rate"),
            "source": "derived"}


def _seed_variance(per_seed: Dict[str, Any], seed_list: Sequence[int]) -> Dict[str, Any]:
    """多 seed 到底提供了多少方差？（【M1/M2 回顾】新增）

    离线夹具里 `fake_noise = 0`，`provider._jitter` 在 `noise <= 0` 时直接 return，
    于是 seed 对结果**零影响** —— 三个 seed 的 Δ / CI / P 会完全相同。
    这时"≥3 seed + 配对 bootstrap CI"只是**机械满足规范**，统计上是空的：
    bootstrap 重抽的是**题**，不是**运行**，它测的是"题目样本带来的不确定性"，
    不是"运行随机性"。两者不能混为一谈。

    依据（asta-bench `scripts/inspect_score_with_task.py:35-36` 原文）：
      "All other tasks … are deterministic, so their copied .eval scores
       should be preserved as-is"
    即：确定性任务不需要重复跑；同理，确定性夹具下多 seed 不产生新信息。
    deep_research_bench `utils/stat.py` 干脆不报 CI；ScholarQABench 用 ICC
    报**评分者一致性** —— 真正的方差来源是 judge / 采样，不是 seed。

    只有 1 个 seed 时不作判断（`zero_variance` 取 False）—— "只跑一次结果当然一致"，
    那不是证据。要声称零方差必须真的跑 ≥2 个 seed 且结果相同。
    """
    sigs = {(round(float(per_seed[str(s)]["baseline_accuracy"]), 6),
             round(float(per_seed[str(s)]["final_accuracy"]), 6),
             round(float(per_seed[str(s)]["delta_pp"]), 6))
            for s in seed_list}
    # 只有 1 个 seed 时"结果一致"是平凡的，不能据此声称零方差 —— 必须 ≥2 个 seed 才作数。
    seeds = list(seed_list)
    return {"n_seeds": len(seeds),
            "distinct_results": len(sigs),
            "zero_variance": len(sigs) == 1 and len(seeds) >= 2,
            "detail": ("%d 个 seed 产生 %d 种结果" % (len(seeds), len(sigs)))}


ZERO_VARIANCE_NOTE = (
    "本夹具下 fake_noise=0 且 provider 的抖动在 noise<=0 时直接关闭，"
    "多个 seed 的结果完全相同 —— 报告里的 95% CI 是题级配对 bootstrap 的"
    "「题目抽样不确定性」，**不是运行间方差**；多 seed 在此不提供额外信息，"
    "不得解读为「多次独立重复实验」。依据 asta-bench "
    "scripts/inspect_score_with_task.py:35-36：确定性任务无需重复跑。"
    "接真实 API 后方差来源变为模型采样 / judge 一致性，届时应改用 "
    "n_samples 或 ICC（ScholarQABench rubric_eval.py:81-94）来报告。"
)


def run_final_assessment(cfg: Dict[str, Any], state: ArtifactState,
                         datasets: Sequence[str], seeds: Optional[Sequence[int]] = None,
                         limit: Optional[int] = None, actor: str = "evaluator",
                         reason: str = "M4 冻结终评",
                         B: int = 10000, rng_seed: int = 20260924,
                         strict_protocol: bool = True,
                         transfer_datasets: Optional[Sequence[str]] = None,
                         budget_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """M4 终评：把当前 release 跑在 final_holdout 上，与封存基线**逐题配对**比较。

    报什么（Spec §14）：
    - 主指标 Δ、95% CI、P(Δ>0)（题级配对 bootstrap）
    - 每个 seed 的准确率与跨 seed 范围
    - 成本（cost_usd / tokens）
    - **负面结果**：回归题（基线对、终评错）、失败归因分布
    """
    baseline = load_baseline()
    # 护栏：ACTIVE 指针有可能被测试重置回 baseline-v0（`init_baseline` 会 write_active）。
    # 这时终评会拿基线跟自己比，Δ 恒为 0 且不报错 —— 极难发现。这里直接硬失败。
    if state.release_id and state.release_id == baseline.get("release_id"):
        raise FinalHoldoutError(
            f"当前生效 release 就是基线 release（{state.release_id}），Δ 恒为 0。"
            "若 ACTIVE 指针被测试重置过，请显式指定 --release <候选 release>。"
        )
    proto = check_protocol(cfg, baseline)
    if strict_protocol and not proto["ok"]:
        raise ProtocolMismatch(
            "终评与基线不在同一协议下，Δ 不可解释：" + json.dumps(proto["diffs"], ensure_ascii=False)
            + "。请先用当前夹具重跑基线（final-baseline --force），或加 --no-strict-protocol 明确放弃可比性。"
        )

    require_verified(list(datasets))

    seed_list = [int(s) for s in seeds] if seeds else [int(baseline.get("seed", 0))]
    # Reject an unpaired seed before final task evaluation starts. Falling
    # back to an unrelated baseline seed after running the candidate would spend
    # a sealed evaluation budget on a comparison that was not frozen correctly.
    for ds in datasets:
        base_runs = (baseline.get("datasets", {}).get(ds, {}) or {}).get("runs", {})
        if not base_runs:
            raise FinalHoldoutError(f"基线里没有 {ds} 的逐 seed 结果，无法配对")
        missing_seeds = [sd for sd in seed_list if str(sd) not in base_runs]
        if missing_seeds:
            raise FinalHoldoutError(
                f"基线 {ds} 没有计划 seed 的逐 seed 结果：{missing_seeds}；"
                "请冻结与基线 seed 完全匹配的终评计划"
            )

    # role 只由数据集本身决定：
    #   primary_final_datasets（A3 新封存的 suppqa）→ main，唯一允许作为结论
    #   transfer_datasets（CS）→ smoke，只证明"机制在另一领域能跑通"，不作结论（A2）
    transfer_set = set(transfer_datasets or
                       cfg.get("evaluation", {}).get("transfer_datasets") or [])
    primary_set = set(cfg.get("evaluation", {}).get("primary_final_datasets") or [])
    if not primary_set:
        primary_set = set(datasets) - transfer_set
    per_ds: Dict[str, Any] = {}

    for ds in datasets:
        runs = _run_final_seeds(cfg, state, ds, seed_list,
                                "final-assessment", limit, actor, reason, False,
                                budget_override)
        base_runs = (baseline.get("datasets", {}).get(ds, {}) or {}).get("runs", {})

        per_seed: Dict[str, Any] = {}
        for sd in seed_list:
            key = str(sd)
            cur = runs[key]
            base = base_runs.get(key) or base_runs.get(str(seed_list[0]))
            paired = pair_seed(base.get("per_task", []), cur["per_task"], B=B, rng_seed=rng_seed)
            estimates = paired_estimands(base["per_task"], cur["per_task"], B, rng_seed)
            per_seed[key] = {
                "estimands": estimates,
                "primary_axis_inference": estimates[str(cfg.get("evaluation", {}).get(
                    "primary_axis") or DEFAULT_PRIMARY_AXIS)],
                "primary_test_estimand": "accuracy",
                "n_paired": paired["n_paired"],
                "baseline_accuracy": paired["baseline_accuracy"],
                "final_accuracy": paired["final_accuracy"],
                "delta_pp": paired["delta_pp"],
                "ci95": paired["ci95"],
                "p_delta_gt_0": paired["p_delta_gt_0"],
                "primary_test": paired["primary_test"],
                "cost_usd": cur["metrics"].get("cost_usd"),
                # P2：成本口径显式化 —— 估算值单独放，不混进 cost_usd
                "cost_estimated": cur["metrics"].get("cost_estimated", True),
                "cost_basis": cur["metrics"].get("cost_basis", ""),
                "cost_usd_nominal": cur["metrics"].get("cost_usd_nominal"),
                "prompt_tokens": cur["metrics"].get("prompt_tokens"),
                "completion_tokens": cur["metrics"].get("completion_tokens"),
                "secondary": {k: cur["metrics"].get(k) for k in
                              ("citation_precision", "abstain_rate",
                               "unsupported_answer_rate", "skill_recall_rate",
                               "gold_in_topk_rate")},
                "secondary_baseline": {k: base["metrics"].get(k) for k in
                                       ("citation_precision", "abstain_rate",
                               "unsupported_answer_rate", "skill_recall_rate",
                               "gold_in_topk_rate")},
                # 【决策 2】三指标口径：accuracy 降参考、precision 主判据、coverage 必报
                "triple": {"baseline": _triple_of_run(base.get("per_task", []),
                                                      base["metrics"]),
                           "candidate": _triple_of_run(cur["per_task"], cur["metrics"])},
                "negative_results": {
                    "n_regressed": paired["n_regressed"],
                    "n_improved": paired["n_improved"],
                    "regressed_task_ids": paired["regressed_task_ids"][:20],
                    "improved_task_ids": paired["improved_task_ids"][:20],
                },
            }

        deltas = [per_seed[str(s)]["delta_pp"] for s in seed_list]
        per_ds[ds] = {
            "n_tasks": runs[str(seed_list[0])]["n_tasks"],
            "per_seed": per_seed,
            "primary_seed": seed_list[0],
            "mean_delta_pp": round(sum(deltas) / len(deltas), 4),
            "delta_range_pp": [round(min(deltas), 4), round(max(deltas), 4)],
            "role": ("main" if ds in primary_set else "smoke" if ds in transfer_set else "main"),
            "seed_variance": _seed_variance(per_seed, seed_list),
        }

    # 【B1】多重比较校正：3 个数据集同时报，不做校正等于"挑最好看的那个报"。
    # 主检验取每个数据集 primary_seed 的精确 McNemar p 值，做 Holm 逐步降级。
    mc_ds = [d for d in per_ds]
    holm = holm_bonferroni([per_ds[d]["per_seed"][str(per_ds[d]["primary_seed"])]
                            ["primary_test"]["p_value"] for d in mc_ds])
    multiplicity = {
        "estimand": "accuracy",
        "role": ("primary" if str(cfg.get("evaluation", {}).get("primary_axis")
                                  or DEFAULT_PRIMARY_AXIS) == "accuracy" else "secondary"),
        "datasets": mc_ds,
        "primary_test": "exact McNemar",
        "holm": holm,
        "interpretation": (
            "下列结论按 Holm 校正后的 p 值判定；未通过校正的数据集只能称"
            "「方向性改善」，不得称「显著改善」。"
        ),
    }
    for d, adj, rej in zip(mc_ds, holm["adjusted"], holm["reject"]):
        per_ds[d]["holm_adjusted_p"] = adj
        per_ds[d]["significant_after_holm"] = bool(rej)

    zero_var_ds = [d for d, v in per_ds.items() if v["seed_variance"]["zero_variance"]]
    variance_note = {
        "zero_variance_datasets": zero_var_ds,
        "all_zero_variance": len(zero_var_ds) == len(per_ds) and bool(per_ds),
        "note": ZERO_VARIANCE_NOTE if zero_var_ds else "",
    }

    # 【决策 2】主结论按三指标口径出（不再由单一 accuracy 出）
    primary_conclusion = {}
    for d in per_ds:
        ps = per_ds[d]["per_seed"][str(per_ds[d]["primary_seed"])]
        primary_conclusion[d] = pair_triple(ps["triple"]["baseline"],
                                            ps["triple"]["candidate"], cfg)

    report = {
        "created_at": now_iso(),
        "milestone": "M4",
        "release_id": state.release_id,
        "system_prompt_sha256": sha256_obj(state.system_prompt),
        "skills_fingerprint": state.registry.fingerprint(),
        "task_demos_sha256": sha256_obj(state.task_demos),
        "baseline_release_id": baseline.get("release_id"),
        "baseline_created_at": baseline.get("created_at"),
        "seeds": seed_list,
        "actor": actor,
        "reason": reason,
        "protocol_check": proto,
        "pairing": "per_task（题级配对 bootstrap）",
        "primary_test": ("exact McNemar for accuracy" if str(cfg.get("evaluation", {}).get(
            "primary_axis") or DEFAULT_PRIMARY_AXIS) == "accuracy" else
            "paired bootstrap CI for configured primary axis; McNemar/Holm assess secondary accuracy"),
        "multiplicity": multiplicity,
        "B": B,
        # 【M1/M2 回顾】必须在报告里显式说明：多 seed 在这个夹具下不提供任何额外方差。
        # 否则"3 seed + 配对 bootstrap CI"看起来满足规范，统计上却是空的。
        "variance": variance_note,
        "datasets": per_ds,
        # 【决策 2】三指标并列的主结论（主判据 = answered_precision，见 verdict.py）
        "primary_conclusion": primary_conclusion,
        "unblind_budget": {ds: budget_state(ds) for ds in datasets},
        "audit_entries": len(audit_log()),
    }
    write_json(ASSESSMENT_FILE, report)
    return report
