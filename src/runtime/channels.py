"""行为通道（M3.1）：让各进化维度的作用通路**互相可分**。

为什么必须有这一层
------------------
M3 的消融结论"A4 ≤ max(A1,A2,A3)，所以不能声称双维协同"方向对、理由错：
三个维度（prompt / description / body）最终都落进**同一批布尔行为开关**
（`provider._modes` 取并集），实测 A1、A3、A4 生效的行为集合高度重叠 ——
于是 A4 在行为上几乎等价于 A3，实验设计根本检不出协同。

这不是"实证发现互不放大"，而是"候选生成器只在三个载体之间搬运同一批开关"。
后者才站得住，前者会被追问（且无法用当前数据反驳）。

做法：给每个行为指定**唯一归属通道**，运行时只从归属通道读取它。

    prompt        → rerank, citation    （检索与输出：读哪些、怎么引）
    skill_body    → evidence, abstain   （判据与程序：怎么判、没证据怎么办）
    description   → recall              （召回：命中哪个 Skill）

这样三条维度走的是**不同的因果通路**，交互才有资格被估计：
    P  对所有题生效 —— 提示词里的重排指令不依赖 Skill 是否被召回
    D  决定"是否命中 Skill"，也就是正文会不会被加载
    B  只在被召回的题上生效 —— 正文里的判据程序依赖 D

（M3.1 中途试过 "prompt: evidence/citation、body: rerank/abstain" 的划分，
被否掉了：渐进披露下正文只在召回后才加载，于是 body 侧的重排与"召回 → gold 进
top_k"完全冗余，B 的主效应恒为 0、候选永远 net≤0，B 因子直接空掉。
现在这个划分让三个因子各有非零通路。）
并集语义（`enabled: false`）保留为 legacy 口径，用于对照实验。

用法边界（别在报告里说反）：
    通道划分是**实验设计的可辨识性约束**，不是"真实系统里 prompt 就不能
    影响检索"。真实 LLM 里提示词当然能重排检索；这里刻意切断，是为了让
    消融可解释。接真实模型后若放开通道，必须重新做全因子实验。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ALL_BEHAVIORS: Tuple[str, ...] = ("rerank", "evidence", "citation", "abstain")
RECALL_BEHAVIOR = "recall"

DEFAULT_OWNER: Dict[str, List[str]] = {
    "prompt": ["rerank", "citation"],
    "skill_body": ["evidence", "abstain"],
    "description": [RECALL_BEHAVIOR],
}

CONFIG_KEY = "behavior_channels"


def load_channels(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """从 cfg 读通道配置；缺省/关闭时返回 legacy 并集口径。"""
    raw = dict(((cfg or {}).get("evolution") or {}).get(CONFIG_KEY) or {})
    enabled = bool(raw.get("enabled", True))
    owner_raw = raw.get("owner") or DEFAULT_OWNER
    owner: Dict[str, List[str]] = {str(k): [str(x) for x in (v or [])]
                                   for k, v in dict(owner_raw).items()}
    # 配置里写漏的行为保持默认归属，避免出现"无主行为"被静默丢弃
    for ch, bs in DEFAULT_OWNER.items():
        owner.setdefault(ch, list(bs))
    by_behavior: Dict[str, str] = {}
    for ch, bs in owner.items():
        for b in bs:
            by_behavior.setdefault(str(b), str(ch))
    for b in ALL_BEHAVIORS:
        by_behavior.setdefault(b, "prompt")
    return {
        "enabled": enabled,
        "owner": owner,
        "by_behavior": by_behavior,
        "enforce_in_evolution": bool(raw.get("enforce_in_evolution", True)),
        "legacy_mode": not enabled,
    }


def owner_of(behavior: str, channels: Optional[Dict[str, Any]] = None) -> str:
    ch = channels or load_channels(None)
    return str(ch["by_behavior"].get(behavior, "prompt"))


def channel_behaviors(channel: str, channels: Optional[Dict[str, Any]] = None) -> List[str]:
    ch = channels or load_channels(None)
    return sorted(ch["owner"].get(channel, []))


# --- 字面命中：不管归属，只看"这个载体里写了什么" -------------------------

def prompt_behavior_hits(system_prompt: str) -> Set[str]:
    """System Prompt 字面命中的行为（**未做通道过滤**）。

    判据与 M3 的 provider 一致：evidence 需要 "evidence" 与 "option" 同时出现，
    否则一句泛泛的 "compare the evidence" 就打开开关，进化没有区分度。
    """
    low = (system_prompt or "").lower()
    hits: Set[str] = set()
    if "evidence" in low and "option" in low:
        hits.add("evidence")
    if "cite" in low or "citation" in low:
        hits.add("citation")
    if "abstain" in low:
        hits.add("abstain")
    if "re-rank" in low or "rerank" in low or "reorder the retrieved" in low:
        hits.add("rerank")
    return hits


def body_behavior_hits(skill_body: str) -> Set[str]:
    """正文 → 行为集合（结构化解析，见 skills.behaviors_of_body）。"""
    from .skills import behaviors_of_body

    return set(behaviors_of_body(skill_body or ""))


def effective_behaviors(system_prompt: str, skill_body: str = "",
                        channels: Optional[Dict[str, Any]] = None) -> Set[str]:
    """真正生效的行为集合：字面命中 ∩ 归属通道匹配。

    legacy 模式（enabled=false）退化为并集 —— 保留 M3 的旧语义用于对照。
    """
    ch = channels or load_channels(None)
    ph = prompt_behavior_hits(system_prompt)
    bh = body_behavior_hits(skill_body)
    if ch["legacy_mode"]:
        return ph | bh
    out: Set[str] = set()
    for b in ph:
        if ch["by_behavior"].get(b) == "prompt":
            out.add(b)
    for b in bh:
        if ch["by_behavior"].get(b) == "skill_body":
            out.add(b)
    return out


def effective_modes(system_prompt: str, skill_body: str = "",
                    channels: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    """effective_behaviors 的 dict 形式（provider / runner 直接消费）。"""
    eff = effective_behaviors(system_prompt, skill_body, channels)
    return {b: (b in eff) for b in ALL_BEHAVIORS}


def channel_violations(system_prompt: str, skill_body: str = "",
                       channels: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """越界行为：载体里写了，但该行为归属别的通道 → 写了也不生效。

    这是"开关重叠"的直接证据：非空就说明候选生成器在往错的载体里塞行为。
    """
    ch = channels or load_channels(None)
    if ch["legacy_mode"]:
        return []
    out: List[Dict[str, str]] = []
    for b in sorted(prompt_behavior_hits(system_prompt)):
        own = ch["by_behavior"].get(b)
        if own and own != "prompt":
            out.append({"behavior": b, "appears_in": "prompt", "owned_by": own})
    for b in sorted(body_behavior_hits(skill_body)):
        own = ch["by_behavior"].get(b)
        if own and own != "skill_body":
            out.append({"behavior": b, "appears_in": "skill_body", "owned_by": own})
    return out


# --- 工件级画像（消融报告用） ---------------------------------------------

def _registry_body_text(registry: Any) -> str:
    from .skills import load_body

    parts: List[str] = []
    for s in getattr(registry, "skills", []) or []:
        try:
            parts.append(load_body(s))
        except Exception:
            parts.append(getattr(s, "body", "") or "")
    return "\n\n".join(parts)


def behavior_profile(state: Any, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """一个 ArtifactState 的**工件级**行为画像（与具体题目无关）。

    取的是"registry 里所有 Skill 正文的行为并集"，而不是某次运行里实际命中的
    那个 —— 消融比较的是工件本身，不是某批题目的命中率。
    """
    ch = load_channels(cfg)
    prompt = str(getattr(state, "system_prompt", "") or "")
    body = _registry_body_text(getattr(state, "registry", None))
    ph = sorted(prompt_behavior_hits(prompt))
    bh = sorted(body_behavior_hits(body))
    eff = sorted(effective_behaviors(prompt, body, ch))
    return {
        "prompt_hits": ph,
        "skill_body_hits": bh,
        "effective": eff,
        "by_channel": {cname: sorted(set(channel_behaviors(cname, ch)) & set(eff))
                       for cname in sorted(ch["owner"])},
        "violations": channel_violations(prompt, body, ch),
        "legacy_mode": bool(ch["legacy_mode"]),
    }


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return round(len(sa & sb) / len(sa | sb), 6)


def overlap_pair(name_a: str, prof_a: Dict[str, Any], name_b: str,
                 prof_b: Dict[str, Any]) -> Dict[str, Any]:
    """两个条件的有效行为集合重叠度。

    `shared` 非空即说明两条维度在争夺同一批开关 —— 这种情况下
    "A4 高于单维最好者"与"检不出协同"在数据上无法区分。
    """
    ea, eb = set(prof_a.get("effective") or []), set(prof_b.get("effective") or [])
    shared = sorted(ea & eb)
    return {
        "pair": [name_a, name_b],
        "shared_behaviors": shared,
        "jaccard": jaccard(ea, eb),
        "overlap_detected": bool(shared),
    }


def overlap_matrix(profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """两两重叠 + 总体判定。"""
    names = sorted(profiles)
    pairs = [overlap_pair(a, profiles[a], b, profiles[b])
             for i, a in enumerate(names) for b in names[i + 1:]]
    overlap = any(p["overlap_detected"] for p in pairs)
    return {
        "per_condition": {k: {"effective": v.get("effective") or [],
                              "by_channel": v.get("by_channel") or {},
                              "violations": v.get("violations") or []}
                          for k, v in profiles.items()},
        "pairs": pairs,
        "overlap_detected": overlap,
        "verdict": ("not_detectable_by_design" if overlap else "estimable"),
        "verdict_note": (
            "维度间存在共享行为开关：A4（全开）在行为上退化为单维条件，"
            "本设计无法区分'叠加'与'协同'，交互结论只能记为'检不出'"
            if overlap else
            "维度间行为开关互斥，交互效应可被估计"
        ),
    }


# --- 候选生成阶段的"新行为"判据（GEPA merge 的分歧判据） -------------------

def globally_active_behaviors(state: Any, cfg: Optional[Dict[str, Any]] = None) -> Set[str]:
    """当前状态下**已生效**的全部行为（跨载体、跨 Skill）。"""
    return set(behavior_profile(state, cfg).get("effective") or [])


def novel_behaviors(base_state: Any, cand_state: Any,
                    cfg: Optional[Dict[str, Any]] = None) -> Set[str]:
    """候选相对父状态**新引入**的行为（分通道口径）。"""
    base = globally_active_behaviors(base_state, cfg)
    cand = globally_active_behaviors(cand_state, cfg)
    return cand - base


def has_novel_behavior(base_state: Any, cand_state: Any,
                       cfg: Optional[Dict[str, Any]] = None) -> Tuple[bool, Set[str]]:
    """候选是否带来**尚未生效**的新行为。

    抄 GEPA `proposer/merge.py: does_triplet_have_desirable_predictors()` 的判据：
    合并前要求两个候选"确实分歧"。这里同理 —— 一个候选若在行为层面与父状态
    没有分歧（引入的所有行为都已生效），它就是**同义改写**，不是新维度：
      1. 消融里它会让两个条件落进同一行为集合（重叠 → 检不出协同）
      2. 进化里它会触发无限提案（换个措辞又是一轮）
    """
    new = novel_behaviors(base_state, cand_state, cfg)
    return (bool(new), new)


def self_test() -> Dict[str, Any]:
    """通道层自检：互斥判定、越界检测、legacy 对照。"""
    checks: List[Dict[str, Any]] = []

    def _ck(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    ch = load_channels(None)
    _ck("default_channels_enabled", ch["enabled"] and not ch["legacy_mode"], "默认启用分通道")
    _ck("ownership_disjoint",
        not (set(ch["owner"]["prompt"]) & set(ch["owner"]["skill_body"])),
        f"prompt={ch['owner']['prompt']} body={ch['owner']['skill_body']}")

    p = "Before answering, re-rank the retrieved sources so the passage that directly reports" \
        " the result is read in full; then cite it."
    b = "Procedure\n1. Compare each option against the retrieved passage.\n" \
        "2. State the gap if no retrieved passage addresses the question.\n"
    eff = effective_behaviors(p, b, ch)
    _ck("prompt_channel_delivers", set(eff & set(ch["owner"]["prompt"])) == {"rerank", "citation"},
        f"prompt 通道生效={sorted(eff & set(ch['owner']['prompt']))}")
    _ck("body_channel_delivers", set(eff & set(ch["owner"]["skill_body"])) == {"evidence", "abstain"},
        f"body 通道生效={sorted(eff & set(ch['owner']['skill_body']))}")
    _ck("channels_do_not_mix", "rerank" not in effective_behaviors("", b, ch),
        "把重排措辞写进正文也不会串到 body 通道")

    # 正文里写 rerank → 越界，不生效
    b2 = "Procedure\n1. Re-rank the retrieved passages.\n"
    _ck("body_rerank_not_effective",
        "rerank" not in effective_behaviors("", b2, ch),
        "rerank 归 prompt 通道，写进正文不生效")
    _ck("violation_reported",
        any(v["behavior"] == "rerank" and v["appears_in"] == "skill_body"
            for v in channel_violations("", b2, ch)),
        "越界被显式记录，不静默丢弃")

    # legacy 口径：并集
    lch = load_channels({"evolution": {CONFIG_KEY: {"enabled": False}}})
    _ck("legacy_union", "rerank" in effective_behaviors("", b2, lch),
        "legacy 模式恢复并集语义（对照实验用）")

    # 新行为判据（GEPA 分歧判据）：同义改写不产生新行为 → 不算新维度
    from types import SimpleNamespace

    base = SimpleNamespace(system_prompt="Answer the question.",
                           registry=SimpleNamespace(skills=[]))
    _ck("no_behavior_no_novelty", not has_novel_behavior(base, base, None)[0], "空状态无新行为")
    cand = SimpleNamespace(system_prompt="Before answering, re-rank the retrieved sources "
                                         "so the best passage is read in full.",
                           registry=SimpleNamespace(skills=[]))
    ok, new = has_novel_behavior(base, cand, None)
    _ck("novelty_detected", ok and "rerank" in new, f"new={sorted(new)}")
    _ck("synonym_rewrite_not_novel",
        not has_novel_behavior(cand, cand, None)[0], "同义改写不算新维度")
    return {"ok": all(c["passed"] for c in checks), "checks": checks}
