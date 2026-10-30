"""候选生成：deterministic reflection stub（**不是 GEPA**）。

覆盖两类工件：
  - `system_prompt`：归因 plan.failed → 追加一条固定指令（M1 起）
  - `skill_description`：归因 skill.not_recalled → 给**反事实命中率最高**的那个 skill
    的 description 注入失败题目里的高频关键词（M2 起开放）

选 skill 的逻辑值得一提：不是拍脑袋指派，而是做一次**反事实模拟**——
把候选关键词加进每个可变 skill 的 description，看哪个能让最多失败题目被召回，
就改哪个。可解释、可复现，且不依赖任何 LLM 调用。

为什么先写它：
  Spec 附录 D 第 3 步要求"先跑通零成本 E2E"，用真实 LLM 反思调试链路既慢又贵。
  这里用一个**固定的指令库**做反思：给定归因类别，产出一条确定性的 System Prompt 指令补丁。

它的边界（务必记住，别在简历/报告里说错）：
  - 它是**测试替身**，用于验证「归因 → 候选 → 门禁 → 评测 → review」这条链路本身是否连通
  - 它**不产出任何效果结论**。真实候选生成必须由 DSPy/GEPA 承担，
    见 `gepa_adapter.py`：依赖缺失时显式报错，不允许拿本 stub 冒充 GEPA
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..contracts.types import (ATTRIBUTION_TARGET, Candidate, content_id, content_sha256,
                               new_id)
from ..util import stable_json
from ..data.split import load_tasks
from ..runtime.channels import has_novel_behavior, load_channels, owner_of
from ..runtime.runner import ArtifactState
from ..runtime.skills import (_stems, append_step, behaviors_of_body, load_body, recall,
                              with_body, with_description)
from .counterfactual import rerun_many, tasks_of

STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "have", "has", "been", "are",
    "was", "were", "which", "what", "when", "into", "using", "used", "may", "can",
    "should", "following", "these", "those", "than", "then", "you", "your", "their",
    "not", "but", "all", "any", "how", "why", "who", "its", "would", "could", "there",
    "about", "after", "before", "over", "under", "more", "most", "some", "such",
}

_WORD_RE = re.compile(r"[a-z][a-z\-]{3,}")

MAX_KEYWORDS = 4
KW_SUFFIX_TEMPLATE = " Use when the task mentions: {kws}."

MORE_STOPWORDS = {
    "between", "into", "during", "within", "without", "among", "both", "each",
    "other", "than", "that", "these", "those", "while", "where", "which", "from",
    "with", "this", "have", "been", "were", "does", "did", "following", "based",
    "given", "using", "used", "also", "only", "same", "such", "into", "onto",
    "what", "when", "best", "most", "least", "following", "however", "therefore",
}
STOPWORDS = STOPWORDS | MORE_STOPWORDS


def _looks_like_identifier(word: str) -> bool:
    """过滤题面里的编号/位点噪声（如 311-476、adr-1、pm-1），它们没有语义泛化能力。"""
    if any(ch.isdigit() for ch in word):
        return True
    if "-" in word and len(word) < 8:
        return True
    return False

# ===========================================================================
# 反思候选：维度枚举 × 措辞变体 × 可插失败证据（P4 L1）
# ===========================================================================
#
# 为什么不再写死一整句
#   旧实现是"每个归因类别 → 一两句固定指令"。它能跑，但问题在于
#   **候选内容完全由我预先写好**，于是"反思"这一层名不副实：只能说"链路通了"，
#   不能说"自进化机制有效"。
#
#   对齐 pk-pi 的 `style_hint`（`python_backend/.../backend.py:860`）：把
#   "要改哪个方面"枚举成几条固定的**维度**，每个维度配多个**措辞变体**，
#   再允许把失败的证据（题面关键词）插进模板槽位 `{scope}`。于是：
#     · 维度仍然固定 → 归因依然成立，不破坏"一次只改一个维度"的约束；
#     · 内容不再固定 → 候选是"按维度 + 失败证据"生成的，不是固定句子。
#
#   注意 GEPA `default_adapter.py:184` 的 `assert len(components_to_update) == 1`
#   说明"一次只改一个组件"是学界默认 —— 本项目的维度约束与之相符，不是自创。
#
# 【M3.1】措辞必须携带 **prompt 通道** 的行为关键词（rerank / citation）。
# 写 "evaluate every option against the evidence" 这类判据措辞是无效的：
# evidence 归 skill_body 通道，写进提示词不生效（见 runtime/channels.py）。
#
# 各类的**首选指令**逐字保持与 L1 之前一致（`PRIMARY_DIRECTIVES`），
# 这样历史 campaign 的结果仍可比；新增的是变体与槽位。

# 首选指令：每类的第一条，`next_directive` 会优先用它
PRIMARY_DIRECTIVES: Dict[str, str] = {
    "plan.failed":
        "Before answering, re-rank the retrieved sources so that the passage which directly "
        "reports the requested result is read in full, then cite that source for the chosen "
        "answer.",
    "skill.not_recalled":
        "When the task mentions a known literature-synthesis procedure, first state which "
        "specialized procedure applies, then re-rank the retrieved sources before choosing.",
    "skill.misapplied":
        "After loading a procedure, re-rank the retrieved sources so that the passage which "
        "directly addresses the question is read in full before deciding.",
    "skill.missing":
        "If no retrieved source addresses the question, state the gap explicitly instead of "
        "choosing the most familiar option.",
}

# 维度 → 措辞变体。`{scope}` 是失败证据槽位，由 `expand_directive` 填充
# （填失败题的实词；无证据时退化为空串，保证模板始终可解析）。
DIRECTIVE_VARIANTS: Dict[str, List[str]] = {
    "rerank": [
        "Before answering, re-rank the retrieved sources so that the passage which directly "
        "reports the requested result is read in full, then cite that source for the chosen "
        "answer.",
        "Before answering, re-order the retrieved passages by how directly they report the "
        "result the question asks about, and cite the source you rely on.",
        "Before deciding{scope}, re-rank the retrieved sources by how directly each one "
        "reports the requested finding, then read the top passage in full.",
        "Re-rank the retrieved sources{scope} so that any passage reporting the requested "
        "result is read in full before an answer is chosen.",
    ],
    "citation": [
        "Cite the specific source that supports the chosen answer{scope}, naming the "
        "passage you relied on.",
        "For the option you select, cite the retrieved source{scope} that reports it.",
    ],
    # 无行为关键词的一类：用于"证据缺口"这类不靠行为开关生效的措辞
    "gap": [
        "If no retrieved source addresses the question, state the gap explicitly instead of "
        "choosing the most familiar option.",
        "When no retrieved passage addresses the question{scope}, state that the evidence "
        "is missing rather than selecting an option.",
    ],
}

# 归因类别 → 允许使用的维度（**这就是"维度枚举"**：维度固定，所以还能归因）
CLASS_DIMENSIONS: Dict[str, List[str]] = {
    "plan.failed": ["rerank", "citation"],
    "skill.not_recalled": ["rerank"],
    "skill.misapplied": ["rerank"],
    "skill.missing": ["gap"],
}

# 刻意的**退化**补丁：它不带来增益，用来验证非回归门真的会拦住退化候选 ——
# 没被拦住就说明门坏了。
DEGRADED_DIRECTIVES: Dict[str, str] = {
    "plan.failed": "Answer from prior knowledge first; use the retrieved passages only if "
                   "time permits.",
}


def _build_directive_library() -> Dict[str, List[str]]:
    """由「首选指令 + 维度变体 + 退化补丁」拼出指令库。"""
    lib: Dict[str, List[str]] = {}
    for cls, primary in PRIMARY_DIRECTIVES.items():
        out = [primary]
        for dim in CLASS_DIMENSIONS.get(cls, []):
            for v in DIRECTIVE_VARIANTS.get(dim, []):
                if v not in out:
                    out.append(v)
        if cls in DEGRADED_DIRECTIVES:
            out.append(DEGRADED_DIRECTIVES[cls])
        lib[cls] = out
    return lib


DIRECTIVE_LIBRARY: Dict[str, List[str]] = _build_directive_library()


def expand_directive(text: str, failure_rows: Optional[Sequence[Dict[str, Any]]] = None) -> str:
    """把模板里的 `{scope}` 槽位填成失败证据（P4 L1）。

    失败证据 = 这批失败题的实词（复用 `keywords_for_failures`）。填进去之后，
    同一维度的不同轮次会产出**不同**的候选文本 —— 这就是"内容不再固定"的来源。
    没有证据时退化为空串，保证模板永远可解析。
    """
    if "{scope}" not in text:
        return text
    kws: List[str] = []
    if failure_rows:
        try:
            tasks = _task_index([str(r.get("dataset", "")) for r in failure_rows
                                 if r.get("dataset")])
            kws = list(keywords_for_failures(failure_rows, tasks) or [])
        except Exception:
            kws = []      # 拿不到证据就退化为不带槽位，不让"证据注入"成为失败点
    scope = f" for tasks about {', '.join(kws[:3])}" if kws else ""
    return text.replace("{scope}", scope)


def unified_diff(old: str, new: str, from_name: str = "a/system_prompt",
                 to_name: str = "b/system_prompt") -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile=from_name, tofile=to_name, n=3,
        )
    )


def directive_for(cls: str) -> Optional[str]:
    """该类别的第一条指令（兼容旧接口）。"""
    lst = DIRECTIVE_LIBRARY.get(cls) or []
    return lst[0] if lst else None


def next_directive_detail(cls: str, current_prompt: str, state: Optional[ArtifactState] = None,
                          cfg: Optional[Dict[str, Any]] = None,
                          failure_rows: Optional[Sequence[Dict[str, Any]]] = None
                          ) -> Optional[Dict[str, Any]]:
    """同 `next_directive`，但额外返回该指令所属的**维度**与是否为退化补丁。

    【P4 L1】维度是"候选归因"的关键：一个候选属于哪个维度是**固定枚举**出来的，
    不是事后猜的 —— 这正是"内容自由但维度固定"这对约束的落点。
    """
    low = (current_prompt or "").lower()
    enforce = state is not None and _enforce_novelty(cfg)
    degraded_text = DEGRADED_DIRECTIVES.get(cls)
    for d in DIRECTIVE_LIBRARY.get(cls) or []:
        candidate = expand_directive(d, failure_rows)
        if candidate.strip().lower() in low:
            continue
        dim = "degraded" if (degraded_text and d == degraded_text) else _dimension_of(d)
        if enforce:
            new_prompt = (current_prompt or "").rstrip() + "\n" + candidate + "\n"
            trial = ArtifactState(release_id=state.release_id, system_prompt=new_prompt,
                                  registry=state.registry, task_demos=state.task_demos)
            ok, _ = has_novel_behavior(state, trial, cfg)
            if not ok:
                continue
        return {"text": candidate, "dimension": dim,
                "degraded": dim == "degraded",
                "evidence_injected": "{scope}" in d and bool(failure_rows)}
    return None


def _dimension_of(template: str) -> str:
    """反查某条指令模板属于哪个维度。"""
    for dim, variants in DIRECTIVE_VARIANTS.items():
        if template in variants:
            return dim
    return "primary"


def next_directive(cls: str, current_prompt: str, state: Optional[ArtifactState] = None,
                   cfg: Optional[Dict[str, Any]] = None,
                   failure_rows: Optional[Sequence[Dict[str, Any]]] = None) -> Optional[str]:
    """返回本类别中**尚未写进当前 prompt**的第一条指令；全都用过则返回 None（防重复提案）。

    【M3.1】若给了 `state` 且开启了通道约束，会再过一道"新行为"判据：
    候选指令若在行为层面与当前状态没有分歧（引入的行为都已生效），它就是
    同义改写而非新维度 —— 提它只会让消融的两个条件落进同一行为集合。
    （判据取自 GEPA `proposer/merge.py` 的"合并前要求分歧"检查。）

    【P4 L1】若给了 `failure_rows`，会把模板里的 `{scope}` 槽位填成失败证据，
    于是同一维度在不同轮次产出不同文本（内容不再固定，但维度仍固定）。
    """
    d = next_directive_detail(cls, current_prompt, state=state, cfg=cfg,
                              failure_rows=failure_rows)
    return d["text"] if d else None


def _enforce_novelty(cfg: Optional[Dict[str, Any]]) -> bool:
    return bool(load_channels(cfg).get("enforce_in_evolution", True))


def _prompt_candidate_from_backend(backend: Any, cls: str, state: ArtifactState,
                                   failure_ids: List[str], cfg: Dict[str, Any],
                                   failure_rows: Optional[Sequence[Dict[str, Any]]]
                                   ) -> Optional[Candidate]:
    """用外部反思后端产出 prompt 候选（P4 L2）。

    与 `_prompt_candidate` 的唯一区别是"文本从哪来"（后端 vs 本地模板），
    其余（内容哈希 ID、diff、rationale）完全一致 —— 这样换成 LlmBackend 时，
    发布的候选格式不变，历史产物仍可比。
    """
    patch = backend.propose(cls, state.system_prompt, failure_rows, cfg, state=state)
    if patch is None or not getattr(patch, "text", ""):
        return None
    old = state.system_prompt
    # 【2026-09-27】产出形态二选一：
    #   mode="patch"   → 追加一句（历史形态，所有已发布 release 都是这个形态）
    #   mode="rewrite" → 整条替换（GEPA 形态，见 prompt_rewrite.py）
    # 追加形态的候选空间只有"加哪句触发词"，量级约 2^4 —— 这是"自进化测不出
    # 收益"的结构性原因；整段改写把它扩到自由文本空间。
    is_rewrite = (getattr(patch, "mode", "patch") == "rewrite"
                  and bool(getattr(patch, "full_prompt", "")))
    new = patch.full_prompt if is_rewrite else (old.rstrip() + "\n" + patch.text + "\n")
    cand = Candidate(
        candidate_id=content_id("cand", state.release_id, "system_prompt", new),
        parent_release_id=state.release_id,
        artifact_type="system_prompt",
        artifact_id="system_prompt",
        diff=unified_diff(old, new),
        rationale=getattr(patch, "rationale", "") or f"归因 {cls}：由反思后端产出候选。",
        status="draft",
        new_content=new,
        attributed_class=cls,
        source_failure_ids=list(failure_ids),
        generator={
            "backend": getattr(patch, "source", "backend"),
            "is_gepa": False,
            "dimension": getattr(patch, "dimension", "primary"),
            # 产出形态（patch / rewrite）与 few-shot 注入条数 —— 没有它，
            # 事后无法区分"候选是追加还是整段改写"，也就无法归因。
            "proposal_mode": ("rewrite" if is_rewrite else "patch"),
            "fewshot_demos": int(getattr(patch, "demos_used", 0) or 0),
            "degraded": bool(getattr(patch, "degraded", False)),
            "evidence_injected": bool(getattr(patch, "evidence_injected", False)),
            # 真实模型反思的留痕：每次尝试的原始输出与被拒原因。
            # 没有它，"提不出候选"无法区分"模型不行"和"链路坏了"。
            "reflection_attempts": list(getattr(backend, "attempts", []) or [])[:3],
            "reflection_error": getattr(backend, "last_error", None),
            "reflection_model": (cfg.get("models", {}).get("reflection", {})
                                 .get("model_id", "none")),
        },
        content_sha256=content_sha256(new),
    )
    return cand


def generate_candidate(cls: str, state: ArtifactState, failure_ids: List[str],
                       cfg: Dict[str, Any], artifact_type: Optional[str] = None,
                       failure_rows: Optional[Sequence[Dict[str, Any]]] = None,
                       backend: Optional[Any] = None) -> Optional[Candidate]:
    """按目标工件类型分派；补丁已存在（重复提案）则返回 None（FR-207）。

    【P4 L2】`backend` 是反思后端（`reflection_backend.ReflectionBackend`）。
    传入时由它产出 prompt 维度的候选，而不是直接用本地模板库 ——
    接真实 API 后换成 LlmBackend 即可，本函数签名不变。
    """
    atype = artifact_type or ATTRIBUTION_TARGET.get(cls)
    if atype == "system_prompt":
        if backend is not None:
            return _prompt_candidate_from_backend(backend, cls, state, failure_ids,
                                                  cfg, failure_rows)
        return _prompt_candidate(cls, state, failure_ids, cfg, failure_rows=failure_rows)
    if atype == "skill_description":
        return generate_description_candidate(state, list(failure_rows or []), cfg)
    if atype == "skill_body":
        return generate_body_candidate(state, list(failure_rows or []), cfg)
    if atype == "new_skill":
        from .crystallize import generate_new_skill_candidate
        return generate_new_skill_candidate(state, list(failure_rows or []), cfg)
    return None


def _prompt_candidate(cls: str, state: ArtifactState, failure_ids: List[str],
                      cfg: Dict[str, Any],
                      failure_rows: Optional[Sequence[Dict[str, Any]]] = None) -> Optional[Candidate]:
    """产出单个候选；若所有候选指令都已写进 prompt（重复提案）则返回 None（FR-207）。

    【P4 L1】`failure_rows` 传给 `next_directive` 用于填充 `{scope}` 证据槽位。
    """
    detail = next_directive_detail(cls, state.system_prompt, state=state, cfg=cfg,
                                   failure_rows=failure_rows)
    if not detail:
        return None
    directive = detail["text"]
    old = state.system_prompt
    new = old.rstrip() + "\n" + directive + "\n"
    cand = Candidate(
        # 【M1/M2 回顾】候选身份用**内容哈希**（对齐 GEPA `core/state.py:37-39`），
        # 不用 uuid —— 这样两次等价运行产出同一个 ID，可做字节级重放比对。
        candidate_id=content_id("cand", state.release_id, "system_prompt", new),
        parent_release_id=state.release_id,
        artifact_type="system_prompt",
        artifact_id="system_prompt",
        diff=unified_diff(old, new),
        rationale=(f"归因 {cls}：按维度「{detail['dimension']}」产出候选"
                   + ("（已注入失败证据）" if detail.get("evidence_injected") else "")
                   + f"，针对 {len(failure_ids)} 条失败样本。"),
        status="draft",
        new_content=new,
        attributed_class=cls,
        source_failure_ids=list(failure_ids),
        generator={
            "backend": "deterministic-stub",
            "is_gepa": False,
            # 【P4 L1】维度是固定枚举出来的，写进产物以便事后核对"这个候选改的是哪个方面"
            "dimension": detail["dimension"],
            "degraded": bool(detail.get("degraded")),
            "evidence_injected": bool(detail.get("evidence_injected")),
            "reflection_model": cfg.get("models", {}).get("reflection", {}).get("model_id", "none"),
            "note": "测试替身，不产生效果结论；真实候选用 DSPy/GEPA（见 gepa_adapter.py）",
        },
        content_sha256=content_sha256(new),
    )
    return cand


# ---------------------------------------------------------------------------
# M2：Skill description 进化
# ---------------------------------------------------------------------------

def _task_index(datasets: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """按 task_id 索引已重新校验的 dev_optimize 题目。"""
    idx: Dict[str, Dict[str, Any]] = {}
    for ds in sorted(set(datasets)):
        rows = load_tasks(ds, splits=["dev_optimize"])
        if any(t.get("split") != "dev_optimize" for t in rows):
            raise ValueError("candidate generation tasks must come from dev_optimize")
        per_dataset = {t["task_id"]: t for t in rows}
        if len(per_dataset) != len(rows):
            raise ValueError("candidate task IDs must be unique within each dataset")
        if idx.keys() & per_dataset.keys():
            raise ValueError("candidate task IDs must be unique across datasets")
        idx.update(per_dataset)
    return idx


def keywords_for_failures(failure_rows: Sequence[Dict[str, Any]],
                          tasks: Dict[str, Dict[str, Any]],
                          k: int = MAX_KEYWORDS) -> List[str]:
    """从失败题目的**题干**里抽高频实词，作为 description 的补充触发词。

    三条刻意的约束：
    1. **只用题干，不用选项** —— 把选项里的词写进 description 等于把答案特征
       硬编码进触发条件，换个题目就失效（且是拿 optimize 的答案在训练）
    2. **文档频次 ≥ 2** —— 只在单题出现的词通常是专有名词噪声，没有泛化价值
    3. **过滤编号型 token**（含数字 / 短连字符）—— 例如 311-476、adr-1
    """
    df: Counter = Counter()
    tf: Counter = Counter()
    for r in failure_rows:
        t = tasks.get(str(r.get("task_id")))
        if not t:
            continue
        words = {w for w in _WORD_RE.findall(str(t.get("question", "")).lower())
                 if w not in STOPWORDS and not _looks_like_identifier(w)}
        df.update(words)
        tf.update(words)

    shared = [w for w, n in df.items() if n >= 2]
    pool = shared or list(df)
    if not pool:
        return []
    # 频次优先，同频次取更长的词（信息量更大）
    return [w for w, _ in sorted(((w, (df[w], len(w), w)) for w in pool),
                                 key=lambda x: (-x[1][0], -x[1][1], x[1][2]))[:k]]


def _mutable_skills(state: ArtifactState, cfg: Dict[str, Any]) -> List[Any]:
    """description 进化的候选 skill。

    优先 `skills.description_evolution_groups`（默认 G-main / G-adj）：
    description 承载的是"什么时候该用我"的语义，只有领域/任务型 skill 才有意义；
    G-common 里的格式化、术语规范化这类通用 skill 塞关键词进去只会污染触发条件。
    """
    preferred = [g for g in cfg.get("skills", {}).get("description_evolution_groups", [])
                 or ["G-main", "G-adj"]]
    groups = set(cfg.get("skills", {}).get("mutable_groups", []) or [])
    pool = [s for s in state.registry.skills if (not groups or s.group in groups) and not s.frozen]
    preferred_pool = [s for s in pool if s.group in preferred]
    return preferred_pool or pool


def pick_target_skill(state: ArtifactState, failure_rows: Sequence[Dict[str, Any]],
                      keywords: Sequence[str], cfg: Dict[str, Any]) -> Optional[Tuple[Any, int]]:
    """反事实选择：把关键词加进每个可变 skill，看谁能让更多失败题被召回。

    返回 (skill, 命中得分总和)。命中得分相同则按名字字典序（保证可复现）。
    """
    cands = _mutable_skills(state, cfg)
    if not cands or not keywords:
        return None
    tasks = _task_index([str(r.get("dataset", "")) for r in failure_rows if r.get("dataset")])
    suffix = KW_SUFFIX_TEMPLATE.format(kws=", ".join(keywords))
    max_chars = int(cfg.get("gates", {}).get("static", {}).get("max_description_chars", 400))
    min_score = int(cfg.get("skills", {}).get("recall_min_score", 2))

    best: Optional[Tuple[Any, int]] = None
    for s in cands:
        new_desc = (s.description.rstrip() + suffix)[:max_chars]
        trial = with_description(state.registry, s.name, new_desc)
        hits = 0
        score = 0
        for r in failure_rows:
            t = tasks.get(str(r.get("task_id")))
            if not t:
                continue
            hit = recall(f"{t.get('question','')} " + " ".join(t.get("options", [])), trial,
                         min_score=min_score)
            if hit:
                hits += 1
                score += int(hit["score"])
        key = (hits, score)
        if best is None or key > best[1] or (key == best[1] and s.name < best[0].name):
            best = (s, key)
    return best


def generate_description_candidate(state: ArtifactState, failure_rows: Sequence[Dict[str, Any]],
                                   cfg: Dict[str, Any]) -> Optional[Candidate]:
    """归因 skill.not_recalled → 改目标 skill 的 description。

    返回 None 的情形（都写进 round 报告的 skipped 原因）：
      - 没有失败样本 / 抽不出关键词
      - 关键词已全部在该 skill 的 description 或 triggers 中（防重复提案）
      - 没有可变 skill
    """
    if not failure_rows:
        return None
    tasks = _task_index([str(r.get("dataset", "")) for r in failure_rows if r.get("dataset")])
    keywords = keywords_for_failures(failure_rows, tasks)
    if not keywords:
        return None

    picked = pick_target_skill(state, failure_rows, keywords, cfg)
    if picked is None:
        return None
    skill, (hits, score) = picked

    max_chars = int(cfg.get("gates", {}).get("static", {}).get("max_description_chars", 400))
    # 逐步减少关键词，直到新 description 能放进长度上限
    while keywords:
        new_desc = skill.description.rstrip() + KW_SUFFIX_TEMPLATE.format(kws=", ".join(keywords))
        if len(new_desc) <= max_chars:
            break
        keywords = keywords[:-1]
    if not keywords:
        return None
    new_desc = skill.description.rstrip() + KW_SUFFIX_TEMPLATE.format(kws=", ".join(keywords))

    old_desc = skill.description
    if new_desc.strip().lower() == old_desc.strip().lower():
        return None
    # 防重复提案：关键词若已在 description 或 triggers 中出现过，就不再提
    existing = _stems(old_desc + " " + " ".join(skill.triggers))
    if all(_stems(k).issubset(existing) or (set(_stems(k)) & existing) for k in keywords):
        return None

    triggers = sorted(set(list(skill.triggers) + list(keywords)))[:8]
    cand = Candidate(
        candidate_id=content_id("cand", state.release_id, "skill_description",
                                skill.name, new_desc, triggers),
        parent_release_id=state.release_id,
        artifact_type="skill_description",
        artifact_id=skill.name,
        diff=unified_diff(old_desc, new_desc,
                          from_name=f"a/{skill.name}.description",
                          to_name=f"b/{skill.name}.description"),
        rationale=(f"归因 skill.not_recalled：{len(failure_rows)} 条失败题目未召回任何 Skill。"
                   f"反事实模拟显示给 '{skill.name}' 的 description 注入关键词 "
                   f"{keywords} 后，可命中其中 {hits}/{len(failure_rows)} 题（同类最高）。"),
        status="draft",
        new_content=new_desc,
        attributed_class="skill.not_recalled",
        source_failure_ids=[str(r.get("failure_id", "")) for r in failure_rows],
        generator={
            "backend": "deterministic-stub",
            "is_gepa": False,
            "selection": "counterfactual-recall-coverage",
            "note": "测试替身，不产生效果结论；真实候选用 DSPy/GEPA（见 gepa_adapter.py）",
        },
        content_sha256=content_sha256(new_desc),
        extra={
            "skill_name": skill.name,
            "old_description": old_desc,
            "new_triggers": triggers,
            "keywords": list(keywords),
            "counterfactual_recall": {"hits": int(hits), "score": int(score),
                                      "n_failures": len(failure_rows)},
        },
    )
    return cand


# ---------------------------------------------------------------------------
# M3：Skill 正文进化（归因 skill.misapplied → 改 SKILL.md 正文）
# ---------------------------------------------------------------------------

# 候选步骤库：**按行为分类**，每类一条。
#
# 选择哪一类不是拍脑袋，而是**反事实重跑**（counterfactual.rerun）决定：
# 把这一步加进目标 Skill 的正文，在同一批失败题上重跑，看 net = 修好数 − 弄坏数。
# 只提 net > 0 的候选 —— 因此退化性的步骤在这里就出不来，不必等门禁拦。
BODY_PATCH_LIBRARY: Dict[str, str] = {
    # evidence / abstain 归 skill_body 通道（rerank 归 prompt，写进正文不生效）
    "evidence": (
        "Compare each option against the retrieved passage and discard options that are "
        "supported only by surface similarity to the question wording."
    ),
    "abstain": (
        "If no retrieved passage addresses the question, state the gap instead of choosing "
        "the most familiar option."
    ),
}

BODY_PATCH_ORDER = ["evidence", "abstain"]


def _body_mutable_skills(state: ArtifactState, cfg: Dict[str, Any]) -> List[Any]:
    """正文可改的 skill：与 description 进化同一套分组限制，且 frozen 的不能动。

    G-verify（引用核查）是治理类 Skill，正文被改等于自己削弱自己的核查标准，
    因此默认不可改。
    """
    return _mutable_skills(state, cfg)


def pick_body_target(state: ArtifactState,
                     failure_rows: Sequence[Dict[str, Any]]) -> Optional[Any]:
    """选目标 skill：失败样本里**被召回次数最多**的那个。

    skill.misapplied 的含义是"用错了这个 Skill"，所以改的必然是当时被召回的那个，
    而不是随便挑一个高频 skill —— 否则 diff 与归因对不上，review 就没法审。
    """
    cnt: Counter = Counter()
    for r in failure_rows:
        name = str(r.get("skill_recalled") or "").strip()
        if name:
            cnt[name] += 1
    if not cnt:
        return None
    name = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return state.registry.get(name)


def pick_body_targets(state: ArtifactState, failure_rows: Sequence[Dict[str, Any]],
                      cfg: Dict[str, Any]) -> List[Any]:
    """【M4+】失败样本里**被召回过的全部**可变 skill（按召回次数降序）。

    为什么不再只挑"被召回最多的那一个"（`pick_body_target`）：
        行为只来自被召回的那个 Skill 的正文（runner.py:130-139）。失败样本是跨多个
        Skill 分布的 —— 只改召回最多的那个，其余 Skill 覆盖的题一分不动，
        反事实净增益被摊平到 0。实测：给**全部**可变 Skill 加 evidence 步
        → protocolqa 49.18% → 72.13%（**+22.95pp**）；只加一个 → **+0.00pp**。
        后果是 campaign 三轮里 `skill.misapplied` 候选的 delta_pp 全是空 dict，
        这条归因分支等于没接线。

    对齐 GEPA `AllReflectionComponentSelector`（`component_selector.py:27-36`）：
    一轮反射**全部**组件，而不是 round_robin 每次一个。
    注意 GEPA 内置的两个 selector 都不做失败归因，我们保留归因 —— 只把"改几个"放开。
    """
    cnt: Counter = Counter()
    for r in failure_rows:
        name = str(r.get("skill_recalled") or "").strip()
        if name:
            cnt[name] += 1
    mutable = {s.name: s for s in _body_mutable_skills(state, cfg) if not s.frozen}
    out: List[Any] = []
    for name, _c in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
        s = mutable.get(name)
        if s is not None:
            out.append(s)
    return out


def _body_trials(state: ArtifactState, targets: Sequence[Any],
                 cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """全部正文试验：单 Skill 逐个 **+** 「所有目标 Skill 同改」的组合试验。

    组合试验为什么不能省（而不是把各单 Skill 的 net 相加）：
        多个 Skill 同时改不是线性叠加 —— 失败题被各 Skill 分批覆盖，
        叠加后的净增益**不等于**各自净增益之和，只有真的跑一遍才知道。
        DSPy COPRO（`copro_optimizer.py:195-200`）在多 predictor 时必须全量重评
        所有候选、不能单组件加分相加，正是同一个理由。
    """
    per: Dict[str, List[Dict[str, Any]]] = {}
    for s in targets:
        cands = _patch_candidates(state, s, cfg)
        if cands:
            per[s.name] = cands

    trials: List[Dict[str, Any]] = []
    for name in sorted(per):
        for c in per[name]:
            trials.append(dict(c, targets=[name], kind="single"))

    by_key: Dict[str, Dict[str, str]] = {}
    for name in sorted(per):
        for c in per[name]:
            by_key.setdefault(c["patch_key"], {})[name] = c["new_body"]
    for key in BODY_PATCH_ORDER:
        bodies = by_key.get(key) or {}
        if len(bodies) < 2:
            continue  # 只有一个 Skill 可改时，组合试验 == 单 Skill 试验，不必重复
        reg = state.registry
        for n in sorted(bodies):
            reg = with_body(reg, n, bodies[n])
        trials.append({
            "patch_key": key,
            "step": BODY_PATCH_LIBRARY[key],
            "targets": sorted(bodies),
            "kind": "combined",
            "new_bodies": dict(bodies),
            "state": ArtifactState(release_id=state.release_id,
                                   system_prompt=state.system_prompt,
                                   registry=reg, task_demos=state.task_demos),
        })
    return trials


def _pre_evaluation_body_trials(state: ArtifactState,
                                trials: Sequence[Dict[str, Any]],
                                cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]],
                                                              List[Dict[str, Any]]]:
    """Keep only body candidates that pass the promotion static gate.

    Body generation historically scored template patches first and left
    `apply_gate()` to the later round boundary. That spent counterfactual
    TaskRunner calls on candidates already destined for rejection. The round
    repeats the gate after selection; this early gate is a budget guard.
    """
    from ..governance.policy import static_gate

    eligible: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for trial in trials:
        targets = sorted(str(name) for name in trial.get("targets") or [])
        if not targets:
            continue
        combined = trial.get("kind") == "combined"
        if combined:
            artifact_id = "+".join(targets)
            new_content = str(trial.get("step") or "")
            bodies = dict(trial.get("new_bodies") or {})
            diff = "\n".join(unified_diff(
                load_body(state.registry.get(name)), bodies[name],
                from_name=f"a/{name}.md", to_name=f"b/{name}.md")
                for name in targets if name in bodies
                and state.registry.get(name) is not None)
            extra = {"skill_bodies": bodies}
        else:
            if len(targets) != 1 or not trial.get("new_body"):
                continue
            artifact_id = targets[0]
            new_content = str(trial["new_body"])
            diff = unified_diff(
                load_body(state.registry.get(artifact_id)), new_content,
                from_name=f"a/{artifact_id}.md", to_name=f"b/{artifact_id}.md")
            extra = {}
        preview = Candidate(
            candidate_id=content_id("cand-preflight", state.release_id,
                                    "skill_body", artifact_id, new_content),
            parent_release_id=state.release_id,
            artifact_type="skill_body", artifact_id=artifact_id,
            diff=diff, rationale="pre-evaluation Skill body patch gate",
            status="draft", new_content=new_content,
            attributed_class="skill.misapplied",
            content_sha256=content_sha256(new_content), extra=extra)
        gate = static_gate(preview, cfg, state)
        if gate.get("passed") is True:
            eligible.append(dict(trial))
        else:
            rejected.append({
                "patch_key": str(trial.get("patch_key") or ""),
                "targets": targets,
                "kind": str(trial.get("kind") or "single"),
                "failed_static_checks": [
                    {"check": row.get("check"), "detail": row.get("detail")}
                    for row in gate.get("results") or []
                    if row.get("passed") is not True
                ],
            })
    return eligible, rejected


def _patch_candidates(state: ArtifactState, skill: Any,
                      cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """列出**尚未在目标正文中生效**的候选步骤（按 BODY_PATCH_ORDER）。

    判重用的是"行为是否已生效"而不是"字符串是否已出现"：
    正文里换成同义写法（reorder / rank the retrieved）也算已经有了，
    否则换个措辞就会被当成新提案，触发无限进化。

    【M3.1】两道额外约束（配 `evolution.behavior_channels`）：
      1. **通道过滤**：`evidence` 归属 prompt 通道，写进正文不生效 —— 直接跳过，
         否则候选看起来有 diff、实际零行为变化（M3 的 A3 正是这么被 A1 吸收的）。
      2. **新行为**：候选引入的行为必须不在**当前整体状态**里生效
         （不只是目标 Skill 的正文）—— 抄 GEPA 的"合并前要求分歧"判据。

    【M4+】第 2 条对**正文**是错的，已改为"目标 Skill 自身粒度"，理由是可测的：
      行为来自**被召回的那一个** Skill 的正文 —— runner.py:130-139 只把 `hit_body`
      传进 `effective_modes`，别的 Skill 正文里写了什么，对这批题毫无影响。
      所以在"全局"粒度判新颖性，等于说"只要任意一个 Skill 有了 evidence，
      其它 Skill 就都不许再加" —— 而失败题是被多个 Skill 分批覆盖的，
      这直接把正文进化判成 no-op。实测：给全部可变 Skill 加 evidence 步，
      protocolqa 49.18% → 72.13%（+22.95pp）；只加被召回最多的那一个 → +0.00pp。
      正确粒度是"目标 Skill 自己还没有这个行为"（上面的 `current` 已经判了）。
    """
    body = load_body(skill)
    channels = load_channels(cfg)
    enforce = bool(channels.get("enforce_in_evolution", True))
    current = set(behaviors_of_body(body))
    out: List[Dict[str, Any]] = []
    for key in BODY_PATCH_ORDER:
        if key in current:
            continue
        if enforce and channels.get("enabled", True) and owner_of(key, channels) != "skill_body":
            continue  # 越界：这个行为归别的通道，写进正文不会生效
        step = BODY_PATCH_LIBRARY[key]
        new_body = append_step(body, step)
        # 追加后行为必须真的生效，否则说明这条措辞没被解析器识别 —— 直接跳过
        if key not in set(behaviors_of_body(new_body)):
            continue
        trial = ArtifactState(release_id=state.release_id,
                              system_prompt=state.system_prompt,
                              registry=with_body(state.registry, skill.name, new_body),
                              task_demos=state.task_demos)
        out.append({
            "patch_key": key,
            "step": step,
            "new_body": new_body,
            "state": trial,
        })
    return out


def generate_body_candidate(state: ArtifactState, failure_rows: Sequence[Dict[str, Any]],
                            cfg: Dict[str, Any]) -> Optional[Candidate]:
    """归因 skill.misapplied → 给目标 Skill 正文追加一步（反事实净增益最大者）。

    【M4+】目标 Skill 不再只有 1 个：`pick_body_targets` 返回失败样本里被召回过的
    **全部**可变 Skill，并额外构造"所有目标同改"的组合试验，一起 `rerun_many`
    联合重评后取 net 最大者（详见 `_body_trials`）。
    """
    if not failure_rows:
        return None
    targets = pick_body_targets(state, failure_rows, cfg)
    if not targets:
        return None
    trials = _body_trials(state, targets, cfg)
    trials, static_gate_rejections = _pre_evaluation_body_trials(state, trials, cfg)
    if not trials:
        return None  # 无候选通过静态门，不消耗反事实评测预算

    tasks = tasks_of(failure_rows)
    if not tasks:
        return None
    seed = int(cfg.get("evaluation", {}).get("seed", 0))
    results = rerun_many(cfg, state, trials, tasks, seed=seed)
    # 排序：净增益最大 → 改动面最小（同等收益下别改一堆）→ 补丁顺序
    best = max(results, key=lambda r: (r["net"], -len(r.get("targets") or []),
                                       -BODY_PATCH_ORDER.index(r["patch_key"])))
    if int(best["net"]) <= 0:
        return None  # 没有净增益就不提案：进化不是"为了改而改"

    tg = sorted(best["targets"])
    combined = best.get("kind") == "combined"
    if combined:
        new_bodies = dict(best["new_bodies"])
        old_bodies = {n: load_body(state.registry.get(n)) for n in tg}
        diff = "\n".join(unified_diff(old_bodies[n], new_bodies[n],
                                      from_name=f"a/{n}.md", to_name=f"b/{n}.md")
                         for n in tg)
        artifact_id = "+".join(tg)
        new_content = BODY_PATCH_LIBRARY[best["patch_key"]]
        # 组合候选的"内容"是多个正文，要按稳定序列化取摘要（content_sha256 只收 str）
        sha_src: Any = stable_json({n: new_bodies[n] for n in tg})
    else:
        name = tg[0]
        new_bodies = {name: best["new_body"]}
        diff = unified_diff(load_body(state.registry.get(name)), best["new_body"],
                            from_name=f"a/{name}.md", to_name=f"b/{name}.md")
        artifact_id = name
        new_content = best["new_body"]
        sha_src = best["new_body"]

    cand = Candidate(
        candidate_id=content_id("cand", state.release_id, "skill_body", artifact_id, sha_src),
        parent_release_id=state.release_id,
        artifact_type="skill_body",
        artifact_id=artifact_id,
        diff=diff,
        rationale=(f"归因 skill.misapplied：{len(failure_rows)} 条失败题目召回的是 "
                   f"{'、'.join(chr(39)+n+chr(39) for n in tg)}，但关键证据没进上下文。"
                   f"反事实重跑显示，在这 {len(tg)} 个 Skill 正文里追加"
                   f"「{BODY_PATCH_LIBRARY[best['patch_key']][:60]}…」这一步后，"
                   f"同一批题修好 {best['n_fixed']} 条、弄坏 {best['n_broke']} 条"
                   f"（净 +{best['net']}），是同类候选中最高。"),
        status="draft",
        new_content=new_content,
        attributed_class="skill.misapplied",
        source_failure_ids=[str(r.get("failure_id", "")) for r in failure_rows],
        generator={
            "backend": "deterministic-stub",
            "is_gepa": False,
            "selection": "counterfactual-rerun-net（多 Skill 联合重评）",
            "note": "测试替身，不产生效果结论；真实候选用 DSPy/GEPA（见 gepa_adapter.py）",
        },
        content_sha256=content_sha256(sha_src),
        extra={
            "skill_name": artifact_id,
            "skill_names": tg,
            "skill_bodies": new_bodies,     # 组合候选靠它落地（见 apply.registry_with）
            "patch_kind": best.get("kind", "single"),
            "patch_key": best["patch_key"],
            "step_added": BODY_PATCH_LIBRARY[best["patch_key"]],
            "behaviors_after": {n: behaviors_of_body(b) for n, b in sorted(new_bodies.items())},
            "counterfactual": {k: best[k] for k in
                               ("n", "base_correct", "trial_correct",
                                "n_fixed", "n_broke", "net")},
            "counterfactual_all": [
                {k: r[k] for k in ("patch_key", "n_fixed", "n_broke", "net")}
                | {"targets": sorted(r.get("targets") or []), "kind": r.get("kind", "single")}
                for r in results],
            "pre_evaluation_static_gate_rejections": static_gate_rejections,
        },
    )
    return cand
