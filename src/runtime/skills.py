"""Skill registry 与渐进披露路由（对应 Pi harness/skills.ts 的能力）。

本模块**不是**重新实现 Skill 运行时（那是 Pi 的职责），只做三件事：
1. 从 registry.yaml 读出 skill 的 name/description/trigger（对应 Pi 的渐进披露：
   上下文里只放 name + description，命中后才加载正文）
2. 用一个可测试的纯函数做候选命中，并把命中结果写进 trace，供归因使用
3. 【M3】解析 Skill 正文里的程序步骤，映射成**行为开关**，交给运行时执行

关于第 3 点，务必如实理解（别在报告里说反）：
真实的 LLM 是"读到正文里的步骤 → 尽量照做"，本项目的离线模拟器做不到这一点。
这里做的是**结构化替代**：正文里出现"re-rank / compare each option / state the gap"
这类程序性措辞时，运行时确定性地切换对应的答题策略。它验证的是
「改正文 → 行为变 → 指标变」这条链路是否闭合，而不是"模型真的读懂了自然语言"。
真实模型接入后，`behaviors_of_body` 应被替换为"模型按正文执行"，
但归因/门禁/发布这些环节不变。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import yaml

from ..util import path, resolve_artifact, sha256_obj

# --- 正文 → 行为开关 -------------------------------------------------------
#
# 每个行为对应运行时可观测的一种策略。关键词刻意写得**具体**：
# "compare each option" 才触发逐项比对，泛泛的 "compare" 不触发 ——
# 否则随便写句 "compare the results" 就打开开关，进化就没有区分度了。
BEHAVIOR_PATTERNS: Dict[str, List[str]] = {
    "rerank": [r"re-?rank", r"rank the retrieved", r"reorder the retrieved"],
    "evidence": [r"compare (?:each|every) option", r"evaluate (?:each|every) option",
                 r"(?:each|every) option against", r"mark (?:each|every) option"],
    "citation": [r"\bcite\b", r"citation", r"source identifier", r"\bdoi\b"],
    "abstain": [r"state the gap", r"\babstain\b", r"insufficient evidence",
                r"no retrieved (?:source|passage)"],
}

_BEHAVIOR_RE: Dict[str, List[re.Pattern]] = {
    k: [re.compile(p, re.I) for p in pats] for k, pats in BEHAVIOR_PATTERNS.items()
}

_STEP_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.*\S)\s*$")
# 允许 `## Procedure` 与裸行的 `Procedure` 两种写法 —— 现有 SKILL.md 用的是后者，
# 只认 `#` 前缀会导致"解析出 0 步 → 追加时新建一个 `## Procedure` 小节"，
# 于是正文里出现两个 Procedure 小节，静态门按后者只数到 1 步（M3 实测踩到）。
_PROC_HEAD_RE = re.compile(r"^(?:#{1,6}\s*)?(?:procedure|steps|workflow)\s*:?\s*$", re.I)
_HEAD_RE = re.compile(r"^#{1,6}\s")


def parse_procedure(body: str) -> List[str]:
    """抽出正文里 `Procedure / Steps / Workflow` 小节下的编号（或 `-`）步骤。

    没有该小节时，退化成"全篇编号列表即步骤"，兼容简写的 SKILL.md。
    """
    lines = (body or "").splitlines()
    start: Optional[int] = None
    end = len(lines)
    for i, ln in enumerate(lines):
        s = ln.strip()
        if start is None:
            if _PROC_HEAD_RE.match(s):
                start = i + 1
        elif _HEAD_RE.match(s):
            end = i
            break
    segment = lines[start:end] if start is not None else lines
    steps: List[str] = []
    for ln in segment:
        m = _STEP_RE.match(ln)
        if m:
            steps.append(m.group(1).strip())
    return steps


def behaviors_of_body(body: str) -> List[str]:
    """正文 → 行为开关列表（已排序，保证可复现与可序列化）。"""
    steps = parse_procedure(body)
    text = " ".join(steps) if steps else (body or "")
    return sorted(k for k, pats in _BEHAVIOR_RE.items()
                  if any(p.search(text) for p in pats))


@dataclass
class Skill:
    name: str
    group: str
    description: str
    triggers: List[str] = field(default_factory=list)
    body_path: str = ""
    frozen: bool = False
    license: str = "UNVERIFIED"
    body: str = ""                 # 【M3】内联正文：进化中的 trial 状态用它，落盘时写到 body_path
    bundle_root: str = ""
    bundle_files: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    lifecycle: str = "active"
    replaced_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        # 键名必须与 registry.yaml 的 schema（`path`）一致，否则 release 落盘后
        # 再读回来正文就丢失了（M3 实测：to_dict 写 body_path、load 读 path → 空串）
        result = {
            "name": self.name,
            "group": self.group,
            "description": self.description,
            "triggers": list(self.triggers),
            "path": self.body_path,
            "frozen": self.frozen,
            "license": self.license,
        }
        if self.bundle_files:
            result.update({"bundle_root": self.bundle_root, "bundle_files": self.bundle_files,
                           "dependencies": self.dependencies, "permissions": self.permissions})
        if self.lifecycle != "active" or self.replaced_by:
            result.update({"lifecycle": self.lifecycle, "replaced_by": self.replaced_by})
        return result


@dataclass
class Registry:
    release_id: str
    skills: List[Skill]

    def to_dict(self) -> Dict[str, Any]:
        return {"release_id": self.release_id, "skills": [s.to_dict() for s in self.skills]}

    def fingerprint(self) -> str:
        # to_dict 只保存正文路径，试验态 with_body 的内联正文此前不进指纹，
        # 导致不同 Skill 候选复用同一 run_id/缓存目录。
        from .skill_bundle import bundle_identity
        bundles = {s.name: bundle_identity(s) for s in self.skills if s.bundle_files}
        return sha256_obj({"registry": self.to_dict(),
                           **({"bundles": bundles} if bundles else {}),
                           "effective_bodies": {s.name: load_body(s)
                                                for s in self.skills}})

    def get(self, name: str) -> Optional[Skill]:
        for s in self.skills:
            if s.name == name:
                return s
        return None


def load_registry(registry_path: str) -> Registry:
    with open(registry_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    skills = [
        Skill(
            name=s["name"],
            group=s.get("group", "G-common"),
            description=s.get("description", ""),
            triggers=list(s.get("triggers", [])),
            body_path=str(s.get("path") or s.get("body_path") or ""),
            frozen=bool(s.get("frozen", False)),
            license=s.get("license", "UNVERIFIED"),
            body=str(s.get("body", "") or ""),
            bundle_root=str(s.get("bundle_root") or ""),
            bundle_files=dict(s.get("bundle_files") or {}),
            dependencies=list(s.get("dependencies") or []),
            permissions=list(s.get("permissions") or []),
            lifecycle=str(s.get("lifecycle") or "active"),
            replaced_by=str(s.get("replaced_by") or ""),
        )
        for s in raw.get("skills", [])
    ]
    names = {skill.name for skill in skills}
    if len(names) != len(skills):
        raise ValueError("Skill names must be unique")
    for skill in skills:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", skill.name):
            raise ValueError("invalid Skill name")
        if skill.lifecycle not in {"active", "retired", "merged"}:
            raise ValueError("unknown Skill lifecycle")
        if skill.lifecycle == "merged" and (skill.replaced_by not in names or skill.replaced_by == skill.name):
            raise ValueError("merged Skill requires a distinct surviving Skill")
        if skill.replaced_by and (skill.replaced_by not in names or
                next(s for s in skills if s.name == skill.replaced_by).lifecycle != "active"):
            raise ValueError("replacement Skill must remain active")
    return Registry(release_id=str(raw.get("release_id", "unknown")), skills=skills)


def _stems(text: str) -> set:
    """取词干前缀（长度 5），避免 resistant/resistance 这类形态差导致误判未命中。"""
    return {t[:5] for t in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", (text or "").lower()) if len(t) >= 4}


def route(query: str, registry: Registry, top_n: int = 3) -> List[Dict[str, Any]]:
    """渐进披露的"候选"阶段：只用 name/description 打分，不读正文。"""
    q = _stems(query)
    scored = []
    for s in registry.skills:
        if s.lifecycle != "active":
            continue
        hay = _stems(s.description + " " + " ".join(s.triggers))
        hit = len(q & hay)
        scored.append({"name": s.name, "score": hit, "group": s.group, "description": s.description})
    scored.sort(key=lambda d: (-d["score"], d["name"]))
    return scored[:top_n]


def render_catalog(registry: Registry, max_chars: int = 16000) -> str:
    catalog = "\n".join(f"{s.name}: {s.description}" for s in registry.skills if s.lifecycle == "active")
    if max_chars < 1 or len(catalog) > max_chars:
        raise ValueError("Skill catalog exceeds context budget; review merge/retirement proposals")
    return catalog


def recall(query: str, registry: Registry, min_score: int = 2) -> Optional[Dict[str, Any]]:
    """返回命中的 skill（score >= min_score 的最高者），未命中返回 None。

    min_score 默认 **2**（M2 调整）：
    min_score=1 时，只要题干/选项里任意一个实词撞上 description 就算命中，
    实测召回率 ~92%~100% —— Skill 机制在夹具里"永远命中"，description 进化因此
    完全没有可观测效果（A2 消融恒为 0）。提到 2 之后召回率落到有区分度的区间
    （litqa2 ~44%、protocolqa ~97%），"改 description → 召回 → 检索质量 → 正确率"
    这条链才真实可测。该阈值由 `skills.recall_min_score` 配置。
    """
    cands = route(query, registry, top_n=1)
    if not cands or cands[0]["score"] < min_score:
        return None
    return cands[0]


def select_skill(query: str, registry: Registry, min_score: int = 2,
                 pinned_skill: str = "") -> Optional[Dict[str, Any]]:
    """Select by ordinary routing or an explicit, protocol-wide Skill pin.

    A pin is useful for testing a Skill body on a known task domain. It is a
    protocol setting applied to both arms, never inferred from an answer key.
    """
    if not pinned_skill:
        return recall(query, registry, min_score=min_score)
    skill = registry.get(pinned_skill)
    if skill is not None and skill.lifecycle != "active":
        raise ValueError("pinned Skill is retired or merged")
    if skill is None:
        raise ValueError(f"pinned Skill is absent from registry: {pinned_skill}")
    return {"name": skill.name, "score": None, "group": skill.group,
            "description": skill.description, "source": "protocol_pin"}


def context_block(registry: Registry) -> str:
    """上下文里只放 name + description（FR-003）。"""
    block = "\n".join(f"- {s.name}: {s.description}" for s in registry.skills if s.lifecycle == "active")
    if len(block) > 16000:
        raise ValueError("Skill catalog exceeds disclosure budget")
    return block


def _clone(registry: Registry) -> Registry:
    import copy

    return Registry(release_id=registry.release_id, skills=copy.deepcopy(registry.skills))


def with_description(registry: Registry, name: str, description: str,
                     triggers: Optional[List[str]] = None) -> Registry:
    """返回一个**新的** Registry，其中指定 skill 的 description/triggers 被替换。

    不改原对象：进化过程中需要同时持有"父状态"与"候选状态"做对比。
    """
    new = _clone(registry)
    tgt = new.get(name)
    if tgt is None:
        raise KeyError(f"registry 中没有 skill: {name}")
    tgt.description = description
    if triggers is not None:
        tgt.triggers = list(triggers)
    return new


def with_body(registry: Registry, name: str, body: str) -> Registry:
    """【M3】返回新 Registry，其中指定 skill 的正文被替换（内联，落盘由 release 负责）。"""
    new = _clone(registry)
    tgt = new.get(name)
    if tgt is None:
        raise KeyError(f"registry 中没有 skill: {name}")
    tgt.body = body
    return new


def add_skill(registry: Registry, skill: Skill) -> Registry:
    """【M3】返回新 Registry，追加一个结晶出来的 Skill（按名字去重）。"""
    new = _clone(registry)
    if new.get(skill.name) is not None:
        raise ValueError(f"registry 中已存在同名 skill: {skill.name}")
    new.skills.append(skill)
    new.skills.sort(key=lambda s: s.name)
    return new


def load_body(skill: Skill, base: Optional[str] = None) -> str:
    """内联正文优先；否则按 body_path 读盘（相对**产物根**解析）。

    P1：`body_path` 存的是形如 `artifacts/releases/rel-xxx/skills/bodies/y.md`
    的相对路径。这里必须走 `resolve_artifact()`（按 `artifacts_root()` 解析），
    而不是 `path()`（按仓库根解析）—— 否则 `RSI_HOME` 一旦把产物重定向
    （测试沙箱、CI），就会读空。
    """
    if skill.body:
        return skill.body
    if not skill.body_path:
        return ""
    p = resolve_artifact(skill.body_path) if base is None else path(base, skill.body_path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def _procedure_span(lines: List[str]) -> tuple:
    """返回 Procedure 小节的行区间 [start, end)；没有该小节则 (None, None)。"""
    start: Optional[int] = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if start is None:
            if _PROC_HEAD_RE.match(s):
                start = i + 1
        elif _HEAD_RE.match(s):
            return start, i
    return (start, len(lines)) if start is not None else (None, None)


def _last_step(lines: List[str], start: int, end: int) -> tuple:
    """在 [start,end) 里找最后一步：返回 (行号, 编号, 是否 bullet)。"""
    last_idx: Optional[int] = None
    last_num = 0
    bullet = False
    for i in range(start, end):
        if _STEP_RE.match(lines[i]):
            last_idx = i
            head = lines[i].strip()
            if head[0].isdigit():
                m = re.match(r"^\s*(\d+)", lines[i])
                last_num = int(m.group(1)) if m else 0
            else:
                bullet = True
    return last_idx, last_num, bullet


def append_step(body: str, step: str) -> str:
    """把一步追加到正文的步骤列表末尾，并维持连续编号。

    正文是给人看的 SKILL.md，编号必须连续（否则"第 3 步"会指向错的地方），
    所以这里按最后一步的实际编号 +1，而不是无脑 append。

    没有 `Procedure` 小节时（例如小节名叫 "Output shape"）不新建小节，
    而是追加到全篇最后一个编号步骤之后 —— 新建小节会让正文出现两处步骤列表，
    静态门按新小节只数到 1 步，反而把合法候选拦掉（M3 实测踩到）。
    """
    lines = (body or "").splitlines()
    start, end = _procedure_span(lines)
    if start is None:
        last_idx, last_num, bullet = _last_step(lines, 0, len(lines))
        if last_idx is None:
            return (body or "").rstrip() + "\n\n## Procedure\n1. " + step + "\n"
    else:
        last_idx, last_num, bullet = _last_step(lines, start, end)

    if last_idx is None:
        lines.insert(end, "1. " + step)
    elif bullet:
        lines.insert(last_idx + 1, "- " + step)
    else:
        lines.insert(last_idx + 1, f"{last_num + 1}. " + step)
    return "\n".join(lines) + "\n"


def behaviors_of_skill(skill: Skill, base: Optional[str] = None) -> List[str]:
    return behaviors_of_body(load_body(skill, base))
