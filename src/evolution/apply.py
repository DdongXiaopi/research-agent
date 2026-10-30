"""把候选应用到工件状态（M3：四类工件的统一入口）。

单独成模块的原因：round（正式评测）、ablation（消融组合）、counterfactual
（反事实重跑）三处都要"拿父状态 + 候选 → 得到候选状态"。三处各写一遍，
迟早会出现"消融里用的组合方式和正式评测不一致"这种隐蔽错误。
"""

from __future__ import annotations
import re

from typing import Any, Dict, Optional

from ..contracts.types import Candidate
from .fewshot import parse_demos_bundle
from .task_program import parse_task_program
from ..runtime.runner import ArtifactState
from ..runtime.skills import Registry, Skill, add_skill, with_body, with_description


def registry_with(registry: Registry, cand: Candidate) -> Registry:
    """只改 registry 的那部分（description / body / 新 Skill）。"""
    if cand.artifact_type == "skill_description":
        return with_description(registry, cand.artifact_id, cand.new_content,
                                list(cand.extra.get("new_triggers", [])) or None)
    if cand.artifact_type == "skill_body":
        # 【M4+】组合补丁：一次给多个 Skill 追加同一步（extra.skill_bodies 有值时生效）。
        # 单 Skill 候选没有这个字段，继续走 artifact_id + new_content 的老路径，行为不变。
        bodies = dict(cand.extra.get("skill_bodies") or {})
        if bodies:
            reg = registry
            for name in sorted(bodies):
                reg = with_body(reg, name, str(bodies[name]))
            return reg
        return with_body(registry, cand.artifact_id, cand.new_content)
    if cand.artifact_type == "new_skill":
        meta = dict(cand.extra.get("skill") or {})
        if (not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cand.artifact_id) or
                str(meta.get("name", cand.artifact_id)) != cand.artifact_id or
                meta.get("lifecycle", "active") != "active" or meta.get("replaced_by")):
            raise ValueError("new Skill metadata must match its name and active lifecycle")
        skill = Skill(
            name=str(meta.get("name", cand.artifact_id)),
            group=str(meta.get("group", "G-crystallized")),
            description=str(meta.get("description", "")),
            triggers=list(meta.get("triggers", [])),
            body_path="",
            frozen=bool(meta.get("frozen", False)),
            license=str(meta.get("license", "SELF-GENERATED")),
            body=cand.new_content,
            bundle_root=str(meta.get("bundle_root") or ""),
            bundle_files=dict(meta.get("bundle_files") or {}),
            dependencies=list(meta.get("dependencies") or []),
            permissions=list(meta.get("permissions") or []),
        )
        from ..runtime.skill_bundle import bundle_files
        bundle_files(skill)
        return add_skill(registry, skill)
    return registry


def state_with(state: ArtifactState, cand: Candidate) -> ArtifactState:
    """按候选类型构造"候选状态"：prompt 型换 prompt，其余换 registry。"""
    if cand.artifact_type == "task_program":
        program = parse_task_program(cand.new_content)
        return ArtifactState(release_id=state.release_id,
                             system_prompt=program["system_prompt"],
                             registry=state.registry,
                             task_demos=program["task_demos"])
    if cand.artifact_type == "system_prompt":
        return ArtifactState(release_id=state.release_id,
                             system_prompt=cand.new_content,
                             registry=state.registry,
                             task_demos=state.task_demos)
    return ArtifactState(release_id=state.release_id,
                         system_prompt=state.system_prompt,
                         registry=registry_with(state.registry, cand),
                         task_demos=(parse_demos_bundle(cand.new_content)
                                     if cand.artifact_type == "task_demos"
                                     else state.task_demos))


def combine(base: ArtifactState, cands: Any,
            kinds: Optional[Any] = None) -> ArtifactState:
    """把多个候选按维度组合到一个状态上（消融 A4 用）。

    `kinds` 指定要应用哪些工件类型；不指定则全部应用。
    """
    wanted = set(kinds) if kinds else None
    prompt = base.system_prompt
    registry = base.registry
    task_demos = base.task_demos
    for c in cands:
        if wanted is not None and c.artifact_type not in wanted:
            continue
        if c.artifact_type == "system_prompt":
            prompt = c.new_content
        elif c.artifact_type == "task_demos":
            task_demos = parse_demos_bundle(c.new_content)
        elif c.artifact_type == "task_program":
            program = parse_task_program(c.new_content)
            prompt = program["system_prompt"]
            task_demos = program["task_demos"]
        else:
            registry = registry_with(registry, c)
    return ArtifactState(release_id=base.release_id, system_prompt=prompt, registry=registry,
                         task_demos=task_demos)
