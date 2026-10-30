"""发布与回滚（主线简化版，Spec §3.1 / FR-206）。

一个 release = 一份可进化工件的不可变快照：
    artifacts/releases/<release_id>/
        release.json          血缘：parent、候选、diff、指标、sha256
        system_prompt.txt     当前生效的 System Prompt
        skills/registry.yaml  Skill 注册表快照
    artifacts/releases/ACTIVE  单指针文件，指向当前生效 release

回滚 = 把 ACTIVE 指向旧 release（单命令、可故障注入测试）。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..contracts.types import Candidate, content_sha256
from . import review as rv
from ..runtime.runner import ArtifactState
from ..evolution.fewshot import parse_demos_bundle, serialize_demos_bundle
from ..evolution.task_program import (changed_task_program_components,
                                      parse_task_program)
from ..runtime.skills import (Registry, Skill, add_skill, load_body, load_registry, with_body,
                              with_description)
from ..runtime.skill_bundle import bundle_identity, bundle_files, materialize_bundle
from ..util import (ensure_dir, now_iso, artifacts_path, artifacts_root,
                    repo_root, path, read_json, sha256_obj, write_json)

RELEASES_ROOT = artifacts_path("releases")
ACTIVE_FILE = RELEASES_ROOT / "ACTIVE"
# 【决策 4】champion / challenger / archived（对齐 MLflow Model Registry 的 alias 机制）
#
# 为什么需要它：原来只有 `ACTIVE` 一个指针，"发布"和"生效"是同一个动作，
# 于是"当前版本不够好"这个判断只能靠 **rollback**（把指针拨回出厂版本）——
# 那等于把进化成果全丢了。MLflow 的做法是 alias：
#   champion（当前生效）/ challenger（待验证）/ archived（保留但不再指向）
#   promotion = 重新指向别名；rollback = 把别名指回旧版本（秒级，不是重新部署）。
# 本项目照抄：`ACTIVE` 仍是唯一**生效**指针（所有既有路径不变），
# alias 是**治理层**：它让"先当 challenger 验证、达标再 repoint"成为可能，
# 也让被换下的版本进 archived 而不是消失 —— 回滚永远只是改指针。
ALIASES_FILE = RELEASES_ROOT / "ALIASES.json"
BASELINE_PROMPT = path("configs", "prompts", "system_prompt.v0.txt")
BASELINE_REGISTRY = path("skills", "registry.yaml")


class ReleaseError(RuntimeError):
    pass


def _dir(release_id: str) -> Path:
    return RELEASES_ROOT / release_id


def list_releases() -> List[str]:
    if not RELEASES_ROOT.exists():
        return []
    return sorted(p.name for p in RELEASES_ROOT.iterdir() if p.is_dir())


def current_release_id() -> str:
    _recover_activation()
    if not ACTIVE_FILE.exists():
        raise ReleaseError("没有 ACTIVE 指针，请先运行 `scripts/rsi.py init` 建立 baseline-v0 发布")
    release_id = ACTIVE_FILE.read_text(encoding="utf-8").strip()
    if not release_id:
        raise ReleaseError("当前 release 已暂停；请人工 rollback 或晋升已验证版本")
    return release_id


def _recover_activation() -> None:
    journal = RELEASES_ROOT / "activation-pending.json"
    if journal.exists():
        desired = read_json(journal)
        _write_aliases(desired["aliases"])
        write_active(desired["active"])
        journal.unlink()


def _activate_release(release_id: str, aliases: Dict[str, Any]) -> None:
    """One durable desired state for ACTIVE and aliases, replayable after crash."""
    if aliases.get("champion") != (release_id or None):
        raise ReleaseError("activation alias must match ACTIVE")
    journal = RELEASES_ROOT / "activation-pending.json"
    write_json(journal, {"active": release_id, "aliases": aliases})
    _recover_activation()


def write_active(release_id: str) -> None:
    ensure_dir(RELEASES_ROOT)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n",
                dir=ACTIVE_FILE.parent, prefix=ACTIVE_FILE.name + ".",
                suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(release_id + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, ACTIVE_FILE)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def init_baseline(prompt_path: Path | None = None, registry_path: Path | None = None,
                  release_id: str = "baseline-v0", note: str | None = None) -> str:
    prompt_path = Path(prompt_path or BASELINE_PROMPT)
    registry_path = Path(registry_path or BASELINE_REGISTRY)
    if not prompt_path.exists():
        raise ReleaseError(f"基线 System Prompt 不存在: {prompt_path}")
    if not registry_path.exists():
        raise ReleaseError(f"Skill registry 不存在: {registry_path}")

    d = _dir(release_id)
    if d.exists():
        # A release ID denotes an immutable snapshot. Repeated initialization
        # may reactivate identical content, but must never overwrite a changed
        # Prompt, registry or Skill body under the same ID.
        existing = load_state(release_id)
        source = load_registry(str(registry_path))

        def comparable(registry: Registry) -> Dict[str, Any]:
            return {s.name: {"group": s.group, "description": s.description,
                             "triggers": list(s.triggers), "frozen": s.frozen,
                             "lifecycle": s.lifecycle, "replaced_by": s.replaced_by,
                             "license": s.license,
                             "body_sha256": content_sha256(load_body(s))}
                             | ({"bundle_identity": bundle_identity(s)}
                                if s.bundle_files else {})
                    for s in registry.skills}

        if (existing.system_prompt != prompt_path.read_text(encoding="utf-8") or
                comparable(existing.registry) != comparable(source)):
            raise ReleaseError(f"release ID 已存在且内容不同，禁止覆盖不可变快照: {release_id}")
        write_active(release_id)
        return release_id
    ensure_dir(d / "skills")
    shutil.copyfile(prompt_path, d / "system_prompt.txt")
    registry = load_registry(str(registry_path))
    # The source registry points at mutable workspace files. Snapshot every
    # effective body so this baseline can actually be replayed later.
    for skill in registry.skills:
        skill.body = load_body(skill)
    registry = _materialize_bodies(registry, release_id)
    registry.release_id = release_id
    with open(d / "skills" / "registry.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(registry.to_dict(), f, allow_unicode=True, sort_keys=True)
    write_json(d / "release.json", {
        "release_id": release_id,
        "parent": None,
        "created_at": now_iso(),
        "artifacts": {
            "system_prompt_sha256": sha256_obj(prompt_path.read_text(encoding="utf-8")),
            "skills_registry_sha256": sha256_obj(registry.to_dict()),
            "skills_body_sha256": {s.name: content_sha256(load_body(s))
                                   for s in registry.skills},
        },
        "candidate": None,
        "note": note or "M1 基线（A0）：未进化的 v0",
    })
    write_active(release_id)
    return release_id


def load_state(release_id: Optional[str] = None) -> ArtifactState:
    rid = release_id or current_release_id()
    d = _dir(rid)
    if not d.exists():
        raise ReleaseError(f"release 不存在: {rid}")
    prompt = (d / "system_prompt.txt").read_text(encoding="utf-8")
    registry = load_registry(str(d / "skills" / "registry.yaml"))
    # Verifies dependencies as well as inline bodies, including older snapshots.
    for skill in registry.skills:
        bundle_files(skill)
    artifacts = (read_json(d / "release.json").get("artifacts") or {})
    expected_bodies = artifacts.get("skills_body_sha256")
    if expected_bodies is not None:
        if sha256_obj(prompt) != artifacts.get("system_prompt_sha256"):
            raise ReleaseError(f"System Prompt 快照哈希不匹配: {rid}")
        actual_bodies = {s.name: content_sha256(load_body(s))
                         for s in registry.skills}
        if actual_bodies != expected_bodies:
            raise ReleaseError(f"Skill 正文快照哈希不匹配: {rid}")
        if sha256_obj(registry.to_dict()) != artifacts.get("skills_registry_sha256"):
            raise ReleaseError(f"Skill 注册表快照哈希不匹配: {rid}")
    demos_path = d / "task_demos.json"
    task_demos = []
    if demos_path.exists():
        raw_demos = demos_path.read_text(encoding="utf-8")
        expected_demos = artifacts.get("task_demos_sha256")
        if expected_demos and sha256_obj(raw_demos) != expected_demos:
            raise ReleaseError(f"Task demo 快照哈希不匹配: {rid}")
        try:
            task_demos = parse_demos_bundle(raw_demos)
        except ValueError as exc:
            raise ReleaseError(f"Task demo 快照无法解析: {rid}: {exc}") from exc
    return ArtifactState(release_id=rid, system_prompt=prompt, registry=registry,
                         task_demos=task_demos)


def _materialize_bodies(registry: Registry, release_id: str) -> Registry:
    """把进化过的正文（内联）落盘到 release 目录，并把 path 改指过去。

    为什么必须落盘：release 是**不可变快照**，正文不能继续指向 `skills/main/*.md`
    —— 那个文件是可变的工作区文件，指过去就等于"发布内容会被后续编辑改写"。
    """
    import copy

    out = Registry(release_id=registry.release_id, skills=copy.deepcopy(registry.skills))
    d = _dir(release_id)
    bodies = d / "skills" / "bodies"
    for s in out.skills:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", s.name):
            raise ReleaseError("invalid Skill snapshot name")
        if s.bundle_files:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", s.name):
                raise ReleaseError("invalid bundled Skill name")
            materialize_bundle(s, d / "skills" / "bundles" / s.name)
            s.bundle_root = f"artifacts/releases/{release_id}/skills/bundles/{s.name}"
        body = s.body or load_body(s)
        if not body and s.body_path:
            raise ReleaseError(f"Skill 正文路径存在但不可读取，不能建立不可变快照: {s.name}")
        if not body:
            continue
        ensure_dir(bodies)
        (bodies / f"{s.name}.md").write_text(body, encoding="utf-8")
        s.body_path = f"artifacts/releases/{release_id}/skills/bodies/{s.name}.md"
        s.body = ""
    return out


def publish(parent_release_id: str, candidate: Candidate, metrics: Dict[str, Any],
            new_registry: Optional[Registry] = None) -> str:
    """把已批准的候选落成新 release，并移动 ACTIVE 指针。

    【M1/M2 回顾】父指针必须是**当前生效的 release**，不能是候选自己记的那个。
    事故：跨轮候选池把第 1 轮度量于 `baseline-v0` 的候选在第 3 轮发布，
    `parent_release_id` 取了候选里的旧值 → 新 release 的父被设成 baseline-v0，
    前两轮的正文补丁与提示词一并丢失，而 campaign 报告每一轮都写着 published=ok。
    发布链断裂却没有任何报错，是最难发现的一类 bug —— 所以这里 fail-closed。
    """
    if new_registry is not None:
        raise ReleaseError(
            "unreviewed new_registry overrides are unsupported; derive registry changes from the approved candidate")
    active = current_release_id() if ACTIVE_FILE.exists() else None
    if active and str(parent_release_id) != str(active):
        raise ReleaseError(
            f"父指针与当前生效 release 不一致：候选记的是 {parent_release_id}，"
            f"当前是 {active}。按候选记的父发布会让发布链静默回退（丢失中间轮次的改动）。"
            "若确需跨轮复用，请先在当前 release 上重新度量该候选。")
    if candidate.artifact_type not in ("system_prompt", "skill_description",
                                       "skill_body", "new_skill", "task_program"):
        raise ReleaseError(
            "FR-201 只支持发布 system_prompt / skill_description / skill_body / new_skill 候选 "
            "（task_program 仅作 system_prompt 候选的存储封装）；"
            f"收到 {candidate.artifact_type}"
        )
    try:
        req = rv.read_request(candidate.candidate_id)
        review_path = artifacts_path("reviews", f"{candidate.candidate_id}.review.json")
        record = read_json(review_path)
    except (rv.ReviewError, OSError, ValueError) as exc:
        raise ReleaseError(f"发布前缺少有效 review 请求与批准记录: {exc}") from exc
    actual_sha = content_sha256(candidate.new_content)
    if req.get("candidate_artifact_sha256") != rv.candidate_artifact_sha256(candidate):
        raise ReleaseError("review 请求：候选正文或元数据与评审身份不一致；旧请求须重新生成并评审")
    cost_exception_required = bool(
        req.get("evidence_type") == "aggregate_selection_gate_v1"
        and ((req.get("evaluation_summary") or {}).get("cost_exception_reviewable") is True))
    exception_approved = bool(
        (record.get("metrics") or {}).get("cost_overrun_exception_accepted") is True)
    approved_request_sha = (record.get("metrics") or {}).get("review_request_sha256")
    if not approved_request_sha or approved_request_sha != rv.request_sha256(req):
        raise ReleaseError("review 请求在批准后发生变化，必须重新评审")
    if (req.get("candidate_id") != candidate.candidate_id or
            req.get("parent_release_id") != parent_release_id or
            req.get("diff") != candidate.diff or
            req.get("new_content_sha256") != actual_sha or
            not rv.request_effect_gate_ok(req) or
            (candidate.content_sha256 and candidate.content_sha256 != actual_sha) or
            not (req.get("static_gate") or {}).get("passed")):
        raise ReleaseError("review 请求与候选、父版本或静态门不一致")
    approved_metrics = {
        "optimize": req.get("optimize_metrics", {}),
        "gate": req.get("gate_metrics", {}),
    }
    if metrics != approved_metrics:
        raise ReleaseError("release metrics differ from the metrics approved in the review request")
    if cost_exception_required != exception_approved:
        raise ReleaseError("成本超支例外必须在人工批准记录中显式确认")
    if (record.get("decision") != "approve" or
            record.get("subject_id") != candidate.candidate_id or
            record.get("subject_sha256") != actual_sha or
            record.get("parent_release_id") != parent_release_id or
            record.get("diff") != candidate.diff or
            not str(record.get("reviewer") or "").strip()):
        raise ReleaseError("发布必须有与当前候选完全一致的批准记录")
    if record["reviewer"] == "auto-sim" and (
            not os.environ.get("RSI_HOME") or
            artifacts_root().resolve() == (repo_root() / "artifacts").resolve()):
        raise ReleaseError("模拟批准记录只能在 RSI_HOME 指定的独立产物目录发布")
    parent = load_state(parent_release_id)
    if candidate.artifact_type == "task_program":
        try:
            changed_components = changed_task_program_components(
                candidate.new_content, parent)
        except (TypeError, ValueError) as exc:
            raise ReleaseError(f"task_program parent comparison failed: {exc}") from exc
        gate_rows = list((req.get("static_gate") or {}).get("results") or [])
        component_gate = next((row for row in gate_rows
                               if row.get("check") == "task_program_single_component"), None)
        if (changed_components != ["system_prompt"] or component_gate is None
                or component_gate.get("passed") is not True
                or component_gate.get("detail") != "system_prompt"):
            raise ReleaseError(
                "FR-201 task_program release must change only system_prompt and carry "
                "a current parent-relative single-component gate")
    release_id = f"rel-{candidate.candidate_id}"
    d = _dir(release_id)
    ensure_dir(d / "skills")

    if candidate.artifact_type == "system_prompt":
        (d / "system_prompt.txt").write_text(candidate.new_content, encoding="utf-8")
        new_registry = new_registry or parent.registry
    elif candidate.artifact_type == "skill_description":
        # 改 description：prompt 原样继承，只替换目标 skill 的 description/triggers
        (d / "system_prompt.txt").write_text(parent.system_prompt, encoding="utf-8")
        new_registry = new_registry or with_description(
            parent.registry,
            candidate.artifact_id,
            candidate.new_content,
            list(candidate.extra.get("new_triggers", [])) or None,
        )
    elif candidate.artifact_type == "skill_body":
        # 【M3】改正文：prompt 与 description 都不动，只替换目标 skill 的 body
        # 【M4+】组合补丁（extra.skill_bodies）：一次替换多个 skill 的 body
        (d / "system_prompt.txt").write_text(parent.system_prompt, encoding="utf-8")
        bodies = dict(candidate.extra.get("skill_bodies") or {})
        if bodies:
            reg = parent.registry
            for name in sorted(bodies):
                reg = with_body(reg, name, str(bodies[name]))
            new_registry = new_registry or reg
        else:
            new_registry = new_registry or with_body(parent.registry, candidate.artifact_id,
                                                     candidate.new_content)
    elif candidate.artifact_type == "task_program":
        try:
            program = parse_task_program(candidate.new_content)
        except ValueError as exc:
            raise ReleaseError(f"invalid task_program candidate: {exc}") from exc
        (d / "system_prompt.txt").write_text(
            program["system_prompt"], encoding="utf-8")
        (d / "task_demos.json").write_text(
            serialize_demos_bundle(program["task_demos"]), encoding="utf-8")
        new_registry = new_registry or parent.registry
    else:
        # 【M3】结晶新 Skill：registry 里多一个成员
        (d / "system_prompt.txt").write_text(parent.system_prompt, encoding="utf-8")
        meta = dict(candidate.extra.get("skill") or {})
        if not meta.get("name"):
            raise ReleaseError("new_skill 候选缺少 extra.skill 元数据，无法发布")
        from ..evolution.apply import registry_with
        new_registry = registry_with(parent.registry, candidate)

    src_registry = _dir(parent_release_id) / "skills" / "registry.yaml"
    shutil.copyfile(src_registry, d / "skills" / "registry.yaml")
    parent_demos = _dir(parent_release_id) / "task_demos.json"
    if candidate.artifact_type != "task_program" and parent_demos.exists():
        shutil.copyfile(parent_demos, d / "task_demos.json")
    if new_registry is not None:
        new_registry = _materialize_bodies(new_registry, release_id)
        new_registry.release_id = release_id
        with open(d / "skills" / "registry.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(new_registry.to_dict(), f, allow_unicode=True, sort_keys=True)

    prompt_sha = sha256_obj((d / "system_prompt.txt").read_text(encoding="utf-8"))
    published_registry = load_registry(str(d / "skills" / "registry.yaml"))
    demos_path = d / "task_demos.json"
    demos_sha = (sha256_obj(demos_path.read_text(encoding="utf-8"))
                 if demos_path.exists() else None)
    write_json(d / "release.json", {
        "release_id": release_id,
        "parent": parent_release_id,
        "created_at": now_iso(),
        "candidate": candidate.to_dict(),
        "metrics": metrics,
        "review": record,
        "evidence_type": req.get("evidence_type", "round_evaluation"),
        "engineering_only": bool((req.get("evaluation_summary") or {}).get("engineering_only")),
        "artifacts": {
            "system_prompt_sha256": prompt_sha,
            "skills_registry_sha256": sha256_obj(published_registry.to_dict()),
            "skills_body_sha256": {s.name: content_sha256(load_body(s))
                                   for s in published_registry.skills},
            **({"task_demos_sha256": demos_sha} if demos_sha else {}),
        },
        "rollback_point": parent_release_id,
    })
    aliases = load_aliases()
    old_champion = aliases.get("champion")
    aliases["champion"] = release_id
    if aliases.get("challenger") == release_id:
        aliases["challenger"] = None
    if old_champion != release_id:
        aliases["champion_notes"] = []
        if old_champion:
            aliases["archived"] = sorted(set(aliases.get("archived", []) + [old_champion]))
    aliases.setdefault("history", []).append({"at": now_iso(), "action": "publish",
        "from": old_champion, "to": release_id})
    _activate_release(release_id, aliases)
    return release_id


def load_aliases() -> Dict[str, Any]:
    """读取 alias 治理层；不存在时按当前 ACTIVE 现场推算（不写文件）。"""
    _recover_activation()
    if ALIASES_FILE.exists():
        try:
            return json.loads(ALIASES_FILE.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - 文件损坏时按空账本处理
            raise ReleaseError(f"ALIASES.json 无法解析：{exc}")
    champ = current_release_id() if ACTIVE_FILE.exists() else None
    return {
        "champion": champ,
        "challenger": None,
        "archived": [r for r in list_releases() if r != champ],
        "history": [],
        "migrated": False,
        "note": "尚未建立 alias 文件（本次为按当前 ACTIVE 推算的视图）",
    }


def ensure_aliases(actor: str = "system", reason: str = "初始化 champion/challenger 治理层") -> Dict[str, Any]:
    """建立 alias 文件（幂等）。首次调用把当前 ACTIVE 记为 champion。"""
    if ALIASES_FILE.exists():
        return load_aliases()
    champ = current_release_id() if ACTIVE_FILE.exists() else None
    if champ is None:
        raise ReleaseError("没有 ACTIVE 指针，请先运行 `scripts/rsi.py init`")
    data = {
        "champion": champ,
        "challenger": None,
        "archived": [r for r in list_releases() if r != champ],
        "history": [{"at": now_iso(), "action": "adopt_champion",
                     "release": champ, "actor": actor, "reason": reason}],
        "migrated": True,
        "note": "champion=当前 ACTIVE；旧版本一律 archived 保留（回滚=改指针，秒级）",
    }
    ensure_dir(RELEASES_ROOT)
    ALIASES_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    return data


def _write_aliases(data: Dict[str, Any]) -> None:
    ensure_dir(RELEASES_ROOT)
    write_json(ALIASES_FILE, data)


def get_alias(name: str) -> Optional[str]:
    return (load_aliases() or {}).get(name)


def set_alias(name: str, release_id: str, actor: str = "system",
              reason: str = "") -> Dict[str, Any]:
    """设置 champion / challenger（challenger 只登记，不生效）。"""
    if name not in ("champion", "challenger"):
        raise ReleaseError(f"未知 alias：{name}（只允许 champion / challenger；"
                           "archived 由 promote/archive 维护）")
    if release_id not in list_releases():
        raise ReleaseError(f"目标 release 不存在: {release_id}")
    data = ensure_aliases(actor=actor)
    before = data.get(name)
    data[name] = release_id
    data.setdefault("history", []).append({
        "at": now_iso(), "action": f"set_{name}", "from": before, "to": release_id,
        "actor": actor, "reason": reason,
    })
    _write_aliases(data)
    return {"alias": name, "from": before, "to": release_id, "actor": actor,
            "reason": reason, "at": now_iso()}


def note_champion(note: str, actor: str = "system") -> Dict[str, Any]:
    """给当前 champion 挂 caveat（决策 4：保持 champion 的前提是**披露**）。

    为什么不写进文档就算了：champion 会被 promote 换掉、也会被 rollback 拨回，
    挂在版本上的 caveats 必须跟着版本走 —— 换人时自动清空，避免"新版本的旧警告"。
    """
    if not note:
        raise ReleaseError("caveat 不能为空")
    data = ensure_aliases(actor=actor)
    data.setdefault("champion_notes", [])
    seq = max([int(n.get("id") or 0) for n in data["champion_notes"]] or [0]) + 1
    data["champion_notes"].append({"id": seq, "at": now_iso(), "actor": actor,
                                   "note": note, "champion": data.get("champion")})
    _write_aliases(data)
    return {"champion": data.get("champion"), "notes": data["champion_notes"]}


def resolve_caveat(match: Any, actor: str = "system", reason: str = "") -> Dict[str, Any]:
    """把某条 caveat 标记为**已解决**（不删除，留痕）。

    caveat 挂上去之后必须能闭环：否则"待办事项"会变成永久噪声，
    真正还开着的问题反而被淹没。按 id 或按子串匹配未解决的条目。
    """
    data = ensure_aliases(actor=actor)
    notes = data.setdefault("champion_notes", [])
    for i, n in enumerate(notes, 1):
        n.setdefault("id", i)          # 迁移：加 id 之前挂的条目没有 id
    key = str(match)
    hit = [n for n in notes
           if not n.get("resolved") and (key == str(n.get("id")) or key in str(n.get("note", "")))]
    if not hit:
        raise ReleaseError(f"没有匹配到未解决的 caveat: {match}")
    for n in hit:
        n["resolved"] = {"at": now_iso(), "actor": actor, "reason": reason}
    _write_aliases(data)
    return {"champion": data.get("champion"), "resolved": [n.get("id") for n in hit],
            "open": [n.get("id") for n in notes if not n.get("resolved")]}


def archive(release_id: str, actor: str = "system", reason: str = "") -> Dict[str, Any]:
    """把 release 标记为 archived（**永不删除**，回滚要从这里取最近版本）。"""
    data = ensure_aliases(actor=actor)
    arch = [r for r in data.get("archived", []) if r != release_id]
    arch.append(release_id)
    data["archived"] = sorted(set(arch))
    data.setdefault("history", []).append(
        {"at": now_iso(), "action": "archive", "release": release_id,
         "actor": actor, "reason": reason})
    _write_aliases(data)
    return {"archived": release_id, "archived_all": data["archived"]}


def promote_challenger(reviewer: str, reason: str,
                       promotion: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """把 challenger 升为 champion（**移动指针**，秒级；旧 champion 进 archived）。

    `promotion` 必须是 `governance.promotion.promotion_decision` 的完整输出；
    不接受缺失 gate 或只有 `promotable=true` 的手工拼装摘要。
    """
    if not isinstance(promotion, dict):
        raise ReleaseError("晋升必须提供完整发布门报告；不能跳过 gate")
    if promotion.get("promotable") is not True:
        raise ReleaseError("发布门未通过，禁止晋升：" + "; ".join(promotion.get("blocking") or []))
    checks = promotion.get("checks")
    if not isinstance(checks, list):
        raise ReleaseError("发布门报告缺少结构化 checks")
    check_names = [str(row.get("check")) for row in checks if isinstance(row, dict)]
    if len(check_names) != len(checks) or len(check_names) != len(set(check_names)):
        raise ReleaseError("发布门报告 checks 无效或包含重复项")
    by_name = {str(row.get("check")): row for row in checks if isinstance(row, dict)}
    protocol = by_name.get("protocol_same_origin")
    main_result = by_name.get("main_dataset_results")
    secondary = by_name.get("secondary_tolerances")
    dataset_checks = [row for name, row in by_name.items()
                      if name.startswith("dataset:")
                      and (row.get("detail") or {}).get("role") == "main"]
    if (protocol is None or protocol.get("passed") is not True
            or main_result is None or main_result.get("passed") is not True
            or not dataset_checks or any(row.get("passed") is not True
                                         for row in dataset_checks)
            or secondary is None or secondary.get("passed") is not True):
        raise ReleaseError(
            "发布门报告必须包含协议状态、主数据集结果和全部通过的次指标检查")
    if not reviewer or not reason:
        raise ReleaseError("晋升必须带 reviewer 与 reason（可审计；对齐 MLflow 的 sign-off）")
    data = ensure_aliases(actor=reviewer)
    challenger = data.get("challenger")
    if not challenger:
        raise ReleaseError("当前没有 challenger；先用 `rsi.py alias --set challenger <release>`")
    if challenger not in list_releases():
        raise ReleaseError(f"challenger release 不存在: {challenger}")
    before = data.get("champion")
    if promotion.get("challenger_release_id") != challenger:
        raise ReleaseError("发布门报告对应的 challenger 与当前 alias 不一致")
    if promotion.get("champion_release_id") != before:
        raise ReleaseError("发布门报告对应的 champion 已变化，必须重新评测")
    data["champion"] = challenger
    data["challenger"] = None
    # caveats 挂的是**那个版本**，换人就清空 —— 否则会变成"新版本顶着旧警告"
    data["champion_notes"] = []
    if before and before != challenger:
        data["archived"] = sorted(set(data.get("archived", []) + [before]))
    data.setdefault("history", []).append({
        "at": now_iso(), "action": "promote", "from": before, "to": challenger,
        "reviewer": reviewer, "reason": reason,
        "gate": {"promotable": promotion.get("promotable"),
                 "blocking": promotion.get("blocking")},
    })
    _activate_release(challenger, data)
    return {"from": before, "to": challenger, "reviewer": reviewer, "reason": reason,
            "archived": data["archived"], "at": now_iso()}


def demote_champion(reviewer: str, reason: str) -> Dict[str, Any]:
    """把当前 champion 降为 challenger（决策 4：复核发现问题时降级，**不删除**）。"""
    data = ensure_aliases(actor=reviewer)
    champ = data.get("champion")
    if not champ:
        raise ReleaseError("当前没有 champion")
    data["challenger"] = champ
    fallback = release_info(champ).get("parent")
    if fallback not in list_releases() or fallback == champ:
        fallback = None
    data["champion"] = fallback
    data["champion_notes"] = []
    data.setdefault("history", []).append({
        "at": now_iso(), "action": "demote", "release": champ,
        "reviewer": reviewer, "reason": reason})
    _activate_release(fallback or "", data)
    return {"demoted": champ, "now_challenger": champ, "reviewer": reviewer,
            "reason": reason, "active_release": fallback,
            "note": "已回退到父版本" if fallback else "已暂停生效；请人工 rollback 或晋升新候选"}


def rollback(target: str) -> Dict[str, Any]:
    if target not in list_releases():
        raise ReleaseError(f"目标 release 不存在: {target}")
    # A paused release has an intentionally empty ACTIVE; rollback is its recovery path.
    _recover_activation()
    before = (ACTIVE_FILE.read_text(encoding="utf-8").strip() or None
              if ACTIVE_FILE.exists() else None)
    write_active(target)
    governance_sync: Dict[str, Any] = {"status": "not_initialized"}
    # 回滚也要治理层同步：否则 ACTIVE 与 champion 会各说各话（"生效的是谁"变成谜）。
    if ALIASES_FILE.exists():
        try:
            data = load_aliases()
            prev_champion = data.get("champion")
            data["champion"] = target
            # 【事故修复】caveat 只在**真的换了人**时才清空。
            # 之前无条件清空，于是"批准候选 → 判定不通过 → 回滚到原 champion"
            # 这一整套标准操作会把原 champion 的 caveat（含 A1 结论）全部抹掉 ——
            # 回到同一个人，他的已知问题当然还在，不该当作新人。
            if prev_champion != target:
                data["champion_notes"] = []
            if before and before != target:
                data["archived"] = sorted(set(data.get("archived", []) + [before]))
            data.setdefault("history", []).append(
                {"at": now_iso(), "action": "rollback", "from": before, "to": target})
            _write_aliases(data)
            governance_sync = {"status": "synchronized"}
        except Exception as exc:
            # 治理层损坏不应阻断紧急回滚；把部分失败明确交给调用方修复。
            governance_sync = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "active_pointer_rolled_back": True,
                "repair_action": "重试回滚命令以重试 champion alias 同步",
            }
    return {"from": before, "to": target, "at": now_iso(),
            "governance_sync": governance_sync}


def release_info(release_id: Optional[str] = None) -> Dict[str, Any]:
    rid = release_id or current_release_id()
    return read_json(_dir(rid) / "release.json")
