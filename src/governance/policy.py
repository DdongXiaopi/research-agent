"""静态门：先过门再花评测预算（FR-203）。

门项（主线）：
1. 工件类型在 `evolution.allowed_artifacts` 内
2. 工件类型是已知四类之一
3. 体积不超过 `gates.static.max_prompt_chars`
4. 不含危险指令（`forbidden_patterns`）
5. 配置工具范围只能使用受控运行时注册工具（FR-203）
6. 单个候选只改一个类别（FR-201）；组合 Skill 正文逐个过门
7. diff 非空、内容非空
8. 不含疑似密钥（隐私门）
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from ..contracts.types import ARTIFACT_TYPES, Candidate
from ..contracts.validate import validate_artifact_type
from ..runtime.skills import parse_procedure
from ..retrieval.intent import query_for, query_instruction
from ..util import sha256_obj, sha256_text

SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9]{16,}",
    r"(?i)api[_-]?key\s*[=:]\s*\S{8,}",
    r"(?i)authorization\s*:\s*bearer\s+\S{8,}",
]

# A negative safety instruction is not a dependency. Remove clear prohibitions
# before checking for positive search/retrieval requirements; keep explicit
# machine markers such as RETRIEVAL_QUERY: detectable and disallowed.
_NEGATED_RETRIEVAL_RE = re.compile(
    r"\b(?:do\s+not|don't|must\s+not|should\s+not|never)\s+"
    r"(?:(?:request|use|perform|access|search|retrieve|query)\s+)?"
    r"(?:external\s+|online\s+|web\s+|academic\s+)?"
    r"(?:retriev\w*|search\w*|pubmed|academic\.search)\b"
    r"(?:\s+(?:(?:external|online|web|academic)\s+)?"
    r"(?:papers?|articles?|sources?|databases?|documents?|tools?)\b)?"
    r"(?:\s+(?:or|and)\s+(?:(?:request|use|perform|access|search|retrieve|query)\s+)?"
    r"(?:external\s+)?(?:retriev\w*|search\w*|pubmed|academic\.search)\b)*|"
    r"\bno\s+(?:external\s+)?(?:retrieval|search)\s+"
    r"(?:tool|capability|access)\b|"
    r"\bwithout\s+(?:external\s+)?(?:retrieval|search)\b|"
    r"\b(?:avoid|remove|prohibit)\s+(?:external\s+)?retrieval"
    r"(?:[- ]specific)?\b",
    re.I,
)


def _has_retrieval_dependency(text: str) -> bool:
    """Return whether prompt text positively requires search/retrieval.

    Plain mentions in negated restrictions (e.g. "do not retrieve") are safe.
    Explicit runtime query markers remain dependencies even if quoted.
    """
    if re.search(r"RETRIEVAL_QUERY:|RETRIEVAL_INSTRUCTION:|academic\.search", text, re.I):
        return True
    without_prohibitions = _NEGATED_RETRIEVAL_RE.sub(" ", text)
    # A task-input prompt may name the provenance of the already supplied
    # material (e.g. ``supplied PubMed material``).  That is a source label,
    # not an instruction to call PubMed.  Strip only these tightly bounded
    # source-description phrases; positive verbs such as ``search PubMed`` or
    # ``retrieve articles`` remain dependencies and are still rejected.
    without_prohibitions = re.sub(
        r"\b(?:supplied|provided|task[- ]provided|given)\s+"
        r"(?:PubMed|scientific)\s+(?:material|abstracts?|evidence)\b",
        " ", without_prohibitions, flags=re.I)
    return bool(re.search(r"\bretriev\w*\b|\bsearch\w*\b|\bpubmed\b",
                          without_prohibitions, re.I))

# These are evaluation-only fields. A candidate may learn from labelled dev
# feedback, but its runtime instructions cannot ask for the hidden answer key.
ORACLE_REFERENCE_PATTERNS = [
    r"\b(?:expected|gold|reference)\s+(?:answer|label|option)\b",
    r"\bground[- ]truth\s+(?:answer|label|option)\b",
    r"\b(?:correct_index|gold_index|is_gold|gold_in_topk)\b",
]


def _check(name: str, passed: bool, detail: str = "") -> Dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail}


def _visible_demo_tasks(dataset: str) -> Dict[str, Dict[str, Any]]:
    from ..data.split import load_tasks

    rows = load_tasks(dataset, splits=["dev_optimize"])
    return {str(row.get("task_id")): row for row in rows}


def _skill_body_payload(candidate: Candidate) -> Dict[str, Any]:
    """Resolve the effective body text(s) represented by a skill_body candidate.

    Ordinary candidates carry one full body in ``new_content``. M4+ reflection
    combinations carry full per-Skill bodies in ``extra.skill_bodies`` and put
    the shared added step in ``new_content``. Every effective body must reach
    the same static checks as a single-body candidate.
    """
    extra = candidate.extra if isinstance(candidate.extra, dict) else {}
    if "skill_bodies" not in extra:
        return {"targets": [candidate.artifact_id],
                "contents": [str(candidate.new_content or "")],
                "composite": False, "valid": True, "detail": "single body"}

    raw = extra.get("skill_bodies")
    if not isinstance(raw, dict) or not raw:
        return {"targets": [], "contents": [str(candidate.new_content or "")],
                "composite": False, "valid": False,
                "detail": "extra.skill_bodies must be a non-empty object"}
    if any(not isinstance(name, str) or not name.strip()
           or not isinstance(body, str) or not body.strip()
           for name, body in raw.items()):
        return {"targets": [], "contents": [str(candidate.new_content or "")],
                "composite": len(raw) > 1, "valid": False,
                "detail": "body targets and contents must be non-empty strings"}

    bodies = {name: raw[name] for name in sorted(raw)}
    targets = list(bodies)
    composite = len(targets) > 1
    valid = candidate.artifact_id == "+".join(targets)
    if composite:
        valid = valid and all("+" not in name for name in targets)
        patch = str(candidate.new_content or "").strip()
        valid = valid and bool(patch) and all(patch in body for body in bodies.values())
    declared_names = extra.get("skill_names")
    if declared_names is not None:
        names_valid = (isinstance(declared_names, (list, tuple))
                       and all(isinstance(name, str) for name in declared_names)
                       and sorted(declared_names) == targets)
        valid = valid and names_valid
    declared_name = extra.get("skill_name")
    if declared_name is not None:
        valid = valid and declared_name == candidate.artifact_id
    if not composite:
        valid = valid and candidate.new_content == bodies[targets[0]]

    return {"targets": targets, "contents": list(bodies.values()),
            "bodies": bodies, "composite": composite, "valid": valid,
            "detail": f"skill_body targets={targets}"}


def static_gate(candidate: Candidate, cfg: Dict[str, Any], state: Any = None) -> Dict[str, Any]:
    gates = cfg.get("gates", {}).get("static", {})
    allowed = list(cfg.get("evolution", {}).get("allowed_artifacts", []))
    max_chars = int(gates.get("max_prompt_chars", 8000))
    max_desc_chars = int(gates.get("max_description_chars", 400))
    max_body_chars = int(gates.get("max_body_chars", 6000))
    min_steps = int(gates.get("min_procedure_steps", 3))
    max_steps = int(gates.get("max_procedure_steps", 12))
    is_desc = candidate.artifact_type == "skill_description"
    is_body = candidate.artifact_type == "skill_body"
    is_new = candidate.artifact_type == "new_skill"
    is_demos = candidate.artifact_type == "task_demos"
    is_program = candidate.artifact_type == "task_program"
    body_payload = _skill_body_payload(candidate) if is_body else None
    body_texts = (body_payload["contents"] if body_payload else
                  [str(candidate.new_content or "")])
    if is_desc:
        max_chars = max_desc_chars
    if is_demos:
        # Long biomedical abstracts can make a small labeled-demo bundle exceed
        # the prompt-only limit. Keep this opt-in per experiment; the default
        # task_demos budget remains max_prompt_chars.
        max_chars = int(gates.get("max_task_demos_chars", max_chars))
    if is_body or is_new:
        # 【M3】正文类工件按正文长度上限判体积，不再套用 prompt 的 8000
        max_chars = max_body_chars
    forbidden = list(gates.get("forbidden_patterns", []))
    results: List[Dict[str, Any]] = []

    try:
        validate_artifact_type(candidate.artifact_type)
        results.append(_check("artifact_type_known", True, candidate.artifact_type))
    except Exception as exc:
        results.append(_check("artifact_type_known", False, str(exc)))

    results.append(
        _check("artifact_type_allowed", candidate.artifact_type in allowed,
               f"{candidate.artifact_type} in {allowed}")
    )

    from ..runtime.tool_policy import CONTROLLED_TOOLS
    active_tools = (cfg.get("tools") or {}).get("active", [])
    valid_tool_list = (isinstance(active_tools, list)
                       and all(isinstance(name, str) and name for name in active_tools))
    unsupported_tools = (sorted(set(active_tools) - CONTROLLED_TOOLS)
                         if valid_tool_list else [])
    results.append(_check(
        "tool_scope_within_runtime",
        valid_tool_list and not unsupported_tools,
        (f"unsupported active tools: {unsupported_tools}" if unsupported_tools else
         f"active tools are restricted to the controlled runtime: {active_tools}"
         if valid_tool_list else "tools.active must be a list of non-empty names"),
    ))

    if candidate.artifact_type in {"system_prompt", "skill_description",
                                   "skill_body", "new_skill"}:
        from ..evolution.method_rule_gate import validate_statistical_method_rules
        method_rules = validate_statistical_method_rules(
            "\n\n".join(body_texts if is_body else [candidate.new_content or ""]))
        results.append(_check("statistical_method_rule_scope", method_rules["passed"],
                              str(method_rules["reason"])))

    if is_body:
        body_sizes = [len(text) for text in body_texts]
        size_ok = (bool(body_sizes) and body_payload["valid"]
                   and all(0 < size <= max_chars for size in body_sizes))
        size_detail = ", ".join(
            f"{name}:{len(text)}/{max_chars}"
            for name, text in zip(body_payload["targets"], body_texts))
    else:
        n = len(candidate.new_content or "")
    # A task_program is a transport envelope for one parent-relative component
    # change. Its size cap allows both bounded fields; each component is gated
    # below and `task_program_single_component` enforces FR-201 against parent.
    size_limit = (
        max_chars + int(gates.get("max_task_demos_chars", max_chars))
        if is_program else max_chars)
    results.append(_check("size_limit",
                          size_ok if is_body else 0 < n <= size_limit,
                          size_detail if is_body else
                          f"{n}/{size_limit} chars ({candidate.artifact_type})"))

    if is_desc:
        # description 会整体塞进上下文的渐进披露列表里，必须单行、可控长
        results.append(_check("description_single_line",
                              "\n" not in (candidate.new_content or ""),
                              "description 不得包含换行"))
        # 改 description 不允许偷偷改别的 skill（FR-201 单工件）
        results.append(_check("description_target_exists",
                              bool(state is None or state.registry.get(candidate.artifact_id)),
                              candidate.artifact_id))

    if is_body:
        skills = ([state.registry.get(name) for name in body_payload["targets"]]
                  if state is not None else [])
        targets_exist = bool(skills) and all(skill is not None for skill in skills)
        targets_mutable = targets_exist and all(not skill.frozen for skill in skills)
        results.append(_check("body_payload_matches_targets", body_payload["valid"],
                              body_payload["detail"]))
        results.append(_check("body_target_exists", targets_exist,
                              ",".join(body_payload["targets"])))
        results.append(_check("body_target_mutable", targets_mutable,
                              ",".join(body_payload["targets"])))

    if is_new:
        # 结晶必须是不存在的新名字 —— 否则就是"借结晶之名改已有 Skill"，绕过归因约束
        exists = bool(state is not None and state.registry.get(candidate.artifact_id))
        results.append(_check("new_skill_not_exists", not exists, candidate.artifact_id))
        results.append(_check("new_skill_has_metadata", bool(candidate.extra.get("skill")),
                              "extra.skill 缺失则无法落盘发布"))

    if is_demos:
        from ..evolution.fewshot import parse_demos_bundle
        demos = []
        try:
            demos = parse_demos_bundle(candidate.new_content)
            valid = bool(demos) and (cfg.get("runtime") or {}).get("input_mode") == "task_input"
            details = []
            for item in demos:
                task = item["task"]
                dataset = str(task.get("dataset") or "")
                source = _visible_demo_tasks(dataset).get(str(item["task_id"]))
                if source is None:
                    valid = False
                    details.append(f"{item['task_id']}:not_dev_optimize")
                    continue
                if (cfg.get("evaluation") or {}).get("insufficient_option", True):
                    from ..eval.options import inject_insufficient
                    source = inject_insufficient(source)
                if (str(source.get("question")) != str(task.get("question"))
                        or list(source.get("options") or []) != list(task.get("options") or [])
                        or str(source.get("evidence") or source.get("context") or "") !=
                        str(task.get("evidence") or task.get("context") or "")
                        or int(source.get("correct_index", -1)) != int(
                            task.get("correct_index", -2))):
                    valid = False
                    details.append(f"{item['task_id']}:task_content_mismatch")
            results.append(_check("task_demos_visible_and_unmodified", valid,
                                  ";".join(details) or f"{len(demos)} visible dev examples"))
        except (ValueError, TypeError, KeyError) as exc:
            results.append(_check("task_demos_visible_and_unmodified", False, str(exc)))
        results.append(_check("task_demos_count", 1 <= len(demos) <= 4,
                              f"{len(demos)}/4 examples"))

    program_prompt = ""
    if is_program:
        from ..evolution.fewshot import serialize_demos_bundle
        from ..evolution.task_program import (changed_task_program_components,
                                              parse_task_program)
        try:
            program = parse_task_program(candidate.new_content)
            program_prompt = str(program["system_prompt"])
            changed_components = changed_task_program_components(
                candidate.new_content, state)
            single_component = len(changed_components) == 1
            change_detail = (changed_components[0] if single_component else
                             "no_component_changed" if not changed_components else
                             "multiple_components_changed:" + ",".join(changed_components))
            results.append(_check("task_program_single_component", single_component,
                                  change_detail))
            if not single_component:
                results.append(_check(
                    "task_program_components_valid", False,
                    "task program must change exactly one parent-relative component"))
            elif changed_components[0] == "system_prompt":
                prompt_cfg = dict(cfg)
                prompt_cfg["evolution"] = dict(cfg.get("evolution") or {})
                prompt_cfg["evolution"]["allowed_artifacts"] = ["system_prompt"]
                prompt_candidate = Candidate(
                    candidate_id=candidate.candidate_id,
                    parent_release_id=candidate.parent_release_id,
                    artifact_type="system_prompt", artifact_id="system_prompt",
                    diff="validate task-program Prompt component",
                    new_content=program_prompt)
                prompt_gate = static_gate(prompt_candidate, prompt_cfg, state)
                results.append(_check("task_program_components_valid",
                                      prompt_gate["passed"],
                                      "system_prompt=" + str(prompt_gate["passed"])))
            else:
                demo_cfg = dict(cfg)
                demo_cfg["evolution"] = dict(cfg.get("evolution") or {})
                demo_cfg["evolution"]["allowed_artifacts"] = ["task_demos"]
                demo_candidate = Candidate(
                    candidate_id=candidate.candidate_id,
                    parent_release_id=candidate.parent_release_id,
                    artifact_type="task_demos", artifact_id="task_demos",
                    diff="validate task-program demo component",
                    new_content=serialize_demos_bundle(program["task_demos"]))
                demo_gate = static_gate(demo_candidate, demo_cfg, state)
                demo_failures = [
                    f"{result['check']}:{result['detail']}"
                    for result in demo_gate["results"]
                    if not result["passed"]
                ]
                results.append(_check("task_program_components_valid",
                                      demo_gate["passed"],
                                      "task_demos=" + str(demo_gate["passed"])
                                      + ("; failures=" + repr(demo_failures)
                                         if demo_failures else "")))
        except (ValueError, TypeError, KeyError) as exc:
            results.append(_check("task_program_single_component", False, str(exc)))
            results.append(_check("task_program_components_valid", False, str(exc)))

    if is_body or is_new:
        # 正文类工件必须有可解析的 Procedure：否则"进化"出来的只是散文，行为不会变
        checked_texts = body_texts if is_body else [candidate.new_content or ""]
        counts = [len(parse_procedure(text)) for text in checked_texts]
        detail = (", ".join(f"{name}:{count} steps"
                             for name, count in zip(body_payload["targets"], counts))
                  if is_body else f"{counts[0]} steps")
        results.append(_check("procedure_structure",
                              bool(counts) and all(count >= min_steps for count in counts),
                              f"{detail} (min {min_steps})"))
        results.append(_check("procedure_step_limit",
                              bool(counts) and all(count <= max_steps for count in counts),
                              f"{detail} (max {max_steps})"))

    # task_demos embeds authenticated optimize examples, including their labels.
    # Treat their evidence as data after source/label verification above; applying
    # instruction-only retrieval/oracle rules to quoted data creates false positives.
    if is_demos:
        gate_text = ""
    elif is_program:
        gate_text = program_prompt
    elif is_body:
        gate_text = "\n\n".join(body_texts + (
            [str(candidate.new_content)]
            if body_payload["composite"]
            and candidate.new_content not in body_texts else []))
    else:
        gate_text = candidate.new_content or ""
    low = gate_text.lower()
    hits = [p for p in forbidden if p.lower() in low]
    results.append(_check("no_forbidden_instruction", not hits, f"hits={hits}"))

    secrets = [p for p in SECRET_PATTERNS if re.search(p, gate_text)]
    results.append(_check("no_secret", not secrets, f"hits={len(secrets)}"))

    oracle_hits = [p for p in ORACLE_REFERENCE_PATTERNS
                   if re.search(p, gate_text, re.I)]
    results.append(_check("no_oracle_reference", not oracle_hits,
                          f"hits={oracle_hits}"))

    if (cfg.get("runtime") or {}).get("input_mode") == "task_input":
        requires_retrieval = _has_retrieval_dependency(gate_text)
        results.append(_check("no_retrieval_dependency", not requires_retrieval,
                              "task_input candidates must use task-provided material"))

    if "RETRIEVAL_QUERY:" in gate_text.upper():
        try:
            sample = {"question": "sample question", "options": ["yes", "no"]}
            if is_body:
                query_for(sample, "", gate_text, enabled=True)
            else:
                query_for(sample, gate_text, "", enabled=True)
            results.append(_check("retrieval_query_template", True))
        except ValueError as exc:
            results.append(_check("retrieval_query_template", False, str(exc)))

    if "RETRIEVAL_INSTRUCTION:" in gate_text.upper():
        try:
            query_instruction(gate_text if not is_body else "",
                              gate_text if is_body else "")
            results.append(_check("retrieval_instruction", True))
        except ValueError as exc:
            results.append(_check("retrieval_instruction", False, str(exc)))

    results.append(_check("diff_non_empty", bool(candidate.diff.strip()), f"{len(candidate.diff)} chars"))

    # FR-201 allows multiple Skill body targets only within the same category.
    if is_body:
        single_artifact = (body_payload["valid"] and bool(body_payload["targets"])
                           and all("|" not in name for name in body_payload["targets"]))
        artifact_detail = body_payload["detail"] + " (skill_body category)"
    else:
        single_artifact = (bool(candidate.artifact_id)
                           and candidate.artifact_id.count("|") == 0)
        artifact_detail = candidate.artifact_id
    results.append(_check("single_artifact", single_artifact, artifact_detail))

    if state is not None and hasattr(state, "registry") and all(r["passed"] for r in results):
        try:
            from ..evolution.apply import registry_with
            from ..runtime.skills import render_catalog
            trial_registry = registry_with(state.registry, candidate)
            render_catalog(trial_registry, int((cfg.get("skills") or {}).get("max_catalog_chars", 16000)))
            max_active = int((cfg.get("skills") or {}).get("max_active_skills", 64))
            if sum(s.lifecycle == "active" for s in trial_registry.skills) > max_active:
                raise ValueError("active Skill count exceeds registry budget")
            results.append(_check("skill_catalog_budget", True, "rendered catalog and active count within budget"))
        except (ValueError, TypeError, KeyError) as exc:
            results.append(_check("skill_catalog_budget", False, str(exc)))
    passed = all(r["passed"] for r in results)
    content_hash = (sha256_obj(body_payload["bodies"])
                    if is_body and body_payload.get("composite")
                    and body_payload.get("bodies") is not None
                    else sha256_text(candidate.new_content or ""))
    return {"passed": passed, "results": results, "content_sha256": content_hash}


def apply_gate(candidate: Candidate, cfg: Dict[str, Any], state: Any = None) -> Candidate:
    """跑门并把结果写回候选；失败即置为 static_gate_rejected。"""
    res = static_gate(candidate, cfg, state)
    candidate.static_gate_results = res["results"]
    if not res["passed"]:
        candidate.status = "static_gate_rejected"
    return candidate


def rejection_reason(results: List[Dict[str, Any]]) -> str:
    failed = [r["check"] for r in results if not r["passed"]]
    return "static_gate_failed: " + (",".join(failed) if failed else "unknown")
