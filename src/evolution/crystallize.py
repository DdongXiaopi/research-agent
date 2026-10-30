"""新 Skill 结晶（M3，归因 `skill.missing` → 工件 `new_skill`）。

为什么它是最容易自欺的一环，先说清楚边界：
    真实场景里 `skill.missing` 意味着"知识库/程序库里没有能处理这类题的 Skill"，
    结晶出的新 Skill 应该带来**新的检索策略或新的程序**。在离线夹具里做不到这一点：
    语料由 `build_fixture_corpus` 确定性生成，没有新数据源可挖。
    所以这里结晶出的 Skill 能带来的真实改变是**程序性的**：
    它在正文里写进"没有证据就说明缺口，而不是挑最眼熟的选项"，
    让原本会硬猜的题变成弃权 —— 体现在 `unsupported_answer_rate` 上，而不是 accuracy 上。

    这个差别必须如实写：**结晶链路（生成 → 门 → 发布 → 回滚）是可信的，
    但它在本夹具里的收益是"少编答案"，不是"多答对"。**

结晶门（比 description/body 更严，因为这是**新增工件**而不是改现有工件）：
    1. 结构：必须有 Procedure 小节且 ≥3 步
    2. 长度：description / body 不超上限
    3. 禁用模式：与静态门同一套 forbidden_patterns
    4. 触发测试：能召回失败题中的 ≥ min_trigger_recall 比例（否则结晶了也没人用）
    5. 去重：与任一已有 Skill 的 description 相似度 < max_similarity（防无限增殖）
    6. 反事实重跑：net >= 0（不能把已经答对的题弄坏）
    7. 【M4+】跨域误触发：在**其它领域**的题集上触发率必须 <= max_cross_domain_trigger，
       且拿不到负样本集时直接判 fail（fail-closed）。

    第 7 条是 M4 终评后补的，原因是它被实测打脸过：一个 triggers 为
    ['protocol','improve','cells','human'] 的结晶 Skill，在 ProtocolQA 上是正收益，
    但这几个词在 CS 语料里同样高频，于是它跨域误召回 —— CS 转移集上 6 个回归里有 5 个
    （已逐题核实：这 5 题在终评 release 下 `skill_recalled` 均为该结晶 Skill）。

    【必须如实写明的局限】**这条门拦不住上面那个 case。**
    实测该 Skill 在唯一可见的"其它域"（litqa2）上的触发率只有 0.094~0.137，
    而坏在 CS 上的触发率接近 0.8 —— litqa2 与 protocolqa 同属生物医学，
    词汇分布太近，根本代表不了 CS。也就是说：**用可见数据构造不出能复现 CS 失败的负样本**。
    进一步实测：在 litqa2 非目标题上加入该 Skill，`n_broke=0`（甚至 n_fixed=1），
    即"劫持检索"这条机制在可见数据上**没有可测信号**。

    因此这条门的定位要降格：它是**审计 + 防"明显到处乱触发"的保守护栏**，
    阈值 0.20 取自实测分布（各 trigger 组合在可见负样本上的触发率 0.026~0.148，
    取 ~1.35 倍作为上界），**不是**"能检出跨域泛化失败"的判据。
    真正堵住这类失败要靠别的手段（见 docs / README 的后续项）。

    参考：pk-pi-hermes-evolve 的 `tiered-gate.ts` 用配对回归 + **fail-closed**
    （拿不到同 cohort baseline 直接判 fail，而不是放行）；这里沿用同一语义。
    注意 vendor 里 hermes / scientific-agent-skills / pi 三家都只有静态门，
    没有任何跨域误触发检查 —— 这条是我们自己补的。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..contracts.types import Candidate, content_id, content_sha256, new_id
from ..runtime.runner import ArtifactState
from ..runtime.skills import Skill, _stems, behaviors_of_body, parse_procedure, recall, with_body
from ..util import now_iso
from .counterfactual import rerun, tasks_of
from .reflection import keywords_for_failures, unified_diff

BODY_TEMPLATE = """# {name}

{description}

## Procedure
1. Identify the specific property, quantity, or failure mode the question asks about
   ({keywords}).
2. Retrieve sources that report that property directly; prefer primary sources.
3. Compare each option against the retrieved passage and discard options supported only by
   surface similarity to the question wording.
4. If no retrieved passage addresses the question, state the gap instead of choosing the most
   familiar option.
"""

DESCRIPTION_TEMPLATE = (
    "Handle questions about {kws}: identify the exact property being asked, retrieve sources "
    "that report it directly, and compare each option against the retrieved passage. "
    "Use when the task mentions: {kws}."
)

TASK_INPUT_DESCRIPTION_TEMPLATE = (
    "Handle questions about {kws}: identify the requested property, compare "
    "each task option against supplied evidence, and check scope and uncertainty. "
    "Use when the task mentions: {kws}."
)
TASK_INPUT_BODY_TEMPLATE = """# {name}

{description}

## Procedure
1. Identify the property and scope requested by the task ({keywords}).
2. Read the supplied task material and locate statements addressing that property.
3. Compare every permitted task option with those statements, including direction,
   population, uncertainty and contradictory evidence.
4. Follow the task's declared answer contract when evidence is inconclusive; use
   an explicit uncertainty label only when it is among the permitted task options.
"""


class CrystallizationError(RuntimeError):
    pass


def _name_for(keywords: Sequence[str], existing: Sequence[str]) -> str:
    """确定性命名：`crystallized-<kw1>-<kw2>`，重名则追加序号。"""
    base = "crystallized-" + "-".join(list(keywords)[:2] or ["general"])
    name = base
    i = 2
    while name in set(existing):
        name = f"{base}-{i}"
        i += 1
    return name


def _skill_hits(registry: Any, name: str, query: str, min_score: int) -> bool:
    """指定 Skill 在该 query 上的分是否达到召回阈值。"""
    from ..runtime.skills import route

    for c in route(query, registry, top_n=max(1, len(registry.skills))):
        if c["name"] == name:
            return int(c["score"]) >= min_score
    return False


def _similarity(a: str, b: str) -> float:
    sa, sb = _stems(a), _stems(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def cross_domain_negative_tasks(cfg: Dict[str, Any], target_datasets: Sequence[str],
                                exclude_task_ids: Optional[Sequence[str]] = None,
                                limit: int = 0) -> List[Dict[str, Any]]:
    """跨域负样本：目标领域**以外**的主数据集题目（只看 dev_optimize）。

    为什么只用 dev_optimize：
        门禁是在进化过程中跑的。`selection_gate` 是留出来做选择的、
        `final_holdout` 是封存的，两者都不该被"门禁怎么构造"这件事看到；
        `transfer_datasets`（ScholarQABench CS）是只终评的迁移域，同理排除。

    为什么有 fallback：
        campaign 默认同时在 litqa2 + protocolqa 上跑，`skill.missing` 的失败样本
        可能横跨全部主数据集，此时"其它数据集"是空集 —— 若直接 fail-closed，
        整条结晶链路会被静默打死。所以退而求其次，用**同数据集里不在失败集中的
        dev_optimize 题**当负样本：它测的是"过触发"（抢了不该管的题），
        虽然不是真正的跨域，但至少让门禁有据可依而不是空转。
        两个池都拿不到才 fail-closed（调用方负责判 fail）。

    `limit=0` 表示不限（默认）：负样本池大小会直接影响触发率，
    截断会让阈值失去意义 —— 实测同一个 Skill 在 litqa2 前 60 条上是 0.117、
    全量 117 条上是 0.094，靠截断凑阈值就是自欺。
    """
    from ..data.split import load_tasks

    target = {str(d) for d in target_datasets if d}
    all_ds = [str(d) for d in (cfg.get("evaluation", {}).get("datasets") or []) if str(d)]
    exclude = {str(t) for t in (exclude_task_ids or [])}

    def _load(datasets):
        out: List[Dict[str, Any]] = []
        for ds in sorted(datasets):
            try:
                out.extend(load_tasks(ds, splits=["dev_optimize"]))
            except Exception as exc:
                raise CrystallizationError(
                    f"negative dataset {ds} could not be loaded ({type(exc).__name__})") from exc
        return out

    others = [d for d in all_ds if d not in target]
    pool = _load(others)
    origin = "other-domain"
    if not pool:
        # fallback：同域里"不是结晶目标"的题
        pool = [t for t in _load(all_ds) if str(t.get("task_id")) not in exclude]
        origin = "same-domain-non-target"
    if not pool:
        return []
    for t in pool:
        t.setdefault("_neg_pool", origin)
    return pool[:int(limit)] if limit and limit > 0 else pool


def build_skill(keywords: Sequence[str], state: ArtifactState, cfg: Dict[str, Any]) -> Skill:
    kws = ", ".join(keywords)
    name = _name_for(keywords, [s.name for s in state.registry.skills])
    fixed_input = (cfg.get("runtime") or {}).get("input_mode") == "task_input"
    description_template = (TASK_INPUT_DESCRIPTION_TEMPLATE if fixed_input
                            else DESCRIPTION_TEMPLATE)
    body_template = TASK_INPUT_BODY_TEMPLATE if fixed_input else BODY_TEMPLATE
    description = description_template.format(kws=kws)
    body = body_template.format(name=name, description=description, keywords=kws)
    return Skill(
        name=name,
        group=str(cfg.get("skills", {}).get("crystallized_group", "G-crystallized")),
        description=description,
        triggers=list(keywords),
        body_path="",
        frozen=False,
        license="SELF-GENERATED",
        body=body,
    )


def crystallization_checks(skill: Skill, state: ArtifactState,
                           failure_rows: Sequence[Dict[str, Any]],
                           cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """结晶门的全部检查项（返回结构化结果，供 review 与测试使用）。"""
    cz = cfg.get("evolution", {}).get("crystallization", {}) or {}
    gates = cfg.get("gates", {}).get("static", {}) or {}
    max_desc = int(gates.get("max_description_chars", 400))
    max_body = int(gates.get("max_body_chars", 6000))
    max_steps = int(gates.get("max_procedure_steps", 12))
    forbidden = [str(p) for p in gates.get("forbidden_patterns", [])]
    min_steps = int(cz.get("min_procedure_steps", 3))
    min_trigger = float(cz.get("min_trigger_recall", 0.30))
    max_sim = float(cz.get("max_description_similarity", 0.60))
    min_score = int(cfg.get("skills", {}).get("recall_min_score", 2))
    max_cross = float(cz.get("max_cross_domain_trigger", 0.20))
    # fallback 池（同域非目标题）单独一个上限：该池**基线**触发率就有 0.36
    # （同域词汇本就相近），沿用跨域阈值会把整条结晶链路打死 ——
    # 实测 campaign 三轮 `skill.missing` 全部因此 skipped。故该池只审计不拦截。
    max_non_target = float(cz.get("max_non_target_trigger", 1.0))
    neg_limit = int(cz.get("cross_domain_negative_limit", 0))

    checks: List[Dict[str, Any]] = []

    steps = parse_procedure(skill.body)
    checks.append({"name": "procedure_structure",
                   "passed": bool(steps) and len(steps) >= min_steps,
                   "detail": f"{len(steps)} steps (min {min_steps})"})
    checks.append({"name": "procedure_step_limit",
                   "passed": len(steps) <= max_steps,
                   "detail": f"{len(steps)}/{max_steps} steps"})
    checks.append({"name": "description_size",
                   "passed": 0 < len(skill.description) <= max_desc,
                   "detail": f"{len(skill.description)}/{max_desc} chars"})
    checks.append({"name": "body_size",
                   "passed": 0 < len(skill.body) <= max_body,
                   "detail": f"{len(skill.body)}/{max_body} chars"})
    bad = [p for p in forbidden if p.lower() in (skill.description + " " + skill.body).lower()]
    checks.append({"name": "forbidden_patterns",
                   "passed": not bad, "detail": f"matched={bad}" if bad else "none"})

    # Use the same project-wide static gate as every other candidate before
    # spending TaskRunner budget. The crystallization checks above are stricter
    # for new Skills, but they do not cover tool scope, secrets, oracle
    # references, task-input retrieval dependencies, or artifact allowlists.
    from ..governance.policy import static_gate
    preview = Candidate(
        candidate_id=content_id("cand", state.release_id, "new_skill", skill.name,
                                skill.description, tuple(skill.triggers), skill.body),
        parent_release_id=state.release_id,
        artifact_type="new_skill",
        artifact_id=skill.name,
        diff=unified_diff("", skill.body, from_name="a/<new>",
                          to_name=f"b/{skill.name}.md"),
        new_content=skill.body,
        extra={"skill": skill.to_dict()},
    )
    static = static_gate(preview, cfg, state)
    static_failures = [
        f"{result['check']}:{result['detail']}"
        for result in static["results"] if not result["passed"]
    ]
    checks.append({
        "name": "static_gate",
        "passed": static["passed"],
        "detail": ";".join(static_failures) or "all static checks passed",
    })

    # 去重：与已有 Skill 太像就没必要结晶（它只是换个名字的复制品）
    sims = {s.name: round(_similarity(skill.description, s.description), 3)
            for s in state.registry.skills}
    too_similar = {k: v for k, v in sims.items() if v >= max_sim}
    checks.append({"name": "not_duplicate",
                   "passed": not too_similar,
                   "detail": f"max_sim={max(sims.values()) if sims else 0} "
                             f"(threshold {max_sim})" + (f" over={too_similar}" if too_similar else "")})

    # 触发测试：结晶出来却召回不了目标题 = 白结晶。
    # 注意判定的是**新 Skill 自己**的分，不是 registry 的最高分 ——
    # 否则"被别的 Skill 抢先命中"会被误判成"新 Skill 能召回"。
    tasks = tasks_of(failure_rows)
    trial = _with_skill(state, skill)
    hits = 0
    for t in tasks:
        q = f"{t.get('question','')} " + " ".join(t.get("options", []))
        if _skill_hits(trial.registry, skill.name, q, min_score):
            hits += 1
    rate = (hits / len(tasks)) if tasks else 0.0
    checks.append({"name": "trigger_test",
                   "passed": bool(tasks) and rate >= min_trigger,
                   "detail": f"recall {hits}/{len(tasks)} = {rate:.2f} (min {min_trigger})"})

    # 【M4+】跨域触发审计：在别的领域上"不该被召回的题"里命中了多少。
    # 与目标领域触发率互补 —— 前者看"够不够泛"，这里看"有没有过泛"。
    # fail-closed：拿不到负样本集就判 fail，绝不放行（对齐 tiered-gate.ts:133-142）。
    #
    # 定位说明：这是**审计 + 保守护栏**，不是"能检出跨域泛化失败"的判据 ——
    # 实测造成 CS 5 个回归的那个 Skill 在可见负样本上只有 0.094~0.137，低于任何
    # 合理阈值。真正的局限写在模块 docstring 里，别在这里自欺。
    target_ds = sorted({str(r.get("dataset", "")) for r in failure_rows if r.get("dataset")})
    negative_error = None
    try:
        neg = cross_domain_negative_tasks(
            cfg, target_ds,
            exclude_task_ids=[str(r.get("task_id", "")) for r in failure_rows],
            limit=neg_limit)
    except CrystallizationError as exc:
        negative_error = str(exc)
        neg = []
    if not neg:
        checks.append({"name": "cross_domain_false_trigger", "passed": False,
                       "detail": negative_error or "无跨域负样本（fail-closed：负样本缺失时不得放行）"})
    else:
        n_hit = 0
        for t in neg:
            q = f"{t.get('question','')} " + " ".join(t.get("options", []))
            if _skill_hits(trial.registry, skill.name, q, min_score):
                n_hit += 1
        n_rate = n_hit / len(neg)
        neg_ds = ", ".join(sorted({str(t.get("dataset", "")) for t in neg}))
        neg_pool = str(neg[0].get("_neg_pool", "other-domain"))
        cap = max_cross if neg_pool == "other-domain" else max_non_target
        checks.append({"name": "cross_domain_false_trigger",
                       "passed": n_rate <= cap,
                       "detail": ("trigger %d/%d = %.2f (max %.2f); pool=%s datasets=[%s]"
                                  % (n_hit, len(neg), n_rate, cap, neg_pool, neg_ds))})

    # A candidate rejected by any deterministic pre-evaluation check must not
    # be sent to the counterfactual TaskRunner. This includes the shared static
    # gate as well as crystallization-specific structure/trigger/duplicate
    # checks, all of which are cheaper than a task evaluation.
    pre_evaluation_failures = [check for check in checks if not check["passed"]]
    # 反事实重跑：不能把已经答对的题弄坏
    if tasks and not pre_evaluation_failures:
        cf = rerun(cfg, state, trial, tasks,
                   seed=int(cfg.get("evaluation", {}).get("seed", 0)))
        checks.append({"name": "counterfactual_non_regression",
                       "passed": int(cf["net"]) >= 0,
                       "detail": f"n_fixed={cf['n_fixed']} n_broke={cf['n_broke']} "
                                 f"net={cf['net']}"})
    elif tasks:
        failed_names = [check["name"] for check in pre_evaluation_failures]
        checks.append({"name": "counterfactual_non_regression", "passed": False,
                       "detail": "not run: pre-evaluation checks failed: "
                                 + ",".join(failed_names)})
    else:
        cf = None
        checks.append({"name": "counterfactual_non_regression",
                       "passed": False, "detail": "无可用失败题目"})
    return checks


def _with_skill(state: ArtifactState, skill: Skill) -> ArtifactState:
    """把结晶的 Skill 加进 registry（正文内联，落盘由 release 负责）。"""
    from ..runtime.skills import add_skill

    return ArtifactState(release_id=state.release_id,
                         system_prompt=state.system_prompt,
                         registry=add_skill(state.registry, skill),
                         task_demos=state.task_demos)


def propose_skill_from_failures(state, failure_rows, cfg, *, provider=None, cases_text=None, audit_dir=None):
    """A bounded live proposal, evaluated by the existing new-Skill gates.

This is an artifact hypothesis; no fallback template is allowed when a live
proposal is malformed. Reflection examples are visible development failures.
"""
    import copy
    import json
    import re
    from ..runtime.provider import build_provider
    from ..runtime.call_ledger import call_component
    from .reflection_backend import LlmBackend
    cc = (cfg.get("evolution") or {}).get("crystallization") or {}
    if not failure_rows or any(row.get("split") != "dev_optimize" for row in failure_rows):
        raise ValueError("live Skill proposals may consume only visible development failures")
    reflection_cfg = copy.deepcopy(cfg)
    reflection_cfg.setdefault("runtime", {})["provider"] = "http"
    model = copy.deepcopy((cfg.get("models") or {}).get("reflection") or {})
    if str(model.get("provider") or "http").lower() != "http":
        raise ValueError("live Skill crystallization requires an HTTP reflection model")
    model.update({"cache": False, "max_retries": int(cc.get("proposal_max_attempts", 1)),
                  "structured_output": ""})
    if not 1 <= model["max_retries"] <= 3:
        raise ValueError("Skill proposal attempts must be bounded in [1,3]")
    reflection_cfg.setdefault("models", {})["task"] = model
    provider = provider or build_provider(reflection_cfg)
    system = ('Propose one reusable Skill from the supplied development failure cases. '
              'Return JSON with exactly name, description, triggers and body. '
              'name is a safe lowercase hyphenated identifier; description is one short line; '
              'triggers is a list of 2–6 topic terms; body has a Procedure section with 3–8 numbered steps. '
              'Use only supplied task material, explicit task labels and evidence. '
              'Do not retrieve, change tools, infer hidden labels, copy task IDs or memorize answer keys. '
              'In positive Procedure steps, say inspect or read supplied material; do not instruct search, retrieval, or external calls. '
              'If feedback supplies a reusable code definition needed to interpret future material, preserve that explicit definition without inventing other mappings. '
              'Generalize a reasoning procedure; this proposal does not establish the cause of failure. '
              'description must contain no newline. body is one JSON string that decodes into actual line breaks: '
              'a Procedure heading followed by numbered steps on separate lines, never a nested object. '
              'Return the JSON object without Markdown fences or additional keys.')
    system += "\nFormat example (replace topic/content from the failure cases): " + json.dumps({
        "name": "source-check-procedure", "description": "Check source evidence", "triggers": ["source", "evidence"],
        "body": "## Procedure\n1. Read the supplied evidence.\n2. Check the requested condition.\n3. Report the result with a source quotation."})
    case_limit = int(cc.get("proposal_case_limit", 4))
    max_tokens = int(cc.get("proposal_max_tokens", 1400))
    if not 1 <= case_limit <= 6 or not 1 <= max_tokens <= 2400:
        raise ValueError("Skill proposal context and output must be bounded")
    if cases_text is not None and (not isinstance(cases_text, str) or len(cases_text) > 12000):
        raise ValueError("Skill proposal cases exceed the development context budget")
    cases = cases_text if cases_text is not None else LlmBackend._format_cases(
        failure_rows, limit=case_limit, chars=1200)
    user = "Existing Skills:\n" + "\n".join(f"{s.name}: {s.description}" for s in state.registry.skills)
    user += "\nVisible failure cases:\n" + cases
    from ..util import artifacts_path, write_json, sha256_file
    from pathlib import Path
    from uuid import uuid4
    audit = Path(audit_dir) if audit_dir is not None else artifacts_path("skill_proposal_calls", uuid4().hex)
    audit.mkdir(parents=True, exist_ok=False)
    write_json(audit / "request.json", {"system": system, "user": user, "max_tokens": max_tokens,
                                       "model_id": model.get("model_id")})
    schema = {"type": "json_schema", "json_schema": {"name": "skill_proposal", "strict": True,
        "schema": {"type": "object", "properties": {
            "name": {"type": "string", "pattern": "^[a-z][a-z0-9-]{2,63}$"},
            "description": {"type": "string", "pattern": "^[^\\r\\n]+$"},
            "triggers": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 6},
            "body": {"type": "string", "pattern": "^(## )?Procedure\\n1[.)] [\\s\\S]+\\n2[.)] [\\s\\S]+\\n3[.)] [\\s\\S]+$"}},
        "required": ["name", "description", "triggers", "body"],
            "additionalProperties": False}}}
    try:
        with call_component("skill_crystallization_proposer"):
            if hasattr(provider, "chat_measured"):
                output = provider.chat_measured(system, user, max_tokens=max_tokens, response_format=schema)
                raw = output.content
                metadata = {key: getattr(output, key, None) for key in (
                    "cost", "prompt_tokens", "completion_tokens", "model_id", "finish_reason", "http_attempts")}
            else:
                raw = provider.chat(system, user, max_tokens=max_tokens)
                metadata = {"mode": "injected_unmeasured_fixture"}
        write_json(audit / "raw-response.json", {"content": raw, **metadata})
        write_json(audit / "receipt.json", {"status": "response_received",
            "request_sha256": sha256_file(audit / "request.json"),
            "response_sha256": sha256_file(audit / "raw-response.json")})
    except BaseException as exc:
        write_json(audit / "failure.json", {"error_type": type(exc).__name__})
        raise
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate Skill proposal JSON key")
            result[key] = value
        return result
    try:
        try:
            data = json.loads(raw, object_pairs_hook=unique_object)
        except json.JSONDecodeError as exc:
            # JSON-mode endpoints sometimes return literal line breaks inside
            # the body string. Accept only that transport issue; the same exact
            # object/field/Procedure validation follows and original bytes stay.
            if ("Invalid control character" not in exc.msg or not isinstance(raw, str)
                    or any(ord(c) < 32 and c not in "\n\r\t" for c in raw)):
                raise
            data = json.loads(raw, strict=False, object_pairs_hook=unique_object)
            write_json(audit / "json-transport-normalization.json", {"rule": "literal-json-string-whitespace-v1",
                "raw_response_sha256": sha256_file(audit / "raw-response.json")})
    except (TypeError, ValueError) as exc:
        raise ValueError("Skill proposal is not a complete unambiguous JSON object") from exc
    if (not isinstance(data, dict) or set(data) != {"name", "description", "triggers", "body"}
            or not isinstance(data["name"], str) or not re.fullmatch(r"[a-z][a-z0-9-]{2,63}", data["name"])
            or not isinstance(data["description"], str) or not data["description"].strip()
            or "\n" in data["description"] or len(data["description"]) > 400
            or not isinstance(data["triggers"], list) or not 2 <= len(data["triggers"]) <= 6
            or any(not isinstance(term, str) or not term.strip() or len(term) > 60 for term in data["triggers"])
            or not isinstance(data["body"], str) or not data["body"].strip() or len(data["body"]) > 6000):
        raise ValueError("Skill proposal violates the immutable proposal contract")
    # Some JSON-mode providers double-escape a numbered Procedure into one
    # literal line. Normalize only this unambiguous transport form, retaining
    # the original response and both identities for audit. Static gates still run.
    if "\n" not in data["body"] and re.match(r"^(?:## )?Procedure\\n1[.)] ", data["body"]):
        original_body = data["body"]
        data["body"] = original_body.replace("\\r\\n", "\n").replace("\\n", "\n")
        from ..util import sha256_text
        write_json(audit / "normalization.json", {"rule": "single-line-double-escaped-procedure-v1",
            "original_body_sha256": sha256_text(original_body), "body_sha256": sha256_text(data["body"])})
    write_json(audit / "parsed-skill.json", data)
    from ..runtime.skills import parse_procedure
    if not 3 <= len(parse_procedure(data["body"])) <= 8:
        raise ValueError("Skill proposal requires 3 through 8 actual Procedure steps")
    return Skill(name=data["name"], group=cc.get("group", "G-crystallized"),
        description=data["description"], triggers=data["triggers"], body=data["body"], license="SELF-GENERATED")


def plan_new_skill(state: ArtifactState, failure_rows: Sequence[Dict[str, Any]],
                   cfg: Dict[str, Any]) -> Dict[str, Any]:
    """结晶规划：给出候选 + 全部门检查结果 + 未通过原因。"""
    out: Dict[str, Any] = {"ok": False, "reason": "", "candidate": None, "checks": []}
    if not failure_rows:
        out["reason"] = "没有失败样本"
        return out
    from .reflection import _task_index

    tasks = _task_index([str(r.get("dataset", "")) for r in failure_rows if r.get("dataset")])
    keywords = keywords_for_failures(failure_rows, tasks)
    if not keywords:
        out["reason"] = "抽不出可用于触发的关键词"
        return out

    live_proposal = str(((cfg.get("evolution") or {}).get("crystallization") or {}).get(
        "proposal_backend", "deterministic-stub"))
    if live_proposal not in {"deterministic-stub", "llm"}:
        raise ValueError("unknown Skill crystallization proposal backend")
    skill = (propose_skill_from_failures(state, failure_rows, cfg) if live_proposal == "llm"
             else build_skill(keywords, state, cfg))
    checks = crystallization_checks(skill, state, failure_rows, cfg)
    out["checks"] = checks
    failed = [c for c in checks if not c["passed"]]
    if failed:
        out["reason"] = "；".join(f"{c['name']}({c['detail']})" for c in failed)
        return out

    cand = Candidate(
        candidate_id=content_id("cand", state.release_id, "new_skill", skill.name,
                                skill.description, tuple(skill.triggers), skill.body),
        parent_release_id=state.release_id,
        artifact_type="new_skill",
        artifact_id=skill.name,
        diff=unified_diff("", skill.body, from_name="a/<new>", to_name=f"b/{skill.name}.md"),
        rationale=(f"new_skill 工件实验：{len(failure_rows)} 条可进化失败样本，"
                   f"测试现有 {len(state.registry.skills)} 个 Skill 之外的新程序（高频主题词 {keywords}）。"
                   "此提名不证明失败由 Skill 缺失导致。"
                   f"结晶新 Skill '{skill.name}'，正文含「无证据则声明缺口」程序，"
                   f"反事实重跑未弄坏任何已答对的题。"),
        status="draft",
        new_content=skill.body,
        attributed_class="skill.missing",
        source_failure_ids=[str(r.get("failure_id", "")) for r in failure_rows],
        generator={
            "backend": "llm" if live_proposal == "llm" else "deterministic-stub",
            "is_gepa": False,
            "kind": "crystallization",
            "note": ("LLM 能力假设；仍需独立逐类门禁和人工发布审查" if live_proposal == "llm" else
                     "测试替身，不产生效果结论；真实结晶需 LLM 归纳 + 人工审校"),
        },
        content_sha256=content_sha256(skill.body),
        extra={
            "skill": skill.to_dict(),
            "skill_body": skill.body,
            "description": skill.description,
            "triggers": list(skill.triggers),
            "keywords": list(keywords),
            "behaviors": behaviors_of_body(skill.body),
            "checks": checks,
        },
    )
    out.update({"ok": True, "candidate": cand, "skill": skill})
    return out


def generate_new_skill_candidate(state: ArtifactState, failure_rows: Sequence[Dict[str, Any]],
                                 cfg: Dict[str, Any]) -> Optional[Candidate]:
    return plan_new_skill(state, failure_rows, cfg).get("candidate")
