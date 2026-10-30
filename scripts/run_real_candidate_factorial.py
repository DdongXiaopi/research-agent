"""真实模型探索实验：历史 RewriteBackend 候选 × 实际 few-shot × Skill。

四臂共用新 query_control 协议；先做静态/结构接地/10 题实测接地门，
再跑 dev_optimize。历史 200 题已被多次查看，结果只能标探索性。
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rsi import load_config
from src.contracts.types import Candidate
from src.eval.harness import evaluate
from src.evolution.fewshot import bootstrap_demos, render_demos_block
from src.evolution.prompt_rewrite import validate_rewrite
from src.governance.policy import static_gate
from src.governance.release import load_state
from src.runtime.runner import ArtifactState
from src.runtime.skills import load_body, with_body
from src.util import sha256_obj

QUERY_LINE = "RETRIEVAL_QUERY: {question} {options}"
SKILL_NAME = "litqa-literature-synthesis"


def _save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(bool(x["correct"]) for x in rows)
    answered = sum(str(x.get("status")) == "ok" for x in rows)
    return {"n": n, "correct": correct, "answered": answered,
            "accuracy": correct / n if n else 0,
            "coverage": answered / n if n else 0,
            "answered_precision": correct / answered if answered else 0}


def _rows(run_id: str) -> list[dict]:
    path = ROOT / "artifacts" / "runs" / run_id / "per_task.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=230)
    ap.add_argument("--smoke", type=int, default=10)
    args = ap.parse_args()
    # This historical four-arm runner performs a baseline demo-bootstrap call,
    # then smoke/full candidate scoring, without a frozen plan or request
    # ledger. Keep its analysis helpers for provenance, but do not allow the old
    # entrypoint to spend model budget or reinterpret repeatedly viewed tasks.
    ap.error(
        "this historical factorial runner is retired: it has no challenged "
        "execution plan or request ledger; use a current controlled probe")
    cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
    cfg["retrieval"]["query_control"] = {"enabled": True}
    base = load_state("baseline-v1-cited")
    pilot = json.loads((ROOT / "artifacts/probe/pilot_rewrite_iterative.json")
                       .read_text(encoding="utf-8"))
    historical = next(x for x in pilot["frontier"] if x["id"] == "r2c0")
    # 真正执行 few-shot：示范进入答题 system prompt，而非只给提案模型看。
    demos = bootstrap_demos(cfg, base, "pubmedqa", "dev_optimize", k=3,
                            limit=30, seed=20260924,
                            run_tag="factorial-fewshot-bootstrap")
    if not demos:
        raise RuntimeError("few-shot bootstrap returned no correct examples")
    demos_block = render_demos_block(demos)
    prompt_text = (historical["prompt"].rstrip() + "\n\n" + QUERY_LINE +
                   "\n\n" + demos_block)
    skill = base.registry.get(SKILL_NAME)
    skill_text = load_body(skill).rstrip() + "\n\n" + QUERY_LINE + "\n"

    run_id = "factorial-" + uuid.uuid4().hex[:12]
    out = ROOT / "artifacts/probe/real_factorial" / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    prompt_cand = Candidate(run_id + ":prompt", base.release_id,
                            "system_prompt", "system_prompt",
                            diff="rewrite + executed few-shot + query template",
                            new_content=prompt_text)
    skill_cand = Candidate(run_id + ":skill", base.release_id,
                           "skill_body", SKILL_NAME,
                           diff="skill body + query template",
                           new_content=skill_text)
    static = {"prompt": static_gate(prompt_cand, cfg, base),
              "skill": static_gate(skill_cand, cfg, base)}
    rewrite_reason = validate_rewrite(prompt_text, base.system_prompt, cfg)
    if rewrite_reason or not all(x["passed"] for x in static.values()):
        raise RuntimeError(f"static gate rejected: {rewrite_reason}; {static}")
    skill_reg = with_body(base.registry, SKILL_NAME, skill_text)
    states = {
        "base": ArtifactState(run_id + ":base", base.system_prompt, base.registry),
        "prompt_only": ArtifactState(run_id + ":prompt", prompt_text, base.registry),
        "skill_only": ArtifactState(run_id + ":skill", base.system_prompt, skill_reg),
        "joint": ArtifactState(run_id + ":joint", prompt_text, skill_reg),
    }
    # 接地性预检：输出契约和来源要求在运行前可见；数值代理再用 smoke 评测。
    baseline_requires_abstain = "insufficient" in base.system_prompt.lower()
    structural = {arm: {"answer_contract": "answer" in st.system_prompt.lower(),
                        "abstain_contract_preserved": (not baseline_requires_abstain or
                                                        "insufficient" in st.system_prompt.lower()),
                        "cite_contract": "cite" in st.system_prompt.lower()}
                  for arm, st in states.items()}
    if not all(all(v.values()) for v in structural.values()):
        raise RuntimeError(f"structural grounding rejected: {structural}")
    report = {"run_id": run_id, "status": "preflight",
              "cohort_status": "exploratory_reused_dev_optimize",
              "dataset": "pubmedqa", "split": "dev_optimize",
              "new_protocol": "real-v3 + query-template-v1",
              "historical_optimizer": "RewriteBackend (not GEPA library)",
              "historical_candidate_local_id": "r2c0",
              "historical_prompt_sha256": sha256_obj(historical["prompt"]),
              "fewshot_demo_ids": [d["task_id"] for d in demos],
              "fewshot_execution": "render_demos_block embedded in answer system prompt",
              "candidate_ids": {arm: run_id + ":" + arm for arm in states},
              "content_hashes": {arm: {"prompt": sha256_obj(st.system_prompt),
                                       "skills": st.registry.fingerprint()}
                                 for arm, st in states.items()},
              "static_gate": static, "structural_grounding": structural,
              "config_sha256": sha256_obj(cfg), "smoke": {}, "full": {}}
    _save(out, report)

    for arm, state in states.items():
        result = evaluate(cfg, state, "pubmedqa", "dev_optimize",
                          limit=args.smoke, seed=20260924, run_tag=run_id + ":" + arm,
                          write_per_task=True, cfg_fp=run_id + ":smoke")
        report["smoke"][arm] = {"run_id": result.run_id,
                                 "metrics": result.metric_values}
        _save(out, report)
        print("smoke", arm, result.metric_values["accuracy"],
              result.metric_values["unsupported_answer_rate"], flush=True)
    base_uns = report["smoke"]["base"]["metrics"]["unsupported_answer_rate"]
    allowed = base_uns + 0.05
    for arm in ("prompt_only", "skill_only", "joint"):
        uns = report["smoke"][arm]["metrics"]["unsupported_answer_rate"]
        if uns > allowed:
            report.setdefault("rejected", {})[arm] = {
                "reason": "grounding_smoke_unsupported_answer_rate",
                "observed": uns, "allowed": allowed}
    _save(out, report)

    for arm, state in states.items():
        if arm in report.get("rejected", {}):
            continue
        result = evaluate(cfg, state, "pubmedqa", "dev_optimize",
                          limit=args.limit, seed=20260924, run_tag=run_id + ":" + arm,
                          write_per_task=True, cfg_fp=run_id + ":full")
        rows = _rows(result.run_id)
        report["full"][arm] = {"run_id": result.run_id,
                                "metrics": result.metric_values,
                                "exploratory_after_30": _summary(rows[30:])}
        _save(out, report)
        print("full", arm, report["full"][arm]["exploratory_after_30"], flush=True)
    report["status"] = "complete"
    _save(out, report)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
