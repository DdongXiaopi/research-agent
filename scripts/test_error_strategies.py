"""对历史 23+23 失败样本分别试验弃权校准和选项判断提示词。

样本由旧基线失败选择，只能解释机制，不能据此声明总体收益。
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rsi import load_config
from src.data.split import load_tasks
from src.eval.options import inject_all
from src.evolution.prompt_rewrite import validate_rewrite
from src.governance.release import load_state
from src.runtime.runner import ArtifactState, TaskRunner
from src.runtime.trace import Trace


ABSTAIN_RULE = (
    "\nBefore choosing ANSWER: INSUFFICIENT, check each option against the "
    "study's reported result and the retrieved passages. An indirect but "
    "unambiguous result can support an option; require a matching population, "
    "intervention, and outcome. If exactly one option is supported, answer "
    "and cite it. If none is supported, use ANSWER: INSUFFICIENT."
)
JUDGMENT_RULE = (
    "\nBefore ANSWER, compare yes, no, and maybe separately: identify the "
    "population, intervention or exposure, comparator, measured outcome, "
    "and direction of effect in the cited study. Use yes only for a supported "
    "positive claim, no for a supported negative claim, and maybe for mixed "
    "or genuinely inconclusive findings. Cite the passage carrying that result. "
    "If the retrieved text cannot establish any option, use ANSWER: INSUFFICIENT."
)


def main() -> int:
    raise SystemExit(
        "unplanned historical-error scoring is retired: this diagnostic uses "
        "failure-selected tasks and lacks an FR-203 static gate, frozen "
        "execution-plan hash, request caps, and shared call ledger"
    )

    # Retained as historical implementation only. Re-enable after replacing it
    # with a frozen, hash-challenged plan and FR-203-compliant execution path.
    cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
    base = load_state("baseline-v1-cited")
    selected = json.loads((ROOT / "artifacts/probe/query_mechanism_and_errors_2026-09-29.json")
                          .read_text(encoding="utf-8"))["errors"]
    tasks = {str(t["task_id"]): t for t in load_tasks("pubmedqa", ["dev_optimize"])}
    run_id = "strategy-" + uuid.uuid4().hex[:12]
    out = ROOT / "artifacts/probe/strategy_diagnostics" / f"{run_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    for bucket, rule in (("abstain", ABSTAIN_RULE), ("wrong", JUDGMENT_RULE)):
        prompt = base.system_prompt.rstrip() + rule
        reason = validate_rewrite(prompt, base.system_prompt, cfg)
        if reason:
            raise ValueError(f"static prompt gate: {reason}")
        state = ArtifactState(run_id + "-" + bucket, prompt, base.registry)
        for n, item in enumerate(r for r in selected if r["bucket"] == bucket):
            task = inject_all([tasks[item["task_id"]]])[0]
            trace = Trace(run_id + "-" + bucket, out_path=None)
            result = TaskRunner(cfg, state, trace=trace, seed=20260924).run(task)
            row = {"run_id": run_id, "bucket": bucket, "task_id": item["task_id"],
                   "strategy": "abstain_calibration" if bucket == "abstain" else "option_judgment",
                   "correct": result.correct, "status": result.status,
                   "answer_index": result.answer_index,
                   "gold_index": result.gold_index,
                   "cited": result.cited, "gold_rank": result.gold_rank,
                   "cost": result.cost, "error": result.error}
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{bucket} {n+1}/23 {item['task_id']} {result.status} "
                  f"correct={result.correct}", flush=True)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
