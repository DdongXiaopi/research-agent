"""Smoke-test the proposed BioASQ protocol on public train questions only."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rsi import load_config
from src.data.bioasq_unitor import BioASQGateway, PROTOCOL_VERSION, load_tasks
from src.eval.metrics import aggregate
from src.governance.release import load_state
from src.runtime.runner import TaskRunner
from src.util import sha256_obj


def main() -> int:
    raise SystemExit(
        "unplanned historical BioASQ scoring is retired: direct train "
        "TaskRunner calls have no frozen execution-plan hash, request caps, "
        "or shared call ledger"
    )

    # Retained as historical implementation only pending governed replay.
    cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
    cfg["retrieval"]["query_control"] = {"enabled": True}
    cfg["skills"]["pinned_skill"] = "litqa-literature-synthesis"
    tasks = load_tasks("train", limit=20)
    state = load_state("baseline-v1-cited")
    outcomes = []
    for task in tasks:
        runner = TaskRunner(cfg, state, gateway=BioASQGateway(top_k=3), seed=20260929)
        outcomes.append(runner.run(task))
    metrics = aggregate(outcomes)
    report = {
        "id": "bioasq-train-pilot-" + uuid.uuid4().hex[:12],
        "status": "exploratory_train_only",
        "protocol": PROTOCOL_VERSION,
        "source": "UniTor answer_generation/train_set_answGEN.json",
        "config_sha256": sha256_obj(cfg),
        "skill_selection": "protocol_pin:litqa-literature-synthesis",
        "n": len(tasks), "metrics": metrics,
        "gold_in_corpus": sum(x.gold_in_corpus for x in outcomes),
        "gold_in_top3": sum(x.gold_in_topk for x in outcomes),
        "n_correct": sum(x.correct for x in outcomes),
        "n_abstain": sum(x.status == "abstain" for x in outcomes),
        "n_error": sum(x.status == "error" for x in outcomes),
    }
    dest = ROOT / "artifacts/probe" / (report["id"] + ".json")
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
