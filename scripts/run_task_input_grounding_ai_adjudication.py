"""Freeze or execute a third-AI adjudication of disputed grounding references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.eval import ai_grounding_reference as grounding_reference  # noqa: E402
from src.util import read_json, sha256_obj, write_json  # noqa: E402


def _config(path: Path, model_id: str) -> dict:
    cfg = load_config(path)
    model = cfg.setdefault("models", {}).setdefault("task", {})
    if model.get("provider") != "http":
        raise ValueError("AI adjudication requires an HTTP task model")
    model["model_id"] = model_id
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet-a", "packet-b", "responses-a", "responses-b", "plan"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.http.pubmedqa-final.yaml")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--max-cost-usd", type=float, default=0.25)
    parser.add_argument("--previous", type=Path,
                        help="verified partial adjudications to continue without replaying completed calls")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args()

    packet_a, packet_b = read_json(args.packet_a), read_json(args.packet_b)
    response_a, response_b = read_json(args.responses_a), read_json(args.responses_b)
    previous = read_json(args.previous) if args.previous else None
    cfg = _config(args.config, args.model_id)
    report = grounding_reference.score_references(packet_a, packet_b, response_a, response_b)
    plan = grounding_reference.adjudication_plan(
        packet_a, packet_b, response_a, response_b, cfg,
        previous=previous,
        max_calls=report["n_disagreement_unresolved"] - len(previous["rows"]) if previous else
                  report["n_disagreement_unresolved"], tokens=1200,
        max_input_chars=60000, max_cost_usd=args.max_cost_usd)
    if not args.execute:
        write_json(args.plan, plan)
        print(json.dumps({"plan_path": str(args.plan), "plan_sha256": plan["plan_sha256"],
                          "model_id": args.model_id, "disagreement_count": report["n_disagreement_unresolved"],
                          "max_calls": plan["max_calls"], "max_cost_usd": args.max_cost_usd}, indent=2))
        return 0
    if args.expected_plan_sha256 != plan["plan_sha256"]:
        raise ValueError("execution requires the exact reviewed adjudication plan SHA256")
    if read_json(args.plan) != plan:
        raise ValueError("saved adjudication plan differs from the current references/model/budget")
    result = grounding_reference.adjudicate(packet_a, packet_b, response_a, response_b,
                                            cfg, plan, previous=previous)
    if args.output:
        write_json(args.output, result["output"])
    print(json.dumps({"run_dir": result["run_dir"], "receipt": result["receipt"],
                      "adjudications": len(result["output"].get("rows", [])),
                      "output_sha256": sha256_obj(result["output"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
