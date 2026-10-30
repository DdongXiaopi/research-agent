"""Plan/execute AI disagreement adjudication and measured runtime calibration."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.eval import ai_grounding_reference as reference, grounding_calibration as calibration
from src.util import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("adjudicate", "calibrate"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--max-cost-usd", type=float, default=0.25)
        p.add_argument("--max-calls", type=int)
        p.add_argument("--execute", action="store_true")
        p.add_argument("--expected-plan-sha256")
        p.add_argument("--output", type=Path)
        if name == "calibrate":
            p.add_argument("--prompt-version", choices=("v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"), default="v4")
            p.add_argument("--output-mode", choices=("schema", "json_object"), default="schema")
        else:
            p.add_argument("--tokens", type=int, default=1200,
                           help="Frozen completion token cap for each adjudication")
        if name == "adjudicate":
            for field in ("packet-a", "packet-b", "responses-a", "responses-b"):
                p.add_argument("--" + field, type=Path, required=True)
            p.add_argument("--previous-adjudications", type=Path,
                help="verified partial AI adjudications from an earlier failed batch")
        else:
            for field in ("packet", "reference"):
                p.add_argument("--" + field, type=Path, required=True)
    p = sub.add_parser("recompute")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--prompt-version", choices=("v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"), default="v4")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "recompute":
        cfg.setdefault("evaluation", {}).setdefault("semantic_citation", {})[
            "prompt_version"] = args.prompt_version
        result = calibration.recompute_completed_job(args.run_dir, cfg)
        if args.output:
            write_json(args.output,result)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return
    budget = {"max_cost_usd": args.max_cost_usd}
    if args.max_calls is not None:
        budget["max_calls"] = args.max_calls
    if args.command == "adjudicate":
        inputs = tuple(read_json(p) for p in (args.packet_a, args.packet_b, args.responses_a, args.responses_b))
        previous = read_json(args.previous_adjudications) if args.previous_adjudications else None
        plan = reference.adjudication_plan(*inputs, cfg, previous=previous,
                                           tokens=args.tokens, **budget)
    else:
        cfg.setdefault("evaluation", {}).setdefault("semantic_citation", {})[
            "prompt_version"] = args.prompt_version
        inputs = (read_json(args.packet), read_json(args.reference))
        plan = calibration.build_plan(*inputs, cfg, output_mode=args.output_mode, **budget)
    if not args.execute:
        result = plan
    else:
        if args.expected_plan_sha256 != plan["plan_sha256"]:
            raise ValueError("execution requires the exact previously reviewed plan SHA256")
        if args.command == "adjudicate":
            job = reference.adjudicate(*inputs, cfg, plan, previous=previous)
            result = {"job": {"run_dir": job["run_dir"], "receipt": job["receipt"]},
                "reference": reference.resolve(reference.score_references(*inputs), *inputs, job["output"])}
        else:
            job = calibration.execute(*inputs, cfg, plan)
            result = {"run_dir": job["run_dir"], "receipt": job["receipt"], "calibration": job["calibration"]}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
