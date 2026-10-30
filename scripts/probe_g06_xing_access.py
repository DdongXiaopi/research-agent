"""Prepare or run a single zero-price Xing4 account-access probe for G06."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.eval.ai_jobs import build_plan, execute
from src.runtime.provider import load_project_dotenv
from src.util import read_json, repo_root, sha256_file, write_json


MODEL_ID = "XingChenAGI/Xing4.0-29B"
OUTPUT_ROOT = Path(".workbuddy/g06-xing-access-probe-20261016")
SCHEMA = "g06-xing-cn-free-model-access-probe-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
        help="send exactly one availability-only request after reviewing the frozen plan")
    args = parser.parse_args()
    root = repo_root()
    output_root = root / OUTPUT_ROOT
    plan_path = output_root / "plan.json"
    load_project_dotenv()
    if not os.environ.get("SILICONFLOW_API_KEY"):
        raise SystemExit("SILICONFLOW_API_KEY is absent; no request was sent")

    cfg = load_config("configs/experiment.http.yaml")
    cfg["models"]["task"].update({
        "provider": "http", "model_id": MODEL_ID,
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key_env": "SILICONFLOW_API_KEY", "enable_thinking": False,
        "max_retries": 1, "cache": False,
    })
    inputs = {"probe": "availability_only", "candidate_data_included": False}
    plan = build_plan("open_annotations", inputs, cfg, max_calls=1, tokens=220,
        max_input_chars=1000, max_cost_usd=0.01)
    if not output_root.exists():
        output_root.mkdir(parents=True)
    elif not plan_path.is_file():
        raise SystemExit(f"probe output already exists without a frozen plan: {output_root}")
    if plan_path.is_file():
        if read_json(plan_path) != plan:
            raise SystemExit("existing Xing probe plan differs; refusing to replace or rerun")
    else:
        write_json(plan_path, plan)

    if not args.execute:
        print(json.dumps({"status": "plan_ready", "model_id": MODEL_ID,
            "plan_sha256": plan["plan_sha256"], "max_calls": plan["max_calls"],
            "max_physical_attempts": plan["max_physical_attempts"],
            "max_cost_usd": plan["max_cost_usd"], "paid_fallback": False}, ensure_ascii=True))
        return

    runtime_home = output_root / "runtime"
    if runtime_home.exists() and any(runtime_home.iterdir()):
        raise SystemExit("Xing probe runtime already contains a run; automatic replay is forbidden")
    os.environ["RSI_HOME"] = str(runtime_home)

    def worker(call):
        response, _provenance = call("open_annotations",
            {"probe": "reply_json_availability"},
            'Return exactly one JSON object: {"available": true}. This is an API reachability check, not an annotation.')
        return {"response": response}

    result = execute(plan, inputs, cfg, worker)
    run_dir = Path(result["run_dir"]).resolve()
    if output_root.resolve() not in run_dir.parents:
        raise SystemExit("probe runtime escaped its isolated output directory")
    receipt = result["receipt"]
    call = read_json(run_dir / "call-0.json")
    output = read_json(run_dir / "output.json")
    if (receipt.get("call_count") != 1 or receipt.get("test_only") is not False
            or receipt.get("measured_usd") != 0.0
            or receipt.get("usage_complete") is not True
            or receipt.get("request_accounting", {}).get("physical_http_attempts") != 1
            or receipt.get("request_accounting", {}).get("physical_status_counts") != {"200": 1}
            or call.get("cost", {}).get("usd") != 0.0
            or call.get("cost", {}).get("usage_complete") is not True
            or output.get("response") != {"available": True}):
        raise SystemExit("probe did not meet the one-call zero-price HTTP 200 contract; preserve artifacts and do not retry")

    names = ("plan.json", "inputs.json", "output.json", "receipt.json",
             "call-0.json", "call-ledger.jsonl")
    report = {"schema": SCHEMA, "status": "completed", "model_id": MODEL_ID,
        "base_url": cfg["models"]["task"]["base_url"], "paid_fallback": False,
        "http_status": 200, "plan_sha256": plan["plan_sha256"],
        "run_dir": str(run_dir), "receipt": receipt, "cost": call["cost"],
        "files": {name: {"path": str(run_dir / name),
                         "sha256": sha256_file(run_dir / name)} for name in names}}
    write_json(output_root / "report.json", report)
    print(json.dumps({"status": "completed", "model_id": MODEL_ID,
        "plan_sha256": plan["plan_sha256"], "http_status": 200,
        "calls": 1, "measured_usd": receipt["measured_usd"],
        "usage_complete": receipt["usage_complete"], "paid_fallback": False,
        "report": str(output_root / "report.json")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
