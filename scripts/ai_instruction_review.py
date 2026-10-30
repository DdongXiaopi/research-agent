"""Prepare/annotate/measure a contract-specific AI instruction reviewer challenge."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.eval import ai_instruction_calibration as calibration
from src.util import read_json, write_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","reference","calibrate"))
    parser.add_argument("--packet",required=True,type=Path)
    parser.add_argument("--config")
    parser.add_argument("--model-id")
    parser.add_argument("--reference-run")
    parser.add_argument("--max-cost-usd",type=float,default=0.25)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.output and args.output.exists(): raise FileExistsError("output already exists")
    if args.command=="prepare":
        if args.packet.exists(): raise FileExistsError("packet already exists")
        result=calibration.build_packet()
        write_json(args.packet,result)
    else:
        if not args.config: parser.error("--config required")
        cfg=load_config(args.config)
        if args.model_id:
            cfg["models"]["judge"]={**cfg["models"]["task"],"provider":"http","model_id":args.model_id}
        packet=read_json(args.packet)
        if args.command=="reference":
            plan=calibration.reference_plan(packet,cfg,max_cost_usd=args.max_cost_usd)
        else:
            if not args.reference_run: parser.error("--reference-run required")
            plan=calibration.calibration_plan(packet,args.reference_run,cfg,max_cost_usd=args.max_cost_usd)
        result=plan
        if args.execute:
            if args.expected_plan_sha256!=plan["plan_sha256"]:
                raise ValueError("execution requires the exact frozen plan SHA256")
            job=(calibration.reference_execute(packet,cfg,plan) if args.command=="reference"
                 else calibration.calibration_execute(packet,args.reference_run,cfg,plan))
            result={"run_dir":job["run_dir"],"receipt":job["receipt"],"output":job["output"]}
    if args.output: write_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
