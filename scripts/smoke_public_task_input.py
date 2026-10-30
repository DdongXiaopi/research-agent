"""Public supplied-task acceptance, optionally with a live HTTP/Pi synthetic lifecycle."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(provider, model_id="Qwen/Qwen3-8B"):
    import yaml
    from src.config import load_config, config_fingerprint
    from src.contracts.types import Candidate, content_id, content_sha256
    from src.data.public_tasks import prepare
    from src.data.freeze import freeze, verify
    from scripts.make_splits import _commit_prepared
    from src.eval.controlled_experiment import build_plan, execute
    from src.governance import release, review
    from src.util import artifacts_path, write_json

    cfg = load_config(str(ROOT / "configs/experiment.public-task-input.yaml"))
    assets = artifacts_path("engineering-fixture")
    assets.mkdir(parents=True, exist_ok=False)
    if provider != "fake":
        from src.runtime.free_model import verify_current
        if not os.environ.get("SILICONFLOW_API_KEY"):
            raise ValueError("live acceptance credential is unavailable")
        verify_current(model_id, assets / "zero-price-verification")
        cfg["runtime"].update(provider=provider, http_no_cache_required=True,
            pi_provider_max_retries=0, pi_timeout_max_attempts=1, pi_request_cooldown_seconds=0)
        cfg["models"]["task"].update(provider="http", model_id=model_id,
            base_url="https://api.siliconflow.cn/v1", api_key_env="SILICONFLOW_API_KEY",
            temperature=0, max_tokens=160, timeout=90, max_retries=1, enable_thinking=False,
            cache=False, structured_output="json_schema")
    cfg["evolution"]["behavior_channels"].update(enabled=False, enforce_in_evolution=False)
    dataset = "public_acceptance"
    cfg["evaluation"].update(datasets=[dataset], manifests=[dataset + ".json"],
                              transfer_datasets=[], primary_final_datasets=[dataset])
    rows = [{"task_id": f"synthetic-acceptance-task-{i:04d}", "question": f"Which marker does code K7Q4 represent in case {i}?",
             "options": ["alpha marker", "beta marker"], "correct_index": 1,
             "evidence": f"Synthetic acceptance case {i} reports encoded marker K7Q4. Apply the active release's codebook to decode it.",
             "sources": [f"synthetic://acceptance/{i}"]} for i in range(80)]
    manifest, visible, sealed = prepare(dataset, rows, source="synthetic public engineering acceptance fixture", license="SELF-AUTHORED")
    _commit_prepared([(dataset, manifest, visible, sealed)])
    freeze([dataset])
    if not verify([dataset])["ok"]:
        raise ValueError("public prepared tasks did not freeze")
    registry = assets / "registry.yaml"
    registry.write_text(yaml.safe_dump({"release_id": "public-fixed-base", "skills": []}), encoding="utf-8")
    prompt = assets / "prompt.txt"
    prompt.write_text("Synthetic engineering baseline codebook: K7Q4 means alpha marker. Decode the supplied case using this codebook and return the corresponding option in the required format. This deliberately incorrect codebook measures only publication wiring.", encoding="utf-8")
    release.init_baseline(prompt_path=prompt, registry_path=registry, release_id="public-fixed-base")
    cap = 12 if provider != "fake" else 0
    spec = {"base": "public-fixed-base", "compare": None}
    candidate = None
    if provider != "fake":
        text = "Synthetic engineering corrected codebook: K7Q4 means beta marker. Decode the supplied case using this codebook and return the corresponding option in the required format. Cite input:0 when citation is enabled. CITED: source"
        candidate = Candidate(content_id("cand", "public-fixed-base", "system_prompt", text), "public-fixed-base",
            "system_prompt", "system_prompt", new_content=text, content_sha256=content_sha256(text),
            diff="Replace intentionally incorrect synthetic baseline with evidence reading.",
            rationale="Synthetic integration acceptance only; no scientific improvement claim.",
            generator={"backend": "authored-engineering-fixture", "engineering_only": True})
        spec["candidate"] = candidate.to_dict()
    plan = build_plan(cfg, kind="multiseed", spec=spec, datasets=[dataset],
        splits=["dev_optimize", "selection_gate"], seeds=[11, 12, 13],
        limit=1 if provider != "fake" else 2,
        logical_request_cap=cap, physical_request_cap=cap)
    write_json(assets / "planned-experiment.json", plan)
    measured = execute(cfg, plan, plan["plan_sha256"])
    summary = {"status": "ok", "provider": provider, "engineering_only": True,
        "protocol_epoch": cfg["runtime"]["protocol_epoch"], "protocol_sha256": config_fingerprint(cfg),
        "request_accounting": measured["request_accounting"], "final_holdout_read": False,
        "source_groups": manifest["n_source_groups"], "scientific_gain_claim": False}
    if candidate:
        from src.governance.engineering_fixed import create_request
        requested = create_request(Path(measured["run_dir"]) / "report.json", cfg, candidate.to_dict())
        if requested["verdict"] != "proposed_accept":
            write_json(assets / "failed-gates.json", requested)
            raise ValueError("live synthetic candidate did not pass its measured acceptance gates")
        review.decide(candidate.candidate_id, "approve", reviewer="auto-sim",
                      reason="Isolated measured synthetic HTTP/Pi engineering acceptance only.")
        req = review.read_request(candidate.candidate_id)
        published = release.publish(candidate.parent_release_id, candidate,
                                    {"optimize": req["optimize_metrics"], "gate": req["gate_metrics"]})
        release.rollback(candidate.parent_release_id)
        summary.update(published_release=published, rollback_release=release.current_release_id())
    else:
        ablate = build_plan(cfg, kind="ablate", spec={"base": "public-fixed-base",
            "conditions": {"A0": "public-fixed-base", "A1": "public-fixed-base"}}, datasets=[dataset],
            splits=["dev_optimize"], seeds=[11, 12, 13], limit=2)
        execute(cfg, ablate, ablate["plan_sha256"])
        summary["ablation_executed"] = True
        campaign = subprocess.run(
            [sys.executable, str(ROOT / "scripts/rsi.py"), "--config",
             str(ROOT / "configs/experiment.public-task-input.yaml"), "campaign",
             "--datasets", dataset, "--rounds", "3", "--limit", "2", "--no-approve"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if campaign.returncode or "=== campaign ===" not in campaign.stdout:
            raise RuntimeError("public three-round campaign failed: " + campaign.stderr[-2000:])
        report = json.loads(campaign.stdout.split("=== campaign ===", 1)[1])
        if (report.get("rounds_executed") != 3
                or [step.get("round") for step in report.get("steps", [])] != [1, 2, 3]
                or report.get("rounds_published") != 0
                or report.get("final_release") != "public-fixed-base"):
            raise RuntimeError("public campaign did not complete the three-round fixture")
        from src.eval import leakage
        if not leakage.check_all([artifacts_path(name) for name in
                                  ("runs", "rounds", "candidates", "reviews", "campaigns")])["ok"]:
            raise RuntimeError("public campaign leaked final holdout identities")
        summary["campaign_rounds_executed"] = 3
        summary["campaign_published"] = 0
    write_json(assets / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--provider", choices=["fake", "http", "pi"], default="fake")
    parser.add_argument("--model-id", choices=["Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct", "THUDM/GLM-4-9B-0414"], default="Qwen/Qwen3-8B")
    parser.add_argument("--run-home", type=Path)
    args = parser.parse_args()
    if args.worker:
        if os.environ.get("RSI_PUBLIC_CAPABILITY_WORKER") != "1":
            parser.error("worker requires the isolated launcher")
        print(json.dumps(worker(args.provider, args.model_id), sort_keys=True))
        return 0
    if args.provider != "fake" and args.run_home is None:
        parser.error("live acceptance requires a new persistent --run-home")
    if args.provider != "fake":
        from src.runtime.provider import load_project_dotenv
        load_project_dotenv()
    from scripts.build_public_snapshot import source_files
    with tempfile.TemporaryDirectory(prefix="rsi-public-task-input-") as tmp:
        destination = (args.run_home or Path(tmp) / "acceptance").resolve()
        if destination.exists():
            parser.error("acceptance destination must not exist")
        workspace = destination / "workspace"
        for source in source_files():
            target = workspace / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        if args.provider == "pi":
            installed = ROOT / "tools/pi-bridge/node_modules"
            if not installed.is_dir():
                parser.error("install the locked Pi bridge with bootstrap_project.py --install-pi")
            shutil.copytree(installed, workspace / "tools/pi-bridge/node_modules")
        env = {**os.environ, "RSI_HOME": str(destination / "runtime"), "RSI_PUBLIC_CAPABILITY_WORKER": "1",
               "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run([sys.executable, str(workspace / "scripts/smoke_public_task_input.py"),
            "--worker", "--provider", args.provider, "--model-id", args.model_id], cwd=workspace, env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            raise RuntimeError("public task-input acceptance failed: " + result.stderr[-2200:])
        print(result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
