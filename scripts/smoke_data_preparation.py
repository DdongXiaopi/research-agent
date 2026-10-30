"""Check lawful-input preparation and split freeze using disposable synthetic Parquet."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        import pandas as pd
        import pyarrow  # noqa: F401 - parquet engine required by pandas
    except ImportError as exc:
        raise SystemExit("install requirements-data.txt before this optional smoke") from exc
    from scripts.build_public_snapshot import source_files

    with tempfile.TemporaryDirectory(prefix="rsi-public-data-") as directory:
        sandbox = Path(directory)
        for source in source_files():
            target = sandbox / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        config = yaml.safe_load((sandbox / "configs/experiment.yaml").read_text(encoding="utf-8"))
        model_providers = [config.get("runtime", {}).get("provider"),
                           config.get("models", {}).get("task", {}).get("provider"),
                           config.get("models", {}).get("reflection", {}).get("provider")]
        if any(provider != "fake" for provider in model_providers):
            raise RuntimeError("public data smoke requires only fake providers")
        inputs = sandbox / "inputs"
        inputs.mkdir()
        for dataset in ("litqa2", "protocolqa"):
            rows = [{
                "id": f"synthetic-{dataset}-{index:03d}",
                "question": f"Which assay marker is reported in synthetic study {index}?",
                "ideal": ("alpha", "beta", "gamma")[index % 3],
                "distractors": [label for label in ("alpha", "beta", "gamma")
                                if label != ("alpha", "beta", "gamma")[index % 3]],
                "key-passage": (
                    f"Synthetic study {dataset} case {index} records assay marker "
                    f"{('alpha', 'beta', 'gamma')[index % 3]}. This unique report "
                    f"describes sample {index} and its control measurements, "
                    "documenting the observed result for this case."),
                "protocol": (
                    f"Synthetic study {dataset} case {index} records assay marker "
                    f"{('alpha', 'beta', 'gamma')[index % 3]}. This unique report "
                    f"describes sample {index} and its control measurements, "
                    "documenting the observed result for this case."),
                "sources": [f"synthetic://{dataset}/{index}"],
            } for index in range(40)]
            pd.DataFrame(rows).to_parquet(inputs / f"{dataset}.parquet", index=False)

        environment = dict(os.environ)
        environment["RSI_HOME"] = str(sandbox / "run-home")
        environment["PYTHONIOENCODING"] = "utf-8"

        def checked(command):
            run = subprocess.run(command, cwd=sandbox, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", env=environment)
            if run.returncode:
                raise RuntimeError(f"public data smoke failed: {run.stdout}\n{run.stderr}")
            return run.stdout

        checked([sys.executable, "scripts/make_splits.py", "--litqa2-parquet",
                 "inputs/litqa2.parquet", "--protocolqa-parquet",
                 "inputs/protocolqa.parquet"])
        checked([sys.executable, "scripts/rsi.py", "freeze", "--datasets", "litqa2",
                 "protocolqa"])
        frozen = json.loads((sandbox / "data/manifests/freeze.json").read_text(encoding="utf-8"))
        counts = {}
        for dataset in ("litqa2", "protocolqa"):
            manifest = json.loads((sandbox / f"data/manifests/{dataset}.json").read_text(encoding="utf-8"))
            split_counts = {name: len(info["ids"]) for name, info in manifest["splits"].items()}
            if (sum(split_counts.values()) != 40 or not all(split_counts.values())
                    or dataset not in frozen["datasets"]
                    or not frozen["sealed_sha256"][dataset]):
                raise RuntimeError("synthetic data freeze is incomplete")
            counts[dataset] = split_counts
        checked([sys.executable, "scripts/rsi.py", "init"])
        baseline = checked([sys.executable, "scripts/rsi.py", "baseline", "--datasets",
                            "litqa2", "protocolqa", "--limit", "10", "--validate-only"])
        baseline_plan = json.loads(baseline.split("=== baseline ===", 1)[1])
        if baseline_plan.get("final_holdout_read") is not False:
            raise RuntimeError("baseline plan touched synthetic final holdout")
        checked([sys.executable, "scripts/rsi.py", "baseline", "--datasets", "litqa2",
                 "protocolqa", "--limit", "10", "--execute", "--expected-plan-sha256",
                 baseline_plan["plan_sha256"]])
        round_output = checked([sys.executable, "scripts/rsi.py", "round", "--datasets",
                                "litqa2", "protocolqa", "--limit", "10"])
        round_report = json.loads(round_output.split("=== round ===", 1)[1])
        candidates = [row for row in round_report.get("candidates", [])
                      if row.get("candidate_id") and row.get("status") == "proposed_reject"]
        if not candidates:
            raise RuntimeError("synthetic round did not create a reviewable rejection")
        checked([sys.executable, "scripts/rsi.py", "review", "--candidate-id",
                 candidates[0]["candidate_id"], "--decision", "reject", "--reason",
                 "synthetic public integration smoke", "--reviewer", "public-smoke"])
        rollback = checked([sys.executable, "scripts/rsi.py", "rollback", "--to",
                            "baseline-v1"])
        if not json.loads(rollback.split("=== rollback ===", 1)[1]).get("ok"):
            raise RuntimeError("synthetic rollback failed")
        print(json.dumps({"status": "ok", "synthetic_only": True,
                          "datasets": counts, "baseline_executed": True,
                          "round_reviewed": True, "rollback_ok": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
