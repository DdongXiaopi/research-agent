"""Run the fixed-task evolution loop on disposable synthetic data only."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _copy_public_runtime(destination: Path) -> None:
    for folder, suffixes in (("src", {".py"}),
                             ("skills", {".yaml", ".md", ".txt"}),
                             ("configs", {".yaml", ".json", ".txt", ".md"})):
        for source in (ROOT / folder).rglob("*"):
            if (source.is_file() and not source.is_symlink() and source.suffix in suffixes
                    and source.name != "experiment.docker-tools.yaml"):
                target = destination / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    for relative in ("scripts/rsi.py", "scripts/smoke_evolution_loop.py"):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)


def _make_fixture() -> Path:
    import yaml

    from src.data.freeze import freeze
    from src.data.split import assign_split
    from src.util import sha256_obj

    dataset = "litqa2"
    salt = "synthetic-public-evolution-smoke-v1"
    splits = {name: [] for name in ("dev_optimize", "selection_gate", "final_holdout")}
    visible, sealed = [], []
    labels = ["alpha", "beta", "gamma", "delta"]
    for index in range(120):
        task_id = f"synthetic-public-{index:03d}"
        split = assign_split(dataset, task_id, salt)
        splits[split].append(task_id)
        answer = index % len(labels)
        row = {
            "task_id": task_id, "task_type": "mcq", "dataset": dataset,
            "domain": "biomedical", "split": split,
            "question": f"Which label is reported for synthetic case {index}?",
            "options": labels, "correct_index": answer,
            "evidence": f"Synthetic case {index} reports label {labels[answer]}.",
            "context": "", "sources": [f"synthetic://case/{index}"],
        }
        (sealed if split == "final_holdout" else visible).append(row)
    for relative, rows in (("data/normalized/litqa2.jsonl", visible),
                           ("data/sealed/litqa2.final_holdout.jsonl", sealed)):
        target = ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if not visible or not sealed or not splits["selection_gate"]:
        raise RuntimeError("synthetic fixture did not populate all split roles")
    manifest = {
        "dataset": dataset, "n_questions": 120, "source_file": "generated synthetic fixture",
        "source_sha256": hashlib.sha256(b"generated synthetic fixture").hexdigest(),
        "salt": salt, "ratios": {"dev_optimize": 60, "selection_gate": 20,
                             "final_holdout": 20},
        "split_algorithm": "sha256(salt|dataset|task_id)[:8] % 100",
        "splits": {name: {"ids": sorted(ids), "ids_sha256": sha256_obj(sorted(ids))}
                   for name, ids in splits.items()},
    }
    manifest_path = ROOT / "data/manifests/litqa2.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    freeze([dataset], operator="synthetic-public-evolution-smoke")
    config = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text(encoding="utf-8"))
    config["retrieval"]["corpus_version"] = "synthetic-v2"
    config["evaluation"]["sealed_pairing"] = False
    config["evaluation"]["datasets"] = [dataset]
    config["evaluation"]["manifests"] = ["litqa2.json"]
    config["evaluation"]["transfer_datasets"] = []
    config["evaluation"]["primary_final_datasets"] = []
    config["runtime"]["fake_noise"] = 0.3
    config_path = ROOT / "configs/experiment.synthetic-smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _worker() -> int:
    config_path = _make_fixture()
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts/rsi.py"), "--config", str(config_path),
         "e2e", "--limit", "10"], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    if run.returncode:
        raise RuntimeError(f"synthetic e2e failed:\n{run.stdout}\n{run.stderr}")
    marker = "=== e2e ==="
    if marker not in run.stdout:
        raise RuntimeError("synthetic e2e did not return a report")
    report = json.loads(run.stdout.split(marker, 1)[1])
    steps = report.get("steps", [])
    names = {step.get("step") for step in steps}
    required = {"init", "baseline", "round", "reject", "rollback", "leakage"}
    if (not report.get("ok") or not required.issubset(names)
            or not ({"approve", "gate_blocks_approval"} & names)
            or not all(step.get("ok") for step in steps)):
        raise RuntimeError("synthetic e2e skipped or failed a governance step")
    print(json.dumps({"status": "ok", "synthetic_only": True,
                      "provider_requests": 0, "steps": sorted(names)}, sort_keys=True))
    return 0


def main() -> int:
    if sys.argv[1:] == ["--worker"]:
        if os.environ.get("RSI_PUBLIC_SMOKE_WORKER") != "1":
            raise SystemExit("--worker is internal to the disposable smoke runner")
        return _worker()
    if len(sys.argv) != 1:
        raise SystemExit("usage: python scripts/smoke_evolution_loop.py")
    with tempfile.TemporaryDirectory(prefix="rsi-public-evolution-") as directory:
        sandbox = Path(directory)
        _copy_public_runtime(sandbox)
        env = os.environ.copy()
        env["RSI_HOME"] = str(sandbox / "run-home")
        env["RSI_PUBLIC_SMOKE_WORKER"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env.pop("PYTHONPATH", None)
        run = subprocess.run([sys.executable, str(sandbox / "scripts/smoke_evolution_loop.py"),
                              "--worker"], cwd=sandbox, env=env, check=False)
        return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
