"""Check the public aggregate record; optionally match private source artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "PUBLIC-EVIDENCE-2026-10-19.json"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify(private_dir: Path | None = None) -> dict:
    record = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    if record.get("schema") != "rsi-public-evidence-summary-v1":
        raise ValueError("unexpected public evidence schema")
    if record["protocol"]["seeds"] != [20261019, 20261020, 20261021]:
        raise ValueError("historical seed list changed")
    results = record["results"]
    if set(results) != {"g08_qasper", "m4_bioasq"}:
        raise ValueError("historical cohort list changed")
    total_requests = 0
    for row in results.values():
        tasks = row["tasks"]
        if (not isinstance(tasks, int) or tasks <= 0 or row["article_groups"] > tasks
                or not 0 <= row["baseline_correct"] <= tasks
                or not 0 <= row["candidate_correct"] <= tasks):
            raise ValueError("invalid aggregate task counts")
        calculated = 100 * (row["candidate_correct"] - row["baseline_correct"]) / tasks
        if abs(calculated - row["paired_delta_pp"]) > 0.0001:
            raise ValueError("aggregate paired delta differs from counts")
        total_requests += row["http_200_requests_all_seeds"]
    if total_requests != 882:
        raise ValueError("historical request total changed")

    source_verified = private_dir is not None
    if private_dir is not None:
        private_dir = private_dir.resolve()
        for relative, expected in record["private_artifact_sha256"].items():
            path = private_dir / relative
            if not path.is_file() or digest(path) != expected:
                raise ValueError(f"private source artifact differs: {relative}")
        prereg = json.loads((private_dir / "preregistration.json").read_text(encoding="utf-8"))
        summary = json.loads((private_dir / "final-evidence-summary.json").read_text(encoding="utf-8"))
        provenance = json.loads((private_dir / "sources" / "provenance.json").read_text(encoding="utf-8"))
        for field in ("model", "parent", "candidate", "seeds", "primary_seed", "automatic_retries"):
            if prereg[field] != record["protocol"][field]:
                raise ValueError(f"private protocol differs: {field}")
        for key, private_key in (("g08_qasper", "g08"), ("m4_bioasq", "m4")):
            row, source = results[key], summary[private_key]
            seed = str(record["protocol"]["primary_seed"])
            primary = source["dataset_summary"]["per_seed"][seed]
            if (row["dataset"] != source["dataset"]
                    or row["assessment_sha256"] != source["assessment_sha256"]
                    or row["tasks"] != source["dataset_summary"]["n_tasks"]
                    or row["http_200_requests_all_seeds"] != source["requests_by_status"]["200"]
                    or abs(row["baseline_correct"] / row["tasks"] - primary["baseline_accuracy"]) > 1e-5
                    or abs(row["candidate_correct"] / row["tasks"] - primary["final_accuracy"]) > 1e-5):
                raise ValueError(f"private aggregate differs: {key}")
        source_hashes = {entry["sha256"] for entry in provenance["files"]}
        if any(item["sha256"] not in source_hashes for item in record["source_archives"]):
            raise ValueError("public source archive hashes differ from local provenance")
    return {"status": "ok", "cohorts": len(results), "http_200_requests": total_requests,
            "private_source_verified": source_verified}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", type=Path,
                        help="optional local closure directory; not included in the public repository")
    args = parser.parse_args()
    print(json.dumps(verify(args.private_dir), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
