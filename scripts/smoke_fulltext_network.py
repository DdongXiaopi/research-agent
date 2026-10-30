"""Optional real OA -> native document retrieval -> zero-model answer acceptance."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-home", type=Path, required=True)
    args = parser.parse_args()
    destination = args.run_home.resolve()
    if destination.exists():
        parser.error("network acceptance requires a new isolated run-home")
    os.environ["RSI_HOME"] = str(destination)
    import requests
    from src.util import artifacts_path, write_json, sha256_obj, sha256_file
    from src.retrieval import documents
    from src.runtime import research_assistant as research
    from src.runtime.research_tools import verify_citations, audit_open_research_trace
    from src.governance.release import init_baseline
    from src.config import load_config

    root = artifacts_path()
    root.mkdir(parents=True, exist_ok=False)
    url = "https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001745"
    response = requests.get(url, timeout=20, stream=True, allow_redirects=False)
    try:
        response.raise_for_status()
        payload = bytearray()
        for block in response.iter_content(65536):
            if len(payload) + len(block) > 2_000_000:
                raise ValueError("license page exceeded byte ceiling")
            payload.extend(block)
    finally:
        response.close()
    if b"Creative Commons Attribution" not in payload:
        raise ValueError("source article's open license was not verified")
    (root / "license-page.html").write_bytes(payload)
    snapshot = {"schema": "literature-snapshot-v1", "records": [{"source_id": "plos-wilson2014",
        "title": "Best Practices for Scientific Computing", "doi": "10.1371/journal.pbio.1001745",
        "oa_license": "cc-by", "license_source": url,
        "oa_url": "https://journals.plos.org/plosbiology/article/file?id=10.1371/journal.pbio.1001745&type=manuscript",
        "raw_metadata_sha256": sha256_file(root / "license-page.html")} ]}
    snapshot["snapshot_sha256"] = sha256_obj(snapshot)
    write_json(root / "metadata.json", snapshot)
    plan = documents.build_plan(snapshot, "plos-wilson2014", timeout=30)
    fetched = documents.execute(snapshot, plan, plan["plan_sha256"])
    query = "Which version control and testing practices are recommended?"
    selected = documents.retrieval_snapshot(fetched, query, backend="paperqa", top_k=3)
    write_json(root / "selected-documents.json", selected)
    init_baseline(release_id="baseline-v1")
    cfg = load_config(str(ROOT / "configs/experiment.yaml"))
    request = research.request_from_dict({"task_id": "fulltext-integration", "task_type": "open_research",
        "query": query, "project_id": "document-acceptance", "session_id": "one",
        "evidence": documents.selected_evidence(selected)})
    answer_plan = research.build_plan(request, cfg, retrieval_snapshot=selected)
    result = research.execute(request, cfg, answer_plan["plan_sha256"], retrieval_snapshot=selected)
    citations = verify_citations(Path(result["run_dir"]))
    audit_open_research_trace(Path(result["run_dir"]))
    summary = {"status": "ok", "full_text_downloaded": True,
        "characters": len(fetched["documents"][0]["text"]),
        "download_sha256": fetched["documents"][0]["download_sha256"],
        "native_adapter": selected["adapter"], "citation_status": citations["status"],
        "provider_requests": 0, "download_requests": fetched["physical_requests"], "license_source": url}
    write_json(root / "summary.json", summary)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
