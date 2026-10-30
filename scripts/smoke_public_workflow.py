"""Exercise the public research path with synthetic input and zero API calls."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    # RSI_HOME must be bound before importing modules with release path constants.
    with tempfile.TemporaryDirectory(prefix="rsi-public-workflow-") as sandbox:
        previous = os.environ.get("RSI_HOME")
        os.environ["RSI_HOME"] = sandbox
        try:
            from src.config import load_config
            from src.evolution.feedback_skill import build_feedback_skill_plan
            from src.evolution.session_feedback import export_development, record_feedback
            from src.governance.release import init_baseline
            from src.runtime.research_assistant import build_plan, execute, request_from_dict
            from src.runtime.research_tools import audit_open_research_trace, verify_citations

            init_baseline(
                registry_path=ROOT / "skills" / "registry.m1-v1.yaml",
                release_id="baseline-v1",
            )
            config = load_config(str(ROOT / "configs" / "experiment.yaml"))
            source = json.loads((ROOT / "configs" / "research" / "request.example.json").read_text(encoding="utf-8"))
            request = request_from_dict(source)
            plan = build_plan(request, config)
            result = execute(request, config, plan["plan_sha256"])
            run_dir = Path(result["run_dir"])
            citations = verify_citations(run_dir)
            trace = audit_open_research_trace(run_dir)
            accounting = result["response"]["execution_summary"]["request_accounting"]
            # Synthetic feedback exists only inside this disposable smoke fixture.
            record_feedback(run_dir, category="procedure",
                            correction="State that the synthetic measurements lack controls.",
                            reviewer="synthetic-smoke-fixture", use_for_evolution=True)
            development = export_development("assay-project", Path(sandbox) / "development.json")
            proposal = build_feedback_skill_plan(development, config)
            if (result["response"]["status"] != "ok"
                    or citations["status"] != "identity_verified"
                    or not trace["answer_selected"]
                    or trace["provider_calls"] != 0
                    or accounting["physical_http_attempts"] != 0
                    or len(development["examples"]) != 1
                    or proposal["model_calls"] != 0
                    or proposal["publication_authorized"] is not False):
                raise RuntimeError("public workflow integration check failed")
            print(json.dumps({
                "status": "ok", "release_id": "baseline-v1",
                "plan_sha256": plan["plan_sha256"],
                "citation_identity_verified": True,
                "answer_selected": True, "provider_requests": 0,
                "development_examples": 1,
                "proposal_plan_only": True,
            }, sort_keys=True))
            return 0
        finally:
            if previous is None:
                os.environ.pop("RSI_HOME", None)
            else:
                os.environ["RSI_HOME"] = previous


if __name__ == "__main__":
    raise SystemExit(main())
