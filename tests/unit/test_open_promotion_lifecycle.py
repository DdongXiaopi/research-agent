"""Exercise the public feedback/promotion interface in a fresh process."""
import json
import subprocess
import sys
from pathlib import Path


def test_feedback_to_approved_release_rollback_and_tamper_guards():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, str(root / "scripts/smoke_open_promotion.py")],
                             cwd=root, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "ok" and report["tamper_guards_verified"]
    assert report["candidate_kind"] == "new_skill"
    assert report["physical_http_attempts"] == 0
    assert report["published_release"] != report["rollback_release"]
    assert report["engineering_only"] and not report["scientific_gain_claim"]
