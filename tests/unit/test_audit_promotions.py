import json

import pytest

from scripts import audit_promotions as audit


def test_corrupt_round_cannot_be_silently_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROUNDS_ROOT", tmp_path)
    (tmp_path / "round-1.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed round"):
        audit._candidate_rows()


def test_missing_champion_metrics_produces_inconclusive_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROUNDS_ROOT", tmp_path)
    monkeypatch.setattr(audit.rel, "load_aliases", lambda: {"champion": "rel-c1"})
    (tmp_path / "round-1.json").write_text(json.dumps({"candidates": [{"candidate_id": "c1"}]}), encoding="utf-8")
    result = audit.audit({})
    assert result["n_candidates"] == 1
    assert result["champion_would_be_blocked"] is None
    assert not result["findings"][0]["audit_complete"]
    assert "无法判断" in result["verdict"]


def test_unmeasurable_tolerance_does_not_prove_champion_passed(monkeypatch):
    monkeypatch.setattr(audit.rel, "load_aliases", lambda: {"champion": "rel-c1"})
    monkeypatch.setattr(audit, "_candidate_rows", lambda: [{"round_id": "r", "candidate_id": "c1", "status": "promoted", "target": "x", "parent_release_id": "p", "candidate_metrics": {"d": {"optimize": {"x": 1}}}, "baseline_metrics": {"d": {"optimize": {"x": 1}}}}])
    monkeypatch.setattr(audit, "check_tolerances", lambda *args: {"items": [{"status": "n_a"}], "passed": True, "not_measurable": ["x"]})
    assert audit.audit({})["champion_would_be_blocked"] is None
