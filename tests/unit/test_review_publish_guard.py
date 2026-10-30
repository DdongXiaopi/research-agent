"""A release cannot be activated without a matching approved review."""

from __future__ import annotations

import copy
import os
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.rsi import build_parser
from src.contracts.types import Candidate, ReviewRecord, content_id, content_sha256
from src.evolution.campaign import run_campaign
from src.governance import release as rel
from src.governance import review as rv
from src.util import artifacts_path, write_json


class TestReviewPublishGuard(unittest.TestCase):
    def _candidate(self) -> Candidate:
        parent = rel.current_release_id()
        content = "Review guard test prompt. ANSWER: option; CITED: source; INSUFFICIENT: abstain."
        return Candidate(
            candidate_id=content_id("cand", parent, "system_prompt", content),
            parent_release_id=parent,
            artifact_type="system_prompt", artifact_id="system_prompt",
            diff="test diff", new_content=content,
            content_sha256=content_sha256(content),
        )

    @staticmethod
    def _attach_complete_round_evidence(req: dict, *, persist_report: bool = True) -> None:
        req.update({
            "round_id": req.get("round_id") or "round-test-complete",
            "optimize_metrics": {"pubmedqa": {"accuracy": 0.6}},
            "gate_metrics": {"pubmedqa": {"accuracy": 0.6}},
            "evaluation_summary": {
                "baseline_metrics": {"pubmedqa": {"optimize": {"accuracy": 0.5}}},
                "candidate_metrics": {"pubmedqa": {
                    "optimize": {"accuracy": 0.6}, "gate": {"accuracy": 0.6}}},
                "task_support": {"n_paired": 10},
                "evaluation_support": {
                    "passed": True, "per_dataset_seed": [{"dataset": "pubmedqa",
                                                           "seed": 1,
                                                           "passed": True}]},
                "seed_requirement": {"passed": True},
                "tolerance": {"passed": True, "per_dataset": {}},
                "measurement_control": {
                    "control_type": "exact_baseline_repeat", "contaminated": False},
            },
        })
        if persist_report:
            TestReviewPublishGuard._write_round_report(req)

    @staticmethod
    def _write_round_report(req: dict) -> None:
        evidence = req["evaluation_summary"]
        candidate = {
            "candidate_id": req["candidate_id"],
            "parent_release_id": req["parent_release_id"],
            "content_sha256": req["new_content_sha256"],
            "status": "proposed_accept",
            "static_gate_passed": True,
            "candidate_metrics": evidence["candidate_metrics"],
            "baseline_metrics": evidence["baseline_metrics"],
            "task_support": evidence["task_support"],
            "evaluation_support": evidence["evaluation_support"],
            "tolerance": evidence["tolerance"],
            "seed_requirement": evidence["seed_requirement"],
        }
        write_json(artifacts_path("rounds", f"{req['round_id']}.json"), {
            "round_id": req["round_id"],
            "release_id": req["parent_release_id"],
            "measurement_control": evidence["measurement_control"],
            "candidates": [candidate],
        })

    def test_publish_requires_approved_matching_review(self) -> None:
        cand = self._candidate()
        before = rel.current_release_id()
        with self.assertRaisesRegex(rel.ReleaseError, "review"):
            rel.publish(before, cand, {})

        req = rv.build_request(cand, {}, {}, before,
                               {"passed": True, "results": []})
        req["verdict"] = "proposed_accept"
        rv.write_request(req)
        with self.assertRaisesRegex(rv.ReviewError, "完整证据摘要"):
            rv.decide(cand.candidate_id, "approve", reviewer="unit-test")
        req["evaluation_summary"] = {
            "baseline_metrics": {"test": {"accuracy": 0.5}},
            "candidate_metrics": {"test": {"accuracy": 0.6}},
            "task_support": {"n_paired": 10}, "tolerance": {"passed": True},
            "measurement_control": {"status": "ok",
                                    "control_type": "exact_baseline_repeat",
                                    "contaminated": True}}
        rv.write_request(req)
        with self.assertRaisesRegex(rv.ReviewError, "完整证据摘要"):
            rv.decide(cand.candidate_id, "approve", reviewer="unit-test")
        rv.decide(cand.candidate_id, "reject", reviewer="unit-test")
        with self.assertRaisesRegex(rel.ReleaseError, "review 请求"):
            rel.publish(before, cand, {})

        self._attach_complete_round_evidence(req)
        rv.write_request(req)
        approved = rv.decide(cand.candidate_id, "approve", reviewer="unit-test")
        review_record = approved["review"]
        self.assertEqual(review_record["diff"], cand.diff)
        self.assertEqual(review_record["rollback_point"], before)
        self.assertEqual(review_record["metrics"]["optimize"], req["optimize_metrics"])
        self.assertEqual(review_record["metrics"]["gate"], req["gate_metrics"])
        changed = Candidate(**cand.to_dict())
        changed.new_content += " Altered after review."
        with self.assertRaisesRegex(rel.ReleaseError, "review 请求"):
            rel.publish(before, changed, {})
        self.assertEqual(rel.current_release_id(), before)

    def test_rejected_candidate_cannot_be_approved(self) -> None:
        cand = self._candidate()
        req = rv.build_request(cand, {}, {}, cand.parent_release_id,
                               {"passed": True, "results": []})
        req["verdict"] = "proposed_reject"
        rv.write_request(req)
        with self.assertRaisesRegex(rv.ReviewError, "未通过评测门"):
            rv.decide(cand.candidate_id, "approve", reviewer="unit-test")

    def test_metadata_cannot_change_under_an_approved_body_hash(self) -> None:
        cand = self._candidate()
        cand.extra = {"reviewed_metadata": "original"}
        req = rv.build_request(cand, {}, {}, cand.parent_release_id, {"passed": True, "results": []})
        req["verdict"] = "proposed_accept"
        self._attach_complete_round_evidence(req)
        rv.write_request(req)
        rv.decide(cand.candidate_id, "approve", reviewer="isolated-test-reviewer")
        cand.extra["reviewed_metadata"] = "changed after review"
        with self.assertRaisesRegex(rel.ReleaseError, "元数据"):
            rel.publish(cand.parent_release_id, cand, {"optimize": req["optimize_metrics"], "gate": req["gate_metrics"]})

    def test_publish_rejects_review_request_changed_after_approval(self) -> None:
        cand = self._candidate()
        parent = rel.current_release_id()
        req = rv.build_request(cand, {}, {}, parent,
                               {"passed": True, "results": []})
        req["verdict"] = "proposed_accept"
        self._attach_complete_round_evidence(req)
        rv.write_request(req)
        rv.decide(cand.candidate_id, "approve", reviewer="unit-test")

        req["rationale"] = "Changed after human approval"
        rv.write_request(req)
        with self.assertRaisesRegex(rel.ReleaseError, "批准后发生变化"):
            rel.publish(parent, cand, {})
        self.assertEqual(rel.current_release_id(), parent)

    def test_publish_metrics_must_match_the_approved_review_request(self) -> None:
        cand = self._candidate()
        cand.new_content += "\nmetrics-binding regression"
        cand.content_sha256 = content_sha256(cand.new_content)
        cand.candidate_id = content_id(
            "cand", cand.parent_release_id, cand.artifact_type, cand.new_content)
        parent = rel.current_release_id()
        req = rv.build_request(cand, {}, {}, parent,
                               {"passed": True, "results": []})
        req["verdict"] = "proposed_accept"
        self._attach_complete_round_evidence(req)
        rv.write_request(req)
        rv.decide(cand.candidate_id, "approve", reviewer="unit-test")

        approved_metrics = {
            "optimize": req["optimize_metrics"],
            "gate": req["gate_metrics"],
        }
        with self.assertRaisesRegex(rel.ReleaseError, "metrics approved"):
            rel.publish(parent, cand, {"optimize": {}, "gate": {}})
        self.assertEqual(rel.current_release_id(), parent)

        try:
            release_id = rel.publish(parent, cand, approved_metrics)
            self.assertEqual(rel.release_info(release_id)["metrics"], approved_metrics)
            self.assertEqual(rel.current_release_id(), release_id)
        finally:
            rel.rollback(parent)

    def test_publish_rejects_unreviewed_registry_override(self) -> None:
        cand = self._candidate()
        cand.new_content += "\nunreviewed registry override regression"
        cand.content_sha256 = content_sha256(cand.new_content)
        cand.candidate_id = content_id(
            "cand", cand.parent_release_id, cand.artifact_type, cand.new_content)
        parent = rel.current_release_id()
        req = rv.build_request(cand, {}, {}, parent,
                               {"passed": True, "results": []})
        req["verdict"] = "proposed_accept"
        self._attach_complete_round_evidence(req)
        rv.write_request(req)
        rv.decide(cand.candidate_id, "approve", reviewer="unit-test")

        altered_registry = copy.deepcopy(rel.load_state(parent).registry)
        altered_registry.skills[0].description += " unreviewed change"
        approved_metrics = {
            "optimize": req["optimize_metrics"],
            "gate": req["gate_metrics"],
        }
        output_release = artifacts_path(
            "releases", f"rel-{cand.candidate_id}")
        with self.assertRaisesRegex(rel.ReleaseError, "new_registry overrides"):
            rel.publish(parent, cand, approved_metrics, new_registry=altered_registry)
        self.assertEqual(rel.current_release_id(), parent)
        self.assertFalse(output_release.exists())

    def test_round_request_requires_final_evidence_before_approval(self) -> None:
        cand = self._candidate()
        req = rv.build_request(cand, {}, {}, cand.parent_release_id,
                               {"passed": True, "results": []})
        req["verdict"] = "proposed_accept"
        req["round_id"] = "round-test"
        rv.write_request(req)
        with self.assertRaisesRegex(rv.ReviewError, "完整证据摘要"):
            rv.decide(cand.candidate_id, "approve", reviewer="unit-test")

    def test_fabricated_round_id_cannot_substitute_for_persisted_round(self) -> None:
        cand = self._candidate()
        req = rv.build_request(cand, {}, {}, cand.parent_release_id,
                               {"passed": True, "results": []})
        req["verdict"] = "proposed_accept"
        req["round_id"] = f"unpersisted-{cand.candidate_id}"
        self._attach_complete_round_evidence(req, persist_report=False)
        rv.write_request(req)
        self.assertFalse(rv.request_effect_gate_ok(req))
        with self.assertRaisesRegex(rv.ReviewError, "完整证据摘要"):
            rv.decide(cand.candidate_id, "approve", reviewer="unit-test")

    def test_real_provider_cost_must_be_measured_before_round_approval(self) -> None:
        req = {
            "candidate_id": "candidate-real-provider",
            "parent_release_id": "release-parent",
            "new_content_sha256": "test-content-hash",
            "verdict": "proposed_accept", "round_id": "round-real-provider",
            "static_gate": {"passed": True},
            "optimize_metrics": {"pubmedqa": {"accuracy": 0.72}},
            "gate_metrics": {"pubmedqa": {"accuracy": 0.72}},
            "evaluation_summary": {
                "task_support": {"n_paired": 90},
                "evaluation_support": {
                    "passed": True, "per_dataset_seed": [{"dataset": "pubmedqa",
                                                           "seed": 1,
                                                           "passed": True}]},
                "seed_requirement": {"passed": True},
                "measurement_control": {
                    "control_type": "exact_baseline_repeat", "contaminated": False},
                "tolerance": {
                    "passed": True,
                    "per_dataset": {"pubmedqa": {
                        "provider_kind": "pi",
                        "items": [{"quantity": "cost", "status": "n_a"}],
                    }},
                },
                "baseline_metrics": {"pubmedqa": {"optimize": {"accuracy": 0.7}}},
                "candidate_metrics": {"pubmedqa": {
                    "optimize": {"accuracy": 0.72},
                    "gate": {"accuracy": 0.72}}},
            },
        }
        self._write_round_report(req)
        self.assertFalse(rv.request_effect_gate_ok(req))
        req["evaluation_summary"]["tolerance"]["per_dataset"]["pubmedqa"]["items"][0]["status"] = "pass"
        self._write_round_report(req)
        self.assertTrue(rv.request_effect_gate_ok(req))
        req["evaluation_summary"]["tolerance"]["passed"] = False
        self.assertFalse(rv.request_effect_gate_ok(req))

    def test_campaign_defaults_to_manual_review(self) -> None:
        args = build_parser().parse_args(["campaign"])
        self.assertFalse(args.approve_simulated)
        self.assertFalse(args.no_approve)
        self.assertFalse(
            inspect.signature(run_campaign).parameters["approve"].default)

    @mock.patch("src.governance.review.verify_confirmation_attestation",
                new=lambda *args, **kwargs: True)
    def test_cost_exception_approval_requires_explicit_flag_and_reason(self) -> None:
        cand = self._candidate()
        cost_runs = {"101": {"passed": False, "items": [{
            "quantity": "cost", "status": "fail", "base": 1.0,
            "cand": 1.2, "limit": 15.0,
        }]}}
        exception = {"eligible": True,
                     "requires_explicit_human_cost_approval": True,
                     "per_execution_label": {"101": {
                         "baseline_usd": 1.0, "candidate_usd": 1.2,
                         "limit_pct": 15.0, "observed_delta_pct": 20.0,
                     }}}
        req = {
            "candidate_id": cand.candidate_id,
            "parent_release_id": cand.parent_release_id,
            "new_content_sha256": cand.content_sha256,
            "diff": cand.diff, "rollback_point": cand.parent_release_id,
            "verdict": "proposed_accept", "evidence_type": "aggregate_selection_gate_v1",
            "static_gate": {"passed": True}, "cost_overrun_review": exception,
            "evaluation_summary": {
                "official_validation_read": True, "official_test_read": False,
                "per_task_output_written": False, "promotion_promotable": False,
                "cost_exception_reviewable": True, "protocol_checked": False,
                "execution_labels": [101], "baseline_metrics": {"accuracy": 0.5},
                "candidate_metrics": {"accuracy": 0.6},
                "tolerance_report": {"authoritative_cost": {
                    "passed": False, "per_execution_label": cost_runs}},
            },
            "gate_metrics": {"promotion": {
                "promotable": False, "cost_overrun_exception_reviewable": True,
                "cost_overrun_exception": exception,
                "checks": [
                    {"check": "protocol_same_origin", "passed": None},
                    {"check": "dataset:main", "passed": True,
                     "detail": {"role": "main", "primary_delta_pp": 1.0}},
                    {"check": "secondary_tolerances", "passed": False},
                ],
            }},
        }
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
                rv, "_dir", return_value=Path(temp_dir)):
            rv.write_request(req)
            with self.assertRaisesRegex(rv.ReviewError, "必须显式设置"):
                rv.decide(cand.candidate_id, "approve", reason="worth the cost")
            with self.assertRaisesRegex(rv.ReviewError, "必须填写"):
                rv.decide(cand.candidate_id, "approve", accept_cost_overrun=True)
            approved = rv.decide(cand.candidate_id, "approve",
                                 reason="measured gain justifies the cost",
                                 accept_cost_overrun=True)
            self.assertTrue(approved["review"]["metrics"][
                "cost_overrun_exception_accepted"])

    def test_review_cli_exposes_explicit_cost_exception_flag(self) -> None:
        args = build_parser().parse_args([
            "review", "--candidate-id", "candidate", "--decision", "approve",
            "--accept-cost-overrun", "--reason", "measured gain",
        ])
        self.assertTrue(args.accept_cost_overrun)

    @mock.patch("src.governance.review.verify_confirmation_attestation",
                new=lambda *args, **kwargs: True)
    def test_release_rejects_cost_exception_without_explicit_review_record(self) -> None:
        cand = self._candidate()
        cost_runs = {"101": {"passed": False, "items": [{
            "quantity": "cost", "status": "fail", "base": 1.0,
            "cand": 1.2, "limit": 15.0,
        }]}}
        exception = {"eligible": True,
                     "requires_explicit_human_cost_approval": True,
                     "per_execution_label": {"101": {
                         "baseline_usd": 1.0, "candidate_usd": 1.2,
                         "limit_pct": 15.0, "observed_delta_pct": 20.0,
                     }}}
        req = {
            "candidate_id": cand.candidate_id,
            "candidate_artifact_sha256": rv.candidate_artifact_sha256(cand),
            "parent_release_id": cand.parent_release_id,
            "new_content_sha256": cand.content_sha256,
            "diff": cand.diff, "rollback_point": cand.parent_release_id,
            "verdict": "proposed_accept", "evidence_type": "aggregate_selection_gate_v1",
            "static_gate": {"passed": True}, "cost_overrun_review": exception,
            "evaluation_summary": {
                "official_validation_read": True, "official_test_read": False,
                "per_task_output_written": False, "promotion_promotable": False,
                "cost_exception_reviewable": True, "protocol_checked": False,
                "execution_labels": [101], "baseline_metrics": {"accuracy": 0.5},
                "candidate_metrics": {"accuracy": 0.6},
                "tolerance_report": {"authoritative_cost": {
                    "passed": False, "per_execution_label": cost_runs}},
            },
            "gate_metrics": {"promotion": {
                "promotable": False, "cost_overrun_exception_reviewable": True,
                "cost_overrun_exception": exception,
                "checks": [
                    {"check": "protocol_same_origin", "passed": None},
                    {"check": "dataset:main", "passed": True,
                     "detail": {"role": "main", "primary_delta_pp": 1.0}},
                    {"check": "secondary_tolerances", "passed": False},
                ],
            }},
        }
        rv.write_request(req)
        rv.write_review(ReviewRecord(
            action_type="release", subject_sha256=cand.content_sha256,
            decision="approve", reviewer="unit-test", subject_id=cand.candidate_id,
            metrics={"cost_overrun_exception_accepted": False,
                     "review_request_sha256": rv.request_sha256(req)},
            rollback_point=cand.parent_release_id,
            parent_release_id=cand.parent_release_id, diff=cand.diff,
        ))
        approved_metrics = {
            "optimize": req.get("optimize_metrics", {}),
            "gate": req.get("gate_metrics", {}),
        }
        with self.assertRaisesRegex(rel.ReleaseError, "成本超支例外"):
            rel.publish(cand.parent_release_id, cand, approved_metrics)
        original_active = rel.current_release_id()
        try:
            rv.decide(cand.candidate_id, "approve", reviewer="unit-test",
                      reason="measured gain justifies the cost",
                      accept_cost_overrun=True)
            published = rel.publish(cand.parent_release_id, cand, approved_metrics)
            self.assertEqual(published, f"rel-{cand.candidate_id}")
            self.assertEqual(rel.current_release_id(), published)
        finally:
            rel.write_active(original_active)

    def test_simulated_approval_requires_isolated_artifacts(self) -> None:
        with mock.patch.dict(os.environ, {"RSI_HOME": ""}):
            with self.assertRaisesRegex(ValueError, "RSI_HOME"):
                run_campaign({}, rounds=1, approve=True)


if __name__ == "__main__":
    unittest.main()
