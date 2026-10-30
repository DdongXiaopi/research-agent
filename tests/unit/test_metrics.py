import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.metrics import (  # noqa: E402
    accuracy, aggregate_by_gold_label, citation_precision, delta_pp, p95, total_cost,
)
from src.util import redact, sha256_obj  # noqa: E402


class _O:
    def __init__(self, correct, cited=None, precision=0.0, cost=None, latency=0):
        self.correct = correct
        self.cited = cited or []
        self.citation_precision = precision
        self.cost = cost or {}
        self.latency_ms = latency


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        self.assertAlmostEqual(accuracy([_O(True), _O(False), _O(True), _O(False)]), 0.5)
        self.assertEqual(accuracy([]), 0.0)

    def test_citation_precision_ignores_uncited(self):
        outs = [_O(True, ["a"], 1.0), _O(False), _O(True, ["b"], 0.0)]
        self.assertAlmostEqual(citation_precision(outs), 0.5)

    def test_p95(self):
        vals = list(range(1, 101))
        self.assertGreaterEqual(p95(vals), 94)
        self.assertEqual(p95([]), 0.0)

    def test_total_cost(self):
        outs = [_O(True, cost={"usd": 0.1, "prompt_tokens": 10, "completion_tokens": 5}),
                _O(False, cost={"usd": 0.2, "prompt_tokens": 20, "completion_tokens": 5})]
        c = total_cost(outs)
        self.assertAlmostEqual(c["usd"], 0.3)
        self.assertEqual(c["prompt_tokens"], 30)

    def test_total_cost_does_not_turn_missing_usage_into_zero_cost(self):
        out = _O(True, cost={"usd": None, "usd_nominal": None,
                            "cost_estimated": True, "usage_available": False,
                            "cost_basis": "API usage counters unavailable"})
        c = total_cost([out])
        self.assertIsNone(c["usd"])
        self.assertIsNone(c["usd_nominal"])
        self.assertFalse(c["usage_complete"])
        self.assertTrue(c["cost_estimated"])

    def test_pi_catalog_zero_is_exposed_but_not_authoritative_bill_cost(self):
        out = _O(True, cost={
            "usd": None, "usd_nominal": 0.01, "cost_estimated": True,
            "usage_available": True, "prompt_tokens": 1000,
            "completion_tokens": 20,
            "pi_usage_cost_metadata": {"input": 0, "output": 0,
                                        "cacheRead": 0, "cacheWrite": 0,
                                        "total": 0},
            "pi_usage_cost_authoritative": False,
        })
        c = total_cost([out])
        self.assertIsNone(c["usd"])
        self.assertEqual(c["pi_usage_cost_metadata_usd"], 0.0)
        self.assertTrue(c["pi_usage_cost_metadata_complete"])
        self.assertFalse(c["pi_usage_cost_authoritative"])

    def test_delta_pp(self):
        self.assertAlmostEqual(delta_pp(0.5, 0.6), 10.0)

    def test_class_aggregates_report_metrics_without_task_level_answers(self):
        from types import SimpleNamespace

        tasks = [
            {"task_id": "y1", "options": ["yes", "no"], "correct_index": 0},
            {"task_id": "y2", "options": ["yes", "no"], "correct_index": 0},
            {"task_id": "n1", "options": ["yes", "no"], "correct_index": 1},
        ]
        outcomes = [
            SimpleNamespace(task_id="y1", correct=True, answer_index=0),
            SimpleNamespace(task_id="y2", correct=False, answer_index=-1),
            SimpleNamespace(task_id="n1", correct=False, answer_index=0),
        ]
        classes = aggregate_by_gold_label(outcomes, tasks)
        self.assertEqual(classes["yes"], {
            "n": 2, "accuracy": 0.5, "coverage": 0.5,
            "answered_precision": 1.0,
        })
        self.assertEqual(classes["no"], {
            "n": 1, "accuracy": 0.0, "coverage": 1.0,
            "answered_precision": 0.0,
        })

    def test_class_aggregates_reject_mixed_labeled_scope(self):
        from types import SimpleNamespace

        tasks = [
            {"task_id": "labeled", "options": ["yes", "no"], "correct_index": 0},
            {"task_id": "unlabeled"},
        ]
        outcomes = [
            SimpleNamespace(task_id="labeled", correct=True, answer_index=0),
            SimpleNamespace(task_id="unlabeled", correct=False, answer_index=-1),
        ]
        with self.assertRaisesRegex(ValueError, "mixes labeled and unlabeled"):
            aggregate_by_gold_label(outcomes, tasks)


class TestUtil(unittest.TestCase):
    def test_sha256_obj_stable(self):
        self.assertEqual(sha256_obj({"a": 1, "b": 2}), sha256_obj({"b": 2, "a": 1}))

    def test_redact(self):
        s = redact("Authorization: Bearer abcdef1234567890\napi_key = XYZ12345678")
        self.assertNotIn("abcdef1234567890", s)
        self.assertNotIn("XYZ12345678", s)

    def test_redact_preserves_field_name_at_start_and_masks_key_with_underscores(self):
        self.assertEqual(redact("api_key=abcdefghijklmnop"), "api_key=***REDACTED***")
        self.assertEqual(redact("sk-abcdefgh_ijklmnop"), "***REDACTED***")


if __name__ == "__main__":
    unittest.main()
