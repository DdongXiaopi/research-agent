"""M2：配对 bootstrap / 阈值判定的统计性质。

这些断言校验的是**统计工具本身**，不是实验结论：
- 两份完全一样的逐题结果 → Δ 必须恰好 0，CI 退化为 [0,0]
- 全对 vs 全错 → Δ 必须 100pp
- 一半题由错翻对 → Δ ≈ 50pp，且 P(Δ>0)=1
- 同 rng_seed 必须复现（否则报告里的 CI 不可信）
- `meets_threshold` 的"或"规则：Δ 达阈值 或 P 达阈值，任一满足即可
"""

from __future__ import annotations

import unittest

from src.eval.stats import (
    meets_threshold,
    paired_bootstrap,
    paired_bootstrap_binary,
    self_test,
    summary,
)

N = 60


class TestPairedBootstrap(unittest.TestCase):
    def test_identical_sequences_give_zero_delta(self) -> None:
        base = [1] * 30 + [0] * 30
        ci = paired_bootstrap_binary(base, list(base))
        self.assertEqual(ci["delta_pp"], 0.0)
        self.assertEqual(ci["p_delta_gt_0"], 0.0)
        self.assertEqual(ci["n"], len(base))

    def test_full_flip_gives_100pp(self) -> None:
        base = [0] * N
        cand = [1] * N
        ci = paired_bootstrap_binary(base, cand)
        self.assertAlmostEqual(ci["delta_pp"], 100.0, places=6)
        self.assertAlmostEqual(ci["p_delta_gt_0"], 1.0, places=6)

    def test_partial_improvement(self) -> None:
        base = [0] * N
        cand = [1] * (N // 2) + [0] * (N // 2)
        ci = paired_bootstrap_binary(base, cand)
        self.assertAlmostEqual(ci["delta_pp"], 50.0, places=6)
        self.assertGreaterEqual(ci["p_delta_gt_0"], 0.999)
        self.assertLessEqual(ci["ci_low_pp"], 50.0)
        self.assertGreaterEqual(ci["ci_high_pp"], 50.0)

    def test_deterministic_given_rng_seed(self) -> None:
        import random

        rng = random.Random(7)
        base = [rng.randint(0, 1) for _ in range(80)]
        cand = [1 if (b == 1 or rng.random() < 0.3) else rng.randint(0, 1) for b in base]
        a = paired_bootstrap_binary(base, cand, B=2000, rng_seed=20260924)
        b = paired_bootstrap_binary(base, cand, B=2000, rng_seed=20260924)
        self.assertEqual(a, b)

    def test_continuous_version_agrees_on_binary_input(self) -> None:
        base = [0, 1, 0, 1, 1, 0, 1, 0]
        cand = [1, 1, 0, 1, 1, 1, 1, 0]
        a = paired_bootstrap_binary(base, cand, B=500, rng_seed=1)
        b = paired_bootstrap([float(x) for x in base], [float(x) for x in cand],
                             B=500, rng_seed=1)
        self.assertAlmostEqual(a["delta_pp"], b["delta_pp"], places=6)

    def test_summary_fields(self) -> None:
        s = summary([0.1, 0.2, 0.3])
        self.assertAlmostEqual(s["mean"], 0.2, places=6)
        self.assertIn("std", s)
        self.assertIn("min", s)
        self.assertIn("max", s)


class TestThresholdRule(unittest.TestCase):
    def test_delta_meets_threshold(self) -> None:
        r = meets_threshold({"delta_pp": 2.0, "p_delta_gt_0": 0.4},
                            min_delta_pp=1.0, min_prob=0.90)
        self.assertTrue(r["passed"])

    def test_probability_meets_threshold(self) -> None:
        r = meets_threshold({"delta_pp": 0.5, "p_delta_gt_0": 0.95},
                            min_delta_pp=1.0, min_prob=0.90)
        self.assertTrue(r["passed"])

    def test_neither_meets(self) -> None:
        r = meets_threshold({"delta_pp": 0.5, "p_delta_gt_0": 0.5},
                            min_delta_pp=1.0, min_prob=0.90)
        self.assertFalse(r["passed"])


class TestStatsSelfTest(unittest.TestCase):
    def test_self_test_passes(self) -> None:
        rep = self_test()
        self.assertTrue(rep["ok"], str(rep))


class TestInvalidStatisticalInputs(unittest.TestCase):
    def test_mcnemar_rejects_unpaired_and_nonbinary_values(self) -> None:
        from src.eval.stats import mcnemar_exact
        for before, after in (([0, 1], [1]), ([0, 2], [1, 0]), ([float("nan")], [0])):
            with self.subTest(before=before, after=after), self.assertRaises(ValueError):
                mcnemar_exact(before, after)

    def test_mcnemar_handles_more_than_1023_discordant_pairs(self) -> None:
        from src.eval.stats import mcnemar_exact
        result = mcnemar_exact([0] * 1500, [1] * 1500)
        self.assertEqual(result["n_improved"], 1500)
        self.assertEqual(result["p_value"], 0.0)

    def test_bootstrap_rejects_invalid_parameters_and_nonfinite_data(self) -> None:
        for before, after, kwargs in (([0], [1], {"B": 0}), ([0], [1], {"alpha": 1}),
                                      ([float("nan")], [1], {}), ([-1e308], [1e308], {})):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                paired_bootstrap(before, after, **kwargs)

    def test_binary_bootstrap_rejects_nonbinary_values(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap_binary([0, 2], [1, 0], B=10)

    def test_holm_rejects_invalid_probabilities(self) -> None:
        from src.eval.stats import holm_bonferroni
        for values in ([1.1], [-0.1], [float("nan")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                holm_bonferroni(values)


if __name__ == "__main__":
    unittest.main()
