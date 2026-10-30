import os
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import rsi


class RsiE2eProviderGuardTests(unittest.TestCase):
    def _run_with_config(self, cfg):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"RSI_HOME": directory}), \
                patch.object(rsi, "artifacts_root", return_value=Path(directory)), \
                patch.object(rsi, "load_config", return_value=cfg), \
                patch.object(rsi.rel, "init_baseline") as init_baseline, \
                patch.object(rsi, "evaluate") as evaluate:
            with self.assertRaisesRegex(RuntimeError, "fake-provider-only"):
                rsi.cmd_e2e(Namespace(config="live.yaml", seed=1, limit=1))
        init_baseline.assert_not_called()
        evaluate.assert_not_called()

    def test_http_runtime_is_rejected_before_baseline_or_evaluation(self):
        self._run_with_config({
            "runtime": {"provider": "http"},
            "models": {"task": {"provider": "http"}},
        })

    def test_http_model_override_is_rejected_even_with_fake_runtime(self):
        self._run_with_config({
            "runtime": {"provider": "fake"},
            "models": {"task": {"provider": "http"}},
        })

    def test_http_reflection_and_router_overrides_are_rejected(self):
        self._run_with_config({
            "runtime": {"provider": "fake"},
            "models": {"task": {"provider": "fake"},
                       "reflection": {"provider": "http"}},
            "skills": {"router_provider": "http"},
        })

    def test_fake_provider_config_is_accepted_by_guard(self):
        rsi._require_fake_e2e_providers({
            "runtime": {"provider": "fake"},
            "models": {"task": {"provider": "fake"},
                       "reflection": {"provider": "fake"}},
        })


if __name__ == "__main__":
    unittest.main()
