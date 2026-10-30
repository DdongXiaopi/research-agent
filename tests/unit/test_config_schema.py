import copy
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import ConfigError, load_config, validate_config  # noqa: E402


class ConfigSchemaTests(unittest.TestCase):
    def test_pi_retry_zero_is_valid_and_provider_caps_are_explicit(self):
        cfg=load_config("configs/experiment.yaml")
        cfg["runtime"].update(pi_provider_max_retries=0,pi_timeout_max_attempts=1)
        validate_config(cfg)
        for key,value in (("pi_provider_max_retries",-1),("pi_provider_max_retries",5),
                          ("pi_timeout_max_attempts",0),("pi_timeout_max_attempts",4)):
            with self.subTest(key=key,value=value):
                changed=copy.deepcopy(cfg);changed["runtime"][key]=value
                with self.assertRaises(ConfigError): validate_config(changed)

    def test_all_checked_in_experiment_configs_match_the_schema(self):
        files = sorted((ROOT / "configs").glob("experiment*.yaml"))
        self.assertGreater(len(files), 0)
        for file in files:
            with self.subTest(config=file.name):
                load_config(file.as_posix())

    def test_unknown_key_is_rejected_instead_of_silently_ignored(self):
        cfg = load_config("configs/experiment.yaml")
        cfg["evoluton"] = cfg.pop("evolution")
        with self.assertRaisesRegex(ConfigError, "unknown key.*evoluton"):
            validate_config(cfg)

    def test_runtime_provider_and_task_limits_are_checked(self):
        cfg = copy.deepcopy(load_config("configs/experiment.yaml"))
        cfg["runtime"]["provider"] = "htttp"
        with self.assertRaisesRegex(ConfigError, "runtime.provider"):
            validate_config(cfg)

        cfg = copy.deepcopy(load_config("configs/experiment.yaml"))
        cfg["runtime"]["max_model_calls"] = 0
        with self.assertRaisesRegex(ConfigError, "max_model_calls"):
            validate_config(cfg)

        cfg = copy.deepcopy(load_config("configs/experiment.yaml"))
        cfg["tools"]["active"].append("powershell")
        with self.assertRaisesRegex(ConfigError, "unsupported tools.*powershell"):
            validate_config(cfg)

    def test_boolean_is_not_accepted_as_integer_and_ranges_are_enforced(self):
        cfg = copy.deepcopy(load_config("configs/experiment.yaml"))
        cfg["retrieval"]["top_k"] = True
        with self.assertRaisesRegex(ConfigError, "top_k"):
            validate_config(cfg)

        cfg = copy.deepcopy(load_config("configs/experiment.yaml"))
        cfg["models"]["task"]["temperature"] = 2.1
        with self.assertRaisesRegex(ConfigError, "temperature"):
            validate_config(cfg)

    def test_nested_static_gate_typo_is_rejected(self):
        cfg = copy.deepcopy(load_config("configs/experiment.yaml"))
        cfg["gates"]["static"]["max_promt_chars"] = 8000
        with self.assertRaisesRegex(ConfigError, "max_promt_chars"):
            validate_config(cfg)

    def test_structured_output_mode_is_typed_and_closed(self):
        cfg = copy.deepcopy(load_config("configs/experiment.yaml"))
        cfg["models"]["task"]["structured_output"] = "json_schema"
        validate_config(cfg)
        cfg["models"]["task"]["structured_output"] = "json"
        with self.assertRaisesRegex(ConfigError, "structured_output"):
            validate_config(cfg)

    def test_semantic_citation_prompt_version_is_closed_and_versioned(self):
        cfg = copy.deepcopy(load_config("configs/experiment.yaml"))
        cfg["evaluation"]["semantic_citation"] = {
            "enabled": True, "mode": "proposition", "prompt_version": "v6"}
        validate_config(cfg)
        cfg["evaluation"]["semantic_citation"]["prompt_version"] = "v99"
        with self.assertRaisesRegex(ConfigError, "prompt_version"):
            validate_config(cfg)

        cfg["evaluation"]["semantic_citation"]["prompt_version"] = "v10"
        validate_config(cfg)
        cfg["evaluation"]["semantic_citation"]["prompt_version"] = "v14"
        validate_config(cfg)
        cfg["evaluation"]["semantic_citation"]["prompt_version"] = "v16"
        validate_config(cfg)

    def test_load_config_validates_after_placeholder_resolution(self):
        cfg = load_config("configs/experiment.http.pubmedqa-pilot.yaml")
        cfg["models"]["task"]["base_url"] = "${MISSING_TEST_KEY}"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.yaml"
            path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "unresolved config placeholder"):
                load_config(path.as_posix())


if __name__ == "__main__":
    unittest.main()
