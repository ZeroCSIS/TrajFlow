import copy
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class YJMobConfigTests(unittest.TestCase):
    def _load(self, name):
        with (REPO_ROOT / "src" / "config" / name).open(encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    def test_convergence_config_only_extends_training_horizon(self):
        smoke = self._load("config_yjmob_smoke.yaml")
        convergence = self._load("config_yjmob_convergence.yaml")

        self.assertEqual(convergence["training"]["num_epochs"], 100)
        self.assertEqual(convergence["training"]["early_stop_patience"], 10)
        self.assertEqual(
            convergence["training"]["checkpoint_policy"],
            "best_and_last",
        )

        comparable_smoke = copy.deepcopy(smoke)
        comparable_convergence = copy.deepcopy(convergence)
        for config in (comparable_smoke, comparable_convergence):
            config["project"].pop("name")
            config["project"].pop("output_dir")
            config["training"].pop("num_epochs")

        self.assertEqual(comparable_convergence, comparable_smoke)


if __name__ == "__main__":
    unittest.main()
