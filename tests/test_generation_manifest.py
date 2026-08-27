from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from generate import select_evaluation_indices, write_generation_manifest


class _Dataset:
    sample_indices = np.asarray([10, 11, 12, 13, 14])

    def __len__(self):
        return len(self.sample_indices)


class GenerationManifestTest(unittest.TestCase):
    def test_selection_is_repeatable_and_controls_are_disjoint(self):
        first, first_control = select_evaluation_indices(20, 5, 42, 5)
        second, second_control = select_evaluation_indices(20, 5, 42, 5)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_control, second_control)
        self.assertFalse(set(first.tolist()) & set(first_control.tolist()))

    def test_environment_provenance_and_checkpoint_hash_are_recorded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint fixture")
            config_path = root / "config.yaml"
            config_path.write_text("project:\n  seed: 42\n", encoding="utf-8")
            metrics_path = root / "metrics.json"
            metrics_path.write_text("{}\n", encoding="utf-8")
            output = root / "generation_manifest.json"
            with (
                mock.patch("generate.subprocess.run", side_effect=FileNotFoundError),
                mock.patch("generate.torch.cuda.is_available", return_value=False),
                mock.patch.dict(
                    os.environ,
                    {
                        "TRAJFLOW_GIT_COMMIT": "abc123",
                        "TRAJFLOW_GIT_DIRTY": "false",
                    },
                    clear=False,
                ),
            ):
                write_generation_manifest(
                    output,
                    config_path=config_path,
                    checkpoint_path=checkpoint,
                    config={
                        "project": {"seed": 42, "deterministic": True},
                        "condition": {"condition_type": "full"},
                    },
                    dataset=_Dataset(),
                    selected_indices=np.asarray([0, 2]),
                    control_indices=np.asarray([1, 3]),
                    arrays={"generated": np.zeros((2, 3, 2))},
                    metrics_path=metrics_path,
                    sampling_steps=10,
                    sampling_method="em",
                    condition_mode="real",
                )

            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(manifest["git_commit"], "abc123")
            self.assertFalse(manifest["git_dirty"])
            self.assertEqual(manifest["git_provenance_source"], "environment")
            self.assertEqual(
                manifest["checkpoint_sha256"],
                hashlib.sha256(b"checkpoint fixture").hexdigest(),
            )
            self.assertEqual(manifest["selected_source_sample_indices"], [10, 12])
            self.assertEqual(manifest["real_control_source_sample_indices"], [11, 13])
            self.assertEqual(manifest["array_shapes"]["generated"], [2, 3, 2])


if __name__ == "__main__":
    unittest.main()
