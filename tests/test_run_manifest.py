from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from train import write_run_manifest


class _Dataset:
    split_indices_path = None

    def __len__(self):
        return 3


class RunManifestTest(unittest.TestCase):
    def test_missing_git_uses_explicit_commit_without_marking_tree_clean(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "manifest.json"
            with (
                mock.patch("train.subprocess.run", side_effect=FileNotFoundError),
                mock.patch.dict(
                    os.environ,
                    {"TRAJFLOW_GIT_COMMIT": "abc123"},
                    clear=False,
                ),
            ):
                write_run_manifest(
                    str(output),
                    "config.yaml",
                    {"project": {"seed": 42, "deterministic": True}},
                    torch.device("cpu"),
                    _Dataset(),
                    None,
                )

            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(manifest["git_commit"], "abc123")
            self.assertIsNone(manifest["git_dirty"])


if __name__ == "__main__":
    unittest.main()
