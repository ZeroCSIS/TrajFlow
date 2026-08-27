from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from src.training.trainer import FlowMatchingTrainer
from src.utils.reproducibility import seed_everything


class TinyVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, x, t, c=None):
        del t, c
        return self.scale * x


class TrainerValidationTest(unittest.TestCase):
    def test_validation_draws_are_repeatable_and_select_checkpoint(self):
        config = {
            "project": {"seed": 7},
            "data": {
                "parametrized": False,
                "trajectory_length": 2,
                "batch_size": 2,
                "num_workers": 0,
                "od_finer": False,
                "dataset_type": "open_source",
            },
            "training": {
                "learning_rate": 1e-3,
                "num_epochs": 1,
                "print_every": 1,
                "save_every": 1,
                "viz_bool": False,
                "validation_seed": 123,
                "early_stop_patience": 2,
                "early_stop_delta": 0.0,
            },
            "condition": {"enabled": False},
            "flow_matching": {"enabled": True, "flow_type": "standard"},
            "ddpm": {"enabled": False},
            "baseline": {"enabled": False},
        }
        values = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2) / 10
        train_dataset = TensorDataset(values[:4])
        validation_dataset = TensorDataset(values[4:])
        # TensorDataset returns a one-element tuple, while the production
        # unconditional dataset returns a tensor. Keep the fixture interface exact.
        train_dataset = _TensorOnlyDataset(train_dataset)
        validation_dataset = _TensorOnlyDataset(validation_dataset)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            save_dir = root / "run"
            seed_everything(7)
            trainer = FlowMatchingTrainer(
                config,
                TinyVelocity(),
                train_dataset,
                str(save_dir),
                torch.device("cpu"),
                validation_dataset=validation_dataset,
            )
            first = trainer.validation_loss()
            second = trainer.validation_loss()
            self.assertAlmostEqual(first, second, places=10)
            losses = trainer.train()
            self.assertEqual(len(losses), 1)
            history = json.loads(
                (save_dir / "loss_history.json").read_text(encoding="utf-8")
            )
            self.assertEqual(history["selection_metric"], "validation_loss")
            best_model = root / "models" / "run" / "best_model.pt"
            self.assertTrue(best_model.exists())


class _TensorOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][0]


if __name__ == "__main__":
    unittest.main()
