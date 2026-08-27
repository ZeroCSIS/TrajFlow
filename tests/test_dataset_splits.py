from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_utils.prepare_yjmob import assign_user_split, prepare_dataset
from src.data.dataset import FlowMatchingDataset


class DatasetSplitTest(unittest.TestCase):
    def test_loader_uses_user_disjoint_split_and_fixed_location_vocabulary(self):
        users_by_split = {}
        for uid in range(10_000):
            users_by_split.setdefault(assign_user_split(uid, 42, 0.8, 0.1), uid)
            if len(users_by_split) == 3:
                break
        users = sorted(users_by_split.values())

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "dataset.csv.gz"
            output = root / "processed"
            with gzip.open(source, "wt", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("uid", "d", "t", "x", "y"))
                for uid in users:
                    for day in (0, 1):
                        writer.writerow((uid, day, 1, 1 + uid % 10, 2))
                        writer.writerow((uid, day, 2, 2 + uid % 10, 3))
            prepare_dataset(
                source,
                output,
                max_users=len(users),
                trajectory_length=4,
                expected_md5=None,
            )
            config = {
                "project": {"output_dir": str(root / "outputs")},
                "data": {
                    "trajectory_length": 4,
                    "sample_count": -1,
                    "dataset_type": "open_source",
                    "dataset_folder": str(output),
                    "region": "Fixture",
                    "split_file": "split_indices.npz",
                    "parametrized": False,
                    "norm1by1": False,
                    "od_finer": False,
                    "geohash": False,
                },
                "condition": {"enabled": True, "condition_type": "full"},
            }
            datasets = {
                split: FlowMatchingDataset(config, mode=split)
                for split in ("train", "val", "test")
            }
            self.assertTrue(all(dataset.location_dim == 40_000 for dataset in datasets.values()))
            selected = [set(dataset.sample_indices.tolist()) for dataset in datasets.values()]
            self.assertFalse(selected[0] & selected[1])
            self.assertFalse(selected[0] & selected[2])
            self.assertFalse(selected[1] & selected[2])
            self.assertEqual(set.union(*selected), set(range(len(users) * 2)))
            for dataset in datasets.values():
                np.testing.assert_array_equal(
                    dataset.conditions[:, 6:8].numpy(), dataset.all_head[:, 6:8]
                )


if __name__ == "__main__":
    unittest.main()
