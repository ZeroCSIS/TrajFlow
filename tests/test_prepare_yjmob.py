from __future__ import annotations

import csv
import gzip
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_utils.prepare_yjmob import (
    Sample,
    assign_user_split,
    prepare_dataset,
    raw_condition,
)


def users_for_all_splits(seed: int = 42) -> list[int]:
    selected: dict[str, int] = {}
    for uid in range(10_000):
        split = assign_user_split(uid, seed, 0.8, 0.1)
        selected.setdefault(split, uid)
        if len(selected) == 3:
            return sorted(selected.values())
    raise AssertionError("could not find fixture users for every split")


class PrepareYJMobTest(unittest.TestCase):
    def write_fixture(self, path: Path, users: list[int]) -> None:
        with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("uid", "d", "t", "x", "y"))
            for uid in users:
                offset = uid % 7
                for day in (0, 1, 27):
                    writer.writerow((uid, day, 1, 10 + offset, 20))
                    writer.writerow((uid, day, 3, 12 + offset, 22 + day % 2))
                    writer.writerow((uid, day, 8, 14 + offset, 24 + day % 2))

    def test_conversion_excludes_day_and_keeps_identity_only_in_manifest(self):
        users = users_for_all_splits()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "dataset1.csv.gz"
            output = root / "processed"
            self.write_fixture(source, users)

            summary = prepare_dataset(
                source,
                output,
                max_users=len(users),
                trajectory_length=5,
                expected_md5=None,
            )

            self.assertEqual(summary["sample_count"], len(users) * 2)
            self.assertEqual(summary["excluded_days"], [27])
            self.assertEqual(
                summary["split_user_overlap_counts"],
                {"train_val": 0, "train_test": 0, "val_test": 0},
            )
            with (output / "manifest.csv").open(encoding="utf-8", newline="") as stream:
                manifest = list(csv.DictReader(stream))
            self.assertEqual({int(row["day"]) for row in manifest}, {0, 1})
            self.assertEqual({int(row["uid"]) for row in manifest}, set(users))
            self.assertEqual({row["split"] for row in manifest}, {"train", "val", "test"})

            with (output / "traj_segments.pkl").open("rb") as stream:
                trajectories = pickle.load(stream)
            with (output / "conditions.pkl").open("rb") as stream:
                conditions = pickle.load(stream)
            self.assertEqual(trajectories.shape, (len(users) * 2, 5, 2))
            self.assertEqual(conditions.shape, (len(users) * 2, 9))
            np.testing.assert_allclose(trajectories[0, 0], [10 + users[0] % 7, 20])
            np.testing.assert_allclose(trajectories[0, -1], [14 + users[0] % 7, 24])
            self.assertTrue(np.all((conditions[:, 6:8] >= 0) & (conditions[:, 6:8] < 40_000)))

            with np.load(output / "split_indices.npz") as splits:
                covered = np.sort(np.concatenate([splits[name] for name in splits.files]))
                np.testing.assert_array_equal(covered, np.arange(len(manifest)))
                train_conditions = conditions[splits["train"], 1:6]
            np.testing.assert_allclose(train_conditions.mean(axis=0), 0.0, atol=1e-6)

            metadata = json.loads((output / "grid_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["encoding"], "cartesian_grid")
            self.assertEqual(metadata["fixed_location_dim"], 40_000)
            self.assertNotIn("uid", metadata)
            self.assertNotIn("day", metadata)

    def test_user_hash_split_is_stable_and_user_disjoint(self):
        first = [assign_user_split(uid, 42, 0.8, 0.1) for uid in range(100)]
        second = [assign_user_split(uid, 42, 0.8, 0.1) for uid in range(100)]
        changed = [assign_user_split(uid, 43, 0.8, 0.1) for uid in range(100)]
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_average_distance_matches_existing_trajflow_condition(self):
        sample = Sample(
            uid=1,
            day=0,
            split="train",
            timeslots=(0, 1, 2),
            points=np.asarray([[1, 1], [4, 1], [4, 5]], dtype=np.float32),
        )
        condition = raw_condition(sample, cell_size_m=1.0)
        self.assertAlmostEqual(condition[1], 7.0)
        self.assertAlmostEqual(condition[4], 7.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
