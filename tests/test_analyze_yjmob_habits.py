from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_utils.analyze_yjmob_habits import (
    build_habit_profile,
    load_observation_cube,
    missingness_profile,
    read_manifest_users,
    shuffle_timeslots_within_user_days,
    temporal_holdout_metrics,
    user_slot_stability,
)


class YJMobHabitAnalysisTest(unittest.TestCase):
    def stable_fixture(self) -> tuple[np.ndarray, np.ndarray]:
        cube = np.full((2, 8, 4), -1, dtype=np.int32)
        for day in range(8):
            cube[0, day, 0] = 1
            cube[0, day, 1] = 10 + day
            cube[1, day, 0] = 2
            cube[1, day, 1] = 20 + day
        return cube, np.asarray([100, 200])

    def test_user_slot_top1_and_holdout_detect_personal_modes(self):
        cube, users = self.stable_fixture()
        all_days = np.ones(8, dtype=bool)
        summary, rows = user_slot_stability(
            cube,
            users,
            all_days,
            min_observed_days=4,
        )
        user_zero_slot_zero = next(
            row for row in rows if row["uid"] == 100 and row["timeslot"] == 0
        )
        self.assertEqual(user_zero_slot_zero["top_cell"], 1)
        self.assertEqual(user_zero_slot_zero["top1_location_rate"], 1.0)
        self.assertGreater(summary["weighted_top1_location_rate"], 0.5)

        train = np.asarray([True] * 4 + [False] * 4)
        test = ~train
        holdout = temporal_holdout_metrics(
            cube,
            train,
            test,
            min_train_observations=3,
            grid_height=200,
        )
        self.assertGreater(
            holdout["personalized_top1_exact_hit_rate"],
            holdout["population_timeslot_top1_exact_hit_rate"],
        )
        self.assertGreater(
            holdout["personalized_minus_population_exact_hit_rate"],
            0.0,
        )

    def test_shuffle_preserves_masks_and_daily_location_multisets(self):
        cube, _ = self.stable_fixture()
        shuffled = shuffle_timeslots_within_user_days(cube, 42)
        np.testing.assert_array_equal(cube >= 0, shuffled >= 0)
        for user_index, day_index in np.ndindex(cube.shape[:2]):
            np.testing.assert_array_equal(
                np.sort(cube[user_index, day_index][cube[user_index, day_index] >= 0]),
                np.sort(
                    shuffled[user_index, day_index][
                        shuffled[user_index, day_index] >= 0
                    ]
                ),
            )

    def test_missingness_counts_only_raw_internal_gaps(self):
        cube = np.full((1, 2, 6), -1, dtype=np.int32)
        cube[0, 0, [0, 2, 5]] = [1, 2, 3]
        cube[0, 1, 3] = 4
        profile = missingness_profile(cube, np.asarray([True, True]))
        self.assertEqual(profile["converter_eligible_user_day_count"], 1)
        self.assertEqual(profile["one_observation_user_day_count"], 1)
        self.assertEqual(
            profile["max_internal_gap_slots_per_eligible_user_day"]["max"],
            2.0,
        )
        self.assertEqual(
            profile["positive_internal_gap_slots"]["count"],
            2,
        )

    def test_build_profile_cross_checks_manifest_without_calendar_claim(self):
        cube, users = self.stable_fixture()
        profile, rows = build_habit_profile(
            cube,
            users,
            manifest_sample_count=16,
            excluded_days=set(),
            min_observed_days=4,
            min_train_observations=3,
            null_repeats=3,
            seed=42,
        )
        self.assertTrue(profile["conversion_cross_check"]["counts_match"])
        self.assertFalse(
            profile["weekly_phase_analysis"]["calendar_mapping_available"]
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            profile["chronological_holdout_and_shuffled_null"][
                "null_repeat_count"
            ],
            3,
        )

    def test_raw_loader_uses_exact_manifest_cohort(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("sample_index", "uid", "day"),
                )
                writer.writeheader()
                writer.writerow({"sample_index": 0, "uid": 10, "day": 0})
                writer.writerow({"sample_index": 1, "uid": 20, "day": 0})
            raw = root / "raw.csv.gz"
            with gzip.open(raw, "wt", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("uid", "d", "t", "x", "y"))
                writer.writerow((10, 0, 0, 1, 1))
                writer.writerow((10, 0, 2, 2, 1))
                writer.writerow((20, 0, 1, 3, 1))
                writer.writerow((20, 0, 3, 4, 1))
                writer.writerow((30, 0, 0, 5, 1))
            users, sample_count = read_manifest_users(manifest)
            cube, scan = load_observation_cube(
                raw,
                users,
                day_count=2,
                timeslot_count=4,
            )
            self.assertEqual(sample_count, 2)
            self.assertEqual(scan["rows_retained"], 4)
            self.assertEqual(cube[0, 0, 0], 0)
            self.assertEqual(cube[1, 0, 3], 600)


if __name__ == "__main__":
    unittest.main()
