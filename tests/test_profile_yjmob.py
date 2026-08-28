from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_utils.profile_yjmob import build_dataset_profile, five_distinct_point_pattern


class YJMobProfileTest(unittest.TestCase):
    def test_profile_reports_manifest_quantiles_and_distinct_rdp_points(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (root / "manifest.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "sample_index",
                        "uid",
                        "day",
                        "split",
                        "observation_count",
                        "first_timeslot",
                        "last_timeslot",
                    ),
                )
                writer.writeheader()
                writer.writerow({
                    "sample_index": 0,
                    "uid": 1,
                    "day": 0,
                    "split": "test",
                    "observation_count": 2,
                    "first_timeslot": 1,
                    "last_timeslot": 5,
                })
                writer.writerow({
                    "sample_index": 1,
                    "uid": 2,
                    "day": 0,
                    "split": "test",
                    "observation_count": 4,
                    "first_timeslot": 2,
                    "last_timeslot": 8,
                })
            coefficients = np.asarray([
                [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            ])
            np.save(root / "processed_coeffs_Fixture_rdp_k_3_test.npy", coefficients)

            profile = build_dataset_profile(
                root,
                region="Fixture",
                parameterization="rdp_k",
                control_points=3,
            )
            self.assertEqual(
                profile["manifest_statistics"]["test"][
                    "two_observation_sample_rate"
                ],
                0.5,
            )
            self.assertEqual(
                profile["manifest_statistics"]["test"]["observation_count"]["p50"],
                3.0,
            )
            self.assertEqual(
                profile["rdp_control_point_statistics"]["test"][
                    "distinct_control_point_histogram"
                ],
                {"2": 1, "3": 1},
            )
            self.assertEqual(
                profile["rdp_control_point_statistics"]["test"][
                    "five_distinct_point_pattern"
                ]["trajectory_count"],
                0,
            )

    def test_five_point_spike_identifies_out_and_back_pattern(self):
        outward = np.column_stack([np.arange(5, dtype=float), np.zeros(5)])
        trajectory = np.concatenate([outward, outward[::-1]], axis=0)
        summary = five_distinct_point_pattern(
            trajectory[None, ...],
            np.asarray([5]),
        )
        self.assertEqual(summary["trajectory_count"], 1)
        self.assertEqual(summary["same_start_end_rate"], 1.0)
        self.assertEqual(summary["palindromic_control_points_rate"], 1.0)
        self.assertEqual(summary["nearly_collinear_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
