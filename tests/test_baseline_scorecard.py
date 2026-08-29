from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from data_utils.build_yjmob_baseline_scorecard import (
    build_scorecard,
    render_markdown,
)


class BaselineScorecardTest(unittest.TestCase):
    def test_scorecard_freezes_k_prefix_and_same_od_counterexample(self):
        reference = np.asarray([
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
        ])
        generated = np.asarray([
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            [
                [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]],
            ],
        ])
        candidate_rows = []
        dtw = [[1.0, 1.0], [0.0, 2.0]]
        frechet = [[1.0, 1.0], [0.0, 2.0]]
        for condition_index in range(2):
            for sample_index in range(2):
                candidate_rows.append({
                    "condition_index": condition_index,
                    "sample_index": sample_index,
                    "dtw_km": dtw[condition_index][sample_index],
                    "continuous_frechet_km": frechet[condition_index][
                        sample_index
                    ],
                    "out_of_bounds": False,
                })
        metrics = {
            "condition_count": 2,
            "samples_per_condition": 2,
            "grid_cell_size_km": 0.5,
            "candidate_metrics": candidate_rows,
            "per_condition": [
                {
                    "condition_index": 0,
                    "mean_pairwise_aligned_point_distance_km": 0.0,
                },
                {
                    "condition_index": 1,
                    "mean_pairwise_aligned_point_distance_km": 0.5,
                },
            ],
            "best_of_k_vs_paired_raw_test": {
                "dtw_km": {"p50": 0.5},
                "continuous_frechet_km": {"p50": 0.5},
            },
            "od_straight_line_vs_paired_raw_test": {
                "dtw_median_km": 0.5,
                "continuous_frechet_median_km": 0.5,
            },
            "out_of_bounds": {
                "pooled_candidates": {"out_of_bounds_trajectory_count": 0}
            },
        }
        manifest = {
            "selected_test_local_indices": [4, 7],
            "selected_source_sample_indices": [40, 70],
            "samples_per_condition": 2,
            "git_commit": "abc123",
            "checkpoint_sha256": "checkpoint",
            "seed": 42,
            "sampling_steps": 10,
        }
        with patch(
            "data_utils.build_yjmob_baseline_scorecard.control_distances",
            return_value=(np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])),
        ):
            scorecard, rows = build_scorecard(
                metrics,
                manifest,
                generated,
                reference,
                np.asarray([4, 7]),
            )

        acceptance = scorecard["same_origin_destination_acceptance_set"]
        self.assertEqual(acceptance["condition_count"], 1)
        self.assertEqual(acceptance["all_candidates_constant_condition_count"], 1)
        self.assertEqual(
            acceptance["all_candidates_constant_at_origin_condition_count"],
            1,
        )
        self.assertTrue(acceptance["current_baseline_expected_failure_reproduced"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_sample_index"], 40)
        self.assertEqual(scorecard["reconciliation"]["candidate_metric_matrix_complete"], True)
        self.assertIn("O=D 结构性反例", render_markdown(scorecard))

    def test_rejects_index_drift(self):
        metrics = {
            "condition_count": 1,
            "samples_per_condition": 1,
            "grid_cell_size_km": 0.5,
            "candidate_metrics": [{
                "condition_index": 0,
                "sample_index": 0,
                "dtw_km": 0.0,
                "continuous_frechet_km": 0.0,
                "out_of_bounds": False,
            }],
            "per_condition": [
                {
                    "condition_index": 0,
                    "mean_pairwise_aligned_point_distance_km": 0.0,
                }
            ],
        }
        manifest = {
            "selected_test_local_indices": [1],
            "selected_source_sample_indices": [2],
            "samples_per_condition": 1,
        }
        arrays = np.zeros((1, 1, 2, 2), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "indices differ"):
            build_scorecard(
                metrics,
                manifest,
                arrays,
                arrays[:, 0],
                np.asarray([9]),
            )


if __name__ == "__main__":
    unittest.main()
