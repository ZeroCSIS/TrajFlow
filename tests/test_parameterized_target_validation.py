from __future__ import annotations

import unittest

import numpy as np

from src.data.validation import summarize_parameterized_targets


class ParameterizedTargetValidationTest(unittest.TestCase):
    def test_summary_records_effective_control_points(self):
        trajectories = np.asarray(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]],
            ]
        )
        summary = summarize_parameterized_targets(
            trajectories,
            configured_control_points=3,
            source_has_movement=True,
        )
        self.assertEqual(summary["distinct_control_point_histogram"], {"1": 1, "3": 1})
        self.assertEqual(summary["repeated_single_point_trajectory_count"], 1)

    def test_moving_dataset_cannot_collapse_to_repeated_points(self):
        trajectories = np.ones((3, 4, 2), dtype=np.float64)
        with self.assertRaisesRegex(RuntimeError, "collapsed"):
            summarize_parameterized_targets(
                trajectories,
                configured_control_points=4,
                source_has_movement=True,
            )


if __name__ == "__main__":
    unittest.main()
