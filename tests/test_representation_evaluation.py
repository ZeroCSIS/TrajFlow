from __future__ import annotations

import unittest

import numpy as np

from data_utils.evaluate_representation import build_representation_report


class RepresentationEvaluationTest(unittest.TestCase):
    def test_report_compares_representation_with_straight_line(self):
        raw = np.asarray(
            [
                [[1.0, 1.0], [1.0, 3.0], [3.0, 3.0]],
                [[4.0, 4.0], [4.0, 6.0], [6.0, 6.0]],
            ]
        )
        report = build_representation_report(
            raw,
            raw,
            grid_metadata={"width": 10, "height": 10, "cell_size_m": 500},
            max_pairs=2,
            density_bins=5,
        )
        representation = report[
            "parameterized_representation_vs_paired_raw_test"
        ]
        self.assertAlmostEqual(
            representation["continuous_frechet_median_km"],
            0.0,
        )
        self.assertTrue(
            report["representation_ceiling_comparison"][
                "representation_beats_straight_line_on_continuous_frechet_median"
            ]
        )


if __name__ == "__main__":
    unittest.main()
