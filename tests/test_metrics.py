from __future__ import annotations

import math
import unittest

import numpy as np

from src.eval.metrics import (
    compute_baseline_metrics,
    continuous_frechet,
    density_jensen_shannon,
    dynamic_time_warping,
)


class BaselineMetricsTest(unittest.TestCase):
    def test_identical_curves_have_zero_distance(self):
        curve = np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 2.0]])
        self.assertAlmostEqual(dynamic_time_warping(curve, curve), 0.0)
        self.assertAlmostEqual(continuous_frechet(curve, curve), 0.0, places=7)

    def test_parallel_lines_have_unit_continuous_frechet_distance(self):
        curve_a = np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        curve_b = np.asarray([[0.0, 1.0], [1.0, 1.0]])
        self.assertAlmostEqual(continuous_frechet(curve_a, curve_b), 1.0, places=6)

    def test_disjoint_density_has_log_two_js_divergence(self):
        generated = np.asarray([[[1.0, 1.0], [1.0, 1.0]]])
        reference = np.asarray([[[2.0, 2.0], [2.0, 2.0]]])
        self.assertAlmostEqual(
            density_jensen_shannon(generated, reference, width=2, height=2),
            math.log(2.0),
        )

    def test_metric_bundle_uses_grid_scale(self):
        curves = np.asarray([[[1.0, 1.0], [2.0, 2.0], [3.0, 2.0]]])
        metrics = compute_baseline_metrics(
            curves,
            curves,
            grid_metadata={"width": 200, "height": 200, "cell_size_m": 500},
            max_pairs=1,
        )
        self.assertAlmostEqual(metrics["density_js_divergence"], 0.0)
        self.assertAlmostEqual(metrics["dtw_mean_km"], 0.0)
        self.assertAlmostEqual(metrics["continuous_frechet_mean_km"], 0.0)
        self.assertEqual(metrics["paired_curve_count"], 1)


if __name__ == "__main__":
    unittest.main()
