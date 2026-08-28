from __future__ import annotations

import math
import unittest

import numpy as np

from src.eval.metrics import (
    compute_baseline_metrics,
    compute_best_of_k_metrics,
    compute_control_metrics,
    continuous_frechet,
    density_jensen_shannon,
    dynamic_time_warping,
    origin_destination_straight_lines,
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

    def test_density_can_use_coarser_bins_without_changing_domain(self):
        generated = np.asarray([[[1.0, 1.0], [1.0, 1.0]]])
        reference = np.asarray([[[2.0, 2.0], [2.0, 2.0]]])
        self.assertAlmostEqual(
            density_jensen_shannon(
                generated,
                reference,
                width=4,
                height=4,
                bins=1,
            ),
            0.0,
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

    def test_out_of_bounds_is_reported_by_point_and_trajectory(self):
        reference = np.asarray([
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ])
        generated = reference.copy()
        generated[1, 1] = [999.0, 999.0]
        metrics = compute_baseline_metrics(
            generated,
            reference,
            grid_metadata={"width": 200, "height": 200},
            max_pairs=2,
        )
        self.assertEqual(metrics["generated_out_of_bounds_point_count"], 1)
        self.assertAlmostEqual(metrics["generated_out_of_bounds_rate"], 1.0 / 6.0)
        self.assertEqual(metrics["generated_out_of_bounds_trajectory_count"], 1)
        self.assertAlmostEqual(
            metrics["generated_out_of_bounds_trajectory_rate"], 0.5
        )
        histogram_points = metrics["density_histogram_points"]
        self.assertEqual(histogram_points["generated_total"], 6)
        self.assertEqual(histogram_points["generated_in_bounds"], 5)
        self.assertEqual(
            histogram_points["generated_excluded_out_of_bounds"], 1
        )

    def test_control_bundle_compares_model_with_condition_only_line(self):
        reference = np.asarray([
            [[1.0, 1.0], [1.0, 3.0], [3.0, 3.0]],
            [[5.0, 5.0], [5.0, 7.0], [7.0, 7.0]],
        ])
        real_control = np.asarray([
            [[2.0, 2.0], [2.0, 4.0], [4.0, 4.0]],
            [[6.0, 6.0], [6.0, 8.0], [8.0, 8.0]],
        ])
        straight = origin_destination_straight_lines(reference)
        np.testing.assert_allclose(straight[:, 0], reference[:, 0])
        np.testing.assert_allclose(straight[:, -1], reference[:, -1])

        metrics = compute_control_metrics(
            reference,
            reference,
            real_control,
            parameterized_reference=reference,
            grid_metadata={"width": 10, "height": 10, "cell_size_m": 500},
            max_pairs=2,
            density_bins=5,
        )
        self.assertEqual(
            metrics["array_shapes"]["od_straight_line_control"], [2, 3, 2]
        )
        self.assertTrue(
            metrics["model_improvement_over_straight_line"][
                "model_beats_straight_line_on_dtw_median"
            ]
        )
        self.assertGreater(
            metrics["od_straight_line_vs_paired_raw_test"]["dtw_median_km"],
            0.0,
        )
        self.assertEqual(metrics["metric_semantics"]["density_grid_bins"], [5, 5])
        self.assertTrue(metrics["baseline_acceptance_gate"]["passed"])
        self.assertAlmostEqual(
            metrics["parameterized_representation_vs_paired_raw_test"][
                "continuous_frechet_median_km"
            ],
            0.0,
        )
        self.assertGreater(
            metrics["representation_ceiling_comparison"][
                "continuous_frechet_median_km_straight_minus_representation"
            ],
            0.0,
        )

    def test_best_of_k_reports_oracle_diversity_and_oob_separately(self):
        reference = np.asarray([
            [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]],
            [[4.0, 4.0], [4.0, 5.0], [5.0, 5.0]],
        ])
        generated = np.asarray([
            [
                reference[0],
                reference[0] + np.asarray([0.0, 1.0]),
            ],
            [
                reference[1] + np.asarray([1.0, 0.0]),
                reference[1],
            ],
        ])
        generated[1, 0, 1] = [99.0, 99.0]

        metrics = compute_best_of_k_metrics(
            generated,
            reference,
            grid_metadata={"width": 10, "height": 10, "cell_size_m": 500},
            density_bins=5,
            workers=1,
        )

        self.assertTrue(metrics["diagnostic_only"])
        self.assertEqual(metrics["condition_count"], 2)
        self.assertEqual(metrics["samples_per_condition"], 2)
        self.assertEqual(
            metrics["best_of_k_vs_paired_raw_test"]["dtw_km"]["p50"],
            0.0,
        )
        self.assertEqual(
            metrics["best_of_k_vs_paired_raw_test"][
                "continuous_frechet_km"
            ]["p50"],
            0.0,
        )
        self.assertEqual(
            metrics["candidate_diversity"]["unordered_pairs_per_condition"],
            1,
        )
        self.assertEqual(
            metrics["out_of_bounds"]["pooled_candidates"][
                "out_of_bounds_trajectory_count"
            ],
            1,
        )
        self.assertEqual(len(metrics["candidate_metrics"]), 4)
        self.assertEqual(
            metrics["per_condition"][0]["best_dtw_sample_index"],
            0,
        )
        self.assertEqual(
            metrics["per_condition"][1][
                "best_continuous_frechet_sample_index"
            ],
            1,
        )

    def test_best_of_k_accepts_condition_major_flattened_candidates(self):
        reference = np.asarray([
            [[1.0, 1.0], [2.0, 2.0]],
            [[3.0, 3.0], [4.0, 4.0]],
        ])
        generated = np.repeat(reference[:, None, :, :], 2, axis=1)
        metrics = compute_best_of_k_metrics(
            generated.reshape(4, 2, 2),
            reference,
            grid_metadata={"width": 10, "height": 10},
        )
        self.assertEqual(
            metrics["array_shapes"]["generated_candidates"],
            [2, 2, 2, 2],
        )
        self.assertEqual(
            metrics["candidate_diversity"]["all_pair_distances_km"]["max"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
