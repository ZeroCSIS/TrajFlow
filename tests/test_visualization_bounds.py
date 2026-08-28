from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.utils.visualization import (
    _grid_plot_bounds,
    _out_of_bounds_trajectory_mask,
    visualize_density_comparison,
    visualize_trajectories,
)


class VisualizationBoundsTest(unittest.TestCase):
    def test_grid_bounds_and_trajectory_mask_use_metric_domain(self):
        metadata = {"width": 3, "height": 4, "coordinate_min": 1}
        self.assertEqual(_grid_plot_bounds(metadata), (0.5, 4.5, 0.5, 3.5))
        trajectories = np.asarray(
            [
                [[1.0, 1.0], [3.0, 4.0]],
                [[1.0, 1.0], [3.0, 4.6]],
            ]
        )
        np.testing.assert_array_equal(
            _out_of_bounds_trajectory_mask(trajectories, metadata),
            [False, True],
        )

    def test_domain_plots_keep_outliers_in_a_separate_figure(self):
        metadata = {"width": 3, "height": 4, "coordinate_min": 1}
        reference = np.asarray(
            [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 4.0]],
                [[1.0, 1.0], [2.0, 2.0], [3.0, 4.0]],
            ]
        )
        generated = reference.copy()
        generated[1, 1] = [20.0, -10.0]
        with TemporaryDirectory() as temp_dir:
            visualize_trajectories(
                generated,
                reference,
                3,
                False,
                temp_dir,
                grid_metadata=metadata,
            )
            visualize_density_comparison(
                generated,
                reference,
                3,
                temp_dir,
                grid_metadata=metadata,
                density_bins=2,
            )
            expected = {
                "generated_trajectories.png",
                "ground_truth_trajectories.png",
                "trajectory_comparison.png",
                "generated_outliers.png",
                "density_comparison.png",
            }
            self.assertTrue(
                expected.issubset({path.name for path in Path(temp_dir).iterdir()})
            )


if __name__ == "__main__":
    unittest.main()
