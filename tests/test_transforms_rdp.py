from __future__ import annotations

import unittest

import numpy as np

from src.data.transforms import point2para, point_line_distance_2d


class RDPCompatibilityTest(unittest.TestCase):
    def test_two_dimensional_distance_is_numpy_2_compatible(self):
        self.assertAlmostEqual(
            point_line_distance_2d(
                np.asarray([1.0, 1.0]),
                np.asarray([0.0, 0.0]),
                np.asarray([2.0, 0.0]),
            ),
            1.0,
        )

    def test_rdp_returns_requested_nonzero_control_points(self):
        trajectory = np.asarray([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 0.0],
        ])
        parameterized = point2para(trajectory, method="rdp_k", K=4)
        self.assertIsNotNone(parameterized)
        points = parameterized["simplified_points"]
        self.assertEqual(points.shape, (4, 2))
        self.assertTrue(np.isfinite(points).all())
        self.assertTrue(np.any(np.abs(points) > 0.0))
        np.testing.assert_allclose(points[0], trajectory[0])
        np.testing.assert_allclose(points[-1], trajectory[-1])


if __name__ == "__main__":
    unittest.main()
