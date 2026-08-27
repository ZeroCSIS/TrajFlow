from __future__ import annotations

import unittest

import torch

from src.eval.inference import FlowMatchingInference, _cartesian_grid_point


class ConstantVelocity(torch.nn.Module):
    def forward(self, x, t, c=None):
        del t, c
        return torch.ones_like(x)


class InferenceStepTest(unittest.TestCase):
    def test_configured_step_count_reaches_one_unit_of_constant_flow(self):
        inference = FlowMatchingInference.__new__(FlowMatchingInference)
        inference.config = {"data": {"od_finer": False}}
        inference.M = 2
        inference.device = torch.device("cpu")
        inference.wrapped_model = ConstantVelocity()
        inference.conditional = False
        initial, final = inference._sample_flow_matching(2, n_steps=10, method="em")
        torch.testing.assert_close(final - initial, torch.ones_like(initial))

    def test_cartesian_cell_decode_matches_converter_formula(self):
        metadata = {"width": 200, "height": 200, "coordinate_min": 1}
        point = _cartesian_grid_point((3 - 1) * 200 + (4 - 1), 0.5, 0.5, metadata)
        self.assertEqual(point.tolist(), [3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
