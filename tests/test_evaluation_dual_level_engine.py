"""
Tests for dual-level streaming metrics engine.
"""

import unittest

import torch

from scripts.evaluation.core.contracts import SliceSample
from scripts.evaluation.metrics.engine import DualLevelStreamingMetricsEngine


class TestDualLevelStreamingMetricsEngine(unittest.TestCase):
    def test_volume_boundary_rejects_2d_reconstruction_without_geometry(self):
        engine = DualLevelStreamingMetricsEngine(
            thresholds=[0.5],
            volume_metric_names=["DiceNativeCoefficient"],
        )

        # Volume A with 2 slices, then volume B with 1 slice.
        samples = [
            SliceSample(
                case_id="sub-stroke0001",
                slice_id="sub-stroke0001_slice0",
                volume_id="sub-stroke0001",
                slice_index=0,
                prediction_prob=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
                ground_truth_mask=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            ),
            SliceSample(
                case_id="sub-stroke0001",
                slice_id="sub-stroke0001_slice1",
                volume_id="sub-stroke0001",
                slice_index=1,
                prediction_prob=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
                ground_truth_mask=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
            ),
            SliceSample(
                case_id="sub-stroke0002",
                slice_id="sub-stroke0002_slice0",
                volume_id="sub-stroke0002",
                slice_index=0,
                prediction_prob=torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]),
                ground_truth_mask=torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]),
            ),
        ]

        finalized_midstream = engine.update(samples[0])
        self.assertEqual(len(finalized_midstream), 0)
        finalized_midstream = engine.update(samples[1])
        self.assertEqual(len(finalized_midstream), 0)

        # The first volume boundary is the earliest point at which the old path
        # would claim a reconstructed 3D grid.
        with self.assertRaisesRegex(ValueError, "deferred 2D reconstruction contract"):
            engine.update(samples[2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
