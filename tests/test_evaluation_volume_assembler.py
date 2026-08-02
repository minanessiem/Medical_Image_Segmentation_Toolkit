"""
Tests for volume assembly from slice streams.
"""

import unittest

import torch

from scripts.evaluation.core.contracts import SliceSample
from scripts.evaluation.io.volume_assembler import VolumeAssembler


class TestVolumeAssembler(unittest.TestCase):
    def _sample(
        self,
        *,
        case_id: str,
        volume_id: str,
        slice_index: int,
        pred_value: float,
        gt_value: float,
    ) -> SliceSample:
        return SliceSample(
            case_id=case_id,
            slice_id=f"{volume_id}_slice{slice_index}",
            volume_id=volume_id,
            slice_index=slice_index,
            prediction_prob=torch.full((1, 2, 2), fill_value=pred_value, dtype=torch.float32),
            ground_truth_mask=torch.full((1, 2, 2), fill_value=gt_value, dtype=torch.float32),
        )

    def test_finalize_volume_rejects_untyped_2d_reconstruction(self):
        assembler = VolumeAssembler()
        # Add out of order intentionally: 2, 0, 1
        assembler.add_sample(
            "n3_mean",
            self._sample(
                case_id="sub-stroke0001", volume_id="sub-stroke0001", slice_index=2, pred_value=2.0, gt_value=20.0
            ),
        )
        assembler.add_sample(
            "n3_mean",
            self._sample(
                case_id="sub-stroke0001", volume_id="sub-stroke0001", slice_index=0, pred_value=0.0, gt_value=0.0
            ),
        )
        assembler.add_sample(
            "n3_mean",
            self._sample(
                case_id="sub-stroke0001", volume_id="sub-stroke0001", slice_index=1, pred_value=1.0, gt_value=10.0
            ),
        )

        with self.assertRaisesRegex(ValueError, "deferred 2D reconstruction contract"):
            assembler.finalize_volume("n3_mean", "sub-stroke0001")

    def test_each_analysis_case_rejects_reconstruction_without_geometry(self):
        assembler = VolumeAssembler()
        assembler.add_sample(
            "n1_single",
            self._sample(
                case_id="sub-stroke0002", volume_id="sub-stroke0002", slice_index=0, pred_value=1.0, gt_value=1.0
            ),
        )
        assembler.add_sample(
            "n5_soft_staple",
            self._sample(
                case_id="sub-stroke0002", volume_id="sub-stroke0002", slice_index=0, pred_value=5.0, gt_value=1.0
            ),
        )
        with self.assertRaisesRegex(ValueError, "will not manufacture an affine"):
            assembler.finalize_volume("n1_single", "sub-stroke0002")
        with self.assertRaisesRegex(ValueError, "will not manufacture an affine"):
            assembler.finalize_volume("n5_soft_staple", "sub-stroke0002")

    def test_duplicate_slice_index_raises(self):
        assembler = VolumeAssembler()
        first = self._sample(
            case_id="sub-stroke0003", volume_id="sub-stroke0003", slice_index=4, pred_value=0.0, gt_value=0.0
        )
        second = self._sample(
            case_id="sub-stroke0003", volume_id="sub-stroke0003", slice_index=4, pred_value=1.0, gt_value=1.0
        )
        assembler.add_sample("n1_single", first)
        with self.assertRaises(ValueError):
            assembler.add_sample("n1_single", second)

    def test_finalize_all_rejects_2d_volume_claim(self):
        assembler = VolumeAssembler()
        assembler.add_sample(
            "n1_single",
            self._sample(
                case_id="sub-stroke0010", volume_id="sub-stroke0010", slice_index=0, pred_value=0.0, gt_value=0.0
            ),
        )
        assembler.add_sample(
            "n1_single",
            self._sample(
                case_id="sub-stroke0011", volume_id="sub-stroke0011", slice_index=0, pred_value=1.0, gt_value=1.0
            ),
        )
        self.assertEqual(assembler.buffer_size(), 2)
        with self.assertRaisesRegex(ValueError, "geometry-aware evaluation"):
            assembler.finalize_all()
        self.assertEqual(assembler.buffer_size(), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
