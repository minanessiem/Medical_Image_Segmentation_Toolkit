"""
Tests for reconstructed volume exporter.
"""

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from scripts.evaluation.core.contracts import SliceSample
from scripts.evaluation.core.contracts import VolumeSample
from scripts.evaluation.io.volume_assembler import VolumeAssembler
from scripts.evaluation.io.volume_exporter import export_reconstructed_volumes
from src.inference.contracts import SpatialGeometry


class TestEvaluationVolumeExporter(unittest.TestCase):
    def test_exports_pred_and_gt_nifti_with_expected_shape(self):
        pred = torch.zeros((1, 4, 5, 3), dtype=torch.float32)
        gt = torch.ones((1, 4, 5, 3), dtype=torch.float32)
        sample = VolumeSample(
            case_id="sub-stroke0010",
            volume_id="sub-stroke0010",
            prediction_volume=pred,
            ground_truth_volume=gt,
            prediction_space="model_preprocessed",
            reference_space="model_preprocessed",
            prediction_geometry=SpatialGeometry(
                shape=(4, 5, 3),
                affine=(
                    (1.0, 0.0, 0.0, 10.0),
                    (0.0, 1.0, 0.0, 20.0),
                    (0.0, 0.0, 2.0, 30.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                spacing=(1.0, 1.0, 2.0),
                orientation="RAS",
            ),
            reference_geometry=SpatialGeometry(
                shape=(4, 5, 3),
                affine=(
                    (1.0, 0.0, 0.0, 10.0),
                    (0.0, 1.0, 0.0, 20.0),
                    (0.0, 0.0, 2.0, 30.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                spacing=(1.0, 1.0, 2.0),
                orientation="RAS",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            written = export_reconstructed_volumes(
                grouped_volumes={"n1_single": [sample]},
                output_dir=Path(tmp),
            )
            self.assertEqual(len(written), 2)
            pred_path = Path(tmp) / "n1_single" / "sub-stroke0010__pred.nii.gz"
            gt_path = Path(tmp) / "n1_single" / "sub-stroke0010__gt.nii.gz"
            self.assertTrue(pred_path.exists())
            self.assertTrue(gt_path.exists())
            pred_img = nib.load(pred_path)
            gt_img = nib.load(gt_path)
            self.assertEqual(pred_img.shape, (4, 5, 3))
            self.assertEqual(gt_img.shape, (4, 5, 3))
            self.assertAlmostEqual(float(pred_img.affine[2, 3]), 30.0)

    def test_slice_reconstruction_is_rejected_before_export(self):
        h = 64
        w = 64
        depth = 5
        center_y = (h - 1) / 2.0
        center_x = (w - 1) / 2.0
        radii_by_slice = [4, 7, 10, 7, 4]

        yy, xx = np.ogrid[:h, :w]
        assembler = VolumeAssembler()
        volume_id = "sub-stroke0900"
        analysis_case_key = "n1_single"
        for z in range(depth):
            radius = radii_by_slice[z]
            circle = ((yy - center_y) ** 2 + (xx - center_x) ** 2) <= float(radius * radius)
            circle_t = torch.from_numpy(circle.astype(np.float32)).unsqueeze(0)  # [1,H,W]
            sample = SliceSample(
                case_id=volume_id,
                slice_id=f"{volume_id}_slice{z}",
                volume_id=volume_id,
                slice_index=z,
                prediction_mask=circle_t,
                ground_truth_mask=circle_t.clone(),
                metadata={
                    "source_affine": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "pre_resize_shape_hw": [h, w],
                },
            )
            assembler.add_sample(analysis_case_key, sample)

        with self.assertRaisesRegex(ValueError, "deferred 2D reconstruction contract"):
            assembler.finalize_volume(analysis_case_key, volume_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
