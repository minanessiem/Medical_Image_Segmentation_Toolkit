"""
Integration-style tests for evaluation entrypoints with dual-level outputs.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.evaluation.core.contracts import SliceSample
from src.metrics.metrics import DiceNativeCoefficient


class _DummyValLoader:
    class _Dataset:
        def __init__(self):
            self.return_metadata = False

    def __init__(self):
        self.dataset = self._Dataset()


class TestEvaluationEntrypointsIntegration(unittest.TestCase):
    def _build_prob_sample(self, volume_id: str, slice_index: int, value: float) -> SliceSample:
        return SliceSample(
            case_id=volume_id,
            slice_id=f"{volume_id}_slice{slice_index}",
            volume_id=volume_id,
            slice_index=slice_index,
            prediction_prob=torch.full((1, 2, 2), fill_value=value, dtype=torch.float32),
            ground_truth_mask=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float32),
        )

    def _build_mask_sample(self, volume_id: str, slice_index: int, value: float) -> SliceSample:
        return SliceSample(
            case_id=volume_id,
            slice_id=f"{volume_id}_s{slice_index:04d}",
            volume_id=volume_id,
            slice_index=slice_index,
            prediction_mask=torch.full((1, 2, 2), fill_value=value, dtype=torch.float32),
            ground_truth_mask=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float32),
        )

    def test_nnunet_2d_entrypoint_rejects_volume_reconstruction(self):
        from scripts.evaluation import compute_segmentation_metrics_for_nnunet_2d_predictions as entry

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pred_dir = base / "pred"
            gt_dir = base / "gt"
            out_dir = base / "out"
            pred_dir.mkdir(parents=True, exist_ok=True)
            gt_dir.mkdir(parents=True, exist_ok=True)
            (pred_dir / "sub-stroke0001_s0000.nii.gz").touch()
            (gt_dir / "sub-stroke0001_s0000.nii.gz").touch()

            samples = [
                self._build_mask_sample("sub-stroke0001", 0, 1.0),
                self._build_mask_sample("sub-stroke0001", 1, 1.0),
                self._build_mask_sample("sub-stroke0002", 0, 1.0),
            ]

            argv = [
                "prog",
                "--pred-dir",
                str(pred_dir),
                "--gt-dir",
                str(gt_dir),
                "--output-dir",
                str(out_dir),
                "--fixed-threshold",
                "0.5",
            ]

            with patch.dict("scripts.evaluation.metrics.engine.THREED_METRIC_CLASSES", {"DiceNativeCoefficient": DiceNativeCoefficient}, clear=True), patch(
                "scripts.evaluation.compute_segmentation_metrics_for_nnunet_2d_predictions.count_matched_pairs",
                return_value=(1, 0, 1),
            ), patch(
                "scripts.evaluation.compute_segmentation_metrics_for_nnunet_2d_predictions.iter_nnunet_slice_samples",
                return_value=iter(samples),
            ), patch("sys.argv", argv), self.assertRaisesRegex(
                ValueError, "deferred 2D reconstruction contract"
            ):
                entry.main()

            self.assertFalse((out_dir / "canonical_results.json").exists())

    def test_repository_2d_entrypoint_rejects_volume_reconstruction(self):
        from scripts.evaluation import compute_segmentation_metrics_for_diffusionmodel_2d_predictions as entry

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_dir = base / "run"
            out_dir = base / "out"
            run_dir.mkdir(parents=True, exist_ok=True)

            yielded_samples = [
                ("n1_single", self._build_prob_sample("sub-stroke0001", 0, 0.9)),
                ("n1_single", self._build_prob_sample("sub-stroke0001", 1, 0.9)),
                ("n1_single", self._build_prob_sample("sub-stroke0002", 0, 0.9)),
            ]

            argv = [
                "prog",
                "--run-dir",
                str(run_dir),
                "--model-name",
                "best_model",
                "--output-dir",
                str(out_dir),
                "--fixed-threshold",
                "0.5",
                "--ensemble-samples",
                "1",
                "--ensemble-method",
                "single",
            ]

            with patch.dict("scripts.evaluation.metrics.engine.THREED_METRIC_CLASSES", {"DiceNativeCoefficient": DiceNativeCoefficient}, clear=True), patch(
                "scripts.evaluation.compute_segmentation_metrics_for_diffusionmodel_2d_predictions.load_config_from_run_dir",
                return_value={},
            ), patch(
                "scripts.evaluation.compute_segmentation_metrics_for_diffusionmodel_2d_predictions.validate_eval_config_contract",
                return_value=None,
            ), patch(
                "scripts.evaluation.compute_segmentation_metrics_for_diffusionmodel_2d_predictions.find_checkpoint",
                return_value=str(run_dir / "models" / "best_model.pth"),
            ), patch(
                "scripts.evaluation.compute_segmentation_metrics_for_diffusionmodel_2d_predictions.load_model",
                return_value=object(),
            ), patch(
                "scripts.evaluation.compute_segmentation_metrics_for_diffusionmodel_2d_predictions.get_dataloaders",
                return_value={"val": _DummyValLoader()},
            ), patch(
                "scripts.evaluation.compute_segmentation_metrics_for_diffusionmodel_2d_predictions.iter_diffusion_case_slice_samples",
                return_value=iter(yielded_samples),
            ), patch("sys.argv", argv), self.assertRaisesRegex(
                ValueError, "deferred 2D reconstruction contract"
            ):
                entry.main()

            self.assertFalse((out_dir / "canonical_results.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
