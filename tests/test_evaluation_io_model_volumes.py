"""
Tests for live-model 3D volume IO producer.
"""

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from scripts.evaluation.core.contracts import VolumeSample
from scripts.evaluation.io.model_volumes import (
    iter_model_volume_samples,
    normalize_probability_prediction,
    resolve_batch_item_identity,
    validate_model_evaluation_mode,
)


class DummyModel(nn.Module):
    def forward(self, x):
        return x


def _base_cfg(diffusion_type="Discriminative", dim="3d"):
    return OmegaConf.create(
        {
            "data_mode": {
                "dim": dim,
                "loader_mode": "full_volumes_3d",
            },
            "diffusion": {"type": diffusion_type},
            "model": {
                "spatial_dims": dim,
                "image_channels": 1,
                "out_channels": 1,
            },
            "validation": {"inference": {"mode": "direct"}},
            "dataset": {
                "active_subsets": {"val": "val_fast"},
                "preprocessing_configs": {
                    "roi": {
                        "slice_2d": [2, 2],
                        "volume_3d": [2, 2, 2],
                    }
                },
            },
        }
    )


class TestModelVolumeIO(unittest.TestCase):
    def test_unsupported_current_3d_diffusion_raises(self):
        cfg = _base_cfg(diffusion_type="OpenAI_DDPM", dim="3d")

        with self.assertRaises(ValueError) as ctx:
            validate_model_evaluation_mode(cfg)

        message = str(ctx.exception)
        self.assertIn("3D live-model evaluation", message)
        self.assertIn("discriminative", message.lower())
        self.assertIn("OpenAI_DDPM", message)

    def test_2d_mode_is_allowed_by_mode_validator(self):
        cfg = _base_cfg(diffusion_type="OpenAI_DDPM", dim="2d")

        validate_model_evaluation_mode(cfg)

    def test_normalize_logits_with_sigmoid(self):
        logits = torch.tensor([[[[[-2.0, 0.0, 2.0]]]]])

        probabilities = normalize_probability_prediction(logits)

        self.assertGreaterEqual(float(probabilities.min()), 0.0)
        self.assertLessEqual(float(probabilities.max()), 1.0)
        self.assertAlmostEqual(float(probabilities[0, 0, 0, 0, 1]), 0.5)
        self.assertFalse(probabilities.requires_grad)
        self.assertEqual(probabilities.device.type, "cpu")

    def test_normalize_probability_inputs_remain_probabilities(self):
        prediction = torch.tensor([[[[[0.0, 0.25, 1.0]]]]])

        probabilities = normalize_probability_prediction(prediction)

        self.assertEqual(float(probabilities.min()), 0.0)
        self.assertEqual(float(probabilities.max()), 1.0)
        self.assertAlmostEqual(float(probabilities[0, 0, 0, 0, 1]), 0.25)

    def test_resolve_batch_item_identity_from_list(self):
        case_id = resolve_batch_item_identity(
            sample_ids=["case_a", "case_b"],
            batch_index=3,
            item_index=1,
        )

        self.assertEqual(case_id, "case_b")

    def test_resolve_batch_item_identity_fallback(self):
        case_id = resolve_batch_item_identity(
            sample_ids=None,
            batch_index=3,
            item_index=1,
        )

        self.assertEqual(case_id, "batch3_item1")

    def test_3d_discriminative_dummy_model_emits_volume_samples(self):
        cfg = _base_cfg(diffusion_type="Discriminative", dim="3d")
        model = DummyModel()
        image = torch.zeros(2, 1, 2, 2, 2)
        label = torch.ones(2, 1, 2, 2, 2)
        sample_ids = ["case_a", "case_b"]
        metas = {
            "source_spacing_xyz": [(1.0, 2.0, 3.0), (2.0, 2.0, 2.0)],
            "site_id": ["site_1", "site_2"],
        }
        dataloader = [(image, label, sample_ids, metas)]

        def inferer(conditioned_image, progress_label=None, show_window_progress=True):
            self.assertEqual(progress_label, "case_a")
            self.assertFalse(show_window_progress)
            return conditioned_image + 0.25

        with patch(
            "scripts.evaluation.io.model_volumes.build_model_probability_executor",
            return_value=inferer,
        ):
            samples = list(
                iter_model_volume_samples(
                    model=model,
                    dataloader=dataloader,
                    cfg=cfg,
                    device="cpu",
                    show_progress=False,
                )
            )

        self.assertEqual(len(samples), 2)
        self.assertIsInstance(samples[0], VolumeSample)
        self.assertEqual(samples[0].case_id, "case_a")
        self.assertEqual(samples[1].case_id, "case_b")
        self.assertEqual(tuple(samples[0].prediction_volume.shape), (1, 2, 2, 2))
        self.assertEqual(tuple(samples[0].ground_truth_volume.shape), (1, 2, 2, 2))
        self.assertAlmostEqual(float(samples[0].prediction_volume.mean()), 0.25)
        self.assertEqual(samples[0].metadata["loader_mode"], "full_volumes_3d")
        self.assertEqual(samples[0].metadata["validation_inference_mode"], "direct")
        self.assertEqual(samples[0].metadata["subset"], "val_fast")
        self.assertEqual(samples[0].metadata["site_id"], "site_1")
        self.assertEqual(samples[1].metadata["site_id"], "site_2")

    def test_iter_model_volume_samples_respects_max_samples(self):
        cfg = _base_cfg(diffusion_type="Discriminative", dim="3d")
        model = DummyModel()
        image = torch.zeros(2, 1, 2, 2, 2)
        label = torch.ones(2, 1, 2, 2, 2)
        dataloader = [(image, label, ["case_a", "case_b"])]

        def inferer(conditioned_image, progress_label=None, show_window_progress=True):
            del progress_label, show_window_progress
            return conditioned_image + 0.5

        with patch(
            "scripts.evaluation.io.model_volumes.build_model_probability_executor",
            return_value=inferer,
        ):
            samples = list(
                iter_model_volume_samples(
                    model=model,
                    dataloader=dataloader,
                    cfg=cfg,
                    device="cpu",
                    show_progress=False,
                    max_samples=1,
                )
            )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].case_id, "case_a")


if __name__ == "__main__":
    unittest.main(verbosity=2)
