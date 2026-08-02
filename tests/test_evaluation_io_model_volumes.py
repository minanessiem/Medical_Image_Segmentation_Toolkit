"""
Tests for live-model 3D volume IO producer.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from monai.data import MetaTensor
from omegaconf import OmegaConf

from scripts.evaluation.core.contracts import VolumeSample
from scripts.evaluation.io.model_volumes import (
    iter_model_volume_samples,
    resolve_batch_item_identity,
    validate_model_evaluation_mode,
)


class DummyModel(nn.Module):
    def forward(self, x):
        return x


class DummyExecutor:
    def __init__(self, value: float):
        self.value = float(value)
        self.policy = SimpleNamespace(
            output_space="model_preprocessed",
            precision="fp32",
        )
        self.policy_source = "explicit_top_level"

    def __call__(self, conditioned_image, progress_label=None, show_window_progress=True):
        del progress_label, show_window_progress
        return torch.full_like(conditioned_image, self.value)


def _meta_batch(values: torch.Tensor, affines: torch.Tensor) -> MetaTensor:
    return MetaTensor(values, affine=affines)


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
        self.assertIn("ProbabilityPredictor", message)

    def test_2d_mode_is_rejected_until_reconstruction_contract_exists(self):
        cfg = _base_cfg(diffusion_type="OpenAI_DDPM", dim="2d")

        with self.assertRaisesRegex(ValueError, "deferred 2D reconstruction contract"):
            validate_model_evaluation_mode(cfg)

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
        affines = torch.stack(
            [
                torch.diag(torch.tensor([1.0, 2.0, 3.0, 1.0])),
                torch.diag(torch.tensor([2.0, 2.0, 2.0, 1.0])),
            ]
        )
        image = _meta_batch(torch.zeros(2, 1, 2, 2, 2), affines)
        label = _meta_batch(torch.ones(2, 1, 2, 2, 2), affines.clone())
        sample_ids = ["case_a", "case_b"]
        metas = {
            "source_spacing_xyz": [(1.0, 2.0, 3.0), (2.0, 2.0, 2.0)],
            "site_id": ["site_1", "site_2"],
        }
        dataloader = [(image, label, sample_ids, metas)]

        with patch(
            "scripts.evaluation.io.model_volumes.build_model_probability_executor",
            return_value=DummyExecutor(0.234567),
        ), patch("scripts.evaluation.io.model_volumes.torch.sigmoid") as sigmoid_mock:
            samples = list(
                iter_model_volume_samples(
                    model=model,
                    dataloader=dataloader,
                    cfg=cfg,
                    device="cpu",
                    show_progress=False,
                )
            )

        sigmoid_mock.assert_not_called()

        self.assertEqual(len(samples), 2)
        self.assertIsInstance(samples[0], VolumeSample)
        self.assertEqual(samples[0].case_id, "case_a")
        self.assertEqual(samples[1].case_id, "case_b")
        self.assertEqual(tuple(samples[0].prediction_volume.shape), (1, 2, 2, 2))
        self.assertEqual(tuple(samples[0].ground_truth_volume.shape), (1, 2, 2, 2))
        self.assertAlmostEqual(float(samples[0].prediction_volume.mean()), 0.234567)
        self.assertEqual(samples[0].prediction_space, "model_preprocessed")
        self.assertEqual(samples[0].reference_space, "model_preprocessed")
        self.assertEqual(samples[0].prediction_geometry.spacing, (1.0, 2.0, 3.0))
        self.assertEqual(samples[1].prediction_geometry.spacing, (2.0, 2.0, 2.0))
        self.assertEqual(samples[0].metadata["loader_mode"], "full_volumes_3d")
        self.assertEqual(samples[0].metadata["inference_policy_source"], "explicit_top_level")
        self.assertEqual(samples[0].metadata["subset"], "val_fast")
        self.assertEqual(samples[0].metadata["site_id"], "site_1")
        self.assertEqual(samples[1].metadata["site_id"], "site_2")

    def test_iter_model_volume_samples_respects_max_samples(self):
        cfg = _base_cfg(diffusion_type="Discriminative", dim="3d")
        model = DummyModel()
        affines = torch.stack([torch.eye(4), torch.eye(4)])
        image = _meta_batch(torch.zeros(2, 1, 2, 2, 2), affines)
        label = _meta_batch(torch.ones(2, 1, 2, 2, 2), affines.clone())
        dataloader = [(image, label, ["case_a", "case_b"])]

        with patch(
            "scripts.evaluation.io.model_volumes.build_model_probability_executor",
            return_value=DummyExecutor(0.5),
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
