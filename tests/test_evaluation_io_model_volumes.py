"""
Tests for live-model 3D volume IO producer.
"""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
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
from src.inference.contracts import (
    LabeledPreprocessedCase,
    NativeImageMetadata,
    PreprocessedCase,
    SpatialGeometry,
    SpatialTrace,
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
            decision=SimpleNamespace(threshold=0.5),
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

    def test_native_output_uses_case_aware_preprocessing_and_native_label(self):
        cfg = _base_cfg(diffusion_type="Discriminative", dim="3d")
        cfg.inference = OmegaConf.create(
            {
                "output_space": "native_input",
                "precision": "fp32",
                "sliding_window": {
                    "enabled": False,
                    "roi_size": [2, 2, 2],
                    "sw_batch_size": 1,
                    "overlap": 0.5,
                    "blend_mode": "gaussian",
                    "padding_mode": "constant",
                },
                "decision": {"threshold": 0.5},
            }
        )
        cfg.dataset.id = "isles26"
        cfg.dataset.modalities = ["T1_RAW"]
        cfg.validation.val_batch_size = 1
        native_affine = np.array(
            [
                [-1.0, 0.0, 0.0, 7.0],
                [0.0, 2.0, 0.0, -3.0],
                [0.0, 0.0, 1.5, 4.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        model_geometry = SpatialGeometry.identity((2, 2, 2))
        native_geometry = SpatialGeometry(
            shape=(3, 4, 5),
            affine=tuple(tuple(float(value) for value in row) for row in native_affine),
            spacing=(1.0, 2.0, 1.5),
            orientation="LAS",
        )
        native_metadata = NativeImageMetadata(
            canonical_key="T1",
            shape=native_geometry.shape,
            dtype="float32",
            affine=native_geometry.affine,
            spacing=native_geometry.spacing,
            orientation=native_geometry.orientation,
            qform=native_geometry.affine,
            sform=native_geometry.affine,
            qform_code=1,
            sform_code=1,
            source_reference="synthetic",
        )
        case = PreprocessedCase(
            case_id="native-case",
            image=MetaTensor(torch.zeros((1, 1, 2, 2, 2)), affine=torch.eye(4)),
            spatial_trace=SpatialTrace(
                original=native_geometry,
                model=model_geometry,
                transform_history=(
                    {"class": "SpatialResample", "do_transforms": True},
                ),
            ),
            native_metadata={"T1": native_metadata},
            reference_key="T1",
        )
        labeled = LabeledPreprocessedCase(
            case=case,
            model_label=torch.zeros((1, 1, 2, 2, 2)),
            model_label_geometry=model_geometry,
            native_label=torch.ones((1, 1, 3, 4, 5)),
            native_label_metadata=replace(native_metadata, canonical_key="label"),
        )
        dataloader = SimpleNamespace(
            dataset=SimpleNamespace(
                database=[
                    {
                        "caseID": "native-case",
                        "T1": ["t1.nii.gz"],
                        "label": "label.nii.gz",
                    }
                ]
            ),
            batch_size=1,
        )
        native_executor = DummyExecutor(0.75)
        native_executor.policy.output_space = "native_input"

        with patch(
            "scripts.evaluation.io.model_volumes.build_model_probability_executor",
            return_value=native_executor,
        ), patch(
            "scripts.evaluation.io.model_volumes.preprocess_case",
            return_value=labeled,
        ) as preprocess_mock:
            samples = list(
                iter_model_volume_samples(
                    model=DummyModel(),
                    dataloader=dataloader,
                    cfg=cfg,
                    device="cpu",
                    show_progress=False,
                )
            )

        preprocess_mock.assert_called_once()
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].prediction_space, "native_input")
        self.assertEqual(samples[0].reference_space, "native_input")
        self.assertEqual(tuple(samples[0].prediction_volume.shape), (1, 3, 4, 5))
        self.assertEqual(tuple(samples[0].ground_truth_volume.shape), (1, 3, 4, 5))
        self.assertEqual(samples[0].prediction_geometry, native_geometry)
        self.assertEqual(samples[0].reference_geometry, native_geometry)
        self.assertTrue(samples[0].metadata["spatial_restoration_applied"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
