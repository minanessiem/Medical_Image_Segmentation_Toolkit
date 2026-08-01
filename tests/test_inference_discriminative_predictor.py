"""Contracts for the initial discriminative probability-predictor backend."""

import unittest

import torch
from omegaconf import OmegaConf

from src.inference.contracts import UnsupportedModelError
from src.inference.pipeline import build_model_probability_executor
from src.inference.predictors import build_probability_predictor


class StubDiscriminativeBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.disable_tqdm = None

    def sample(self, conditioned_image, disable_tqdm=False):
        self.calls += 1
        self.disable_tqdm = disable_tqdm
        return torch.sigmoid(conditioned_image[:, :1])


def _cfg(diffusion_type="Discriminative"):
    return OmegaConf.create(
        {
            "diffusion": {"type": diffusion_type},
            "model": {
                "spatial_dims": "3d",
                "image_channels": 1,
                "out_channels": 1,
            },
            "data_mode": {"dim": "3d", "loader_mode": "full_volumes_3d"},
            "dataset": {
                "preprocessing_configs": {
                    "roi": {"volume_3d": [4, 4, 4]},
                }
            },
            "validation": {
                "inference": {
                    "mode": "direct",
                    "sliding_window": {"roi_size": None},
                }
            },
        }
    )


class TestDiscriminativeProbabilityPredictor(unittest.TestCase):
    def test_builder_wraps_backend_probability_sampling_without_model_branching(self):
        backend = StubDiscriminativeBackend()
        predictor = build_probability_predictor(backend=backend, cfg=_cfg())
        image = torch.tensor([[[[[-1.0, 0.0, 1.0]]]]])

        probability = predictor.predict(image)

        torch.testing.assert_close(probability, torch.sigmoid(image))
        self.assertEqual(backend.calls, 1)
        self.assertTrue(backend.disable_tqdm)
        self.assertEqual(predictor.capabilities.model_family, "discriminative")
        self.assertEqual(predictor.capabilities.spatial_dims, 3)
        self.assertEqual(predictor.capabilities.input_channels, 1)
        self.assertEqual(predictor.capabilities.output_channels, 1)

    def test_generative_backend_fails_with_future_registration_hook(self):
        backend = StubDiscriminativeBackend()

        with self.assertRaises(UnsupportedModelError) as ctx:
            build_probability_predictor(
                backend=backend,
                cfg=_cfg(diffusion_type="OpenAI_DDPM"),
            )

        message = str(ctx.exception)
        self.assertIn("OpenAI_DDPM", message)
        self.assertIn("ProbabilityPredictor", message)
        self.assertIn("register", message.lower())
        self.assertEqual(backend.calls, 0)

    def test_discriminative_backend_requires_sample_boundary(self):
        with self.assertRaisesRegex(UnsupportedModelError, "sample"):
            build_probability_predictor(backend=object(), cfg=_cfg())

    def test_shared_executor_records_legacy_policy_source(self):
        executor = build_model_probability_executor(
            backend=StubDiscriminativeBackend(),
            cfg=_cfg(),
        )

        self.assertEqual(executor.policy_source, "legacy_validation")
        self.assertFalse(executor.policy.sliding_window.enabled)
        self.assertEqual(executor.policy.sliding_window.roi_size, (4, 4, 4))


if __name__ == "__main__":
    unittest.main()
