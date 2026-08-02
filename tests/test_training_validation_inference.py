"""Training validation consumes the shared Cut 4 probability executor."""

import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from src.inference.policy import parse_inference_policy
from src.inference.contracts import InvalidInferenceRuntimeError
from src.training.trainer import validate_one_epoch


class DummyDiffusion:
    def __init__(self) -> None:
        self.eval_calls = 0
        self.train_calls = 0

    def eval(self):
        self.eval_calls += 1

    def train(self):
        self.train_calls += 1


class RecordingMetric:
    def __init__(self) -> None:
        self.predictions = []
        self.targets = []
        self.reset_calls = 0

    def __call__(self, prediction, target):
        self.predictions.append(prediction.detach().clone())
        self.targets.append(target.detach().clone())

    def compute(self):
        return {"test_score": torch.tensor(0.75)}

    def reset(self):
        self.reset_calls += 1


class TestTrainingValidationInference(unittest.TestCase):
    def test_validate_one_epoch_builds_and_calls_shared_executor(self):
        cfg = OmegaConf.create(
            {
                "device": "cpu",
                "validation": {
                    "ensemble": {"enabled": False},
                    "progress_metrics": ["test_score"],
                },
            }
        )
        diffusion = DummyDiffusion()
        metric = RecordingMetric()
        image = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
        label = torch.ones((1, 1, 2, 2, 2), dtype=torch.float32)
        executor_calls = []

        def executor(conditioned_image, progress_label=None, show_window_progress=True):
            executor_calls.append(
                (conditioned_image.detach().clone(), progress_label, show_window_progress)
            )
            return torch.full_like(conditioned_image[:, :1], 0.25)

        executor.policy = parse_inference_policy(
            {"sliding_window": {"enabled": False}},
            model_roi=(2, 2, 2),
        )

        with patch(
            "src.inference.pipeline.build_model_probability_executor",
            return_value=executor,
        ) as build_executor:
            results = validate_one_epoch(
                diffusion=diffusion,
                val_dl=[(image, label, ["case-1"])],
                metrics=[metric],
                logger=None,
                global_step=10,
                cfg=cfg,
            )

        build_executor.assert_called_once_with(backend=diffusion, cfg=cfg)
        self.assertEqual(len(executor_calls), 1)
        self.assertEqual(executor_calls[0][1], "case-1")
        self.assertTrue(executor_calls[0][2])
        torch.testing.assert_close(
            metric.predictions[0],
            torch.full_like(metric.predictions[0], 0.25),
        )
        torch.testing.assert_close(metric.targets[0], torch.ones_like(metric.targets[0]))
        self.assertAlmostEqual(float(results["test_score"]), 0.75)
        self.assertEqual(metric.reset_calls, 1)
        self.assertEqual(diffusion.eval_calls, 1)
        self.assertEqual(diffusion.train_calls, 1)

    def test_validation_rejects_runtime_that_forbids_ground_truth(self):
        cfg = OmegaConf.create(
            {
                "device": "cpu",
                "validation": {
                    "ensemble": {"enabled": False},
                },
                "inference_runtime": {
                    "profile": "native",
                    "constraints": {"allow_ground_truth": False},
                },
            }
        )
        diffusion = DummyDiffusion()
        image = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
        label = torch.ones((1, 1, 2, 2, 2), dtype=torch.float32)
        executor_calls = []

        def executor(conditioned_image, progress_label=None, show_window_progress=True):
            executor_calls.append(conditioned_image)
            return torch.full_like(conditioned_image[:, :1], 0.25)

        executor.policy = parse_inference_policy(
            {"sliding_window": {"enabled": False}},
            model_roi=(2, 2, 2),
        )

        with patch(
            "src.inference.pipeline.build_model_probability_executor",
            return_value=executor,
        ):
            with self.assertRaisesRegex(
                InvalidInferenceRuntimeError,
                "does not permit ground truth",
            ):
                validate_one_epoch(
                    diffusion=diffusion,
                    val_dl=[(image, label, ["case-1"])],
                    metrics=[RecordingMetric()],
                    logger=None,
                    global_step=10,
                    cfg=cfg,
                )

        self.assertEqual(executor_calls, [])
        self.assertEqual(diffusion.eval_calls, 0)
        self.assertEqual(diffusion.train_calls, 0)


if __name__ == "__main__":
    unittest.main()
