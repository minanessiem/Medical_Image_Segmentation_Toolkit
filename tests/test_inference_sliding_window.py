"""Architecture-neutral direct and sliding-window probability execution tests."""

import unittest
from unittest.mock import patch

import torch

from src.inference.contracts import (
    InvalidInferencePolicyError,
    InvalidPredictionError,
    PredictorCapabilities,
    UnsupportedModelError,
)
from src.inference.pipeline import ModelProbabilityExecutor
from src.inference.policy import InferencePolicy, SlidingWindowPolicy
from src.utils.monai_sliding_window_backport import sliding_window_inference


class FunctionPredictor:
    def __init__(
        self,
        function,
        *,
        spatial_dims=3,
        supported_precisions=("fp32", "fp16", "bf16"),
    ) -> None:
        self.function = function
        self.calls = []
        self.capabilities = PredictorCapabilities(
            model_family="discriminative",
            spatial_dims=spatial_dims,
            input_channels=1,
            output_channels=1,
            supported_precisions=supported_precisions,
        )

    def predict(self, conditioned_image):
        self.calls.append(conditioned_image.detach().clone())
        return self.function(conditioned_image)


def _policy(
    *,
    enabled=True,
    roi_size=(4, 4, 4),
    sw_batch_size=1,
    overlap=0.5,
    blend_mode="gaussian",
    padding_mode="constant",
    precision="fp32",
    output_space="model_preprocessed",
):
    return InferencePolicy(
        output_space=output_space,
        precision=precision,
        sliding_window=SlidingWindowPolicy(
            enabled=enabled,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            blend_mode=blend_mode,
            padding_mode=padding_mode,
        ),
    )


class TestModelProbabilityExecutor(unittest.TestCase):
    def test_direct_execution_accepts_generic_probability_predictor(self):
        predictor = FunctionPredictor(lambda image: torch.full_like(image[:, :1], 0.25))
        executor = ModelProbabilityExecutor(
            predictor=predictor,
            policy=_policy(enabled=False),
            policy_source="explicit_top_level",
        )
        image = torch.zeros((1, 1, 3, 5, 7), dtype=torch.float32)

        probability = executor(image)

        self.assertEqual(tuple(probability.shape), (1, 1, 3, 5, 7))
        torch.testing.assert_close(probability, torch.full_like(probability, 0.25))
        self.assertEqual(len(predictor.calls), 1)

    def test_sliding_window_matches_probability_before_blending_reference(self):
        image = torch.linspace(-2.0, 2.0, steps=5 * 6 * 7).reshape(1, 1, 5, 6, 7)
        probability_function = lambda patch: torch.sigmoid((2.0 * patch[:, :1]) - 0.5)
        predictor = FunctionPredictor(probability_function)
        policy = _policy(roi_size=(4, 4, 4), sw_batch_size=2, overlap=0.25)

        actual = ModelProbabilityExecutor(
            predictor=predictor,
            policy=policy,
            policy_source="explicit_top_level",
        )(image, show_window_progress=False)
        expected = sliding_window_inference(
            image,
            roi_size=policy.sliding_window.roi_size,
            sw_batch_size=policy.sliding_window.sw_batch_size,
            predictor=probability_function,
            overlap=policy.sliding_window.overlap,
            mode=policy.sliding_window.blend_mode,
            padding_mode=policy.sliding_window.padding_mode,
            progress=False,
        )

        torch.testing.assert_close(actual, expected)
        self.assertGreater(len(predictor.calls), 1)
        self.assertGreaterEqual(float(actual.min()), 0.0)
        self.assertLessEqual(float(actual.max()), 1.0)

    def test_sliding_window_preserves_shape_when_input_is_smaller_than_roi(self):
        predictor = FunctionPredictor(lambda patch: torch.full_like(patch[:, :1], 0.75))
        image = torch.zeros((1, 1, 3, 5, 7), dtype=torch.float32)

        probability = ModelProbabilityExecutor(
            predictor=predictor,
            policy=_policy(roi_size=(6, 8, 10), padding_mode="constant"),
            policy_source="model_contract",
        )(image, show_window_progress=False)

        self.assertEqual(tuple(probability.shape), tuple(image.shape))
        torch.testing.assert_close(probability, torch.full_like(probability, 0.75))

    def test_sliding_window_forwards_established_blend_padding_and_progress_fields(self):
        predictor = FunctionPredictor(lambda patch: torch.full_like(patch[:, :1], 0.25))
        policy = _policy(
            roi_size=(2, 3, 4),
            sw_batch_size=3,
            overlap=0.125,
            blend_mode="constant",
            padding_mode="reflect",
        )
        image = torch.zeros((1, 1, 4, 5, 6), dtype=torch.float32)

        def fake_sliding_window(inputs, **kwargs):
            self.assertIs(inputs, image)
            self.assertEqual(kwargs["roi_size"], (2, 3, 4))
            self.assertEqual(kwargs["sw_batch_size"], 3)
            self.assertEqual(kwargs["overlap"], 0.125)
            self.assertEqual(kwargs["mode"], "constant")
            self.assertEqual(kwargs["padding_mode"], "reflect")
            self.assertTrue(kwargs["progress"])
            self.assertEqual(kwargs["progress_desc"], "SW case-17")
            patch_probability = kwargs["predictor"](inputs)
            torch.testing.assert_close(
                patch_probability,
                torch.full_like(patch_probability, 0.25),
            )
            return patch_probability

        with patch(
            "src.inference.sliding_window.monai_sliding_window_inference",
            side_effect=fake_sliding_window,
        ):
            probability = ModelProbabilityExecutor(
                predictor=predictor,
                policy=policy,
                policy_source="legacy_validation",
            )(
                image,
                progress_label="case-17",
                show_window_progress=True,
            )

        torch.testing.assert_close(probability, torch.full_like(probability, 0.25))

    def test_invalid_probability_fails_in_direct_and_window_paths(self):
        for enabled in (False, True):
            with self.subTest(sliding_window=enabled):
                predictor = FunctionPredictor(
                    lambda patch: torch.full_like(patch[:, :1], 1.25)
                )
                executor = ModelProbabilityExecutor(
                    predictor=predictor,
                    policy=_policy(enabled=enabled),
                    policy_source="explicit_top_level",
                )
                with self.assertRaisesRegex(InvalidPredictionError, r"\[0, 1\]"):
                    executor(
                        torch.zeros((1, 1, 4, 4, 4)),
                        show_window_progress=False,
                    )

    def test_native_output_fails_until_spatial_restoration_cut(self):
        predictor = FunctionPredictor(lambda image: torch.full_like(image[:, :1], 0.5))

        with self.assertRaisesRegex(InvalidInferencePolicyError, "Cut 7"):
            ModelProbabilityExecutor(
                predictor=predictor,
                policy=_policy(output_space="native_input"),
                policy_source="explicit_top_level",
            )

    def test_selected_precision_must_be_supported_by_predictor(self):
        predictor = FunctionPredictor(
            lambda image: torch.full_like(image[:, :1], 0.5),
            supported_precisions=("fp32",),
        )

        with self.assertRaisesRegex(UnsupportedModelError, "fp16"):
            ModelProbabilityExecutor(
                predictor=predictor,
                policy=_policy(precision="fp16"),
                policy_source="explicit_top_level",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for FP16 execution")
    def test_window_batch_one_runs_under_cuda_fp16_autocast(self):
        autocast_states = []

        def predict(patch):
            autocast_states.append(torch.is_autocast_enabled())
            return torch.sigmoid(patch[:, :1])

        predictor = FunctionPredictor(predict)
        image = torch.zeros((1, 1, 3, 3, 3), device="cuda", dtype=torch.float32)
        executor = ModelProbabilityExecutor(
            predictor=predictor,
            policy=_policy(
                roi_size=(2, 2, 2),
                sw_batch_size=1,
                precision="fp16",
            ),
            policy_source="explicit_top_level",
        )

        probability = executor(image, show_window_progress=False)

        self.assertTrue(autocast_states)
        self.assertTrue(all(autocast_states))
        self.assertEqual(tuple(probability.shape), tuple(image.shape))
        self.assertEqual(probability.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
