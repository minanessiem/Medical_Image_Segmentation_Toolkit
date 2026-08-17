"""Invertible, configurable probability-space test-time augmentation."""

from __future__ import annotations

import unittest

import torch

from src.inference.augmentation import TtaView, build_tta_views
from src.inference.contracts import InvalidPredictionError, PredictorCapabilities
from src.inference.pipeline import ModelProbabilityExecutor
from src.inference.policy import InferencePolicy, SlidingWindowPolicy, TtaPolicy


def _ramp(shape=(2, 3, 4)) -> torch.Tensor:
    values = torch.arange(float(torch.tensor(shape).prod().item()))
    values = values / values.max()
    return values.reshape(1, 1, *shape)


class _RecordingIdentityPredictor:
    capabilities = PredictorCapabilities(
        model_family="discriminative",
        spatial_dims=3,
        input_channels=1,
        output_channels=1,
        supported_precisions=("fp32",),
    )

    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def predict(self, conditioned_image: torch.Tensor) -> torch.Tensor:
        self.inputs.append(conditioned_image.detach().clone())
        return conditioned_image.detach().clone()


def _policy(*axes: str) -> InferencePolicy:
    return InferencePolicy(
        sliding_window=SlidingWindowPolicy(
            roi_size=(2, 3, 4),
            enabled=False,
        ),
        precision="fp32",
        tta=TtaPolicy(enabled=bool(axes), flip_axes=tuple(axes)),
    )


class TestInvertibleFlipTta(unittest.TestCase):
    def test_named_axes_map_to_model_grid_dimensions_and_round_trip_exactly(self):
        original = _ramp()
        expected = {
            "x": torch.flip(original, dims=(-3,)),
            "y": torch.flip(original, dims=(-2,)),
            "z": torch.flip(original, dims=(-1,)),
        }

        for axis in ("x", "y", "z"):
            with self.subTest(axis=axis):
                view = TtaView(
                    name=f"flip_{axis}",
                    spatial_dims=3,
                    flip_axis=axis,
                )
                transformed = view.apply(original)
                torch.testing.assert_close(transformed, expected[axis], rtol=0, atol=0)
                torch.testing.assert_close(view.invert(transformed), original, rtol=0, atol=0)
                torch.testing.assert_close(original, _ramp(), rtol=0, atol=0)

    def test_xy_policy_is_identity_plus_two_independent_views(self):
        views = build_tta_views(
            TtaPolicy(enabled=True, flip_axes=("x", "y")),
            spatial_dims=3,
        )

        self.assertEqual(tuple(view.name for view in views), ("identity", "flip_x", "flip_y"))
        self.assertNotIn("flip_xy", tuple(view.name for view in views))

    def test_z_flip_is_available_when_explicitly_configured_for_3d(self):
        original = _ramp()
        views = build_tta_views(
            TtaPolicy(enabled=True, flip_axes=("z",)),
            spatial_dims=3,
        )

        self.assertEqual(tuple(view.name for view in views), ("identity", "flip_z"))
        torch.testing.assert_close(
            views[1].apply(original),
            torch.flip(original, dims=(-1,)),
            rtol=0,
            atol=0,
        )

    def test_each_probability_is_inverse_flipped_before_the_mean(self):
        image = _ramp()
        predictor = _RecordingIdentityPredictor()
        executor = ModelProbabilityExecutor(
            predictor=predictor,
            policy=_policy("x", "y"),
            policy_source="explicit_top_level",
        )

        probability = executor(image, progress_label="case", show_window_progress=False)

        self.assertEqual(len(predictor.inputs), 3)
        torch.testing.assert_close(predictor.inputs[0], image, rtol=0, atol=0)
        torch.testing.assert_close(
            predictor.inputs[1], torch.flip(image, dims=(-3,)), rtol=0, atol=0
        )
        torch.testing.assert_close(
            predictor.inputs[2], torch.flip(image, dims=(-2,)), rtol=0, atol=0
        )
        torch.testing.assert_close(probability, image, rtol=0, atol=1e-7)
        self.assertEqual(probability.dtype, torch.float32)

    def test_tta_disabled_uses_one_prediction_without_averaging(self):
        image = _ramp()
        predictor = _RecordingIdentityPredictor()
        executor = ModelProbabilityExecutor(
            predictor=predictor,
            policy=_policy(),
            policy_source="explicit_top_level",
        )

        probability = executor(image, show_window_progress=False)

        self.assertEqual(len(predictor.inputs), 1)
        torch.testing.assert_close(probability, image, rtol=0, atol=0)

    def test_view_rejects_the_wrong_tensor_rank_or_unavailable_axis(self):
        with self.assertRaisesRegex(InvalidPredictionError, "layout"):
            TtaView(name="flip_x", spatial_dims=3, flip_axis="x").apply(
                torch.zeros((1, 1, 4, 4))
            )
        with self.assertRaisesRegex(InvalidPredictionError, "not available"):
            TtaView(name="flip_z", spatial_dims=2, flip_axis="z").apply(
                torch.zeros((1, 1, 4, 4))
            )


if __name__ == "__main__":
    unittest.main()
