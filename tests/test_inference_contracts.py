import unittest

import torch

from src.inference.contracts import (
    InferenceInputError,
    InvalidPredictionError,
    PredictionResult,
    PredictorCapabilities,
    PreprocessedCase,
    SpatialGeometry,
    SpatialRestorationError,
    SpatialTrace,
    UnsupportedModelError,
)
from src.inference.predictors import (
    validate_probability_output,
    validate_predictor_capabilities,
)


class TestInferenceContracts(unittest.TestCase):
    def test_prediction_result_declares_supported_output_space(self):
        geometry = SpatialGeometry(
            shape=(9, 11, 13),
            affine=(
                (1.0, 0.0, 0.0, 4.0),
                (0.0, 2.0, 0.0, -3.0),
                (0.0, 0.0, 3.0, 7.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            spacing=(1.0, 2.0, 3.0),
            orientation="RAS",
        )
        trace = SpatialTrace(original=geometry, model=geometry)
        probability = torch.zeros((1, 1, 9, 11, 13), dtype=torch.float32)

        result = PredictionResult(
            probability=probability,
            output_space="model_preprocessed",
            spatial_trace=trace,
        )

        self.assertEqual(result.output_space, "model_preprocessed")
        self.assertIs(result.probability, probability)

    def test_prediction_result_rejects_unknown_output_space(self):
        geometry = SpatialGeometry.identity((3, 4, 5))
        with self.assertRaisesRegex(InvalidPredictionError, "output_space"):
            PredictionResult(
                probability=torch.zeros((1, 1, 3, 4, 5)),
                output_space="reoriented_but_untracked",
                spatial_trace=SpatialTrace(original=geometry, model=geometry),
            )

    def test_prediction_result_rejects_probability_geometry_mismatch(self):
        geometry = SpatialGeometry.identity((3, 4, 5))

        with self.assertRaisesRegex(
            SpatialRestorationError,
            "probability shape.*does not match",
        ):
            PredictionResult(
                probability=torch.zeros((1, 1, 3, 4, 6)),
                output_space="native_input",
                spatial_trace=SpatialTrace(original=geometry, model=geometry),
            )

    def test_prediction_result_rejects_invalid_binary_mask(self):
        geometry = SpatialGeometry.identity((3, 4, 5))
        probability = torch.zeros((1, 1, 3, 4, 5))

        for mask, message in (
            (torch.zeros_like(probability), "integer or boolean"),
            (
                torch.full(probability.shape, 2, dtype=torch.uint8),
                r"binary values \{0, 1\}",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(InvalidPredictionError, message):
                    PredictionResult(
                        probability=probability,
                        output_space="native_input",
                        spatial_trace=SpatialTrace(
                            original=geometry,
                            model=geometry,
                        ),
                        mask=mask,
                    )


class TestProbabilityOutputValidation(unittest.TestCase):
    def test_valid_3d_probability_tensor_is_returned_unchanged(self):
        image = torch.zeros((1, 1, 8, 9, 10), dtype=torch.float32)
        probability = torch.full((1, 1, 8, 9, 10), 0.25, dtype=torch.float32)

        validated = validate_probability_output(
            probability,
            conditioned_image=image,
            spatial_dims=3,
            output_channels=1,
        )

        self.assertIs(validated, probability)

    def test_rejects_logits_outside_probability_domain(self):
        with self.assertRaisesRegex(InvalidPredictionError, r"\[0, 1\]"):
            validate_probability_output(
                torch.tensor([[[[[-0.1, 1.1]]]]]),
                spatial_dims=3,
                output_channels=1,
            )

    def test_rejects_nan_and_infinity(self):
        for invalid_value in (float("nan"), float("inf")):
            with self.subTest(value=invalid_value):
                with self.assertRaisesRegex(InvalidPredictionError, "finite"):
                    validate_probability_output(
                        torch.tensor([[[[[invalid_value]]]]]),
                        spatial_dims=3,
                        output_channels=1,
                    )

    def test_rejects_wrong_rank_channel_and_spatial_shape(self):
        with self.assertRaisesRegex(InvalidPredictionError, "rank"):
            validate_probability_output(
                torch.zeros((1, 1, 8, 9)),
                spatial_dims=3,
                output_channels=1,
            )

        with self.assertRaisesRegex(InvalidPredictionError, "channels"):
            validate_probability_output(
                torch.zeros((1, 2, 8, 9, 10)),
                spatial_dims=3,
                output_channels=1,
            )

        with self.assertRaisesRegex(InvalidPredictionError, "spatial shape"):
            validate_probability_output(
                torch.zeros((1, 1, 8, 9, 11)),
                conditioned_image=torch.zeros((1, 1, 8, 9, 10)),
                spatial_dims=3,
                output_channels=1,
            )

    def test_current_3d_diffusion_capability_fails_clearly(self):
        capabilities = PredictorCapabilities(
            model_family="diffusion",
            spatial_dims=3,
            input_channels=1,
            output_channels=1,
            supported_precisions=("fp32",),
        )

        with self.assertRaisesRegex(
            UnsupportedModelError,
            "3D non-discriminative diffusion.*not supported",
        ):
            validate_predictor_capabilities(capabilities)


class TestSpecializedContractErrors(unittest.TestCase):
    def test_spatial_contract_uses_spatial_restoration_error(self):
        with self.assertRaisesRegex(SpatialRestorationError, "positive integers"):
            SpatialGeometry.identity((3, 0, 5))

        with self.assertRaisesRegex(SpatialRestorationError, "anatomical code"):
            SpatialGeometry(
                shape=(3, 4, 5),
                affine=SpatialGeometry.identity((3, 4, 5)).affine,
                spacing=(1.0, 1.0, 1.0),
                orientation="RRR",
            )

    def test_preprocessed_case_uses_inference_input_error(self):
        geometry = SpatialGeometry.identity((3, 4, 5))
        trace = SpatialTrace(original=geometry, model=geometry)

        with self.assertRaisesRegex(InferenceInputError, "case_id"):
            PreprocessedCase(
                case_id="",
                image=torch.zeros((1, 1, 3, 4, 5)),
                spatial_trace=trace,
            )

        with self.assertRaisesRegex(InferenceInputError, "floating-point"):
            PreprocessedCase(
                case_id="case",
                image=torch.zeros((1, 1, 3, 4, 5), dtype=torch.uint8),
                spatial_trace=trace,
            )


if __name__ == "__main__":
    unittest.main()
