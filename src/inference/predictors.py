"""Tensor-level predictor protocols and probability-contract validation."""

from __future__ import annotations

from numbers import Integral
from typing import Optional

import torch

from src.inference.contracts import (
    InferenceInputError,
    InvalidPredictionError,
    PredictorCapabilities,
    UnsupportedModelError,
    _validate_probability_tensor,
)


SUPPORTED_PRECISIONS = frozenset({"fp16", "fp32", "bf16"})


def validate_predictor_capabilities(
    capabilities: PredictorCapabilities,
) -> PredictorCapabilities:
    """Validate the presently certified predictor capability boundary."""
    if not isinstance(capabilities, PredictorCapabilities):
        raise UnsupportedModelError(
            "Predictor capabilities must be a PredictorCapabilities instance."
        )
    if not isinstance(capabilities.model_family, str):
        raise UnsupportedModelError("Predictor model_family must be a string.")
    family = capabilities.model_family.strip().lower()
    if (
        isinstance(capabilities.spatial_dims, bool)
        or not isinstance(capabilities.spatial_dims, Integral)
        or capabilities.spatial_dims not in (2, 3)
    ):
        raise UnsupportedModelError(
            f"Unsupported predictor spatial dimensionality: {capabilities.spatial_dims}."
        )
    if capabilities.spatial_dims == 3 and family != "discriminative":
        raise UnsupportedModelError(
            "3D non-discriminative diffusion inference is not supported by the current "
            "shared predictor contract. Use a certified 3D discriminative backend."
        )
    if family != "discriminative":
        raise UnsupportedModelError(
            f"Unsupported predictor model family {capabilities.model_family!r}. "
            "The initial shared inference implementation supports discriminative models only."
        )
    if (
        isinstance(capabilities.input_channels, bool)
        or not isinstance(capabilities.input_channels, Integral)
        or capabilities.input_channels <= 0
        or isinstance(capabilities.output_channels, bool)
        or not isinstance(capabilities.output_channels, Integral)
        or capabilities.output_channels <= 0
    ):
        raise UnsupportedModelError("Predictor input and output channel counts must be positive.")
    if capabilities.returns_probabilities is not True:
        raise UnsupportedModelError("Predictors must declare probability-domain output.")
    if not isinstance(capabilities.supported_precisions, tuple) or not capabilities.supported_precisions:
        raise UnsupportedModelError("Predictor must declare at least one supported precision.")
    normalized_precisions = {
        str(precision).lower() for precision in capabilities.supported_precisions
    }
    unknown_precisions = normalized_precisions - SUPPORTED_PRECISIONS
    if unknown_precisions:
        raise UnsupportedModelError(
            f"Predictor declares unsupported precisions: {sorted(unknown_precisions)}."
        )
    return capabilities


def validate_probability_output(
    probability: torch.Tensor,
    *,
    conditioned_image: Optional[torch.Tensor] = None,
    spatial_dims: int,
    output_channels: int,
) -> torch.Tensor:
    """Validate finite ``[B,C,*spatial]`` floating probabilities in ``[0, 1]``."""
    if isinstance(spatial_dims, bool) or spatial_dims not in (2, 3):
        raise InvalidPredictionError(
            f"spatial_dims must be 2 or 3 for predictor validation, got {spatial_dims!r}."
        )
    if (
        isinstance(output_channels, bool)
        or not isinstance(output_channels, Integral)
        or output_channels <= 0
    ):
        raise InvalidPredictionError(
            f"output_channels must be a positive integer, got {output_channels!r}."
        )
    _validate_probability_tensor(probability, field_name="Predictor probability output")
    expected_rank = int(spatial_dims) + 2
    if probability.ndim != expected_rank:
        raise InvalidPredictionError(
            "Predictor probability output has the wrong rank: "
            f"expected {expected_rank} for spatial_dims={spatial_dims}, "
            f"got {probability.ndim} with shape {tuple(probability.shape)}."
        )
    if int(probability.shape[1]) != int(output_channels):
        raise InvalidPredictionError(
            "Predictor probability output has incorrect channels: "
            f"expected {output_channels}, got {probability.shape[1]}."
        )

    if conditioned_image is not None:
        if not torch.is_tensor(conditioned_image):
            raise InferenceInputError(
                "conditioned_image must be a torch.Tensor when provided."
            )
        if conditioned_image.ndim != expected_rank:
            raise InferenceInputError(
                "conditioned_image has the wrong rank for predictor validation: "
                f"expected {expected_rank}, got {conditioned_image.ndim}."
            )
        if probability.shape[0] != conditioned_image.shape[0]:
            raise InvalidPredictionError(
                "Predictor probability batch size does not match conditioned image: "
                f"{probability.shape[0]} != {conditioned_image.shape[0]}."
            )
        if tuple(probability.shape[2:]) != tuple(conditioned_image.shape[2:]):
            raise InvalidPredictionError(
                "Predictor probability spatial shape does not match conditioned image: "
                f"{tuple(probability.shape[2:])} != {tuple(conditioned_image.shape[2:])}."
            )
    return probability
