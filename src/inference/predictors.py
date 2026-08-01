"""Tensor-level predictor protocols and probability-contract validation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from src.inference.contracts import (
    InferenceInputError,
    InvalidPredictionError,
    ProbabilityPredictor,
    PredictorCapabilities,
    UnsupportedModelError,
    _validate_probability_tensor,
)


SUPPORTED_PRECISIONS = frozenset({"fp16", "fp32", "bf16"})


@dataclass(frozen=True)
class DiscriminativeProbabilityPredictor:
    """Expose an existing discriminative adapter through the shared protocol."""

    backend: Any
    capabilities: PredictorCapabilities

    def predict(self, conditioned_image: torch.Tensor) -> torch.Tensor:
        """Delegate probability generation without interpreting model outputs."""

        return self.backend.sample(conditioned_image, disable_tqdm=True)


def build_probability_predictor(
    *,
    backend: Any,
    cfg: Mapping[str, Any] | DictConfig,
) -> ProbabilityPredictor:
    """Prepare the first supported backend without inspecting its architecture."""

    root = OmegaConf.create(cfg) if not OmegaConf.is_config(cfg) else cfg
    try:
        configured_family = str(
            OmegaConf.select(root, "diffusion.type", default="Discriminative")
        ).strip()
    except OmegaConfBaseException as exc:
        raise UnsupportedModelError(
            f"Could not resolve the configured predictor backend family: {exc}"
        ) from exc

    if configured_family.lower() != "discriminative":
        raise UnsupportedModelError(
            f"Configured inference backend {configured_family!r} is not supported by "
            "the initial shared inference implementation. Implement and register a "
            "backend adapter satisfying src.inference.contracts.ProbabilityPredictor "
            "before enabling generative inference."
        )
    if not callable(getattr(backend, "sample", None)):
        raise UnsupportedModelError(
            "The discriminative inference backend must expose sample(conditioned_image, "
            "disable_tqdm=...) returning probabilities."
        )

    capabilities = PredictorCapabilities(
        model_family="discriminative",
        spatial_dims=_resolve_spatial_dims(root, backend),
        input_channels=_resolve_positive_int(
            root,
            path="model.image_channels",
            fallback=getattr(backend, "image_channels", None),
        ),
        output_channels=_resolve_positive_int(
            root,
            path="model.out_channels",
            fallback=getattr(backend, "mask_channels", None),
        ),
        supported_precisions=tuple(sorted(SUPPORTED_PRECISIONS)),
    )
    validate_predictor_capabilities(capabilities)
    return DiscriminativeProbabilityPredictor(
        backend=backend,
        capabilities=capabilities,
    )


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


def predict_probabilities(
    predictor: ProbabilityPredictor,
    conditioned_image: torch.Tensor,
) -> torch.Tensor:
    """Execute one prepared predictor call and enforce its tensor contract."""

    capabilities = validate_predictor_capabilities(predictor.capabilities)
    _validate_conditioned_image(conditioned_image, capabilities)
    probability = predictor.predict(conditioned_image)
    return validate_probability_output(
        probability,
        conditioned_image=conditioned_image,
        spatial_dims=capabilities.spatial_dims,
        output_channels=capabilities.output_channels,
    )


def _validate_conditioned_image(
    conditioned_image: torch.Tensor,
    capabilities: PredictorCapabilities,
) -> None:
    if not torch.is_tensor(conditioned_image):
        raise InferenceInputError("conditioned_image must be a torch.Tensor.")
    if not conditioned_image.is_floating_point():
        raise InferenceInputError("conditioned_image must be floating-point.")
    expected_rank = int(capabilities.spatial_dims) + 2
    if conditioned_image.ndim != expected_rank:
        raise InferenceInputError(
            "conditioned_image has the wrong rank: "
            f"expected {expected_rank}, got {conditioned_image.ndim}."
        )
    if int(conditioned_image.shape[1]) != int(capabilities.input_channels):
        raise InferenceInputError(
            "conditioned_image has incorrect channels: "
            f"expected {capabilities.input_channels}, got {conditioned_image.shape[1]}."
        )


def _resolve_spatial_dims(root: Any, backend: Any) -> int:
    try:
        value = (
            OmegaConf.select(root, "model.spatial_dims", default=None)
            or OmegaConf.select(root, "data_mode.dim", default=None)
            or getattr(backend, "spatial_dims", None)
        )
    except OmegaConfBaseException as exc:
        raise UnsupportedModelError(
            f"Could not resolve predictor spatial dimensionality: {exc}"
        ) from exc
    token = str(value).strip().lower()
    if token.endswith("d"):
        token = token[:-1]
    if token not in {"2", "3"}:
        raise UnsupportedModelError(
            "Predictor spatial dimensionality must resolve to 2D or 3D; "
            f"got {value!r}."
        )
    return int(token)


def _resolve_positive_int(
    root: Any,
    *,
    path: str,
    fallback: Any,
) -> int:
    try:
        value = OmegaConf.select(root, path, default=None)
    except OmegaConfBaseException as exc:
        raise UnsupportedModelError(
            f"Could not resolve predictor capability {path}: {exc}"
        ) from exc
    if value is None:
        value = fallback
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise UnsupportedModelError(
            f"Predictor capability {path} must be a positive integer, got {value!r}."
        )
    return int(value)


__all__ = [
    "DiscriminativeProbabilityPredictor",
    "build_probability_predictor",
    "predict_probabilities",
    "validate_predictor_capabilities",
    "validate_probability_output",
]
