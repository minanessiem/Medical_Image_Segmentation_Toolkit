"""Typed, transport-independent contracts for shared medical-image inference."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Any, Mapping, Optional, Protocol, Tuple, runtime_checkable

import torch


SUPPORTED_OUTPUT_SPACES = frozenset({"model_preprocessed", "native_input"})


class InferenceError(RuntimeError):
    """Base class for shared inference boundary failures."""


class UnsupportedModelError(InferenceError):
    """The requested model family or dimensionality is not supported."""


class InvalidInferencePolicyError(InferenceError):
    """The shared inference policy is invalid or unsupported."""


class InvalidInferenceRuntimeError(InferenceError):
    """An execution profile or cross-policy request is invalid."""


class InvalidPredictionError(InferenceError):
    """A predictor or pipeline result violates the shared output contract."""


class SpatialRestorationError(InferenceError):
    """A prediction could not be restored to its declared spatial grid."""


class InferenceInputError(InferenceError):
    """An inference consumer supplied an invalid image or interface payload."""


class ResourceLimitError(InferenceError):
    """Inference exceeded a declared runtime resource limit."""


Affine = Tuple[Tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class PredictorCapabilities:
    """Capabilities declared by a tensor-level probability predictor."""

    model_family: str
    spatial_dims: int
    input_channels: int
    output_channels: int
    supported_precisions: Tuple[str, ...]
    returns_probabilities: bool = True


@runtime_checkable
class ProbabilityPredictor(Protocol):
    """Minimal predictor boundary: ``[B,C,*spatial]`` to probabilities."""

    capabilities: PredictorCapabilities

    def predict(self, conditioned_image: torch.Tensor) -> torch.Tensor:
        """Return floating probabilities aligned with the input spatial grid."""


@dataclass(frozen=True)
class SpatialGeometry:
    """Shape and physical-space metadata for one three-dimensional grid."""

    shape: Tuple[int, int, int]
    affine: Affine
    spacing: Tuple[float, float, float]
    orientation: str

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple) or len(self.shape) != 3 or any(
            isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
            for value in self.shape
        ):
            raise SpatialRestorationError(
                f"SpatialGeometry.shape must contain three positive integers, got {self.shape}."
            )
        if (
            not isinstance(self.affine, tuple)
            or len(self.affine) != 4
            or any(not isinstance(row, tuple) or len(row) != 4 for row in self.affine)
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for row in self.affine
                for value in row
            )
        ):
            raise SpatialRestorationError(
                "SpatialGeometry.affine must be a finite numeric 4x4 matrix."
            )
        if not isinstance(self.spacing, tuple) or len(self.spacing) != 3 or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or value <= 0
            for value in self.spacing
        ):
            raise SpatialRestorationError(
                f"SpatialGeometry.spacing must contain three positive values: {self.spacing}"
            )
        if not isinstance(self.orientation, str):
            raise SpatialRestorationError(
                "SpatialGeometry.orientation must be a string anatomical code."
            )
        orientation = self.orientation.strip().upper()
        axis_groups = {"R": 0, "L": 0, "A": 1, "P": 1, "S": 2, "I": 2}
        if (
            len(orientation) != 3
            or any(axis not in axis_groups for axis in orientation)
            or {axis_groups[axis] for axis in orientation} != {0, 1, 2}
        ):
            raise SpatialRestorationError(
                "SpatialGeometry.orientation must be a valid three-axis anatomical code "
                f"such as 'RAS' or 'LPS', got {self.orientation!r}."
            )

    @classmethod
    def identity(cls, shape: Tuple[int, int, int]) -> "SpatialGeometry":
        return cls(
            shape=shape,
            affine=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            spacing=(1.0, 1.0, 1.0),
            orientation="RAS",
        )


@dataclass(frozen=True)
class NativeImageMetadata:
    """Immutable native NIfTI metadata captured before dataset preprocessing."""

    canonical_key: str
    shape: Tuple[int, int, int]
    dtype: str
    affine: Affine
    spacing: Tuple[float, float, float]
    orientation: str
    qform: Optional[Affine]
    sform: Optional[Affine]
    qform_code: int
    sform_code: int
    source_reference: str

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_key, str) or not self.canonical_key.strip():
            raise InferenceInputError(
                "NativeImageMetadata.canonical_key must not be empty."
            )
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise InferenceInputError("NativeImageMetadata.dtype must not be empty.")
        SpatialGeometry(
            shape=self.shape,
            affine=self.affine,
            spacing=self.spacing,
            orientation=self.orientation,
        )
        for field_name in ("qform", "sform"):
            form = getattr(self, field_name)
            if form is not None:
                SpatialGeometry(
                    shape=self.shape,
                    affine=form,
                    spacing=self.spacing,
                    orientation=self.orientation,
                )
        for field_name in ("qform_code", "sform_code"):
            code = getattr(self, field_name)
            if isinstance(code, bool) or not isinstance(code, Integral) or code < 0:
                raise InferenceInputError(
                    f"NativeImageMetadata.{field_name} must be a non-negative integer."
                )
        if not isinstance(self.source_reference, str) or not self.source_reference.strip():
            raise InferenceInputError(
                "NativeImageMetadata.source_reference must not be empty."
            )

    @property
    def geometry(self) -> SpatialGeometry:
        return SpatialGeometry(
            shape=self.shape,
            affine=self.affine,
            spacing=self.spacing,
            orientation=self.orientation,
        )


@dataclass(frozen=True)
class SpatialTrace:
    """Metadata required to relate the model grid to the original input grid."""

    original: SpatialGeometry
    model: SpatialGeometry
    transform_history: Tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.original, SpatialGeometry):
            raise SpatialRestorationError("SpatialTrace.original must be SpatialGeometry.")
        if not isinstance(self.model, SpatialGeometry):
            raise SpatialRestorationError("SpatialTrace.model must be SpatialGeometry.")
        if not isinstance(self.transform_history, tuple) or any(
            not isinstance(step, Mapping) for step in self.transform_history
        ):
            raise SpatialRestorationError(
                "SpatialTrace.transform_history must be a tuple of mapping records."
            )


@dataclass(frozen=True)
class PreprocessedCase:
    """One label-optional model input with its invertible spatial trace."""

    case_id: str
    image: torch.Tensor
    spatial_trace: SpatialTrace
    native_metadata: Mapping[str, NativeImageMetadata] = field(default_factory=dict)
    reference_key: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise InferenceInputError("PreprocessedCase.case_id must not be empty.")
        if not torch.is_tensor(self.image):
            raise InferenceInputError("PreprocessedCase.image must be a torch.Tensor.")
        if self.image.ndim not in (4, 5):
            raise InferenceInputError(
                "PreprocessedCase.image must use [B,C,*spatial] layout for 2D or 3D "
                f"inference, got shape {tuple(self.image.shape)}."
            )
        if self.image.numel() == 0 or self.image.shape[0] <= 0 or self.image.shape[1] <= 0:
            raise InferenceInputError(
                f"PreprocessedCase.image must be non-empty, got shape {tuple(self.image.shape)}."
            )
        if not torch.is_floating_point(self.image):
            raise InferenceInputError("PreprocessedCase.image must use a floating-point dtype.")
        if not bool(torch.isfinite(self.image).all().item()):
            raise InferenceInputError("PreprocessedCase.image must contain only finite values.")
        if not isinstance(self.spatial_trace, SpatialTrace):
            raise InferenceInputError("PreprocessedCase.spatial_trace must be SpatialTrace.")
        if not isinstance(self.native_metadata, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, NativeImageMetadata)
            for key, value in self.native_metadata.items()
        ):
            raise InferenceInputError(
                "PreprocessedCase.native_metadata must map canonical keys to "
                "NativeImageMetadata values."
            )
        if self.native_metadata:
            if not isinstance(self.reference_key, str) or not self.reference_key.strip():
                raise InferenceInputError(
                    "PreprocessedCase.reference_key is required when native metadata is present."
                )
            if self.reference_key not in self.native_metadata:
                raise InferenceInputError(
                    f"PreprocessedCase.reference_key {self.reference_key!r} is absent from "
                    "native_metadata."
                )
        elif self.reference_key is not None:
            raise InferenceInputError(
                "PreprocessedCase.reference_key requires native_metadata."
            )
        if not isinstance(self.metadata, Mapping):
            raise InferenceInputError("PreprocessedCase.metadata must be a mapping.")


@dataclass(frozen=True)
class LabeledPreprocessedCase:
    """A preprocessed inference case accompanied by model- and native-space labels."""

    case: PreprocessedCase
    model_label: torch.Tensor
    native_label: torch.Tensor
    native_label_metadata: NativeImageMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.case, PreprocessedCase):
            raise InferenceInputError(
                "LabeledPreprocessedCase.case must be a PreprocessedCase."
            )
        for field_name in ("model_label", "native_label"):
            tensor = getattr(self, field_name)
            if not torch.is_tensor(tensor) or tensor.ndim != 5:
                raise InferenceInputError(
                    f"LabeledPreprocessedCase.{field_name} must use [B,C,D,H,W] layout."
                )
            if not torch.is_floating_point(tensor):
                raise InferenceInputError(
                    f"LabeledPreprocessedCase.{field_name} must use a floating-point dtype."
                )
            if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all().item()):
                raise InferenceInputError(
                    f"LabeledPreprocessedCase.{field_name} must be non-empty and finite."
                )
        if (
            self.model_label.shape[0] != self.case.image.shape[0]
            or tuple(self.model_label.shape[2:]) != tuple(self.case.image.shape[2:])
        ):
            raise InferenceInputError(
                "LabeledPreprocessedCase.model_label must match the preprocessed image "
                "batch and spatial shape."
            )
        if not isinstance(self.native_label_metadata, NativeImageMetadata):
            raise InferenceInputError(
                "LabeledPreprocessedCase.native_label_metadata must be NativeImageMetadata."
            )
        if tuple(int(value) for value in self.native_label.shape[2:]) != (
            self.native_label_metadata.shape
        ):
            raise InferenceInputError(
                "LabeledPreprocessedCase.native_label spatial shape must match "
                "native_label_metadata."
            )


@dataclass(frozen=True)
class TimingResourceRecord:
    """Optional stage timing and peak-memory measurements for one invocation."""

    stage_seconds: Mapping[str, float] = field(default_factory=dict)
    peak_gpu_allocated_bytes: Optional[int] = None
    peak_gpu_reserved_bytes: Optional[int] = None
    peak_host_rss_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage_seconds, Mapping):
            raise ResourceLimitError(
                "TimingResourceRecord.stage_seconds must be a mapping."
            )
        invalid_stages = {
            name: seconds
            for name, seconds in self.stage_seconds.items()
            if (
                isinstance(seconds, bool)
                or not isinstance(seconds, Real)
                or not math.isfinite(float(seconds))
                or seconds < 0
            )
        }
        if invalid_stages:
            raise ResourceLimitError(
                f"TimingResourceRecord stage durations must be finite and non-negative: "
                f"{invalid_stages}."
            )
        for field_name in (
            "peak_gpu_allocated_bytes",
            "peak_gpu_reserved_bytes",
            "peak_host_rss_bytes",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            ):
                raise ResourceLimitError(
                    f"TimingResourceRecord.{field_name} must be a non-negative integer "
                    f"or None, got {value!r}."
                )


@dataclass(frozen=True)
class PredictionResult:
    """A probability result with an explicit spatial-space declaration."""

    probability: torch.Tensor
    output_space: str
    spatial_trace: SpatialTrace
    mask: Optional[torch.Tensor] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    resources: Optional[TimingResourceRecord] = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.output_space, str)
            or self.output_space not in SUPPORTED_OUTPUT_SPACES
        ):
            raise InvalidPredictionError(
                "PredictionResult.output_space must be one of "
                f"{sorted(SUPPORTED_OUTPUT_SPACES)}, got {self.output_space!r}."
            )
        _validate_probability_tensor(self.probability, field_name="PredictionResult.probability")
        if self.probability.ndim != 5:
            raise InvalidPredictionError(
                "PredictionResult.probability must represent a 3D case using "
                f"[B,C,D,H,W], got shape {tuple(self.probability.shape)}."
            )
        if not isinstance(self.spatial_trace, SpatialTrace):
            raise SpatialRestorationError(
                "PredictionResult.spatial_trace must be a SpatialTrace."
            )
        geometry = (
            self.spatial_trace.original
            if self.output_space == "native_input"
            else self.spatial_trace.model
        )
        observed_shape = tuple(int(value) for value in self.probability.shape[2:])
        if observed_shape != geometry.shape:
            raise SpatialRestorationError(
                f"PredictionResult probability shape {observed_shape} does not match "
                f"the declared {self.output_space} geometry shape {geometry.shape}."
            )
        if self.mask is not None:
            if not torch.is_tensor(self.mask):
                raise InvalidPredictionError(
                    "PredictionResult.mask must be a torch.Tensor when provided."
                )
            if tuple(self.mask.shape) != tuple(self.probability.shape):
                raise InvalidPredictionError(
                    f"PredictionResult.mask shape {tuple(self.mask.shape)} must match "
                    f"probability shape {tuple(self.probability.shape)}."
                )
            if torch.is_floating_point(self.mask) or torch.is_complex(self.mask):
                raise InvalidPredictionError(
                    "PredictionResult.mask must use an integer or boolean dtype."
                )
            if not bool(torch.all((self.mask == 0) | (self.mask == 1)).item()):
                raise InvalidPredictionError(
                    "PredictionResult.mask must contain only binary values {0, 1}."
                )
        if self.resources is not None and not isinstance(
            self.resources, TimingResourceRecord
        ):
            raise ResourceLimitError(
                "PredictionResult.resources must be TimingResourceRecord or None."
            )
        if not isinstance(self.provenance, Mapping):
            raise InvalidPredictionError("PredictionResult.provenance must be a mapping.")
        if not isinstance(self.diagnostics, Mapping):
            raise InvalidPredictionError("PredictionResult.diagnostics must be a mapping.")


def _validate_probability_tensor(
    probability: torch.Tensor,
    *,
    field_name: str,
) -> torch.Tensor:
    if not torch.is_tensor(probability):
        raise InvalidPredictionError(f"{field_name} must be a torch.Tensor.")
    if probability.numel() == 0:
        raise InvalidPredictionError(f"{field_name} must not be empty.")
    if not torch.is_floating_point(probability):
        raise InvalidPredictionError(f"{field_name} must use a floating-point dtype.")
    if not bool(torch.isfinite(probability).all().item()):
        raise InvalidPredictionError(f"{field_name} must contain only finite values.")
    minimum = float(probability.detach().amin().item())
    maximum = float(probability.detach().amax().item())
    if minimum < 0.0 or maximum > 1.0:
        raise InvalidPredictionError(
            f"{field_name} violates the probability contract [0, 1]: "
            f"observed min={minimum}, max={maximum}."
        )
    return probability
