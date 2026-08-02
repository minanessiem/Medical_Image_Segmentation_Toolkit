"""
Shared contracts for the greenfield evaluation package.

These dataclasses are intentionally lightweight and focused on:
- sample payload boundaries between IO adapters and metric engine
- protocol settings for threshold evaluation
- explicit reporting scopes and running-stat semantics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
from torch import Tensor

from src.inference.contracts import SUPPORTED_OUTPUT_SPACES, SpatialGeometry

ScopeName = Literal["all_slices", "foreground_only"]


@dataclass
class SliceSample:
    """
    One evaluation sample yielded by a streaming producer.

    Exactly one of `prediction_prob` or `prediction_mask` should be provided.
    """

    case_id: str
    slice_id: str
    ground_truth_mask: Tensor
    prediction_prob: Optional[Tensor] = None
    prediction_mask: Optional[Tensor] = None
    volume_id: Optional[str] = None
    slice_index: Optional[int] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate basic sample invariants."""
        if self.prediction_prob is None and self.prediction_mask is None:
            raise ValueError(
                "SliceSample must provide either prediction_prob or prediction_mask."
            )
        if self.prediction_prob is not None and self.prediction_mask is not None:
            raise ValueError(
                "SliceSample must not provide both prediction_prob and prediction_mask."
            )
        if self.volume_id is None and self.slice_index is not None:
            raise ValueError(
                "SliceSample with slice_index must also provide volume_id."
            )
        if self.volume_id is not None and self.slice_index is None:
            raise ValueError(
                "SliceSample with volume_id must also provide slice_index."
            )
        if self.volume_id is not None and not self.volume_id.strip():
            raise ValueError("SliceSample volume_id must not be empty.")
        if self.slice_index is not None and int(self.slice_index) < 0:
            raise ValueError("SliceSample slice_index must be >= 0.")


@dataclass(frozen=True)
class ThresholdProtocol:
    """Protocol settings used by evaluation orchestration."""

    mode: Literal["fixed", "sweep"]
    thresholds: List[float]
    optimize_metric: Optional[str] = None


@dataclass(frozen=True)
class PrimaryMetricSelector:
    """Config-driven primary metric selector for threshold choice."""

    level: Literal["slice", "volume"]
    metric: str
    statistic: Literal["mean", "median"]
    direction: Literal["max", "min"]


@dataclass(frozen=True)
class EvaluationThresholdProtocol:
    """Extended threshold protocol for model evaluation workflows."""

    mode: Literal["fixed", "sweep", "oracle_per_case", "sweep_with_oracle"]
    thresholds: List[float]
    fixed_threshold: float
    primary: PrimaryMetricSelector


@dataclass
class RunningStats:
    """
    Numerically stable-enough running statistics for mean/std.

    Tracks values using sum and sum-of-squares to avoid storing samples.
    """

    count: int = 0
    value_sum: float = 0.0
    value_sum_sq: float = 0.0

    def update(self, value: float) -> None:
        """Update with one scalar value."""
        self.count += 1
        self.value_sum += value
        self.value_sum_sq += value * value

    @property
    def mean(self) -> float:
        """Return running mean."""
        if self.count == 0:
            return 0.0
        return self.value_sum / self.count

    @property
    def std(self) -> float:
        """Return population standard deviation."""
        if self.count == 0:
            return 0.0
        mean = self.mean
        variance = (self.value_sum_sq / self.count) - (mean * mean)
        if variance < 0:
            variance = 0.0
        return variance ** 0.5

    def to_dict(self) -> Dict[str, float]:
        """Serialize to a mean/std/count dictionary."""
        return {
            "mean": float(self.mean),
            "std": float(self.std),
            "count": int(self.count),
        }


@dataclass
class ScopedRunningStats:
    """Holds running stats for both denominator scopes."""

    all_slices: RunningStats = field(default_factory=RunningStats)
    foreground_only: RunningStats = field(default_factory=RunningStats)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Serialize to scope-keyed dictionary."""
        return {
            "all_slices": self.all_slices.to_dict(),
            "foreground_only": self.foreground_only.to_dict(),
        }


@dataclass
class VolumeSample:
    """
    One spatially declared 3D prediction/reference pair.

    Tensors are expected in channel-first format [C, H, W, D].
    """

    case_id: str
    volume_id: str
    prediction_volume: Tensor
    ground_truth_volume: Tensor
    prediction_space: str
    reference_space: str
    prediction_geometry: SpatialGeometry
    reference_geometry: SpatialGeometry
    metadata: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate basic volume sample invariants."""
        if not self.volume_id.strip():
            raise ValueError("VolumeSample volume_id must not be empty.")
        if self.prediction_volume.ndim != 4:
            raise ValueError(
                "VolumeSample prediction_volume must be 4D [C,H,W,D], "
                f"got shape={tuple(self.prediction_volume.shape)}."
            )
        if self.ground_truth_volume.ndim != 4:
            raise ValueError(
                "VolumeSample ground_truth_volume must be 4D [C,H,W,D], "
                f"got shape={tuple(self.ground_truth_volume.shape)}."
            )
        if tuple(self.prediction_volume.shape) != tuple(self.ground_truth_volume.shape):
            raise ValueError(
                "VolumeSample prediction and ground truth shape mismatch: "
                f"pred={tuple(self.prediction_volume.shape)} "
                f"gt={tuple(self.ground_truth_volume.shape)}."
            )
        if self.prediction_space not in SUPPORTED_OUTPUT_SPACES:
            raise ValueError(
                "VolumeSample prediction_space must be one of "
                f"{sorted(SUPPORTED_OUTPUT_SPACES)}, got {self.prediction_space!r}."
            )
        if self.reference_space not in SUPPORTED_OUTPUT_SPACES:
            raise ValueError(
                "VolumeSample reference_space must be one of "
                f"{sorted(SUPPORTED_OUTPUT_SPACES)}, got {self.reference_space!r}."
            )
        if not isinstance(self.prediction_geometry, SpatialGeometry):
            raise ValueError(
                "VolumeSample prediction_geometry must be a SpatialGeometry; "
                "geometry is mandatory for 3D evaluation."
            )
        if not isinstance(self.reference_geometry, SpatialGeometry):
            raise ValueError(
                "VolumeSample reference_geometry must be a SpatialGeometry; "
                "geometry is mandatory for 3D evaluation."
            )
        if self.prediction_space != self.reference_space:
            raise ValueError(
                "VolumeSample prediction/reference space mismatch: "
                f"prediction={self.prediction_space!r}, "
                f"reference={self.reference_space!r}."
            )

        prediction_shape = tuple(
            int(value) for value in self.prediction_volume.shape[-3:]
        )
        reference_shape = tuple(
            int(value) for value in self.ground_truth_volume.shape[-3:]
        )
        if self.prediction_geometry.shape != prediction_shape:
            raise ValueError(
                "VolumeSample prediction tensor/geometry shape mismatch: "
                f"tensor={prediction_shape}, geometry={self.prediction_geometry.shape}."
            )
        if self.reference_geometry.shape != reference_shape:
            raise ValueError(
                "VolumeSample reference tensor/geometry shape mismatch: "
                f"tensor={reference_shape}, geometry={self.reference_geometry.shape}."
            )
        if self.prediction_geometry.shape != self.reference_geometry.shape:
            raise ValueError(
                "VolumeSample prediction/reference geometry shape mismatch: "
                f"prediction={self.prediction_geometry.shape}, "
                f"reference={self.reference_geometry.shape}."
            )
        if not np.allclose(
            np.asarray(self.prediction_geometry.affine, dtype=np.float64),
            np.asarray(self.reference_geometry.affine, dtype=np.float64),
            rtol=1.0e-5,
            atol=1.0e-5,
        ):
            raise ValueError(
                "VolumeSample prediction/reference affine mismatch: equal tensor "
                "shapes do not establish a shared physical grid."
            )
        if not np.allclose(
            self.prediction_geometry.spacing,
            self.reference_geometry.spacing,
            rtol=1.0e-5,
            atol=1.0e-5,
        ):
            raise ValueError(
                "VolumeSample prediction/reference spacing mismatch: "
                f"prediction={self.prediction_geometry.spacing}, "
                f"reference={self.reference_geometry.spacing}."
            )
        if self.prediction_geometry.orientation != self.reference_geometry.orientation:
            raise ValueError(
                "VolumeSample prediction/reference orientation mismatch: "
                f"prediction={self.prediction_geometry.orientation!r}, "
                f"reference={self.reference_geometry.orientation!r}."
            )
