"""Exactly invertible model-grid test-time augmentation views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

from src.inference.contracts import InvalidPredictionError
from src.inference.policy import SUPPORTED_TTA_FLIP_AXES, TtaPolicy


@dataclass(frozen=True)
class TtaView:
    """One exactly invertible spatial view on a model-preprocessed grid."""

    name: str
    spatial_dims: int
    flip_axis: str | None = None

    @property
    def tensor_dimension(self) -> int | None:
        if self.flip_axis is None:
            return None
        return _axis_tensor_dimension(self.flip_axis, spatial_dims=self.spatial_dims)

    def apply(self, image: Tensor) -> Tensor:
        """Create this view without changing the separately held spatial trace."""

        return _apply_view(image, self)

    def invert(self, probability: Tensor) -> Tensor:
        """Return a view-aligned probability to the original model-grid orientation."""

        return _apply_view(probability, self)


def build_tta_views(policy: TtaPolicy, *, spatial_dims: int) -> Tuple[TtaView, ...]:
    """Return identity plus every configured independent single-axis flip."""

    if not isinstance(policy, TtaPolicy):
        raise TypeError("policy must be a TtaPolicy.")
    if spatial_dims not in (2, 3):
        raise InvalidPredictionError(
            f"TTA requires a 2D or 3D predictor, got spatial_dims={spatial_dims}."
        )

    views = [TtaView(name="identity", spatial_dims=spatial_dims)]
    if policy.enabled:
        views.extend(
            TtaView(
                name=f"flip_{axis}",
                spatial_dims=spatial_dims,
                flip_axis=axis,
            )
            for axis in policy.flip_axes
        )
    return tuple(views)


def tta_view_names(policy: TtaPolicy) -> Tuple[str, ...]:
    """Return deterministic view names without configuring a separate count."""

    if not isinstance(policy, TtaPolicy):
        raise TypeError("policy must be a TtaPolicy.")
    if not policy.enabled:
        return ("identity",)
    return ("identity", *(f"flip_{axis}" for axis in policy.flip_axes))


def _apply_view(value: Tensor, view: TtaView) -> Tensor:
    if not torch.is_tensor(value):
        raise InvalidPredictionError("TTA inputs and probabilities must be tensors.")
    expected_ndim = view.spatial_dims + 2
    if value.ndim != expected_ndim:
        raise InvalidPredictionError(
            "TTA tensors must use [B,C,*spatial] layout; "
            f"expected {expected_ndim} dimensions, got shape {tuple(value.shape)}."
        )

    tensor = _plain_tensor(value)
    dimension = view.tensor_dimension
    if dimension is None:
        return tensor
    return torch.flip(tensor, dims=(dimension,))


def _plain_tensor(value: Tensor) -> Tensor:
    """Discard tensor-subclass metadata; geometry is owned by SpatialTrace."""

    as_tensor = getattr(value, "as_tensor", None)
    if callable(as_tensor):
        plain = as_tensor()
        if torch.is_tensor(plain):
            return plain
    return value


def _axis_tensor_dimension(axis: str, *, spatial_dims: int) -> int:
    # MONAI/NIfTI model grids retain voxel-axis order after canonical RAS
    # orientation: [B,C,X,Y] or [B,C,X,Y,Z]. Do not reinterpret those axes
    # using PyTorch Conv3d's generic D/H/W labels.
    available = {"x": -2, "y": -1}
    if spatial_dims == 3:
        available = {"x": -3, "y": -2, "z": -1}
    try:
        return available[axis]
    except KeyError as exc:
        raise InvalidPredictionError(
            f"TTA axis {axis!r} is not available for a {spatial_dims}D tensor."
        ) from exc


__all__ = [
    "SUPPORTED_TTA_FLIP_AXES",
    "TtaView",
    "build_tta_views",
    "tta_view_names",
]
