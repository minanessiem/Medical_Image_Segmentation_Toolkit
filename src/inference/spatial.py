"""Architecture-neutral restoration of 3D probability maps to native grids."""

from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import SpatialResample

from src.inference.contracts import (
    SpatialGeometry,
    SpatialRestorationError,
    SpatialTrace,
    _validate_probability_tensor,
)


SUPPORTED_SPATIAL_OPERATIONS = frozenset(
    {
        "Orientation",
        "SpatialResample",
        "SpatialPad",
    }
)
WORLD_COORDINATE_TOLERANCE_MM = 1e-4


def restore_probability_to_native(
    probability: torch.Tensor,
    spatial_trace: SpatialTrace,
) -> torch.Tensor:
    """Continuously resample one model-grid probability map onto its native grid."""
    _validate_probability_tensor(probability, field_name="model probability")
    if probability.ndim != 5 or int(probability.shape[0]) != 1:
        raise SpatialRestorationError(
            "Native restoration requires one 3D case in [1,C,D,H,W] layout, "
            f"got {tuple(probability.shape)}."
        )
    if not isinstance(spatial_trace, SpatialTrace):
        raise SpatialRestorationError("spatial_trace must be a SpatialTrace.")
    observed_shape = tuple(int(value) for value in probability.shape[2:])
    if observed_shape != spatial_trace.model.shape:
        raise SpatialRestorationError(
            f"Probability shape {observed_shape} does not match the recorded model grid "
            f"{spatial_trace.model.shape}."
        )

    geometry_changed = spatial_trace.model != spatial_trace.original
    _validate_transform_history(spatial_trace, require_spatial_step=geometry_changed)
    probability_tensor = (
        probability.as_tensor() if isinstance(probability, MetaTensor) else probability
    )
    if not geometry_changed:
        validate_output_geometry(
            observed_shape=observed_shape,
            observed_affine=np.asarray(spatial_trace.model.affine, dtype=np.float64),
            expected=spatial_trace.original,
        )
        return probability_tensor.detach().to(
            device="cpu",
            dtype=torch.float32,
        ).clone()

    source = MetaTensor(
        probability_tensor.detach().to(device="cpu", dtype=torch.float32)[0],
        affine=torch.tensor(spatial_trace.model.affine, dtype=torch.float64),
    )
    resampler = SpatialResample(
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
        dtype=np.float64,
    )
    try:
        restored = resampler(
            source,
            dst_affine=torch.tensor(spatial_trace.original.affine, dtype=torch.float64),
            spatial_size=spatial_trace.original.shape,
        )
    except Exception as exc:
        raise SpatialRestorationError(
            "Continuous probability restoration from the model grid to the native "
            f"grid failed: {exc}"
        ) from exc

    restored_affine = getattr(restored, "affine", None)
    if restored_affine is None:
        raise SpatialRestorationError(
            "Spatial restoration discarded the output affine."
        )
    restored_tensor = torch.as_tensor(restored).detach().to(dtype=torch.float32).unsqueeze(0)
    _validate_probability_tensor(restored_tensor, field_name="restored native probability")
    validate_output_geometry(
        observed_shape=tuple(int(value) for value in restored_tensor.shape[2:]),
        observed_affine=np.asarray(
            torch.as_tensor(restored_affine).detach().cpu(),
            dtype=np.float64,
        ),
        expected=spatial_trace.original,
    )
    return restored_tensor


def threshold_probability(probability: torch.Tensor, threshold: float) -> torch.Tensor:
    """Threshold a validated probability tensor into an exact uint8 binary mask."""
    _validate_probability_tensor(probability, field_name="probability to threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise SpatialRestorationError("Probability threshold must be numeric.")
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise SpatialRestorationError(
            f"Probability threshold must be within [0, 1], got {threshold}."
        )
    return (probability >= threshold).to(dtype=torch.uint8)


def validate_output_geometry(
    *,
    observed_shape: tuple[int, int, int],
    observed_affine: np.ndarray,
    expected: SpatialGeometry,
    tolerance_mm: float = WORLD_COORDINATE_TOLERANCE_MM,
) -> None:
    """Validate array shape, affine, and corner world coordinates."""
    if tuple(int(value) for value in observed_shape) != expected.shape:
        raise SpatialRestorationError(
            f"Restored shape {observed_shape} does not match native shape {expected.shape}."
        )
    observed_affine = np.asarray(observed_affine, dtype=np.float64)
    expected_affine = np.asarray(expected.affine, dtype=np.float64)
    if observed_affine.shape != (4, 4) or not np.allclose(
        observed_affine,
        expected_affine,
        rtol=0,
        atol=tolerance_mm,
    ):
        raise SpatialRestorationError(
            "Restored affine does not match the native reference affine within "
            f"{tolerance_mm} mm."
        )

    corners = np.asarray(tuple(_corner_indices(expected.shape)), dtype=np.float64)
    homogeneous = np.concatenate(
        [corners, np.ones((corners.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    observed_world = homogeneous @ observed_affine.T
    expected_world = homogeneous @ expected_affine.T
    if not np.allclose(
        observed_world[:, :3],
        expected_world[:, :3],
        rtol=0,
        atol=tolerance_mm,
    ):
        raise SpatialRestorationError(
            "Restored corner world coordinates do not match the native reference grid."
        )


def _validate_transform_history(
    spatial_trace: SpatialTrace,
    *,
    require_spatial_step: bool,
) -> None:
    applied_spatial_steps = 0
    for index, operation in enumerate(spatial_trace.transform_history):
        operation_class = operation.get("class")
        applied = bool(operation.get("do_transforms", True))
        if not applied:
            continue
        if not isinstance(operation_class, str) or not operation_class:
            raise SpatialRestorationError(
                f"Spatial transform history item {index} has no valid class name."
            )
        if operation_class not in SUPPORTED_SPATIAL_OPERATIONS:
            raise SpatialRestorationError(
                f"Spatial restoration does not support recorded operation "
                f"{operation_class!r}. Refusing to infer an inverse transform."
            )
        applied_spatial_steps += 1
    if require_spatial_step and applied_spatial_steps == 0:
        raise SpatialRestorationError(
            "Model and native grids differ, but the transform history contains no "
            "applied supported spatial operation."
        )


def _corner_indices(shape: tuple[int, int, int]) -> Iterable[tuple[int, int, int]]:
    return product(*((0, int(size) - 1) for size in shape))


__all__ = [
    "SUPPORTED_SPATIAL_OPERATIONS",
    "WORLD_COORDINATE_TOLERANCE_MM",
    "restore_probability_to_native",
    "threshold_probability",
    "validate_output_geometry",
]
