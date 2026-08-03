"""NIfTI transport validation around the shared native-space writer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import nibabel as nib
import numpy as np

from scripts.gc_submission_builder.runtime.interfaces import OutputBinding
from src.inference.contracts import InvalidPredictionError, PredictionResult
from src.inference.output import write_native_prediction_mask


class ImageTransportError(RuntimeError):
    """Raised when a platform image cannot satisfy the NIfTI transport contract."""


@dataclass(frozen=True)
class NiftiInputInspection:
    shape: tuple[int, int, int]
    dtype: str
    spacing: tuple[float, float, float]
    orientation: str


def inspect_nifti_input(path: str | Path) -> NiftiInputInspection:
    """Validate one `.nii.gz` header without duplicating preprocessing I/O."""

    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file() or not input_path.name.lower().endswith(".nii.gz"):
        raise ImageTransportError("Input image must be one existing .nii.gz file.")
    try:
        image = nib.load(str(input_path))
    except Exception as exc:
        raise ImageTransportError("Input NIfTI could not be opened.") from exc
    if len(image.shape) != 3:
        raise ImageTransportError(
            f"Input NIfTI must be 3D, got shape={tuple(image.shape)}."
        )
    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ImageTransportError("Input NIfTI affine must be finite and 4x4.")
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(affine))
    if any(not np.isfinite(value) or value <= 0 for value in spacing):
        raise ImageTransportError("Input NIfTI spacing must be finite and positive.")
    return NiftiInputInspection(
        shape=tuple(int(value) for value in image.shape),
        dtype=str(image.get_data_dtype()),
        spacing=spacing,
        orientation="".join(str(value) for value in nib.aff2axcodes(affine)),
    )


def write_nifti_prediction(
    result: PredictionResult,
    *,
    output_root: str | Path,
    binding: OutputBinding,
) -> Mapping[str, Any]:
    """Materialize one GC NIfTI output through the certified native writer."""

    if not isinstance(binding, OutputBinding):
        raise ImageTransportError("binding must be an OutputBinding.")
    if binding.file_type != "nifti":
        raise ImageTransportError("Current GC output transport supports NIfTI only.")
    if result.output_space != "native_input":
        raise ImageTransportError(
            "Production Grand Challenge output requires output_space='native_input'."
        )
    root = Path(output_root).expanduser().resolve()
    relative = _safe_relative_path(binding.relative_path)
    directory = (root / Path(*relative.parts)).resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise ImageTransportError("Output binding escapes the configured /output root.") from exc
    output_path = directory / "output.nii.gz"
    try:
        return write_native_prediction_mask(result, output_path)
    except InvalidPredictionError as exc:
        raise ImageTransportError(f"Native output validation failed: {exc}") from exc


def _safe_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip() or "\\" in value:
        raise ImageTransportError("Output binding must be a safe POSIX relative path.")
    path = PurePosixPath(value.strip())
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ImageTransportError("Output binding must be a safe relative path.")
    return path


__all__ = [
    "ImageTransportError",
    "NiftiInputInspection",
    "inspect_nifti_input",
    "write_nifti_prediction",
]
