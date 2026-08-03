"""Validated NIfTI materialization for native-space binary predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import nibabel as nib
import numpy as np

from src.inference.contracts import InvalidPredictionError, PredictionResult
from src.inference.spatial import validate_output_geometry


def write_native_prediction_mask(
    result: PredictionResult,
    output_path: str | Path,
) -> Mapping[str, Any]:
    """Atomically write and re-open one native-grid uint8 NIfTI segmentation."""
    if not isinstance(result, PredictionResult):
        raise InvalidPredictionError("result must be a PredictionResult.")
    if result.output_space != "native_input":
        raise InvalidPredictionError(
            "Native NIfTI output requires output_space='native_input'; refusing to "
            "write a model_preprocessed mask as a native prediction."
        )
    if result.mask is None:
        raise InvalidPredictionError(
            "PredictionResult.mask is required before native NIfTI materialization."
        )
    if result.native_reference is None:
        raise InvalidPredictionError(
            "PredictionResult.native_reference is required for native NIfTI output."
        )
    if tuple(result.mask.shape[:2]) != (1, 1):
        raise InvalidPredictionError(
            "Native NIfTI output requires a single-case, single-channel mask in "
            f"[1,1,D,H,W] layout, got {tuple(result.mask.shape)}."
        )

    path = Path(output_path)
    if not str(path).endswith(".nii.gz"):
        raise InvalidPredictionError(
            f"Native segmentation output path must end with '.nii.gz', got {path}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.nii.gz")

    metadata = result.native_reference
    mask = result.mask[0, 0].detach().cpu().numpy().astype(np.uint8, copy=False)
    image = nib.Nifti1Image(mask, affine=np.asarray(metadata.affine, dtype=np.float64))
    image.set_data_dtype(np.uint8)
    image.set_qform(
        None if metadata.qform is None else np.asarray(metadata.qform, dtype=np.float64),
        code=int(metadata.qform_code),
    )
    image.set_sform(
        None if metadata.sform is None else np.asarray(metadata.sform, dtype=np.float64),
        code=int(metadata.sform_code),
    )

    try:
        nib.save(image, str(temporary_path))
        validation = _validate_written_native_mask(temporary_path, result)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return {**validation, "path": str(path)}


def _validate_written_native_mask(
    path: Path,
    result: PredictionResult,
) -> Mapping[str, Any]:
    metadata = result.native_reference
    assert metadata is not None
    try:
        reopened = nib.load(str(path))
        data = np.asarray(reopened.dataobj)
    except Exception as exc:
        raise InvalidPredictionError(
            f"Written native segmentation could not be re-opened: {exc}"
        ) from exc

    validate_output_geometry(
        observed_shape=tuple(int(value) for value in reopened.shape),
        observed_affine=np.asarray(reopened.affine, dtype=np.float64),
        expected=result.spatial_trace.original,
    )
    if reopened.get_data_dtype() != np.dtype(np.uint8):
        raise InvalidPredictionError(
            f"Written segmentation dtype must be uint8, got {reopened.get_data_dtype()}."
        )
    values = sorted(int(value) for value in np.unique(data))
    if any(value not in (0, 1) for value in values):
        raise InvalidPredictionError(
            f"Written segmentation contains non-binary values: {values}."
        )

    qform, qform_code = reopened.get_qform(coded=True)
    sform, sform_code = reopened.get_sform(coded=True)
    _validate_form("qform", qform, int(qform_code), metadata.qform, metadata.qform_code)
    _validate_form("sform", sform, int(sform_code), metadata.sform, metadata.sform_code)
    return {
        "path": str(path),
        "shape": list(reopened.shape),
        "dtype": "uint8",
        "allowed_values": values,
        "qform_code": int(qform_code),
        "sform_code": int(sform_code),
        "spatial_validation": "passed",
    }


def _validate_form(
    name: str,
    observed: np.ndarray | None,
    observed_code: int,
    expected: object,
    expected_code: int,
) -> None:
    if observed_code != int(expected_code):
        raise InvalidPredictionError(
            f"Written segmentation {name} code {observed_code} does not match "
            f"native reference code {expected_code}."
        )
    if expected is None:
        if observed is not None:
            raise InvalidPredictionError(
                f"Written segmentation unexpectedly contains a coded {name}."
            )
        return
    if observed is None or not np.allclose(
        np.asarray(observed, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=0,
        atol=1e-4,
    ):
        raise InvalidPredictionError(
            f"Written segmentation {name} does not match the native reference."
        )


__all__ = ["write_native_prediction_mask"]
