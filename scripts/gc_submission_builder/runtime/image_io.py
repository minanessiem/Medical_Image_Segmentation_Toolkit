"""Medical-image transport around one shared native-space prediction result."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from scripts.gc_submission_builder.runtime.interfaces import OutputBinding
from src.inference.contracts import InvalidPredictionError, PredictionResult
from src.inference.output import write_native_prediction_mask


class ImageTransportError(RuntimeError):
    """Raised when a platform image cannot satisfy its transport contract."""


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
        raise ImageTransportError("NIfTI writer requires file_type='nifti'.")
    if binding.result_key != "mask":
        raise ImageTransportError(
            "The current NIfTI transport supports the binary mask result only; "
            "probability NIfTI output is not implemented."
        )
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


def write_mha_prediction(
    result: PredictionResult,
    *,
    output_root: str | Path,
    binding: OutputBinding,
) -> Mapping[str, Any]:
    """Write and reopen one compressed MHA result on the native input grid."""

    if not isinstance(binding, OutputBinding):
        raise ImageTransportError("binding must be an OutputBinding.")
    if binding.file_type != "mha":
        raise ImageTransportError("MHA writer requires file_type='mha'.")
    if result.output_space != "native_input" or result.native_reference is None:
        raise ImageTransportError(
            "Production Grand Challenge output requires output_space='native_input' "
            "with native reference geometry."
        )
    array = _result_array(result, binding.result_key)
    spacing, origin, direction = _simpleitk_geometry(result)
    directory = _output_directory(output_root, binding.relative_path)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / "output.mha"
    image = sitk.GetImageFromArray(array.transpose(2, 1, 0))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    try:
        sitk.WriteImage(image, str(output_path), useCompression=True)
    except Exception as exc:
        raise ImageTransportError("Could not write compressed MHA output.") from exc
    return _inspect_mha_prediction(
        output_path,
        result=result,
        result_key=binding.result_key,
    )


def materialize_prediction_outputs(
    result: PredictionResult,
    *,
    output_root: str | Path,
    bindings: Sequence[OutputBinding],
) -> Mapping[str, Mapping[str, Any]]:
    """Materialize one complete declared output set or leave no declared output."""

    if result.output_space != "native_input":
        raise ImageTransportError(
            "Production Grand Challenge output requires output_space='native_input'."
        )
    resolved_bindings = tuple(bindings)
    if not resolved_bindings:
        raise ImageTransportError("At least one output binding is required.")
    if any(not isinstance(binding, OutputBinding) for binding in resolved_bindings):
        raise ImageTransportError("Every output binding must be an OutputBinding.")

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    final_directories = {
        binding.slug: _output_directory(root, binding.relative_path)
        for binding in resolved_bindings
    }
    _clear_declared_outputs(final_directories.values())
    staging = Path(tempfile.mkdtemp(prefix=".gc-staging-", dir=root))
    try:
        for binding in resolved_bindings:
            _write_prediction_output(
                result,
                output_root=staging,
                binding=binding,
            )
        for binding in resolved_bindings:
            source = _output_directory(staging, binding.relative_path)
            destination = final_directories[binding.slug]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))

        validations: dict[str, Mapping[str, Any]] = {}
        for binding in resolved_bindings:
            validations[binding.slug] = _inspect_materialized_prediction(
                final_directories[binding.slug],
                result=result,
                binding=binding,
            )
        return validations
    except Exception:
        _clear_declared_outputs(final_directories.values())
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _write_prediction_output(
    result: PredictionResult,
    *,
    output_root: Path,
    binding: OutputBinding,
) -> Mapping[str, Any]:
    if binding.file_type == "nifti":
        return write_nifti_prediction(
            result,
            output_root=output_root,
            binding=binding,
        )
    if binding.file_type == "mha":
        return write_mha_prediction(
            result,
            output_root=output_root,
            binding=binding,
        )
    raise ImageTransportError(
        f"No output materializer is registered for file_type={binding.file_type!r}."
    )


def _inspect_materialized_prediction(
    directory: Path,
    *,
    result: PredictionResult,
    binding: OutputBinding,
) -> Mapping[str, Any]:
    if binding.file_type == "mha":
        return _inspect_mha_prediction(
            directory / "output.mha",
            result=result,
            result_key=binding.result_key,
        )
    path = directory / "output.nii.gz"
    try:
        image = nib.load(str(path))
        array = np.asarray(image.dataobj)
    except Exception as exc:
        raise ImageTransportError("Could not reopen materialized NIfTI output.") from exc
    expected = _result_array(result, binding.result_key)
    if tuple(image.shape) != tuple(expected.shape) or not np.array_equal(array, expected):
        raise ImageTransportError(
            "Materialized NIfTI output does not match the declared prediction result."
        )
    return {
        "path": str(path),
        "result_key": binding.result_key,
        "file_type": "nifti",
        "shape": list(image.shape),
        "dtype": str(image.get_data_dtype()),
        "spatial_validation": "passed",
    }


def _inspect_mha_prediction(
    path: Path,
    *,
    result: PredictionResult,
    result_key: str,
) -> Mapping[str, Any]:
    try:
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
    except Exception as exc:
        raise ImageTransportError("Could not reopen materialized MHA output.") from exc
    expected = _result_array(result, result_key)
    spacing, origin, direction = _simpleitk_geometry(result)
    if image.GetDimension() != 3 or tuple(image.GetSize()) != tuple(expected.shape):
        raise ImageTransportError("MHA output size does not match native input geometry.")
    for name, observed, reference in (
        ("spacing", image.GetSpacing(), spacing),
        ("origin", image.GetOrigin(), origin),
        ("direction", image.GetDirection(), direction),
    ):
        if not np.allclose(observed, reference, rtol=0, atol=1e-5):
            raise ImageTransportError(
                f"MHA output {name} does not match native input geometry."
            )
    if result_key == "mask":
        if array.dtype != np.dtype(np.uint8) or not set(np.unique(array)).issubset(
            {0, 1}
        ):
            raise ImageTransportError("MHA mask must be binary uint8.")
        values_match = np.array_equal(array, expected)
    else:
        if array.dtype != np.dtype(np.float32):
            raise ImageTransportError("MHA probability must be float32.")
        if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise ImageTransportError(
                "MHA probability must be finite and constrained to [0, 1]."
            )
        values_match = np.allclose(array, expected, rtol=0, atol=1e-7)
    if not values_match:
        raise ImageTransportError(
            f"MHA {result_key} values changed during materialization."
        )
    try:
        with path.open("rb") as handle:
            header = handle.read(65536).split(b"ElementDataFile", 1)[0]
    except OSError as exc:
        raise ImageTransportError("Could not inspect MHA compression header.") from exc
    if b"CompressedData = True" not in header:
        raise ImageTransportError("MHA output must use embedded compression.")
    return {
        "path": str(path),
        "result_key": result_key,
        "file_type": "mha",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "spacing": list(image.GetSpacing()),
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
        "spatial_validation": "passed",
        "compressed": True,
    }


def _result_array(result: PredictionResult, result_key: str) -> np.ndarray:
    if result_key == "mask":
        tensor = result.mask
        dtype = np.uint8
    elif result_key == "probability":
        tensor = result.probability
        dtype = np.float32
    else:
        raise ImageTransportError(f"Unknown prediction result_key={result_key!r}.")
    if tensor.ndim != 5 or tuple(tensor.shape[:2]) != (1, 1):
        raise ImageTransportError(
            f"Prediction result {result_key!r} must have shape [1,1,X,Y,Z]."
        )
    array = tensor.detach().cpu().numpy()[0, 0].astype(dtype, copy=False)
    if result.native_reference is None or tuple(array.shape) != tuple(
        result.native_reference.shape
    ):
        raise ImageTransportError(
            f"Prediction result {result_key!r} does not match native geometry shape."
        )
    return array


def _simpleitk_geometry(
    result: PredictionResult,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if result.native_reference is None:
        raise ImageTransportError("Native reference geometry is required for MHA.")
    affine = np.asarray(result.native_reference.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ImageTransportError("Native affine must be finite and 4x4.")
    ras_to_lps = np.diag([-1.0, -1.0, 1.0])
    matrix = ras_to_lps @ affine[:3, :3]
    spacing = np.linalg.norm(matrix, axis=0)
    if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ImageTransportError("Native spacing must be finite and positive.")
    direction = matrix / spacing[np.newaxis, :]
    if not np.allclose(direction.T @ direction, np.eye(3), rtol=0, atol=1e-5):
        raise ImageTransportError(
            "Native affine contains shear and is not orthonormal after spacing "
            "decomposition; it cannot be represented losslessly as MHA."
        )
    determinant = float(np.linalg.det(direction))
    if not np.isfinite(determinant) or not np.isclose(
        abs(determinant), 1.0, rtol=0, atol=1e-5
    ):
        raise ImageTransportError(
            "Native affine direction is not losslessly representable as MHA."
        )
    origin = ras_to_lps @ affine[:3, 3]
    return (
        tuple(float(value) for value in spacing),
        tuple(float(value) for value in origin),
        tuple(float(value) for value in direction.reshape(-1)),
    )


def _output_directory(output_root: str | Path, relative_path: str) -> Path:
    root = Path(output_root).expanduser().resolve()
    relative = _safe_relative_path(relative_path)
    directory = (root / Path(*relative.parts)).resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise ImageTransportError(
            "Output binding escapes the configured /output root."
        ) from exc
    return directory


def _clear_declared_outputs(directories: Sequence[Path]) -> None:
    for directory in sorted(set(directories), key=lambda value: len(value.parts), reverse=True):
        if directory.exists():
            shutil.rmtree(directory)


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
    "materialize_prediction_outputs",
    "write_mha_prediction",
    "write_nifti_prediction",
]
