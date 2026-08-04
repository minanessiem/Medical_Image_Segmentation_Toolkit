"""Medical-image transport around one shared native-space prediction result."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import gzip
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from scripts.gc_submission_builder.runtime.interfaces import (
    OutputBinding,
    ResolvedImageInput,
)
from src.inference.contracts import InvalidPredictionError, PredictionResult
from src.inference.output import write_native_prediction_mask


class ImageTransportError(RuntimeError):
    """Raised when a platform image cannot satisfy its transport contract."""


@dataclass(frozen=True)
class MedicalImageInspection:
    source_format: str
    shape: tuple[int, int, int]
    dtype: str
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]
    orientation: str


NiftiInputInspection = MedicalImageInspection


@dataclass(frozen=True)
class CanonicalImageInput:
    dataset_key: str
    source_format: str
    canonical_path: Path
    source_size_bytes: int
    canonical_size_bytes: int
    converted: bool
    source_inspection: MedicalImageInspection
    canonical_inspection: MedicalImageInspection


@dataclass(frozen=True)
class CanonicalizedImageInputs:
    inputs: Mapping[str, CanonicalImageInput]
    scratch_free_before_bytes: int
    scratch_free_after_bytes: int

    @property
    def canonical_modalities(self) -> Mapping[str, Path]:
        return MappingProxyType(
            {
                key: value.canonical_path
                for key, value in self.inputs.items()
            }
        )


def inspect_nifti_input(path: str | Path) -> MedicalImageInspection:
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
    spacing, origin, direction = _decompose_ras_affine(affine)
    return MedicalImageInspection(
        source_format="nii_gz",
        shape=tuple(int(value) for value in image.shape),
        dtype=str(image.get_data_dtype()),
        spacing=spacing,
        origin=origin,
        direction=direction,
        orientation="".join(str(value) for value in nib.aff2axcodes(affine)),
    )


@contextmanager
def canonicalize_image_inputs(
    image_inputs: Mapping[str, ResolvedImageInput],
    *,
    scratch_root: str | Path = "/tmp",
) -> Iterator[CanonicalizedImageInputs]:
    """Yield canonical `.nii.gz` paths and remove invocation scratch on exit."""

    if not isinstance(image_inputs, Mapping) or not image_inputs:
        raise ImageTransportError("At least one resolved image input is required.")
    root = Path(scratch_root).expanduser().resolve()
    if not root.is_dir():
        raise ImageTransportError("Writable invocation scratch directory does not exist.")
    try:
        free_before = int(shutil.disk_usage(root).free)
    except OSError as exc:
        raise ImageTransportError("Could not inspect invocation scratch capacity.") from exc

    with tempfile.TemporaryDirectory(prefix="gc-input-", dir=root) as temporary:
        temporary_root = Path(temporary)
        canonicalized: dict[str, CanonicalImageInput] = {}
        for dataset_key, resolved in image_inputs.items():
            if not isinstance(resolved, ResolvedImageInput):
                raise ImageTransportError(
                    "Resolved input values must be ResolvedImageInput instances."
                )
            if dataset_key != resolved.dataset_key:
                raise ImageTransportError(
                    "Resolved image mapping key disagrees with its canonical dataset key."
                )
            canonicalized[dataset_key] = _canonicalize_image_input(
                resolved,
                output_path=temporary_root / f"modality-{len(canonicalized)}.nii.gz",
                scratch_root=root,
            )
        try:
            free_after = int(shutil.disk_usage(root).free)
        except OSError as exc:
            raise ImageTransportError(
                "Could not inspect scratch capacity after input canonicalization."
            ) from exc
        yield CanonicalizedImageInputs(
            inputs=MappingProxyType(canonicalized),
            scratch_free_before_bytes=free_before,
            scratch_free_after_bytes=free_after,
        )


def _canonicalize_image_input(
    resolved: ResolvedImageInput,
    *,
    output_path: Path,
    scratch_root: Path,
) -> CanonicalImageInput:
    source_path = resolved.source_path
    source_format = resolved.source_format
    if source_format == "nii_gz":
        inspection = inspect_nifti_input(source_path)
        return CanonicalImageInput(
            dataset_key=resolved.dataset_key,
            source_format=source_format,
            canonical_path=source_path,
            source_size_bytes=resolved.source_size_bytes,
            canonical_size_bytes=resolved.source_size_bytes,
            converted=False,
            source_inspection=inspection,
            canonical_inspection=inspection,
        )
    if source_format == "nii":
        source_inspection = _inspect_nifti(source_path, source_format="nii")
        _require_scratch_capacity(
            scratch_root,
            required_bytes=max(resolved.source_size_bytes, 1),
        )
        try:
            with source_path.open("rb") as source, output_path.open("wb") as raw:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw,
                    compresslevel=6,
                    mtime=0,
                ) as compressed:
                    shutil.copyfileobj(source, compressed, length=1024 * 1024)
        except OSError as exc:
            raise ImageTransportError("Could not gzip the selected NIfTI input.") from exc
        canonical_inspection = inspect_nifti_input(output_path)
        _validate_inspection_equivalence(source_inspection, canonical_inspection)
    elif source_format in {"mha", "tif", "tiff"}:
        source_image = _read_scalar_3d_sitk(source_path, source_format=source_format)
        source_inspection = _inspect_sitk_image(
            source_image,
            source_format=source_format,
        )
        voxel_bytes = int(sitk.GetArrayViewFromImage(source_image).nbytes)
        _require_scratch_capacity(
            scratch_root,
            required_bytes=max(resolved.source_size_bytes, voxel_bytes),
        )
        try:
            sitk.WriteImage(source_image, str(output_path), useCompression=True)
            canonical_image = sitk.ReadImage(str(output_path))
        except Exception as exc:
            raise ImageTransportError(
                f"Could not canonicalize the selected {source_format} image to NIfTI."
            ) from exc
        _validate_sitk_equivalence(source_image, canonical_image)
        try:
            values_match = sitk.Hash(source_image) == sitk.Hash(canonical_image)
        except Exception as exc:
            raise ImageTransportError(
                "Could not verify voxel preservation during image canonicalization."
            ) from exc
        if not values_match:
            raise ImageTransportError(
                "Voxel values changed during image canonicalization."
            )
        canonical_inspection = inspect_nifti_input(output_path)
        _validate_inspection_equivalence(source_inspection, canonical_inspection)
    else:
        raise ImageTransportError(
            f"No canonicalizer is registered for source_format={source_format!r}."
        )

    try:
        canonical_size_bytes = int(output_path.stat().st_size)
    except OSError as exc:
        raise ImageTransportError("Could not stat the canonical NIfTI input.") from exc
    return CanonicalImageInput(
        dataset_key=resolved.dataset_key,
        source_format=source_format,
        canonical_path=output_path,
        source_size_bytes=resolved.source_size_bytes,
        canonical_size_bytes=canonical_size_bytes,
        converted=True,
        source_inspection=source_inspection,
        canonical_inspection=canonical_inspection,
    )


def _inspect_nifti(path: Path, *, source_format: str) -> MedicalImageInspection:
    try:
        image = nib.load(str(path))
    except Exception as exc:
        raise ImageTransportError("Input NIfTI could not be opened.") from exc
    if len(image.shape) != 3:
        raise ImageTransportError(
            f"Input NIfTI must be 3D, got shape={tuple(image.shape)}."
        )
    affine = np.asarray(image.affine, dtype=np.float64)
    spacing, origin, direction = _decompose_ras_affine(affine)
    return MedicalImageInspection(
        source_format=source_format,
        shape=tuple(int(value) for value in image.shape),
        dtype=str(image.get_data_dtype()),
        spacing=spacing,
        origin=origin,
        direction=direction,
        orientation="".join(str(value) for value in nib.aff2axcodes(affine)),
    )


def _read_scalar_3d_sitk(path: Path, *, source_format: str) -> sitk.Image:
    try:
        image = sitk.ReadImage(str(path))
    except Exception as exc:
        raise ImageTransportError(
            f"Input {source_format} image could not be opened."
        ) from exc
    if image.GetDimension() != 3:
        raise ImageTransportError(
            f"Input {source_format} image must be 3D, got dimension={image.GetDimension()}."
        )
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ImageTransportError(
            f"Input {source_format} image must be scalar, got "
            f"components={image.GetNumberOfComponentsPerPixel()}."
        )
    return image


def _inspect_sitk_image(
    image: sitk.Image,
    *,
    source_format: str,
) -> MedicalImageInspection:
    spacing = tuple(float(value) for value in image.GetSpacing())
    origin = tuple(float(value) for value in image.GetOrigin())
    direction = tuple(float(value) for value in image.GetDirection())
    if (
        len(spacing) != 3
        or any(not np.isfinite(value) or value <= 0 for value in spacing)
        or len(origin) != 3
        or any(not np.isfinite(value) for value in origin)
        or len(direction) != 9
        or any(not np.isfinite(value) for value in direction)
    ):
        raise ImageTransportError(
            f"Input {source_format} image has invalid physical geometry."
        )
    direction_matrix = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    if not np.allclose(
        direction_matrix.T @ direction_matrix,
        np.eye(3),
        rtol=0,
        atol=1e-5,
    ):
        raise ImageTransportError(
            f"Input {source_format} direction must be orthonormal."
        )
    affine = _sitk_ras_affine(image)
    return MedicalImageInspection(
        source_format=source_format,
        shape=tuple(int(value) for value in image.GetSize()),
        dtype=str(sitk.GetArrayViewFromImage(image).dtype),
        spacing=spacing,
        origin=origin,
        direction=direction,
        orientation="".join(str(value) for value in nib.aff2axcodes(affine)),
    )


def _validate_sitk_equivalence(source: sitk.Image, canonical: sitk.Image) -> None:
    if source.GetDimension() != canonical.GetDimension() or source.GetSize() != canonical.GetSize():
        raise ImageTransportError(
            "Canonical NIfTI shape does not match the platform image."
        )
    for name, observed, expected in (
        ("spacing", canonical.GetSpacing(), source.GetSpacing()),
        ("origin", canonical.GetOrigin(), source.GetOrigin()),
        ("direction", canonical.GetDirection(), source.GetDirection()),
    ):
        if not np.allclose(observed, expected, rtol=0, atol=1e-5):
            raise ImageTransportError(
                f"Canonical NIfTI {name} does not match the platform image."
            )
    size = tuple(int(value) for value in source.GetSize())
    landmarks = {
        (0, 0, 0),
        tuple(max(value - 1, 0) for value in size),
        tuple(value // 2 for value in size),
        tuple(min(max(value // 3, 0), value - 1) for value in size),
    }
    for index in landmarks:
        source_point = source.TransformIndexToPhysicalPoint(index)
        canonical_point = canonical.TransformIndexToPhysicalPoint(index)
        if not np.allclose(source_point, canonical_point, rtol=0, atol=1e-5):
            raise ImageTransportError(
                "Canonical NIfTI world coordinates do not match the platform image."
            )


def _validate_inspection_equivalence(
    source: MedicalImageInspection,
    canonical: MedicalImageInspection,
) -> None:
    if source.shape != canonical.shape or source.dtype != canonical.dtype:
        raise ImageTransportError(
            "Canonical NIfTI shape or dtype does not match the platform image."
        )
    for name in ("spacing", "origin", "direction"):
        if not np.allclose(
            getattr(source, name),
            getattr(canonical, name),
            rtol=0,
            atol=1e-5,
        ):
            raise ImageTransportError(
                f"Canonical NIfTI {name} does not match the platform image."
            )


def _sitk_ras_affine(image: sitk.Image) -> np.ndarray:
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    lps_affine = np.eye(4, dtype=np.float64)
    lps_affine[:3, :3] = direction @ np.diag(spacing)
    lps_affine[:3, 3] = origin
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    return lps_to_ras @ lps_affine


def _require_scratch_capacity(scratch_root: Path, *, required_bytes: int) -> None:
    try:
        free_bytes = int(shutil.disk_usage(scratch_root).free)
    except OSError as exc:
        raise ImageTransportError("Could not inspect invocation scratch capacity.") from exc
    safety_bytes = 64 * 1024 * 1024
    if free_bytes < required_bytes + safety_bytes:
        raise ImageTransportError(
            "Insufficient /tmp capacity for input canonicalization; "
            f"required_bytes={required_bytes + safety_bytes}, free_bytes={free_bytes}."
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
    return _decompose_ras_affine(affine)


def _decompose_ras_affine(
    affine: np.ndarray,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
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
    "CanonicalImageInput",
    "CanonicalizedImageInputs",
    "ImageTransportError",
    "MedicalImageInspection",
    "NiftiInputInspection",
    "canonicalize_image_inputs",
    "inspect_nifti_input",
    "materialize_prediction_outputs",
    "write_mha_prediction",
    "write_nifti_prediction",
]
