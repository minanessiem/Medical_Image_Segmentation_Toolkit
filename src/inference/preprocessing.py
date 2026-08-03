"""Dataset-routed, label-optional preprocessing for shared inference consumers."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel
import numpy as np
import torch

from src.data.loader_stack.preprocessing import get_preprocessing_adapter
from src.inference.contracts import (
    Affine,
    InferenceInputError,
    LabeledPreprocessedCase,
    NativeImageMetadata,
    PreprocessedCase,
    SpatialGeometry,
    SpatialTrace,
)


def _config_field(config: Any, field_name: str) -> Any:
    if isinstance(config, Mapping) and field_name in config:
        return config[field_name]
    if hasattr(config, field_name):
        return getattr(config, field_name)
    raise InferenceInputError(
        f"Dataset inference configuration is missing required field '{field_name}'."
    )


def _as_affine(value: Any) -> Affine:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 4):
        raise InferenceInputError(
            f"Expected a 4x4 NIfTI affine, got shape {tuple(array.shape)}."
        )
    return tuple(tuple(float(item) for item in row) for row in array)


def _optional_affine(value: Any) -> Affine | None:
    return None if value is None else _as_affine(value)


def _coerce_source_path(value: Any, *, canonical_key: str) -> Path:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Path)):
        if len(value) != 1:
            raise InferenceInputError(
                f"Raw modality '{canonical_key}' must identify exactly one 3D NIfTI file."
            )
        value = value[0]
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise InferenceInputError(
            f"Raw modality '{canonical_key}' must contain a non-empty NIfTI path."
        )
    return Path(value).expanduser().resolve()


def _capture_native_metadata(
    *,
    canonical_key: str,
    source_path: Path,
) -> tuple[NativeImageMetadata, nibabel.spatialimages.SpatialImage]:
    try:
        image = nibabel.load(str(source_path))
    except Exception as exc:
        raise InferenceInputError(
            f"Could not load raw modality '{canonical_key}' as NIfTI: {exc}"
        ) from exc
    if len(image.shape) != 3:
        raise InferenceInputError(
            f"Raw modality '{canonical_key}' must be a 3D NIfTI volume, "
            f"got shape {tuple(image.shape)}."
        )

    qform, qform_code = image.get_qform(coded=True)
    sform, sform_code = image.get_sform(coded=True)
    affine = np.asarray(image.affine, dtype=np.float64)
    metadata = NativeImageMetadata(
        canonical_key=canonical_key,
        shape=tuple(int(value) for value in image.shape),
        dtype=str(image.get_data_dtype()),
        affine=_as_affine(affine),
        spacing=tuple(float(value) for value in nibabel.affines.voxel_sizes(affine)),
        orientation="".join(str(code) for code in nibabel.aff2axcodes(affine)),
        qform=_optional_affine(qform),
        sform=_optional_affine(sform),
        qform_code=int(qform_code),
        sform_code=int(sform_code),
        source_reference=sha256(str(source_path).encode("utf-8")).hexdigest(),
    )
    return metadata, image


def _model_geometry(image: torch.Tensor) -> SpatialGeometry:
    affine_value = getattr(image, "affine", None)
    if affine_value is None:
        raise InferenceInputError(
            "Dataset preprocessing discarded the model-space affine before the shared "
            "inference boundary."
        )
    affine_array = np.asarray(torch.as_tensor(affine_value).detach().cpu(), dtype=np.float64)
    return SpatialGeometry(
        shape=tuple(int(value) for value in image.shape[-3:]),
        affine=_as_affine(affine_array),
        spacing=tuple(
            float(value) for value in nibabel.affines.voxel_sizes(affine_array)
        ),
        orientation="".join(
            str(code) for code in nibabel.aff2axcodes(affine_array)
        ),
    )


def _transform_history(image: torch.Tensor) -> tuple[Mapping[str, Any], ...]:
    pending = tuple(getattr(image, "pending_operations", ()))
    if pending:
        raise InferenceInputError(
            "Dataset preprocessing left pending spatial operations on the model input. "
            "Shared inference requires all lazy transforms to be materialized before "
            "capturing the spatial trace."
        )
    operations = getattr(image, "applied_operations", ())
    return tuple(dict(operation) for operation in operations if isinstance(operation, Mapping))


def preprocess_case(
    *,
    dataset_id: str,
    case_id: str,
    raw_modalities: Mapping[str, Any],
    dataset_cfg: Any,
    label_path: str | Path | None = None,
    load_labels: bool = False,
) -> PreprocessedCase | LabeledPreprocessedCase:
    """Preprocess one 3D case through its registered dataset implementation."""
    if not isinstance(raw_modalities, Mapping):
        raise InferenceInputError("raw_modalities must be a canonical-key mapping.")

    adapter = get_preprocessing_adapter(dataset_id)
    modalities = adapter.normalize_modalities(_config_field(dataset_cfg, "modalities"))
    preprocessing_configs = _config_field(dataset_cfg, "preprocessing_configs")
    if not isinstance(preprocessing_configs, Mapping):
        raise InferenceInputError(
            "dataset.preprocessing_configs must be a mapping for shared inference."
        )

    required_keys = adapter.resolve_required_raw_modalities(modalities)
    observed_keys = tuple(str(key) for key in raw_modalities)
    if set(observed_keys) != set(required_keys):
        raise InferenceInputError(
            f"Dataset '{adapter.dataset_id}' raw modality keys must be exactly "
            f"{required_keys}; received {observed_keys}."
        )

    case_input: dict[str, str] = {}
    native_metadata: dict[str, NativeImageMetadata] = {}
    for canonical_key in required_keys:
        source_path = _coerce_source_path(
            raw_modalities[canonical_key],
            canonical_key=canonical_key,
        )
        metadata, _image = _capture_native_metadata(
            canonical_key=canonical_key,
            source_path=source_path,
        )
        native_metadata[canonical_key] = metadata
        case_input[canonical_key] = str(source_path)

    native_label = None
    native_label_metadata = None
    if load_labels:
        if label_path is None:
            raise InferenceInputError(
                "label_path is required when preprocess_case(load_labels=True)."
            )
        resolved_label_path = _coerce_source_path(label_path, canonical_key="label")
        native_label_metadata, label_image = _capture_native_metadata(
            canonical_key="label",
            source_path=resolved_label_path,
        )
        native_label_array = np.asarray(label_image.dataobj, dtype=np.float32)
        native_label = torch.from_numpy(native_label_array.copy()).unsqueeze(0).unsqueeze(0)
        case_input["label"] = str(resolved_label_path)

    pipeline = adapter.build_full_volume_pipeline(
        modalities,
        preprocessing_configs,
        bool(load_labels),
    )
    try:
        processed = pipeline(case_input)
    except Exception as exc:
        raise InferenceInputError(
            f"Dataset '{adapter.dataset_id}' preprocessing failed for case "
            f"'{case_id}': {exc}"
        ) from exc

    image = processed.get("image")
    if not torch.is_tensor(image) or image.ndim != 4:
        observed_shape = tuple(image.shape) if torch.is_tensor(image) else None
        raise InferenceInputError(
            "Dataset preprocessing must return a 3D channel-first 'image' tensor "
            f"[C,D,H,W], got {observed_shape}."
        )
    model_geometry = _model_geometry(image)
    reference_key = adapter.select_native_reference_key(modalities)
    preprocessed = PreprocessedCase(
        case_id=str(case_id),
        image=image.float().unsqueeze(0),
        spatial_trace=SpatialTrace(
            original=native_metadata[reference_key].geometry,
            model=model_geometry,
            transform_history=_transform_history(image),
        ),
        native_metadata=native_metadata,
        reference_key=reference_key,
        metadata={
            "dataset_id": adapter.dataset_id,
            "processed_modalities": modalities,
        },
    )
    if not load_labels:
        return preprocessed

    model_label = processed.get("label")
    if not torch.is_tensor(model_label) or model_label.ndim != 4:
        observed_shape = tuple(model_label.shape) if torch.is_tensor(model_label) else None
        raise InferenceInputError(
            "Label-enabled dataset preprocessing must return a channel-first 'label' "
            f"tensor [C,D,H,W], got {observed_shape}."
        )
    assert native_label is not None
    assert native_label_metadata is not None
    return LabeledPreprocessedCase(
        case=preprocessed,
        model_label=model_label.float().unsqueeze(0),
        native_label=native_label.float(),
        native_label_metadata=native_label_metadata,
    )


__all__ = ["preprocess_case"]
