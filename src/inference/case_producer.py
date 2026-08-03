"""Reusable, dataset-routed production of typed post-training inference cases."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel
import numpy as np
import torch

from src.data.loader_stack.contracts import PreprocessingAdapterError
from src.data.loader_stack.preprocessing import get_preprocessing_adapter
from src.inference.contracts import (
    Affine,
    InferenceInputError,
    LabeledPreprocessedCase,
    NativeImageMetadata,
    PreprocessedCase,
    SpatialGeometry,
    SpatialRestorationError,
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


def _tensor_geometry(tensor: torch.Tensor, *, field_name: str) -> SpatialGeometry:
    affine_value = getattr(tensor, "affine", None)
    if affine_value is None:
        raise InferenceInputError(
            f"Dataset preprocessing discarded the {field_name} affine before the "
            "shared inference boundary."
        )
    affine_array = np.asarray(
        torch.as_tensor(affine_value).detach().cpu(),
        dtype=np.float64,
    )
    return SpatialGeometry(
        shape=tuple(int(value) for value in tensor.shape[-3:]),
        affine=_as_affine(affine_array),
        spacing=tuple(
            float(value) for value in nibabel.affines.voxel_sizes(affine_array)
        ),
        orientation="".join(str(code) for code in nibabel.aff2axcodes(affine_array)),
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
    return tuple(
        dict(operation) for operation in operations if isinstance(operation, Mapping)
    )


class PreprocessedCaseProducer:
    """Preprocess normalized records through one reusable deterministic pipeline."""

    def __init__(
        self,
        *,
        dataset_id: str,
        dataset_cfg: Any,
        load_labels: bool = False,
    ) -> None:
        self.load_labels = bool(load_labels)
        try:
            self.adapter = get_preprocessing_adapter(dataset_id)
        except PreprocessingAdapterError as exc:
            raise InferenceInputError(
                f"Cannot build a case producer for dataset '{dataset_id}': {exc}"
            ) from exc

        try:
            self.modalities = self.adapter.normalize_modalities(
                _config_field(dataset_cfg, "modalities")
            )
        except InferenceInputError:
            raise
        except (KeyError, TypeError, ValueError, PreprocessingAdapterError) as exc:
            raise InferenceInputError(
                f"Cannot resolve modalities for dataset '{self.adapter.dataset_id}': "
                f"{exc}"
            ) from exc
        self.preprocessing_configs = _config_field(
            dataset_cfg,
            "preprocessing_configs",
        )
        if not isinstance(self.preprocessing_configs, Mapping):
            raise InferenceInputError(
                "dataset.preprocessing_configs must be a mapping for shared inference."
            )
        try:
            self.required_raw_keys = self.adapter.resolve_required_raw_modalities(
                self.modalities
            )
            self.reference_key = self.adapter.select_native_reference_key(
                self.modalities
            )
            self.pipeline = self.adapter.build_full_volume_pipeline(
                self.modalities,
                self.preprocessing_configs,
                self.load_labels,
            )
        except Exception as exc:
            raise InferenceInputError(
                f"Could not build deterministic preprocessing for dataset "
                f"'{self.adapter.dataset_id}': {exc}"
            ) from exc

    def preprocess(
        self,
        record: Mapping[str, Any],
    ) -> PreprocessedCase | LabeledPreprocessedCase:
        """Turn one normalized case record into the configured typed case."""
        if not isinstance(record, Mapping):
            raise InferenceInputError(
                "Case producer records must be normalized mappings."
            )
        case_id_value = record.get("caseID")
        case_id = "" if case_id_value is None else str(case_id_value).strip()
        if not case_id:
            raise InferenceInputError(
                f"Dataset '{self.adapter.dataset_id}' case record is missing a "
                "non-empty 'caseID'."
            )

        try:
            raw_modalities = {
                key: record[key]
                for key in self.required_raw_keys
            }
        except KeyError as exc:
            raise InferenceInputError(
                f"Dataset '{self.adapter.dataset_id}' case '{case_id}' is missing "
                f"required raw modality '{exc.args[0]}'."
            ) from exc

        label_path = record.get("label") if self.load_labels else None
        if self.load_labels and label_path is None:
            raise InferenceInputError(
                f"Dataset '{self.adapter.dataset_id}' case '{case_id}' is missing "
                "required normalized record field 'label'."
            )
        try:
            return self._preprocess_values(
                case_id=case_id,
                raw_modalities=raw_modalities,
                label_path=label_path,
                record_metadata={
                    str(key): value
                    for key, value in record.items()
                    if key
                    not in {
                        "caseID",
                        "label",
                        *self.required_raw_keys,
                    }
                },
            )
        except (InferenceInputError, SpatialRestorationError) as exc:
            raise type(exc)(
                f"Dataset '{self.adapter.dataset_id}' preprocessing failed for "
                f"case '{case_id}': {exc}"
            ) from exc

    __call__ = preprocess

    def _preprocess_values(
        self,
        *,
        case_id: str,
        raw_modalities: Mapping[str, Any],
        label_path: Any,
        record_metadata: Mapping[str, Any],
    ) -> PreprocessedCase | LabeledPreprocessedCase:
        case_input: dict[str, str] = {}
        native_metadata: dict[str, NativeImageMetadata] = {}
        for canonical_key in self.required_raw_keys:
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
        if self.load_labels:
            if label_path is None:
                raise InferenceInputError(
                    "A normalized labeled case record must contain 'label'."
                )
            resolved_label_path = _coerce_source_path(
                label_path,
                canonical_key="label",
            )
            native_label_metadata, label_image = _capture_native_metadata(
                canonical_key="label",
                source_path=resolved_label_path,
            )
            native_label_array = np.asarray(label_image.dataobj, dtype=np.float32)
            native_label = (
                torch.from_numpy(native_label_array.copy()).unsqueeze(0).unsqueeze(0)
            )
            case_input["label"] = str(resolved_label_path)

        try:
            processed = self.pipeline(case_input)
        except Exception as exc:
            raise InferenceInputError(
                f"The registered deterministic pipeline raised: {exc}"
            ) from exc
        if not isinstance(processed, Mapping):
            raise InferenceInputError(
                "The registered deterministic pipeline must return a mapping, got "
                f"{type(processed).__name__}."
            )

        image = processed.get("image")
        if not torch.is_tensor(image) or image.ndim != 4:
            observed_shape = tuple(image.shape) if torch.is_tensor(image) else None
            raise InferenceInputError(
                "Dataset preprocessing must return a 3D channel-first 'image' tensor "
                f"[C,D,H,W], got {observed_shape}."
            )
        model_geometry = _tensor_geometry(image, field_name="model image")
        transform_history = _transform_history(image)
        original_geometry = native_metadata[self.reference_key].geometry
        if model_geometry != original_geometry and not any(
            bool(operation.get("do_transforms", True))
            for operation in transform_history
        ):
            raise SpatialRestorationError(
                "Model and native image grids differ, but preprocessing did not retain "
                "an applied transform history."
            )

        preprocessed = PreprocessedCase(
            case_id=case_id,
            image=image.float().unsqueeze(0),
            spatial_trace=SpatialTrace(
                original=original_geometry,
                model=model_geometry,
                transform_history=transform_history,
            ),
            native_metadata=native_metadata,
            reference_key=self.reference_key,
            metadata={
                "dataset_id": self.adapter.dataset_id,
                "processed_modalities": self.modalities,
                "record_metadata": dict(record_metadata),
            },
        )
        if not self.load_labels:
            return preprocessed

        model_label = processed.get("label")
        if not torch.is_tensor(model_label) or model_label.ndim != 4:
            observed_shape = (
                tuple(model_label.shape) if torch.is_tensor(model_label) else None
            )
            raise InferenceInputError(
                "Label-enabled dataset preprocessing must return a channel-first "
                f"'label' tensor [C,D,H,W], got {observed_shape}."
            )
        model_label_geometry = _tensor_geometry(
            model_label,
            field_name="model label",
        )
        if native_label is None or native_label_metadata is None:
            raise InferenceInputError(
                "Labeled case production lost the native label before preprocessing."
            )
        return LabeledPreprocessedCase(
            case=preprocessed,
            model_label=model_label.float().unsqueeze(0),
            model_label_geometry=model_label_geometry,
            native_label=native_label.float(),
            native_label_metadata=native_label_metadata,
        )


def build_case_producer(
    *,
    dataset_id: str,
    dataset_cfg: Any,
    load_labels: bool = False,
) -> PreprocessedCaseProducer:
    """Build one reusable producer for a sequence of normalized records."""
    return PreprocessedCaseProducer(
        dataset_id=dataset_id,
        dataset_cfg=dataset_cfg,
        load_labels=load_labels,
    )


__all__ = ["PreprocessedCaseProducer", "build_case_producer"]
