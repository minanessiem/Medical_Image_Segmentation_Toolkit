"""Registered deterministic preprocessing adapters for repository datasets."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Callable, Mapping, Sequence

from monai.transforms import Compose

from src.data.loader_stack.contracts import PreprocessingAdapterError
from src.data.loader_stack.registry import get_dataset_capabilities


PipelineBuilder = Callable[[Sequence[str], Mapping[str, Any], bool], Compose]
ModalityNormalizer = Callable[[Sequence[str] | None], tuple[str, ...]]
BaseModalityResolver = Callable[[str], str]


@dataclass(frozen=True)
class DatasetPreprocessingAdapter:
    """Dataset-owned deterministic full-volume preprocessing boundary."""

    dataset_id: str
    canonical_raw_modalities: tuple[str, ...]
    build_full_volume_pipeline: PipelineBuilder
    normalize_modalities: ModalityNormalizer
    resolve_base_modality: BaseModalityResolver

    def resolve_required_raw_modalities(
        self,
        modalities: Sequence[str] | None,
    ) -> tuple[str, ...]:
        normalized = self.normalize_modalities(modalities)
        required: list[str] = []
        for modality in normalized:
            raw_key = str(self.resolve_base_modality(modality))
            if raw_key not in self.canonical_raw_modalities:
                raise PreprocessingAdapterError(
                    f"Dataset '{self.dataset_id}' modality '{modality}' resolves to "
                    f"unsupported raw key '{raw_key}'. Supported canonical raw keys: "
                    f"{self.canonical_raw_modalities}."
                )
            if raw_key not in required:
                required.append(raw_key)
        if not required:
            raise PreprocessingAdapterError(
                f"Dataset '{self.dataset_id}' preprocessing requires at least one modality."
            )
        return tuple(required)

    def select_native_reference_key(
        self,
        modalities: Sequence[str] | None,
    ) -> str:
        """Choose the first configured raw modality as the aligned native reference."""
        return self.resolve_required_raw_modalities(modalities)[0]


def get_preprocessing_adapter(dataset_id: str) -> DatasetPreprocessingAdapter:
    """Load the adapter registered for ``dataset_id`` without eager loader imports."""
    try:
        capabilities = get_dataset_capabilities(dataset_id)
    except ValueError as exc:
        raise PreprocessingAdapterError(
            f"No preprocessing adapter is registered for dataset '{dataset_id}'."
        ) from exc

    factory_path = capabilities.preprocessing_adapter_factory
    if not factory_path:
        raise PreprocessingAdapterError(
            f"Dataset '{capabilities.dataset_id}' has no implemented preprocessing adapter."
        )
    module_name, separator, factory_name = factory_path.partition(":")
    if not separator or not module_name or not factory_name:
        raise PreprocessingAdapterError(
            f"Dataset '{capabilities.dataset_id}' has invalid preprocessing adapter "
            f"registration '{factory_path}'."
        )
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
        adapter = factory()
    except (ImportError, AttributeError, TypeError) as exc:
        raise PreprocessingAdapterError(
            f"Dataset '{capabilities.dataset_id}' preprocessing adapter "
            f"'{factory_path}' could not be loaded: {exc}"
        ) from exc
    if not isinstance(adapter, DatasetPreprocessingAdapter):
        raise PreprocessingAdapterError(
            f"Dataset '{capabilities.dataset_id}' preprocessing adapter factory "
            "must return DatasetPreprocessingAdapter."
        )
    if adapter.dataset_id != capabilities.dataset_id:
        raise PreprocessingAdapterError(
            f"Dataset adapter id '{adapter.dataset_id}' does not match registry id "
            f"'{capabilities.dataset_id}'."
        )
    return adapter


__all__ = ["DatasetPreprocessingAdapter", "get_preprocessing_adapter"]
