"""
Internal loader-stack package for phased data-loader refactor.

Phase 1 introduces this package as scaffolding only.
Runtime behavior remains sourced from `src.data.loaders`.
"""

from src.data.loader_stack.contracts import (
    DatasetPreprocessingError,
    InvalidCaseRecordError,
    LabelRequiredError,
    PreprocessingAdapterError,
    SUPPORTED_LOADER_MODES,
    validate_supported_loader_mode,
)
from src.data.loader_stack.core import _build_loader_kwargs, _is_set
from src.data.loader_stack.factory import (
    LoaderResolution,
    resolve_dataset_identity,
    resolve_loader_contract,
)
from src.data.loader_stack.registry import (
    DatasetCapabilities,
    get_dataset_capabilities,
)
from src.data.loader_stack.preprocessing import (
    DatasetPreprocessingAdapter,
    get_preprocessing_adapter,
)

__all__ = [
    "SUPPORTED_LOADER_MODES",
    "DatasetPreprocessingError",
    "InvalidCaseRecordError",
    "LabelRequiredError",
    "PreprocessingAdapterError",
    "validate_supported_loader_mode",
    "_build_loader_kwargs",
    "_is_set",
    "LoaderResolution",
    "resolve_dataset_identity",
    "resolve_loader_contract",
    "DatasetCapabilities",
    "DatasetPreprocessingAdapter",
    "get_preprocessing_adapter",
    "get_dataset_capabilities",
]
