"""
Dataset capability registry for loader routing.

Phase 4 note:
- Used by the loader factory to resolve dataset capability metadata.
- `implementation_state` gates whether a dataset route is active at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetCapabilities:
    """
    Static dataset capability descriptor used by future loader routing.
    """

    dataset_id: str
    supported_loader_modes: tuple[str, ...]
    loader_module: str
    implementation_state: str
    preprocessing_adapter_factory: str | None = None


DEFAULT_DATASET_REGISTRY: dict[str, DatasetCapabilities] = {
    # ISLES24 supports online slices, nnUNet slices, full volumes, and random patches.
    "isles24": DatasetCapabilities(
        dataset_id="isles24",
        supported_loader_modes=(
            "online_slices_3d_to_2d",
            "nnunet_slices_2d",
            "full_volumes_3d",
            "random_patches_3d",
        ),
        loader_module="src.data.loader_stack.isles24_loader",
        implementation_state="legacy-runtime",
        preprocessing_adapter_factory=(
            "src.data.loader_stack.isles24_loader:build_preprocessing_adapter"
        ),
    ),
    # ISLES26 supports online slices, full volumes, and random volume patches.
    "isles26": DatasetCapabilities(
        dataset_id="isles26",
        supported_loader_modes=(
            "online_slices_3d_to_2d",
            "nnunet_slices_2d",
            "full_volumes_3d",
            "random_patches_3d",
        ),
        loader_module="src.data.loader_stack.isles26_loader",
        implementation_state="online-runtime",
        preprocessing_adapter_factory=(
            "src.data.loader_stack.isles26_loader:build_preprocessing_adapter"
        ),
    ),
}


def get_dataset_capabilities(dataset_id: str) -> DatasetCapabilities:
    """
    Resolve dataset capabilities from the default registry.
    """
    key = str(dataset_id).strip().lower()
    if key not in DEFAULT_DATASET_REGISTRY:
        supported = ", ".join(sorted(DEFAULT_DATASET_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported dataset id '{dataset_id}'. Supported dataset ids: [{supported}]"
        )
    return DEFAULT_DATASET_REGISTRY[key]


def phase_marker() -> str:
    """
    Return a stable marker string for incremental refactor checks.
    """
    return "loader_stack.registry.phase4_5"
