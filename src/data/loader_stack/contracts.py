"""
Contract helpers for loader-stack validation.

Phase 1 note:
- Canonical validation still lives in `src.data.loaders`.
- Low-risk shared primitives are migrated here incrementally.
"""

from __future__ import annotations


class DatasetPreprocessingError(ValueError):
    """Base error for dataset preprocessing contract failures."""


class PreprocessingAdapterError(DatasetPreprocessingError):
    """A dataset has no usable registered preprocessing adapter."""


class LabelRequiredError(DatasetPreprocessingError):
    """A requested loader or transform is intrinsically label-dependent."""


class InvalidCaseRecordError(DatasetPreprocessingError):
    """A dataset record violates its explicit image/label contract."""


class CaseRecordSourceError(DatasetPreprocessingError):
    """Validation-only case-record discovery could not satisfy its contract."""


SUPPORTED_LOADER_MODES = (
    "online_slices_3d_to_2d",
    "nnunet_slices_2d",
    "full_volumes_3d",
    "random_patches_3d",
)


def validate_supported_loader_mode(loader_mode: object) -> str:
    """
    Validate and normalize the top-level loader mode contract.

    Returns:
        The loader mode as a string when valid.
    """
    if loader_mode not in SUPPORTED_LOADER_MODES:
        raise ValueError(
            "Invalid data_mode.loader_mode. Expected one of "
            "{online_slices_3d_to_2d, nnunet_slices_2d, full_volumes_3d, random_patches_3d}, "
            f"got: {loader_mode}"
        )
    return str(loader_mode)


def phase_marker() -> str:
    """
    Return a stable marker string for incremental refactor checks.
    """
    return "loader_stack.contracts.phase1"
