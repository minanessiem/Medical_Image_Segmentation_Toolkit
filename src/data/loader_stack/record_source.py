"""Validation-only normalized case-record discovery.

This boundary deliberately stops before Dataset, transform, sampler, or
DataLoader construction. Dataset modules continue to own record semantics.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.data.loader_stack.contracts import CaseRecordSourceError
from src.data.loader_stack.factory import resolve_dataset_identity
from src.data.loader_stack.registry import DatasetCapabilities, get_dataset_capabilities
from src.data.loader_stack.subset_contract import resolve_subset_contract


CaseRecord = dict[str, Any]
CaseRecordReader = Callable[..., Sequence[Mapping[str, Any]]]


def _as_config(cfg: Any) -> DictConfig:
    if OmegaConf.is_dict(cfg):
        return cfg
    if isinstance(cfg, Mapping):
        return OmegaConf.create(cfg)
    raise CaseRecordSourceError(
        "Case-record discovery requires an OmegaConf config or mapping."
    )


def _required_text(cfg: DictConfig, path: str) -> str:
    value = OmegaConf.select(cfg, path, default=None)
    text = "" if value is None else str(value).strip()
    if not text:
        raise CaseRecordSourceError(
            f"Case-record discovery requires a non-empty config value at '{path}'."
        )
    return text


def _load_registered_reader(
    capabilities: DatasetCapabilities,
) -> CaseRecordReader:
    registration = capabilities.case_record_reader
    if not registration:
        raise CaseRecordSourceError(
            f"Dataset '{capabilities.dataset_id}' has no registered case-record reader."
        )
    module_name, separator, reader_name = registration.partition(":")
    if not separator or not module_name or not reader_name:
        raise CaseRecordSourceError(
            f"Dataset '{capabilities.dataset_id}' has invalid case-record reader "
            f"registration '{registration}'."
        )
    try:
        module = importlib.import_module(module_name)
        reader = getattr(module, reader_name)
    except (ImportError, AttributeError) as exc:
        raise CaseRecordSourceError(
            f"Dataset '{capabilities.dataset_id}' case-record reader "
            f"'{registration}' could not be loaded: {exc}"
        ) from exc
    if not callable(reader):
        raise CaseRecordSourceError(
            f"Dataset '{capabilities.dataset_id}' case-record reader "
            f"'{registration}' is not callable."
        )
    return reader


def load_case_records(
    cfg: Any,
    subset_role: str = "val",
    load_labels: bool = True,
) -> list[CaseRecord]:
    """Return a normalized configured subset without building runtime loaders."""
    root = _as_config(cfg)
    role = str(subset_role).strip()
    if not role:
        raise CaseRecordSourceError("subset_role must be a non-empty role name.")

    try:
        dataset_id = resolve_dataset_identity(
            OmegaConf.select(root, "dataset.id", default=None),
            OmegaConf.select(root, "dataset.name", default=None),
        )
        capabilities = get_dataset_capabilities(dataset_id)
    except ValueError as exc:
        requested = OmegaConf.select(root, "dataset.id", default=None)
        raise CaseRecordSourceError(
            f"Cannot resolve a registered case-record reader for dataset {requested!r}: {exc}"
        ) from exc

    try:
        partitioning, subset_definitions, active_subsets = resolve_subset_contract(
            partitioning=OmegaConf.select(root, "dataset.partitioning", default=None),
            subsets=OmegaConf.select(root, "dataset.subsets", default=None),
            active_subsets=OmegaConf.select(
                root, "dataset.active_subsets", default=None
            ),
            fold_value=OmegaConf.select(root, "dataset.fold", default=None),
            required_roles=(role,),
        )
    except ValueError as exc:
        raise CaseRecordSourceError(
            f"Invalid subset contract for dataset='{dataset_id}', role='{role}': {exc}"
        ) from exc

    subset_name = active_subsets[role]
    split_file = _required_text(root, "data_io.paths.split_file")
    data_root = _required_text(root, "data_io.paths.data_root")
    reader = _load_registered_reader(capabilities)
    try:
        records = reader(
            datalist=split_file,
            basedir=data_root,
            subset_name=subset_name,
            partitioning=partitioning,
            subset_definitions=subset_definitions,
            fold_value=OmegaConf.select(root, "dataset.fold", default=None),
            load_labels=bool(load_labels),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise CaseRecordSourceError(
            "Failed to load case records for "
            f"dataset='{dataset_id}', role='{role}', subset='{subset_name}': {exc}"
        ) from exc

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise CaseRecordSourceError(
            f"Dataset '{dataset_id}' reader returned {type(records).__name__}; "
            "expected a sequence of normalized mappings."
        )

    normalized: list[CaseRecord] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise CaseRecordSourceError(
                f"Dataset '{dataset_id}' reader returned a non-mapping record at "
                f"index {index} for role='{role}', subset='{subset_name}'."
            )
        normalized.append(dict(record))
    return normalized


__all__ = ["CaseRecord", "load_case_records"]
