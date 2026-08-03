"""
Subset contract helpers for dataset partition routing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


SUPPORTED_PARTITIONING_MODES = {"fold", "split"}


def normalize_partitioning_mode(partitioning: Any) -> str:
    if partitioning is None:
        raise ValueError(
            "Missing required dataset.partitioning. Expected one of: {fold, split}."
        )
    token = str(partitioning).strip().lower()
    if token not in SUPPORTED_PARTITIONING_MODES:
        raise ValueError(
            "Invalid dataset.partitioning. Expected one of {fold, split}, "
            f"got: {partitioning!r}."
        )
    return token


def _coerce_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value.keys()}
    raise ValueError(f"{field_name} must be a mapping.")


def _coerce_sequence(value: Any, *, field_name: str) -> list[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a non-empty list.")
    items = list(value)
    if len(items) == 0:
        raise ValueError(f"{field_name} must be a non-empty list.")
    return items


def _normalize_fold_values(
    value: Any,
    *,
    field_name: str,
    fold_value: Any | None,
) -> tuple[int, ...]:
    items = _coerce_sequence(value, field_name=field_name)
    normalized: list[int] = []
    for item in items:
        if isinstance(item, str) and item.strip() == "${dataset.fold}":
            if fold_value is None:
                raise ValueError(
                    f"{field_name} uses '${{dataset.fold}}' but dataset.fold is not set."
                )
            normalized.append(int(fold_value))
            continue
        try:
            normalized.append(int(item))
        except Exception as exc:
            raise ValueError(
                f"{field_name} entries must be int-compatible, got {item!r}."
            ) from exc
    if len(normalized) == 0:
        raise ValueError(f"{field_name} must contain at least one value.")
    # Preserve declaration order while dropping duplicates.
    return tuple(dict.fromkeys(normalized))


def _normalize_split_values(value: Any, *, field_name: str) -> tuple[str, ...]:
    items = _coerce_sequence(value, field_name=field_name)
    normalized: list[str] = []
    for item in items:
        token = str(item).strip()
        if not token:
            raise ValueError(f"{field_name} contains an empty split label.")
        normalized.append(token)
    if len(normalized) == 0:
        raise ValueError(f"{field_name} must contain at least one split label.")
    return tuple(dict.fromkeys(normalized))


def resolve_subset_contract(
    *,
    partitioning: Any,
    subsets: Any,
    active_subsets: Any,
    fold_value: Any | None,
    required_roles: Sequence[str] = ("train", "val", "sample"),
) -> tuple[str, dict[str, dict[str, tuple[Any, ...]]], dict[str, str]]:
    """
    Validate and normalize dataset subset contract.
    """
    normalized_partitioning = normalize_partitioning_mode(partitioning)
    subset_mapping = _coerce_mapping(subsets, field_name="dataset.subsets")
    if len(subset_mapping) == 0:
        raise ValueError("dataset.subsets must contain at least one subset definition.")

    normalized_subsets: dict[str, dict[str, tuple[Any, ...]]] = {}
    for subset_name_raw, spec_raw in subset_mapping.items():
        subset_name = str(subset_name_raw).strip()
        if not subset_name:
            raise ValueError("dataset.subsets contains an empty subset key.")
        spec = _coerce_mapping(spec_raw, field_name=f"dataset.subsets.{subset_name}")

        if normalized_partitioning == "fold":
            has_fold_in = "fold_in" in spec
            has_fold_not_in = "fold_not_in" in spec
            if has_fold_in == has_fold_not_in:
                raise ValueError(
                    f"dataset.subsets.{subset_name} must define exactly one of "
                    "{fold_in, fold_not_in} for partitioning='fold'."
                )
            if has_fold_in:
                normalized_subsets[subset_name] = {
                    "fold_in": _normalize_fold_values(
                        spec["fold_in"],
                        field_name=f"dataset.subsets.{subset_name}.fold_in",
                        fold_value=fold_value,
                    )
                }
            else:
                normalized_subsets[subset_name] = {
                    "fold_not_in": _normalize_fold_values(
                        spec["fold_not_in"],
                        field_name=f"dataset.subsets.{subset_name}.fold_not_in",
                        fold_value=fold_value,
                    )
                }
            continue

        if "split_in" not in spec:
            raise ValueError(
                f"dataset.subsets.{subset_name} must define split_in for "
                "partitioning='split'."
            )
        normalized_subsets[subset_name] = {
            "split_in": _normalize_split_values(
                spec["split_in"],
                field_name=f"dataset.subsets.{subset_name}.split_in",
            )
        }

    active_subset_mapping = _coerce_mapping(
        active_subsets, field_name="dataset.active_subsets"
    )
    if isinstance(required_roles, str) or not isinstance(required_roles, Sequence):
        raise ValueError("required_roles must be a non-empty sequence of role names.")
    normalized_required_roles = tuple(
        dict.fromkeys(str(role).strip() for role in required_roles)
    )
    if not normalized_required_roles or any(
        not role for role in normalized_required_roles
    ):
        raise ValueError("required_roles must contain non-empty role names.")

    normalized_active_subsets: dict[str, str] = {}
    for role in normalized_required_roles:
        if role not in active_subset_mapping:
            if normalized_required_roles == ("train", "val", "sample"):
                raise ValueError(
                    "dataset.active_subsets must define train, val, and sample keys. "
                    f"Missing: {role}"
                )
            raise ValueError(
                "dataset.active_subsets must define every requested subset role. "
                f"Missing: {role}"
            )
        subset_ref = str(active_subset_mapping[role]).strip()
        if not subset_ref:
            raise ValueError(f"dataset.active_subsets.{role} must be non-empty.")
        if subset_ref not in normalized_subsets:
            raise ValueError(
                f"dataset.active_subsets.{role} references unknown subset "
                f"'{subset_ref}'. Available subsets: {sorted(normalized_subsets.keys())}."
            )
        normalized_active_subsets[role] = subset_ref

    return normalized_partitioning, normalized_subsets, normalized_active_subsets


def filter_records_for_subset(
    records: Sequence[Mapping[str, Any]],
    *,
    partitioning: str,
    subset_name: str,
    subset_definitions: Mapping[str, Mapping[str, tuple[Any, ...]]],
) -> list[Mapping[str, Any]]:
    """
    Filter normalized records for one configured subset.
    """
    if subset_name not in subset_definitions:
        raise ValueError(
            f"Unknown subset '{subset_name}'. Available: {sorted(subset_definitions.keys())}."
        )
    selector = subset_definitions[subset_name]
    selected: list[Mapping[str, Any]] = []
    if partitioning == "fold":
        fold_in = selector.get("fold_in")
        fold_not_in = selector.get("fold_not_in")
        for index, record in enumerate(records):
            if "fold" not in record:
                raise ValueError(
                    "Fold-based subset selection requires each record to define 'fold'. "
                    f"Missing at record index {index} (caseID={record.get('caseID')})."
                )
            try:
                record_fold = int(record["fold"])
            except Exception as exc:
                raise ValueError(
                    "Invalid fold value in datalist record. "
                    f"Expected int-compatible value, got {record.get('fold')!r} "
                    f"(caseID={record.get('caseID')})."
                ) from exc
            include = record_fold in fold_in if fold_in is not None else record_fold not in fold_not_in
            if include:
                selected.append(record)
        return selected

    split_in = selector.get("split_in")
    for index, record in enumerate(records):
        split_value = record.get("split")
        split_token = "" if split_value is None else str(split_value).strip()
        if not split_token:
            raise ValueError(
                "Split-based subset selection requires each record to define non-empty 'split'. "
                f"Missing at record index {index} (caseID={record.get('caseID')})."
            )
        if split_token in split_in:
            selected.append(record)
    return selected


def describe_subset_selector(
    *,
    partitioning: str,
    subset_name: str,
    subset_definitions: Mapping[str, Mapping[str, tuple[Any, ...]]],
) -> str:
    selector = subset_definitions[subset_name]
    if partitioning == "fold":
        if "fold_in" in selector:
            return f"{subset_name}: fold_in={list(selector['fold_in'])}"
        return f"{subset_name}: fold_not_in={list(selector['fold_not_in'])}"
    return f"{subset_name}: split_in={list(selector['split_in'])}"
