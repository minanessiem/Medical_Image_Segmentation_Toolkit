"""
ISLES26 dataset loaders and parsing helpers.

Phase 4 note:
- Keep structure parallel to `src.data.loader_stack.isles24_loader`.
- Implement datalist parsing plus 3D/2D online dataset foundations.
- nnUNet-specific ISLES26 loader integration is implemented in later phases.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import nibabel
import torch
from monai.transforms import (
    Compose,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    Resize,
    SpatialPadd,
    Spacingd,
)

from src.data.augmentation import AugmentationPipeline2D, AugmentationPipeline3D
from src.data.loader_stack.subset_contract import (
    filter_records_for_subset,
    normalize_partitioning_mode,
)
from src.utils.loader_monai_utils import build_monai_compose_safe
from src.utils.loader_transforms import (
    MergeProcessedChannelsTransform,
    ProcessModalitiesTransform,
)
from src.utils.loader_utils import LoaderDataUtils

ISLES26_MODALITY_KEY = "T1"
ISLES26_SUPPORTED_MODALITY_KEYS = (ISLES26_MODALITY_KEY,)
ISLES26_SUPPORTED_PREPROCESSING_KEYS = ("RAW", "ZSCORE", "PCTNORM", "PCT_ZSCORE")
ISLES26_REQUIRED_RECORD_KEYS = ("caseID", ISLES26_MODALITY_KEY, "label")
ISLES26_OPTIONAL_RECORD_KEYS = ("siteID", "metadata_csv", "metadata", "fold")
ISLES26_VIRTUAL_PATH_TEMPLATE = "{case_id}_slice{slice_idx}"
ISLES26_PATCH_PATH_TEMPLATE = "{case_id}_patch{patch_idx}"


def _parse_modality_config(modality_config: str) -> Tuple[str, str]:
    try:
        return LoaderDataUtils.parse_modality_token(modality_config)
    except ValueError as exc:
        raise ValueError(
            "ISLES26 modality config must use '<modality_key>_<preprocessing_key>' format. "
            f"Got '{modality_config}'."
        ) from exc


def _resolve_base_modality(modality_config: str) -> str:
    base_modality, _ = _parse_modality_config(modality_config)
    return base_modality


def _validate_percentile_range(
    *,
    lower_percentile: float,
    upper_percentile: float,
    field_name: str,
) -> None:
    if lower_percentile < 0.0 or upper_percentile > 100.0:
        raise ValueError(
            f"{field_name} must be within [0, 100]. "
            f"Got lower={lower_percentile}, upper={upper_percentile}."
        )
    if lower_percentile >= upper_percentile:
        raise ValueError(
            f"{field_name} requires lower_percentile < upper_percentile. "
            f"Got lower={lower_percentile}, upper={upper_percentile}."
        )


def _validate_positive_eps(*, eps: float, field_name: str) -> None:
    if eps <= 0.0:
        raise ValueError(f"{field_name} must be > 0. Got {eps}.")


def _resolve_t1_preprocessing_params(
    preprocessing_key: str,
    data_stats: Mapping[str, float],
    preprocessing_configs: Mapping[str, Any],
) -> Dict[str, Any]:
    if preprocessing_key == "RAW":
        return {}

    if "t1" not in preprocessing_configs:
        raise KeyError(
            "ISLES26 preprocessing key "
            f"'{preprocessing_key}' requires dataset.preprocessing_configs.t1 settings. "
            f"Observed stats: {dict(data_stats)}"
        )

    t1_cfg = preprocessing_configs["t1"]
    foreground_cfg = t1_cfg["foreground"]
    shared_params = {
        "foreground_threshold": float(foreground_cfg["threshold"]),
        "use_finite_mask": bool(foreground_cfg["use_finite"]),
    }

    if preprocessing_key == "ZSCORE":
        zscore_cfg = t1_cfg["zscore"]
        eps = float(zscore_cfg["eps"])
        _validate_positive_eps(
            eps=eps,
            field_name="dataset.preprocessing_configs.t1.zscore.eps",
        )
        return {
            **shared_params,
            "eps": eps,
        }

    if preprocessing_key == "PCTNORM":
        pctnorm_cfg = t1_cfg["pctnorm"]
        lower_percentile = float(pctnorm_cfg["lower_percentile"])
        upper_percentile = float(pctnorm_cfg["upper_percentile"])
        _validate_percentile_range(
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            field_name="dataset.preprocessing_configs.t1.pctnorm",
        )
        output_min, output_max = LoaderDataUtils.as_float_tuple(
            pctnorm_cfg["output_range"],
            expected_len=2,
            field_name="dataset.preprocessing_configs.t1.pctnorm.output_range",
        )
        if output_max <= output_min:
            raise ValueError(
                "dataset.preprocessing_configs.t1.pctnorm.output_range requires "
                f"min < max, got [{output_min}, {output_max}]."
            )
        eps = float(pctnorm_cfg["eps"])
        _validate_positive_eps(
            eps=eps,
            field_name="dataset.preprocessing_configs.t1.pctnorm.eps",
        )
        return {
            **shared_params,
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
            "output_min": output_min,
            "output_max": output_max,
            "eps": eps,
        }

    if preprocessing_key == "PCT_ZSCORE":
        pct_zscore_cfg = t1_cfg["pct_zscore"]
        lower_percentile = float(pct_zscore_cfg["lower_percentile"])
        upper_percentile = float(pct_zscore_cfg["upper_percentile"])
        _validate_percentile_range(
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            field_name="dataset.preprocessing_configs.t1.pct_zscore",
        )
        eps = float(pct_zscore_cfg["eps"])
        _validate_positive_eps(
            eps=eps,
            field_name="dataset.preprocessing_configs.t1.pct_zscore.eps",
        )
        return {
            **shared_params,
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
            "eps": eps,
        }

    raise NotImplementedError(
        "Unsupported ISLES26 T1 preprocessing key "
        f"'{preprocessing_key}'. Supported keys: {ISLES26_SUPPORTED_PREPROCESSING_KEYS}. "
        f"Observed stats: {dict(data_stats)}"
    )


def _process_t1_raw(raw_tensor: torch.Tensor, **_: Any) -> torch.Tensor:
    return raw_tensor.float()


def _sanitize_float_tensor(raw_tensor: torch.Tensor) -> torch.Tensor:
    tensor = raw_tensor.float()
    finite_mask = torch.isfinite(tensor)
    if bool(torch.all(finite_mask)):
        return tensor
    if bool(torch.any(finite_mask)):
        fill_value = torch.mean(tensor[finite_mask])
    else:
        fill_value = torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device)
    return torch.where(finite_mask, tensor, fill_value)


def _build_foreground_mask(
    tensor: torch.Tensor,
    *,
    foreground_threshold: float,
    use_finite_mask: bool,
) -> torch.Tensor:
    threshold_mask = tensor > float(foreground_threshold)
    if use_finite_mask:
        finite_mask = torch.isfinite(tensor)
        threshold_mask = threshold_mask & finite_mask
        if bool(torch.any(threshold_mask)):
            return threshold_mask
        if bool(torch.any(finite_mask)):
            return finite_mask
    elif bool(torch.any(threshold_mask)):
        return threshold_mask

    finite_mask = torch.isfinite(tensor)
    if bool(torch.any(finite_mask)):
        return finite_mask
    return torch.ones_like(tensor, dtype=torch.bool)


def _extract_masked_values(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = tensor[mask]
    if values.numel() > 0:
        return values

    finite_values = tensor[torch.isfinite(tensor)]
    if finite_values.numel() > 0:
        return finite_values
    return tensor.reshape(-1)


def _compute_percentile_bounds(
    values: torch.Tensor,
    *,
    lower_percentile: float,
    upper_percentile: float,
) -> Tuple[float, float]:
    lower_q = float(torch.quantile(values, lower_percentile / 100.0).item())
    upper_q = float(torch.quantile(values, upper_percentile / 100.0).item())
    return lower_q, upper_q


def _process_t1_zscore(
    raw_tensor: torch.Tensor,
    *,
    foreground_threshold: float,
    use_finite_mask: bool,
    eps: float,
    **_: Any,
) -> torch.Tensor:
    tensor = _sanitize_float_tensor(raw_tensor)
    foreground_mask = _build_foreground_mask(
        tensor,
        foreground_threshold=foreground_threshold,
        use_finite_mask=use_finite_mask,
    )
    values = _extract_masked_values(tensor, foreground_mask)
    mean_val = torch.mean(values)
    std_val = torch.std(values, unbiased=False)
    denom = torch.clamp(std_val, min=float(eps))
    return (tensor - mean_val) / denom


def _process_t1_pctnorm(
    raw_tensor: torch.Tensor,
    *,
    foreground_threshold: float,
    use_finite_mask: bool,
    lower_percentile: float,
    upper_percentile: float,
    output_min: float,
    output_max: float,
    eps: float,
    **_: Any,
) -> torch.Tensor:
    tensor = _sanitize_float_tensor(raw_tensor)
    foreground_mask = _build_foreground_mask(
        tensor,
        foreground_threshold=foreground_threshold,
        use_finite_mask=use_finite_mask,
    )
    values = _extract_masked_values(tensor, foreground_mask)
    lower_q, upper_q = _compute_percentile_bounds(
        values,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    lower_bound = torch.tensor(lower_q, dtype=tensor.dtype, device=tensor.device)
    upper_bound = torch.tensor(upper_q, dtype=tensor.dtype, device=tensor.device)

    clipped = torch.clamp(tensor, min=lower_bound, max=upper_bound)
    denom = torch.clamp(upper_bound - lower_bound, min=float(eps))
    normalized = (clipped - lower_bound) / denom
    scaled = normalized * float(output_max - output_min) + float(output_min)
    return scaled


def _process_t1_pct_zscore(
    raw_tensor: torch.Tensor,
    *,
    foreground_threshold: float,
    use_finite_mask: bool,
    lower_percentile: float,
    upper_percentile: float,
    eps: float,
    **_: Any,
) -> torch.Tensor:
    tensor = _sanitize_float_tensor(raw_tensor)
    foreground_mask = _build_foreground_mask(
        tensor,
        foreground_threshold=foreground_threshold,
        use_finite_mask=use_finite_mask,
    )
    values = _extract_masked_values(tensor, foreground_mask)
    lower_q, upper_q = _compute_percentile_bounds(
        values,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    lower_bound = torch.tensor(lower_q, dtype=tensor.dtype, device=tensor.device)
    upper_bound = torch.tensor(upper_q, dtype=tensor.dtype, device=tensor.device)
    clipped = torch.clamp(tensor, min=lower_bound, max=upper_bound)

    clipped_values = _extract_masked_values(clipped, foreground_mask)
    mean_val = torch.mean(clipped_values)
    std_val = torch.std(clipped_values, unbiased=False)
    denom = torch.clamp(std_val, min=float(eps))
    return (clipped - mean_val) / denom


ISLES26_T1_PROCESSORS = {
    "RAW": _process_t1_raw,
    "ZSCORE": _process_t1_zscore,
    "PCTNORM": _process_t1_pctnorm,
    "PCT_ZSCORE": _process_t1_pct_zscore,
}


def _process_t1_modality(
    modality_config: str,
    raw_tensor: torch.Tensor,
    *,
    preprocessing_configs: Mapping[str, Any],
) -> torch.Tensor:
    base_modality, preprocessing_key = _parse_modality_config(modality_config)
    if base_modality != ISLES26_MODALITY_KEY:
        raise ValueError(
            "ISLES26 currently supports only T1-based modality tokens. "
            f"Got '{modality_config}'."
        )

    data_stats = LoaderDataUtils.compute_tensor_data_stats(raw_tensor)
    params = _resolve_t1_preprocessing_params(
        preprocessing_key=preprocessing_key,
        data_stats=data_stats,
        preprocessing_configs=preprocessing_configs,
    )
    processor = ISLES26_T1_PROCESSORS.get(preprocessing_key)
    if processor is None:
        raise NotImplementedError(
            "Unsupported ISLES26 T1 preprocessing key "
            f"'{preprocessing_key}'. Supported keys: {ISLES26_SUPPORTED_PREPROCESSING_KEYS}."
        )
    return processor(raw_tensor, **params)


def _build_common_preprocess_transforms(
    modalities: Sequence[str],
    preprocessing_configs: Mapping[str, Any],
) -> list[Any]:
    base_modalities = sorted(set(_resolve_base_modality(modality) for modality in modalities))
    keys_to_load = [*base_modalities, "label"]

    common_cfg = preprocessing_configs["common"]
    orientation_cfg = common_cfg["orientation"]
    spacing_cfg = common_cfg["spacing"]
    spacing_enabled = bool(spacing_cfg["enabled"])
    allow_native_spacing = bool(spacing_cfg["allow_native_spacing"])

    transforms: list[Any] = [
        LoadImaged(
            keys=keys_to_load,
            reader="NibabelReader",
            ensure_channel_first=True,
        )
    ]
    if bool(orientation_cfg["enabled"]):
        transforms.append(
            Orientationd(
                keys=keys_to_load,
                axcodes=str(orientation_cfg["axcodes"]),
            )
        )

    if not spacing_enabled and not allow_native_spacing:
        raise ValueError(
            "ISLES26 common.spacing.enabled is false while allow_native_spacing is false. "
            "Enable spacing resampling or set allow_native_spacing=true to opt into native spacing."
        )

    if spacing_enabled:
        pixdim = LoaderDataUtils.as_float_tuple(
            spacing_cfg["pixdim"],
            expected_len=3,
            field_name="dataset.preprocessing_configs.common.spacing.pixdim",
        )
        image_interp = str(spacing_cfg["interpolation"]["image"])
        label_interp = str(spacing_cfg["interpolation"]["label"])
        spacing_modes = tuple([image_interp] * len(base_modalities) + [label_interp])
        transforms.append(
            Spacingd(
                keys=keys_to_load,
                pixdim=pixdim,
                mode=spacing_modes,
            )
        )

    def _process_t1_modality_with_config(
        modality_config: str,
        raw_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return _process_t1_modality(
            modality_config=modality_config,
            raw_tensor=raw_tensor,
            preprocessing_configs=preprocessing_configs,
        )

    transforms.extend(
        [
            ProcessModalitiesTransform(
                modalities,
                resolve_base_modality=_resolve_base_modality,
                process_modality=_process_t1_modality_with_config,
            ),
            MergeProcessedChannelsTransform(
                modalities,
                resolve_base_modality=_resolve_base_modality,
            ),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )
    return transforms


def build_common_preprocessed_volume_pipeline(
    modalities: Sequence[str],
    preprocessing_configs: Mapping[str, Any],
) -> Compose:
    return build_monai_compose_safe(
        _build_common_preprocess_transforms(modalities, preprocessing_configs)
    )


def build_online_slices_3d_to_2d_case_pipeline(
    modalities: Sequence[str],
    preprocessing_configs: Mapping[str, Any],
) -> Compose:
    return build_common_preprocessed_volume_pipeline(
        modalities=modalities,
        preprocessing_configs=preprocessing_configs,
    )


def build_full_volumes_3d_pipeline(
    modalities: Sequence[str],
    preprocessing_configs: Mapping[str, Any],
) -> Compose:
    transforms = _build_common_preprocess_transforms(modalities, preprocessing_configs)
    fullvol_cfg = preprocessing_configs["full_volumes_3d"]
    if bool(fullvol_cfg["pad_to_divisible"]):
        roi_3d = LoaderDataUtils.as_int_tuple(
            preprocessing_configs["roi"]["volume_3d"],
            expected_len=3,
            field_name="dataset.preprocessing_configs.roi.volume_3d",
        )
        transforms.append(SpatialPadd(keys=["image", "label"], spatial_size=roi_3d))
    return build_monai_compose_safe(transforms)


def _build_random_patches_3d_sampling_transforms(
    preprocessing_configs: Mapping[str, Any],
    patches_per_volume: Optional[int] = None,
) -> list[Any]:
    transforms: list[Any] = []
    random_patch_cfg = preprocessing_configs["random_patches_3d"]
    roi_3d = LoaderDataUtils.as_int_tuple(
        preprocessing_configs["roi"]["volume_3d"],
        expected_len=3,
        field_name="dataset.preprocessing_configs.roi.volume_3d",
    )
    if patches_per_volume is None:
        num_samples = int(random_patch_cfg["patches_per_volume"]["train"])
    else:
        num_samples = int(patches_per_volume)
    if num_samples <= 0:
        raise ValueError(
            "random_patches_3d requires patches_per_volume to be > 0, "
            f"got {num_samples}."
        )

    crop_cfg = random_patch_cfg["rand_crop_by_pos_neg_label"]
    transforms.append(
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_3d,
            pos=float(crop_cfg["pos"]),
            neg=float(crop_cfg["neg"]),
            num_samples=num_samples,
            image_key="image",
            image_threshold=float(crop_cfg["image_threshold"]),
            allow_smaller=bool(crop_cfg["allow_smaller"]),
        )
    )
    if bool(random_patch_cfg["pad_to_divisible"]):
        transforms.append(SpatialPadd(keys=["image", "label"], spatial_size=roi_3d))
    transforms.append(EnsureTyped(keys=["image", "label"], dtype=torch.float32))
    return transforms


def build_random_patches_3d_pipeline(
    modalities: Sequence[str],
    preprocessing_configs: Mapping[str, Any],
    patches_per_volume: Optional[int] = None,
    include_common_preprocess: bool = True,
) -> Compose:
    transforms: list[Any] = []
    if include_common_preprocess:
        transforms.extend(_build_common_preprocess_transforms(modalities, preprocessing_configs))
    transforms.extend(
        _build_random_patches_3d_sampling_transforms(
            preprocessing_configs=preprocessing_configs,
            patches_per_volume=patches_per_volume,
        )
    )
    return build_monai_compose_safe(transforms)


def resolve_random_patch_return_patches_per_case(
    preprocessing_configs: Mapping[str, Any],
) -> int:
    """
    Resolve random-patch grouped return factor for training.

    Backward compatibility contract:
    - Missing `return_patches_per_case.train` defaults to 1.
    """
    random_patch_cfg = preprocessing_configs.get("random_patches_3d", {})
    if not hasattr(random_patch_cfg, "get"):
        return 1

    return_cfg = random_patch_cfg.get("return_patches_per_case", {})
    if not hasattr(return_cfg, "get"):
        return_patches_per_case = 1
    elif "train" not in return_cfg:
        return_patches_per_case = 1
    else:
        return_patches_per_case = int(return_cfg["train"])

    if return_patches_per_case <= 0:
        raise ValueError(
            "dataset.preprocessing_configs.random_patches_3d.return_patches_per_case.train "
            f"must be > 0, got {return_patches_per_case}."
        )
    return return_patches_per_case


def _normalize_modalities(modalities: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """
    Normalize modality contract for ISLES26.

    Modality tokens must follow `<modality_key>_<preprocessing_key>` format
    (for example `T1_RAW`, `T1_ZSCORE`, `T1_PCTNORM`, `T1_PCT_ZSCORE`).
    Defaults to the single-modality `T1_RAW` channel when no explicit list is given.
    """
    if modalities is None:
        return (f"{ISLES26_MODALITY_KEY}_RAW",)
    normalized = tuple(str(modality) for modality in modalities)
    if len(normalized) == 0:
        raise ValueError("ISLES26 loaders require a non-empty modalities list.")
    for modality in normalized:
        base_modality, preprocessing_key = _parse_modality_config(modality)
        if base_modality != ISLES26_MODALITY_KEY:
            raise ValueError(
                "ISLES26 loaders currently support only T1-derived modality tokens. "
                f"Got modality '{modality}'."
            )
        if preprocessing_key not in ISLES26_SUPPORTED_PREPROCESSING_KEYS:
            raise ValueError(
                "Unsupported ISLES26 preprocessing key "
                f"'{preprocessing_key}' in modality '{modality}'. "
                f"Supported keys: {ISLES26_SUPPORTED_PREPROCESSING_KEYS}."
            )
    return normalized


def _normalize_case_record_paths(record: Mapping[str, Any], basedir: str) -> Dict[str, Any]:
    """
    Return a shallow normalized copy of one ISLES26 datalist record.
    """
    normalized = dict(record)
    for key in (ISLES26_MODALITY_KEY, "label", "metadata_csv"):
        if key in normalized:
            normalized[key] = LoaderDataUtils.resolve_path_value(basedir, normalized[key])
    return normalized


def _normalize_t1_paths(t1_value: object, basedir: str) -> list[str]:
    """
    Normalize T1 field to a non-empty list of absolute/expanded paths.
    """
    if isinstance(t1_value, str):
        candidate_paths = [t1_value]
    elif isinstance(t1_value, list):
        candidate_paths = t1_value
    else:
        raise ValueError(
            "ISLES26 record field 'T1' must be either a string path or list of string paths."
        )

    resolved_paths: list[str] = []
    for path_value in candidate_paths:
        if not LoaderDataUtils.is_non_empty(path_value):
            continue
        resolved = LoaderDataUtils.resolve_path_value(basedir, path_value)
        resolved_paths.append(str(resolved))

    if len(resolved_paths) == 0:
        raise ValueError("ISLES26 record requires non-empty 'T1' path list.")
    return resolved_paths


def _normalize_case_record(
    record: Mapping[str, Any],
    basedir: str,
    partitioning: str = "split",
) -> Dict[str, Any]:
    """
    Validate and normalize one ISLES26 datalist record.
    """
    normalized_partitioning = normalize_partitioning_mode(partitioning)
    partition_key = "fold" if normalized_partitioning == "fold" else "split"
    required_keys = (*ISLES26_REQUIRED_RECORD_KEYS, partition_key)
    missing = [key for key in required_keys if key not in record]
    if missing:
        raise ValueError(
            "ISLES26 record is missing required keys: "
            f"{missing}. Required keys for partitioning='{normalized_partitioning}': "
            f"{required_keys}."
        )

    normalized = _normalize_case_record_paths(record=record, basedir=basedir)
    normalized["caseID"] = LoaderDataUtils.normalize_case_id(
        record.get("caseID"),
        field_name="caseID",
    )

    if "split" in record:
        split_value = record.get("split")
        if not LoaderDataUtils.is_non_empty(split_value):
            raise ValueError("ISLES26 record requires non-empty 'split' label when provided.")
        normalized["split"] = str(split_value).strip()

    if "fold" in record:
        if not LoaderDataUtils.is_non_empty(record.get("fold")):
            raise ValueError("ISLES26 record requires non-empty 'fold' value when provided.")
        try:
            normalized["fold"] = int(record.get("fold"))
        except Exception as exc:
            raise ValueError(
                f"ISLES26 record has invalid 'fold' value: {record.get('fold')}."
            ) from exc

    normalized[ISLES26_MODALITY_KEY] = _normalize_t1_paths(
        t1_value=record.get(ISLES26_MODALITY_KEY),
        basedir=basedir,
    )

    label_value = record.get("label")
    if not LoaderDataUtils.is_non_empty(label_value):
        raise ValueError("ISLES26 record requires non-empty 'label' path.")
    normalized["label"] = str(LoaderDataUtils.resolve_path_value(basedir, label_value))

    if "siteID" in record and LoaderDataUtils.is_non_empty(record.get("siteID")):
        normalized["siteID"] = str(record.get("siteID"))

    if "metadata" in record:
        metadata = record.get("metadata")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("ISLES26 record field 'metadata' must be a mapping when provided.")
        normalized["metadata"] = metadata

    if "metadata_csv" in record and LoaderDataUtils.is_non_empty(
        record.get("metadata_csv")
    ):
        normalized["metadata_csv"] = str(
            LoaderDataUtils.resolve_path_value(basedir, record.get("metadata_csv"))
        )

    return normalized


def _read_normalized_records(
    datalist,
    basedir,
    key="training",
    partitioning: str = "split",
):
    datalist_path = os.path.expanduser(str(datalist))
    basedir_path = os.path.expanduser(str(basedir))

    with open(datalist_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    if key not in payload:
        raise ValueError(
            f"ISLES26 datalist missing key '{key}'. Available top-level keys: {list(payload.keys())}."
        )

    raw_records = payload[key]
    if not isinstance(raw_records, list):
        raise ValueError(
            f"ISLES26 datalist key '{key}' must contain a list, got: {type(raw_records).__name__}."
        )

    normalized_records: list[Dict[str, Any]] = []
    for index, record in enumerate(raw_records):
        if not isinstance(record, Mapping):
            raise ValueError(
                f"ISLES26 datalist entry at index {index} must be a mapping, "
                f"got: {type(record).__name__}."
            )
        normalized_records.append(
            _normalize_case_record(
                record=record,
                basedir=basedir_path,
                partitioning=partitioning,
            )
        )
    return normalized_records


def datafold_read(
    datalist,
    basedir,
    subset_name="train",
    key="training",
    partitioning: str = "split",
    subset_definitions: Optional[Mapping[str, Mapping[str, tuple[Any, ...]]]] = None,
):
    """
    Read and normalize ISLES26 datalist entries with subset-based selection.
    """
    normalized_partitioning = normalize_partitioning_mode(partitioning)
    normalized_records = _read_normalized_records(
        datalist=datalist,
        basedir=basedir,
        key=key,
        partitioning=normalized_partitioning,
    )
    requested_subset = str(subset_name).strip()
    if len(requested_subset) == 0:
        raise ValueError("ISLES26 subset_name must be non-empty.")

    normalized_subset_definitions = subset_definitions
    if normalized_subset_definitions is None:
        if normalized_partitioning != "split":
            raise ValueError(
                "ISLES26 datafold_read requires partitioning='split' when "
                "subset_definitions are omitted."
            )
        normalized_subset_definitions = {
            requested_subset: {"split_in": (requested_subset,)}
        }

    return filter_records_for_subset(
        records=normalized_records,
        partitioning=normalized_partitioning,
        subset_name=requested_subset,
        subset_definitions=normalized_subset_definitions,
    )


class ISLES26Dataset3D(torch.utils.data.Dataset):
    """
    ISLES26 3D full-volume dataset scaffold.

    Signature intentionally mirrors ISLES24Dataset3D for parity.
    """

    def __init__(
        self,
        directory,
        datalist_json,
        fold=0,
        subset_name: Optional[str] = None,
        partitioning: str = "split",
        subset_definitions: Optional[Mapping[str, Mapping[str, tuple[Any, ...]]]] = None,
        transform=None,
        modalities=None,
        test_flag=False,
        image_size=32,
        preprocessing_configs: Optional[Mapping[str, Any]] = None,
        aug_cfg=None,
        is_training=False,
    ):
        super().__init__()
        self.directory = os.path.expanduser(str(directory))
        self.datalist_json = datalist_json
        self.fold = int(fold)
        self.transform = transform
        self.test_flag = bool(test_flag)
        self.subset_name = (
            str(subset_name).strip()
            if subset_name is not None
            else ("val" if self.test_flag else "train")
        )
        self.partitioning = str(partitioning).strip().lower()
        self.subset_definitions = subset_definitions
        if len(self.subset_name) == 0:
            raise ValueError("ISLES26Dataset3D requires a non-empty subset_name.")
        self.image_size = int(image_size)
        self.aug_cfg = aug_cfg
        self.is_training = bool(is_training)
        if not isinstance(preprocessing_configs, Mapping):
            raise ValueError(
                "ISLES26Dataset3D requires dataset.preprocessing_configs mapping. "
                f"Got: {type(preprocessing_configs).__name__}."
            )
        self.preprocessing_configs = dict(preprocessing_configs)
        self.modalities = _normalize_modalities(modalities)
        self.base_modalities = sorted(
            list(set(_parse_modality_config(modality)[0] for modality in self.modalities))
        )

        self.database = datafold_read(
            datalist=self.datalist_json,
            basedir=self.directory,
            subset_name=self.subset_name,
            partitioning=self.partitioning,
            subset_definitions=self.subset_definitions,
        )
        self.full_volume_pipeline = build_full_volumes_3d_pipeline(
            modalities=self.modalities,
            preprocessing_configs=self.preprocessing_configs,
        )
        self.augmentation = None
        if self.is_training and self.aug_cfg is not None:
            self.augmentation = AugmentationPipeline3D(self.aug_cfg)
            print("Initialized 3D augmentation pipeline for ISLES26 full-volume training dataset")

    def __len__(self):
        return len(self.database)

    def _load_case_volume(self, filedict: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        case_input = LoaderDataUtils.resolve_case_input_paths(
            filedict=filedict,
            base_modalities=self.base_modalities,
        )
        case_data = self.full_volume_pipeline(case_input)
        return {
            "image": case_data["image"],
            "label": case_data["label"],
        }

    def __getitem__(self, index):
        filedict = self.database[index]
        case_data = self._load_case_volume(filedict)
        image = case_data["image"]
        label = case_data["label"]

        if self.augmentation is not None:
            data_dict = {"image": image, "label": label}
            data_dict = self.augmentation(data_dict)
            image = data_dict["image"]
            label = data_dict["label"]

        if self.transform:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            label = self.transform(label)

        return image, label, filedict["caseID"]


class ISLES26RandomPatches3D(torch.utils.data.Dataset):
    """
    ISLES26 random 3D patch dataset.

    Flatten policy:
    - __len__ = num_cases * (patches_per_volume.train / return_patches_per_case.train)
    - __getitem__ returns one patch when return_patches_per_case.train == 1
    - __getitem__ returns grouped patches when return_patches_per_case.train > 1
    """

    def __init__(
        self,
        directory,
        datalist_json,
        fold=0,
        subset_name: Optional[str] = None,
        partitioning: str = "split",
        subset_definitions: Optional[Mapping[str, Mapping[str, tuple[Any, ...]]]] = None,
        transform=None,
        modalities=None,
        test_flag=False,
        image_size=32,
        preprocessing_configs: Optional[Mapping[str, Any]] = None,
        aug_cfg=None,
        is_training=False,
    ):
        super().__init__()
        self.directory = os.path.expanduser(str(directory))
        self.datalist_json = datalist_json
        self.fold = int(fold)
        self.transform = transform
        self.test_flag = bool(test_flag)
        self.subset_name = (
            str(subset_name).strip()
            if subset_name is not None
            else ("val" if self.test_flag else "train")
        )
        self.partitioning = str(partitioning).strip().lower()
        self.subset_definitions = subset_definitions
        if len(self.subset_name) == 0:
            raise ValueError("ISLES26RandomPatches3D requires a non-empty subset_name.")
        self.image_size = int(image_size)
        self.aug_cfg = aug_cfg
        self.is_training = bool(is_training)
        if not isinstance(preprocessing_configs, Mapping):
            raise ValueError(
                "ISLES26RandomPatches3D requires dataset.preprocessing_configs mapping. "
                f"Got: {type(preprocessing_configs).__name__}."
            )
        self.preprocessing_configs = dict(preprocessing_configs)
        self.modalities = _normalize_modalities(modalities)
        self.base_modalities = sorted(
            list(set(_parse_modality_config(modality)[0] for modality in self.modalities))
        )
        self.database = datafold_read(
            datalist=self.datalist_json,
            basedir=self.directory,
            subset_name=self.subset_name,
            partitioning=self.partitioning,
            subset_definitions=self.subset_definitions,
        )

        self.patches_per_volume = int(
            self.preprocessing_configs["random_patches_3d"]["patches_per_volume"]["train"]
        )
        if self.patches_per_volume <= 0:
            raise ValueError(
                "dataset.preprocessing_configs.random_patches_3d.patches_per_volume.train "
                f"must be > 0, got {self.patches_per_volume}."
            )
        self.return_patches_per_case = resolve_random_patch_return_patches_per_case(
            self.preprocessing_configs
        )
        if self.return_patches_per_case > self.patches_per_volume:
            raise ValueError(
                "dataset.preprocessing_configs.random_patches_3d.return_patches_per_case.train "
                "must be <= dataset.preprocessing_configs.random_patches_3d.patches_per_volume.train. "
                f"Got return_patches_per_case={self.return_patches_per_case}, "
                f"patches_per_volume={self.patches_per_volume}."
            )
        if self.patches_per_volume % self.return_patches_per_case != 0:
            raise ValueError(
                "dataset.preprocessing_configs.random_patches_3d.patches_per_volume.train must be "
                "divisible by dataset.preprocessing_configs.random_patches_3d.return_patches_per_case.train. "
                f"Got patches_per_volume={self.patches_per_volume}, "
                f"return_patches_per_case={self.return_patches_per_case}."
            )
        self.patch_groups_per_case = self.patches_per_volume // self.return_patches_per_case
        self.preprocessed_volume_pipeline = build_common_preprocessed_volume_pipeline(
            modalities=self.modalities,
            preprocessing_configs=self.preprocessing_configs,
        )
        self.random_patch_sampler_pipeline = build_random_patches_3d_pipeline(
            modalities=self.modalities,
            preprocessing_configs=self.preprocessing_configs,
            patches_per_volume=self.patches_per_volume,
            include_common_preprocess=False,
        )
        self.augmentation = None
        if self.is_training and self.aug_cfg is not None:
            self.augmentation = AugmentationPipeline3D(self.aug_cfg)
            print("Initialized 3D augmentation pipeline for ISLES26 random-patch training dataset")

    def __len__(self):
        return len(self.database) * self.patch_groups_per_case

    def _load_case_patches(self, filedict: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        case_input = LoaderDataUtils.resolve_case_input_paths(
            filedict=filedict,
            base_modalities=self.base_modalities,
        )
        case_data = self.preprocessed_volume_pipeline(case_input)
        data_dict = {
            "image": case_data["image"],
            "label": case_data["label"],
        }
        patch_samples = LoaderDataUtils.as_sample_list(
            self.random_patch_sampler_pipeline(data_dict)
        )
        if len(patch_samples) != self.patches_per_volume:
            raise RuntimeError(
                "Random patch pipeline returned unexpected number of samples. "
                f"Expected {self.patches_per_volume}, got {len(patch_samples)}."
            )
        return patch_samples

    def _apply_patch_postprocessing(
        self, image: torch.Tensor, label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Random-patch mode policy: crop first, then augment per returned patch.
        if self.augmentation is not None:
            data_dict = {"image": image, "label": label}
            data_dict = self.augmentation(data_dict)
            image = data_dict["image"]
            label = data_dict["label"]

        if self.transform:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            label = self.transform(label)
        return image, label

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Patch-group index out of range: {index}. Dataset length is {len(self)}."
            )
        case_idx = int(index) // self.patch_groups_per_case
        patch_group_idx = int(index) % self.patch_groups_per_case
        patch_start_idx = patch_group_idx * self.return_patches_per_case
        patch_end_idx = patch_start_idx + self.return_patches_per_case
        filedict = self.database[case_idx]

        patch_samples = self._load_case_patches(filedict)
        selected_patch_samples = patch_samples[patch_start_idx:patch_end_idx]
        if len(selected_patch_samples) != self.return_patches_per_case:
            raise RuntimeError(
                "Random patch group selection returned unexpected number of samples. "
                f"Expected {self.return_patches_per_case}, got {len(selected_patch_samples)}."
            )

        processed_images: list[torch.Tensor] = []
        processed_labels: list[torch.Tensor] = []
        patch_paths: list[str] = []
        for local_offset, patch_sample in enumerate(selected_patch_samples):
            image = patch_sample["image"]
            label = patch_sample["label"]
            image, label = self._apply_patch_postprocessing(image=image, label=label)
            processed_images.append(image)
            processed_labels.append(label)
            patch_idx = patch_start_idx + local_offset
            patch_paths.append(
                ISLES26_PATCH_PATH_TEMPLATE.format(
                    case_id=filedict["caseID"],
                    patch_idx=int(patch_idx),
                )
            )

        if self.return_patches_per_case == 1:
            return processed_images[0], processed_labels[0], patch_paths[0]

        grouped_images = torch.stack(processed_images, dim=0)
        grouped_labels = torch.stack(processed_labels, dim=0)
        return grouped_images, grouped_labels, patch_paths


class ISLES26Dataset2D(torch.utils.data.Dataset):
    """
    ISLES26 online 3D->2D slice dataset scaffold.

    Signature intentionally mirrors ISLES24Dataset2D for parity.
    """

    def __init__(
        self,
        directory,
        datalist_json,
        fold=0,
        subset_name: Optional[str] = None,
        partitioning: str = "split",
        subset_definitions: Optional[Mapping[str, Mapping[str, tuple[Any, ...]]]] = None,
        transform=None,
        modalities=None,
        test_flag=False,
        image_size=32,
        use_caching=False,
        cache_prefix: Optional[str] = None,
        shared_cache=None,
        cache_lock=None,
        aug_cfg=None,
        is_training=False,
        preprocessing_configs: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        self.directory = os.path.expanduser(str(directory))
        self.datalist_json = datalist_json
        self.fold = int(fold)
        self.transform = transform
        self.test_flag = bool(test_flag)
        self.subset_name = (
            str(subset_name).strip()
            if subset_name is not None
            else ("val" if self.test_flag else "train")
        )
        self.partitioning = str(partitioning).strip().lower()
        self.subset_definitions = subset_definitions
        if len(self.subset_name) == 0:
            raise ValueError("ISLES26Dataset2D requires a non-empty subset_name.")
        self.modalities = _normalize_modalities(modalities)
        self.base_modalities = sorted(
            list(set(_parse_modality_config(modality)[0] for modality in self.modalities))
        )

        self.database = datafold_read(
            datalist=self.datalist_json,
            basedir=self.directory,
            subset_name=self.subset_name,
            partitioning=self.partitioning,
            subset_definitions=self.subset_definitions,
        )

        self.all_slices = []
        self.image_size = int(image_size)
        if not isinstance(preprocessing_configs, Mapping):
            raise ValueError(
                "ISLES26Dataset2D requires dataset.preprocessing_configs mapping. "
                f"Got: {type(preprocessing_configs).__name__}."
            )
        self.preprocessing_configs = dict(preprocessing_configs)
        self.slice_axis = int(
            self.preprocessing_configs["online_slices_3d_to_2d"]["slice_axis"]
        )
        if self.slice_axis not in (0, 1, 2):
            raise ValueError(
                "dataset.preprocessing_configs.online_slices_3d_to_2d.slice_axis "
                f"must be one of {{0,1,2}}, got {self.slice_axis}."
            )
        self.slice_order = str(
            self.preprocessing_configs["online_slices_3d_to_2d"]["slice_order"]
        )
        if self.slice_order not in {"sequential", "reverse"}:
            raise ValueError(
                "dataset.preprocessing_configs.online_slices_3d_to_2d.slice_order "
                f"must be one of {{sequential, reverse}}, got '{self.slice_order}'."
            )
        self.slice_roi = LoaderDataUtils.as_int_tuple(
            self.preprocessing_configs["roi"]["slice_2d"],
            expected_len=2,
            field_name="dataset.preprocessing_configs.roi.slice_2d",
        )
        self.case_pipeline = build_online_slices_3d_to_2d_case_pipeline(
            modalities=self.modalities,
            preprocessing_configs=self.preprocessing_configs,
        )
        self.image_slice_resize = Resize(
            spatial_size=self.slice_roi,
            mode="bilinear",
        )
        self.label_slice_resize = Resize(
            spatial_size=self.slice_roi,
            mode="nearest",
        )
        self.use_caching = bool(use_caching)
        self._cache_prefix = (
            str(cache_prefix).strip()
            if cache_prefix is not None
            else f"subset:{self.subset_name}"
        )
        self.shared_cache = shared_cache
        self._shared_cache_lock = cache_lock
        self.aug_cfg = aug_cfg
        self.is_training = bool(is_training)
        self.augmentation = None
        self.return_metadata = False

        if self.is_training and self.aug_cfg is not None:
            self.augmentation = AugmentationPipeline2D(self.aug_cfg)
            print("Initialized augmentation pipeline for ISLES26 training dataset")

        for case_idx, filedict in enumerate(self.database):
            first_mod_key = self.base_modalities[0]
            filepath = (
                filedict[first_mod_key][0]
                if isinstance(filedict[first_mod_key], list)
                else filedict[first_mod_key]
            )
            if os.path.exists(filepath):
                num_slices = int(nibabel.load(filepath).shape[self.slice_axis])
                if self.slice_order == "reverse":
                    slice_indices = range(num_slices - 1, -1, -1)
                else:
                    slice_indices = range(num_slices)
                self.all_slices.extend(
                    [(case_idx, slice_idx) for slice_idx in slice_indices]
                )

        self.cache = None
        self.cache_lock = None
        if self.use_caching:
            if self.shared_cache is not None:
                self.cache = self.shared_cache
                self.cache_lock = self._shared_cache_lock or threading.Lock()
            else:
                self.cache = {}
                self.cache_lock = threading.Lock()

    def __len__(self):
        return len(self.all_slices)

    def _load_case_volume(self, filedict: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        case_input = LoaderDataUtils.resolve_case_input_paths(
            filedict=filedict,
            base_modalities=self.base_modalities,
        )
        case_data = self.case_pipeline(case_input)
        return {
            "image": case_data["image"],
            "label": case_data["label"],
        }

    def _build_slice_metadata(self, filedict: Mapping[str, Any], slice_idx: int) -> Dict[str, Any]:
        metadata = {
            "caseID": str(filedict.get("caseID", "")),
            "slice_index": int(slice_idx),
            "source_path": (
                filedict.get(ISLES26_MODALITY_KEY, [""])[0]
                if isinstance(filedict.get(ISLES26_MODALITY_KEY, []), list)
                else filedict.get(ISLES26_MODALITY_KEY, "")
            ),
            "virtual_path": ISLES26_VIRTUAL_PATH_TEMPLATE.format(
                case_id=str(filedict.get("caseID", "")),
                slice_idx=int(slice_idx),
            ),
        }
        if "siteID" in filedict:
            metadata["siteID"] = filedict["siteID"]
        if "metadata_csv" in filedict:
            metadata["metadata_csv"] = filedict["metadata_csv"]
        if "metadata" in filedict:
            metadata["metadata"] = filedict["metadata"]
        return metadata

    def __getitem__(self, index):
        case_idx, slice_idx = self.all_slices[index]
        filedict = self.database[case_idx]
        cache_key = (self._cache_prefix, int(case_idx))

        if self.use_caching:
            if cache_key not in self.cache:
                with self.cache_lock:
                    if cache_key not in self.cache:
                        self.cache[cache_key] = self._load_case_volume(filedict)
            case_data = self.cache[cache_key]
        else:
            case_data = self._load_case_volume(filedict)

        image_volume = case_data["image"]
        label_volume = case_data["label"]

        # image/label are [C, X, Y, Z] after common preprocessing.
        slice_dim = int(self.slice_axis) + 1
        image = image_volume.select(dim=slice_dim, index=int(slice_idx))
        label = label_volume.select(dim=slice_dim, index=int(slice_idx))

        image = self.image_slice_resize(image)
        label = self.label_slice_resize(label)

        if self.augmentation is not None:
            data_dict = {"image": image, "label": label}
            data_dict = self.augmentation(data_dict)
            image = data_dict["image"]
            label = data_dict["label"]

        if self.transform:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            label = self.transform(label)

        virtual_path = ISLES26_VIRTUAL_PATH_TEMPLATE.format(
            case_id=filedict["caseID"],
            slice_idx=int(slice_idx),
        )
        if self.return_metadata:
            return image, label, virtual_path, self._build_slice_metadata(filedict, slice_idx)
        return image, label, virtual_path


ISLES26OnlineProc2D = ISLES26Dataset2D


def phase_marker() -> str:
    """
    Return a stable marker string for incremental refactor checks.
    """
    return "loader_stack.isles26_loader.phase4_7"


__all__ = [
    "ISLES26_MODALITY_KEY",
    "ISLES26_SUPPORTED_MODALITY_KEYS",
    "ISLES26_SUPPORTED_PREPROCESSING_KEYS",
    "ISLES26_REQUIRED_RECORD_KEYS",
    "ISLES26_OPTIONAL_RECORD_KEYS",
    "ISLES26_VIRTUAL_PATH_TEMPLATE",
    "ISLES26_PATCH_PATH_TEMPLATE",
    "build_common_preprocessed_volume_pipeline",
    "build_online_slices_3d_to_2d_case_pipeline",
    "build_full_volumes_3d_pipeline",
    "build_random_patches_3d_pipeline",
    "_normalize_modalities",
    "_normalize_case_record_paths",
    "datafold_read",
    "ISLES26Dataset3D",
    "ISLES26RandomPatches3D",
    "ISLES26Dataset2D",
    "ISLES26OnlineProc2D",
]
