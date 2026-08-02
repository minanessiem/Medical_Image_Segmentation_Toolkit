"""
Core conversion orchestration for nnU-Net dataset export.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from scripts.nnunet.core.exporters import (
    SliceExportStrategy,
    VolumeExportStrategy,
    clear_directory,
    count_files_in_directory,
    write_provenance_jsonl,
)
from src.data.loader_stack.subset_contract import (
    describe_subset_selector,
    resolve_subset_contract,
)
from src.data.loader_stack.factory import resolve_loader_contract
from src.data.loaders import get_dataloaders, validate_dataset_contract
from src.inference.contracts import SUPPORTED_OUTPUT_SPACES
from src.utils.train_utils import setup_seeds


@dataclass(frozen=True)
class ConversionRequest:
    output_base: Path
    dataset_folder: Path
    images_tr: Path
    labels_tr: Path
    images_ts: Path
    labels_ts: Path
    is_test_mode: bool
    max_slices: Optional[int]
    use_parallel: bool
    num_workers: int
    export_train: bool
    export_test: bool


def _is_set(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    return True


def _resolve_dataset_label(cfg: DictConfig) -> str:
    return str(
        OmegaConf.select(
            cfg,
            "dataset.id",
            default=OmegaConf.select(cfg, "dataset.name", default="unknown"),
        )
    )


def _resolve_augmentation_label(cfg: DictConfig) -> str:
    label = OmegaConf.select(cfg, "augmentation._name_")
    if _is_set(label):
        return str(label)
    return "custom"


def _resolve_export_strategy(cfg: DictConfig) -> Any:
    loader_mode = str(OmegaConf.select(cfg, "data_mode.loader_mode", default="")).strip()
    dim = str(OmegaConf.select(cfg, "data_mode.dim", default="")).strip().lower()
    if loader_mode == "online_slices_3d_to_2d" and dim == "2d":
        return SliceExportStrategy()
    if loader_mode == "full_volumes_3d" and dim == "3d":
        return VolumeExportStrategy(export_space=_resolve_3d_export_space(cfg))
    raise ValueError(
        "nnUNet converter requires one of the supported mode combinations: "
        "(loader_mode='online_slices_3d_to_2d', dim='2d') or "
        "(loader_mode='full_volumes_3d', dim='3d'). "
        f"Got loader_mode='{loader_mode}', dim='{dim}'."
    )


def _resolve_3d_export_space(cfg: DictConfig) -> str:
    value = OmegaConf.select(cfg, "nnunet.export_space", default=None)
    if not _is_set(value):
        raise ValueError(
            "3D nnUNet conversion requires explicit nnunet.export_space. "
            "Set it to 'native_input' or 'model_preprocessed' in the complete "
            "conversion preset; the converter will not infer spatial meaning."
        )
    export_space = str(value).strip()
    if export_space not in SUPPORTED_OUTPUT_SPACES:
        raise ValueError(
            "nnunet.export_space must be one of "
            f"{sorted(SUPPORTED_OUTPUT_SPACES)}, got {export_space!r}."
        )
    return export_space


def _resolve_nnunet_configuration(cfg: DictConfig) -> str:
    dim = str(OmegaConf.select(cfg, "data_mode.dim", default="2d")).strip().lower()
    if dim == "3d":
        return "3d_fullres"
    return "2d"


def _resolve_subset_context(cfg: DictConfig) -> Dict[str, Any]:
    partitioning, subset_definitions, active_subsets = resolve_subset_contract(
        partitioning=OmegaConf.select(cfg, "dataset.partitioning"),
        subsets=OmegaConf.select(cfg, "dataset.subsets"),
        active_subsets=OmegaConf.select(cfg, "dataset.active_subsets"),
        fold_value=OmegaConf.select(cfg, "dataset.fold", default=None),
    )
    selectors_by_role = {
        role: describe_subset_selector(
            partitioning=partitioning,
            subset_name=active_subsets[role],
            subset_definitions=subset_definitions,
        )
        for role in ("train", "val", "sample")
    }
    context: Dict[str, Any] = {
        "partitioning": partitioning,
        "active_subsets": dict(active_subsets),
        "selectors_by_role": selectors_by_role,
    }
    fold_value = OmegaConf.select(cfg, "dataset.fold", default=None)
    if partitioning == "fold" and _is_set(fold_value):
        context["fold"] = int(fold_value)
    return context


def validate_converter_contract(cfg: DictConfig) -> None:
    """
    Validate converter-specific requirements on top of the global data contract.

    Supported conversion modes:
      - online_slices_3d_to_2d (2D slice export)
      - full_volumes_3d (3D volume export)

    Loader routes are currently limited to ISLES24/ISLES26 modules.
    """
    validate_dataset_contract(cfg)

    loader_mode = OmegaConf.select(cfg, "data_mode.loader_mode")
    dim = OmegaConf.select(cfg, "data_mode.dim")
    data_root = OmegaConf.select(cfg, "data_io.paths.data_root")
    split_file = OmegaConf.select(cfg, "data_io.paths.split_file")
    dataset_id = OmegaConf.select(cfg, "dataset.id")
    dataset_name = OmegaConf.select(cfg, "dataset.name")
    modalities = OmegaConf.select(cfg, "dataset.modalities")
    num_modalities = OmegaConf.select(cfg, "dataset.num_modalities")

    is_slice_mode = loader_mode == "online_slices_3d_to_2d" and dim == "2d"
    is_volume_mode = loader_mode == "full_volumes_3d" and dim == "3d"
    if not (is_slice_mode or is_volume_mode):
        raise ValueError(
            "nnUNet converter requires one of the supported mode combinations: "
            "(loader_mode='online_slices_3d_to_2d', dim='2d') or "
            "(loader_mode='full_volumes_3d', dim='3d'). "
            f"Got loader_mode='{loader_mode}', dim='{dim}'."
        )
    if is_volume_mode:
        _resolve_3d_export_space(cfg)
    resolution = resolve_loader_contract(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        loader_mode=loader_mode,
    )
    supported_loader_modules = {
        "src.data.loader_stack.isles24_loader",
        "src.data.loader_stack.isles26_loader",
    }
    if resolution.capabilities.loader_module not in supported_loader_modules:
        raise ValueError(
            "scripts.nnunet.convert_to_nnunet currently supports "
            "dataset routes backed by ISLES24/ISLES26 slice/volume loaders only. "
            f"Got dataset='{resolution.dataset_id}', "
            f"loader_module='{resolution.capabilities.loader_module}'."
        )
    if not _is_set(data_root):
        raise ValueError("nnUNet converter requires data_io.paths.data_root.")
    if not _is_set(split_file):
        raise ValueError("nnUNet converter requires data_io.paths.split_file.")
    modalities_is_sequence = isinstance(modalities, (list, tuple)) or OmegaConf.is_list(modalities)
    if not modalities_is_sequence or len(modalities) == 0:
        raise ValueError("nnUNet converter requires dataset.modalities to be a non-empty list.")
    if int(num_modalities) != len(modalities):
        raise ValueError(
            "nnUNet converter requires len(dataset.modalities) == dataset.num_modalities. "
            f"Got {len(modalities)} vs {num_modalities}."
        )

    nn_dataset_id = OmegaConf.select(cfg, "nnunet.dataset_id")
    nn_dataset_name = OmegaConf.select(cfg, "nnunet.dataset_name")
    output_dir = OmegaConf.select(cfg, "nnunet.output_dir")
    test_mode = bool(OmegaConf.select(cfg, "nnunet.test"))
    test_max_slices = OmegaConf.select(cfg, "nnunet.test_max_slices")

    if not _is_set(nn_dataset_id):
        raise ValueError("nnUNet converter requires nnunet.dataset_id.")
    if not _is_set(nn_dataset_name):
        raise ValueError("nnUNet converter requires nnunet.dataset_name.")
    if not _is_set(output_dir):
        raise ValueError("nnUNet converter requires nnunet.output_dir.")
    if test_mode and (test_max_slices is None or int(test_max_slices) <= 0):
        raise ValueError("When nnunet.test=true, nnunet.test_max_slices must be > 0.")


def build_conversion_request(cfg: DictConfig) -> ConversionRequest:
    """
    Resolve a typed conversion request from Hydra config.
    """
    output_base = Path(cfg.nnunet.output_dir)
    dataset_folder = output_base / f"Dataset{cfg.nnunet.dataset_id}_{cfg.nnunet.dataset_name}"

    images_tr = dataset_folder / "imagesTr"
    labels_tr = dataset_folder / "labelsTr"
    images_ts = dataset_folder / "imagesTs"
    labels_ts = dataset_folder / "labelsTs"

    is_test_mode = bool(cfg.nnunet.test)
    max_slices = int(cfg.nnunet.test_max_slices) if is_test_mode else None
    use_parallel = bool(cfg.nnunet.parallel.enabled)
    num_workers = int(cfg.nnunet.parallel.num_workers)
    export_train = bool(cfg.nnunet.export_train)
    export_test = bool(cfg.nnunet.export_test)

    if not export_train and not export_test:
        raise ValueError("At least one of export_train or export_test must be True")

    return ConversionRequest(
        output_base=output_base,
        dataset_folder=dataset_folder,
        images_tr=images_tr,
        labels_tr=labels_tr,
        images_ts=images_ts,
        labels_ts=labels_ts,
        is_test_mode=is_test_mode,
        max_slices=max_slices,
        use_parallel=use_parallel,
        num_workers=num_workers,
        export_train=export_train,
        export_test=export_test,
    )


def _prepare_output_directories(request: ConversionRequest) -> None:
    for directory in [request.images_tr, request.labels_tr, request.images_ts, request.labels_ts]:
        directory.mkdir(parents=True, exist_ok=True)


def _configure_dataset_for_export(dataset: Any) -> None:
    # Ensure export path gets per-slice metadata and no augmentation side-effects.
    current = dataset
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "return_metadata"):
            current.return_metadata = True
        if hasattr(current, "augmentation"):
            current.augmentation = None
        current = getattr(current, "dataset", None)


def _build_dataset_json_payload(
    cfg: DictConfig,
    num_channels: int,
    num_training: int,
    num_test: int,
    is_test_mode: bool,
    mode_key: str,
    subset_context: Dict[str, Any],
) -> Dict[str, Any]:
    dataset_label = _resolve_dataset_label(cfg)
    if mode_key == "volumes_3d":
        description = f"{dataset_label} 3D volumes preprocessed with MedSegDiff pipeline"
    else:
        description = f"{dataset_label} 2D slices preprocessed with MedSegDiff pipeline"

    channel_names = {str(i): cfg.dataset.modalities[i] for i in range(num_channels)}
    source_config: Dict[str, Any] = {
        "dataset": dataset_label,
        "modalities": list(cfg.dataset.modalities),
        "image_size": cfg.model.image_size,
        "partitioning": subset_context["partitioning"],
        "active_subsets": dict(subset_context["active_subsets"]),
        "active_subset_selectors": dict(subset_context["selectors_by_role"]),
        "export_split_mapping": {
            "train": subset_context["active_subsets"]["train"],
            "test": subset_context["active_subsets"]["val"],
        },
        "test_mode": is_test_mode,
        "num_test": num_test,
    }
    if mode_key == "volumes_3d":
        source_config["export_space"] = _resolve_3d_export_space(cfg)
    if "fold" in subset_context:
        source_config["fold"] = subset_context["fold"]

    return {
        "channel_names": channel_names,
        "labels": {
            "background": 0,
            "lesion": 1,
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "description": description,
        "reference": "Converted from MedSegDiff preprocessing",
        "source_config": source_config,
    }


def run_conversion(cfg: DictConfig) -> Dict[str, Any]:
    """
    Run nnU-Net conversion for supported slice/volume dataset routes.
    """
    setup_seeds(cfg)
    validate_converter_contract(cfg)
    request = build_conversion_request(cfg)
    _prepare_output_directories(request)
    strategy = _resolve_export_strategy(cfg)

    dim_label = str(OmegaConf.select(cfg, "data_mode.dim", default="2d")).upper()
    nnunet_configuration = _resolve_nnunet_configuration(cfg)
    unit_label = strategy.sample_unit_name
    subset_context = _resolve_subset_context(cfg)
    active_subsets = subset_context["active_subsets"]
    train_subset = active_subsets["train"]
    val_subset = active_subsets["val"]
    sample_subset = active_subsets["sample"]

    print(f"\n{'=' * 60}")
    print(f"{_resolve_dataset_label(cfg)} {dim_label} Dataset → nnU-Net Conversion")
    print(f"{'=' * 60}")
    print(f"Mode: {f'TEST (limited {unit_label})' if request.is_test_mode else 'FULL EXPORT'}")
    if request.is_test_mode:
        print(f"Max {unit_label} per split: {request.max_slices}")
    print(
        "Export splits: "
        f"{'train' if request.export_train else ''}"
        f"{' + ' if request.export_train and request.export_test else ''}"
        f"{'test' if request.export_test else ''}"
    )
    print(
        "Parallel: "
        f"{'enabled (' + str(request.num_workers) + ' threads)' if request.use_parallel else 'disabled'}"
    )
    print(f"Source dataset: {_resolve_dataset_label(cfg)}")
    print(f"Modalities: {list(cfg.dataset.modalities)}")
    print(f"Image size: {cfg.model.image_size}")
    print(f"Partitioning: {subset_context['partitioning']}")
    print(
        "Active subsets: "
        f"train={train_subset}, val={val_subset}, sample={sample_subset}"
    )
    print(
        "Subset selectors: "
        f"{subset_context['selectors_by_role']['train']} | "
        f"{subset_context['selectors_by_role']['val']} | "
        f"{subset_context['selectors_by_role']['sample']}"
    )
    print(f"Augmentation: {_resolve_augmentation_label(cfg)}")
    if str(strategy.mode_key) == "volumes_3d":
        print(f"Export space: {strategy.export_space}")
    print(f"Output: {request.dataset_folder}")
    print(f"{'=' * 60}\n")

    dataloaders = get_dataloaders(cfg)
    train_dataset = dataloaders["train"].dataset
    val_dataset = dataloaders["val"].dataset
    _configure_dataset_for_export(train_dataset)
    _configure_dataset_for_export(val_dataset)

    num_channels = len(cfg.dataset.modalities)

    print(f"Total training {unit_label}: {len(train_dataset)}")
    print(f"Total validation {unit_label}: {len(val_dataset)}")
    print(f"Channels per sample: {num_channels}")
    if request.is_test_mode:
        if request.export_train:
            print(f"Will export train: {min(request.max_slices, len(train_dataset))} {unit_label}")
        if request.export_test:
            print(f"Will export test: {min(request.max_slices, len(val_dataset))} {unit_label}")
    print()
    train_case_ids = set()
    val_case_ids = set()
    provenance_records: List[Dict[str, Any]] = []

    if request.export_train:
        clear_directory(request.images_tr, "training images")
        clear_directory(request.labels_tr, "training labels")
        train_case_ids, train_records = strategy.export_split(
            dataset=train_dataset,
            images_dir=request.images_tr,
            labels_dir=request.labels_tr,
            num_channels=num_channels,
            desc=f"Exporting training set (subset: {train_subset})",
            max_slices=request.max_slices,
            split_name="train",
            use_parallel=request.use_parallel,
            num_workers=request.num_workers,
        )
        provenance_records.extend(train_records)
    else:
        print("Skipping training set export (export_train=false)")

    # Memory cleanup between export phases
    del train_dataset
    del dataloaders
    gc.collect()

    if request.export_test:
        clear_directory(request.images_ts, "test images")
        clear_directory(request.labels_ts, "test labels")
        val_case_ids, val_records = strategy.export_split(
            dataset=val_dataset,
            images_dir=request.images_ts,
            labels_dir=request.labels_ts,
            num_channels=num_channels,
            desc=f"Exporting test set (validation subset: {val_subset})",
            max_slices=request.max_slices,
            split_name="test",
            use_parallel=request.use_parallel,
            num_workers=request.num_workers,
        )
        provenance_records.extend(val_records)
    else:
        print("Skipping test set export (export_test=false)")

    del val_dataset
    gc.collect()

    # numTraining: use exported count if we exported, otherwise count from disk.
    if request.export_train:
        num_training = len(train_case_ids)
    else:
        num_training = count_files_in_directory(request.labels_tr, "*.nii.gz")
        print(f"Counted {num_training} existing training labels on disk")

    num_test = len(val_case_ids)

    dataset_json = _build_dataset_json_payload(
        cfg=cfg,
        num_channels=num_channels,
        num_training=num_training,
        num_test=num_test,
        is_test_mode=request.is_test_mode,
        mode_key=str(strategy.mode_key),
        subset_context=subset_context,
    )

    with (request.dataset_folder / "dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_json, handle, indent=2)
    provenance_path = request.dataset_folder / "export_provenance.jsonl"
    write_provenance_jsonl(provenance_records, provenance_path)

    print(f"\n{'=' * 60}")
    print("Conversion complete!")
    print(f"{'=' * 60}")
    print(f"Training {unit_label}: {num_training}")
    print(f"Test {unit_label} ({val_subset} subset): {num_test}")
    if request.is_test_mode:
        print(f"\n⚠️  TEST MODE: Only exported {request.max_slices} {unit_label} per split")
        print("   Run with nnunet.test=false for full export")
    print("\nOutput structure:")
    print(f"  {request.dataset_folder}/")
    print("  ├── dataset.json")
    print(f"  ├── export_provenance.jsonl ({len(provenance_records)} records)")
    print(f"  ├── imagesTr/  ({num_training * num_channels} files)")
    print(f"  ├── labelsTr/  ({num_training} files)")
    print(f"  ├── imagesTs/  ({num_test * num_channels} files)")
    print(f"  └── labelsTs/  ({num_test} files)")

    dataset_id = cfg.nnunet.dataset_id
    pred_dir = f"/mnt/outputs/nnunet_results/predictionsTs_{cfg.nnunet.dataset_name}"
    eval_output_dir = f"{pred_dir}/evaluation_v2"

    print("\nNext steps — Raw nnU-Net commands:")
    print(f"  1. export nnUNet_raw={request.output_base}")
    print(
        f"  2. nnUNetv2_plan_and_preprocess -d {dataset_id} "
        f"-c {nnunet_configuration} --verify_dataset_integrity"
    )
    print(f"  3. nnUNetv2_train {dataset_id} {nnunet_configuration} all")
    print(
        f"  4. nnUNetv2_predict -i {request.images_ts} "
        f"-o {pred_dir} -d {dataset_id} -c {nnunet_configuration} -f all"
    )
    print("  5. python3 -m scripts.nnunet.evaluate_nnunet_results \\")
    print("       --convert-config-name <nnunet/convert/...> --eval-config-name <nnunet/eval/...> \\")
    print(
        f"       nnunet_eval.pred_dir={pred_dir} nnunet_eval.gt_dir={request.labels_ts} "
        f"nnunet_eval.output_dir={eval_output_dir}"
    )

    print("\nNext steps — SLURM runners:")
    print("  1. python3 -m scripts.nnunet.slurm_runners.run_nnunet_command preprocess \\")
    print(f"       -d {dataset_id} -c {nnunet_configuration} --verify")
    print("  2. python3 -m scripts.nnunet.slurm_runners.run_nnunet_command train \\")
    print(f"       -d {dataset_id} -c {nnunet_configuration} -f all")
    print("  3. python3 -m scripts.nnunet.slurm_runners.run_nnunet_command predict \\")
    print(
        f"       -d {dataset_id} -c {nnunet_configuration} "
        f"-i {request.images_ts} -o {pred_dir}"
    )
    print("  4. python3 -m scripts.nnunet.slurm_runners.run_evaluate_nnunet_results \\")
    print("       --convert-config-name <nnunet/convert/...> --eval-config-name <nnunet/eval/...> \\")
    print(
        f"       nnunet_eval.pred_dir={pred_dir} nnunet_eval.gt_dir={request.labels_ts} "
        f"nnunet_eval.output_dir={eval_output_dir}"
    )

    return {
        "dataset_folder": str(request.dataset_folder),
        "num_training": num_training,
        "num_test": num_test,
        "provenance_records": len(provenance_records),
        "partitioning": subset_context["partitioning"],
        "train_subset": train_subset,
        "val_subset": val_subset,
        "export_space": (
            strategy.export_space if str(strategy.mode_key) == "volumes_3d" else None
        ),
    }
