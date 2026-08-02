"""
Config-driven nnU-Net evaluation orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from scripts.evaluation.core.contracts import SliceSample
from scripts.evaluation.metrics.engine import DualLevelStreamingMetricsEngine, VolumeThresholdState
from scripts.evaluation.metrics.engine import StreamingMetricsEngine
from scripts.evaluation.metrics.registry_3d import THREED_METRIC_CLASSES, compute_metrics_3d_at_threshold
from scripts.evaluation.reporting import (
    build_report_payload,
    build_text_summary,
    write_json_report,
    write_threshold_csv,
    write_volume_threshold_csv,
)
from scripts.evaluation.reporting.threshold_protocol import (
    enforce_post_threshold_mode,
    make_fixed_protocol,
    select_primary_threshold,
)
from scripts.nnunet.core.io_adapters import (
    SUPPORTED_INPUT_FORMATS,
    count_nnunet_slice_pairs,
    count_nnunet_volume_pairs,
    iter_nnunet_slice_samples,
    iter_nnunet_volume_samples,
)
from src.inference.contracts import SUPPORTED_OUTPUT_SPACES


@dataclass(frozen=True)
class NnunetEvaluationRequest:
    dataset_id: str
    dataset_name: str
    pred_dir: Path
    gt_dir: Path
    output_dir: Path
    input_format: str
    volume_space: str
    levels: Sequence[str]
    threshold: float
    allow_shape_mismatch: bool
    foreground_only_all_metrics: bool


def _log(message: str) -> None:
    print(f"[nnunet-eval] {message}", flush=True)


def _is_set(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    return True


def _select_dataset_id(cfg: DictConfig) -> str:
    value = OmegaConf.select(
        cfg,
        "nnunet.dataset_id",
        default=OmegaConf.select(cfg, "dataset.nnunet.dataset_id", default=None),
    )
    if not _is_set(value):
        raise ValueError("Evaluation requires nnunet.dataset_id (or dataset.nnunet.dataset_id).")
    return str(value)


def _select_dataset_name(cfg: DictConfig) -> str:
    value = OmegaConf.select(
        cfg,
        "nnunet.dataset_name",
        default=OmegaConf.select(cfg, "dataset.nnunet.dataset_name", default=None),
    )
    if not _is_set(value):
        raise ValueError("Evaluation requires nnunet.dataset_name (or dataset.nnunet.dataset_name).")
    return str(value)


def build_evaluation_request(cfg: DictConfig) -> NnunetEvaluationRequest:
    """
    Resolve a typed evaluation request from composed config.
    """
    dataset_id = _select_dataset_id(cfg)
    dataset_name = _select_dataset_name(cfg)

    pred_dir_cfg = OmegaConf.select(cfg, "nnunet_eval.pred_dir", default=None)
    if not _is_set(pred_dir_cfg):
        raise ValueError("Evaluation requires nnunet_eval.pred_dir.")
    pred_dir = Path(str(pred_dir_cfg))

    gt_dir_cfg = OmegaConf.select(cfg, "nnunet_eval.gt_dir", default=None)
    if not _is_set(gt_dir_cfg):
        raise ValueError("Evaluation requires nnunet_eval.gt_dir.")
    gt_dir = Path(str(gt_dir_cfg))

    output_dir_cfg = OmegaConf.select(cfg, "nnunet_eval.output_dir", default=None)
    if not _is_set(output_dir_cfg):
        raise ValueError("Evaluation requires nnunet_eval.output_dir.")
    output_dir = Path(str(output_dir_cfg))

    input_format_value = OmegaConf.select(cfg, "nnunet_eval.input_format", default=None)
    if not _is_set(input_format_value):
        raise ValueError("Evaluation requires nnunet_eval.input_format.")
    input_format = str(input_format_value).strip().lower()
    if input_format not in SUPPORTED_INPUT_FORMATS:
        raise ValueError(
            "nnunet_eval.input_format must be one of "
            f"{SUPPORTED_INPUT_FORMATS}. Got '{input_format}'."
        )
    if input_format == "slices_2d":
        raise ValueError(
            "Geometry-aware nnU-Net evaluation currently supports 3D volumes only. "
            "The slices_2d path lacks a certified parent-volume geometry, "
            "in-plane resize inversion, slice-placement trace, and reconstruction "
            "contract. Use input_format='volumes_3d' or complete the deferred 2D "
            "geometry-compliance work before evaluation."
        )

    volume_space_value = OmegaConf.select(cfg, "nnunet.export_space", default=None)
    if not _is_set(volume_space_value):
        raise ValueError(
            "3D nnU-Net evaluation requires nnunet.export_space from the composed "
            "conversion preset; it is not inferred or redeclared by evaluation."
        )
    volume_space = str(volume_space_value).strip()
    if volume_space not in SUPPORTED_OUTPUT_SPACES:
        raise ValueError(
            "nnunet.export_space must be one of "
            f"{sorted(SUPPORTED_OUTPUT_SPACES)}, got {volume_space!r}."
        )

    raw_levels = OmegaConf.select(cfg, "nnunet_eval.levels", default=None)
    if raw_levels is None:
        raise ValueError("Evaluation requires nnunet_eval.levels.")
    levels: List[str] = [str(level).strip().lower() for level in list(raw_levels)]
    if not levels:
        raise ValueError("Evaluation requires nnunet_eval.levels to be non-empty.")
    invalid_levels = [level for level in levels if level not in {"slice", "volume"}]
    if invalid_levels:
        raise ValueError(
            f"nnunet_eval.levels contains unsupported values: {invalid_levels}. "
            "Allowed values are 'slice' and 'volume'."
        )

    threshold_value = OmegaConf.select(cfg, "nnunet_eval.threshold", default=None)
    if not _is_set(threshold_value):
        raise ValueError("Evaluation requires nnunet_eval.threshold.")
    threshold = float(threshold_value)

    allow_shape_mismatch_value = OmegaConf.select(cfg, "nnunet_eval.allow_shape_mismatch", default=None)
    if not _is_set(allow_shape_mismatch_value):
        raise ValueError("Evaluation requires nnunet_eval.allow_shape_mismatch.")
    allow_shape_mismatch = bool(allow_shape_mismatch_value)

    foreground_only_value = OmegaConf.select(cfg, "nnunet_eval.foreground_only_all_metrics", default=None)
    if not _is_set(foreground_only_value):
        raise ValueError("Evaluation requires nnunet_eval.foreground_only_all_metrics.")
    foreground_only_all_metrics = bool(foreground_only_value)

    return NnunetEvaluationRequest(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        output_dir=output_dir,
        input_format=input_format,
        volume_space=volume_space,
        levels=levels,
        threshold=threshold,
        allow_shape_mismatch=allow_shape_mismatch,
        foreground_only_all_metrics=foreground_only_all_metrics,
    )


def run_nnunet_evaluation(
    cfg: DictConfig,
    convert_config_name: Optional[str] = None,
    eval_config_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute nnU-Net evaluation for slice-form or native-volume predictions.
    """
    request = build_evaluation_request(cfg)
    _log("Resolved evaluation request from composed config.")
    _log(
        f"input_format={request.input_format}, levels={list(request.levels)}, "
        f"volume_space={request.volume_space}, "
        f"threshold={request.threshold}, allow_shape_mismatch={request.allow_shape_mismatch}, "
        f"foreground_only_all_metrics={request.foreground_only_all_metrics}"
    )
    _log(f"pred_dir={request.pred_dir}")
    _log(f"gt_dir={request.gt_dir}")
    _log(f"output_dir={request.output_dir}")

    if not request.pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {request.pred_dir}")
    if not request.gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory does not exist: {request.gt_dir}")

    protocol = enforce_post_threshold_mode(make_fixed_protocol(request.threshold))
    threshold = float(protocol.thresholds[0])
    _log(f"Using fixed post-threshold protocol at threshold={threshold:.4f}.")
    include_slice_level = "slice" in request.levels
    include_volume_level = "volume" in request.levels

    if request.input_format == "slices_2d":
        _log("Counting matched slice pairs...")
        matched, missing, total_gt = count_nnunet_slice_pairs(request.pred_dir, request.gt_dir)
        _log(f"Slice pairs: matched={matched}, missing={missing}, total_gt={total_gt}")
        if matched == 0:
            raise RuntimeError(
                f"No matched prediction/GT pairs found in slice mode: "
                f"pred_dir='{request.pred_dir}', gt_dir='{request.gt_dir}'."
            )
        slice_results, volume_results = _evaluate_slices_2d(
            pred_dir=request.pred_dir,
            gt_dir=request.gt_dir,
            allow_shape_mismatch=request.allow_shape_mismatch,
            thresholds=protocol.thresholds,
            assembler_case_key=request.dataset_name,
            foreground_only_all_metrics=request.foreground_only_all_metrics,
            expected_pairs=matched,
        )
    elif request.input_format == "volumes_3d":
        _log("Counting matched volume pairs...")
        matched, missing, total_gt = count_nnunet_volume_pairs(request.pred_dir, request.gt_dir)
        _log(f"Volume pairs: matched={matched}, missing={missing}, total_gt={total_gt}")
        if matched == 0:
            raise RuntimeError(
                f"No matched prediction/GT pairs found in volume mode: "
                f"pred_dir='{request.pred_dir}', gt_dir='{request.gt_dir}'."
            )
        slice_results, volume_results = _evaluate_volumes_3d(
            pred_dir=request.pred_dir,
            gt_dir=request.gt_dir,
            allow_shape_mismatch=request.allow_shape_mismatch,
            threshold=threshold,
            compute_slice_metrics=include_slice_level,
            foreground_only_all_metrics=request.foreground_only_all_metrics,
            expected_pairs=matched,
            volume_space=request.volume_space,
        )
    else:
        raise ValueError(
            f"Unsupported input format '{request.input_format}'. "
            f"Allowed values are {SUPPORTED_INPUT_FORMATS}."
        )

    _log("Selecting primary threshold and building report payload...")
    selected_threshold = select_primary_threshold(slice_results, protocol)

    include_volume_level = include_volume_level and volume_results is not None
    payload_volume_results = volume_results if include_volume_level else None

    payload = build_report_payload(
        finalized_results=slice_results,
        protocol=protocol,
        entrypoint_name="evaluate_nnunet_results",
        metadata={
            "pred_dir": str(request.pred_dir.resolve()),
            "gt_dir": str(request.gt_dir.resolve()),
            "matched_pairs": int(matched),
            "missing_predictions": int(missing),
            "input_format": request.input_format,
            "volume_space": request.volume_space,
            "levels": list(request.levels),
            "foreground_only_all_metrics": bool(request.foreground_only_all_metrics),
            "convert_config_name": convert_config_name,
            "eval_config_name": eval_config_name,
        },
        selected_threshold=selected_threshold,
        volume_finalized_results=payload_volume_results,
    )

    _log("Writing report artifacts...")
    json_path = write_json_report(payload, output_dir=request.output_dir)
    summary_path = request.output_dir / "evaluation_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_text = build_text_summary(payload)
    summary_path.write_text(summary_text, encoding="utf-8")

    slice_csv_path: Optional[Path] = None
    if include_slice_level:
        slice_csv_path = write_threshold_csv(slice_results, output_dir=request.output_dir)

    volume_csv_path: Optional[Path] = None
    if include_volume_level and payload_volume_results is not None:
        volume_csv_path = write_volume_threshold_csv(payload_volume_results, output_dir=request.output_dir)
    _log("Evaluation completed and artifacts written.")

    return {
        "input_format": request.input_format,
        "volume_space": request.volume_space,
        "levels": list(request.levels),
        "pred_dir": str(request.pred_dir),
        "gt_dir": str(request.gt_dir),
        "output_dir": str(request.output_dir),
        "matched_pairs": int(matched),
        "missing_predictions": int(missing),
        "total_gt_files": int(total_gt),
        "json_path": str(json_path),
        "summary_path": str(summary_path),
        "slice_csv_path": str(slice_csv_path) if slice_csv_path is not None else None,
        "volume_csv_path": str(volume_csv_path) if volume_csv_path is not None else None,
        "summary_text": summary_text,
    }


def _evaluate_slices_2d(
    pred_dir: Path,
    gt_dir: Path,
    allow_shape_mismatch: bool,
    thresholds: Sequence[float],
    assembler_case_key: str,
    foreground_only_all_metrics: bool,
    expected_pairs: int,
) -> Tuple[Dict[float, Dict[str, Any]], Dict[float, Dict[str, Any]]]:
    _log("Starting slice-level streaming evaluation...")
    samples = iter_nnunet_slice_samples(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        strict_shape=not allow_shape_mismatch,
    )
    engine = DualLevelStreamingMetricsEngine(
        thresholds=thresholds,
        assembler_case_key=assembler_case_key,
    )
    processed = 0
    pbar = tqdm(total=int(expected_pairs), desc="Evaluating slices", unit="slice", dynamic_ncols=True)
    try:
        for sample in samples:
            engine.update(sample)
            processed += 1
            pbar.update(1)
    finally:
        pbar.close()
    _log(f"Slice-level streaming evaluation complete: processed={processed}.")
    dual_level_results = engine.finalize()
    slice_results = dual_level_results["slice_level"]
    if foreground_only_all_metrics:
        _log("Applying foreground_only_all_metrics policy to slice-level report values.")
        _apply_foreground_only_all_metrics_policy(slice_results)
    volume_results = dual_level_results["volume_level"]
    return slice_results, volume_results


def _evaluate_volumes_3d(
    pred_dir: Path,
    gt_dir: Path,
    allow_shape_mismatch: bool,
    threshold: float,
    compute_slice_metrics: bool,
    foreground_only_all_metrics: bool,
    expected_pairs: int,
    volume_space: str,
) -> Tuple[Dict[float, Dict[str, Any]], Dict[float, Dict[str, Any]]]:
    _log(f"Starting 3D volume evaluation in {volume_space} space...")
    metric_names = tuple(THREED_METRIC_CLASSES.keys())
    state = VolumeThresholdState(
        threshold=threshold,
        metrics={name: state_metric for name, state_metric in _new_running_stats(metric_names).items()},
    )
    total_slices = 0
    foreground_slices = 0
    empty_slices = 0

    slice_engine: Optional[StreamingMetricsEngine] = None
    slice_pbar = None
    if compute_slice_metrics:
        _log("Slice-level metrics requested for volumes_3d; deriving per-slice metrics from each volume.")
        slice_engine = StreamingMetricsEngine(thresholds=[threshold])
        slice_pbar = tqdm(
            total=None,
            desc="Evaluating derived slices",
            unit="slice",
            dynamic_ncols=True,
        )

    processed_volumes = 0
    pbar = tqdm(total=int(expected_pairs), desc="Evaluating volumes", unit="volume", dynamic_ncols=True)
    try:
        for volume_sample in iter_nnunet_volume_samples(
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            volume_space=volume_space,
            strict_shape=not allow_shape_mismatch,
        ):
            pbar.set_postfix_str(str(volume_sample.volume_id), refresh=False)
            pred = volume_sample.prediction_volume
            gt = volume_sample.ground_truth_volume

            if slice_engine is not None:
                num_volume_slices = int(pred.shape[-1])
                for slice_index in range(num_volume_slices):
                    sample = SliceSample(
                        case_id=str(volume_sample.case_id),
                        slice_id=f"{volume_sample.volume_id}_s{slice_index:04d}",
                        volume_id=str(volume_sample.volume_id),
                        slice_index=int(slice_index),
                        prediction_mask=pred[..., slice_index],
                        ground_truth_mask=gt[..., slice_index],
                        metadata={
                            "source": "nnunet_volumes",
                            "volume_space": volume_space,
                            "derived_from_volume": True,
                        },
                    )
                    slice_engine.update(sample)
                    if slice_pbar is not None:
                        slice_pbar.update(1)

            num_slices = int(volume_sample.metadata.get("num_slices", int(pred.shape[-1])))
            state.update_volume_slice_count(num_slices)
            state.volume_counts["total"] += 1
            has_foreground = bool((gt > 0.5).sum() > 0)
            if has_foreground:
                state.volume_counts["foreground"] += 1
            else:
                state.volume_counts["empty"] += 1

            slice_fg_flags = (gt > 0.5).reshape(-1, gt.shape[-1]).any(dim=0)
            fg_count = int(slice_fg_flags.sum().item())
            total_count = int(slice_fg_flags.numel())
            empty_count = total_count - fg_count
            foreground_slices += fg_count
            total_slices += total_count
            empty_slices += empty_count

            metric_values = compute_metrics_3d_at_threshold(
                pred=pred,
                gt=gt,
                threshold=threshold,
                metric_configs=_build_spacing_metric_configs(volume_sample.metadata),
                metric_names=metric_names,
            )
            for metric_name in metric_names:
                state.metrics[metric_name].update(float(metric_values.get(metric_name, 0.0)))
            processed_volumes += 1
            pbar.update(1)
    finally:
        pbar.close()
        if slice_pbar is not None:
            slice_pbar.close()
    _log(f"3D volume evaluation complete: processed={processed_volumes}.")

    volume_results: Dict[float, Dict[str, Any]] = {threshold: state.to_dict()}
    if slice_engine is not None:
        slice_results = slice_engine.finalize()
        if foreground_only_all_metrics:
            _log("Applying foreground_only_all_metrics policy to derived slice-level report values.")
            _apply_foreground_only_all_metrics_policy(slice_results)
        _log(
            "Derived slice-level evaluation complete: "
            f"processed={slice_results[float(threshold)]['slice_counts']['total']}."
        )
    else:
        # Volume-mode evaluation without slice-level request still returns stable
        # slice counts for report payload compatibility.
        slice_results = {
            threshold: {
                "threshold": float(threshold),
                "slice_counts": {
                    "total": int(total_slices),
                    "foreground": int(foreground_slices),
                    "empty": int(empty_slices),
                },
                "metrics": {},
            }
        }
    return slice_results, volume_results


def _build_spacing_metric_configs(metadata: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    spacing = metadata.get("source_spacing_xyz")
    if spacing is None:
        spacing_xyz = (1.0, 1.0, 1.0)
    else:
        spacing_xyz = tuple(float(v) for v in spacing[:3])
        if len(spacing_xyz) != 3:
            spacing_xyz = (1.0, 1.0, 1.0)
    voxel_size = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    return {
        "AbsoluteVolumeDifferenceNative": {"voxel_size": voxel_size},
        "HausdorffDistance95MonaiMm": {"spacing": spacing_xyz},
        "SurfaceDiceMonai": {"spacing": spacing_xyz, "tolerance_mm": 1.0},
        "PredictedVolumeMm3": {"spacing": spacing_xyz},
        "GroundTruthVolumeMm3": {"spacing": spacing_xyz},
    }


def _new_running_stats(metric_names: Sequence[str]) -> Dict[str, Any]:
    from scripts.evaluation.core.contracts import RunningStats

    return {name: RunningStats() for name in metric_names}


def _apply_foreground_only_all_metrics_policy(
    slice_results: Dict[float, Dict[str, Any]],
) -> None:
    """
    Legacy 2D compatibility policy:
    use foreground-only denominator for all metrics.
    """
    for threshold_result in slice_results.values():
        metrics = threshold_result.get("metrics", {})
        for scoped_stats in metrics.values():
            if not isinstance(scoped_stats, dict):
                continue
            fg_stats = scoped_stats.get("foreground_only")
            if isinstance(fg_stats, dict):
                scoped_stats["all_slices"] = dict(fg_stats)
