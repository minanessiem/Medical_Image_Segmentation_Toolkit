"""Live repository-model producer for geometry-aware 3D evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from tqdm import tqdm

from scripts.evaluation.core.contracts import VolumeSample
from scripts.evaluation.core.model_loader import is_discriminative_config, resolve_diffusion_type
from src.inference.contracts import LabeledPreprocessedCase
from src.inference.pipeline import ModelProbabilityExecutor, predict_preprocessed_case


def validate_model_evaluation_mode(cfg: DictConfig) -> None:
    """Validate whether the current config can be evaluated by live model IO."""
    data_dim = _normalize_dim_token(OmegaConf.select(cfg, "data_mode.dim", default=None))
    diffusion_type = resolve_diffusion_type(cfg)

    if data_dim == "2d":
        raise ValueError(
            "Geometry-aware repository-model evaluation supports 3D volumes only. "
            "The deferred 2D reconstruction contract must define parent-volume "
            "geometry, slice placement, and inverse resize/preprocessing before "
            "2D samples can enter volume assessment."
        )
    if data_dim == "3d" and is_discriminative_config(cfg):
        return
    if data_dim == "3d":
        raise ValueError(
            "3D live-model evaluation currently supports discriminative adapters only. "
            f"Got diffusion.type='{diffusion_type}'. Current non-discriminative diffusion "
            "adapters are 2D-shaped and do not satisfy the 3D volume inference contract. "
            "A future generative backend must register a compatible ProbabilityPredictor."
        )
    raise ValueError(
        "Unsupported data_mode.dim for model evaluation. "
        f"Expected '2d' or '3d', got {OmegaConf.select(cfg, 'data_mode.dim', default=None)!r}."
    )


def iter_model_volume_samples(
    *,
    executor: ModelProbabilityExecutor,
    cases: Iterable[LabeledPreprocessedCase],
    device: str | torch.device,
    loader_mode: str = "",
    subset: object = None,
    show_progress: bool = True,
    max_samples: Optional[int] = None,
    total_cases: Optional[int] = None,
) -> Iterator[VolumeSample]:
    """Yield one spatially declared sample per typed preprocessed case."""
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be > 0 when provided.")

    resolved_device = torch.device(device)
    selected_total = total_cases
    if max_samples is not None and selected_total is not None:
        selected_total = min(selected_total, max_samples)
    case_iterable = _wrap_with_progress(cases, selected_total, show_progress)

    for case_index, labeled in enumerate(case_iterable):
        if max_samples is not None and case_index >= max_samples:
            return
        if not isinstance(labeled, LabeledPreprocessedCase):
            raise TypeError(
                "Repository-model volume evaluation requires each producer value to be "
                "LabeledPreprocessedCase."
            )
        if int(labeled.case.image.shape[0]) != 1:
            raise ValueError(
                "Repository-model full-volume evaluation processes one complete case at "
                "a time. A job may process many cases sequentially; "
                "inference.sliding_window.sw_batch_size independently controls windows "
                "within the current case."
            )

        device_case = replace(
            labeled.case,
            image=labeled.case.image.to(resolved_device),
        )
        prediction = predict_preprocessed_case(
            executor,
            device_case,
            progress_label=labeled.case_id,
            show_window_progress=show_progress,
        )
        if prediction.output_space == "model_preprocessed":
            reference = labeled.model_label
            reference_geometry = labeled.model_label_geometry
            prediction_geometry = prediction.spatial_trace.model
        else:
            reference = labeled.native_label
            reference_geometry = labeled.native_label_geometry
            prediction_geometry = prediction.spatial_trace.original

        pred_volume = _ensure_channel_first_volume(prediction.probability[0])
        gt_volume = _ensure_channel_first_volume(reference[0])
        metadata = _case_provenance(labeled.case.metadata)
        metadata.update({
            "source": "live_model_volume",
            "case_index": int(case_index),
            "loader_mode": str(loader_mode),
            "inference_policy_source": str(executor.policy_source),
            "output_space": prediction.output_space,
            "precision": executor.policy.precision,
            "shape": tuple(int(dim) for dim in pred_volume.shape),
            "reference_spacing_xyz": list(reference_geometry.spacing),
            "spatial_restoration_applied": _restoration_was_applied(
                prediction.provenance
            ),
        })
        if subset is not None:
            metadata["subset"] = str(subset)

        sample = VolumeSample(
            case_id=labeled.case_id,
            volume_id=labeled.case_id,
            prediction_volume=pred_volume,
            ground_truth_volume=gt_volume,
            prediction_space=prediction.output_space,
            reference_space=prediction.output_space,
            prediction_geometry=prediction_geometry,
            reference_geometry=reference_geometry,
            metadata=metadata,
        )
        sample.validate()
        yield sample


def _case_provenance(metadata: Mapping[str, Any]) -> dict[str, object]:
    provenance: dict[str, object] = {}
    record_metadata = metadata.get("record_metadata", {})
    if isinstance(record_metadata, Mapping):
        provenance.update(
            {
                str(key): _metadata_value_to_python(value)
                for key, value in record_metadata.items()
            }
        )
    for key in ("dataset_id", "processed_modalities"):
        if key in metadata:
            provenance[key] = _metadata_value_to_python(metadata[key])
    return provenance


def _restoration_was_applied(provenance: Mapping[str, Any]) -> bool:
    restoration = provenance.get("spatial_restoration", {})
    if not isinstance(restoration, Mapping):
        return False
    return bool(restoration.get("applied", False))


def _ensure_channel_first_volume(volume: Tensor) -> Tensor:
    if volume.ndim == 4:
        return volume.detach().float().cpu()
    if volume.ndim == 3:
        return volume.detach().float().cpu().unsqueeze(0)
    raise ValueError(
        "Expected item volume tensor to be 3D [H,W,D] or 4D [C,H,W,D], "
        f"got shape={tuple(volume.shape)}."
    )


def _metadata_value_to_python(value: object) -> object:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_metadata_value_to_python(item) for item in value]
    if isinstance(value, list):
        return [_metadata_value_to_python(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _metadata_value_to_python(item)
            for key, item in value.items()
        }
    return value


def _normalize_dim_token(value: object) -> str:
    if value is None:
        raise ValueError("Missing data_mode.dim for model evaluation.")
    token = str(value).strip().lower()
    if token in {"2", "2d"}:
        return "2d"
    if token in {"3", "3d"}:
        return "3d"
    return token


def _wrap_with_progress(
    cases: Iterable[LabeledPreprocessedCase],
    total_cases: Optional[int],
    show_progress: bool,
) -> Iterable[LabeledPreprocessedCase]:
    if not show_progress:
        return cases
    return tqdm(
        cases,
        total=total_cases,
        desc="Evaluating validation volumes",
        leave=True,
    )


__all__ = ["iter_model_volume_samples", "validate_model_evaluation_mode"]
