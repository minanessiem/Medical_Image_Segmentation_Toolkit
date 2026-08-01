"""
Live repository-model IO producer for native 3D volume evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Dict, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from tqdm import tqdm

from scripts.evaluation.core.contracts import VolumeSample
from scripts.evaluation.core.model_loader import is_discriminative_config, resolve_diffusion_type
from src.inference.pipeline import build_model_probability_executor


BatchType = Any


def validate_model_evaluation_mode(cfg: DictConfig) -> None:
    """
    Validate whether the current config can be evaluated by live model IO.
    """
    data_dim = _normalize_dim_token(OmegaConf.select(cfg, "data_mode.dim", default=None))
    diffusion_type = resolve_diffusion_type(cfg)

    if data_dim == "2d":
        return
    if data_dim == "3d" and is_discriminative_config(cfg):
        return
    if data_dim == "3d":
        raise ValueError(
            "3D live-model evaluation currently supports discriminative adapters only. "
            f"Got diffusion.type='{diffusion_type}'. Current non-discriminative diffusion "
            "adapters are 2D-shaped and do not satisfy the 3D volume inference contract."
        )
    raise ValueError(
        "Unsupported data_mode.dim for model evaluation. "
        f"Expected '2d' or '3d', got {OmegaConf.select(cfg, 'data_mode.dim', default=None)!r}."
    )


def iter_model_volume_samples(
    model: Any,
    dataloader: Iterable[BatchType],
    cfg: DictConfig,
    device: str | torch.device,
    show_progress: bool = True,
    max_samples: Optional[int] = None,
) -> Iterator[VolumeSample]:
    """
    Yield 3D ``VolumeSample`` objects from a live repository model.
    """
    validate_model_evaluation_mode(cfg)
    data_dim = _normalize_dim_token(OmegaConf.select(cfg, "data_mode.dim", default=None))
    if data_dim != "3d":
        raise ValueError("iter_model_volume_samples requires data_mode.dim='3d'.")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be > 0 when provided.")

    resolved_device = torch.device(device)
    inferer = build_model_probability_executor(backend=model, cfg=cfg)
    total_batches = _safe_len(dataloader)
    batch_iterable = _wrap_with_progress(dataloader, total_batches, show_progress)
    loader_mode = str(OmegaConf.select(cfg, "data_mode.loader_mode", default="") or "")
    validation_mode = str(
        OmegaConf.select(cfg, "validation.inference.mode", default="direct") or "direct"
    )
    subset = OmegaConf.select(cfg, "dataset.active_subsets.val", default=None)

    yielded = 0
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(batch_iterable):
            image, label, sample_ids, metas = _unpack_batch(batch)
            image = image.to(resolved_device)
            prediction = inferer(
                image,
                progress_label=_resolve_batch_label(sample_ids, batch_index),
                show_window_progress=show_progress,
            )
            probabilities = normalize_probability_prediction(prediction)
            labels = label.detach().float().cpu()

            batch_size = int(probabilities.shape[0])
            if labels.shape[0] != batch_size:
                raise ValueError(
                    "Prediction and label batch size mismatch: "
                    f"pred={tuple(probabilities.shape)}, label={tuple(labels.shape)}."
                )

            for item_index in range(batch_size):
                if max_samples is not None and yielded >= max_samples:
                    return

                case_id = resolve_batch_item_identity(
                    sample_ids=sample_ids,
                    batch_index=batch_index,
                    item_index=item_index,
                )
                pred_volume = _ensure_channel_first_volume(probabilities[item_index])
                gt_volume = _ensure_channel_first_volume(labels[item_index])
                metadata = {
                    "source": "live_model_volume",
                    "batch_index": int(batch_index),
                    "item_index": int(item_index),
                    "loader_mode": loader_mode,
                    "validation_inference_mode": validation_mode,
                    "shape": tuple(int(dim) for dim in pred_volume.shape),
                }
                if subset is not None:
                    metadata["subset"] = str(subset)
                metadata.update(_resolve_item_meta(metas, item_index))

                sample = VolumeSample(
                    case_id=case_id,
                    volume_id=case_id,
                    prediction_volume=pred_volume,
                    ground_truth_volume=gt_volume,
                    metadata=metadata,
                )
                sample.validate()
                yield sample
                yielded += 1


def normalize_probability_prediction(prediction: Tensor) -> Tensor:
    """
    Normalize model output to CPU probabilities in ``[0, 1]``.
    """
    pred = prediction.detach().float()
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    return pred.clamp(0, 1).cpu()


def resolve_batch_item_identity(
    sample_ids: object,
    batch_index: int,
    item_index: int,
) -> str:
    """
    Resolve a stable case ID for one item in a collated batch.
    """
    value = _index_collated_value(sample_ids, item_index)
    if value is not None:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if torch.is_tensor(value):
            if value.ndim == 0:
                value = value.item()
            else:
                value = value.detach().cpu().tolist()
        text = str(value).strip()
        if text:
            return text
    return f"batch{batch_index}_item{item_index}"


def _unpack_batch(
    batch: BatchType,
) -> Tuple[Tensor, Tensor, Optional[object], Optional[object]]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("Batch must be tuple/list like (image, label, [case_id], [metadata]).")
    image = batch[0]
    label = batch[1]
    sample_ids = batch[2] if len(batch) > 2 else None
    metas = batch[3] if len(batch) > 3 else None
    if not torch.is_tensor(image) or not torch.is_tensor(label):
        raise ValueError("Batch image and label must be tensors.")
    return image, label, sample_ids, metas


def _ensure_channel_first_volume(volume: Tensor) -> Tensor:
    if volume.ndim == 4:
        return volume.detach().float().cpu()
    if volume.ndim == 3:
        return volume.detach().float().cpu().unsqueeze(0)
    raise ValueError(
        "Expected item volume tensor to be 3D [H,W,D] or 4D [C,H,W,D], "
        f"got shape={tuple(volume.shape)}."
    )


def _resolve_batch_label(sample_ids: object, batch_index: int) -> str:
    first_id = resolve_batch_item_identity(sample_ids, batch_index=batch_index, item_index=0)
    return first_id


def _resolve_item_meta(metas: object, item_index: int) -> Dict[str, object]:
    if metas is None:
        return {}
    if isinstance(metas, Mapping):
        item_meta: Dict[str, object] = {}
        for key, value in metas.items():
            item_value = _index_collated_value(value, item_index)
            if item_value is not None:
                item_meta[str(key)] = _metadata_value_to_python(item_value)
        return item_meta
    item_value = _index_collated_value(metas, item_index)
    if isinstance(item_value, Mapping):
        return {str(key): _metadata_value_to_python(value) for key, value in item_value.items()}
    return {}


def _index_collated_value(value: object, item_index: int) -> Optional[object]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return value if item_index == 0 else None
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value
        if item_index < int(value.shape[0]):
            return value[item_index]
        return None
    if isinstance(value, Sequence):
        if item_index < len(value):
            return value[item_index]
        return None
    return value if item_index == 0 else None


def _metadata_value_to_python(value: object) -> object:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
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


def _safe_len(iterable: Iterable[Any]) -> Optional[int]:
    try:
        return len(iterable)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return None


def _wrap_with_progress(
    dataloader: Iterable[BatchType],
    total_batches: Optional[int],
    show_progress: bool,
) -> Iterable[BatchType]:
    if not show_progress:
        return dataloader
    return tqdm(
        dataloader,
        total=total_batches,
        desc="Evaluating validation volumes",
        leave=True,
    )
