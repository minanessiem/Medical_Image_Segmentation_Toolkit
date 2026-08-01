"""Probability-domain MONAI sliding-window execution."""

from __future__ import annotations

from typing import Optional

import torch

from src.inference.contracts import ProbabilityPredictor
from src.inference.policy import SlidingWindowPolicy
from src.inference.predictors import predict_probabilities, validate_probability_output
from src.utils.monai_sliding_window_backport import (
    sliding_window_inference as monai_sliding_window_inference,
)


def predict_sliding_window_probabilities(
    predictor: ProbabilityPredictor,
    conditioned_image: torch.Tensor,
    policy: SlidingWindowPolicy,
    *,
    progress_label: Optional[str] = None,
    show_window_progress: bool = True,
) -> torch.Tensor:
    """Predict each window in probability space, then blend probabilities."""

    def _predict_window(window_batch: torch.Tensor) -> torch.Tensor:
        return predict_probabilities(predictor, window_batch)

    probability = monai_sliding_window_inference(
        conditioned_image,
        roi_size=policy.roi_size,
        sw_batch_size=policy.sw_batch_size,
        predictor=_predict_window,
        overlap=policy.overlap,
        mode=policy.blend_mode,
        padding_mode=policy.padding_mode,
        progress=bool(show_window_progress),
        progress_desc=_progress_description(progress_label),
        progress_position=1,
        progress_leave=False,
    )
    capabilities = predictor.capabilities
    return validate_probability_output(
        probability,
        conditioned_image=conditioned_image,
        spatial_dims=capabilities.spatial_dims,
        output_channels=capabilities.output_channels,
    )


def _progress_description(progress_label: Optional[str]) -> str:
    if progress_label is None:
        return "SW Volume"
    trimmed = str(progress_label).strip()
    return f"SW {trimmed}" if trimmed else "SW Volume"


__all__ = ["predict_sliding_window_probabilities"]
