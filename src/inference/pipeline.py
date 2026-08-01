"""Policy-driven model-space probability execution independent of consumers."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Mapping, Optional

import torch
from omegaconf import DictConfig

from src.inference.contracts import (
    InvalidInferencePolicyError,
    ProbabilityPredictor,
    UnsupportedModelError,
)
from src.inference.policy import InferencePolicy, resolve_inference_policy
from src.inference.predictors import (
    build_probability_predictor,
    predict_probabilities,
    validate_predictor_capabilities,
)
from src.inference.sliding_window import predict_sliding_window_probabilities


@dataclass(frozen=True)
class ModelProbabilityExecutor:
    """Execute one prepared predictor according to a resolved shared policy."""

    predictor: ProbabilityPredictor
    policy: InferencePolicy
    policy_source: str

    def __post_init__(self) -> None:
        capabilities = validate_predictor_capabilities(self.predictor.capabilities)
        if self.policy.precision not in capabilities.supported_precisions:
            raise UnsupportedModelError(
                f"Predictor does not support requested precision {self.policy.precision!r}; "
                f"supported={capabilities.supported_precisions}."
            )
        if self.policy.output_space != "model_preprocessed":
            raise InvalidInferencePolicyError(
                "Cut 4 produces model_preprocessed probabilities only. "
                "output_space='native_input' requires Cut 7 spatial restoration."
            )

    def __call__(
        self,
        conditioned_image: torch.Tensor,
        progress_label: Optional[str] = None,
        show_window_progress: bool = True,
    ) -> torch.Tensor:
        with torch.inference_mode(), _autocast_context(
            conditioned_image,
            self.policy.precision,
        ):
            if self.policy.sliding_window.enabled:
                return predict_sliding_window_probabilities(
                    self.predictor,
                    conditioned_image,
                    self.policy.sliding_window,
                    progress_label=progress_label,
                    show_window_progress=show_window_progress,
                )
            return predict_probabilities(self.predictor, conditioned_image)


def build_model_probability_executor(
    backend: Any,
    cfg: Mapping[str, Any] | DictConfig,
) -> ModelProbabilityExecutor:
    """Resolve policy and prepare the configured backend for shared execution."""

    resolved = resolve_inference_policy(cfg)
    predictor = build_probability_predictor(backend=backend, cfg=cfg)
    return ModelProbabilityExecutor(
        predictor=predictor,
        policy=resolved.policy,
        policy_source=resolved.source,
    )


def _autocast_context(
    conditioned_image: torch.Tensor,
    precision: str,
) -> ContextManager[Any]:
    if precision == "fp32":
        return nullcontext()
    if not torch.is_tensor(conditioned_image):
        return nullcontext()

    device_type = conditioned_image.device.type
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    if device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    if device_type == "cpu" and precision == "bf16":
        return torch.autocast(device_type="cpu", dtype=dtype)
    raise UnsupportedModelError(
        f"Precision {precision!r} is not supported for inference execution on "
        f"device type {device_type!r}. Use fp32 or a compatible CUDA runtime."
    )


__all__ = [
    "ModelProbabilityExecutor",
    "build_model_probability_executor",
]
