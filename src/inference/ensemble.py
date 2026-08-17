"""Strict, streaming probability aggregation for model ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import Tensor
from omegaconf import OmegaConf

from src.inference.contracts import InvalidPredictionError


SUPPORTED_ENSEMBLE_METHODS = frozenset({"mean"})


@dataclass
class MeanProbabilityAccumulator:
    """Accumulate equally weighted probabilities without stacking full volumes."""

    _sum: Tensor | None = None
    _count: int = 0

    @property
    def count(self) -> int:
        return self._count

    def add(self, probability: Tensor) -> None:
        if not torch.is_tensor(probability) or not probability.is_floating_point():
            raise InvalidPredictionError(
                "Ensemble members must produce floating-point probability tensors."
            )
        if probability.numel() == 0:
            raise InvalidPredictionError("Ensemble probability tensors must not be empty.")
        if not bool(torch.isfinite(probability).all()):
            raise InvalidPredictionError("Ensemble probabilities contain non-finite values.")
        minimum = float(probability.detach().amin().cpu())
        maximum = float(probability.detach().amax().cpu())
        if minimum < 0.0 or maximum > 1.0:
            raise InvalidPredictionError(
                "Ensemble members must produce probabilities within [0, 1]; "
                f"observed range [{minimum}, {maximum}]."
            )

        value = probability.detach().to(dtype=torch.float32)
        if self._sum is None:
            self._sum = value.clone()
        else:
            if value.device != self._sum.device:
                raise InvalidPredictionError(
                    "All ensemble probability tensors must be on the same device."
                )
            if tuple(value.shape) != tuple(self._sum.shape):
                raise InvalidPredictionError(
                    "All ensemble probability tensors must have identical shapes; "
                    f"expected {tuple(self._sum.shape)}, got {tuple(value.shape)}."
                )
            self._sum.add_(value)
        self._count += 1

    def mean(self) -> Tensor:
        if self._sum is None or self._count == 0:
            raise InvalidPredictionError(
                "Cannot produce an ensemble mean without at least one member."
            )
        return self._sum.div(float(self._count))


def mean_probability_ensemble(probabilities: Iterable[Tensor]) -> Tensor:
    """Return the equal-weight FP32 mean over an arbitrary number of members."""

    accumulator = MeanProbabilityAccumulator()
    for probability in probabilities:
        accumulator.add(probability)
    return accumulator.mean()


def model_ensemble_contract(cfg: Any) -> dict[str, Any]:
    """Project fields that must agree for model-grid probability averaging."""

    contract: dict[str, Any] = {}
    for path in (
        "dataset.id",
        "dataset.modalities",
        "dataset.num_modalities",
        "dataset.preprocessing_configs",
        "data_mode.dim",
        "model",
        "diffusion.type",
    ):
        value = OmegaConf.select(cfg, path, default=None)
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        contract[path] = value
    return contract


__all__ = [
    "MeanProbabilityAccumulator",
    "SUPPORTED_ENSEMBLE_METHODS",
    "mean_probability_ensemble",
    "model_ensemble_contract",
]
