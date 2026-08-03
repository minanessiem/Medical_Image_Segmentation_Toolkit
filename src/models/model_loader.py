"""Single-model construction and loading shared by repository consumers."""

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.diffusion.diffusion import Diffusion
from src.models.checkpoint_loading import load_model_state_dict_compat
from src.models.model_factory import build_model


class StrictModelLoadError(RuntimeError):
    """Raised when a release-boundary checkpoint is not an exact model match."""


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: Path,
    device: str | torch.device,
) -> Tuple[List[str], List[str]]:
    """Load a checkpoint using the repository's existing compatibility path."""
    checkpoint_state = torch.load(
        Path(checkpoint_path),
        map_location=torch.device(device),
    )
    return load_model_state_dict_compat(model, checkpoint_state)


def load_model(
    cfg: DictConfig,
    checkpoint_path: Path,
    device: str | torch.device,
) -> Tuple[nn.Module, List[str], List[str]]:
    """Build one configured model and load it as the evaluation path already did."""
    resolved_device = torch.device(device)
    base_model = build_model(cfg)
    model = Diffusion.build_diffusion(base_model, cfg, resolved_device)
    missing_keys, unexpected_keys = load_checkpoint_into_model(
        model=model,
        checkpoint_path=checkpoint_path,
        device=resolved_device,
    )
    model.to(resolved_device)
    model.eval()
    return model, missing_keys, unexpected_keys


def load_model_strict(
    cfg: DictConfig,
    checkpoint_path: Path,
    device: str | torch.device,
) -> nn.Module:
    """Load one release model and reject every missing or unexpected key."""

    model, missing_keys, unexpected_keys = load_model(
        cfg,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    if missing_keys or unexpected_keys:
        details = []
        if missing_keys:
            details.append(f"missing keys={missing_keys}")
        if unexpected_keys:
            details.append(f"unexpected keys={unexpected_keys}")
        raise StrictModelLoadError(
            "Release checkpoint loading requires an exact state-dict match; "
            + "; ".join(details)
            + "."
        )
    return model


__all__ = [
    "StrictModelLoadError",
    "load_checkpoint_into_model",
    "load_model",
    "load_model_strict",
]
