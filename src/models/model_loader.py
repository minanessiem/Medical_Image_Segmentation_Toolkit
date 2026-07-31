"""Single-model construction and loading shared by repository consumers."""

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.diffusion.diffusion import Diffusion
from src.models.checkpoint_loading import load_model_state_dict_compat
from src.models.model_factory import build_model


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
