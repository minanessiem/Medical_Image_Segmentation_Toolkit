"""
Model and checkpoint loading utilities for repository-model evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.models.model_loader import (
    load_checkpoint_into_model as load_shared_checkpoint_into_model,
    load_model,
)


CHECKPOINT_DIRS: Tuple[str, ...] = (
    "models/best",
    "models/checkpoints",
    "models/checkpoint",
    "models",
)


def find_checkpoint(
    run_dir: Path,
    model_name: str,
    use_ema: bool = False,
) -> Path:
    """
    Find a checkpoint in the repository's standard run-output locations.
    """
    run_dir = Path(run_dir)
    normalized_name = _strip_pth_suffix(model_name)
    searched_paths: List[Path] = []

    for relative_dir in CHECKPOINT_DIRS:
        checkpoint_dir = run_dir / relative_dir
        if use_ema:
            exact_candidates = [
                checkpoint_dir / f"{normalized_name}_ema.pth",
                checkpoint_dir / f"{normalized_name}.ema.pth",
            ]
            searched_paths.extend(exact_candidates)
            for candidate in exact_candidates:
                if candidate.exists():
                    return candidate

            pattern = f"{normalized_name}_ema_*.pth"
            searched_paths.append(checkpoint_dir / pattern)
            matches = sorted(checkpoint_dir.glob(pattern)) if checkpoint_dir.exists() else []
            if matches:
                return matches[0]
        else:
            candidate = checkpoint_dir / f"{normalized_name}.pth"
            searched_paths.append(candidate)
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        _format_checkpoint_not_found_message(
            model_name=normalized_name,
            use_ema=use_ema,
            searched_paths=searched_paths,
        )
    )


def build_model_for_evaluation(
    cfg: DictConfig,
    checkpoint_path: Path,
    device: str | torch.device,
) -> nn.Module:
    """
    Build the configured model/adapter and load checkpoint weights.

    The returned module exposes the repository's evaluation-time inference API,
    including ``sample(conditioned_image, ...)`` for diffusion and
    discriminative adapters.
    """
    model, missing_keys, unexpected_keys = load_model(
        cfg,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    _log_state_dict_diagnostics(missing_keys, unexpected_keys)
    return model


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: Path,
    device: str | torch.device,
) -> Tuple[List[str], List[str]]:
    """
    Load a checkpoint into ``model`` using repository prefix compatibility.
    """
    report = load_shared_checkpoint_into_model(
        model,
        checkpoint_path,
        device=device,
    )
    return report


def resolve_diffusion_type(cfg: DictConfig) -> str:
    """
    Resolve ``diffusion.type`` with the repository's discriminative default.
    """
    value = OmegaConf.select(cfg, "diffusion.type", default="Discriminative")
    return str(value)


def is_discriminative_config(cfg: DictConfig) -> bool:
    """
    Return whether the config requests the discriminative adapter path.
    """
    return resolve_diffusion_type(cfg) == "Discriminative"


def _strip_pth_suffix(model_name: str) -> str:
    name = str(model_name).strip()
    if name.endswith(".pth"):
        return name[:-4]
    return name


def _format_checkpoint_not_found_message(
    model_name: str,
    use_ema: bool,
    searched_paths: Sequence[Path],
) -> str:
    checkpoint_kind = "EMA checkpoint" if use_ema else "checkpoint"
    searched = "\n".join(f"  - {path}" for path in searched_paths)
    return (
        f"{checkpoint_kind} not found for model '{model_name}'.\n"
        f"Searched paths:\n{searched}"
    )


def _log_state_dict_diagnostics(
    missing_keys: Sequence[str],
    unexpected_keys: Sequence[str],
) -> None:
    if not missing_keys and not unexpected_keys:
        print("  Loaded checkpoint state dict successfully.")
        return
    if missing_keys:
        preview = list(missing_keys[:5])
        print(f"  Missing checkpoint keys ({len(missing_keys)}): {preview}")
    if unexpected_keys:
        preview = list(unexpected_keys[:5])
        print(f"  Unexpected checkpoint keys ({len(unexpected_keys)}): {preview}")
