"""Checkpoint-to-model loading helpers shared by training and evaluation."""

from typing import List, Tuple

import torch.nn as nn


def _extract_checkpoint_state_dict(payload: dict) -> dict:
    """Extract a model state dict from common checkpoint payload structures."""
    if not isinstance(payload, dict):
        return payload
    if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
        return payload["model_state_dict"]
    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    return payload


def _normalize_state_dict_keys_for_model(
    model: nn.Module,
    state_dict: dict,
) -> dict:
    """Normalize checkpoint keys to maximize overlap with target model keys."""
    state_dict = _extract_checkpoint_state_dict(state_dict)
    if not isinstance(state_dict, dict):
        return state_dict

    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    if model_keys & checkpoint_keys:
        return state_dict

    prefixes_to_strip = [
        "module.",
        "wrapped_model.base_model.",
        "model.model.",
        "model.",
    ]

    candidates = [state_dict]
    for prefix in prefixes_to_strip:
        if any(key.startswith(prefix) for key in state_dict):
            stripped = {
                key[len(prefix) :] if key.startswith(prefix) else key: value
                for key, value in state_dict.items()
            }
            candidates.append(stripped)

    if any(key.startswith("module.") for key in model_keys):
        candidates.append({f"module.{key}": value for key, value in state_dict.items()})

    best = state_dict
    best_overlap = len(model_keys & checkpoint_keys)
    for candidate in candidates:
        overlap = len(model_keys & set(candidate.keys()))
        if overlap > best_overlap:
            best = candidate
            best_overlap = overlap

    return best


def load_model_state_dict_compat(
    model: nn.Module,
    checkpoint_state: dict,
) -> Tuple[List[str], List[str]]:
    """Load model state with the repository's existing prefix compatibility."""
    normalized_state = _normalize_state_dict_keys_for_model(model, checkpoint_state)

    try:
        model.load_state_dict(normalized_state, strict=True)
        return [], []
    except RuntimeError:
        missing, unexpected = model.load_state_dict(normalized_state, strict=False)
        return list(missing), list(unexpected)
