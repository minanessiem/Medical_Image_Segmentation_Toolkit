"""
Configuration utilities for repository-model evaluation.

The evaluation entrypoint needs to start from a saved Hydra run config, then
layer evaluation policy and explicit overrides on top. These helpers keep that
logic out of the runtime pipeline.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import yaml
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "configs").is_dir()
)
CONFIG_ROOT = PROJECT_ROOT / "configs"
RESOLVED_CONFIG_FILENAME = "resolved_evaluation_config.yaml"
COMPLETE_POLICY_GROUPS = frozenset({"inference", "inference_runtime"})


def load_run_config(run_dir: Path) -> DictConfig:
    """
    Load a saved Hydra run config from ``<run_dir>/.hydra/config.yaml``.
    """
    config_path = Path(run_dir) / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            "Run config not found. Expected Hydra config at "
            f"{config_path}."
        )

    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)
    return cfg


def load_evaluation_config(name: str = "default") -> DictConfig:
    """
    Load an evaluation policy config from ``configs/evaluation/<name>.yaml``.
    """
    cfg, package = _load_config_group("evaluation", name)
    if _is_global_package(package):
        result = cfg
    else:
        result = OmegaConf.create({"evaluation": cfg})
    OmegaConf.set_struct(result, False)
    return result


def merge_evaluation_config(run_cfg: DictConfig, eval_cfg: DictConfig) -> DictConfig:
    """
    Merge evaluation policy into a trained run config without changing data or
    training sections unless the evaluation config explicitly contains them.
    """
    base = _mutable_copy(run_cfg)
    merged = OmegaConf.merge(base, eval_cfg)
    for group in COMPLETE_POLICY_GROUPS:
        selected = OmegaConf.select(eval_cfg, group, default=None)
        if selected is not None:
            OmegaConf.update(
                merged,
                group,
                OmegaConf.to_container(selected, resolve=False),
                merge=False,
            )
    OmegaConf.set_struct(merged, False)
    return merged


def apply_evaluation_overrides(
    cfg: DictConfig,
    overrides: Sequence[str],
) -> DictConfig:
    """
    Apply Hydra-style ``key=value`` overrides to an evaluation config.

    Supports dotted-key overrides and top-level config group overrides such as
    ``validation=sliding_window_3d_metrics_full``.
    """
    if not overrides:
        return cfg

    updated = _mutable_copy(cfg)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected key=value.")

        raw_key, raw_value = override.split("=", 1)
        key = raw_key.strip()
        value = raw_value.strip()
        if not key:
            raise ValueError(f"Invalid override '{override}'. Override key is empty.")

        if "." not in key:
            group_cfg = _try_load_config_group_override(key, value)
            if group_cfg is not None:
                if key in COMPLETE_POLICY_GROUPS:
                    selected = OmegaConf.select(group_cfg, key, default=None)
                    if selected is None:
                        raise ValueError(
                            f"Config group override {key}={value} did not define {key}."
                        )
                    OmegaConf.update(
                        updated,
                        key,
                        OmegaConf.to_container(selected, resolve=False),
                        merge=False,
                    )
                else:
                    updated = OmegaConf.merge(updated, group_cfg)
                OmegaConf.set_struct(updated, False)
                continue

        OmegaConf.update(
            updated,
            key,
            _parse_override_value(value),
            merge=True,
        )

    OmegaConf.set_struct(updated, False)
    return updated


def resolve_evaluation_output_dir(
    cfg: DictConfig,
    timestamp: Optional[str] = None,
) -> Path:
    """
    Resolve the output directory for model evaluation artifacts.
    """
    explicit = OmegaConf.select(cfg, "evaluation.output_dir", default=None)
    if _is_set(explicit):
        return Path(str(explicit))

    run_dir = OmegaConf.select(cfg, "evaluation.run_dir", default=None)
    model_name = OmegaConf.select(cfg, "evaluation.model_name", default=None)
    if not _is_set(run_dir):
        raise ValueError("evaluation.run_dir is required to resolve output_dir.")
    if not _is_set(model_name):
        raise ValueError("evaluation.model_name is required to resolve output_dir.")

    resolved_timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (
        Path(str(run_dir))
        / "analysis"
        / "evaluation_v3"
        / f"{str(model_name)}_{resolved_timestamp}"
    )


def write_resolved_evaluation_config(cfg: DictConfig, output_dir: Path) -> Path:
    """
    Persist the final composed evaluation config to ``output_dir``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / RESOLVED_CONFIG_FILENAME
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(OmegaConf.to_yaml(cfg, resolve=True))
    return output_path


def _mutable_copy(cfg: DictConfig) -> DictConfig:
    copied = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.set_struct(copied, False)
    return copied


def _is_set(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _try_load_config_group_override(group: str, name: str) -> Optional[DictConfig]:
    config_path = CONFIG_ROOT / group / f"{name}.yaml"
    if not config_path.exists():
        return None

    group_cfg, package = _load_config_group(group, name)
    if _is_global_package(package):
        return group_cfg
    return OmegaConf.create({group: group_cfg})


def _load_config_group(group: str, name: str) -> Tuple[DictConfig, Optional[str]]:
    config_path = CONFIG_ROOT / group / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config group file not found: {config_path}")

    package = _read_package_directive(config_path)
    cfg = OmegaConf.load(config_path)
    defaults = list(cfg.get("defaults", [])) if "defaults" in cfg else []
    body = _config_body_without_defaults(cfg)

    if not defaults:
        return body, package

    composed = OmegaConf.create({})
    saw_self = False
    for default in defaults:
        if default == "_self_":
            composed = OmegaConf.merge(composed, body)
            saw_self = True
            continue
        composed = OmegaConf.merge(
            composed,
            _resolve_default_config(default=default, current_group=group),
        )

    if not saw_self:
        composed = OmegaConf.merge(composed, body)

    OmegaConf.set_struct(composed, False)
    return composed, package


def _config_body_without_defaults(cfg: DictConfig) -> DictConfig:
    body = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if "defaults" in body:
        del body["defaults"]
    OmegaConf.set_struct(body, False)
    return body


def _resolve_default_config(default: Any, current_group: str) -> DictConfig:
    if isinstance(default, str):
        group_cfg, package = _load_config_group(current_group, default)
        if _is_global_package(package):
            return group_cfg
        return group_cfg

    if isinstance(default, dict) or OmegaConf.is_dict(default):
        default_map = OmegaConf.to_container(default, resolve=False)
        if not isinstance(default_map, dict) or len(default_map) != 1:
            raise ValueError(f"Invalid defaults entry: {default}")
        raw_group, name = next(iter(default_map.items()))
        group, _is_override = _normalize_default_group_key(str(raw_group))
        group_cfg, package = _load_config_group(group, str(name))
        if _is_global_package(package):
            return group_cfg
        return OmegaConf.create({group: group_cfg})

    raise ValueError(f"Unsupported defaults entry: {default}")


def _normalize_default_group_key(raw_key: str) -> Tuple[str, bool]:
    key = raw_key.strip()
    is_override = False
    if key.startswith("override "):
        is_override = True
        key = key[len("override ") :].strip()
    if key.startswith("/"):
        key = key[1:].strip()
    if not key:
        raise ValueError(f"Invalid defaults key '{raw_key}'.")
    return key, is_override


def _read_package_directive(config_path: Path) -> Optional[str]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if stripped.startswith("# @package "):
                    return stripped[len("# @package ") :].strip()
                continue
            break
    return None


def _is_global_package(package: Optional[str]) -> bool:
    return package == "_global_"


def _parse_override_value(raw_value: str) -> Any:
    try:
        return yaml.safe_load(raw_value)
    except Exception:
        return raw_value
