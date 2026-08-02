#!/usr/bin/env python3
"""
Config-driven repository-model evaluation entrypoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from omegaconf import DictConfig, OmegaConf

from scripts.evaluation.core.evaluation_pipeline import run_model_evaluation
from scripts.evaluation.core.model_config import (
    apply_evaluation_overrides,
    load_evaluation_config,
    load_run_config,
    merge_evaluation_config,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a repository-trained segmentation model.",
    )
    parser.add_argument(
        "--evaluation-config-name",
        type=str,
        default="default",
        help=(
            "Evaluation config preset under configs/evaluation/ without .yaml "
            "(default: default)."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Hydra-style key=value overrides. Required: evaluation.run_dir and "
            "evaluation.model_name unless provided by the evaluation preset."
        ),
    )
    return parser


def compose_evaluation_config(
    evaluation_config_name: str,
    overrides: Iterable[str],
) -> DictConfig:
    """
    Compose saved run config + evaluation preset + CLI overrides.
    """
    override_list = list(overrides)
    run_dir = _extract_run_dir_override(override_list)
    if run_dir is None:
        preset_cfg = load_evaluation_config(evaluation_config_name)
        run_dir_value = OmegaConf.select(preset_cfg, "evaluation.run_dir", default=None)
        if _is_set(run_dir_value):
            run_dir = Path(str(run_dir_value))
    if run_dir is None:
        raise ValueError(
            "evaluation.run_dir is required. Provide it as "
            "`evaluation.run_dir=/path/to/run`."
        )

    run_cfg = load_run_config(run_dir)
    eval_cfg = load_evaluation_config(evaluation_config_name)
    cfg = merge_evaluation_config(run_cfg, eval_cfg)
    cfg = apply_evaluation_overrides(cfg, override_list)
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        cfg = compose_evaluation_config(
            evaluation_config_name=args.evaluation_config_name,
            overrides=args.overrides,
        )
        results = run_model_evaluation(cfg)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("Repository Model Evaluation")
    print("=" * 70)
    print(f"Output dir: {results['output_dir']}")
    print("Outputs:")
    for key in (
        "json_path",
        "volume_csv_path",
        "per_case_csv_path",
        "oracle_csv_path",
        "config_path",
        "summary_path",
    ):
        value = results.get(key)
        if value:
            print(f"  {value}")
    print("-" * 70)
    summary_text = results.get("summary_text")
    if summary_text:
        print(summary_text)
    return 0


def _extract_run_dir_override(overrides: Iterable[str]) -> Optional[Path]:
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        if key.strip() == "evaluation.run_dir" and _is_set(value):
            return Path(value.strip())
    return None


def _is_set(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


if __name__ == "__main__":
    sys.exit(main())
