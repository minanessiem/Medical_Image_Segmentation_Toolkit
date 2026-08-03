"""Command-line entrypoint for Grand Challenge submission artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from scripts.gc_submission_builder.build_config import (
    DEFAULT_CONFIG_PATH,
    load_model_artifact_build_config,
)
from scripts.gc_submission_builder.model_artifact import build_model_artifact


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build independently replaceable Grand Challenge artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("build-model", "Build and load-test a model artifact archive."),
        (
            "build-all",
            "Build every artifact currently implemented; Cut 10 adds the image.",
        ),
    ):
        child = subparsers.add_parser(command, help=help_text)
        _add_model_arguments(child)
    return parser


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--use-ema", action="store_true", default=None)
    parser.add_argument("--inference-policy", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--archive-name")
    parser.add_argument("--validation-device")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        config = load_model_artifact_build_config(
            args.config,
            overrides={
                "run_dir": args.run_dir,
                "checkpoint": args.checkpoint,
                "use_ema": args.use_ema,
                "inference_policy": args.inference_policy,
                "output_dir": args.output_dir,
                "archive_name": args.archive_name,
                "validation_device": args.validation_device,
            },
        )
        result = build_model_artifact(config)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Grand Challenge model artifact build passed")
    print(f"Artifact directory: {result.artifact_dir}")
    print(f"Archive:            {result.archive_path}")
    print(f"Build report:       {result.report_path}")
    if args.command == "build-all":
        print("Container image build remains intentionally deferred to Cut 10.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
