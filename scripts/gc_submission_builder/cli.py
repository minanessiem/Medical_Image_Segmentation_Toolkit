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
from scripts.gc_submission_builder.container_builder import (
    build_container_image,
    save_container_image,
    test_container_image,
)
from scripts.gc_submission_builder.container_config import (
    DEFAULT_CONTAINER_CONFIG_PATH,
    load_container_build_config,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build independently replaceable Grand Challenge artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    model = subparsers.add_parser(
        "build-model",
        help="Build and load-test a model artifact archive.",
    )
    _add_model_arguments(model)

    image = subparsers.add_parser(
        "build-image",
        help="Build and inspect the model-independent container image.",
    )
    _add_container_arguments(image)

    test = subparsers.add_parser(
        "test",
        help="Run the isolated Grand Challenge HTTP lifecycle.",
    )
    _add_container_arguments(test)
    test.add_argument("--model-dir", type=Path, required=True)
    test.add_argument("--input-dir", type=Path, required=True)
    test.add_argument("--test-output-dir", type=Path, required=True)
    test.add_argument("--readiness-timeout-seconds", type=int, default=300)

    save = subparsers.add_parser(
        "save",
        help="Save the inspected image as a gzip-compressed Docker archive.",
    )
    _add_container_arguments(save)

    all_parser = subparsers.add_parser(
        "build-all",
        help="Build the model archive and build/save the independent image.",
    )
    _add_model_arguments(all_parser)
    _add_container_arguments(all_parser, prefixed=True)
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


def _add_container_arguments(
    parser: argparse.ArgumentParser,
    *,
    prefixed: bool = False,
) -> None:
    option_prefix = "container-" if prefixed else ""
    destination_prefix = "container_" if prefixed else ""
    parser.add_argument(
        f"--{option_prefix}config",
        dest=f"{destination_prefix}config",
        type=Path,
        default=DEFAULT_CONTAINER_CONFIG_PATH,
    )
    for option, value_type in (
        ("image-name", str),
        ("image-tag", str),
        ("dockerfile", Path),
        ("interface-manifest", Path),
        ("output-dir", Path),
        ("archive-name", str),
    ):
        rendered = f"--{option_prefix}{option}"
        destination = f"{destination_prefix}{option.replace('-', '_')}"
        parser.add_argument(rendered, dest=destination, type=value_type)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.command == "build-model":
            model_result = _build_model_from_args(args)
            _print_model_result(model_result)
        elif args.command in {"build-image", "test", "save"}:
            container_config = _load_container_from_args(args)
            if args.command == "build-image":
                image_result = build_container_image(container_config)
                _print_image_result(image_result)
            elif args.command == "test":
                test_result = test_container_image(
                    container_config,
                    model_dir=args.model_dir,
                    input_dir=args.input_dir,
                    output_dir=args.test_output_dir,
                    readiness_timeout_seconds=args.readiness_timeout_seconds,
                )
                print("Grand Challenge container lifecycle passed")
                print(f"Output:              {test_result.output_path}")
                if test_result.runtime_log.strip():
                    print("Container runtime log:")
                    print(test_result.runtime_log.rstrip())
            else:
                archive_path = save_container_image(container_config)
                print("Grand Challenge container image save passed")
                print(f"Archive:             {archive_path}")
        else:
            model_result = _build_model_from_args(args)
            container_config = _load_container_from_args(args, prefixed=True)
            image_result = build_container_image(container_config)
            archive_path = save_container_image(container_config)
            _print_model_result(model_result)
            _print_image_result(image_result)
            print(f"Image archive:       {archive_path}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def _build_model_from_args(args: argparse.Namespace):
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
    return build_model_artifact(config)


def build_model_artifact(config):
    """Lazily enter the model builder so image-only commands stay lightweight."""

    from scripts.gc_submission_builder.model_artifact import (
        build_model_artifact as build_model_artifact_impl,
    )

    return build_model_artifact_impl(config)


def _load_container_from_args(
    args: argparse.Namespace,
    *,
    prefixed: bool = False,
):
    prefix = "container_" if prefixed else ""
    return load_container_build_config(
        getattr(args, f"{prefix}config"),
        overrides={
            "image_name": getattr(args, f"{prefix}image_name"),
            "image_tag": getattr(args, f"{prefix}image_tag"),
            "dockerfile": getattr(args, f"{prefix}dockerfile"),
            "interface_manifest": getattr(args, f"{prefix}interface_manifest"),
            "output_dir": getattr(args, f"{prefix}output_dir"),
            "archive_name": getattr(args, f"{prefix}archive_name"),
        },
    )


def _print_model_result(result) -> None:
    print("Grand Challenge model artifact build passed")
    print(f"Artifact directory: {result.artifact_dir}")
    print(f"Archive:            {result.archive_path}")
    print(f"Build report:       {result.report_path}")


def _print_image_result(result) -> None:
    print("Grand Challenge container image build passed")
    print(f"Image:               {result.inspection.image_reference}")
    print(f"Image ID:            {result.inspection.image_id}")
    print(f"Build report:        {result.report_path}")


if __name__ == "__main__":
    raise SystemExit(main())
