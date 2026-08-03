"""Non-production container diagnostic for either declared prediction space."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts.gc_submission_builder.runtime.inference import initialize_runtime


DEFAULT_DIAGNOSTIC_PROFILE = Path("/opt/app/runtime_profiles/gc_container_test.yaml")
PRODUCTION_OUTPUT_ROOT = Path("/output")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one mounted case without writing the production output socket."
    )
    parser.add_argument("--model-dir", type=Path, default=Path("/opt/ml/model"))
    parser.add_argument(
        "--interface-manifest",
        type=Path,
        default=Path("/opt/app/interface_manifest.yaml"),
    )
    parser.add_argument("--runtime-profile", type=Path, default=DEFAULT_DIAGNOSTIC_PROFILE)
    parser.add_argument("--input-dir", type=Path, default=Path("/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/diagnostic"))
    parser.add_argument("--device", default=os.environ.get("GC_DEVICE") or None)
    parser.add_argument(
        "--output-space",
        choices=("model_preprocessed", "native_input"),
        default=None,
        help=(
            "Diagnostic-only override of the artifact result space. This is rejected "
            "by the production runtime profile."
        ),
    )
    parser.add_argument("--retain", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    production_root = PRODUCTION_OUTPUT_ROOT.resolve()
    if output_dir == production_root or production_root in output_dir.parents:
        raise RuntimeError(
            "Diagnostics must not write data to the production /output socket or "
            "any of its descendants."
        )
    runtime = initialize_runtime(
        model_dir=args.model_dir,
        interface_manifest_path=args.interface_manifest,
        runtime_profile_path=args.runtime_profile,
        device=args.device,
        output_space_override=args.output_space,
    )
    prediction = runtime.predict(input_root=args.input_dir)
    result = prediction.result
    report = {
        "interface": prediction.interface.name,
        "runtime_profile": runtime.runtime_profile.profile,
        "inference_policy_origin": runtime.inference_policy_origin,
        "output_space": result.output_space,
        "probability_shape": list(result.probability.shape),
        "probability_dtype": str(result.probability.dtype),
        "mask_shape": list(result.mask.shape),
        "mask_dtype": str(result.mask.dtype),
        "elapsed_seconds": prediction.elapsed_seconds,
        "peak_cuda_memory_bytes": prediction.peak_cuda_memory_bytes,
        "retained": bool(args.retain),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.retain:
        if not runtime.runtime_profile.constraints.allow_intermediate_artifacts:
            raise RuntimeError(
                "Selected runtime profile does not permit diagnostic artifacts."
            )
        np.save(
            output_dir / "probability.npy",
            result.probability.detach().cpu().numpy(),
            allow_pickle=False,
        )
        np.save(
            output_dir / "mask.npy",
            result.mask.detach().cpu().numpy(),
            allow_pickle=False,
        )
    report_path = output_dir / "diagnostic_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
