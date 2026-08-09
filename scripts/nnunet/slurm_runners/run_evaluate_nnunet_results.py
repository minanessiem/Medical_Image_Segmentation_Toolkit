#!/usr/bin/env python3
"""
Submit nnU-Net evaluation as a SLURM job.

This script wraps evaluate_nnunet_results.py and submits it
as a non-interactive SLURM job.

Usage:
    python3 -m scripts.nnunet.slurm_runners.run_evaluate_nnunet_results \
        [--time 02:00:00] [--cpus 12] [--mem 32G] \
        --convert-config-name nnunet/convert/isles26_cluster_3d_t1raw \
        --eval-config-name nnunet/eval/volumes_3d \
        [hydra_overrides...] \
        [--dry-run]

Examples:
    # 3D evaluation dry-run
    python3 -m scripts.nnunet.slurm_runners.run_evaluate_nnunet_results \
        --convert-config-name nnunet/convert/isles26_cluster_3d_t1raw \
        --eval-config-name nnunet/eval/volumes_3d \
        nnunet_eval.pred_dir=/mnt/outputs/nnunet_results/predictionsTs_isles26_t1_raw \
        nnunet_eval.gt_dir=/mnt/datasets/nnunet_raw/Dataset260_isles26_t1_raw/labelsTs \
        nnunet_eval.output_dir=/mnt/outputs/nnunet_results/predictionsTs_isles26_t1_raw \
        --dry-run
"""

import argparse
import os
import re
import sys
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.slurm.base_run_config import BASE_CONFIG, SLURM_TEMPLATE, update_logdir_paths
from scripts.slurm.job_runner import SlurmJobRunner
from scripts.slurm.single_job_runner import load_config
from scripts.nnunet.slurm_runners.nnunet_env import CPU_RESOURCE_PROFILE


# Default resource configuration for evaluation
# CPU/IO bound task, can be heavier for native 3D metrics.
EVAL_DEFAULTS = {
    **CPU_RESOURCE_PROFILE,
    "time": "02:00:00",
}


def _sanitize_job_token(value: str) -> str:
    """Sanitize a token for safe SLURM job-name usage."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown"


def _resolve_dataset_identity(convert_config_name: str, hydra_overrides: list[str]) -> tuple[str, str]:
    """
    Resolve nnUNet dataset identity from composed conversion config.
    """
    composed = load_config(convert_config_name, hydra_overrides, resolve_final=True)

    nnunet_cfg = composed.get("nnunet", {}) if isinstance(composed, dict) else {}
    dataset_cfg = composed.get("dataset", {}) if isinstance(composed, dict) else {}
    dataset_nnunet = dataset_cfg.get("nnunet", {}) if isinstance(dataset_cfg, dict) else {}

    dataset_id = nnunet_cfg.get("dataset_id") or dataset_nnunet.get("dataset_id")
    dataset_name = nnunet_cfg.get("dataset_name") or dataset_nnunet.get("dataset_name")

    if not dataset_id or not dataset_name:
        raise ValueError(
            "Could not resolve nnUNet dataset identity from conversion config. "
            "Expected nnunet.dataset_id and nnunet.dataset_name (or dataset.nnunet.*)."
        )

    return _sanitize_job_token(str(dataset_id)), _sanitize_job_token(str(dataset_name))


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Submit nnU-Net evaluation as SLURM job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # SLURM resource arguments
    slurm_group = parser.add_argument_group("SLURM resources")
    slurm_group.add_argument(
        "--partition",
        type=str,
        default=None,
        help=f"SLURM partition (default: {EVAL_DEFAULTS['partition']})",
    )
    slurm_group.add_argument(
        "--qos",
        type=str,
        default=None,
        help=f"SLURM QoS (default: {EVAL_DEFAULTS['qos']})",
    )
    slurm_group.add_argument(
        "--time",
        type=str,
        default=EVAL_DEFAULTS["time"],
        help=f"Time limit (default: {EVAL_DEFAULTS['time']})",
    )
    slurm_group.add_argument(
        "--cpus",
        type=int,
        default=EVAL_DEFAULTS["cpus_per_task"],
        help=f"CPUs per task (default: {EVAL_DEFAULTS['cpus_per_task']})",
    )
    slurm_group.add_argument(
        "--mem",
        type=str,
        default=EVAL_DEFAULTS["mem"],
        help=f"Memory allocation (default: {EVAL_DEFAULTS['mem']})",
    )

    # Configuration
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--convert-config-name",
        type=str,
        required=True,
        help="Conversion config name (required), e.g. nnunet/convert/isles26_cluster_3d_t1raw.",
    )
    config_group.add_argument(
        "--eval-config-name",
        type=str,
        required=True,
        help="Evaluation config name (required), e.g. nnunet/eval/volumes_3d.",
    )

    # Hydra-style overrides as positional
    parser.add_argument(
        "hydra_overrides",
        nargs="*",
        help=(
            "Hydra-style overrides passed to evaluate_nnunet_results "
            "(e.g., nnunet_eval.pred_dir=..., nnunet_eval.gt_dir=...)."
        ),
    )

    # Control arguments
    control_group = parser.add_argument_group("Control")
    control_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print job configuration without submitting",
    )

    return parser.parse_args()


def build_python_command(args: argparse.Namespace) -> str:
    """Build the Python command to run evaluation."""
    cmd_parts = [
        "python3 -m scripts.nnunet.evaluate_nnunet_results",
        f"--convert-config-name={args.convert_config_name}",
        f"--eval-config-name={args.eval_config_name}",
    ]

    if args.hydra_overrides:
        cmd_parts.extend(args.hydra_overrides)

    return " ".join(cmd_parts)


def main() -> None:
    args = parse_arguments()

    python_command = build_python_command(args)

    # Resolve dataset identity for traceable job naming.
    try:
        dataset_id, dataset_name = _resolve_dataset_identity(args.convert_config_name, args.hydra_overrides)
    except Exception as exc:
        print(f"[WARN] Could not resolve dataset identity for job name: {exc}")
        dataset_id, dataset_name = "unknown", "unknown"

    config = BASE_CONFIG.copy()
    config.update(EVAL_DEFAULTS)
    config["python_command"] = python_command
    config["cpus_per_task"] = args.cpus
    config["mem"] = args.mem
    config["time"] = args.time

    if args.partition:
        config["partition"] = args.partition
    if args.qos:
        config["qos"] = args.qos

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_tag = f"d{dataset_id}_{dataset_name}"
    config["job_name"] = f"nnunet_evaluate_{dataset_tag}_{timestamp}"
    config["logdir_name"] = os.path.join(
        "nnunet_jobs",
        "evaluate",
        f"evaluate_{dataset_tag}_{timestamp}",
    )
    config = update_logdir_paths(config)

    print("\n" + "=" * 60)
    print("nnU-Net Evaluation SLURM Job")
    print("=" * 60)
    print(f"  Convert config: {args.convert_config_name}")
    print(f"  Eval config:    {args.eval_config_name}")
    if args.hydra_overrides:
        print(f"  Overrides:      {' '.join(args.hydra_overrides)}")
    print("-" * 60)
    print("  Resources:")
    print(f"    Partition: {config['partition']}")
    print(f"    QoS:       {config['qos']}")
    print(f"    GPUs:      {config['gpus']}")
    print(f"    CPUs:      {args.cpus}")
    print(f"    Memory:    {args.mem}")
    print(f"    Time:      {args.time}")
    print("-" * 60)
    print("  Command:")
    print(f"    {python_command}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would submit the following SLURM job:\n")

        config["output_file"] = f"{config['host_logdir']}/output.out"
        config["error_file"] = f"{config['host_logdir']}/error.err"

        print("Configuration:")
        for key in [
            "job_name",
            "partition",
            "qos",
            "gpus",
            "time",
            "cpus_per_task",
            "mem",
            "host_logdir",
            "python_command",
        ]:
            print(f"  {key}: {config[key]}")

        print("\n[DRY RUN] No job submitted.")
        return

    runner = SlurmJobRunner(config)
    job_id = runner.submit_job(config, SLURM_TEMPLATE, dry_run=False)

    if job_id:
        print(f"\n✓ Job submitted successfully: {job_id}")
        print(f"  Monitor with: squeue -j {job_id}")
        print(f"  Logs at: {config['host_logdir']}")
    else:
        print("\n✗ Job submission failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
