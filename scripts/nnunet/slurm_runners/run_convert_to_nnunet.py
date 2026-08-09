#!/usr/bin/env python3
"""
Submit dataset conversion to nnU-Net format as a SLURM job.

This script wraps convert_to_nnunet.py and submits it
as a non-interactive SLURM job.

Usage:
    python3 -m scripts.nnunet.slurm_runners.run_convert_to_nnunet \
        [--time 02:00:00] [--cpus 12] [--mem 32G] \
        --convert-config-name nnunet/convert/isles24_cluster_baseline \
        [hydra_overrides...] \
        [--dry-run]

Examples:
    # Cluster conversion (test mode)
    python3 -m scripts.nnunet.slurm_runners.run_convert_to_nnunet \
        --convert-config-name nnunet/convert/isles24_cluster_baseline

    # Full export with val_full as test subset (split-based datasets)
    python3 -m scripts.nnunet.slurm_runners.run_convert_to_nnunet \
        --convert-config-name nnunet/convert/isles26_cluster_t1raw \
        nnunet.test=false \
        dataset.active_subsets.val=val_full \
        dataset.active_subsets.sample=val_full

    # Custom resources then overrides
    python3 -m scripts.nnunet.slurm_runners.run_convert_to_nnunet \
        --convert-config-name nnunet/convert/isles26_cluster_3d_t1raw \
        --time 04:00:00 --cpus 12 --mem 32G \
        nnunet.test=false

    # Dry run to see what would be submitted
    python3 -m scripts.nnunet.slurm_runners.run_convert_to_nnunet \
        --convert-config-name nnunet/convert/isles24_cluster_baseline \
        nnunet.test=false --dry-run
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


# Default resource configuration for dataset conversion
# CPU/IO bound task - no heavy GPU usage needed
CONVERT_DEFAULTS = {
    **CPU_RESOURCE_PROFILE,
    "time": "02:00:00",
}


def _sanitize_job_token(value: str) -> str:
    """Sanitize a token for safe SLURM job-name usage."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown"


def _resolve_dataset_identity(config_name: str, hydra_overrides: list[str]) -> tuple[str, str]:
    """
    Resolve nnUNet dataset identity from composed config.

    Uses the same custom YAML composer as SLURM training launch so this works
    without Hydra/OmegaConf on the login node.
    """
    composed = load_config(config_name, hydra_overrides, resolve_final=True)

    nnunet_cfg = composed.get("nnunet", {}) if isinstance(composed, dict) else {}
    dataset_cfg = composed.get("dataset", {}) if isinstance(composed, dict) else {}
    dataset_nnunet = dataset_cfg.get("nnunet", {}) if isinstance(dataset_cfg, dict) else {}

    dataset_id = nnunet_cfg.get("dataset_id") or dataset_nnunet.get("dataset_id")
    dataset_name = nnunet_cfg.get("dataset_name") or dataset_nnunet.get("dataset_name")

    if not dataset_id or not dataset_name:
        raise ValueError(
            "Could not resolve nnUNet dataset identity from config. "
            "Expected nnunet.dataset_id and nnunet.dataset_name (or dataset.nnunet.*)."
        )

    return _sanitize_job_token(str(dataset_id)), _sanitize_job_token(str(dataset_name))


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Submit dataset conversion to nnU-Net as SLURM job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # SLURM resource arguments (first)
    slurm_group = parser.add_argument_group("SLURM resources")
    slurm_group.add_argument(
        "--partition",
        type=str,
        default=None,
        help=f"SLURM partition (default: {CONVERT_DEFAULTS['partition']})",
    )
    slurm_group.add_argument(
        "--qos",
        type=str,
        default=None,
        help=f"SLURM QoS (default: {CONVERT_DEFAULTS['qos']})",
    )
    slurm_group.add_argument(
        "--time",
        type=str,
        default=CONVERT_DEFAULTS["time"],
        help=f"Time limit (default: {CONVERT_DEFAULTS['time']})",
    )
    slurm_group.add_argument(
        "--cpus",
        type=int,
        default=CONVERT_DEFAULTS["cpus_per_task"],
        help=f"CPUs per task (default: {CONVERT_DEFAULTS['cpus_per_task']})",
    )
    slurm_group.add_argument(
        "--mem",
        type=str,
        default=CONVERT_DEFAULTS["mem"],
        help=f"Memory allocation (default: {CONVERT_DEFAULTS['mem']})",
    )

    # Configuration (second)
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--convert-config-name",
        "--config-name",
        dest="convert_config_name",
        type=str,
        required=True,
        help=(
            "Conversion config name (required), e.g. nnunet/convert/isles24_cluster_baseline. "
            "--config-name is accepted as a compatibility alias."
        ),
    )

    # Hydra overrides as positional (third)
    parser.add_argument(
        "hydra_overrides",
        nargs="*",
        help=(
            "Hydra config overrides "
            "(e.g., nnunet.test=false dataset.active_subsets.val=val_full "
            "dataset.active_subsets.sample=val_full)"
        ),
    )

    # Control arguments (last)
    control_group = parser.add_argument_group("Control")
    control_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print job configuration without submitting",
    )

    return parser.parse_args()


def build_python_command(args: argparse.Namespace) -> str:
    """Build the Python command to run dataset conversion."""
    cmd_parts = [
        "python3 -m scripts.nnunet.convert_to_nnunet",
        f"--config-name={args.convert_config_name}",
    ]

    # Add Hydra overrides
    if args.hydra_overrides:
        cmd_parts.extend(args.hydra_overrides)

    return " ".join(cmd_parts)


def main():
    args = parse_arguments()

    # Build the Python command
    python_command = build_python_command(args)

    # Resolve dataset identity for traceable job naming.
    try:
        dataset_id, dataset_name = _resolve_dataset_identity(args.convert_config_name, args.hydra_overrides)
    except Exception as exc:
        print(f"[WARN] Could not resolve dataset identity for job name: {exc}")
        dataset_id, dataset_name = "unknown", "unknown"

    # Create job configuration
    config = BASE_CONFIG.copy()
    config.update(CONVERT_DEFAULTS)

    # Override with conversion-specific settings
    config["python_command"] = python_command
    config["cpus_per_task"] = args.cpus
    config["mem"] = args.mem
    config["time"] = args.time

    if args.partition:
        config["partition"] = args.partition
    if args.qos:
        config["qos"] = args.qos

    # Create job name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_tag = f"d{dataset_id}_{dataset_name}"
    config["job_name"] = f"nnunet_convert_{dataset_tag}_{timestamp}"

    # Set up log directory
    config["logdir_name"] = os.path.join(
        "nnunet_jobs",
        "convert",
        f"convert_{dataset_tag}_{timestamp}",
    )
    config = update_logdir_paths(config)

    # Print job summary
    print("\n" + "=" * 60)
    print("Dataset -> nnU-Net Conversion SLURM Job")
    print("=" * 60)
    print(f"  Config:      {args.convert_config_name}")
    if args.hydra_overrides:
        print(f"  Overrides:   {' '.join(args.hydra_overrides)}")
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

        # Show what the script would look like
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

    # Submit the job
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
