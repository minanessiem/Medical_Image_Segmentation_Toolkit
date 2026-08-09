#!/usr/bin/env python3
"""
Generic SLURM runner for nnU-Net CLI commands.

Submits nnU-Net commands (preprocess, train, predict) as non-interactive
SLURM jobs with proper environment setup.

Usage:
    python scripts/nnunet/slurm_runners/run_nnunet_command.py <command> [args] [--dry-run]

Commands:
    preprocess  Run nnUNetv2_plan_and_preprocess
    train       Run nnUNetv2_train  
    predict     Run nnUNetv2_predict

Examples:
    # Plan and preprocess dataset 050 with verification
    python scripts/nnunet/slurm_runners/run_nnunet_command.py preprocess \\
        -d 050 --verify --dry-run

    # Train 2D model on all folds
    python scripts/nnunet/slurm_runners/run_nnunet_command.py train \\
        -d 050 -c 2d -f all

    # Train with custom time limit
    python scripts/nnunet/slurm_runners/run_nnunet_command.py train \\
        -d 050 -c 2d -f all --time 72:00:00

    # Predict on test set
    python scripts/nnunet/slurm_runners/run_nnunet_command.py predict \\
        -d 050 -i /mnt/datasets/nnunet_raw/Dataset050_isles24/imagesTs \\
        -o /mnt/outputs/nnunet_results/predictions
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Callable, Dict, Any

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.slurm.base_run_config import BASE_CONFIG, update_logdir_paths
from scripts.slurm.job_runner import SlurmJobRunner
from scripts.nnunet.slurm_runners.nnunet_env import (
    NNUNET_ENV,
    NNUNET_SLURM_TEMPLATE,
    COMMAND_DEFAULTS,
    get_env_export_string,
)


# ============================================================================
# Preprocess Command
# ============================================================================

def add_preprocess_args(subparser: argparse.ArgumentParser) -> None:
    """Add arguments for nnUNetv2_plan_and_preprocess."""
    subparser.add_argument(
        '-d', '--dataset-id',
        required=True,
        help='Dataset ID (e.g., 050)'
    )
    subparser.add_argument(
        '-c', '--configurations',
        default='2d',
        help='Configurations to preprocess (default: 2d)'
    )
    subparser.add_argument(
        '--verify',
        action='store_true',
        help='Verify dataset integrity before preprocessing'
    )
    subparser.add_argument(
        '--clean',
        action='store_true',
        help='Clean existing preprocessed data first'
    )
    subparser.add_argument(
        '-np', '--num-processes',
        type=int,
        default=None,
        help='Number of parallel processes (default: auto)'
    )


def build_preprocess_command(args: argparse.Namespace) -> str:
    """Build nnUNetv2_plan_and_preprocess command string."""
    cmd_parts = [
        'nnUNetv2_plan_and_preprocess',
        f'-d {args.dataset_id}',
        f'-c {args.configurations}',
    ]
    
    if args.verify:
        cmd_parts.append('--verify_dataset_integrity')
    if args.clean:
        cmd_parts.append('--clean')
    if args.num_processes:
        cmd_parts.append(f'-np {args.num_processes}')
    
    return ' '.join(cmd_parts)


# ============================================================================
# Train Command
# ============================================================================

def add_train_args(subparser: argparse.ArgumentParser) -> None:
    """Add arguments for nnUNetv2_train."""
    subparser.add_argument(
        '-d', '--dataset-id',
        required=True,
        help='Dataset ID (e.g., 050)'
    )
    subparser.add_argument(
        '-c', '--configuration',
        default='2d',
        help='Configuration (default: 2d)'
    )
    subparser.add_argument(
        '-f', '--fold',
        default='all',
        help='Fold to train (default: all)'
    )
    subparser.add_argument(
        '--npz',
        action='store_true',
        help='Save softmax outputs during validation'
    )
    subparser.add_argument(
        '-tr', '--trainer',
        default=None,
        help='Trainer class name (default: nnUNetTrainer)'
    )
    subparser.add_argument(
        '-p', '--plans',
        default=None,
        help='Plans identifier (default: nnUNetPlans)'
    )


def build_train_command(args: argparse.Namespace) -> str:
    """Build nnUNetv2_train command string."""
    cmd_parts = [
        'nnUNetv2_train',
        args.dataset_id,
        args.configuration,
        args.fold,
    ]
    
    if args.npz:
        cmd_parts.append('--npz')
    if args.trainer:
        cmd_parts.append(f'-tr {args.trainer}')
    if args.plans:
        cmd_parts.append(f'-p {args.plans}')
    
    return ' '.join(cmd_parts)


# ============================================================================
# Predict Command
# ============================================================================

def add_predict_args(subparser: argparse.ArgumentParser) -> None:
    """Add arguments for nnUNetv2_predict."""
    subparser.add_argument(
        '-i', '--input-dir',
        required=True,
        help='Input folder with images to predict'
    )
    subparser.add_argument(
        '-o', '--output-dir',
        required=True,
        help='Output folder for predictions'
    )
    subparser.add_argument(
        '-d', '--dataset-id',
        required=True,
        help='Dataset ID (e.g., 050)'
    )
    subparser.add_argument(
        '-c', '--configuration',
        default='2d',
        help='Configuration (default: 2d)'
    )
    subparser.add_argument(
        '-f', '--fold',
        default='all',
        help='Fold(s) to use for prediction (default: all)'
    )
    subparser.add_argument(
        '--save-probabilities',
        action='store_true',
        help='Save softmax probabilities'
    )
    subparser.add_argument(
        '-chk', '--checkpoint',
        default='checkpoint_final.pth',
        help='Checkpoint name (default: checkpoint_final.pth)'
    )
    subparser.add_argument(
        '-npp', '--num-processes-preprocessing',
        type=int,
        default=None,
        help='Num processes for preprocessing'
    )
    subparser.add_argument(
        '-nps', '--num-processes-segmentation',
        type=int,
        default=None,
        help='Num processes for segmentation export'
    )


def build_predict_command(args: argparse.Namespace) -> str:
    """Build nnUNetv2_predict command string."""
    cmd_parts = [
        'nnUNetv2_predict',
        f'-i {args.input_dir}',
        f'-o {args.output_dir}',
        f'-d {args.dataset_id}',
        f'-c {args.configuration}',
        f'-f {args.fold}',
    ]
    
    if args.save_probabilities:
        cmd_parts.append('--save_probabilities')
    if args.checkpoint != 'checkpoint_final.pth':
        cmd_parts.append(f'-chk {args.checkpoint}')
    if args.num_processes_preprocessing:
        cmd_parts.append(f'-npp {args.num_processes_preprocessing}')
    if args.num_processes_segmentation:
        cmd_parts.append(f'-nps {args.num_processes_segmentation}')
    
    return ' '.join(cmd_parts)


# ============================================================================
# Command Registry
# ============================================================================

COMMANDS: Dict[str, Dict[str, Any]] = {
    "preprocess": {
        "add_args": add_preprocess_args,
        "build_cmd": build_preprocess_command,
        "defaults": COMMAND_DEFAULTS["preprocess"],
        "help": "Run nnUNetv2_plan_and_preprocess",
    },
    "train": {
        "add_args": add_train_args,
        "build_cmd": build_train_command,
        "defaults": COMMAND_DEFAULTS["train"],
        "help": "Run nnUNetv2_train",
    },
    "predict": {
        "add_args": add_predict_args,
        "build_cmd": build_predict_command,
        "defaults": COMMAND_DEFAULTS["predict"],
        "help": "Run nnUNetv2_predict",
    },
}


# ============================================================================
# Shared Argument Handling
# ============================================================================

def add_slurm_args(parser: argparse.ArgumentParser, defaults: Dict[str, Any]) -> None:
    """Add SLURM resource arguments with command-specific defaults."""
    group = parser.add_argument_group('SLURM resources')
    group.add_argument(
        '--partition',
        type=str,
        default=None,
        help=f'SLURM partition (default: {defaults["partition"]})'
    )
    group.add_argument(
        '--qos',
        type=str,
        default=None,
        help=f'SLURM QoS (default: {defaults["qos"]})'
    )
    group.add_argument(
        '--time',
        type=str,
        default=defaults["time"],
        help=f'Time limit (default: {defaults["time"]})'
    )
    group.add_argument(
        '--cpus',
        type=int,
        default=defaults["cpus_per_task"],
        help=f'CPUs per task (default: {defaults["cpus_per_task"]})'
    )
    group.add_argument(
        '--mem',
        type=str,
        default=defaults["mem"],
        help=f'Memory allocation (default: {defaults["mem"]})'
    )


def add_control_args(parser: argparse.ArgumentParser) -> None:
    """Add control arguments (must be last in help output)."""
    group = parser.add_argument_group('Control')
    group.add_argument(
        '--dry-run',
        action='store_true',
        help='Print job configuration without submitting'
    )


# ============================================================================
# Job Submission
# ============================================================================

def build_slurm_config(
    args: argparse.Namespace, 
    command: str, 
    command_name: str
) -> Dict[str, Any]:
    """Build SLURM job configuration."""
    config = BASE_CONFIG.copy()
    defaults = COMMAND_DEFAULTS[command_name]
    config.update({key: value for key, value in defaults.items() if key != 'description'})
    
    # Set command with nnU-Net env exports
    config['command'] = command
    config['nnunet_env_exports'] = get_env_export_string()
    
    # Override resources
    config['cpus_per_task'] = args.cpus
    config['mem'] = args.mem
    config['time'] = args.time
    
    if args.partition:
        config['partition'] = args.partition
    if args.qos:
        config['qos'] = args.qos
    
    # Create job name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_id = getattr(args, 'dataset_id', 'unknown')
    config['job_name'] = f'nnunet_{command_name}_{dataset_id}_{timestamp}'
    
    # Set up log directory
    config['logdir_name'] = os.path.join(
        'nnunet_jobs', command_name,
        f'{command_name}_{dataset_id}_{timestamp}'
    )
    config = update_logdir_paths(config)
    
    return config


def print_job_summary(
    args: argparse.Namespace,
    command_name: str,
    command: str,
    config: Dict[str, Any]
) -> None:
    """Print job summary before submission."""
    print("\n" + "=" * 70)
    print(f"nnU-Net {command_name.upper()} SLURM Job")
    print("=" * 70)
    
    # Command-specific info
    print(f"  Command: {command_name}")
    print(f"  Dataset: {getattr(args, 'dataset_id', 'N/A')}")
    
    if command_name == 'train':
        print(f"  Config:  {args.configuration}")
        print(f"  Fold:    {args.fold}")
    elif command_name == 'predict':
        print(f"  Input:   {args.input_dir}")
        print(f"  Output:  {args.output_dir}")
    
    print("-" * 70)
    print("  nnU-Net Environment:")
    for key, value in NNUNET_ENV.items():
        print(f"    {key}={value}")
    
    print("-" * 70)
    print("  SLURM Resources:")
    print(f"    Partition: {config['partition']}")
    print(f"    QoS:       {config['qos']}")
    print(f"    GPUs:      {config['gpus']}")
    print(f"    CPUs:      {args.cpus}")
    print(f"    Memory:    {args.mem}")
    print(f"    Time:      {args.time}")
    
    print("-" * 70)
    print(f"  Full command:")
    print(f"    {command}")
    print("=" * 70)


def submit_job(config: Dict[str, Any], dry_run: bool) -> None:
    """Submit the SLURM job."""
    if dry_run:
        print("\n[DRY RUN] Would submit the following SLURM job:\n")
        
        config['output_file'] = f"{config['host_logdir']}/output.out"
        config['error_file'] = f"{config['host_logdir']}/error.err"
        
        print("Configuration:")
        for key in ['job_name', 'partition', 'qos', 'gpus', 'time',
                    'cpus_per_task', 'mem', 'host_logdir']:
            print(f"  {key}: {config[key]}")
        
        print("\n[DRY RUN] No job submitted.")
        return
    
    runner = SlurmJobRunner(config)
    job_id = runner.submit_job(config, NNUNET_SLURM_TEMPLATE, dry_run=False)
    
    if job_id:
        print(f"\n✓ Job submitted successfully: {job_id}")
        print(f"  Monitor with: squeue -j {job_id}")
        print(f"  Logs at: {config['host_logdir']}")
    else:
        print("\n✗ Job submission failed")
        sys.exit(1)


# ============================================================================
# Main
# ============================================================================

def main():
    # Create main parser
    parser = argparse.ArgumentParser(
        description='Generic SLURM runner for nnU-Net CLI commands',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Create subparsers for each command
    subparsers = parser.add_subparsers(
        dest='command',
        title='Commands',
        description='Available nnU-Net commands',
        required=True
    )
    
    # Register each command
    for cmd_name, cmd_info in COMMANDS.items():
        subparser = subparsers.add_parser(
            cmd_name,
            help=cmd_info["help"],
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Add command-specific arguments first
        cmd_info["add_args"](subparser)
        
        # Add SLURM resource arguments
        add_slurm_args(subparser, cmd_info["defaults"])
        
        # Add control arguments last
        add_control_args(subparser)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get command info
    cmd_info = COMMANDS[args.command]
    
    # Build the nnU-Net command
    command = cmd_info["build_cmd"](args)
    
    # Build SLURM config
    config = build_slurm_config(args, command, args.command)
    
    # Print summary
    print_job_summary(args, args.command, command, config)
    
    # Submit job
    submit_job(config, args.dry_run)


if __name__ == '__main__':
    main()
