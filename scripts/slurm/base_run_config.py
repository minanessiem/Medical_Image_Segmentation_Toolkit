import os

HOME_DIR = os.path.expanduser("~")

# Get the SSD_STORE environment variable
ssd_store_path = os.environ.get("SSD_STORE")

def update_logdir_paths(config: dict) -> dict:
    """Update the logdir paths in the config based on current settings."""
    config["container_logdir"] = os.path.join(config["container_outputs_dir"], config["logdir_name"])
    config["host_logdir"] = os.path.join(config["host_outputs_dir"], config["logdir_name"])
    return config

BASE_CONFIG = {
    "partition": "mcml-dgx-a100-40x8",
    "qos": "mcml",
    "gpus": 1,
    "gpu_directive": "#SBATCH --gres=gpu:1",
    "time": "47:00:00",
    "cpus_per_task": 128,
    "mem": "256G",
    "dependency": None,
    "include": None,
    "exclude": None,
    
    # Host paths (where files exist on the host system)
    "host_code_dir": os.path.join(HOME_DIR, "code"),
    "host_datasets_dir": os.path.join(ssd_store_path, "datasets"),
    "host_outputs_dir": os.path.join(ssd_store_path, "outputs"),
    "container_image": os.path.join(ssd_store_path, "MedSegDiff_nnUNet_010226.sqsh"),
    
    # Container mount points (where directories appear inside container)
    "container_code_dir": "/mnt/code",
    "container_datasets_dir": "/mnt/datasets",
    "container_outputs_base": "/mnt/outputs/",  # General base, experiment subdir from config
    "container_prefix": "/mnt/",
    "host_base": ssd_store_path,
    "container_outputs_dir": "/mnt/outputs",
    
    # Output directory configuration
    "logdir_name": "medsegdiff_outputs",
    
    # Hydra configuration
    "hydra_config_name": "cluster",
    "hydra_overrides": []  # List of override strings, e.g., ["dataset.fold=0", "training.max_epochs=300"]
}

# Initialize logdir paths
BASE_CONFIG = update_logdir_paths(BASE_CONFIG)

SLURM_TEMPLATE = """#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
{gpu_directive}
#SBATCH --time={time}
#SBATCH --dependency={dependency}
#SBATCH --nodelist={include}
#SBATCH --exclude={exclude}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --output={output_file}
#SBATCH --error={error_file}

# Define paths
CODE_DIR={host_code_dir}
DATASETS_DIR={host_datasets_dir}
OUTPUTS_DIR={host_outputs_dir}
IMAGE='{container_image}'

# Ensure log directories exist
mkdir -p logs

# Set ulimit
ulimit -n 4096

# Execute the job
srun \\
  --container-mounts=$CODE_DIR:{container_code_dir},$DATASETS_DIR:{container_datasets_dir},$OUTPUTS_DIR:{container_outputs_dir} \\
  --container-image=$IMAGE \\
  bash -c "cd {container_code_dir}/medseg-diffusion_ISLES24 && export PYTHONNOUSERSITE=1 && export TORCH_MULTIPROCESSING_SHARING_STRATEGY=file_system && {python_command}"
"""
