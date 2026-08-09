#!/usr/bin/env python3

import glob
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, Optional

class SlurmJobRunner:
    def __init__(self, base_config: Dict):
        self.base_config = base_config
        
    def get_existing_runstrings(self) -> set:
        """Get all existing runstrings from output files."""
        existing_runstrings = set()
        output_pattern = os.path.join(self.base_config["host_logdir"], "*.out")
        
        print(f"\nScanning for existing runs:")
        print(f"  Pattern: {output_pattern}")
        print(f"  Absolute path: {os.path.abspath(output_pattern)}")
        
        output_files = glob.glob(output_pattern)
        print(f"  Found {len(output_files)} output files")
        
        for file in output_files:
            filename = os.path.basename(file)
            parts = filename.split('_')
            if len(parts) >= 3:
                # For medseg-diffusion, adjust parsing if needed (e.g., based on Hydra log naming)
                # Assuming similar format: remove timestamp
                runstring = '_'.join(parts[:-2])  # Flexible for new naming
                existing_runstrings.add(runstring)
                print(f"    Found runstring: {runstring} from file: {file}")
        
        return existing_runstrings
    
    def verify_paths(self) -> bool:
        """Verify all required paths exist."""
        required_paths = [
            self.base_config["host_code_dir"],
            self.base_config["host_datasets_dir"],
            self.base_config["container_image"]
        ]
        
        for path in required_paths:
            if not os.path.exists(path):
                print(f"Error: Required path does not exist: {path}")
                return False
        return True
    
    def check_job_status(self, job_id: str, error_file: Optional[str] = None) -> None:
        """Check and report job status."""
        status_check = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%t"],
            capture_output=True,
            text=True
        )
        
        if status_check.stdout.strip():
            state = status_check.stdout.strip()
            print(f"Job {job_id} state: {state}")
        else:
            print(f"Warning: Job {job_id} not found in queue. Checking detailed status...")
            sacct_check = subprocess.run(
                ["sacct", "-j", job_id, "--format=State,Elapsed,ExitCode", "-n"],
                capture_output=True,
                text=True
            )
            
            if sacct_check.stdout.strip():
                print(f"Job {job_id} details:\n{sacct_check.stdout.strip()}")
                if error_file and os.path.exists(error_file):
                    with open(error_file, 'r') as f:
                        error_content = f.read().strip()
                        if error_content:
                            print(f"\nError file contents:\n{error_content}")
            else:
                print(f"Warning: Could not find status information for job {job_id}")
    
    def submit_job(self, config: Dict, template: str, dry_run: bool = False) -> Optional[str]:
        """Submit a SLURM job with the given configuration."""
        # Create job name for logging
        job_name = config["job_name"]
        
        # Check paths first
        if not self.verify_paths():
            if dry_run:
                print(f"Would skip job {job_name}: Required paths missing")
            return None
        
        # Create output files using the logdir from config
        # Note: We use host_logdir here since SLURM runs on the host system
        host_logdir = config["host_logdir"]

        output_file = os.path.join(host_logdir, "output.out")
        error_file = os.path.join(host_logdir, "error.err")
        
        config["output_file"] = output_file
        config["error_file"] = error_file 
        
        print(f"\nJob output paths:")
        print(f"  Output file: {os.path.abspath(output_file)}")
        print(f"  Error file: {os.path.abspath(error_file)}")
        
        if dry_run:
            print(f"\nWould submit job: {job_name}")
            print("Configuration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
            return None
        else:
            os.makedirs(host_logdir, exist_ok=True)
        
        # Build Hydra overrides string
        if 'hydra_overrides' in config and config['hydra_overrides']:
            config['hydra_overrides_str'] = ' '.join(config['hydra_overrides'])
        else:
            config['hydra_overrides_str'] = ''

        include = config.get("include")
        exclude = config.get("exclude")
        if include and exclude:
            raise ValueError(
                "SLURM include/exclude are mutually exclusive. "
                "Provide only one of 'include' or 'exclude'."
            )

        # Keep SBATCH syntax centralized in the template and drop optional lines
        # when their corresponding values are unset.
        slurm_template = template
        if not config.get("dependency"):
            slurm_template = slurm_template.replace(
                "#SBATCH --dependency={dependency}\n", ""
            )
        if not include:
            slurm_template = slurm_template.replace(
                "#SBATCH --nodelist={include}\n", ""
            )
        if not exclude:
            slurm_template = slurm_template.replace(
                "#SBATCH --exclude={exclude}\n", ""
            )

        # Keep the optional GPU directive consistent with the requested count.
        # CPU-only profiles set gpus=0, which must not emit a GPU GRES request.
        gpu_count = int(config.get("gpus", 0))
        config["gpu_directive"] = (
            f"#SBATCH --gres=gpu:{gpu_count}" if gpu_count > 0 else ""
        )

        # Create the SLURM script
        slurm_script = slurm_template.format(**config)
        script_file = f"temp_{job_name}.sbatch"
        
        try:
            with open(script_file, "w") as f:
                f.write(slurm_script)
            
            result = subprocess.run(
                ["sbatch", script_file],
                check=True,
                capture_output=True,
                text=True
            )
            
            if "Submitted batch job" in result.stdout:
                job_id = result.stdout.strip().split()[-1]
                print(f"Successfully submitted job: {job_name} (Job ID: {job_id})")
                
                time.sleep(2)  # Wait for job to enter system
                self.check_job_status(job_id, error_file)
                return job_id
                
            print(f"Warning: Unexpected sbatch output: {result.stdout}")
            return None
            
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job {job_name}: {e}")
            print(f"Error output: {e.stderr}")
            return None
            
        finally:
            if os.path.exists(script_file):
                os.remove(script_file)
