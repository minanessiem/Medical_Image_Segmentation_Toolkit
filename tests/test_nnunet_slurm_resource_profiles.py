import os
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("SSD_STORE", "/tmp/ssd_store")

from scripts.nnunet.slurm_runners.nnunet_env import (  # noqa: E402
    COMMAND_DEFAULTS,
    CPU_RESOURCE_PROFILE,
    GPU_RESOURCE_PROFILE,
    NNUNET_SLURM_TEMPLATE,
)
from scripts.nnunet.slurm_runners.run_convert_to_nnunet import (  # noqa: E402
    CONVERT_DEFAULTS,
)
from scripts.nnunet.slurm_runners.run_evaluate_nnunet_results import (  # noqa: E402
    EVAL_DEFAULTS,
)
from scripts.nnunet.slurm_runners.run_nnunet_command import (  # noqa: E402
    build_slurm_config,
)
from scripts.slurm.base_run_config import BASE_CONFIG, SLURM_TEMPLATE  # noqa: E402


def _render(template: str, profile: dict) -> str:
    config = BASE_CONFIG.copy()
    config.update(profile)
    config.update(
        {
            "job_name": "resource_test",
            "output_file": "/tmp/output.out",
            "error_file": "/tmp/error.err",
            "python_command": "python3 workload.py",
            "command": "nnUNetv2_train 001 3d_fullres 0",
            "nnunet_env_exports": "export nnUNet_raw=/tmp/raw",
        }
    )
    return template.format(**config)


def _command_args(**overrides) -> SimpleNamespace:
    values = {
        "cpus": 12,
        "mem": "32G",
        "time": "02:00:00",
        "partition": None,
        "qos": None,
        "dataset_id": "266",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cpu_resource_profile_matches_lrz_policy():
    assert CPU_RESOURCE_PROFILE == {
        "partition": "lrz-cpu",
        "qos": "cpu",
        "gpus": 0,
        "gpu_directive": "",
        "cpus_per_task": 12,
        "mem": "32G",
    }


def test_gpu_resource_profile_matches_lrz_policy():
    assert GPU_RESOURCE_PROFILE == {
        "partition": "mcml-dgx-a100-40x8",
        "qos": "mcml",
        "gpus": 1,
        "gpu_directive": "#SBATCH --gres=gpu:1",
        "cpus_per_task": 24,
        "mem": "64G",
    }


def test_cpu_profiles_render_without_gpu_request():
    for template in (SLURM_TEMPLATE, NNUNET_SLURM_TEMPLATE):
        script = _render(template, CPU_RESOURCE_PROFILE)
        assert "#SBATCH --partition=lrz-cpu" in script
        assert "#SBATCH --qos=cpu" in script
        assert "--gres=gpu" not in script


def test_gpu_profiles_render_with_one_gpu_request():
    for template in (SLURM_TEMPLATE, NNUNET_SLURM_TEMPLATE):
        script = _render(template, GPU_RESOURCE_PROFILE)
        assert "#SBATCH --partition=mcml-dgx-a100-40x8" in script
        assert "#SBATCH --qos=mcml" in script
        assert "#SBATCH --gres=gpu:1" in script


def test_nnunet_commands_use_the_expected_profiles():
    preprocess = build_slurm_config(
        _command_args(), "nnUNetv2_plan_and_preprocess -d 266", "preprocess"
    )
    train = build_slurm_config(
        _command_args(cpus=24, mem="64G", time="47:00:00"),
        "nnUNetv2_train 266 3d_fullres 0",
        "train",
    )
    predict = build_slurm_config(
        _command_args(cpus=24, mem="64G", time="04:00:00"),
        "nnUNetv2_predict -d 266",
        "predict",
    )

    assert preprocess["partition"] == "lrz-cpu"
    assert preprocess["qos"] == "cpu"
    assert preprocess["gpus"] == 0
    for config in (train, predict):
        assert config["partition"] == "mcml-dgx-a100-40x8"
        assert config["qos"] == "mcml"
        assert config["gpus"] == 1


def test_explicit_slurm_overrides_take_precedence():
    config = build_slurm_config(
        _command_args(
            partition="custom-cpu",
            qos="custom-qos",
            cpus=8,
            mem="16G",
            time="01:00:00",
        ),
        "nnUNetv2_plan_and_preprocess -d 266",
        "preprocess",
    )

    assert config["partition"] == "custom-cpu"
    assert config["qos"] == "custom-qos"
    assert config["cpus_per_task"] == 8
    assert config["mem"] == "16G"
    assert config["time"] == "01:00:00"


def test_convert_evaluate_and_export_workers_use_cpu_capacity():
    assert CONVERT_DEFAULTS["gpus"] == 0
    assert CONVERT_DEFAULTS["cpus_per_task"] == 12
    assert CONVERT_DEFAULTS["mem"] == "32G"
    assert EVAL_DEFAULTS["gpus"] == 0
    assert EVAL_DEFAULTS["cpus_per_task"] == 12
    assert EVAL_DEFAULTS["mem"] == "32G"

    cluster_config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "nnunet" / "cluster.yaml").read_text()
    )
    assert cluster_config["parallel"]["num_workers"] == 12


def test_command_defaults_keep_expected_time_limits():
    assert COMMAND_DEFAULTS["preprocess"]["time"] == "02:00:00"
    assert COMMAND_DEFAULTS["train"]["time"] == "47:00:00"
    assert COMMAND_DEFAULTS["predict"]["time"] == "04:00:00"


if __name__ == "__main__":
    tests = [
        value
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"{len(tests)} resource-profile tests passed")
