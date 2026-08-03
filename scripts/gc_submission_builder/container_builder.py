"""Build, inspect, smoke-test, and save one Grand Challenge image."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import monotonic, sleep
from typing import Any, Mapping, Sequence
from uuid import uuid4

import nibabel as nib
import numpy as np

from scripts.gc_submission_builder.container_config import (
    PROJECT_ROOT,
    ContainerBuildConfig,
)
from scripts.gc_submission_builder.release_manifest import sha256_file
from scripts.gc_submission_builder.runtime.interfaces import (
    load_interface_manifest,
    resolve_invocation,
)


CONTAINER_BUILD_REPORT_NAME = "container_build_report.json"
GRAND_CHALLENGE_LABEL = "org.grand-challenge.api-method"
MODEL_WEIGHT_SUFFIXES = (".pth", ".pt", ".ckpt")


class ContainerBuildError(RuntimeError):
    """Raised when the independently replaceable image fails its build contract."""


@dataclass(frozen=True)
class ContainerImageInspection:
    image_reference: str
    image_id: str
    architecture: str
    configured_user: str
    api_method: str
    model_payload_audited: bool


@dataclass(frozen=True)
class ContainerBuildResult:
    inspection: ContainerImageInspection
    report_path: Path


@dataclass(frozen=True)
class ContainerTestResult:
    image_reference: str
    interface_name: str
    output_path: Path
    output_shape: tuple[int, int, int]
    output_dtype: str
    output_values: tuple[int, ...]
    geometry_matches_native_input: bool | None
    non_root_uid: int
    runtime_log: str


def build_container_image(config: ContainerBuildConfig) -> ContainerBuildResult:
    """Build Linux/amd64 without embedding a model artifact, then inspect it."""

    _require_config(config)
    manifest_arg = config.interface_manifest_path.relative_to(PROJECT_ROOT).as_posix()
    _run(
        [
            "docker",
            "build",
            "--platform",
            config.platform,
            "--file",
            str(config.dockerfile),
            "--tag",
            config.image_reference,
            "--build-arg",
            f"INTERFACE_MANIFEST={manifest_arg}",
            str(PROJECT_ROOT),
        ]
    )
    inspection = inspect_container_image(config.image_reference)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.output_dir / CONTAINER_BUILD_REPORT_NAME
    report = {
        "image": asdict(inspection),
        "platform": config.platform,
        "dockerfile_sha256": sha256_file(config.dockerfile),
        "requirements_sha256": sha256_file(
            config.dockerfile.parent / "requirements.lock"
        ),
        "interface_manifest_sha256": sha256_file(config.interface_manifest_path),
        "model_embedded": not inspection.model_payload_audited,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ContainerBuildResult(inspection=inspection, report_path=report_path)


def inspect_container_image(image_reference: str) -> ContainerImageInspection:
    completed = _run(
        ["docker", "image", "inspect", image_reference],
        capture_output=True,
    )
    try:
        payload = json.loads(completed.stdout)
        image = payload[0]
        config = image["Config"]
    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
        raise ContainerBuildError("Docker returned an invalid image inspection.") from exc
    architecture = str(image.get("Architecture", ""))
    configured_user = str(config.get("User", ""))
    api_method = str((config.get("Labels") or {}).get(GRAND_CHALLENGE_LABEL, ""))
    if architecture != "amd64":
        raise ContainerBuildError(
            f"Container architecture must be amd64, got {architecture!r}."
        )
    if not configured_user or configured_user in {"0", "root", "0:0", "root:root"}:
        raise ContainerBuildError("Container image must configure a non-root user.")
    if api_method != "invoke":
        raise ContainerBuildError(
            f"Container label {GRAND_CHALLENGE_LABEL!r} must equal 'invoke'."
        )
    _audit_image_model_payload(image_reference)
    return ContainerImageInspection(
        image_reference=image_reference,
        image_id=str(image.get("Id", "")),
        architecture=architecture,
        configured_user=configured_user,
        api_method=api_method,
        model_payload_audited=True,
    )


def test_container_image(
    config: ContainerBuildConfig,
    *,
    model_dir: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    readiness_timeout_seconds: int = 300,
) -> ContainerTestResult:
    """Run the platform lifecycle with isolated networking and read-only inputs."""

    _require_config(config)
    model_root = _required_directory(model_dir, "model_dir")
    input_root = _required_directory(input_dir, "input_dir")
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise ContainerBuildError(
            "output_dir must be empty so stale predictions cannot satisfy the smoke test."
        )
    manifest = load_interface_manifest(config.interface_manifest_path)
    invocation = resolve_invocation(manifest, input_root)
    output_path = (
        output_root
        / Path(*invocation.interface.output.relative_path.split("/"))
        / "output.nii.gz"
    )
    container_name = f"gc-smoke-{uuid4().hex[:12]}"
    started = False
    try:
        run_command = [
            "docker",
            "run",
            "--detach",
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--memory",
            "32g",
            "--cpus",
            "8",
            "--shm-size",
            "1g",
            "--gpus",
            "all",
            "--mount",
            f"type=bind,src={model_root},dst=/opt/ml/model,readonly",
            "--mount",
            f"type=bind,src={input_root},dst=/input,readonly",
            "--mount",
            f"type=bind,src={output_root},dst=/output",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=2147483648",
            config.image_reference,
        ]
        _run(run_command)
        started = True
        _wait_for_health(container_name, timeout_seconds=readiness_timeout_seconds)
        uid_result = _run(
            ["docker", "exec", container_name, "id", "-u"],
            capture_output=True,
        )
        try:
            uid = int(uid_result.stdout.strip())
        except ValueError as exc:
            raise ContainerBuildError("Could not verify the container runtime UID.") from exc
        if uid == 0:
            raise ContainerBuildError("Smoke-test container is running as root.")
        _invoke_inside_container(container_name)
        if not output_path.is_file():
            raise ContainerBuildError("HTTP 201 returned without the declared output file.")
        image = nib.load(str(output_path))
        values = tuple(int(value) for value in np.unique(np.asarray(image.dataobj)))
        if image.get_data_dtype() != np.dtype(np.uint8) or any(
            value not in (0, 1) for value in values
        ):
            raise ContainerBuildError(
                "Container output must be a binary uint8 NIfTI segmentation."
            )
        geometry_matches_native_input: bool | None = None
        if len(invocation.raw_modalities) == 1:
            validate_single_input_nifti_output(
                output_path=output_path,
                input_path=next(iter(invocation.raw_modalities.values())),
            )
            geometry_matches_native_input = True
        runtime_logs = _run(
            ["docker", "logs", container_name],
            capture_output=True,
        )
        return ContainerTestResult(
            image_reference=config.image_reference,
            interface_name=invocation.interface.name,
            output_path=output_path,
            output_shape=tuple(int(value) for value in image.shape),
            output_dtype="uint8",
            output_values=values,
            geometry_matches_native_input=geometry_matches_native_input,
            non_root_uid=uid,
            runtime_log=_combined_output(runtime_logs),
        )
    finally:
        if started:
            _run(["docker", "rm", "--force", container_name], check=False)


def save_container_image(config: ContainerBuildConfig) -> Path:
    """Export the image as one deterministic-gzip Docker archive."""

    _require_config(config)
    inspect_container_image(config.image_reference)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = config.output_dir / config.archive_name
    if archive_path.exists():
        raise ContainerBuildError(f"Refusing to overwrite existing archive: {archive_path}")
    with tempfile.TemporaryDirectory(dir=config.output_dir) as temporary:
        tar_path = Path(temporary) / "image.tar"
        _run(["docker", "save", "--output", str(tar_path), config.image_reference])
        temporary_archive = Path(temporary) / config.archive_name
        with tar_path.open("rb") as source, temporary_archive.open("wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                compresslevel=9,
                mtime=0,
            ) as compressed:
                shutil.copyfileobj(source, compressed, length=1024 * 1024)
        shutil.move(str(temporary_archive), str(archive_path))
    return archive_path


def validate_single_input_nifti_output(
    *,
    output_path: str | Path,
    input_path: str | Path,
) -> None:
    """Independently require fixture output to occupy the sole input NIfTI grid."""

    try:
        output = nib.load(str(Path(output_path).expanduser().resolve()))
        native_input = nib.load(str(Path(input_path).expanduser().resolve()))
    except Exception as exc:
        raise ContainerBuildError(
            "Could not reopen the fixture input/output NIfTI geometry."
        ) from exc
    if tuple(output.shape) != tuple(native_input.shape):
        raise ContainerBuildError(
            "Container output shape does not match the sole native input: "
            f"output={tuple(output.shape)}, input={tuple(native_input.shape)}."
        )
    if not np.allclose(
        np.asarray(output.affine, dtype=np.float64),
        np.asarray(native_input.affine, dtype=np.float64),
        rtol=0,
        atol=1e-4,
    ):
        raise ContainerBuildError(
            "Container output affine does not match the sole native input affine."
        )
    for name, output_form, input_form in (
        ("qform", output.get_qform(coded=True), native_input.get_qform(coded=True)),
        ("sform", output.get_sform(coded=True), native_input.get_sform(coded=True)),
    ):
        _validate_nifti_form(name, output_form, input_form)


def _audit_image_model_payload(image_reference: str) -> None:
    script = (
        "import json,pathlib; "
        "model=pathlib.Path('/opt/ml/model'); app=pathlib.Path('/opt/app'); "
        "model_files=sorted(str(p) for p in model.rglob('*') if p.is_file()); "
        "weight_files=sorted(str(p) for root in (app,model) for p in root.rglob('*') "
        f"if p.is_file() and p.name.lower().endswith({MODEL_WEIGHT_SUFFIXES!r})); "
        "print(json.dumps({'model_directory_files':model_files,"
        "'weight_like_files':weight_files},sort_keys=True))"
    )
    completed = _run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--entrypoint",
            "python",
            image_reference,
            "-c",
            script,
        ],
        capture_output=True,
    )
    try:
        payload = json.loads(completed.stdout)
        model_files = payload["model_directory_files"]
        weight_files = payload["weight_like_files"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ContainerBuildError(
            "Container returned an invalid embedded-model audit result."
        ) from exc
    if not isinstance(model_files, list) or not isinstance(weight_files, list):
        raise ContainerBuildError(
            "Container embedded-model audit fields must be lists."
        )
    embedded = sorted({str(path) for path in (*model_files, *weight_files)})
    if embedded:
        raise ContainerBuildError(
            "Container image contains an embedded model payload; "
            f"files={embedded}."
        )


def _validate_nifti_form(
    name: str,
    observed: tuple[np.ndarray | None, int],
    expected: tuple[np.ndarray | None, int],
) -> None:
    observed_form, observed_code = observed
    expected_form, expected_code = expected
    if int(observed_code) != int(expected_code):
        raise ContainerBuildError(
            f"Container output {name} code does not match the sole native input."
        )
    if expected_form is None:
        if observed_form is not None:
            raise ContainerBuildError(
                f"Container output unexpectedly contains a coded {name}."
            )
        return
    if observed_form is None or not np.allclose(
        np.asarray(observed_form, dtype=np.float64),
        np.asarray(expected_form, dtype=np.float64),
        rtol=0,
        atol=1e-4,
    ):
        raise ContainerBuildError(
            f"Container output {name} does not match the sole native input."
        )


def _wait_for_health(container_name: str, *, timeout_seconds: int) -> None:
    deadline = monotonic() + timeout_seconds
    code = (
        "import urllib.request; "
        "r=urllib.request.urlopen('http://127.0.0.1:4743/health', timeout=2); "
        "raise SystemExit(0 if r.status == 200 else 1)"
    )
    while monotonic() < deadline:
        result = _run(
            ["docker", "exec", container_name, "python", "-c", code],
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            return
        status = _run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
            check=False,
            capture_output=True,
        )
        if status.returncode != 0 or status.stdout.strip().lower() != "true":
            logs = _run(
                ["docker", "logs", container_name],
                check=False,
                capture_output=True,
            )
            raise ContainerBuildError(
                "Container exited before readiness; "
                f"log_tail={_combined_output(logs)[-2000:]!r}."
            )
        sleep(1)
    raise ContainerBuildError(
        f"Container did not become healthy within {timeout_seconds} seconds."
    )


def _invoke_inside_container(container_name: str) -> None:
    code = (
        "import urllib.request; "
        "q=urllib.request.Request('http://127.0.0.1:4743/invoke', data=b'', method='POST'); "
        "r=urllib.request.urlopen(q, timeout=610); "
        "raise SystemExit(0 if r.status == 201 else 1)"
    )
    result = _run(
        ["docker", "exec", container_name, "python", "-c", code],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        logs = _run(
            ["docker", "logs", container_name],
            check=False,
            capture_output=True,
        )
        raise ContainerBuildError(
            "Container /invoke lifecycle failed; "
            "error_type=HttpInvocationFailure, "
            f"log_tail={_combined_output(logs)[-2000:]!r}."
        )


def _required_directory(value: str | Path, field: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ContainerBuildError(f"{field} is not a directory: {path}")
    return path


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(
        value.strip()
        for value in (result.stdout or "", result.stderr or "")
        if value.strip()
    )


def _require_config(config: ContainerBuildConfig) -> None:
    if not isinstance(config, ContainerBuildConfig):
        raise ContainerBuildError("config must be a ContainerBuildConfig instance.")


def _run(
    command: Sequence[str],
    *,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [str(value) for value in command],
            check=check,
            capture_output=capture_output,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContainerBuildError(
            f"Container command failed: {' '.join(str(value) for value in command[:4])}"
        ) from exc


__all__ = [
    "CONTAINER_BUILD_REPORT_NAME",
    "ContainerBuildError",
    "ContainerBuildResult",
    "ContainerImageInspection",
    "ContainerTestResult",
    "build_container_image",
    "inspect_container_image",
    "save_container_image",
    "test_container_image",
    "validate_single_input_nifti_output",
]
