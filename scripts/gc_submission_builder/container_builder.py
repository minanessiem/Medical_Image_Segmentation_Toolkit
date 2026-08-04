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
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from uuid import uuid4

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from scripts.gc_submission_builder.container_config import (
    PROJECT_ROOT,
    ContainerBuildConfig,
)
from scripts.gc_submission_builder.release_manifest import sha256_file
from scripts.gc_submission_builder.runtime.interfaces import (
    OutputBinding,
    load_interface_manifest,
    resolve_invocation,
)


CONTAINER_BUILD_REPORT_NAME = "container_build_report.json"
GRAND_CHALLENGE_LABEL = "org.grand-challenge.api-method"
MODEL_WEIGHT_SUFFIXES = (".pth", ".pt", ".ckpt")
LOCAL_INVOKE_TIMEOUT_SECONDS = 300


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
    output_paths: Mapping[str, Path]
    output_validations: Mapping[str, Mapping[str, Any]]
    geometry_matches_native_input: bool | None
    external_http_tested: bool
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
    container_name = f"gc-smoke-{uuid4().hex[:12]}"
    network_name = f"{container_name}-network"
    started = False
    network_created = False
    try:
        _run(["docker", "network", "create", "--internal", network_name])
        network_created = True
        run_command = [
            "docker",
            "run",
            "--detach",
            "--name",
            container_name,
            "--network",
            network_name,
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
        _wait_for_health(
            container_name,
            image_reference=config.image_reference,
            network_name=network_name,
            timeout_seconds=readiness_timeout_seconds,
        )
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
        _invoke_from_sidecar(
            container_name,
            image_reference=config.image_reference,
            network_name=network_name,
            timeout_seconds=LOCAL_INVOKE_TIMEOUT_SECONDS,
        )
        output_paths, output_validations = validate_single_input_output_set(
            bindings=invocation.interface.outputs,
            output_root=output_root,
            input_path=(
                next(iter(invocation.raw_modalities.values()))
                if len(invocation.raw_modalities) == 1
                else None
            ),
        )
        geometry_matches_native_input: bool | None = None
        if len(invocation.raw_modalities) == 1:
            geometry_matches_native_input = True
        runtime_logs = _run(
            ["docker", "logs", container_name],
            capture_output=True,
        )
        return ContainerTestResult(
            image_reference=config.image_reference,
            interface_name=invocation.interface.name,
            output_paths=MappingProxyType(output_paths),
            output_validations=MappingProxyType(
                {
                    slug: MappingProxyType(validation)
                    for slug, validation in output_validations.items()
                }
            ),
            geometry_matches_native_input=geometry_matches_native_input,
            external_http_tested=True,
            non_root_uid=uid,
            runtime_log=_combined_output(runtime_logs),
        )
    finally:
        if started:
            _run(["docker", "rm", "--force", container_name], check=False)
        if network_created:
            _run(["docker", "network", "rm", network_name], check=False)


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


def validate_single_input_output_set(
    *,
    bindings: Sequence[OutputBinding],
    output_root: str | Path,
    input_path: str | Path | None,
) -> tuple[dict[str, Path], dict[str, Mapping[str, Any]]]:
    """Independently reopen every declared output and verify its transport contract."""

    root = Path(output_root).expanduser().resolve()
    paths: dict[str, Path] = {}
    validations: dict[str, Mapping[str, Any]] = {}
    reference = None
    if input_path is not None:
        try:
            reference = sitk.ReadImage(str(Path(input_path).expanduser().resolve()))
        except Exception as exc:
            raise ContainerBuildError(
                "Could not reopen the sole native input for output-set validation."
            ) from exc
    for binding in bindings:
        extension = "output.nii.gz" if binding.file_type == "nifti" else "output.mha"
        path = root / Path(*binding.relative_path.split("/")) / extension
        if not path.is_file():
            raise ContainerBuildError(
                f"HTTP 201 returned without output socket {binding.slug!r}."
            )
        if binding.file_type == "nifti":
            if binding.result_key != "mask":
                raise ContainerBuildError(
                    "Independent NIfTI validation currently supports mask results only."
                )
            if input_path is not None:
                validate_single_input_nifti_output(
                    output_path=path,
                    input_path=input_path,
                )
            image = nib.load(str(path))
            array = np.asarray(image.dataobj)
            if image.get_data_dtype() != np.dtype(np.uint8) or not set(
                np.unique(array)
            ).issubset({0, 1}):
                raise ContainerBuildError("NIfTI mask output must be binary uint8.")
            validation = {
                "result_key": "mask",
                "file_type": "nifti",
                "shape": list(image.shape),
                "dtype": "uint8",
                "spatial_validation": "passed" if reference is not None else None,
            }
        else:
            validation = _validate_mha_output(
                path,
                result_key=binding.result_key,
                reference=reference,
            )
        paths[binding.slug] = path
        validations[binding.slug] = validation
    return paths, validations


def _validate_mha_output(
    path: Path,
    *,
    result_key: str,
    reference: sitk.Image | None,
) -> Mapping[str, Any]:
    try:
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)
    except Exception as exc:
        raise ContainerBuildError("Could not reopen declared MHA output.") from exc
    if image.GetDimension() != 3:
        raise ContainerBuildError("Declared MHA output must be 3D.")
    if result_key == "mask":
        if array.dtype != np.dtype(np.uint8) or not set(np.unique(array)).issubset(
            {0, 1}
        ):
            raise ContainerBuildError("MHA mask output must be binary uint8.")
    elif result_key == "probability":
        if array.dtype != np.dtype(np.float32):
            raise ContainerBuildError("MHA probability output must be float32.")
        if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise ContainerBuildError(
                "MHA probability output must be finite and constrained to [0, 1]."
            )
    else:
        raise ContainerBuildError(f"Unknown output result_key={result_key!r}.")
    if reference is not None:
        for name, observed, expected in (
            ("size", image.GetSize(), reference.GetSize()),
            ("spacing", image.GetSpacing(), reference.GetSpacing()),
            ("origin", image.GetOrigin(), reference.GetOrigin()),
            ("direction", image.GetDirection(), reference.GetDirection()),
        ):
            if not np.allclose(observed, expected, rtol=0, atol=1e-5):
                raise ContainerBuildError(
                    f"MHA output {name} does not match the sole native input."
                )
        _validate_world_coordinate_landmarks(image, reference)
    try:
        with path.open("rb") as handle:
            header = handle.read(65536).split(b"ElementDataFile", 1)[0]
    except OSError as exc:
        raise ContainerBuildError("Could not inspect MHA compression header.") from exc
    if b"CompressedData = True" not in header:
        raise ContainerBuildError("MHA output must use embedded compression.")
    return {
        "result_key": result_key,
        "file_type": "mha",
        "shape": list(reversed(array.shape)),
        "dtype": str(array.dtype),
        "spatial_validation": "passed" if reference is not None else None,
        "compressed": True,
    }


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


def _validate_world_coordinate_landmarks(
    observed: sitk.Image,
    expected: sitk.Image,
) -> None:
    size = expected.GetSize()
    landmarks = {
        (0, 0, 0),
        tuple(max(int(length) - 1, 0) for length in size),
        tuple(max((int(length) - 1) // 2, 0) for length in size),
    }
    for index in landmarks:
        observed_point = observed.TransformIndexToPhysicalPoint(index)
        expected_point = expected.TransformIndexToPhysicalPoint(index)
        if not np.allclose(observed_point, expected_point, rtol=0, atol=1e-5):
            raise ContainerBuildError(
                "MHA output world-coordinate landmarks do not match the sole "
                "native input."
            )


def _wait_for_health(
    container_name: str,
    *,
    image_reference: str,
    network_name: str,
    timeout_seconds: int,
) -> None:
    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        result = _run_http_sidecar(
            container_name=container_name,
            image_reference=image_reference,
            network_name=network_name,
            method="GET",
            path="/health",
            expected_status=200,
            timeout_seconds=2,
        )
        if result.returncode == 0:
            return
        observed_status = result.stdout.strip()
        if observed_status.isdigit():
            raise ContainerBuildError(
                "Container /health must return exact HTTP 200 without redirect; "
                f"observed_status={observed_status}."
            )
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


def _invoke_from_sidecar(
    container_name: str,
    *,
    image_reference: str,
    network_name: str,
    timeout_seconds: int,
) -> None:
    result = _run_http_sidecar(
        container_name=container_name,
        image_reference=image_reference,
        network_name=network_name,
        method="POST",
        path="/invoke",
        expected_status=201,
        timeout_seconds=timeout_seconds,
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
            f"sidecar_output={_combined_output(result)[-500:]!r}, "
            f"log_tail={_combined_output(logs)[-2000:]!r}."
        )


def _run_http_sidecar(
    *,
    container_name: str,
    image_reference: str,
    network_name: str,
    method: str,
    path: str,
    expected_status: int,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    """Call the service from a separate container without following redirects."""

    code = (
        "import http.client,sys; "
        "connection=http.client.HTTPConnection(sys.argv[1],4743,"
        "timeout=float(sys.argv[4])); "
        "connection.request(sys.argv[2],sys.argv[3],body=b'' if "
        "sys.argv[2]=='POST' else None); "
        "response=connection.getresponse(); response.read(); "
        "print(response.status); "
        "raise SystemExit(0 if response.status==int(sys.argv[5]) else 22)"
    )
    return _run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            network_name,
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=67108864",
            "--entrypoint",
            "python",
            image_reference,
            "-c",
            code,
            container_name,
            method,
            path,
            str(timeout_seconds),
            str(expected_status),
        ],
        check=False,
        capture_output=True,
        timeout_seconds=timeout_seconds + 5,
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
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [str(value) for value in command],
            check=check,
            capture_output=capture_output,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise ContainerBuildError(
            "Container command exceeded its external timeout; "
            f"timeout_seconds={timeout_seconds}, "
            f"command={' '.join(str(value) for value in command[:4])}."
        ) from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContainerBuildError(
            f"Container command failed: {' '.join(str(value) for value in command[:4])}"
        ) from exc


__all__ = [
    "CONTAINER_BUILD_REPORT_NAME",
    "LOCAL_INVOKE_TIMEOUT_SECONDS",
    "ContainerBuildError",
    "ContainerBuildResult",
    "ContainerImageInspection",
    "ContainerTestResult",
    "build_container_image",
    "inspect_container_image",
    "save_container_image",
    "test_container_image",
    "validate_single_input_output_set",
    "validate_single_input_nifti_output",
]
