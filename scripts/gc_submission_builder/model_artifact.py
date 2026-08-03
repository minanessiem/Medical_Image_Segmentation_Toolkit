"""Deterministic staging, validation, and archiving of one release model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any

import yaml
from omegaconf import OmegaConf

from scripts.evaluation.core.model_loader import CHECKPOINT_DIRS
from scripts.gc_submission_builder.build_config import (
    ModelArtifactBuildConfig,
    compose_inference_policy_file,
)
from scripts.gc_submission_builder.release_manifest import (
    ARTIFACT_FILENAMES,
    create_artifact_manifest,
    sha256_file,
    verify_artifact_manifest,
    write_artifact_manifest,
)
from src.data.loader_stack.contracts import PreprocessingAdapterError
from src.data.loader_stack.preprocessing import get_preprocessing_adapter
from src.inference.case_producer import build_case_producer
from src.inference.contracts import PredictorCapabilities
from src.inference.policy import resolve_inference_policy
from src.inference.predictors import build_probability_predictor
from src.inference.runtime import (
    AssessmentContext,
    parse_inference_runtime,
    validate_runtime_compatibility,
)
from src.models.model_loader import StrictModelLoadError, load_model_strict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIRECTORY_NAME = "algorithmmodel"
MODEL_BUILD_REPORT_NAME = "model_build_report.json"


class ModelArtifactError(RuntimeError):
    """Raised when one singular release artifact cannot be built or validated."""


@dataclass(frozen=True)
class ModelArtifactValidationResult:
    dataset_id: str
    preprocessing_adapter: str
    model_family: str
    spatial_dims: int
    input_channels: int
    output_channels: int
    inference_policy_source: str
    output_space: str
    precision: str
    sliding_window_batch_size: int
    runtime_profile: str
    strict_checkpoint_load: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelArtifactBuildResult:
    artifact_dir: Path
    archive_path: Path
    report_path: Path
    manifest: dict[str, Any]
    validation: ModelArtifactValidationResult


def resolve_release_checkpoint(
    run_dir: str | Path,
    checkpoint: str,
    *,
    use_ema: bool = False,
) -> Path:
    """Resolve exactly one checkpoint while retaining evaluation directory conventions."""

    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise ModelArtifactError(f"Training run directory does not exist: {root}")
    token = str(checkpoint).strip()
    if not token:
        raise ModelArtifactError("Checkpoint specification must be non-empty.")

    checkpoint_value = Path(token).expanduser()
    path_explicit = checkpoint_value.is_absolute() or checkpoint_value.parent != Path(".")
    if path_explicit:
        candidate = (
            checkpoint_value
            if checkpoint_value.is_absolute()
            else root / checkpoint_value
        ).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ModelArtifactError(
                "An explicit release checkpoint path must remain within the selected "
                f"training run: {candidate}"
            ) from exc
        if not candidate.is_file() or candidate.suffix != ".pth":
            raise ModelArtifactError(
                f"Explicit release checkpoint is not a .pth file: {candidate}"
            )
        return candidate

    normalized_name = token[:-4] if token.endswith(".pth") else token
    candidates: list[Path] = []
    for relative_dir in CHECKPOINT_DIRS:
        checkpoint_dir = root / relative_dir
        if use_ema:
            for exact_name in (
                f"{normalized_name}_ema.pth",
                f"{normalized_name}.ema.pth",
            ):
                candidate = checkpoint_dir / exact_name
                if candidate.is_file():
                    candidates.append(candidate.resolve())
            if checkpoint_dir.is_dir():
                candidates.extend(
                    path.resolve()
                    for path in sorted(
                        checkpoint_dir.glob(f"{normalized_name}_ema_*.pth")
                    )
                    if path.is_file()
                )
        else:
            candidate = checkpoint_dir / f"{normalized_name}.pth"
            if candidate.is_file():
                candidates.append(candidate.resolve())
    unique_candidates = sorted(set(candidates), key=str)
    if not unique_candidates:
        kind = "EMA checkpoint" if use_ema else "checkpoint"
        searched = ", ".join(str(root / directory) for directory in CHECKPOINT_DIRS)
        raise ModelArtifactError(
            f"Release {kind} {normalized_name!r} was not found under: {searched}."
        )
    if len(unique_candidates) != 1:
        listed = "\n".join(f"  - {path}" for path in unique_candidates)
        raise ModelArtifactError(
            f"Release checkpoint selection is ambiguous for {normalized_name!r}; "
            f"provide an exact run-relative path. Matches:\n{listed}"
        )
    return unique_candidates[0]


def validate_model_artifact(
    artifact_dir: str | Path,
    *,
    device: str = "cpu",
) -> ModelArtifactValidationResult:
    """Hash-check and fully load-test an extracted `/opt/ml/model` equivalent."""

    root = Path(artifact_dir).expanduser().resolve()
    try:
        verify_artifact_manifest(root)
        saved_cfg = OmegaConf.load(root / "config.yaml")
        policy_payload = yaml.safe_load(
            (root / "inference_policy.yaml").read_text(encoding="utf-8")
        )
    except ModelArtifactError:
        raise
    except Exception as exc:
        raise ModelArtifactError(f"Could not read model artifact configuration: {exc}") from exc
    if not isinstance(policy_payload, dict):
        raise ModelArtifactError("Archived inference_policy.yaml must be a mapping.")
    if "defaults" in policy_payload:
        raise ModelArtifactError(
            "Archived inference_policy.yaml must be standalone and contain no defaults."
        )

    composed = OmegaConf.create(OmegaConf.to_container(saved_cfg, resolve=False))
    OmegaConf.set_struct(composed, False)
    OmegaConf.update(composed, "inference", policy_payload, merge=False)
    try:
        resolved_policy = resolve_inference_policy(composed)
        runtime = parse_inference_runtime(
            OmegaConf.load(
                PROJECT_ROOT / "configs" / "inference_runtime" / "gc_submission.yaml"
            )
        )
        validate_runtime_compatibility(
            resolved_policy.policy,
            runtime,
            AssessmentContext(requires_ground_truth=False, threshold_sweep=False),
        )
    except Exception as exc:
        raise ModelArtifactError(
            f"Artifact inference policy is incompatible with gc_submission: {exc}"
        ) from exc
    if resolved_policy.source != "explicit_top_level":
        raise ModelArtifactError(
            "Artifact validation requires inference_policy.yaml to win as the explicit "
            "top-level inference policy."
        )

    dataset_id, spatial_dims = _validate_initial_release_config(composed)
    try:
        adapter = get_preprocessing_adapter(dataset_id)
    except PreprocessingAdapterError as exc:
        raise ModelArtifactError(
            f"Saved dataset.id={dataset_id!r} has no registered preprocessing adapter: {exc}"
        ) from exc

    try:
        model = load_model_strict(composed, root / "weights.pth", device=device)
    except StrictModelLoadError as exc:
        raise ModelArtifactError(f"Strict release checkpoint validation failed: {exc}") from exc
    except Exception as exc:
        raise ModelArtifactError(f"Release model construction/loading failed: {exc}") from exc
    try:
        predictor = build_probability_predictor(backend=model, cfg=composed)
        capabilities = predictor.capabilities
        _validate_release_predictor_capabilities(capabilities)
        producer = build_case_producer(
            dataset_id=dataset_id,
            dataset_cfg=composed.dataset,
            load_labels=False,
        )
    except Exception as exc:
        raise ModelArtifactError(
            f"Artifact predictor/preprocessing capability validation failed: {exc}"
        ) from exc
    if producer.adapter.dataset_id != adapter.dataset_id:
        raise ModelArtifactError(
            "Preprocessing producer resolved a different dataset adapter than the "
            "saved model contract."
        )
    if int(capabilities.spatial_dims) != spatial_dims:
        raise ModelArtifactError(
            "Prepared predictor dimensionality disagrees with the saved model contract."
        )

    return ModelArtifactValidationResult(
        dataset_id=dataset_id,
        preprocessing_adapter=adapter.dataset_id,
        model_family=str(capabilities.model_family),
        spatial_dims=int(capabilities.spatial_dims),
        input_channels=int(capabilities.input_channels),
        output_channels=int(capabilities.output_channels),
        inference_policy_source=resolved_policy.source,
        output_space=resolved_policy.policy.output_space,
        precision=resolved_policy.policy.precision,
        sliding_window_batch_size=resolved_policy.policy.sliding_window.sw_batch_size,
        runtime_profile=runtime.profile,
        strict_checkpoint_load=True,
    )


def build_model_artifact(config: ModelArtifactBuildConfig) -> ModelArtifactBuildResult:
    """Build, load-test, and archive one independently replaceable model artifact."""

    if not isinstance(config, ModelArtifactBuildConfig):
        raise ModelArtifactError("config must be a ModelArtifactBuildConfig instance.")
    run_dir = Path(config.run_dir).expanduser().resolve()
    saved_config_path = run_dir / ".hydra" / "config.yaml"
    if not saved_config_path.is_file():
        raise ModelArtifactError(
            "Run config not found. Expected complete Hydra config at "
            f"{saved_config_path}."
        )
    checkpoint_path = resolve_release_checkpoint(
        run_dir,
        config.checkpoint,
        use_ema=config.use_ema,
    )
    try:
        inference_policy = compose_inference_policy_file(config.inference_policy_path)
    except Exception as exc:
        raise ModelArtifactError(f"Could not compose selected inference policy: {exc}") from exc

    output_dir = Path(config.output_dir).expanduser().resolve()
    try:
        output_dir.relative_to(run_dir)
    except ValueError:
        pass
    else:
        raise ModelArtifactError(
            "Model build output_dir must remain outside the immutable training run."
        )
    artifact_dir = output_dir / ARTIFACT_DIRECTORY_NAME
    archive_path = output_dir / config.archive_name
    report_path = output_dir / MODEL_BUILD_REPORT_NAME
    existing = [path for path in (artifact_dir, archive_path, report_path) if path.exists()]
    if existing:
        raise ModelArtifactError(
            "Refusing to overwrite existing model build outputs: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    code_commit, code_dirty, created_at_utc = _resolve_code_provenance(config)

    with tempfile.TemporaryDirectory(prefix=".cut9-model-", dir=output_dir) as tmp:
        temporary_root = Path(tmp)
        temporary_artifact = temporary_root / ARTIFACT_DIRECTORY_NAME
        temporary_artifact.mkdir()
        shutil.copyfile(saved_config_path, temporary_artifact / "config.yaml")
        shutil.copyfile(checkpoint_path, temporary_artifact / "weights.pth")
        (temporary_artifact / "inference_policy.yaml").write_text(
            yaml.safe_dump(inference_policy, sort_keys=True),
            encoding="utf-8",
        )
        manifest = create_artifact_manifest(
            artifact_dir=temporary_artifact,
            created_at_utc=created_at_utc,
            code_commit=code_commit,
            code_dirty=code_dirty,
            source_run=run_dir.name,
            source_checkpoint=str(checkpoint_path.relative_to(run_dir).as_posix()),
        )
        write_artifact_manifest(
            manifest,
            temporary_artifact / "artifact_manifest.json",
        )
        validation = validate_model_artifact(
            temporary_artifact,
            device=config.validation_device,
        )
        temporary_archive = temporary_root / config.archive_name
        _write_deterministic_tar_gz(temporary_artifact, temporary_archive)
        temporary_report = temporary_root / MODEL_BUILD_REPORT_NAME
        report = {
            "status": "passed",
            "source": {
                "run_dir": str(run_dir),
                "config_path": str(saved_config_path),
                "checkpoint_path": str(checkpoint_path),
                "inference_policy_path": str(
                    Path(config.inference_policy_path).expanduser().resolve()
                ),
            },
            "outputs": {
                "artifact_dir": str(artifact_dir),
                "archive_path": str(archive_path),
                "archive_size_bytes": temporary_archive.stat().st_size,
                "archive_sha256": sha256_file(temporary_archive),
            },
            "manifest": manifest,
            "validation": validation.as_dict(),
        }
        temporary_report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.move(str(temporary_artifact), str(artifact_dir))
        shutil.move(str(temporary_archive), str(archive_path))
        shutil.move(str(temporary_report), str(report_path))

    return ModelArtifactBuildResult(
        artifact_dir=artifact_dir,
        archive_path=archive_path,
        report_path=report_path,
        manifest=manifest,
        validation=validation,
    )


def _validate_initial_release_config(cfg: Any) -> tuple[str, int]:
    diffusion_type = str(
        OmegaConf.select(cfg, "diffusion.type", default="Discriminative")
    ).strip()
    if diffusion_type.lower() != "discriminative":
        raise ModelArtifactError(
            "The initial release rejects generative inference artifacts; configured "
            f"diffusion.type={diffusion_type!r}."
        )
    raw_dims = OmegaConf.select(cfg, "model.spatial_dims", default=None)
    if raw_dims is None:
        raw_dims = OmegaConf.select(cfg, "data_mode.dim", default=None)
    dims_token = str(raw_dims).strip().lower()
    if dims_token.endswith("d"):
        dims_token = dims_token[:-1]
    if dims_token != "3":
        raise ModelArtifactError(
            "The initial release model artifact must be a certified 3D model; "
            f"resolved spatial dimensionality={raw_dims!r}."
        )
    dataset_id_value = OmegaConf.select(cfg, "dataset.id", default=None)
    if not isinstance(dataset_id_value, str) or not dataset_id_value.strip():
        raise ModelArtifactError(
            "Saved model config must declare dataset.id for registered preprocessing."
        )
    return dataset_id_value.strip().lower(), 3


def _validate_release_predictor_capabilities(capabilities: Any) -> None:
    if not isinstance(capabilities, PredictorCapabilities):
        raise ModelArtifactError(
            "Prepared release predictor did not expose PredictorCapabilities."
        )
    if capabilities.model_family.lower() != "discriminative":
        raise ModelArtifactError("Initial release predictor must be discriminative.")
    if capabilities.spatial_dims != 3:
        raise ModelArtifactError("Initial release predictor must be 3D.")


def _resolve_code_provenance(
    config: ModelArtifactBuildConfig,
) -> tuple[str, bool, str]:
    code_commit = config.code_commit or _git_output("rev-parse", "HEAD")
    if config.code_dirty is None:
        code_dirty = bool(_git_output("status", "--porcelain"))
    else:
        code_dirty = config.code_dirty
    if config.created_at_utc:
        created_at = config.created_at_utc
    else:
        commit_time = _git_output("show", "-s", "--format=%cI", code_commit)
        try:
            created_at = (
                datetime.fromisoformat(commit_time.replace("Z", "+00:00"))
                .astimezone(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
        except ValueError as exc:
            raise ModelArtifactError(
                f"Could not normalize commit timestamp {commit_time!r}."
            ) from exc
    return code_commit, bool(code_dirty), created_at


def _git_output(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ModelArtifactError(f"Could not resolve Git build provenance: {exc}") from exc
    return completed.stdout.strip()


def _write_deterministic_tar_gz(artifact_dir: Path, archive_path: Path) -> None:
    with archive_path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            compresslevel=9,
            mtime=0,
        ) as gzip_handle:
            with tarfile.open(
                fileobj=gzip_handle,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as archive:
                for file_name in ARTIFACT_FILENAMES:
                    data = (artifact_dir / file_name).read_bytes()
                    info = tarfile.TarInfo(name=file_name)
                    info.size = len(data)
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    archive.addfile(info, BytesIO(data))


__all__ = [
    "ARTIFACT_DIRECTORY_NAME",
    "MODEL_BUILD_REPORT_NAME",
    "ModelArtifactBuildResult",
    "ModelArtifactError",
    "ModelArtifactValidationResult",
    "build_model_artifact",
    "resolve_release_checkpoint",
    "validate_model_artifact",
]
