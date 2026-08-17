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
from typing import Any, Mapping

import yaml
from omegaconf import OmegaConf

from scripts.evaluation.core.model_loader import CHECKPOINT_DIRS
from scripts.gc_submission_builder.build_config import (
    ModelArtifactBuildConfig,
    ModelArtifactMemberBuildConfig,
    compose_inference_policy_file,
)
from scripts.gc_submission_builder.release_manifest import (
    artifact_members,
    create_artifact_manifest,
    create_ensemble_artifact_manifest,
    iter_artifact_file_paths,
    sha256_file,
    verify_artifact_manifest,
    write_artifact_manifest,
)
from src.data.loader_stack.contracts import PreprocessingAdapterError
from src.data.loader_stack.preprocessing import get_preprocessing_adapter
from src.inference.case_producer import build_case_producer
from src.inference.contracts import PredictorCapabilities
from src.inference.ensemble import model_ensemble_contract
from src.inference.policy import resolve_inference_policy
from src.inference.pipeline import build_ensemble_probability_executor
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
    ensemble_enabled: bool = False
    ensemble_method: str = "mean"
    member_count: int = 1
    member_ids: tuple[str, ...] = ("model",)
    tta_enabled: bool = False
    tta_flip_axes: tuple[str, ...] = ()
    tta_view_count_per_model: int = 1
    effective_prediction_count: int = 1

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
        manifest = verify_artifact_manifest(root)
        members = artifact_members(root, manifest)
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

    composed_members: list[tuple[str, Any, Any]] = []
    canonical_contract: Mapping[str, Any] | None = None
    canonical_dataset_id: str | None = None
    canonical_spatial_dims: int | None = None
    canonical_capabilities: PredictorCapabilities | None = None
    canonical_policy = None
    canonical_policy_source: str | None = None
    canonical_adapter = None
    runtime = parse_inference_runtime(
        OmegaConf.load(
            PROJECT_ROOT / "configs" / "inference_runtime" / "gc_submission.yaml"
        )
    )

    for member in members:
        try:
            saved_cfg = OmegaConf.load(member.config_path)
            composed = OmegaConf.create(OmegaConf.to_container(saved_cfg, resolve=False))
            OmegaConf.set_struct(composed, False)
            OmegaConf.update(composed, "inference", policy_payload, merge=False)
            resolved_policy = resolve_inference_policy(composed)
            validate_runtime_compatibility(
                resolved_policy.policy,
                runtime,
                AssessmentContext(requires_ground_truth=False, threshold_sweep=False),
            )
        except Exception as exc:
            raise ModelArtifactError(
                f"Artifact member {member.member_id!r} inference policy is incompatible "
                f"with gc_submission: {exc}"
            ) from exc
        if resolved_policy.source != "explicit_top_level":
            raise ModelArtifactError(
                "Artifact validation requires inference_policy.yaml to win as the "
                "explicit top-level inference policy."
            )

        dataset_id, spatial_dims = _validate_initial_release_config(composed)
        try:
            adapter = get_preprocessing_adapter(dataset_id)
            model = load_model_strict(composed, member.weights_path, device=device)
            predictor = build_probability_predictor(backend=model, cfg=composed)
            capabilities = predictor.capabilities
            _validate_release_predictor_capabilities(capabilities)
            producer = build_case_producer(
                dataset_id=dataset_id,
                dataset_cfg=composed.dataset,
                load_labels=False,
            )
        except (PreprocessingAdapterError, StrictModelLoadError) as exc:
            raise ModelArtifactError(
                f"Artifact member {member.member_id!r} strict validation failed: {exc}"
            ) from exc
        except Exception as exc:
            raise ModelArtifactError(
                f"Artifact member {member.member_id!r} construction/capability "
                f"validation failed: {exc}"
            ) from exc
        if producer.adapter.dataset_id != adapter.dataset_id:
            raise ModelArtifactError(
                f"Artifact member {member.member_id!r} preprocessing producer resolved "
                "a different adapter than its saved model contract."
            )
        if int(capabilities.spatial_dims) != spatial_dims:
            raise ModelArtifactError(
                f"Artifact member {member.member_id!r} predictor dimensionality "
                "disagrees with its saved model contract."
            )

        contract = model_ensemble_contract(composed)
        if canonical_contract is None:
            canonical_contract = contract
            canonical_dataset_id = dataset_id
            canonical_spatial_dims = spatial_dims
            canonical_capabilities = capabilities
            canonical_policy = resolved_policy.policy
            canonical_policy_source = resolved_policy.source
            canonical_adapter = adapter
        elif contract != canonical_contract:
            raise ModelArtifactError(
                f"Artifact member {member.member_id!r} has an incompatible model, "
                "dataset, or preprocessing contract. Fold/provenance differences are "
                "allowed, scientific inference-contract differences are not."
            )
        composed_members.append((member.member_id, model, composed))

    if canonical_policy is None or canonical_capabilities is None or canonical_adapter is None:
        raise ModelArtifactError("Artifact did not provide any model members.")
    if len(members) > 1 and not canonical_policy.ensemble.enabled:
        raise ModelArtifactError(
            "A multi-member artifact requires inference.ensemble.enabled=true; member "
            "count is discovered from the artifact and must not be configured separately."
        )
    if canonical_policy.ensemble.enabled:
        try:
            build_ensemble_probability_executor(composed_members)
        except Exception as exc:
            raise ModelArtifactError(f"Ensemble executor validation failed: {exc}") from exc

    return ModelArtifactValidationResult(
        dataset_id=str(canonical_dataset_id),
        preprocessing_adapter=canonical_adapter.dataset_id,
        model_family=str(canonical_capabilities.model_family),
        spatial_dims=int(canonical_spatial_dims),
        input_channels=int(canonical_capabilities.input_channels),
        output_channels=int(canonical_capabilities.output_channels),
        inference_policy_source=str(canonical_policy_source),
        output_space=canonical_policy.output_space,
        precision=canonical_policy.precision,
        sliding_window_batch_size=canonical_policy.sliding_window.sw_batch_size,
        runtime_profile=runtime.profile,
        strict_checkpoint_load=True,
        ensemble_enabled=canonical_policy.ensemble.enabled,
        ensemble_method=canonical_policy.ensemble.method,
        member_count=len(members),
        member_ids=tuple(member.member_id for member in members),
        tta_enabled=canonical_policy.tta.enabled,
        tta_flip_axes=canonical_policy.tta.flip_axes,
        tta_view_count_per_model=1 + len(canonical_policy.tta.flip_axes),
        effective_prediction_count=(
            len(members) * (1 + len(canonical_policy.tta.flip_axes))
        ),
    )


def build_model_artifact(config: ModelArtifactBuildConfig) -> ModelArtifactBuildResult:
    """Build, load-test, and archive one arbitrary-N model artifact."""

    if not isinstance(config, ModelArtifactBuildConfig):
        raise ModelArtifactError("config must be a ModelArtifactBuildConfig instance.")
    source_members: list[
        tuple[ModelArtifactMemberBuildConfig, Path, Path, Path]
    ] = []
    for member in config.resolved_members():
        run_dir = Path(member.run_dir).expanduser().resolve()
        saved_config_path = run_dir / ".hydra" / "config.yaml"
        if not saved_config_path.is_file():
            raise ModelArtifactError(
                f"Run config for member {member.member_id!r} not found. Expected "
                f"complete Hydra config at {saved_config_path}."
            )
        checkpoint_path = resolve_release_checkpoint(
            run_dir,
            member.checkpoint,
            use_ema=member.use_ema,
        )
        source_members.append((member, run_dir, saved_config_path, checkpoint_path))
    try:
        inference_policy = compose_inference_policy_file(config.inference_policy_path)
    except Exception as exc:
        raise ModelArtifactError(f"Could not compose selected inference policy: {exc}") from exc

    output_dir = Path(config.output_dir).expanduser().resolve()
    for _member, run_dir, _saved_config, _checkpoint in source_members:
        try:
            output_dir.relative_to(run_dir)
        except ValueError:
            pass
        else:
            raise ModelArtifactError(
                "Model build output_dir must remain outside every immutable training run."
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
        (temporary_artifact / "inference_policy.yaml").write_text(
            yaml.safe_dump(inference_policy, sort_keys=True),
            encoding="utf-8",
        )
        if config.members:
            member_manifest_sources: list[dict[str, str]] = []
            members_root = temporary_artifact / "members"
            members_root.mkdir()
            for member, run_dir, saved_config_path, checkpoint_path in source_members:
                member_root = members_root / member.member_id
                member_root.mkdir()
                shutil.copyfile(saved_config_path, member_root / "config.yaml")
                shutil.copyfile(checkpoint_path, member_root / "weights.pth")
                member_manifest_sources.append(
                    {
                        "id": member.member_id,
                        "source_run": run_dir.name,
                        "source_checkpoint": str(
                            checkpoint_path.relative_to(run_dir).as_posix()
                        ),
                    }
                )
            manifest = create_ensemble_artifact_manifest(
                artifact_dir=temporary_artifact,
                created_at_utc=created_at_utc,
                code_commit=code_commit,
                code_dirty=code_dirty,
                members=member_manifest_sources,
            )
        else:
            _member, run_dir, saved_config_path, checkpoint_path = source_members[0]
            shutil.copyfile(saved_config_path, temporary_artifact / "config.yaml")
            shutil.copyfile(checkpoint_path, temporary_artifact / "weights.pth")
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
                "inference_policy_path": str(
                    Path(config.inference_policy_path).expanduser().resolve()
                ),
                "members": [
                    {
                        "id": member.member_id,
                        "run_dir": str(run_dir),
                        "config_path": str(saved_config_path),
                        "checkpoint_path": str(checkpoint_path),
                    }
                    for member, run_dir, saved_config_path, checkpoint_path in source_members
                ],
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
        if not config.members:
            _member, run_dir, saved_config_path, checkpoint_path = source_members[0]
            report["source"].update(
                {
                    "run_dir": str(run_dir),
                    "config_path": str(saved_config_path),
                    "checkpoint_path": str(checkpoint_path),
                }
            )
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
                for file_path in iter_artifact_file_paths(artifact_dir):
                    relative_name = file_path.relative_to(artifact_dir).as_posix()
                    data = file_path.read_bytes()
                    info = tarfile.TarInfo(name=relative_name)
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
