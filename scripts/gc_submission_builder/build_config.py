"""Strict builder-owned configuration for Grand Challenge model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

import yaml
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"
ALLOWED_CONFIG_KEYS = frozenset(
    {
        "run_dir",
        "checkpoint",
        "use_ema",
        "inference_policy",
        "output_dir",
        "archive_name",
        "validation_device",
        "code_commit",
        "code_dirty",
        "created_at_utc",
        "members",
    }
)
ALLOWED_MEMBER_KEYS = frozenset({"id", "run_dir", "checkpoint", "use_ema"})


class BuilderConfigError(ValueError):
    """Raised when builder configuration is missing, ambiguous, or out of scope."""


@dataclass(frozen=True)
class ModelArtifactMemberBuildConfig:
    """One explicitly selected source model; member count is list-derived."""

    member_id: str
    run_dir: Path
    checkpoint: str
    use_ema: bool = False


@dataclass(frozen=True)
class ModelArtifactBuildConfig:
    """Builder-only inputs; trained model attributes deliberately do not appear."""

    run_dir: Path | None
    checkpoint: str | None
    use_ema: bool
    inference_policy_path: Path
    output_dir: Path
    archive_name: str
    validation_device: str = "cpu"
    code_commit: str | None = None
    code_dirty: bool | None = None
    created_at_utc: str | None = None
    members: Tuple[ModelArtifactMemberBuildConfig, ...] = ()

    def resolved_members(self) -> Tuple[ModelArtifactMemberBuildConfig, ...]:
        if self.members:
            return self.members
        if self.run_dir is None or self.checkpoint is None:
            raise BuilderConfigError(
                "Specify either members or the legacy run_dir/checkpoint pair."
            )
        return (
            ModelArtifactMemberBuildConfig(
                member_id="model",
                run_dir=self.run_dir,
                checkpoint=self.checkpoint,
                use_ema=self.use_ema,
            ),
        )


def load_model_artifact_build_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> ModelArtifactBuildConfig:
    """Load one allowlisted builder config and apply explicit CLI-owned overrides."""

    path = Path(config_path).expanduser().resolve()
    raw = _load_yaml_mapping(path, field_name="builder config")
    unknown = sorted(set(raw) - ALLOWED_CONFIG_KEYS)
    if unknown:
        raise BuilderConfigError(f"Builder config contains unknown keys: {unknown}.")

    override_values = {
        str(key): value
        for key, value in (overrides or {}).items()
        if value is not None
    }
    unknown_overrides = sorted(set(override_values) - ALLOWED_CONFIG_KEYS)
    if unknown_overrides:
        raise BuilderConfigError(
            f"Builder overrides contain unknown keys: {unknown_overrides}."
        )
    values = {**raw, **override_values}

    members = _parse_members(
        values.get("members"),
        base_dir=Path.cwd() if "members" in override_values else path.parent,
    )
    run_dir = None
    checkpoint = None
    if members:
        if values.get("run_dir") is not None or values.get("checkpoint") is not None:
            raise BuilderConfigError(
                "members is mutually exclusive with run_dir/checkpoint."
            )
        if values.get("use_ema", False) is not False:
            raise BuilderConfigError(
                "Top-level use_ema is not valid with members; set it per member."
            )
    else:
        run_dir = _required_path(
            values.get("run_dir"),
            field_name="run_dir",
            base_dir=Path.cwd() if "run_dir" in override_values else path.parent,
        )
        checkpoint = _required_string(values.get("checkpoint"), "checkpoint")
    output_dir = _required_path(
        values.get("output_dir"),
        field_name="output_dir",
        base_dir=Path.cwd() if "output_dir" in override_values else path.parent,
    )
    policy_path = _required_path(
        values.get("inference_policy"),
        field_name="inference_policy",
        base_dir=(
            Path.cwd() if "inference_policy" in override_values else path.parent
        ),
    )
    raw_archive_name = values.get("archive_name")
    archive_name = (
        "algorithmmodel.tar.gz"
        if raw_archive_name is None
        else _required_string(raw_archive_name, "archive_name")
    )
    if (
        Path(archive_name).name != archive_name
        or not archive_name.endswith(".tar.gz")
        or "/" in archive_name
        or "\\" in archive_name
    ):
        raise BuilderConfigError(
            "archive_name must be a root-level filename ending in '.tar.gz'."
        )
    use_ema = values.get("use_ema", False)
    if type(use_ema) is not bool:
        raise BuilderConfigError("use_ema must be a boolean.")
    code_dirty = values.get("code_dirty")
    if code_dirty is not None and type(code_dirty) is not bool:
        raise BuilderConfigError("code_dirty must be a boolean or null.")
    validation_device = _required_string(
        values.get("validation_device", "cpu"),
        "validation_device",
    )
    code_commit = _optional_string(values.get("code_commit"), "code_commit")
    created_at_utc = _optional_string(
        values.get("created_at_utc"),
        "created_at_utc",
    )
    if not policy_path.is_file():
        raise BuilderConfigError(f"Inference policy file does not exist: {policy_path}")

    return ModelArtifactBuildConfig(
        run_dir=run_dir,
        checkpoint=checkpoint,
        use_ema=use_ema,
        inference_policy_path=policy_path,
        output_dir=output_dir,
        archive_name=archive_name,
        validation_device=validation_device,
        code_commit=code_commit,
        code_dirty=code_dirty,
        created_at_utc=created_at_utc,
        members=members,
    )


def _parse_members(value: Any, *, base_dir: Path) -> Tuple[ModelArtifactMemberBuildConfig, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not value:
        raise BuilderConfigError("members must be a non-empty list when provided.")
    parsed: list[ModelArtifactMemberBuildConfig] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise BuilderConfigError(f"members[{index}] must be a mapping.")
        unknown = sorted(set(raw) - ALLOWED_MEMBER_KEYS)
        if unknown:
            raise BuilderConfigError(
                f"members[{index}] contains unknown keys: {unknown}."
            )
        member_id = _required_string(raw.get("id"), f"members[{index}].id")
        allowed_characters = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        )
        if (
            any(character not in allowed_characters for character in member_id)
            or not member_id[0].isalnum()
        ):
            raise BuilderConfigError(
                f"members[{index}].id must match [A-Za-z0-9][A-Za-z0-9._-]*."
            )
        if member_id in seen:
            raise BuilderConfigError(f"Duplicate member id {member_id!r}.")
        seen.add(member_id)
        use_ema = raw.get("use_ema", False)
        if type(use_ema) is not bool:
            raise BuilderConfigError(f"members[{index}].use_ema must be a boolean.")
        parsed.append(
            ModelArtifactMemberBuildConfig(
                member_id=member_id,
                run_dir=_required_path(
                    raw.get("run_dir"),
                    field_name=f"members[{index}].run_dir",
                    base_dir=base_dir,
                ),
                checkpoint=_required_string(
                    raw.get("checkpoint"), f"members[{index}].checkpoint"
                ),
                use_ema=use_ema,
            )
        )
    return tuple(parsed)


def compose_inference_policy_file(path: str | Path) -> dict[str, Any]:
    """Materialize the selected inference preset into one standalone mapping."""

    resolved_path = Path(path).expanduser().resolve()
    composed = _compose_policy(resolved_path, stack=())
    if "defaults" in composed:
        raise BuilderConfigError(
            "Composed inference policy unexpectedly retained a defaults block."
        )
    try:
        resolved = OmegaConf.to_container(OmegaConf.create(composed), resolve=True)
    except Exception as exc:
        raise BuilderConfigError(
            f"Could not resolve inference policy {resolved_path}: {exc}"
        ) from exc
    if not isinstance(resolved, dict):
        raise BuilderConfigError("Composed inference policy must be a mapping.")
    return resolved


def _compose_policy(path: Path, *, stack: tuple[Path, ...]) -> dict[str, Any]:
    if path in stack:
        chain = " -> ".join(item.name for item in (*stack, path))
        raise BuilderConfigError(f"Inference policy inheritance cycle: {chain}.")
    raw = _load_yaml_mapping(path, field_name="inference policy")
    defaults = raw.pop("defaults", [])
    if defaults is None:
        defaults = []
    if not isinstance(defaults, list):
        raise BuilderConfigError(
            f"Inference policy defaults must be a list in {path}."
        )

    result = OmegaConf.create({})
    body_applied = False
    for entry in defaults:
        if entry == "_self_":
            result = OmegaConf.merge(result, OmegaConf.create(raw))
            body_applied = True
            continue
        if not isinstance(entry, str) or not entry.strip():
            raise BuilderConfigError(
                f"Inference policy {path} contains unsupported defaults entry {entry!r}."
            )
        parent_name = entry.strip()
        parent_path = path.parent / (
            parent_name if parent_name.endswith(".yaml") else f"{parent_name}.yaml"
        )
        parent = _compose_policy(parent_path.resolve(), stack=(*stack, path))
        result = OmegaConf.merge(result, OmegaConf.create(parent))
    if not body_applied:
        result = OmegaConf.merge(result, OmegaConf.create(raw))
    plain = OmegaConf.to_container(result, resolve=False)
    if not isinstance(plain, dict):
        raise BuilderConfigError(f"Inference policy {path} did not compose to a mapping.")
    return plain


def _load_yaml_mapping(path: Path, *, field_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise BuilderConfigError(f"{field_name.capitalize()} file does not exist: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BuilderConfigError(f"Could not load {field_name} {path}: {exc}") from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise BuilderConfigError(f"{field_name.capitalize()} must be a mapping: {path}")
    return dict(loaded)


def _required_path(value: Any, *, field_name: str, base_dir: Path) -> Path:
    token = _required_string(value, field_name)
    candidate = Path(token).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise BuilderConfigError(f"{field_name} must be a non-empty string/path.")
    return str(value).strip()


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field_name)


__all__ = [
    "BuilderConfigError",
    "DEFAULT_CONFIG_PATH",
    "ModelArtifactBuildConfig",
    "ModelArtifactMemberBuildConfig",
    "compose_inference_policy_file",
    "load_model_artifact_build_config",
]
