"""Strict, builder-owned configuration for the Grand Challenge image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from scripts.gc_submission_builder.runtime.interfaces import load_interface_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTAINER_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "container.yaml"
)
ALLOWED_CONTAINER_KEYS = frozenset(
    {
        "image_name",
        "image_tag",
        "dockerfile",
        "interface_manifest",
        "output_dir",
        "archive_name",
        "platform",
    }
)
_IMAGE_TOKEN = re.compile(r"^[a-z0-9]+(?:[._/-][a-z0-9]+)*$")
_TAG_TOKEN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


class ContainerConfigError(ValueError):
    """Raised when image-builder configuration is unsafe or ambiguous."""


@dataclass(frozen=True)
class ContainerBuildConfig:
    image_name: str
    image_tag: str
    dockerfile: Path
    interface_manifest_path: Path
    output_dir: Path
    archive_name: str
    platform: str = "linux/amd64"

    @property
    def image_reference(self) -> str:
        return f"{self.image_name}:{self.image_tag}"


def load_container_build_config(
    config_path: str | Path = DEFAULT_CONTAINER_CONFIG_PATH,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> ContainerBuildConfig:
    path = Path(config_path).expanduser().resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContainerConfigError(f"Could not load container config: {exc}") from exc
    if not isinstance(raw, dict):
        raise ContainerConfigError("Container config must be a mapping.")
    unknown = sorted(set(raw) - ALLOWED_CONTAINER_KEYS)
    if unknown:
        raise ContainerConfigError(f"Container config contains unknown keys: {unknown}.")
    supplied = {
        str(key): value
        for key, value in (overrides or {}).items()
        if value is not None
    }
    unknown_overrides = sorted(set(supplied) - ALLOWED_CONTAINER_KEYS)
    if unknown_overrides:
        raise ContainerConfigError(
            f"Container overrides contain unknown keys: {unknown_overrides}."
        )
    values = {**raw, **supplied}

    image_name = _required_string(values.get("image_name"), "image_name").lower()
    image_tag = _required_string(values.get("image_tag"), "image_tag")
    if not _IMAGE_TOKEN.fullmatch(image_name):
        raise ContainerConfigError("image_name is not a valid lower-case Docker name.")
    if not _TAG_TOKEN.fullmatch(image_tag):
        raise ContainerConfigError("image_tag is not a valid Docker tag.")
    platform = _required_string(values.get("platform", "linux/amd64"), "platform")
    if platform != "linux/amd64":
        raise ContainerConfigError(
            "Grand Challenge release images must target platform='linux/amd64'."
        )

    dockerfile = _resolve_path(
        values.get("dockerfile"),
        field="dockerfile",
        base=Path.cwd() if "dockerfile" in supplied else path.parent,
    )
    interface_manifest = _resolve_path(
        values.get("interface_manifest"),
        field="interface_manifest",
        base=Path.cwd() if "interface_manifest" in supplied else path.parent,
    )
    output_dir = _resolve_path(
        values.get("output_dir"),
        field="output_dir",
        base=Path.cwd() if "output_dir" in supplied else path.parent,
    )
    if not dockerfile.is_file():
        raise ContainerConfigError(f"Dockerfile does not exist: {dockerfile}")
    if not interface_manifest.is_file():
        raise ContainerConfigError(
            f"Interface manifest does not exist: {interface_manifest}"
        )
    try:
        dockerfile.relative_to(PROJECT_ROOT)
        interface_manifest.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ContainerConfigError(
            "Dockerfile and interface_manifest must be files inside the build context."
        ) from exc
    load_interface_manifest(interface_manifest)

    archive_name = values.get("archive_name") or f"{image_name.replace('/', '-')}-{image_tag}.tar.gz"
    archive_name = _required_string(archive_name, "archive_name")
    if Path(archive_name).name != archive_name or not archive_name.endswith(".tar.gz"):
        raise ContainerConfigError(
            "archive_name must be a root-level filename ending in '.tar.gz'."
        )
    return ContainerBuildConfig(
        image_name=image_name,
        image_tag=image_tag,
        dockerfile=dockerfile,
        interface_manifest_path=interface_manifest,
        output_dir=output_dir,
        archive_name=archive_name,
        platform=platform,
    )


def _resolve_path(value: Any, *, field: str, base: Path) -> Path:
    token = _required_string(value, field)
    candidate = Path(token).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ContainerConfigError(f"{field} must be a non-empty string/path.")
    return str(value).strip()


__all__ = [
    "ContainerBuildConfig",
    "ContainerConfigError",
    "DEFAULT_CONTAINER_CONFIG_PATH",
    "load_container_build_config",
]
