"""Versioned, path-safe manifest and hash verification for model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


INFERENCE_API_VERSION = 1
ENSEMBLE_ARTIFACT_SCHEMA_VERSION = 2
ARTIFACT_FILENAMES = (
    "artifact_manifest.json",
    "config.yaml",
    "weights.pth",
    "inference_policy.yaml",
)
PAYLOAD_FILENAMES = ARTIFACT_FILENAMES[1:]
LEGACY_MANIFEST_KEYS = frozenset(
    {
        "inference_api_version",
        "created_at_utc",
        "code_commit",
        "code_dirty",
        "source_run",
        "source_checkpoint",
        "config_path",
        "weights_path",
        "inference_policy_path",
        "config_sha256",
        "weights_sha256",
        "inference_policy_sha256",
    }
)
ENSEMBLE_MANIFEST_KEYS = frozenset(
    {
        "inference_api_version",
        "artifact_schema_version",
        "created_at_utc",
        "code_commit",
        "code_dirty",
        "inference_policy_path",
        "inference_policy_sha256",
        "members",
    }
)
MEMBER_MANIFEST_KEYS = frozenset(
    {
        "id",
        "source_run",
        "source_checkpoint",
        "config_path",
        "weights_path",
        "config_sha256",
        "weights_sha256",
    }
)
_MEMBER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ArtifactManifestError(RuntimeError):
    """Raised when an artifact manifest or its declared files are invalid."""


@dataclass(frozen=True)
class ArtifactMember:
    member_id: str
    config_path: Path
    weights_path: Path
    source_run: str
    source_checkpoint: str


def sha256_file(path: str | Path) -> str:
    """Return the lower-case SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_artifact_manifest(
    *,
    artifact_dir: str | Path,
    created_at_utc: str,
    code_commit: str,
    code_dirty: bool,
    source_run: str,
    source_checkpoint: str,
) -> dict[str, Any]:
    """Create the legacy single-model artifact manifest."""

    root = Path(artifact_dir)
    return {
        "inference_api_version": INFERENCE_API_VERSION,
        "created_at_utc": str(created_at_utc),
        "code_commit": str(code_commit),
        "code_dirty": bool(code_dirty),
        "source_run": str(source_run),
        "source_checkpoint": str(source_checkpoint),
        "config_path": "config.yaml",
        "weights_path": "weights.pth",
        "inference_policy_path": "inference_policy.yaml",
        "config_sha256": sha256_file(root / "config.yaml"),
        "weights_sha256": sha256_file(root / "weights.pth"),
        "inference_policy_sha256": sha256_file(root / "inference_policy.yaml"),
    }


def create_ensemble_artifact_manifest(
    *,
    artifact_dir: str | Path,
    created_at_utc: str,
    code_commit: str,
    code_dirty: bool,
    members: Iterable[Mapping[str, str]],
) -> dict[str, Any]:
    """Create a generated arbitrary-N manifest; no member count is configured."""

    root = Path(artifact_dir)
    member_records: list[dict[str, Any]] = []
    for raw in sorted(members, key=lambda item: str(item["id"])):
        member_id = _validate_member_id(str(raw["id"]))
        config_path = f"members/{member_id}/config.yaml"
        weights_path = f"members/{member_id}/weights.pth"
        member_records.append(
            {
                "id": member_id,
                "source_run": str(raw["source_run"]),
                "source_checkpoint": str(raw["source_checkpoint"]),
                "config_path": config_path,
                "weights_path": weights_path,
                "config_sha256": sha256_file(root / config_path),
                "weights_sha256": sha256_file(root / weights_path),
            }
        )
    if not member_records:
        raise ArtifactManifestError("An ensemble artifact requires at least one member.")
    return {
        "inference_api_version": INFERENCE_API_VERSION,
        "artifact_schema_version": ENSEMBLE_ARTIFACT_SCHEMA_VERSION,
        "created_at_utc": str(created_at_utc),
        "code_commit": str(code_commit),
        "code_dirty": bool(code_dirty),
        "inference_policy_path": "inference_policy.yaml",
        "inference_policy_sha256": sha256_file(root / "inference_policy.yaml"),
        "members": member_records,
    }


def write_artifact_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    """Write one deterministic JSON manifest."""

    output = Path(path)
    output.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def verify_artifact_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    """Verify exact contents, safe member paths, versions, and payload hashes."""

    root = Path(artifact_dir).expanduser().resolve()
    if not root.is_dir():
        raise ArtifactManifestError(f"Model artifact directory does not exist: {root}")
    manifest_path = root / "artifact_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactManifestError(f"Could not read artifact manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ArtifactManifestError("Artifact manifest must be a JSON object.")
    if "artifact_schema_version" not in manifest:
        _verify_legacy_artifact(root, manifest)
    else:
        _verify_ensemble_artifact(root, manifest)
    return manifest


def artifact_members(
    artifact_dir: str | Path,
    manifest: Mapping[str, Any] | None = None,
) -> tuple[ArtifactMember, ...]:
    """Return verified members discovered from the extracted artifact topology."""

    root = Path(artifact_dir).expanduser().resolve()
    verified = dict(manifest) if manifest is not None else verify_artifact_manifest(root)
    if "artifact_schema_version" not in verified:
        return (
            ArtifactMember(
                member_id="model",
                config_path=root / "config.yaml",
                weights_path=root / "weights.pth",
                source_run=str(verified["source_run"]),
                source_checkpoint=str(verified["source_checkpoint"]),
            ),
        )
    return tuple(
        ArtifactMember(
            member_id=str(record["id"]),
            config_path=root / str(record["config_path"]),
            weights_path=root / str(record["weights_path"]),
            source_run=str(record["source_run"]),
            source_checkpoint=str(record["source_checkpoint"]),
        )
        for record in verified["members"]
    )


def iter_artifact_file_paths(artifact_dir: str | Path) -> tuple[Path, ...]:
    """Return every regular artifact file in deterministic root-relative order."""

    root = Path(artifact_dir).expanduser().resolve()
    return tuple(
        sorted(
            (path for path in root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )


def _verify_legacy_artifact(root: Path, manifest: Mapping[str, Any]) -> None:
    actual_files = {path.name for path in root.iterdir()}
    expected_files = set(ARTIFACT_FILENAMES)
    if actual_files != expected_files:
        raise ArtifactManifestError(
            "Model artifact must contain exactly the release allowlist; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}."
        )
    _validate_fields(manifest, LEGACY_MANIFEST_KEYS, "Legacy artifact manifest")
    _validate_api_version(manifest)
    for field_name, expected_name in (
        ("config_path", "config.yaml"),
        ("weights_path", "weights.pth"),
        ("inference_policy_path", "inference_policy.yaml"),
    ):
        if manifest[field_name] != expected_name:
            raise ArtifactManifestError(
                f"Manifest {field_name} must be the root-relative path {expected_name!r}."
            )
    _verify_hash(root / "config.yaml", manifest["config_sha256"])
    _verify_hash(root / "weights.pth", manifest["weights_sha256"])
    _verify_hash(
        root / "inference_policy.yaml", manifest["inference_policy_sha256"]
    )


def _verify_ensemble_artifact(root: Path, manifest: Mapping[str, Any]) -> None:
    _validate_fields(manifest, ENSEMBLE_MANIFEST_KEYS, "Ensemble artifact manifest")
    _validate_api_version(manifest)
    if manifest["artifact_schema_version"] != ENSEMBLE_ARTIFACT_SCHEMA_VERSION:
        raise ArtifactManifestError(
            "Unsupported artifact_schema_version "
            f"{manifest['artifact_schema_version']!r}; expected "
            f"{ENSEMBLE_ARTIFACT_SCHEMA_VERSION}."
        )
    actual_root = {path.name for path in root.iterdir()}
    expected_root = {"artifact_manifest.json", "inference_policy.yaml", "members"}
    if actual_root != expected_root or not (root / "members").is_dir():
        raise ArtifactManifestError(
            "Ensemble artifact root must contain only artifact_manifest.json, "
            "inference_policy.yaml, and the members directory."
        )
    if manifest["inference_policy_path"] != "inference_policy.yaml":
        raise ArtifactManifestError(
            "Ensemble manifest inference_policy_path must be 'inference_policy.yaml'."
        )
    _verify_hash(
        root / "inference_policy.yaml", manifest["inference_policy_sha256"]
    )

    raw_members = manifest["members"]
    if not isinstance(raw_members, list) or not raw_members:
        raise ArtifactManifestError("Ensemble manifest members must be a non-empty list.")
    discovered_dirs = sorted(
        path.name for path in (root / "members").iterdir() if path.is_dir()
    )
    unexpected_files = sorted(
        path.name for path in (root / "members").iterdir() if not path.is_dir()
    )
    if unexpected_files:
        raise ArtifactManifestError(
            f"Ensemble members directory contains non-directory entries: {unexpected_files}."
        )
    manifest_ids: list[str] = []
    for record in raw_members:
        if not isinstance(record, dict):
            raise ArtifactManifestError("Each ensemble member manifest entry must be an object.")
        _validate_fields(record, MEMBER_MANIFEST_KEYS, "Ensemble member manifest")
        member_id = _validate_member_id(str(record["id"]))
        if member_id in manifest_ids:
            raise ArtifactManifestError(f"Duplicate ensemble member id {member_id!r}.")
        manifest_ids.append(member_id)
        expected_config = f"members/{member_id}/config.yaml"
        expected_weights = f"members/{member_id}/weights.pth"
        if record["config_path"] != expected_config or record["weights_path"] != expected_weights:
            raise ArtifactManifestError(
                f"Ensemble member {member_id!r} paths must remain under its member directory."
            )
        member_dir = root / "members" / member_id
        actual_member_files = {path.name for path in member_dir.iterdir()}
        if actual_member_files != {"config.yaml", "weights.pth"}:
            raise ArtifactManifestError(
                f"Ensemble member {member_id!r} must contain exactly config.yaml and weights.pth."
            )
        _verify_hash(root / expected_config, record["config_sha256"])
        _verify_hash(root / expected_weights, record["weights_sha256"])
    if sorted(manifest_ids) != discovered_dirs:
        raise ArtifactManifestError(
            "Ensemble manifest members do not match discovered member directories; "
            f"manifest={sorted(manifest_ids)}, discovered={discovered_dirs}."
        )


def _validate_fields(
    mapping: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    unknown = sorted(set(mapping) - expected)
    missing = sorted(expected - set(mapping))
    if unknown or missing:
        raise ArtifactManifestError(
            f"{label} fields are invalid; missing={missing}, unknown={unknown}."
        )


def _validate_api_version(manifest: Mapping[str, Any]) -> None:
    if manifest["inference_api_version"] != INFERENCE_API_VERSION:
        raise ArtifactManifestError(
            "Unsupported inference_api_version "
            f"{manifest['inference_api_version']!r}; expected {INFERENCE_API_VERSION}."
        )


def _validate_member_id(value: str) -> str:
    if not _MEMBER_ID_PATTERN.fullmatch(value):
        raise ArtifactManifestError(
            "Ensemble member ids must match [A-Za-z0-9][A-Za-z0-9._-]*; "
            f"got {value!r}."
        )
    return value


def _verify_hash(path: Path, expected: Any) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ArtifactManifestError(
            f"{path.name} SHA-256 mismatch: manifest={expected}, observed={observed}."
        )


__all__ = [
    "ARTIFACT_FILENAMES",
    "ArtifactManifestError",
    "ArtifactMember",
    "ENSEMBLE_ARTIFACT_SCHEMA_VERSION",
    "INFERENCE_API_VERSION",
    "artifact_members",
    "create_artifact_manifest",
    "create_ensemble_artifact_manifest",
    "iter_artifact_file_paths",
    "sha256_file",
    "verify_artifact_manifest",
    "write_artifact_manifest",
]
