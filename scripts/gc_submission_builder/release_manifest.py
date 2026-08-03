"""Versioned, path-safe manifest and hash verification for model artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


INFERENCE_API_VERSION = 1
ARTIFACT_FILENAMES = (
    "artifact_manifest.json",
    "config.yaml",
    "weights.pth",
    "inference_policy.yaml",
)
PAYLOAD_FILENAMES = ARTIFACT_FILENAMES[1:]
MANIFEST_KEYS = frozenset(
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


class ArtifactManifestError(RuntimeError):
    """Raised when an artifact manifest or its declared files are invalid."""


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
    """Create generated artifact facts without duplicating model attributes."""

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


def write_artifact_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    """Write one deterministic JSON manifest."""

    output = Path(path)
    output.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def verify_artifact_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    """Verify exact archive contents, safe paths, version, and payload hashes."""

    root = Path(artifact_dir).expanduser().resolve()
    if not root.is_dir():
        raise ArtifactManifestError(f"Model artifact directory does not exist: {root}")
    actual_files = {path.name for path in root.iterdir()}
    expected_files = set(ARTIFACT_FILENAMES)
    if actual_files != expected_files:
        raise ArtifactManifestError(
            "Model artifact must contain exactly the release allowlist; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}."
        )
    manifest_path = root / "artifact_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactManifestError(f"Could not read artifact manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ArtifactManifestError("Artifact manifest must be a JSON object.")
    unknown = sorted(set(manifest) - MANIFEST_KEYS)
    missing = sorted(MANIFEST_KEYS - set(manifest))
    if unknown or missing:
        raise ArtifactManifestError(
            f"Artifact manifest fields are invalid; missing={missing}, unknown={unknown}."
        )
    if manifest["inference_api_version"] != INFERENCE_API_VERSION:
        raise ArtifactManifestError(
            "Unsupported inference_api_version "
            f"{manifest['inference_api_version']!r}; expected {INFERENCE_API_VERSION}."
        )
    for field_name, expected_name in (
        ("config_path", "config.yaml"),
        ("weights_path", "weights.pth"),
        ("inference_policy_path", "inference_policy.yaml"),
    ):
        if manifest[field_name] != expected_name:
            raise ArtifactManifestError(
                f"Manifest {field_name} must be the root-relative path {expected_name!r}."
            )
    for file_name, hash_field in (
        ("config.yaml", "config_sha256"),
        ("weights.pth", "weights_sha256"),
        ("inference_policy.yaml", "inference_policy_sha256"),
    ):
        observed = sha256_file(root / file_name)
        if observed != manifest[hash_field]:
            raise ArtifactManifestError(
                f"{file_name} SHA-256 mismatch: manifest={manifest[hash_field]}, "
                f"observed={observed}."
            )
    return manifest


__all__ = [
    "ARTIFACT_FILENAMES",
    "ArtifactManifestError",
    "INFERENCE_API_VERSION",
    "create_artifact_manifest",
    "sha256_file",
    "verify_artifact_manifest",
    "write_artifact_manifest",
]
