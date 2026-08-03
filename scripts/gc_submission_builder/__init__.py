"""Grand Challenge artifact, runtime, and container workflows.

Builder exports stay lazy so importing the container transport does not import the
post-training evaluation application or construct heavyweight scientific modules.
"""

from __future__ import annotations

from typing import Any


__all__ = [
    "ModelArtifactBuildConfig",
    "ModelArtifactBuildResult",
    "build_model_artifact",
    "validate_model_artifact",
]


def __getattr__(name: str) -> Any:
    if name == "ModelArtifactBuildConfig":
        from scripts.gc_submission_builder.build_config import ModelArtifactBuildConfig

        return ModelArtifactBuildConfig
    if name in {
        "ModelArtifactBuildResult",
        "build_model_artifact",
        "validate_model_artifact",
    }:
        from scripts.gc_submission_builder import model_artifact

        return getattr(model_artifact, name)
    raise AttributeError(name)
