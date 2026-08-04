"""Bounded, privacy-safe observability for the Grand Challenge runtime."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import shutil
import traceback
from types import MappingProxyType
from typing import Any, Mapping

import torch

try:
    import resource
except ImportError:  # pragma: no cover - the release runtime is Linux.
    resource = None


EVENT_MARKER = "GC_EVENT"
UNKNOWN_BUILD_FINGERPRINT = "unknown"


@dataclass(frozen=True)
class RuntimeStageError(RuntimeError):
    """A failed bounded runtime stage with safe diagnostic context."""

    stage: str
    error_code: str
    safe_detail: str
    timings_seconds: Mapping[str, float]
    total_seconds: float
    locations: tuple[Mapping[str, Any], ...] = ()

    def __str__(self) -> str:
        return f"{self.error_code} during {self.stage}: {self.safe_detail}"


def emit_gc_event(
    logger: logging.Logger,
    event: str,
    **fields: Any,
) -> None:
    """Emit one deterministic JSON line suitable for hosted log capture."""

    payload = {"event": event, **fields}
    logger.info(
        "%s %s",
        EVENT_MARKER,
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str),
    )
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass


def normalized_timings(values: Mapping[str, float]) -> Mapping[str, float]:
    """Return finite, nonnegative stage durations as an immutable mapping."""

    normalized: dict[str, float] = {}
    for name, value in values.items():
        seconds = float(value)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"Timing {name!r} must be finite and nonnegative.")
        normalized[str(name)] = seconds
    return MappingProxyType(normalized)


def resource_summary(
    *,
    device: torch.device | None = None,
    scratch_root: str | Path = "/tmp",
) -> Mapping[str, int | None]:
    """Collect bounded process, cgroup, CUDA, and scratch resource facts."""

    summary: dict[str, int | None] = {
        "host_peak_rss_bytes": _peak_rss_bytes(),
        "cgroup_memory_current_bytes": _read_cgroup_integer("memory.current"),
        "cgroup_memory_limit_bytes": _read_cgroup_integer("memory.max"),
        "scratch_total_bytes": None,
        "scratch_used_bytes": None,
        "scratch_free_bytes": None,
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
    }
    try:
        usage = shutil.disk_usage(Path(scratch_root))
        summary.update(
            scratch_total_bytes=int(usage.total),
            scratch_used_bytes=int(usage.used),
            scratch_free_bytes=int(usage.free),
        )
    except (OSError, ValueError):
        pass
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        try:
            summary["cuda_peak_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
            summary["cuda_peak_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(device)
            )
        except (RuntimeError, ValueError):
            # Telemetry must never replace the primary inference failure.
            pass
    return MappingProxyType(summary)


def runtime_identity() -> Mapping[str, Any]:
    """Return dependency and hardware identity without importing patient state."""

    versions = {
        package: _package_version(package)
        for package in ("monai", "SimpleITK", "nibabel")
    }
    visible_cuda_devices = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    gpu_name = None
    gpu_total_memory_bytes = None
    if visible_cuda_devices:
        properties = torch.cuda.get_device_properties(0)
        gpu_name = properties.name
        gpu_total_memory_bytes = int(properties.total_memory)
    return MappingProxyType(
        {
            "build_source_sha256": os.environ.get(
                "GC_SOURCE_SHA256", UNKNOWN_BUILD_FINGERPRINT
            ),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            **{f"{name.lower()}_version": value for name, value in versions.items()},
            "cpu_count": os.cpu_count(),
            "visible_cuda_device_count": visible_cuda_devices,
            "gpu_name": gpu_name,
            "gpu_total_memory_bytes": gpu_total_memory_bytes,
        }
    )


def stage_error(
    exc: BaseException,
    *,
    stage: str,
    error_code: str,
    timings_seconds: Mapping[str, float],
    total_seconds: float,
    safe_detail: str | None = None,
) -> RuntimeStageError:
    """Convert an exception into a bounded error without uncontrolled text."""

    detail = safe_detail
    if detail is None:
        if type(exc).__module__.startswith("scripts.gc_submission_builder"):
            detail = str(exc)
        else:
            detail = "Unexpected internal runtime failure."
    return RuntimeStageError(
        stage=stage,
        error_code=error_code,
        safe_detail=detail,
        timings_seconds=normalized_timings(timings_seconds),
        total_seconds=max(0.0, float(total_seconds)),
        locations=_exception_locations(exc),
    )


def failure_event_fields(exc: BaseException) -> Mapping[str, Any]:
    """Return safe structured fields for an application-boundary failure."""

    if isinstance(exc, RuntimeStageError) or all(
        hasattr(exc, name)
        for name in (
            "stage",
            "error_code",
            "safe_detail",
            "timings_seconds",
            "total_seconds",
            "locations",
        )
    ):
        return MappingProxyType(
            {
                "outcome": "failed",
                "failed_stage": exc.stage,
                "error_code": exc.error_code,
                "error_type": type(exc.__cause__ or exc).__name__,
                "detail": exc.safe_detail,
                "timings_seconds": dict(exc.timings_seconds),
                "total_seconds": exc.total_seconds,
                "locations": [dict(value) for value in exc.locations],
            }
        )
    return MappingProxyType(
        {
            "outcome": "failed",
            "failed_stage": "application_boundary",
            "error_code": "UNEXPECTED_APPLICATION_FAILURE",
            "error_type": type(exc).__name__,
            "detail": "Unexpected application-boundary failure.",
            "timings_seconds": {},
            "locations": [dict(value) for value in _exception_locations(exc)],
        }
    )


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    # Linux reports KiB; macOS reports bytes. The release container is Linux.
    return value * 1024 if platform.system() == "Linux" else value


def _read_cgroup_integer(name: str) -> int | None:
    for root in (Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")):
        path = root / name
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if token == "max":
            return None
        try:
            return int(token)
        except ValueError:
            return None
    return None


def _exception_locations(exc: BaseException) -> tuple[Mapping[str, Any], ...]:
    frames = traceback.extract_tb(exc.__traceback__)[-4:]
    return tuple(
        MappingProxyType(
            {
                "module": Path(frame.filename).name,
                "function": frame.name,
                "line": int(frame.lineno),
            }
        )
        for frame in frames
    )


__all__ = [
    "EVENT_MARKER",
    "RuntimeStageError",
    "emit_gc_event",
    "failure_event_fields",
    "normalized_timings",
    "resource_summary",
    "runtime_identity",
    "stage_error",
]
