"""Inference runtime profiles and cross-policy capability validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from omegaconf import DictConfig, ListConfig, OmegaConf

from src.inference.contracts import InvalidInferenceRuntimeError, SUPPORTED_OUTPUT_SPACES
from src.inference.policy import InferencePolicy, SUPPORTED_PRECISIONS


SUPPORTED_PROFILES = frozenset({"native", "gc_container_test", "gc_submission"})


@dataclass(frozen=True)
class RuntimeConstraints:
    allowed_output_spaces: Tuple[str, ...]
    allowed_precisions: Tuple[str, ...]
    allow_ground_truth: bool
    allow_threshold_sweep: bool
    allow_intermediate_artifacts: bool


@dataclass(frozen=True)
class InferenceRuntime:
    profile: str
    case_batch_size: int
    num_workers: int
    require_cuda: bool
    timeout_seconds: int | None
    constraints: RuntimeConstraints


@dataclass(frozen=True)
class AssessmentContext:
    requires_ground_truth: bool = False
    threshold_sweep: bool = False


@dataclass(frozen=True)
class ValidatedInferenceRequest:
    inference: InferencePolicy
    runtime: InferenceRuntime
    assessment: AssessmentContext


def parse_inference_runtime(raw: Mapping[str, Any] | DictConfig) -> InferenceRuntime:
    data = _mapping(raw, "inference_runtime")
    _unknown(
        data,
        {"profile", "case_batch_size", "num_workers", "require_cuda", "timeout_seconds", "constraints"},
        "inference_runtime",
    )
    profile = str(data.get("profile", "native"))
    if profile not in SUPPORTED_PROFILES:
        raise InvalidInferenceRuntimeError(
            f"inference_runtime.profile must be one of {sorted(SUPPORTED_PROFILES)}."
        )
    case_batch_size = _integer(data.get("case_batch_size", 1), "case_batch_size")
    num_workers = _integer(data.get("num_workers", 0), "num_workers")
    if case_batch_size <= 0:
        raise InvalidInferenceRuntimeError("inference_runtime.case_batch_size must be > 0.")
    if num_workers < 0:
        raise InvalidInferenceRuntimeError("inference_runtime.num_workers must be >= 0.")
    require_cuda = _boolean(data.get("require_cuda", False), "require_cuda")
    timeout = data.get("timeout_seconds")
    if timeout is not None:
        timeout = _integer(timeout, "timeout_seconds")
        if timeout <= 0:
            raise InvalidInferenceRuntimeError("inference_runtime.timeout_seconds must be > 0.")

    constraints_data = _mapping(data.get("constraints"), "inference_runtime.constraints")
    _unknown(
        constraints_data,
        {
            "allowed_output_spaces",
            "allowed_precisions",
            "allow_ground_truth",
            "allow_threshold_sweep",
            "allow_intermediate_artifacts",
        },
        "inference_runtime.constraints",
    )
    spaces = _string_tuple(
        constraints_data.get("allowed_output_spaces", tuple(sorted(SUPPORTED_OUTPUT_SPACES))),
        "allowed_output_spaces",
    )
    if not spaces or set(spaces) - SUPPORTED_OUTPUT_SPACES:
        raise InvalidInferenceRuntimeError(
            "inference_runtime.constraints.allowed_output_spaces contains unsupported values."
        )
    precisions = _string_tuple(
        constraints_data.get("allowed_precisions", tuple(sorted(SUPPORTED_PRECISIONS))),
        "allowed_precisions",
    )
    if not precisions or set(precisions) - SUPPORTED_PRECISIONS:
        raise InvalidInferenceRuntimeError(
            "inference_runtime.constraints.allowed_precisions contains unsupported values."
        )
    constraints = RuntimeConstraints(
        allowed_output_spaces=spaces,
        allowed_precisions=precisions,
        allow_ground_truth=_boolean(
            constraints_data.get("allow_ground_truth", True), "allow_ground_truth"
        ),
        allow_threshold_sweep=_boolean(
            constraints_data.get("allow_threshold_sweep", True), "allow_threshold_sweep"
        ),
        allow_intermediate_artifacts=_boolean(
            constraints_data.get("allow_intermediate_artifacts", True),
            "allow_intermediate_artifacts",
        ),
    )
    return InferenceRuntime(
        profile=profile,
        case_batch_size=case_batch_size,
        num_workers=num_workers,
        require_cuda=require_cuda,
        timeout_seconds=timeout,
        constraints=constraints,
    )


def validate_runtime_compatibility(
    inference: InferencePolicy,
    runtime: InferenceRuntime,
    assessment: AssessmentContext | None = None,
) -> ValidatedInferenceRequest:
    assessment = assessment or AssessmentContext()
    if runtime.profile == "gc_submission":
        _validate_gc_hard_constraints(runtime)
    if inference.output_space not in runtime.constraints.allowed_output_spaces:
        required = (
            "native_input"
            if runtime.profile == "gc_submission"
            else str(runtime.constraints.allowed_output_spaces)
        )
        raise InvalidInferenceRuntimeError(
            f"Runtime profile {runtime.profile!r} does not allow output_space={inference.output_space!r}; "
            f"required/allowed output includes {required}."
        )
    if inference.precision not in runtime.constraints.allowed_precisions:
        raise InvalidInferenceRuntimeError(
            f"Runtime precision {inference.precision!r} is not allowed by profile "
            f"{runtime.profile!r}; allowed={runtime.constraints.allowed_precisions}."
        )
    if assessment.requires_ground_truth and not runtime.constraints.allow_ground_truth:
        raise InvalidInferenceRuntimeError(
            f"Runtime profile {runtime.profile!r} does not permit ground truth access."
        )
    if assessment.threshold_sweep and not runtime.constraints.allow_threshold_sweep:
        raise InvalidInferenceRuntimeError(
            f"Runtime profile {runtime.profile!r} does not permit a threshold sweep."
        )
    if (
        inference.artifacts.enabled
        and not runtime.constraints.allow_intermediate_artifacts
    ):
        raise InvalidInferenceRuntimeError(
            f"Runtime profile {runtime.profile!r} does not permit intermediate artifacts."
        )
    return ValidatedInferenceRequest(
        inference=inference,
        runtime=runtime,
        assessment=assessment,
    )


def _validate_gc_hard_constraints(runtime: InferenceRuntime) -> None:
    if runtime.case_batch_size != 1:
        raise InvalidInferenceRuntimeError("gc_submission requires case_batch_size == 1.")
    if runtime.num_workers != 0:
        raise InvalidInferenceRuntimeError("gc_submission requires num_workers == 0.")
    if not runtime.require_cuda:
        raise InvalidInferenceRuntimeError("gc_submission requires CUDA.")
    if runtime.timeout_seconds is None or runtime.timeout_seconds > 600:
        raise InvalidInferenceRuntimeError("gc_submission timeout_seconds must be <= 600.")
    constraints = runtime.constraints
    relaxed = (
        set(constraints.allowed_output_spaces) != {"native_input"}
        or not set(constraints.allowed_precisions).issubset({"fp16", "fp32"})
        or constraints.allow_ground_truth
        or constraints.allow_threshold_sweep
        or constraints.allow_intermediate_artifacts
    )
    if relaxed:
        raise InvalidInferenceRuntimeError(
            "gc_submission hard constraints cannot be relaxed by runtime configuration."
        )


def _mapping(raw: Any, path: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(raw, Mapping):
        raise InvalidInferenceRuntimeError(f"{path} must be a mapping.")
    return dict(raw)


def _unknown(data: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise InvalidInferenceRuntimeError(f"{path} contains unknown keys: {unknown}.")


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise InvalidInferenceRuntimeError(f"inference_runtime.{path} must be a boolean.")
    return value


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInferenceRuntimeError(f"inference_runtime.{path} must be an integer.")
    return int(value)


def _string_tuple(value: Any, path: str) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple, ListConfig)):
        raise InvalidInferenceRuntimeError(f"inference_runtime.constraints.{path} must be a sequence.")
    normalized = tuple(str(item) for item in value)
    if any(not item for item in normalized):
        raise InvalidInferenceRuntimeError(f"inference_runtime.constraints.{path} contains an empty value.")
    return normalized


__all__ = [
    "AssessmentContext",
    "InferenceRuntime",
    "InvalidInferenceRuntimeError",
    "RuntimeConstraints",
    "ValidatedInferenceRequest",
    "parse_inference_runtime",
    "validate_runtime_compatibility",
]
