"""Policy-driven model-space probability execution independent of consumers."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Iterable, Mapping, Optional, Tuple

import torch
from monai.data import MetaTensor
from omegaconf import DictConfig

from src.inference.contracts import (
    PredictionResult,
    PreprocessedCase,
    ProbabilityPredictor,
    UnsupportedModelError,
)
from src.inference.ensemble import MeanProbabilityAccumulator
from src.inference.policy import InferencePolicy, resolve_inference_policy
from src.inference.predictors import (
    build_probability_predictor,
    predict_probabilities,
    validate_predictor_capabilities,
)
from src.inference.sliding_window import predict_sliding_window_probabilities
from src.inference.spatial import (
    WORLD_COORDINATE_TOLERANCE_MM,
    restore_probability_to_native,
    threshold_probability,
)


@dataclass(frozen=True)
class ModelProbabilityExecutor:
    """Execute one prepared predictor according to a resolved shared policy."""

    predictor: ProbabilityPredictor
    policy: InferencePolicy
    policy_source: str

    def __post_init__(self) -> None:
        capabilities = validate_predictor_capabilities(self.predictor.capabilities)
        if self.policy.precision not in capabilities.supported_precisions:
            raise UnsupportedModelError(
                f"Predictor does not support requested precision {self.policy.precision!r}; "
                f"supported={capabilities.supported_precisions}."
            )

    def __call__(
        self,
        conditioned_image: torch.Tensor,
        progress_label: Optional[str] = None,
        show_window_progress: bool = True,
    ) -> torch.Tensor:
        """Return model/preprocessed-grid probabilities for the supplied tensor."""
        with torch.inference_mode(), _autocast_context(
            conditioned_image,
            self.policy.precision,
        ):
            if self.policy.sliding_window.enabled:
                return predict_sliding_window_probabilities(
                    self.predictor,
                    conditioned_image,
                    self.policy.sliding_window,
                    progress_label=progress_label,
                    show_window_progress=show_window_progress,
                )
            return predict_probabilities(self.predictor, conditioned_image)


@dataclass(frozen=True)
class EnsembleMemberExecutor:
    """One named, independently prepared member of a probability ensemble."""

    member_id: str
    executor: ModelProbabilityExecutor

    def __post_init__(self) -> None:
        if not isinstance(self.member_id, str) or not self.member_id.strip():
            raise UnsupportedModelError("Ensemble member_id must be a non-empty string.")


@dataclass(frozen=True)
class EnsembleProbabilityExecutor:
    """Execute and equally average every discovered model member."""

    members: Tuple[EnsembleMemberExecutor, ...]
    policy: InferencePolicy
    policy_source: str

    def __post_init__(self) -> None:
        if not self.members:
            raise UnsupportedModelError("An ensemble requires at least one model member.")
        member_ids = tuple(member.member_id for member in self.members)
        if len(set(member_ids)) != len(member_ids):
            raise UnsupportedModelError("Ensemble member identifiers must be unique.")
        if not self.policy.ensemble.enabled:
            raise UnsupportedModelError(
                "A multi-member executor requires inference.ensemble.enabled=true."
            )
        expected_capabilities = self.members[0].executor.predictor.capabilities
        for member in self.members:
            if member.executor.policy != self.policy:
                raise UnsupportedModelError(
                    f"Ensemble member {member.member_id!r} resolved a different inference policy."
                )
            if member.executor.predictor.capabilities != expected_capabilities:
                raise UnsupportedModelError(
                    "All ensemble members must expose identical predictor capabilities; "
                    f"member {member.member_id!r} is incompatible."
                )

    @property
    def member_ids(self) -> Tuple[str, ...]:
        return tuple(member.member_id for member in self.members)

    @property
    def predictor_capabilities(self):
        return self.members[0].executor.predictor.capabilities

    def __call__(
        self,
        conditioned_image: torch.Tensor,
        progress_label: Optional[str] = None,
        show_window_progress: bool = True,
    ) -> torch.Tensor:
        accumulator = MeanProbabilityAccumulator()
        for member in self.members:
            member_label = (
                f"{progress_label}:{member.member_id}"
                if progress_label
                else member.member_id
            )
            accumulator.add(
                member.executor(
                    conditioned_image,
                    progress_label=member_label,
                    show_window_progress=show_window_progress,
                )
            )
        return accumulator.mean()


def build_model_probability_executor(
    backend: Any,
    cfg: Mapping[str, Any] | DictConfig,
) -> ModelProbabilityExecutor:
    """Resolve policy and prepare the configured backend for shared execution."""

    resolved = resolve_inference_policy(cfg)
    predictor = build_probability_predictor(backend=backend, cfg=cfg)
    return ModelProbabilityExecutor(
        predictor=predictor,
        policy=resolved.policy,
        policy_source=resolved.source,
    )


def build_ensemble_probability_executor(
    members: Iterable[tuple[str, Any, Mapping[str, Any] | DictConfig]],
) -> EnsembleProbabilityExecutor:
    """Build an arbitrary-N equal-weight ensemble from discovered members."""

    prepared: list[EnsembleMemberExecutor] = []
    canonical_policy: InferencePolicy | None = None
    canonical_source: str | None = None
    for member_id, backend, cfg in members:
        executor = build_model_probability_executor(backend=backend, cfg=cfg)
        if canonical_policy is None:
            canonical_policy = executor.policy
            canonical_source = executor.policy_source
        prepared.append(
            EnsembleMemberExecutor(member_id=str(member_id), executor=executor)
        )
    if canonical_policy is None or canonical_source is None:
        raise UnsupportedModelError("No ensemble model members were supplied.")
    return EnsembleProbabilityExecutor(
        members=tuple(prepared),
        policy=canonical_policy,
        policy_source=canonical_source,
    )


def predict_preprocessed_case(
    executor: ModelProbabilityExecutor | EnsembleProbabilityExecutor,
    case: PreprocessedCase,
    *,
    progress_label: Optional[str] = None,
    show_window_progress: bool = True,
) -> PredictionResult:
    """Execute one prepared 3D case and return a truthfully declared output grid."""
    if not isinstance(executor, ModelProbabilityExecutor) and not (
        callable(executor)
        and hasattr(executor, "policy")
        and hasattr(executor, "policy_source")
    ):
        raise TypeError("executor must satisfy the ModelProbabilityExecutor interface.")
    if not isinstance(case, PreprocessedCase):
        raise TypeError("case must be a PreprocessedCase.")

    model_probability = executor(
        case.image,
        progress_label=progress_label or case.case_id,
        show_window_progress=show_window_progress,
    )
    output_space = executor.policy.output_space
    if output_space == "native_input":
        probability = restore_probability_to_native(
            model_probability,
            case.spatial_trace,
        )
        restoration_applied = case.spatial_trace.model != case.spatial_trace.original
    else:
        model_probability_tensor = (
            model_probability.as_tensor()
            if isinstance(model_probability, MetaTensor)
            else model_probability
        )
        probability = model_probability_tensor.detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        restoration_applied = False

    mask = threshold_probability(probability, executor.policy.decision.threshold)
    member_ids = tuple(getattr(executor, "member_ids", ()))
    return PredictionResult(
        probability=probability,
        mask=mask,
        output_space=output_space,
        spatial_trace=case.spatial_trace,
        native_reference=(
            case.native_metadata[case.reference_key]
            if case.reference_key is not None
            else None
        ),
        provenance={
            "case_id": case.case_id,
            "inference_policy_source": str(executor.policy_source),
            "output_space": output_space,
            "threshold": float(executor.policy.decision.threshold),
            "ensemble": {
                "enabled": bool(member_ids),
                "method": executor.policy.ensemble.method if member_ids else None,
                "member_count": len(member_ids) if member_ids else 1,
                "member_ids": list(member_ids),
            },
            "spatial_validation": {
                "status": "passed",
                "shape_matches_declared_geometry": True,
                "native_world_coordinates_validated": output_space == "native_input",
                "world_coordinate_tolerance_mm": (
                    WORLD_COORDINATE_TOLERANCE_MM
                    if output_space == "native_input"
                    else None
                ),
            },
            "spatial_restoration": {
                "applied": restoration_applied,
                "interpolation": "continuous_linear" if restoration_applied else None,
                "source_shape": list(case.spatial_trace.model.shape),
                "output_shape": list(
                    case.spatial_trace.original.shape
                    if output_space == "native_input"
                    else case.spatial_trace.model.shape
                ),
            },
        },
    )


def _autocast_context(
    conditioned_image: torch.Tensor,
    precision: str,
) -> ContextManager[Any]:
    if precision == "fp32":
        return nullcontext()
    if not torch.is_tensor(conditioned_image):
        return nullcontext()

    device_type = conditioned_image.device.type
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    if device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    if device_type == "cpu" and precision == "bf16":
        return torch.autocast(device_type="cpu", dtype=dtype)
    raise UnsupportedModelError(
        f"Precision {precision!r} is not supported for inference execution on "
        f"device type {device_type!r}. Use fp32 or a compatible CUDA runtime."
    )


__all__ = [
    "EnsembleMemberExecutor",
    "EnsembleProbabilityExecutor",
    "ModelProbabilityExecutor",
    "build_ensemble_probability_executor",
    "build_model_probability_executor",
    "predict_preprocessed_case",
]
