"""Shared, transport-independent inference contracts for native and container consumers."""

from src.inference.contracts import (
    InferenceError,
    InferenceInputError,
    InvalidInferencePolicyError,
    InvalidInferenceRuntimeError,
    InvalidPredictionError,
    PredictionResult,
    PredictorCapabilities,
    PreprocessedCase,
    ProbabilityPredictor,
    ResourceLimitError,
    SpatialGeometry,
    SpatialRestorationError,
    SpatialTrace,
    TimingResourceRecord,
    UnsupportedModelError,
)
from src.inference.policy import (
    InferencePolicy,
    ResolvedInferencePolicy,
    parse_inference_policy,
    resolve_inference_policy,
)
from src.inference.runtime import (
    AssessmentContext,
    InferenceRuntime,
    ValidatedInferenceRequest,
    parse_inference_runtime,
    validate_runtime_compatibility,
)

__all__ = [
    "AssessmentContext",
    "InferenceError",
    "InferenceInputError",
    "InferencePolicy",
    "InferenceRuntime",
    "InvalidInferencePolicyError",
    "InvalidInferenceRuntimeError",
    "InvalidPredictionError",
    "PredictionResult",
    "PredictorCapabilities",
    "PreprocessedCase",
    "ProbabilityPredictor",
    "ResolvedInferencePolicy",
    "ResourceLimitError",
    "SpatialGeometry",
    "SpatialRestorationError",
    "SpatialTrace",
    "TimingResourceRecord",
    "UnsupportedModelError",
    "ValidatedInferenceRequest",
    "parse_inference_policy",
    "parse_inference_runtime",
    "resolve_inference_policy",
    "validate_runtime_compatibility",
]
