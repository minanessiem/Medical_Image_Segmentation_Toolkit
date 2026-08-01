"""Shared, transport-independent inference contracts for native and container consumers."""

from src.inference.contracts import (
    InferenceError,
    InferenceInputError,
    InvalidInferencePolicyError,
    InvalidInferenceRuntimeError,
    InvalidPredictionError,
    LabeledPreprocessedCase,
    NativeImageMetadata,
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
from src.inference.preprocessing import preprocess_case
from src.inference.pipeline import (
    ModelProbabilityExecutor,
    build_model_probability_executor,
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
    "LabeledPreprocessedCase",
    "ModelProbabilityExecutor",
    "NativeImageMetadata",
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
    "build_model_probability_executor",
    "parse_inference_policy",
    "parse_inference_runtime",
    "resolve_inference_policy",
    "validate_runtime_compatibility",
    "preprocess_case",
]
