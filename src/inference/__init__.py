"""Shared, transport-independent inference contracts for native and container consumers."""

from src.inference.augmentation import TtaView, build_tta_views, tta_view_names
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
from src.inference.case_producer import (
    PreprocessedCaseProducer,
    build_case_producer,
)
from src.inference.preprocessing import preprocess_case
from src.inference.output import write_native_prediction_mask
from src.inference.pipeline import (
    ModelProbabilityExecutor,
    build_model_probability_executor,
    predict_preprocessed_case,
)
from src.inference.policy import (
    InferencePolicy,
    ResolvedInferencePolicy,
    TtaPolicy,
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
from src.inference.spatial import (
    restore_probability_to_native,
    threshold_probability,
    validate_output_geometry,
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
    "PreprocessedCaseProducer",
    "ProbabilityPredictor",
    "ResolvedInferencePolicy",
    "ResourceLimitError",
    "SpatialGeometry",
    "SpatialRestorationError",
    "SpatialTrace",
    "TimingResourceRecord",
    "TtaPolicy",
    "TtaView",
    "UnsupportedModelError",
    "ValidatedInferenceRequest",
    "build_model_probability_executor",
    "build_tta_views",
    "build_case_producer",
    "predict_preprocessed_case",
    "parse_inference_policy",
    "parse_inference_runtime",
    "resolve_inference_policy",
    "validate_runtime_compatibility",
    "preprocess_case",
    "restore_probability_to_native",
    "threshold_probability",
    "tta_view_names",
    "validate_output_geometry",
    "write_native_prediction_mask",
]
