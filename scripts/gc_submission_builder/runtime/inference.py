"""Grand Challenge orchestration around the shared inference package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import logging
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Mapping

import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from scripts.gc_submission_builder.release_manifest import verify_artifact_manifest
from scripts.gc_submission_builder.runtime.image_io import (
    MedicalImageInspection,
    canonicalize_image_inputs,
    materialize_prediction_outputs,
)
from scripts.gc_submission_builder.runtime.interfaces import (
    InterfaceDefinition,
    InterfaceManifest,
    load_interface_manifest,
    resolve_invocation,
    validate_dataset_bindings,
)
from scripts.gc_submission_builder.runtime.observability import (
    emit_gc_event,
    normalized_timings,
    resource_summary,
    runtime_identity,
    stage_error,
)
from src.inference.case_producer import PreprocessedCaseProducer, build_case_producer
from src.inference.contracts import (
    PredictionResult,
    PreprocessedCase,
    PredictorCapabilities,
)
from src.inference.pipeline import (
    ModelProbabilityExecutor,
    build_model_probability_executor,
    predict_preprocessed_case,
)
from src.inference.policy import resolve_inference_policy
from src.inference.runtime import (
    AssessmentContext,
    InferenceRuntime,
    parse_inference_runtime,
    validate_runtime_compatibility,
)
from src.models.model_loader import load_model_strict


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_PROFILE = (
    PROJECT_ROOT / "configs" / "inference_runtime" / "gc_submission.yaml"
)


class GcRuntimeError(RuntimeError):
    """Raised when the container cannot initialize or complete one invocation."""


class GcRuntimeStageError(GcRuntimeError):
    """A stage-specific failure with bounded telemetry for the app boundary."""

    def __init__(
        self,
        *,
        stage: str,
        error_code: str,
        safe_detail: str,
        timings_seconds: Mapping[str, float],
        total_seconds: float,
        cause: BaseException,
    ) -> None:
        diagnostic = stage_error(
            cause,
            stage=stage,
            error_code=error_code,
            safe_detail=safe_detail,
            timings_seconds=timings_seconds,
            total_seconds=total_seconds,
        )
        super().__init__(str(diagnostic))
        self.stage = diagnostic.stage
        self.error_code = diagnostic.error_code
        self.safe_detail = diagnostic.safe_detail
        self.timings_seconds = diagnostic.timings_seconds
        self.total_seconds = diagnostic.total_seconds
        self.locations = diagnostic.locations


@dataclass(frozen=True)
class GcInvocationReport:
    """Non-identifying facts from one completed platform invocation."""

    interface_name: str
    input_inspections: Mapping[str, MedicalImageInspection]
    input_normalizations: Mapping[str, Mapping[str, Any]]
    output_validations: Mapping[str, Mapping[str, Any]]
    timings_seconds: Mapping[str, float]
    resources: Mapping[str, int | None]
    elapsed_seconds: float
    peak_cuda_memory_bytes: int | None


@dataclass(frozen=True)
class GcCasePrediction:
    """Transport-resolved result before any production output materialization."""

    interface: InterfaceDefinition
    input_inspections: Mapping[str, MedicalImageInspection]
    input_normalizations: Mapping[str, Mapping[str, Any]]
    result: PredictionResult
    timings_seconds: Mapping[str, float]
    elapsed_seconds: float
    peak_cuda_memory_bytes: int | None


@dataclass(frozen=True)
class GcInferenceRuntime:
    """Initialized, single-model Grand Challenge inference service."""

    config: DictConfig
    executor: ModelProbabilityExecutor
    case_producer: PreprocessedCaseProducer
    interface_manifest: InterfaceManifest
    runtime_profile: InferenceRuntime
    artifact_manifest: Mapping[str, Any]
    device: torch.device
    inference_policy_origin: str = "artifact"
    startup_timings_seconds: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    clock: Callable[[], float] = field(default=perf_counter, repr=False, compare=False)

    def invoke(
        self,
        *,
        input_root: str | Path = "/input",
        output_root: str | Path = "/output",
        scratch_root: str | Path = "/tmp",
    ) -> GcInvocationReport:
        """Resolve, preprocess, predict, restore, and write one 3D case."""

        invoke_started = self.clock()
        try:
            prediction = self.predict(
                input_root=input_root,
                scratch_root=scratch_root,
            )
        except GcRuntimeStageError:
            raise
        except Exception as exc:
            raise GcRuntimeStageError(
                stage="prediction_pipeline",
                error_code="PREDICTION_PIPELINE_FAILED",
                safe_detail="Shared prediction did not complete.",
                timings_seconds={},
                total_seconds=self.clock() - invoke_started,
                cause=exc,
            ) from exc

        timings = dict(prediction.timings_seconds)
        writer_started = self.clock()
        try:
            outputs = materialize_prediction_outputs(
                prediction.result,
                output_root=output_root,
                bindings=prediction.interface.outputs,
            )
            _validate_outputs_against_source(
                outputs,
                source_inspections=prediction.input_inspections,
                reference_key=getattr(self.case_producer, "reference_key", None),
            )
        except Exception as exc:
            timings["output_materialization_validation_seconds"] = max(
                0.0, self.clock() - writer_started
            )
            raise GcRuntimeStageError(
                stage="output_materialization_validation",
                error_code="OUTPUT_MATERIALIZATION_VALIDATION_FAILED",
                safe_detail="The complete declared output set could not be validated.",
                timings_seconds=timings,
                total_seconds=self.clock() - invoke_started,
                cause=exc,
            ) from exc
        timings["output_materialization_validation_seconds"] = max(
            0.0, self.clock() - writer_started
        )
        elapsed = max(0.0, self.clock() - invoke_started)
        timings["invoke_total_seconds"] = elapsed
        timings = dict(normalized_timings(timings))
        safe_outputs = {
            slug: {
                key: value
                for key, value in validation.items()
                if key != "path"
            }
            for slug, validation in outputs.items()
        }
        report = GcInvocationReport(
            interface_name=prediction.interface.name,
            input_inspections=prediction.input_inspections,
            input_normalizations=prediction.input_normalizations,
            output_validations=MappingProxyType(
                {
                    slug: MappingProxyType(validation)
                    for slug, validation in safe_outputs.items()
                }
            ),
            timings_seconds=MappingProxyType(timings),
            resources=resource_summary(device=self.device, scratch_root=scratch_root),
            elapsed_seconds=float(elapsed),
            peak_cuda_memory_bytes=prediction.peak_cuda_memory_bytes,
        )
        emit_gc_event(
            LOGGER,
            "invocation_completed",
            outcome="succeeded",
            interface=report.interface_name,
            dataset=self.case_producer.adapter.dataset_id,
            runtime_profile=self.runtime_profile.profile,
            input_contracts={
                key: asdict(inspection) for key, inspection in report.input_inspections.items()
            },
            input_normalizations={
                key: dict(value) for key, value in report.input_normalizations.items()
            },
            outputs={
                slug: dict(validation) for slug, validation in report.output_validations.items()
            },
            timings_seconds=dict(report.timings_seconds),
            resources=dict(report.resources),
        )
        return report

    def predict(
        self,
        *,
        input_root: str | Path = "/input",
        scratch_root: str | Path = "/tmp",
    ) -> GcCasePrediction:
        """Run transport resolution and shared prediction without writing a socket."""

        started = self.clock()
        timings: dict[str, float] = {}
        active_stage = "interface_resolution"
        stage_started = self.clock()
        try:
            invocation = resolve_invocation(
                self.interface_manifest,
                input_root,
                clock=self.clock,
            )
            timings.update(invocation.resolution_timings_seconds)
        except Exception as exc:
            timings["interface_resolution_seconds"] = max(
                0.0, self.clock() - stage_started
            )
            raise GcRuntimeStageError(
                stage=active_stage,
                error_code="INTERFACE_RESOLUTION_FAILED",
                safe_detail="The platform socket set or image inventory did not satisfy the interface manifest.",
                timings_seconds=timings,
                total_seconds=self.clock() - started,
                cause=exc,
            ) from exc

        active_stage = "input_canonicalization"
        stage_started = self.clock()
        try:
            with canonicalize_image_inputs(
                invocation.image_inputs,
                scratch_root=scratch_root,
            ) as canonicalized:
                timings["input_canonicalization_seconds"] = max(
                    0.0, self.clock() - stage_started
                )
                inspections = {
                    key: value.source_inspection
                    for key, value in canonicalized.inputs.items()
                }
                normalizations = {
                    key: MappingProxyType(
                        {
                            "source_format": value.source_format,
                            "canonical_format": "nii_gz",
                            "converted": value.converted,
                            "source_size_bytes": value.source_size_bytes,
                            "canonical_size_bytes": value.canonical_size_bytes,
                            "observed_format_counts": dict(
                                invocation.image_inputs[key].observed_format_counts
                            ),
                            "other_regular_file_count": invocation.image_inputs[
                                key
                            ].other_regular_file_count,
                        }
                    )
                    for key, value in canonicalized.inputs.items()
                }
                emit_gc_event(
                    LOGGER,
                    "input_canonicalized",
                    interface=invocation.interface.name,
                    inputs={
                        key: {
                            "source_format": value.source_format,
                            "source_size_bytes": value.source_size_bytes,
                            "canonical_size_bytes": value.canonical_size_bytes,
                            "converted": value.converted,
                            "source_geometry": asdict(value.source_inspection),
                            "canonical_geometry": asdict(value.canonical_inspection),
                        }
                        for key, value in canonicalized.inputs.items()
                    },
                    metadata_schema={
                        slug: {
                            key: {
                                "type": type(value).__name__,
                                "is_null": value is None,
                            }
                            for key, value in technical.items()
                        }
                        for slug, technical in invocation.technical_inputs.items()
                    },
                    scratch_free_before_bytes=canonicalized.scratch_free_before_bytes,
                    scratch_free_after_bytes=canonicalized.scratch_free_after_bytes,
                    timing_seconds=timings["input_canonicalization_seconds"],
                )
                record = {
                    "caseID": "gc-invocation",
                    **{
                        key: str(path)
                        for key, path in canonicalized.canonical_modalities.items()
                    },
                }

                active_stage = "preprocessing"
                stage_started = self.clock()
                case = self.case_producer.preprocess(record)
                timings["preprocessing_seconds"] = max(
                    0.0, self.clock() - stage_started
                )
                if not isinstance(case, PreprocessedCase):
                    raise GcRuntimeError(
                        "Grand Challenge preprocessing must return an unlabeled PreprocessedCase."
                    )

                active_stage = "device_transfer"
                stage_started = self.clock()
                case_on_device = replace(case, image=case.image.to(self.device))
                timings["device_transfer_seconds"] = max(
                    0.0, self.clock() - stage_started
                )

                active_stage = "prediction_pipeline"
                stage_started = self.clock()
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                result = predict_preprocessed_case(
                    self.executor,
                    case_on_device,
                    progress_label="gc-invocation",
                    show_window_progress=False,
                )
                timings["prediction_pipeline_seconds"] = max(
                    0.0, self.clock() - stage_started
                )
                peak_memory = (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda"
                    else None
                )
        except GcRuntimeStageError:
            raise
        except Exception as exc:
            timings[f"{active_stage}_seconds"] = max(
                0.0, self.clock() - stage_started
            )
            code = f"{active_stage.upper()}_FAILED"
            raise GcRuntimeStageError(
                stage=active_stage,
                error_code=code,
                safe_detail=f"Grand Challenge {active_stage.replace('_', ' ')} did not complete.",
                timings_seconds=timings,
                total_seconds=self.clock() - started,
                cause=exc,
            ) from exc

        elapsed = max(0.0, self.clock() - started)
        return GcCasePrediction(
            interface=invocation.interface,
            input_inspections=MappingProxyType(inspections),
            input_normalizations=MappingProxyType(normalizations),
            result=result,
            timings_seconds=normalized_timings(timings),
            elapsed_seconds=float(elapsed),
            peak_cuda_memory_bytes=peak_memory,
        )


def initialize_runtime(
    *,
    model_dir: str | Path = "/opt/ml/model",
    interface_manifest_path: str | Path,
    runtime_profile_path: str | Path = DEFAULT_RUNTIME_PROFILE,
    device: str | torch.device | None = None,
    output_space_override: str | None = None,
    clock: Callable[[], float] = perf_counter,
) -> GcInferenceRuntime:
    """Load and validate every immutable service dependency before readiness."""

    startup_started = clock()
    timings: dict[str, float] = {}
    root = Path(model_dir).expanduser().resolve()
    stage_started = clock()
    try:
        artifact_manifest = verify_artifact_manifest(root)
        saved_cfg = OmegaConf.load(root / "config.yaml")
        policy_payload = yaml.safe_load(
            (root / "inference_policy.yaml").read_text(encoding="utf-8")
        )
    except Exception as exc:
        timings["artifact_validation_seconds"] = max(0.0, clock() - stage_started)
        raise GcRuntimeStageError(
            stage="artifact_validation",
            error_code="ARTIFACT_VALIDATION_FAILED",
            safe_detail="The mounted model artifact failed integrity or schema validation.",
            timings_seconds=timings,
            total_seconds=clock() - startup_started,
            cause=exc,
        ) from exc
    if not isinstance(policy_payload, dict) or "defaults" in policy_payload:
        exc = GcRuntimeError("Archived inference policy must be a standalone mapping.")
        timings["artifact_validation_seconds"] = max(0.0, clock() - stage_started)
        raise GcRuntimeStageError(
            stage="artifact_validation",
            error_code="ARTIFACT_POLICY_SCHEMA_INVALID",
            safe_detail=str(exc),
            timings_seconds=timings,
            total_seconds=clock() - startup_started,
            cause=exc,
        ) from exc
    timings["artifact_validation_seconds"] = max(0.0, clock() - stage_started)

    stage_started = clock()
    try:
        runtime_profile = parse_inference_runtime(
            OmegaConf.load(Path(runtime_profile_path).expanduser().resolve())
        )
        policy_origin = "artifact"
        active_policy_payload = dict(policy_payload)
        if output_space_override is not None:
            if runtime_profile.profile != "gc_container_test":
                raise GcRuntimeError(
                    "An output-space override is only permitted with the "
                    "gc_container_test runtime profile; production submission policy "
                    "must remain the verified artifact policy."
                )
            active_policy_payload["output_space"] = output_space_override
            policy_origin = "diagnostic_output_space_override"

        config = OmegaConf.create(OmegaConf.to_container(saved_cfg, resolve=False))
        OmegaConf.set_struct(config, False)
        OmegaConf.update(config, "inference", active_policy_payload, merge=False)
        resolved_policy = resolve_inference_policy(config)
        validate_runtime_compatibility(
            resolved_policy.policy,
            runtime_profile,
            AssessmentContext(requires_ground_truth=False, threshold_sweep=False),
        )
        if resolved_policy.source != "explicit_top_level":
            raise GcRuntimeError(
                "The archived inference policy must be the explicit top-level policy."
            )
        resolved_device = _resolve_device(device, runtime_profile)
        dataset_id = _required_dataset_id(config)
    except Exception as exc:
        timings["policy_runtime_resolution_seconds"] = max(
            0.0, clock() - stage_started
        )
        raise GcRuntimeStageError(
            stage="policy_runtime_resolution",
            error_code="POLICY_RUNTIME_RESOLUTION_FAILED",
            safe_detail=str(exc),
            timings_seconds=timings,
            total_seconds=clock() - startup_started,
            cause=exc,
        ) from exc
    timings["policy_runtime_resolution_seconds"] = max(
        0.0, clock() - stage_started
    )

    stage_started = clock()
    try:
        model = load_model_strict(
            config,
            root / "weights.pth",
            device=resolved_device,
        )
        executor = build_model_probability_executor(backend=model, cfg=config)
        _validate_initial_capabilities(executor.predictor.capabilities)
    except Exception as exc:
        timings["model_construction_checkpoint_load_seconds"] = max(
            0.0, clock() - stage_started
        )
        raise GcRuntimeStageError(
            stage="model_construction_checkpoint_load",
            error_code="MODEL_CONSTRUCTION_CHECKPOINT_LOAD_FAILED",
            safe_detail="The configured model could not be constructed and strictly loaded.",
            timings_seconds=timings,
            total_seconds=clock() - startup_started,
            cause=exc,
        ) from exc
    timings["model_construction_checkpoint_load_seconds"] = max(
        0.0, clock() - stage_started
    )

    stage_started = clock()
    try:
        interface_manifest = load_interface_manifest(interface_manifest_path)
        case_producer = build_case_producer(
            dataset_id=dataset_id,
            dataset_cfg=config.dataset,
            load_labels=False,
        )
        validate_dataset_bindings(
            interface_manifest,
            required_raw_keys=case_producer.required_raw_keys,
        )
    except Exception as exc:
        timings["adapter_manifest_initialization_seconds"] = max(
            0.0, clock() - stage_started
        )
        raise GcRuntimeStageError(
            stage="adapter_manifest_initialization",
            error_code="ADAPTER_MANIFEST_INITIALIZATION_FAILED",
            safe_detail=str(exc),
            timings_seconds=timings,
            total_seconds=clock() - startup_started,
            cause=exc,
        ) from exc
    timings["adapter_manifest_initialization_seconds"] = max(
        0.0, clock() - stage_started
    )
    timings["startup_total_seconds"] = max(0.0, clock() - startup_started)
    timings = dict(normalized_timings(timings))
    runtime = GcInferenceRuntime(
        config=config,
        executor=executor,
        case_producer=case_producer,
        interface_manifest=interface_manifest,
        runtime_profile=runtime_profile,
        artifact_manifest=MappingProxyType(dict(artifact_manifest)),
        device=resolved_device,
        inference_policy_origin=policy_origin,
        startup_timings_seconds=MappingProxyType(timings),
        clock=clock,
    )
    emit_gc_event(
        LOGGER,
        "startup_completed",
        outcome="succeeded",
        identity=dict(runtime_identity()),
        artifact={
            "config_sha256": artifact_manifest["config_sha256"],
            "inference_policy_sha256": artifact_manifest[
                "inference_policy_sha256"
            ],
        },
        dataset=dataset_id,
        adapter=case_producer.adapter.dataset_id,
        required_raw_keys=list(case_producer.required_raw_keys),
        runtime_profile=runtime_profile.profile,
        device=str(resolved_device),
        policy_origin=policy_origin,
        output_space=resolved_policy.policy.output_space,
        interface_bindings=[
            {
                "name": interface.name,
                "inputs": [
                    {
                        "slug": binding.slug,
                        "dataset_key": binding.dataset_key,
                        "accepted_formats": list(binding.accepted_formats),
                    }
                    for binding in interface.inputs
                ],
                "outputs": [
                    {
                        "slug": binding.slug,
                        "result_key": binding.result_key,
                        "file_type": binding.file_type,
                    }
                    for binding in interface.outputs
                ],
            }
            for interface in interface_manifest.interfaces
        ],
        model=_model_summary(model),
        timings_seconds=timings,
        resources=dict(resource_summary(device=resolved_device)),
    )
    return runtime


def _resolve_device(
    requested: str | torch.device | None,
    runtime_profile: InferenceRuntime,
) -> torch.device:
    device = torch.device(
        requested
        if requested is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise GcRuntimeError("CUDA was selected but is not available in the container.")
    if runtime_profile.require_cuda and device.type != "cuda":
        raise GcRuntimeError(
            f"Runtime profile {runtime_profile.profile!r} requires a CUDA device."
        )
    return device


def _required_dataset_id(config: DictConfig) -> str:
    value = OmegaConf.select(config, "dataset.id", default=None)
    if not isinstance(value, str) or not value.strip():
        raise GcRuntimeError(
            "Saved model config must declare dataset.id for registered preprocessing."
        )
    return value.strip().lower()


def _validate_initial_capabilities(capabilities: PredictorCapabilities) -> None:
    if not isinstance(capabilities, PredictorCapabilities):
        raise GcRuntimeError("Prepared predictor did not expose PredictorCapabilities.")
    if capabilities.spatial_dims != 3:
        raise GcRuntimeError(
            "The initial Grand Challenge runtime supports 3D predictors only."
        )


def _model_summary(model: Any) -> Mapping[str, Any]:
    parameters_method = getattr(model, "parameters", None)
    if not callable(parameters_method):
        return {"parameter_count": None, "dtypes": [], "devices": []}
    parameters = list(parameters_method())
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "dtypes": sorted({str(parameter.dtype) for parameter in parameters}),
        "devices": sorted({str(parameter.device) for parameter in parameters}),
    }


def _validate_outputs_against_source(
    outputs: Mapping[str, Mapping[str, Any]],
    *,
    source_inspections: Mapping[str, MedicalImageInspection],
    reference_key: str | None,
) -> None:
    if reference_key is None:
        if len(source_inspections) != 1:
            raise GcRuntimeError(
                "Output validation requires an explicit native reference key for "
                "multi-input interfaces."
            )
        reference_key = next(iter(source_inspections))
    try:
        reference = source_inspections[reference_key]
    except KeyError as exc:
        raise GcRuntimeError(
            "The preprocessing native reference key is absent from resolved inputs."
        ) from exc
    for validation in outputs.values():
        if validation.get("file_type") != "mha":
            continue
        if tuple(validation.get("shape", ())) != reference.shape:
            raise GcRuntimeError(
                "Materialized MHA output shape does not match the platform input."
            )
        for name in ("spacing", "origin", "direction"):
            observed = validation.get(name)
            expected = getattr(reference, name)
            if observed is None or not torch.allclose(
                torch.as_tensor(observed, dtype=torch.float64),
                torch.as_tensor(expected, dtype=torch.float64),
                rtol=0,
                atol=1e-5,
            ):
                raise GcRuntimeError(
                    f"Materialized MHA output {name} does not match the platform input."
                )


__all__ = [
    "DEFAULT_RUNTIME_PROFILE",
    "GcCasePrediction",
    "GcInferenceRuntime",
    "GcInvocationReport",
    "GcRuntimeError",
    "GcRuntimeStageError",
    "initialize_runtime",
]
