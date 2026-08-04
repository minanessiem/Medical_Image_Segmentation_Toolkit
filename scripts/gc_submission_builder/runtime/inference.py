"""Grand Challenge orchestration around the shared inference package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import logging
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any, Mapping

import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from scripts.gc_submission_builder.release_manifest import verify_artifact_manifest
from scripts.gc_submission_builder.runtime.image_io import (
    NiftiInputInspection,
    inspect_nifti_input,
    materialize_prediction_outputs,
)
from scripts.gc_submission_builder.runtime.interfaces import (
    InterfaceDefinition,
    InterfaceManifest,
    load_interface_manifest,
    resolve_invocation,
    validate_dataset_bindings,
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


@dataclass(frozen=True)
class GcInvocationReport:
    """Non-identifying facts from one completed platform invocation."""

    interface_name: str
    input_inspections: Mapping[str, NiftiInputInspection]
    output_validations: Mapping[str, Mapping[str, Any]]
    elapsed_seconds: float
    peak_cuda_memory_bytes: int | None


@dataclass(frozen=True)
class GcCasePrediction:
    """Transport-resolved result before any production output materialization."""

    interface: InterfaceDefinition
    input_inspections: Mapping[str, NiftiInputInspection]
    result: PredictionResult
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

    def invoke(
        self,
        *,
        input_root: str | Path = "/input",
        output_root: str | Path = "/output",
    ) -> GcInvocationReport:
        """Resolve, preprocess, predict, restore, and write one 3D case."""

        prediction = self.predict(input_root=input_root)
        writer_started = perf_counter()
        outputs = materialize_prediction_outputs(
            prediction.result,
            output_root=output_root,
            bindings=prediction.interface.outputs,
        )
        elapsed = prediction.elapsed_seconds + (perf_counter() - writer_started)
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
            output_validations=MappingProxyType(
                {
                    slug: MappingProxyType(validation)
                    for slug, validation in safe_outputs.items()
                }
            ),
            elapsed_seconds=float(elapsed),
            peak_cuda_memory_bytes=prediction.peak_cuda_memory_bytes,
        )
        LOGGER.info(
            "GC invocation completed interface=%s dataset=%s runtime=%s "
            "input_contracts=%s outputs=%s elapsed_seconds=%.3f "
            "peak_cuda_memory_bytes=%s",
            report.interface_name,
            self.case_producer.adapter.dataset_id,
            self.runtime_profile.profile,
            {
                key: asdict(inspection)
                for key, inspection in report.input_inspections.items()
            },
            {
                slug: dict(validation)
                for slug, validation in report.output_validations.items()
            },
            report.elapsed_seconds,
            report.peak_cuda_memory_bytes,
        )
        return report

    def predict(
        self,
        *,
        input_root: str | Path = "/input",
    ) -> GcCasePrediction:
        """Run transport resolution and shared prediction without writing a socket."""

        started = perf_counter()
        invocation = resolve_invocation(self.interface_manifest, input_root)
        inspections = {
            key: inspect_nifti_input(path)
            for key, path in invocation.raw_modalities.items()
        }
        record = {
            "caseID": "gc-invocation",
            **{
                key: str(path)
                for key, path in invocation.raw_modalities.items()
            },
        }
        case = self.case_producer.preprocess(record)
        if not isinstance(case, PreprocessedCase):
            raise GcRuntimeError(
                "Grand Challenge preprocessing must return an unlabeled PreprocessedCase."
            )
        case_on_device = replace(case, image=case.image.to(self.device))

        peak_memory: int | None = None
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        result = predict_preprocessed_case(
            self.executor,
            case_on_device,
            progress_label="gc-invocation",
            show_window_progress=False,
        )
        if self.device.type == "cuda":
            peak_memory = int(torch.cuda.max_memory_allocated(self.device))

        elapsed = perf_counter() - started
        return GcCasePrediction(
            interface=invocation.interface,
            input_inspections=MappingProxyType(inspections),
            result=result,
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
) -> GcInferenceRuntime:
    """Load and validate every immutable service dependency before readiness."""

    root = Path(model_dir).expanduser().resolve()
    try:
        artifact_manifest = verify_artifact_manifest(root)
        saved_cfg = OmegaConf.load(root / "config.yaml")
        policy_payload = yaml.safe_load(
            (root / "inference_policy.yaml").read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise GcRuntimeError(f"Model artifact validation failed: {exc}") from exc
    if not isinstance(policy_payload, dict) or "defaults" in policy_payload:
        raise GcRuntimeError(
            "Archived inference_policy.yaml must be a standalone mapping."
        )

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
    except Exception as exc:
        raise GcRuntimeError(f"Inference/runtime policy validation failed: {exc}") from exc
    if resolved_policy.source != "explicit_top_level":
        raise GcRuntimeError(
            "The archived inference policy must be the explicit top-level policy."
        )

    resolved_device = _resolve_device(device, runtime_profile)
    dataset_id = _required_dataset_id(config)
    interface_manifest = load_interface_manifest(interface_manifest_path)
    try:
        model = load_model_strict(
            config,
            root / "weights.pth",
            device=resolved_device,
        )
        executor = build_model_probability_executor(backend=model, cfg=config)
        _validate_initial_capabilities(executor.predictor.capabilities)
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
        raise GcRuntimeError(
            f"Model/predictor/preprocessing initialization failed: {exc}"
        ) from exc

    LOGGER.info(
        "GC runtime ready dataset=%s adapter=%s runtime=%s device=%s policy_origin=%s "
        "artifact_config_sha256=%s artifact_policy_sha256=%s",
        dataset_id,
        case_producer.adapter.dataset_id,
        runtime_profile.profile,
        resolved_device,
        policy_origin,
        artifact_manifest["config_sha256"],
        artifact_manifest["inference_policy_sha256"],
    )
    return GcInferenceRuntime(
        config=config,
        executor=executor,
        case_producer=case_producer,
        interface_manifest=interface_manifest,
        runtime_profile=runtime_profile,
        artifact_manifest=MappingProxyType(dict(artifact_manifest)),
        device=resolved_device,
        inference_policy_origin=policy_origin,
    )


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


__all__ = [
    "DEFAULT_RUNTIME_PROFILE",
    "GcCasePrediction",
    "GcInferenceRuntime",
    "GcInvocationReport",
    "GcRuntimeError",
    "initialize_runtime",
]
