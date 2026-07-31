"""Strict parsing and legacy resolution for the shared inference policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from src.inference.contracts import InvalidInferencePolicyError, SUPPORTED_OUTPUT_SPACES


SUPPORTED_PRECISIONS = frozenset({"fp16", "fp32", "bf16"})
SUPPORTED_BLEND_MODES = frozenset({"constant", "gaussian"})
SUPPORTED_PADDING_MODES = frozenset({"constant", "reflect", "replicate", "circular"})


@dataclass(frozen=True)
class SlidingWindowPolicy:
    roi_size: Tuple[int, ...]
    enabled: bool = True
    sw_batch_size: int = 1
    overlap: float = 0.5
    blend_mode: str = "gaussian"
    padding_mode: str = "constant"


@dataclass(frozen=True)
class TtaPolicy:
    enabled: bool = False


@dataclass(frozen=True)
class EnsemblePolicy:
    enabled: bool = False


@dataclass(frozen=True)
class DecisionPolicy:
    threshold: float = 0.5


@dataclass(frozen=True)
class PostprocessingPolicy:
    enabled: bool = False


@dataclass(frozen=True)
class ArtifactPolicy:
    enabled: bool = False


@dataclass(frozen=True)
class InferencePolicy:
    sliding_window: SlidingWindowPolicy
    output_space: str = "model_preprocessed"
    precision: str = "fp32"
    tta: TtaPolicy = TtaPolicy()
    ensemble: EnsemblePolicy = EnsemblePolicy()
    decision: DecisionPolicy = DecisionPolicy()
    postprocessing: PostprocessingPolicy = PostprocessingPolicy()
    artifacts: ArtifactPolicy = ArtifactPolicy()


@dataclass(frozen=True)
class ResolvedInferencePolicy:
    policy: InferencePolicy
    source: str


def parse_inference_policy(
    raw: Mapping[str, Any] | DictConfig | None,
    *,
    model_roi: Any,
) -> InferencePolicy:
    expected_roi = _parse_roi(model_roi, field_name="model-owned inference ROI")
    data = _plain_mapping(raw, "inference")
    _reject_unknown(
        data,
        {
            "output_space",
            "precision",
            "sliding_window",
            "tta",
            "ensemble",
            "decision",
            "postprocessing",
            "artifacts",
        },
        "inference",
    )

    output_space = str(data.get("output_space", "model_preprocessed"))
    if output_space not in SUPPORTED_OUTPUT_SPACES:
        raise InvalidInferencePolicyError(
            "inference.output_space must be one of "
            f"{sorted(SUPPORTED_OUTPUT_SPACES)}, got {output_space!r}."
        )
    precision = str(data.get("precision", "fp32")).lower()
    if precision not in SUPPORTED_PRECISIONS:
        raise InvalidInferencePolicyError(
            f"inference.precision must be one of {sorted(SUPPORTED_PRECISIONS)}, got {precision!r}."
        )

    policy = InferencePolicy(
        output_space=output_space,
        precision=precision,
        sliding_window=_parse_sliding_window(
            data.get("sliding_window"),
            model_roi=expected_roi,
        ),
        tta=_parse_tta(data.get("tta")),
        ensemble=_parse_ensemble(data.get("ensemble")),
        decision=_parse_decision(data.get("decision")),
        postprocessing=_parse_postprocessing(data.get("postprocessing")),
        artifacts=_parse_artifacts(data.get("artifacts")),
    )
    return policy


def resolve_inference_policy(cfg: Mapping[str, Any] | DictConfig) -> ResolvedInferencePolicy:
    root = OmegaConf.create(cfg) if not OmegaConf.is_config(cfg) else cfg
    model_roi = _resolve_model_roi(root)
    explicit = OmegaConf.select(root, "inference", default=None)
    if explicit is not None:
        policy = parse_inference_policy(explicit, model_roi=model_roi)
        return ResolvedInferencePolicy(
            policy=policy,
            source="explicit_top_level",
        )

    legacy = OmegaConf.select(root, "validation.inference", default=None)
    if legacy is not None:
        policy = _translate_legacy_policy(root, legacy, model_roi=model_roi)
        return ResolvedInferencePolicy(
            policy=policy,
            source="legacy_validation",
        )
    return ResolvedInferencePolicy(
        policy=parse_inference_policy({}, model_roi=model_roi),
        source="model_contract",
    )


def _translate_legacy_policy(
    root: Any,
    legacy: Any,
    *,
    model_roi: Tuple[int, ...],
) -> InferencePolicy:
    legacy_data = _plain_mapping(legacy, "validation.inference")
    _reject_unknown(legacy_data, {"mode", "sliding_window"}, "validation.inference")
    mode = str(legacy_data.get("mode", "auto")).lower()
    if mode not in {"auto", "direct", "sliding_window"}:
        raise InvalidInferencePolicyError(
            "validation.inference.mode must be one of auto, direct, sliding_window, "
            f"got {mode!r}."
        )
    legacy_sw = _plain_mapping(
        legacy_data.get("sliding_window"), "validation.inference.sliding_window"
    )
    _reject_unknown(
        legacy_sw,
        {
            "enabled_loader_modes",
            "roi_size",
            "sw_batch_size",
            "overlap",
            "blend_mode",
            "padding_mode",
        },
        "validation.inference.sliding_window",
    )
    if mode == "auto":
        enabled_modes = legacy_sw.get(
            "enabled_loader_modes", ("full_volumes_3d", "random_patches_3d")
        )
        if not isinstance(enabled_modes, (list, tuple, ListConfig)):
            raise InvalidInferencePolicyError(
                "validation.inference.sliding_window.enabled_loader_modes must be a sequence."
            )
        loader_mode = str(OmegaConf.select(root, "data_mode.loader_mode", default="") or "")
        enabled = loader_mode in {str(value) for value in enabled_modes}
    else:
        enabled = mode == "sliding_window"

    raw_shared: dict[str, Any] = {"sliding_window": {"enabled": enabled}}
    shared_sw = raw_shared["sliding_window"]
    for key in ("sw_batch_size", "overlap", "blend_mode", "padding_mode"):
        if key in legacy_sw:
            shared_sw[key] = legacy_sw[key]
    if legacy_sw.get("roi_size") is not None:
        legacy_roi = _parse_roi(
            legacy_sw["roi_size"],
            field_name="validation.inference.sliding_window.roi_size",
        )
        if legacy_roi != model_roi:
            raise InvalidInferencePolicyError(
                f"Historical validation inference requested roi_size={legacy_roi}, but "
                f"the saved model/data preprocessing contract requires {model_roi}."
            )
    return parse_inference_policy(raw_shared, model_roi=model_roi)


def _parse_sliding_window(
    raw: Any,
    *,
    model_roi: Tuple[int, ...],
) -> SlidingWindowPolicy:
    data = _plain_mapping(raw, "inference.sliding_window")
    _reject_unknown(
        data,
        {"enabled", "sw_batch_size", "overlap", "blend_mode", "padding_mode"},
        "inference.sliding_window",
    )
    sw_batch_size = _integer(data.get("sw_batch_size", 1), "inference.sliding_window.sw_batch_size")
    if sw_batch_size <= 0:
        raise InvalidInferencePolicyError("inference.sliding_window.sw_batch_size must be > 0.")
    overlap = _number(data.get("overlap", 0.5), "inference.sliding_window.overlap")
    if not 0.0 <= overlap < 1.0:
        raise InvalidInferencePolicyError(
            "inference.sliding_window.overlap must satisfy 0 <= overlap < 1."
        )
    blend_mode = str(data.get("blend_mode", "gaussian")).lower()
    if blend_mode not in SUPPORTED_BLEND_MODES:
        raise InvalidInferencePolicyError(
            f"inference.sliding_window.blend_mode must be one of {sorted(SUPPORTED_BLEND_MODES)}."
        )
    padding_mode = str(data.get("padding_mode", "constant")).lower()
    if padding_mode not in SUPPORTED_PADDING_MODES:
        raise InvalidInferencePolicyError(
            f"inference.sliding_window.padding_mode must be one of {sorted(SUPPORTED_PADDING_MODES)}."
        )
    enabled = _boolean(data.get("enabled", True), "inference.sliding_window.enabled")
    return SlidingWindowPolicy(
        enabled=enabled,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        blend_mode=blend_mode,
        padding_mode=padding_mode,
        roi_size=model_roi,
    )


def _parse_tta(raw: Any) -> TtaPolicy:
    data = _plain_mapping(raw, "inference.tta")
    _reject_unknown(data, {"enabled"}, "inference.tta")
    enabled = _disabled_feature(data, "inference.tta")
    return TtaPolicy(enabled=enabled)


def _parse_ensemble(raw: Any) -> EnsemblePolicy:
    data = _plain_mapping(raw, "inference.ensemble")
    _reject_unknown(data, {"enabled"}, "inference.ensemble")
    enabled = _disabled_feature(data, "inference.ensemble")
    return EnsemblePolicy(enabled=enabled)


def _parse_decision(raw: Any) -> DecisionPolicy:
    data = _plain_mapping(raw, "inference.decision")
    _reject_unknown(data, {"threshold"}, "inference.decision")
    threshold = _number(data.get("threshold", 0.5), "inference.decision.threshold")
    if not 0.0 <= threshold <= 1.0:
        raise InvalidInferencePolicyError("inference.decision.threshold must be within [0, 1].")
    return DecisionPolicy(threshold=threshold)


def _parse_postprocessing(raw: Any) -> PostprocessingPolicy:
    data = _plain_mapping(raw, "inference.postprocessing")
    _reject_unknown(data, {"enabled"}, "inference.postprocessing")
    enabled = _disabled_feature(data, "inference.postprocessing")
    return PostprocessingPolicy(enabled=enabled)


def _parse_artifacts(raw: Any) -> ArtifactPolicy:
    data = _plain_mapping(raw, "inference.artifacts")
    _reject_unknown(data, {"enabled"}, "inference.artifacts")
    enabled = _disabled_feature(data, "inference.artifacts")
    return ArtifactPolicy(enabled=enabled)


def _parse_roi(value: Any, *, field_name: str) -> Tuple[int, ...]:
    if value is None:
        raise InvalidInferencePolicyError(
            f"{field_name} must not be null."
        )
    if not isinstance(value, (list, tuple, ListConfig)):
        raise InvalidInferencePolicyError(
            f"{field_name} must be a sequence."
        )
    roi = tuple(_integer(item, field_name) for item in value)
    if len(roi) not in (2, 3) or any(item <= 0 for item in roi):
        raise InvalidInferencePolicyError(
            f"{field_name} must contain two or three positive integers."
        )
    return roi


def _resolve_model_roi(root: Any) -> Tuple[int, ...]:
    try:
        spatial_dims_value = (
            OmegaConf.select(root, "model.spatial_dims", default=None)
            or OmegaConf.select(root, "data_mode.dim", default=None)
        )
    except OmegaConfBaseException as exc:
        raise InvalidInferencePolicyError(
            f"Could not resolve model dimensionality for inference ROI: {exc}"
        ) from exc
    spatial_dims = _parse_spatial_dims_token(spatial_dims_value)
    roi_key = "volume_3d" if spatial_dims == 3 else "slice_2d"
    field_name = f"dataset.preprocessing_configs.roi.{roi_key}"
    try:
        value = OmegaConf.select(root, field_name, default=None)
    except OmegaConfBaseException as exc:
        raise InvalidInferencePolicyError(
            f"Could not resolve model-owned inference ROI from {field_name}: {exc}"
        ) from exc
    if value is None:
        raise InvalidInferencePolicyError(
            f"Missing model-owned inference ROI at {field_name}."
        )
    roi = _parse_roi(value, field_name=field_name)
    if len(roi) != spatial_dims:
        raise InvalidInferencePolicyError(
            f"{field_name} must contain {spatial_dims} values for a {spatial_dims}D model, "
            f"got {roi}."
        )
    return roi


def _parse_spatial_dims_token(value: Any) -> int:
    if value is None:
        raise InvalidInferencePolicyError(
            "Missing model.spatial_dims/data_mode.dim required to resolve inference ROI."
        )
    token = str(value).strip().lower()
    if token.endswith("d"):
        token = token[:-1]
    if token not in {"2", "3"}:
        raise InvalidInferencePolicyError(
            f"Unsupported model spatial dimensionality {value!r}; expected 2, 3, '2d', or '3d'."
        )
    return int(token)


def _disabled_feature(data: Mapping[str, Any], path: str) -> bool:
    enabled = _boolean(data.get("enabled", False), f"{path}.enabled")
    if enabled:
        raise InvalidInferencePolicyError(
            f"{path} is not implemented yet and must remain enabled=false."
        )
    return enabled


def _plain_mapping(raw: Any, path: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if OmegaConf.is_config(raw):
        try:
            raw = OmegaConf.to_container(raw, resolve=True)
        except OmegaConfBaseException as exc:
            raise InvalidInferencePolicyError(
                f"Could not resolve {path} configuration: {exc}"
            ) from exc
    if not isinstance(raw, Mapping):
        raise InvalidInferencePolicyError(f"{path} must be a mapping.")
    return dict(raw)


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise InvalidInferencePolicyError(f"{path} contains unknown keys: {unknown}.")


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise InvalidInferencePolicyError(f"{path} must be a boolean.")
    return value


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInferencePolicyError(f"{path} must be an integer.")
    return int(value)


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInferencePolicyError(f"{path} must be numeric.")
    return float(value)


__all__ = [
    "ArtifactPolicy",
    "DecisionPolicy",
    "EnsemblePolicy",
    "InferencePolicy",
    "InvalidInferencePolicyError",
    "PostprocessingPolicy",
    "ResolvedInferencePolicy",
    "SlidingWindowPolicy",
    "TtaPolicy",
    "parse_inference_policy",
    "resolve_inference_policy",
]
