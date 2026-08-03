"""One-case convenience facade over the reusable preprocessing producer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.inference.case_producer import build_case_producer
from src.inference.contracts import (
    InferenceInputError,
    LabeledPreprocessedCase,
    PreprocessedCase,
)


def preprocess_case(
    *,
    dataset_id: str,
    case_id: str,
    raw_modalities: Mapping[str, Any],
    dataset_cfg: Any,
    label_path: str | Path | None = None,
    load_labels: bool = False,
) -> PreprocessedCase | LabeledPreprocessedCase:
    """Preprocess one 3D case through the registered reusable producer core."""
    if not isinstance(raw_modalities, Mapping):
        raise InferenceInputError("raw_modalities must be a canonical-key mapping.")

    producer = build_case_producer(
        dataset_id=dataset_id,
        dataset_cfg=dataset_cfg,
        load_labels=load_labels,
    )
    observed_keys = tuple(str(key) for key in raw_modalities)
    if set(observed_keys) != set(producer.required_raw_keys):
        raise InferenceInputError(
            f"Dataset '{producer.adapter.dataset_id}' raw modality keys must be exactly "
            f"{producer.required_raw_keys}; received {observed_keys}."
        )

    record = {"caseID": case_id, **dict(raw_modalities)}
    if load_labels:
        record["label"] = label_path
    return producer.preprocess(record)


__all__ = ["preprocess_case"]
