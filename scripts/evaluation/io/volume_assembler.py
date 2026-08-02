"""
Volume assembler for evaluation pipelines.

Accumulates channel-first 2D slices [C, H, W] and reconstructs channel-first
3D volumes [C, H, W, D] per (analysis_case_key, volume_id).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from torch import Tensor

from scripts.evaluation.core.contracts import SliceSample, VolumeSample


@dataclass
class _SliceBufferItem:
    case_id: str
    volume_id: str
    slice_index: int
    prediction_slice: Tensor
    ground_truth_slice: Tensor
    metadata: Dict[str, object]


class VolumeAssembler:
    """
    Assemble streaming slice samples into channel-first volumes.
    """

    def __init__(self) -> None:
        self._buffers: Dict[Tuple[str, str], Dict[int, _SliceBufferItem]] = {}

    def add_sample(self, analysis_case_key: str, sample: SliceSample) -> None:
        sample.validate()
        if sample.volume_id is None or sample.slice_index is None:
            raise ValueError(
                "VolumeAssembler requires SliceSample with volume_id and slice_index."
            )
        pred_slice = self._resolve_prediction_slice(sample)
        gt_slice = sample.ground_truth_mask.detach().cpu().float()
        pred_slice = _ensure_channel_first_slice(pred_slice)
        gt_slice = _ensure_channel_first_slice(gt_slice)
        if tuple(pred_slice.shape) != tuple(gt_slice.shape):
            raise ValueError(
                "Slice prediction/GT shape mismatch for "
                f"{sample.volume_id} slice {sample.slice_index}: "
                f"pred={tuple(pred_slice.shape)} gt={tuple(gt_slice.shape)}."
            )
        key = (str(analysis_case_key), str(sample.volume_id))
        if key not in self._buffers:
            self._buffers[key] = {}
        if int(sample.slice_index) in self._buffers[key]:
            raise ValueError(
                f"Duplicate slice index {sample.slice_index} for volume {sample.volume_id} "
                f"and analysis case {analysis_case_key}."
            )
        self._buffers[key][int(sample.slice_index)] = _SliceBufferItem(
            case_id=sample.case_id,
            volume_id=str(sample.volume_id),
            slice_index=int(sample.slice_index),
            prediction_slice=pred_slice,
            ground_truth_slice=gt_slice,
            metadata=dict(sample.metadata),
        )

    def finalize_volume(self, analysis_case_key: str, volume_id: str) -> Optional[VolumeSample]:
        key = (str(analysis_case_key), str(volume_id))
        item_map = self._buffers.get(key)
        if not item_map:
            return None
        raise ValueError(
            "2D slice-to-volume assembly is not compatible with geometry-aware "
            "evaluation. The deferred 2D reconstruction contract must provide "
            "typed parent/model geometry, slice axis and placement, context-centre "
            "semantics, and an invertible resize/preprocessing trace. VolumeAssembler "
            "will not manufacture an affine or infer space from slice metadata."
        )

    def finalize_case(self, analysis_case_key: str) -> List[VolumeSample]:
        """
        Finalize all currently buffered volumes for one analysis case.
        """
        case_key = str(analysis_case_key)
        volume_ids = sorted(
            volume_id for key_case, volume_id in self._buffers.keys() if key_case == case_key
        )
        finalized: List[VolumeSample] = []
        for volume_id in volume_ids:
            volume_sample = self.finalize_volume(case_key, volume_id)
            if volume_sample is not None:
                finalized.append(volume_sample)
        return finalized

    def finalize_all(self) -> Dict[str, List[VolumeSample]]:
        """
        Finalize all buffered volumes grouped by analysis case key.
        """
        grouped: Dict[str, List[VolumeSample]] = {}
        case_keys = sorted({key_case for key_case, _ in self._buffers.keys()})
        for case_key in case_keys:
            grouped[case_key] = self.finalize_case(case_key)
        return grouped

    def buffer_size(self) -> int:
        """Return number of open (case, volume) buffers."""
        return len(self._buffers)

    @staticmethod
    def _resolve_prediction_slice(sample: SliceSample) -> Tensor:
        if sample.prediction_prob is not None:
            return sample.prediction_prob.detach().cpu().float()
        if sample.prediction_mask is not None:
            return sample.prediction_mask.detach().cpu().float()
        raise ValueError("SliceSample has neither prediction_prob nor prediction_mask.")


def _ensure_channel_first_slice(tensor: Tensor) -> Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3:
        return tensor
    raise ValueError(
        f"Expected 2D/3D slice tensor ([H,W] or [C,H,W]), got shape={tuple(tensor.shape)}."
    )
