"""
IO adapters for nnU-Net prediction/ground-truth evaluation inputs.

This module provides:
- 2D slice adapters (existing nnU-Net post-threshold naming)
- 3D volume adapters (native per-case pairing)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Tuple

import nibabel as nib
import numpy as np
import torch
from torch import Tensor

from scripts.evaluation.core.contracts import VolumeSample
from scripts.evaluation.io.nnunet import (
    count_matched_pairs as count_nnunet_slice_pairs,
    iter_nnunet_slice_samples,
)
from src.inference.contracts import SUPPORTED_OUTPUT_SPACES, SpatialGeometry


SUPPORTED_INPUT_FORMATS = ("slices_2d", "volumes_3d")


def count_nnunet_volume_pairs(pred_dir: Path, gt_dir: Path) -> Tuple[int, int, int]:
    """
    Return (matched_count, missing_prediction_count, total_gt_files) for volume files.
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    gt_files = sorted(gt_dir.glob("*.nii.gz"))
    pred_lookup = {_basename_no_nii(path): path for path in pred_dir.glob("*.nii.gz")}

    missing = 0
    matched = 0
    for gt_file in gt_files:
        case_id = _basename_no_nii(gt_file)
        if case_id in pred_lookup:
            matched += 1
        else:
            missing += 1
    return matched, missing, len(gt_files)


def iter_nnunet_volume_samples(
    pred_dir: Path,
    gt_dir: Path,
    volume_space: str,
    strict_shape: bool = True,
) -> Iterator[VolumeSample]:
    """
    Yield geometry-verified nnU-Net volume pairs as ``VolumeSample``.
    """
    if volume_space not in SUPPORTED_OUTPUT_SPACES:
        raise ValueError(
            "nnU-Net 3D volume_space must be one of "
            f"{sorted(SUPPORTED_OUTPUT_SPACES)}, got {volume_space!r}."
        )
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory does not exist: {gt_dir}")

    gt_files = sorted(gt_dir.glob("*.nii.gz"))
    if not gt_files:
        raise FileNotFoundError(f"No .nii.gz files found in ground truth directory: {gt_dir}")

    pred_lookup: Dict[str, Path] = {_basename_no_nii(path): path for path in pred_dir.glob("*.nii.gz")}
    if not pred_lookup:
        raise FileNotFoundError(f"No .nii.gz files found in prediction directory: {pred_dir}")

    for gt_file in gt_files:
        case_key = _basename_no_nii(gt_file)
        pred_file = pred_lookup.get(case_key)
        if pred_file is None:
            continue

        gt_tensor, gt_geometry = _load_nifti_volume_tensor_with_metadata(gt_file)
        pred_tensor, pred_geometry = _load_nifti_volume_tensor_with_metadata(pred_file)

        if gt_tensor.shape != pred_tensor.shape:
            if strict_shape:
                raise ValueError(
                    f"Shape mismatch for {case_key}: pred={pred_tensor.shape}, gt={gt_tensor.shape}"
                )
            continue

        if not np.allclose(
            np.asarray(pred_geometry.affine),
            np.asarray(gt_geometry.affine),
            rtol=1.0e-5,
            atol=1.0e-5,
        ):
            raise ValueError(
                f"Affine mismatch for {case_key}: prediction and reference do not "
                "occupy the same physical grid."
            )
        if not np.allclose(
            pred_geometry.spacing,
            gt_geometry.spacing,
            rtol=1.0e-5,
            atol=1.0e-5,
        ):
            raise ValueError(f"Spacing mismatch for {case_key}.")
        if pred_geometry.orientation != gt_geometry.orientation:
            raise ValueError(
                f"Orientation mismatch for {case_key}: "
                f"pred={pred_geometry.orientation}, gt={gt_geometry.orientation}."
            )

        # nnUNetv2_predict writes hard labels by default. Keep native values
        # (typically 0/1 for binary lesion tasks) and let metrics apply policy.
        pred_volume = pred_tensor.float()
        gt_volume = gt_tensor.float()

        sample = VolumeSample(
            case_id=case_key,
            volume_id=case_key,
            prediction_volume=pred_volume,
            ground_truth_volume=gt_volume,
            prediction_space=volume_space,
            reference_space=volume_space,
            prediction_geometry=pred_geometry,
            reference_geometry=gt_geometry,
            metadata={
                "source": "nnunet_volumes",
                "pred_path": str(pred_file),
                "gt_path": str(gt_file),
                "case_key": case_key,
                "volume_space": volume_space,
                "source_affine": [list(row) for row in gt_geometry.affine],
                "pred_affine": [list(row) for row in pred_geometry.affine],
                "source_spacing_xyz": list(gt_geometry.spacing),
                "num_slices": int(gt_volume.shape[-1]),
            },
        )
        sample.validate()
        yield sample


def _load_nifti_volume_tensor_with_metadata(path: Path) -> Tuple[Tensor, SpatialGeometry]:
    nii = nib.load(path)
    data = nii.get_fdata()
    tensor = torch.from_numpy(data).float()
    volume_tensor = _to_channel_first_volume(tensor=tensor, path=path)
    affine = np.asarray(nii.affine, dtype=np.float64)
    geometry = SpatialGeometry(
        shape=tuple(int(value) for value in volume_tensor.shape[-3:]),
        affine=tuple(tuple(float(value) for value in row) for row in affine),
        spacing=tuple(float(value) for value in nib.affines.voxel_sizes(affine)),
        orientation="".join(str(code) for code in nib.aff2axcodes(affine)),
    )
    return volume_tensor, geometry


def _to_channel_first_volume(tensor: Tensor, path: Path) -> Tensor:
    """
    Normalize volume tensor to [C, H, W, D].

    Supported shapes:
    - [H, W, D]
    - [H, W, D, 1]
    - [1, H, W, D]
    """
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    if tensor.ndim == 4 and tensor.shape[-1] == 1:
        return tensor.squeeze(-1).unsqueeze(0)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        return tensor
    raise ValueError(
        f"Unsupported NIfTI volume shape for {path}: {tuple(tensor.shape)}. "
        "Expected [H,W,D], [H,W,D,1], or [1,H,W,D]."
    )


def _basename_no_nii(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return path.stem
