"""
Export reconstructed channel-first volumes to NIfTI for visual QA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

from scripts.evaluation.core.contracts import VolumeSample


def export_reconstructed_volumes(
    grouped_volumes: Dict[str, List[VolumeSample]],
    output_dir: Path,
    max_volumes_per_case: Optional[int] = None,
) -> List[Path]:
    """
    Export reconstructed prediction/GT volumes to NIfTI files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for case_key in sorted(grouped_volumes.keys()):
        case_dir = output_dir / str(case_key)
        case_dir.mkdir(parents=True, exist_ok=True)
        volumes = grouped_volumes[case_key]
        if max_volumes_per_case is not None and max_volumes_per_case >= 0:
            volumes = volumes[: max_volumes_per_case]
        for volume_sample in volumes:
            volume_sample.validate()
            prediction_affine = np.asarray(
                volume_sample.prediction_geometry.affine,
                dtype=np.float64,
            )
            reference_affine = np.asarray(
                volume_sample.reference_geometry.affine,
                dtype=np.float64,
            )
            pred_data = _to_nifti_array(volume_sample.prediction_volume)
            gt_data = _to_nifti_array(volume_sample.ground_truth_volume)
            pred_path = case_dir / f"{volume_sample.volume_id}__pred.nii.gz"
            gt_path = case_dir / f"{volume_sample.volume_id}__gt.nii.gz"
            _write_nifti(pred_data, prediction_affine, pred_path)
            _write_nifti(gt_data, reference_affine, gt_path)
            written.extend([pred_path, gt_path])
    return written


def _to_nifti_array(volume_chwd) -> np.ndarray:
    # [C,H,W,D] -> [H,W,D] (single-channel expected)
    arr = volume_chwd.detach().cpu().numpy()
    if arr.ndim != 4:
        raise ValueError(f"Expected [C,H,W,D], got {arr.shape}")
    if arr.shape[0] != 1:
        raise ValueError(f"Expected single-channel volume, got C={arr.shape[0]}")
    return arr[0].astype(np.float32)


def _write_nifti(data_hwd: np.ndarray, affine: np.ndarray, path: Path) -> None:
    nii = nib.Nifti1Image(data_hwd, affine=affine)
    nii.set_qform(affine, code=1)
    nii.set_sform(affine, code=1)
    nib.save(nii, path)
