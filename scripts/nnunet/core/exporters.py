"""
Export strategies and low-level filesystem helpers for nnU-Net conversion.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm

from src.inference.contracts import SUPPORTED_OUTPUT_SPACES


def clear_directory(directory: Path, description: str = "") -> int:
    """
    Remove all files from a directory, keeping the directory itself.

    Args:
        directory: Path to directory to clear.
        description: Human-readable description for logging.

    Returns:
        Number of files removed.
    """
    if not directory.exists():
        return 0

    files = list(directory.glob("*"))
    count = len(files)

    if count > 0:
        # Remove and recreate for speed (faster than individual deletes)
        shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        label = f" ({description})" if description else ""
        print(f"  Cleared {count} stale files from {directory.name}/{label}")

    return count


def count_files_in_directory(directory: Path, pattern: str = "*.nii.gz") -> int:
    """
    Count files matching a pattern in a directory.

    Used to determine numTraining from existing files on disk
    when only exporting the test split (resume scenario).

    Args:
        directory: Path to directory.
        pattern: Glob pattern to match.

    Returns:
        Number of matching files.
    """
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def write_provenance_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Write one JSON record per line for exported slice provenance.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


class SliceExportStrategy:
    """
    Export strategy for online 3D-to-2D slice datasets.
    """

    sample_unit_name = "slices"
    mode_key = "slices_2d"

    @staticmethod
    def _build_export_affine(
        slice_meta: Dict[str, Any],
        out_h: int,
        out_w: int,
        slice_idx: int,
    ) -> np.ndarray:
        """
        Build slice-export affine that preserves source geometry and includes
        index-aware z-offset.
        """
        source_affine = np.asarray(slice_meta.get("source_affine", np.eye(4)), dtype=np.float64)
        if source_affine.shape != (4, 4):
            source_affine = np.eye(4, dtype=np.float64)
        pre_hw = slice_meta.get("pre_resize_shape_hw", [out_h, out_w])
        try:
            pre_h = max(int(pre_hw[0]), 1)
            pre_w = max(int(pre_hw[1]), 1)
        except Exception:
            pre_h, pre_w = out_h, out_w
        out_h = max(int(out_h), 1)
        out_w = max(int(out_w), 1)
        scale_h = float(pre_h / out_h)
        scale_w = float(pre_w / out_w)

        export_affine = np.array(source_affine, dtype=np.float64, copy=True)
        export_affine[:3, 0] = source_affine[:3, 0] * scale_h
        export_affine[:3, 1] = source_affine[:3, 1] * scale_w
        export_affine[:3, 3] = source_affine[:3, 3] + source_affine[:3, 2] * float(slice_idx)
        return export_affine

    @staticmethod
    def _write_nifti_with_affine(data: np.ndarray, affine: np.ndarray, output_path: Path) -> None:
        nii_img = nib.Nifti1Image(data, affine=affine)
        nii_img.set_qform(affine, code=1)
        nii_img.set_sform(affine, code=1)
        nib.save(nii_img, output_path)

    def _process_single_slice(
        self,
        idx: int,
        dataset: Any,
        images_dir: Path,
        labels_dir: Path,
        num_channels: int,
        split_name: str,
    ) -> Dict[str, Any]:
        """
        Process and save a single slice to nnU-Net format.

        Returns:
            Slice export record for provenance manifest.
        """
        sample = dataset[idx]
        if len(sample) == 4:
            image, label, virtual_path, slice_meta = sample
        else:
            image, label, virtual_path = sample
            slice_meta = {}
        # image: [C, H, W], label: [1, H, W]

        # Parse case_id and slice_idx from virtual_path
        # Format from dataset route: "{caseID}_slice{slice_idx}"
        case_id, slice_part = virtual_path.rsplit("_slice", 1)
        slice_idx = int(slice_part)

        # Create case identifier for nnU-Net
        safe_case_id = f"{case_id}_s{slice_idx:04d}"
        out_h = int(image.shape[-2])
        out_w = int(image.shape[-1])
        export_affine = self._build_export_affine(
            slice_meta,
            out_h=out_h,
            out_w=out_w,
            slice_idx=slice_idx,
        )
        image_files: List[str] = []

        # Save each channel as separate file
        for ch_idx in range(num_channels):
            ch_data = image[ch_idx].numpy()  # [H, W]

            # Create 2D NIfTI (shape [H, W, 1] for 2D)
            nii_data = ch_data[..., np.newaxis].astype(np.float32)
            filename = f"{safe_case_id}_{ch_idx:04d}.nii.gz"
            self._write_nifti_with_affine(nii_data, export_affine, images_dir / filename)
            image_files.append(filename)

        # Save label (as uint8 for segmentation)
        label_data = label[0].numpy()  # [H, W]
        label_nii = label_data[..., np.newaxis].astype(np.uint8)
        label_filename = f"{safe_case_id}.nii.gz"
        self._write_nifti_with_affine(label_nii, export_affine, labels_dir / label_filename)

        return {
            "split": split_name,
            "safe_case_id": safe_case_id,
            "case_id": str(case_id),
            "slice_index": int(slice_idx),
            "virtual_path": str(virtual_path),
            "image_files": image_files,
            "label_file": label_filename,
            "source_path": str(slice_meta.get("source_path", "")),
            "source_affine": slice_meta.get("source_affine", np.eye(4).tolist()),
            "source_spacing_xyz": slice_meta.get("source_spacing_xyz", [1.0, 1.0, 1.0]),
            "source_axcodes": slice_meta.get("source_axcodes", ["R", "A", "S"]),
            "source_volume_shape": slice_meta.get("source_volume_shape", [out_h, out_w, 1]),
            "slice_axis": int(slice_meta.get("slice_axis", 2)),
            "pre_resize_shape_hw": slice_meta.get("pre_resize_shape_hw", [out_h, out_w]),
            "post_resize_shape_hw": [out_h, out_w],
            "export_affine": export_affine.tolist(),
        }

    def _export_dataset_sequential(
        self,
        dataset: Any,
        images_dir: Path,
        labels_dir: Path,
        num_channels: int,
        desc: str,
        max_slices: Optional[int] = None,
        split_name: str = "unknown",
    ) -> Tuple[set, List[Dict[str, Any]]]:
        """
        Export a dataset to nnU-Net format sequentially.
        """
        case_ids = set()
        records: List[Dict[str, Any]] = []

        # Determine number of slices to process
        total_slices = len(dataset)
        num_to_process = min(max_slices, total_slices) if max_slices else total_slices

        # Sequential iteration by index
        for idx in tqdm(range(num_to_process), desc=desc):
            record = self._process_single_slice(
                idx,
                dataset,
                images_dir,
                labels_dir,
                num_channels,
                split_name=split_name,
            )
            case_ids.add(record["safe_case_id"])
            records.append(record)

        return case_ids, records

    def _export_dataset_parallel(
        self,
        dataset: Any,
        images_dir: Path,
        labels_dir: Path,
        num_channels: int,
        desc: str,
        max_slices: Optional[int] = None,
        num_workers: int = 32,
        split_name: str = "unknown",
    ) -> Tuple[set, List[Dict[str, Any]]]:
        """
        Export a dataset to nnU-Net format using parallel threads.

        Uses ThreadPoolExecutor for parallel processing. GIL is released
        during I/O operations (nibabel read/write), allowing good parallelism.
        """
        # Determine number of slices to process
        total_slices = len(dataset)
        num_to_process = min(max_slices, total_slices) if max_slices else total_slices

        case_ids = set()
        records: List[Dict[str, Any]] = []

        # Create partial function with fixed arguments
        process_fn = partial(
            self._process_single_slice,
            dataset=dataset,
            images_dir=images_dir,
            labels_dir=labels_dir,
            num_channels=num_channels,
            split_name=split_name,
        )

        # Process slices in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_fn, idx): idx for idx in range(num_to_process)}

            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=num_to_process, desc=desc):
                record = future.result()
                case_ids.add(record["safe_case_id"])
                records.append(record)

        return case_ids, records

    def export_split(
        self,
        dataset: Any,
        images_dir: Path,
        labels_dir: Path,
        num_channels: int,
        desc: str,
        max_slices: Optional[int],
        split_name: str,
        use_parallel: bool,
        num_workers: int,
    ) -> Tuple[set, List[Dict[str, Any]]]:
        """
        Export a split using either sequential or parallel execution.
        """
        if use_parallel:
            return self._export_dataset_parallel(
                dataset=dataset,
                images_dir=images_dir,
                labels_dir=labels_dir,
                num_channels=num_channels,
                desc=desc,
                max_slices=max_slices,
                num_workers=num_workers,
                split_name=split_name,
            )
        return self._export_dataset_sequential(
            dataset=dataset,
            images_dir=images_dir,
            labels_dir=labels_dir,
            num_channels=num_channels,
            desc=desc,
            max_slices=max_slices,
            split_name=split_name,
        )


class VolumeExportStrategy:
    """
    Export strategy for full-volume 3D datasets.
    """

    sample_unit_name = "cases"
    mode_key = "volumes_3d"

    def __init__(self, export_space: str):
        if export_space not in SUPPORTED_OUTPUT_SPACES:
            raise ValueError(
                "VolumeExportStrategy export_space must be one of "
                f"{sorted(SUPPORTED_OUTPUT_SPACES)}, got {export_space!r}."
            )
        self.export_space = export_space

    @staticmethod
    def _to_numpy(data: Any) -> np.ndarray:
        if hasattr(data, "detach"):
            data = data.detach()
        if hasattr(data, "cpu"):
            data = data.cpu()
        if hasattr(data, "numpy"):
            return data.numpy()
        return np.asarray(data)

    @staticmethod
    def _extract_path(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return ""
            value = value[0]
        if isinstance(value, str) and len(value.strip()) > 0:
            candidate = Path(value)
            if candidate.exists():
                return str(candidate)
        return ""

    def _resolve_source_path(self, dataset: Any, idx: int) -> str:
        current = dataset
        resolved_idx = int(idx)
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            indices = getattr(current, "indices", None)
            nested = getattr(current, "dataset", None)
            if indices is None or nested is None:
                break
            if resolved_idx < 0 or resolved_idx >= len(indices):
                return ""
            resolved_idx = int(indices[resolved_idx])
            current = nested

        database = getattr(current, "database", None)
        if (
            not isinstance(database, list)
            or resolved_idx < 0
            or resolved_idx >= len(database)
        ):
            return ""

        record = database[resolved_idx]
        if not isinstance(record, dict):
            return ""

        candidate_keys: List[str] = []
        for key in getattr(current, "base_modalities", []) or []:
            candidate_keys.append(str(key))
        for modality in getattr(current, "modalities", []) or []:
            candidate_keys.append(str(modality).split("_")[0])

        # Keep original order while deduplicating.
        seen = set()
        ordered_keys: List[str] = []
        for key in candidate_keys:
            if key in seen:
                continue
            seen.add(key)
            ordered_keys.append(key)

        for key in ordered_keys:
            path = self._extract_path(record.get(key))
            if path:
                return path

        ignored_keys = {"label", "caseID", "fold", "siteID", "metadata", "metadata_csv"}
        for key, value in record.items():
            if key in ignored_keys:
                continue
            path = self._extract_path(value)
            if path:
                return path
        return ""

    @staticmethod
    def _resolve_metadata_from_source(
        source_path: str,
    ) -> Dict[str, Any]:
        if not source_path or not Path(source_path).exists():
            raise ValueError(
                "3D nnUNet export requires a readable source image path so source "
                "and export geometry can be distinguished."
            )
        try:
            source_img = nib.load(source_path)
        except Exception as exc:
            raise ValueError(
                f"Could not read source NIfTI geometry from {source_path}: {exc}"
            ) from exc
        affine = np.asarray(source_img.affine, dtype=np.float64)
        return {
            "source_path": source_path,
            "source_affine": affine.tolist(),
            "source_spacing_xyz": [
                float(value) for value in nib.affines.voxel_sizes(affine)
            ],
            "source_axcodes": [str(code) for code in nib.aff2axcodes(affine)],
            "source_volume_shape": [int(dim) for dim in source_img.shape[:3]],
        }

    @staticmethod
    def _extract_transformed_affine(
        value: Any, *, field_name: str
    ) -> Optional[np.ndarray]:
        affine_value = getattr(value, "affine", None)
        if affine_value is None:
            return None
        if hasattr(affine_value, "detach"):
            affine_value = affine_value.detach()
        if hasattr(affine_value, "cpu"):
            affine_value = affine_value.cpu()
        affine = np.asarray(affine_value, dtype=np.float64)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError(
                f"{field_name} transformed affine must be a finite 4x4 matrix, "
                f"got shape {tuple(affine.shape)}."
            )
        return affine

    @staticmethod
    def _geometry_from_affine(affine: np.ndarray, shape: List[int]) -> Dict[str, Any]:
        return {
            "export_affine": affine.tolist(),
            "export_spacing_xyz": [
                float(value) for value in nib.affines.voxel_sizes(affine)
            ],
            "export_axcodes": [str(code) for code in nib.aff2axcodes(affine)],
            "export_volume_shape": [int(value) for value in shape],
        }

    @staticmethod
    def _write_nifti_with_affine(data: np.ndarray, affine: np.ndarray, output_path: Path) -> None:
        nii_img = nib.Nifti1Image(data, affine=affine)
        nii_img.set_qform(affine, code=1)
        nii_img.set_sform(affine, code=1)
        nib.save(nii_img, output_path)

    def _process_single_case(
        self,
        idx: int,
        dataset: Any,
        images_dir: Path,
        labels_dir: Path,
        num_channels: int,
        split_name: str,
    ) -> Dict[str, Any]:
        sample = dataset[idx]
        if len(sample) < 3:
            raise ValueError(
                "VolumeExportStrategy expects dataset samples as (image, label, case_id). "
                f"Got sample length {len(sample)}."
            )
        image, label, case_id = sample[0], sample[1], sample[2]
        safe_case_id = str(case_id)

        image_affine = self._extract_transformed_affine(image, field_name="image")
        label_affine = self._extract_transformed_affine(label, field_name="label")
        if label_affine is None:
            raise ValueError(
                "3D nnUNet export requires transformed label MetaTensor geometry; "
                f"case '{safe_case_id}' returned a label without an affine."
            )
        if image_affine is not None and not np.allclose(
            image_affine, label_affine, rtol=1.0e-5, atol=1.0e-5
        ):
            raise ValueError(
                "Transformed image/label affine mismatch for 3D nnUNet export "
                f"case '{safe_case_id}'."
            )

        image_np = self._to_numpy(image)
        if image_np.ndim != 4:
            raise ValueError(
                f"Volume export expects image shape [C,H,W,D], got {tuple(image_np.shape)} "
                f"for case '{safe_case_id}'."
            )
        if int(image_np.shape[0]) != int(num_channels):
            raise ValueError(
                "Volume export channel mismatch: "
                f"expected {num_channels}, got {int(image_np.shape[0])} for case '{safe_case_id}'."
            )

        label_np = self._to_numpy(label)
        if label_np.ndim == 4 and label_np.shape[0] == 1:
            label_np = label_np[0]
        if label_np.ndim != 3:
            raise ValueError(
                f"Volume export expects label shape [H,W,D] or [1,H,W,D], got "
                f"{tuple(label_np.shape)} for case '{safe_case_id}'."
            )

        if tuple(image_np.shape[-3:]) != tuple(label_np.shape):
            raise ValueError(
                "Transformed image/label spatial shape mismatch for 3D nnUNet export "
                f"case '{safe_case_id}': image={tuple(image_np.shape[-3:])}, "
                f"label={tuple(label_np.shape)}."
            )

        fallback_shape = [int(dim) for dim in image_np.shape[-3:]]
        source_path = self._resolve_source_path(dataset, idx)
        metadata = self._resolve_metadata_from_source(source_path)
        export_affine = label_affine
        export_geometry = self._geometry_from_affine(export_affine, fallback_shape)

        if self.export_space == "native_input":
            source_affine = np.asarray(metadata["source_affine"], dtype=np.float64)
            source_shape = tuple(int(value) for value in metadata["source_volume_shape"])
            if source_shape != tuple(fallback_shape) or not np.allclose(
                source_affine, export_affine, rtol=1.0e-5, atol=1.0e-5
            ):
                raise ValueError(
                    "nnunet.export_space='native_input' is inconsistent with the "
                    f"actual exported grid for case '{safe_case_id}': "
                    f"source_shape={source_shape}, export_shape={tuple(fallback_shape)}."
                )

        image_files: List[str] = []
        for ch_idx in range(num_channels):
            ch_volume = np.asarray(image_np[ch_idx], dtype=np.float32)
            filename = f"{safe_case_id}_{ch_idx:04d}.nii.gz"
            self._write_nifti_with_affine(ch_volume, export_affine, images_dir / filename)
            image_files.append(filename)

        label_filename = f"{safe_case_id}.nii.gz"
        label_volume = np.asarray(label_np, dtype=np.uint8)
        self._write_nifti_with_affine(label_volume, export_affine, labels_dir / label_filename)

        return {
            "split": split_name,
            "safe_case_id": safe_case_id,
            "case_id": safe_case_id,
            "image_files": image_files,
            "label_file": label_filename,
            "source_path": metadata["source_path"],
            "source_affine": metadata["source_affine"],
            "source_spacing_xyz": metadata["source_spacing_xyz"],
            "source_axcodes": metadata["source_axcodes"],
            "source_volume_shape": metadata["source_volume_shape"],
            "export_space": self.export_space,
            **export_geometry,
            "post_export_shape_hwd": fallback_shape,
        }

    def _export_dataset_sequential(
        self,
        dataset: Any,
        images_dir: Path,
        labels_dir: Path,
        num_channels: int,
        desc: str,
        max_slices: Optional[int] = None,
        split_name: str = "unknown",
    ) -> Tuple[set, List[Dict[str, Any]]]:
        case_ids = set()
        records: List[Dict[str, Any]] = []

        total_cases = len(dataset)
        num_to_process = min(max_slices, total_cases) if max_slices else total_cases

        for idx in tqdm(range(num_to_process), desc=desc):
            record = self._process_single_case(
                idx,
                dataset,
                images_dir,
                labels_dir,
                num_channels,
                split_name=split_name,
            )
            case_ids.add(record["safe_case_id"])
            records.append(record)

        return case_ids, records

    def _export_dataset_parallel(
        self,
        dataset: Any,
        images_dir: Path,
        labels_dir: Path,
        num_channels: int,
        desc: str,
        max_slices: Optional[int] = None,
        num_workers: int = 32,
        split_name: str = "unknown",
    ) -> Tuple[set, List[Dict[str, Any]]]:
        total_cases = len(dataset)
        num_to_process = min(max_slices, total_cases) if max_slices else total_cases

        case_ids = set()
        records: List[Dict[str, Any]] = []

        process_fn = partial(
            self._process_single_case,
            dataset=dataset,
            images_dir=images_dir,
            labels_dir=labels_dir,
            num_channels=num_channels,
            split_name=split_name,
        )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_fn, idx): idx for idx in range(num_to_process)}
            for future in tqdm(as_completed(futures), total=num_to_process, desc=desc):
                record = future.result()
                case_ids.add(record["safe_case_id"])
                records.append(record)

        return case_ids, records

    def export_split(
        self,
        dataset: Any,
        images_dir: Path,
        labels_dir: Path,
        num_channels: int,
        desc: str,
        max_slices: Optional[int],
        split_name: str,
        use_parallel: bool,
        num_workers: int,
    ) -> Tuple[set, List[Dict[str, Any]]]:
        if use_parallel:
            return self._export_dataset_parallel(
                dataset=dataset,
                images_dir=images_dir,
                labels_dir=labels_dir,
                num_channels=num_channels,
                desc=desc,
                max_slices=max_slices,
                num_workers=num_workers,
                split_name=split_name,
            )
        return self._export_dataset_sequential(
            dataset=dataset,
            images_dir=images_dir,
            labels_dir=labels_dir,
            num_channels=num_channels,
            desc=desc,
            max_slices=max_slices,
            split_name=split_name,
        )
