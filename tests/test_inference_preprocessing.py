import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from src.data.loader_stack.isles24_loader import (
    build_isles24_full_volumes_3d_pipeline,
)
from src.data.loader_stack.isles26_loader import build_full_volumes_3d_pipeline
from src.inference.contracts import (
    InferenceInputError,
    LabeledPreprocessedCase,
    PreprocessedCase,
)
from src.inference.preprocessing import preprocess_case


def _preprocessing_configs(*, orientation: bool = False) -> dict:
    return {
        "common": {
            "orientation": {"enabled": orientation, "axcodes": "RAS"},
            "spacing": {
                "enabled": False,
                "allow_native_spacing": True,
                "pixdim": [1.0, 1.0, 1.0],
                "interpolation": {"image": "bilinear", "label": "nearest"},
            },
        },
        "roi": {"volume_3d": [4, 5, 3], "slice_2d": [4, 5]},
        "online_slices_3d_to_2d": {"slice_axis": 2, "slice_order": "sequential"},
        "full_volumes_3d": {"pad_to_divisible": False},
        "t1": {
            "foreground": {"threshold": 0.0, "use_finite": True},
            "zscore": {"eps": 1.0e-6},
        },
    }


def _dataset_cfg() -> dict:
    return {
        "modalities": ["T1_RAW"],
        "preprocessing_configs": _preprocessing_configs(),
    }


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=2)
    nib.save(image, str(path))


class TestInferencePreprocessing(unittest.TestCase):
    def test_image_only_pipeline_contains_no_label_transform_keys(self):
        pipeline = build_full_volumes_3d_pipeline(
            modalities=["T1_RAW"],
            preprocessing_configs=_preprocessing_configs(),
            load_labels=False,
        )

        for transform in pipeline.transforms:
            keys = tuple(getattr(transform, "keys", ()))
            self.assertNotIn("label", keys)

        isles24_pipeline = build_isles24_full_volumes_3d_pipeline(
            modalities=["CBF_min_0_max_70"],
            preprocessing_configs=_preprocessing_configs(),
            load_labels=False,
        )
        for transform in isles24_pipeline.transforms:
            keys = tuple(getattr(transform, "keys", ()))
            self.assertNotIn("label", keys)

    def test_isles24_adapter_reuses_dataset_preprocessing_without_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "cbf.nii.gz"
            image = np.linspace(0.0, 70.0, 60, dtype=np.float32).reshape(4, 5, 3)
            _write_nifti(image_path, image, np.eye(4))

            result = preprocess_case(
                dataset_id="isles24",
                case_id="isles24-case",
                raw_modalities={"CBF": image_path},
                dataset_cfg={
                    "modalities": ["CBF_min_0_max_70"],
                    "preprocessing_configs": _preprocessing_configs(),
                },
                load_labels=False,
            )

            self.assertIsInstance(result, PreprocessedCase)
            self.assertEqual(tuple(result.image.shape), (1, 1, 4, 5, 3))
            self.assertEqual(result.reference_key, "CBF")

    def test_one_raw_modality_can_materialize_multiple_processed_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "t1.nii.gz"
            _write_nifti(
                image_path,
                np.arange(60, dtype=np.float32).reshape(4, 5, 3) + 1.0,
                np.eye(4),
            )

            result = preprocess_case(
                dataset_id="isles26",
                case_id="multi-channel",
                raw_modalities={"T1": image_path},
                dataset_cfg={
                    "modalities": ["T1_RAW", "T1_ZSCORE"],
                    "preprocessing_configs": _preprocessing_configs(),
                },
                load_labels=False,
            )

            self.assertEqual(tuple(result.image.shape), (1, 2, 4, 5, 3))
            self.assertEqual(tuple(result.native_metadata), ("T1",))

    def test_label_free_and_labeled_preprocessing_have_image_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image_path = base / "image.nii.gz"
            label_path = base / "label.nii.gz"
            affine = np.array(
                [
                    [-2.0, 0.0, 0.0, 12.0],
                    [0.0, 1.5, 0.0, -4.0],
                    [0.0, 0.0, 3.0, 7.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            image = np.arange(60, dtype=np.float32).reshape(4, 5, 3)
            label = np.zeros((4, 5, 3), dtype=np.uint8)
            label[1:3, 2:4, 1] = 1
            _write_nifti(image_path, image, affine)
            _write_nifti(label_path, label, affine)

            unlabeled = preprocess_case(
                dataset_id="isles26",
                case_id="case-a",
                raw_modalities={"T1": image_path},
                dataset_cfg={
                    **_dataset_cfg(),
                    "preprocessing_configs": _preprocessing_configs(orientation=True),
                },
                load_labels=False,
            )
            labeled = preprocess_case(
                dataset_id="isles26",
                case_id="case-a",
                raw_modalities={"T1": image_path},
                dataset_cfg={
                    **_dataset_cfg(),
                    "preprocessing_configs": _preprocessing_configs(orientation=True),
                },
                label_path=label_path,
                load_labels=True,
            )

            self.assertIsInstance(unlabeled, PreprocessedCase)
            self.assertNotIsInstance(unlabeled, LabeledPreprocessedCase)
            self.assertFalse(hasattr(unlabeled, "model_label"))
            self.assertIsInstance(labeled, LabeledPreprocessedCase)
            torch.testing.assert_close(unlabeled.image, labeled.case.image)
            self.assertEqual(tuple(unlabeled.image.shape), (1, 1, 4, 5, 3))
            self.assertEqual(tuple(labeled.model_label.shape), (1, 1, 4, 5, 3))
            self.assertEqual(tuple(labeled.native_label.shape), (1, 1, 4, 5, 3))
            self.assertEqual(labeled.model_label_geometry, labeled.model_geometry)
            self.assertEqual(
                labeled.native_label_geometry,
                labeled.native_label_metadata.geometry,
            )

            metadata = unlabeled.native_metadata["T1"]
            self.assertEqual(metadata.shape, (4, 5, 3))
            self.assertEqual(metadata.dtype, "float32")
            self.assertEqual(metadata.spacing, (2.0, 1.5, 3.0))
            self.assertEqual(metadata.orientation, "LAS")
            self.assertEqual(metadata.qform_code, 1)
            self.assertEqual(metadata.sform_code, 2)
            self.assertEqual(unlabeled.reference_key, "T1")
            self.assertEqual(unlabeled.spatial_trace.original.shape, (4, 5, 3))
            self.assertEqual(unlabeled.spatial_trace.model.shape, (4, 5, 3))
            self.assertEqual(unlabeled.spatial_trace.original.orientation, "LAS")
            self.assertEqual(unlabeled.spatial_trace.model.orientation, "RAS")

    def test_native_metadata_is_case_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            first_path = base / "first.nii.gz"
            second_path = base / "second.nii.gz"
            first_affine = np.diag([1.0, 2.0, 3.0, 1.0])
            second_affine = np.array(
                [
                    [0.0, -1.25, 0.0, 9.0],
                    [1.5, 0.0, 0.0, -2.0],
                    [0.0, 0.0, 2.5, 4.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            _write_nifti(first_path, np.ones((3, 4, 5), dtype=np.float32), first_affine)
            _write_nifti(second_path, np.ones((6, 5, 4), dtype=np.float32), second_affine)

            first = preprocess_case(
                dataset_id="isles26",
                case_id="first",
                raw_modalities={"T1": first_path},
                dataset_cfg=_dataset_cfg(),
                load_labels=False,
            )
            second = preprocess_case(
                dataset_id="isles26",
                case_id="second",
                raw_modalities={"T1": second_path},
                dataset_cfg=_dataset_cfg(),
                load_labels=False,
            )

            self.assertEqual(first.native_metadata["T1"].shape, (3, 4, 5))
            self.assertEqual(second.native_metadata["T1"].shape, (6, 5, 4))
            self.assertNotEqual(
                first.native_metadata["T1"].affine,
                second.native_metadata["T1"].affine,
            )

    def test_missing_or_extra_raw_modalities_fail_before_preprocessing(self):
        with self.assertRaisesRegex(InferenceInputError, "raw modality keys"):
            preprocess_case(
                dataset_id="isles26",
                case_id="missing",
                raw_modalities={},
                dataset_cfg=_dataset_cfg(),
                load_labels=False,
            )

        with self.assertRaisesRegex(InferenceInputError, "raw modality keys"):
            preprocess_case(
                dataset_id="isles26",
                case_id="extra",
                raw_modalities={"T1": "t1.nii.gz", "CBF": "cbf.nii.gz"},
                dataset_cfg=_dataset_cfg(),
                load_labels=False,
            )


if __name__ == "__main__":
    unittest.main()
