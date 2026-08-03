import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor

from src.data.loader_stack.isles26_loader import ISLES26Dataset3D
from src.data.loader_stack.preprocessing import get_preprocessing_adapter
from src.inference.case_producer import build_case_producer
from src.inference.contracts import (
    InferenceInputError,
    LabeledPreprocessedCase,
    PreprocessedCase,
    SpatialRestorationError,
)


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
        "online_slices_3d_to_2d": {
            "slice_axis": 2,
            "slice_order": "sequential",
        },
        "full_volumes_3d": {"pad_to_divisible": False},
    }


def _dataset_cfg(*, orientation: bool = False) -> dict:
    return {
        "modalities": ["T1_RAW"],
        "preprocessing_configs": _preprocessing_configs(orientation=orientation),
    }


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=2)
    nib.save(image, str(path))


def _write_case(
    base: Path,
    case_id: str,
    *,
    shape: tuple[int, int, int],
    affine: np.ndarray,
) -> dict:
    image_path = base / case_id / "t1.nii.gz"
    label_path = base / case_id / "label.nii.gz"
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1.0
    label = np.zeros(shape, dtype=np.uint8)
    label[tuple(size // 2 for size in shape)] = 1
    _write_nifti(image_path, image, affine)
    _write_nifti(label_path, label, affine)
    return {
        "caseID": case_id,
        "split": "val",
        "T1": [str(image_path)],
        "label": str(label_path),
    }


class TestPreprocessedCaseProducer(unittest.TestCase):
    def test_registered_isles24_adapter_produces_the_same_rich_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image_path = base / "isles24" / "cbf.nii.gz"
            label_path = base / "isles24" / "label.nii.gz"
            affine = np.diag([1.5, 2.0, 2.5, 1.0])
            _write_nifti(
                image_path,
                np.linspace(0.0, 70.0, 60, dtype=np.float32).reshape(4, 5, 3),
                affine,
            )
            _write_nifti(
                label_path,
                np.ones((4, 5, 3), dtype=np.uint8),
                affine,
            )

            produced = build_case_producer(
                dataset_id="isles24",
                dataset_cfg={
                    "modalities": ["CBF_min_0_max_70"],
                    "preprocessing_configs": _preprocessing_configs(),
                },
                load_labels=True,
            )(
                {
                    "caseID": "isles24-case",
                    "CBF": [str(image_path)],
                    "label": str(label_path),
                }
            )

            self.assertIsInstance(produced, LabeledPreprocessedCase)
            self.assertEqual(produced.case_id, "isles24-case")
            self.assertEqual(produced.case.reference_key, "CBF")
            self.assertEqual(produced.model_label_geometry, produced.model_geometry)
            self.assertEqual(produced.native_label_geometry.shape, (4, 5, 3))

    def test_one_pipeline_is_reused_across_heterogeneous_labeled_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            first_affine = np.array(
                [
                    [-1.4, 0.0, 0.0, 12.0],
                    [0.0, 2.1, 0.0, -3.0],
                    [0.0, 0.0, 3.2, 4.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            second_affine = np.array(
                [
                    [1.2, 0.15, 0.0, -8.0],
                    [0.0, 1.7, 0.2, 6.0],
                    [0.0, 0.0, 2.4, 11.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            records = [
                _write_case(
                    base,
                    "case-las",
                    shape=(4, 5, 3),
                    affine=first_affine,
                ),
                _write_case(
                    base,
                    "case-oblique",
                    shape=(6, 4, 5),
                    affine=second_affine,
                ),
            ]

            adapter = get_preprocessing_adapter("isles26")
            build_count = 0

            def counting_builder(modalities, preprocessing_configs, load_labels):
                nonlocal build_count
                build_count += 1
                return adapter.build_full_volume_pipeline(
                    modalities,
                    preprocessing_configs,
                    load_labels,
                )

            counted_adapter = replace(
                adapter,
                build_full_volume_pipeline=counting_builder,
            )
            with patch(
                "src.inference.case_producer.get_preprocessing_adapter",
                return_value=counted_adapter,
            ):
                producer = build_case_producer(
                    dataset_id="isles26",
                    dataset_cfg=_dataset_cfg(orientation=True),
                    load_labels=True,
                )
                produced = [producer(record) for record in records]

            self.assertEqual(build_count, 1)
            self.assertTrue(
                all(isinstance(case, LabeledPreprocessedCase) for case in produced)
            )
            self.assertEqual(produced[0].case_id, "case-las")
            self.assertEqual(produced[1].case_id, "case-oblique")
            self.assertEqual(produced[0].native_label_geometry.shape, (4, 5, 3))
            self.assertEqual(produced[1].native_label_geometry.shape, (6, 4, 5))
            self.assertNotEqual(
                produced[0].case.spatial_trace.original,
                produced[1].case.spatial_trace.original,
            )
            for case in produced:
                self.assertEqual(case.model_label_geometry, case.model_geometry)
                self.assertEqual(
                    tuple(case.model_label.shape[2:]),
                    tuple(case.case.image.shape[2:]),
                )

    def test_label_free_and_labeled_producers_have_image_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = _write_case(
                Path(tmp),
                "parity-case",
                shape=(4, 5, 3),
                affine=np.diag([-1.5, 2.0, 2.5, 1.0]),
            )
            unlabeled = build_case_producer(
                dataset_id="isles26",
                dataset_cfg=_dataset_cfg(orientation=True),
                load_labels=False,
            )(record)
            labeled = build_case_producer(
                dataset_id="isles26",
                dataset_cfg=_dataset_cfg(orientation=True),
                load_labels=True,
            )(record)

            self.assertIsInstance(unlabeled, PreprocessedCase)
            self.assertNotIsInstance(unlabeled, LabeledPreprocessedCase)
            self.assertFalse(hasattr(unlabeled, "model_label"))
            self.assertIsInstance(labeled, LabeledPreprocessedCase)
            torch.testing.assert_close(unlabeled.image, labeled.case.image)

    def test_producer_matches_established_dataset_model_tensor_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            record = _write_case(
                base,
                "loader-parity",
                shape=(4, 5, 3),
                affine=np.eye(4),
            )
            record_for_json = {
                **record,
                "T1": [str(Path(record["T1"][0]).relative_to(base))],
                "label": str(Path(record["label"]).relative_to(base)),
            }
            datalist = base / "datalist.json"
            datalist.write_text(
                json.dumps({"training": [record_for_json]}),
                encoding="utf-8",
            )
            preprocessing = _preprocessing_configs()
            dataset = ISLES26Dataset3D(
                directory=str(base),
                datalist_json=str(datalist),
                subset_name="val",
                partitioning="split",
                subset_definitions={"val": {"split_in": ("val",)}},
                modalities=["T1_RAW"],
                preprocessing_configs=preprocessing,
                is_training=False,
                load_labels=True,
            )
            expected_image, expected_label, expected_case_id = dataset[0]

            produced = build_case_producer(
                dataset_id="isles26",
                dataset_cfg={
                    "modalities": ["T1_RAW"],
                    "preprocessing_configs": preprocessing,
                },
                load_labels=True,
            )(record)

            self.assertEqual(produced.case_id, expected_case_id)
            torch.testing.assert_close(produced.case.image[0], expected_image)
            torch.testing.assert_close(produced.model_label[0], expected_label)

    def test_invalid_records_and_unavailable_adapters_fail_with_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image_3d_path = base / "three-dimensional.nii.gz"
            image_path = base / "four-dimensional.nii.gz"
            _write_nifti(
                image_3d_path,
                np.ones((2, 3, 4), dtype=np.float32),
                np.eye(4),
            )
            _write_nifti(
                image_path,
                np.ones((2, 3, 4, 2), dtype=np.float32),
                np.eye(4),
            )
            labeled_producer = build_case_producer(
                dataset_id="isles26",
                dataset_cfg=_dataset_cfg(),
                load_labels=True,
            )
            with self.assertRaisesRegex(
                InferenceInputError,
                "case 'missing-label'.*field 'label'",
            ):
                labeled_producer(
                    {
                        "caseID": "missing-label",
                        "T1": str(image_3d_path),
                    }
                )

            unlabeled_producer = build_case_producer(
                dataset_id="isles26",
                dataset_cfg=_dataset_cfg(),
                load_labels=False,
            )
            with self.assertRaisesRegex(
                InferenceInputError,
                "case 'wrong-rank'.*must be a 3D NIfTI volume",
            ):
                unlabeled_producer(
                    {
                        "caseID": "wrong-rank",
                        "T1": str(image_path),
                    }
                )

            nonfinite_record = _write_case(
                base,
                "nonfinite-label",
                shape=(2, 3, 4),
                affine=np.eye(4),
            )
            _write_nifti(
                Path(nonfinite_record["label"]),
                np.full((2, 3, 4), np.nan, dtype=np.float32),
                np.eye(4),
            )
            with self.assertRaisesRegex(
                InferenceInputError,
                "case 'nonfinite-label'.*non-empty and finite",
            ):
                labeled_producer(nonfinite_record)

        with self.assertRaisesRegex(
            InferenceInputError,
            "dataset 'not-registered'.*No preprocessing adapter",
        ):
            build_case_producer(
                dataset_id="not-registered",
                dataset_cfg=_dataset_cfg(),
                load_labels=False,
            )

    def test_changed_grid_without_transform_history_fails_before_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            record = _write_case(
                base,
                "missing-trace",
                shape=(2, 2, 2),
                affine=np.eye(4),
            )
            adapter = get_preprocessing_adapter("isles26")

            def build_incomplete_pipeline(*_args):
                def incomplete_pipeline(_record):
                    return {
                        "image": MetaTensor(
                            torch.ones((1, 2, 2, 2), dtype=torch.float32),
                            affine=torch.diag(torch.tensor([2.0, 2.0, 2.0, 1.0])),
                        )
                    }

                return incomplete_pipeline

            incomplete_adapter = replace(
                adapter,
                build_full_volume_pipeline=build_incomplete_pipeline,
            )
            with patch(
                "src.inference.case_producer.get_preprocessing_adapter",
                return_value=incomplete_adapter,
            ):
                producer = build_case_producer(
                    dataset_id="isles26",
                    dataset_cfg=_dataset_cfg(),
                    load_labels=False,
                )
                with self.assertRaisesRegex(
                    SpatialRestorationError,
                    "case 'missing-trace'.*did not retain an applied transform history",
                ):
                    producer(record)

    def test_model_image_and_label_grid_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            record = _write_case(
                base,
                "mismatched-label-grid",
                shape=(3, 4, 5),
                affine=np.eye(4),
            )
            shifted_affine = np.eye(4)
            shifted_affine[0, 3] = 0.25
            _write_nifti(
                Path(record["label"]),
                np.ones((3, 4, 5), dtype=np.uint8),
                shifted_affine,
            )
            producer = build_case_producer(
                dataset_id="isles26",
                dataset_cfg=_dataset_cfg(),
                load_labels=True,
            )

            with self.assertRaisesRegex(
                SpatialRestorationError,
                "case 'mismatched-label-grid'.*same model-space physical grid",
            ):
                producer(record)


if __name__ == "__main__":
    unittest.main()
