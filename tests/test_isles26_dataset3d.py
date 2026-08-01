import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from omegaconf import OmegaConf

from src.data.loader_stack.isles26_loader import ISLES26Dataset3D


def _write_nifti(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array.astype(np.float32), affine=np.eye(4)), str(path))


def _build_preprocessing_configs(image_size: int) -> dict:
    return {
        "common": {
            "orientation": {"enabled": False, "axcodes": "RAS"},
            "spacing": {
                "enabled": False,
                "allow_native_spacing": True,
                "pixdim": [1.0, 1.0, 1.0],
                "interpolation": {"image": "bilinear", "label": "nearest"},
            },
        },
        "roi": {
            "volume_3d": [image_size, image_size, image_size],
            "slice_2d": [image_size, image_size],
        },
        "online_slices_3d_to_2d": {"slice_axis": 2, "slice_order": "sequential"},
        "full_volumes_3d": {"pad_to_divisible": False},
    }


class TestIsles26Dataset3D(unittest.TestCase):
    def test_label_free_val_dataset_is_independent_of_test_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            case_t1 = base / "case/t1.nii.gz"
            _write_nifti(case_t1, np.arange(24, dtype=np.float32).reshape(3, 4, 2))
            datalist_path = base / "isles26.json"
            datalist_path.write_text(
                json.dumps(
                    {
                        "training": [
                            {
                                "split": "val",
                                "caseID": "case-label-free",
                                "T1": "case/t1.nii.gz",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            dataset = ISLES26Dataset3D(
                directory=str(base),
                datalist_json=str(datalist_path),
                subset_name="val",
                modalities=["T1_RAW"],
                test_flag=True,
                load_labels=False,
                preprocessing_configs=_build_preprocessing_configs(image_size=3),
            )

            image, case_id = dataset[0]
            self.assertEqual(case_id, "case-label-free")
            self.assertEqual(tuple(image.shape), (1, 3, 4, 2))
            self.assertFalse(dataset.load_labels)
            self.assertTrue(dataset.test_flag)

    def test_dataset3d_train_split_outputs_single_t1_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            case_a_t1 = base / "R001/sub-r001s001/ses-1/anat/t1_a.nii.gz"
            case_a_label = base / "R001/sub-r001s001/ses-1/anat/label_a.nii.gz"
            case_b_t1 = base / "R002/sub-r002s001/ses-1/anat/t1_b.nii.gz"
            case_b_label = base / "R002/sub-r002s001/ses-1/anat/label_b.nii.gz"

            _write_nifti(case_a_t1, np.ones((4, 5, 3), dtype=np.float32) * 11.0)
            _write_nifti(case_a_label, np.ones((4, 5, 3), dtype=np.float32))
            _write_nifti(case_b_t1, np.ones((4, 5, 3), dtype=np.float32) * 22.0)
            _write_nifti(case_b_label, np.ones((4, 5, 3), dtype=np.float32) * 2.0)

            datalist = {
                "training": [
                    {
                        "split": "val_fast",
                        "caseID": "sub-r001s001",
                        "siteID": "R001",
                        "T1": ["R001/sub-r001s001/ses-1/anat/t1_a.nii.gz"],
                        "label": "R001/sub-r001s001/ses-1/anat/label_a.nii.gz",
                    },
                    {
                        "split": "train",
                        "caseID": "sub-r002s001",
                        "siteID": "R002",
                        "T1": ["R002/sub-r002s001/ses-1/anat/t1_b.nii.gz"],
                        "label": "R002/sub-r002s001/ses-1/anat/label_b.nii.gz",
                    },
                ]
            }
            datalist_path = base / "isles26.json"
            datalist_path.write_text(json.dumps(datalist), encoding="utf-8")

            dataset = ISLES26Dataset3D(
                directory=str(base),
                datalist_json=str(datalist_path),
                fold=0,
                subset_name="train",
                transform=None,
                modalities=["T1_RAW"],
                test_flag=False,
                preprocessing_configs=_build_preprocessing_configs(image_size=4),
            )

            self.assertEqual(len(dataset), 1)
            image, label, case_id = dataset[0]

            self.assertEqual(case_id, "sub-r002s001")
            self.assertEqual(tuple(image.shape), (1, 4, 5, 3))
            self.assertEqual(tuple(label.shape), (1, 4, 5, 3))
            self.assertAlmostEqual(float(image[0, 0, 0, 0].item()), 22.0)
            self.assertAlmostEqual(float(label[0, 0, 0, 0].item()), 2.0)

    def test_dataset3d_val_split_supports_t1_suffix_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            case_t1 = base / "R001/sub-r001s001/ses-1/anat/t1.nii.gz"
            case_label = base / "R001/sub-r001s001/ses-1/anat/label.nii.gz"
            _write_nifti(case_t1, np.ones((3, 3, 2), dtype=np.float32) * 9.0)
            _write_nifti(case_label, np.ones((3, 3, 2), dtype=np.float32))

            datalist = {
                "training": [
                    {
                        "split": "val_fast",
                        "caseID": "sub-r001s001",
                        "T1": ["R001/sub-r001s001/ses-1/anat/t1.nii.gz"],
                        "label": "R001/sub-r001s001/ses-1/anat/label.nii.gz",
                    }
                ]
            }
            datalist_path = base / "isles26.json"
            datalist_path.write_text(json.dumps(datalist), encoding="utf-8")

            dataset = ISLES26Dataset3D(
                directory=str(base),
                datalist_json=str(datalist_path),
                fold=0,
                subset_name="val_fast",
                modalities=["T1_RAW"],
                test_flag=True,
                preprocessing_configs=_build_preprocessing_configs(image_size=3),
            )

            self.assertEqual(len(dataset), 1)
            image, label, case_id = dataset[0]
            self.assertEqual(case_id, "sub-r001s001")
            self.assertEqual(tuple(image.shape), (1, 3, 3, 2))
            self.assertEqual(tuple(label.shape), (1, 3, 3, 2))
            self.assertAlmostEqual(float(image[0, 0, 0, 0].item()), 9.0)

    def test_dataset3d_applies_3d_augmentation_for_train_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            case_a_t1 = base / "R001/sub-r001s001/ses-1/anat/t1_a.nii.gz"
            case_a_label = base / "R001/sub-r001s001/ses-1/anat/label_a.nii.gz"
            case_b_t1 = base / "R002/sub-r002s001/ses-1/anat/t1_b.nii.gz"
            case_b_label = base / "R002/sub-r002s001/ses-1/anat/label_b.nii.gz"

            case_a = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
            case_b = (np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3) + 100.0)
            label = np.zeros((4, 5, 3), dtype=np.float32)
            label[1, 1, 1] = 1.0
            _write_nifti(case_a_t1, case_a)
            _write_nifti(case_a_label, label)
            _write_nifti(case_b_t1, case_b)
            _write_nifti(case_b_label, label)

            datalist = {
                "training": [
                    {
                        "split": "val_fast",
                        "caseID": "sub-r001s001",
                        "T1": ["R001/sub-r001s001/ses-1/anat/t1_a.nii.gz"],
                        "label": "R001/sub-r001s001/ses-1/anat/label_a.nii.gz",
                    },
                    {
                        "split": "train",
                        "caseID": "sub-r002s001",
                        "T1": ["R002/sub-r002s001/ses-1/anat/t1_b.nii.gz"],
                        "label": "R002/sub-r002s001/ses-1/anat/label_b.nii.gz",
                    },
                ]
            }
            datalist_path = base / "isles26.json"
            datalist_path.write_text(json.dumps(datalist), encoding="utf-8")

            aug_cfg = OmegaConf.create(
                {
                    "spatial": {
                        "enabled": True,
                        "random_flip": {
                            "enabled": True,
                            "prob": 1.0,
                            "spatial_axis": [0],
                        },
                    },
                    "intensity": {"enabled": False},
                }
            )

            train_dataset = ISLES26Dataset3D(
                directory=str(base),
                datalist_json=str(datalist_path),
                fold=0,
                subset_name="train",
                modalities=["T1_RAW"],
                test_flag=False,
                preprocessing_configs=_build_preprocessing_configs(image_size=4),
                aug_cfg=aug_cfg,
                is_training=True,
            )
            val_dataset = ISLES26Dataset3D(
                directory=str(base),
                datalist_json=str(datalist_path),
                fold=0,
                subset_name="val_fast",
                modalities=["T1_RAW"],
                test_flag=True,
                preprocessing_configs=_build_preprocessing_configs(image_size=4),
                aug_cfg=None,
                is_training=False,
            )

            train_image, _train_label, _train_case_id = train_dataset[0]
            val_image, _val_label, _val_case_id = val_dataset[0]

            self.assertAlmostEqual(float(train_image[0, 0, 0, 0].item()), float(case_b[-1, 0, 0]))
            self.assertAlmostEqual(float(val_image[0, 0, 0, 0].item()), float(case_a[0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
