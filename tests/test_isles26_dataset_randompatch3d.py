import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from omegaconf import OmegaConf

from src.data.loader_stack.contracts import LabelRequiredError
from src.data.loader_stack.isles26_loader import ISLES26RandomPatches3D


def _write_nifti(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array.astype(np.float32), affine=np.eye(4)), str(path))


def _build_preprocessing_configs(
    *,
    roi_3d: list[int],
    patches_per_volume: int,
    pad_to_divisible: bool,
) -> dict:
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
            "volume_3d": roi_3d,
            "slice_2d": [roi_3d[0], roi_3d[1]],
        },
        "online_slices_3d_to_2d": {"slice_axis": 2, "slice_order": "sequential"},
        "random_patches_3d": {
            "patches_per_volume": {"train": patches_per_volume},
            "rand_crop_by_pos_neg_label": {
                "pos": 1,
                "neg": 1,
                "image_threshold": 0.0,
                "allow_smaller": True,
            },
            "pad_to_divisible": pad_to_divisible,
        },
        "full_volumes_3d": {"pad_to_divisible": False},
    }


class TestIsles26RandomPatches3D(unittest.TestCase):
    def test_label_free_random_patch_construction_fails_immediately(self):
        with self.assertRaisesRegex(LabelRequiredError, "requires labels"):
            ISLES26RandomPatches3D(
                directory=".",
                datalist_json="unused.json",
                modalities=["T1_RAW"],
                load_labels=False,
                preprocessing_configs={},
            )

    def _prepare_datalist(self, base: Path) -> Path:
        case_a_t1 = base / "R001/sub-r001s001/ses-1/anat/t1_a.nii.gz"
        case_a_label = base / "R001/sub-r001s001/ses-1/anat/label_a.nii.gz"
        case_b_t1 = base / "R002/sub-r002s001/ses-1/anat/t1_b.nii.gz"
        case_b_label = base / "R002/sub-r002s001/ses-1/anat/label_b.nii.gz"
        case_c_t1 = base / "R003/sub-r003s001/ses-1/anat/t1_c.nii.gz"
        case_c_label = base / "R003/sub-r003s001/ses-1/anat/label_c.nii.gz"

        vol = np.ones((3, 4, 2), dtype=np.float32)
        lbl = np.zeros((3, 4, 2), dtype=np.float32)
        lbl[1, 1, 1] = 1.0
        _write_nifti(case_a_t1, vol * 1.0)
        _write_nifti(case_a_label, lbl)
        _write_nifti(case_b_t1, vol * 2.0)
        _write_nifti(case_b_label, lbl)
        _write_nifti(case_c_t1, vol * 3.0)
        _write_nifti(case_c_label, lbl)

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
                {
                    "split": "train",
                    "caseID": "sub-r003s001",
                    "T1": ["R003/sub-r003s001/ses-1/anat/t1_c.nii.gz"],
                    "label": "R003/sub-r003s001/ses-1/anat/label_c.nii.gz",
                },
            ]
        }
        datalist_path = base / "isles26.json"
        datalist_path.write_text(json.dumps(datalist), encoding="utf-8")
        return datalist_path

    def test_flatten_policy_length_multiplies_cases_by_patch_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            datalist_path = self._prepare_datalist(base)
            dataset = ISLES26RandomPatches3D(
                directory=str(base),
                datalist_json=str(datalist_path),
                fold=0,
                subset_name="train",
                modalities=["T1_RAW"],
                test_flag=False,
                preprocessing_configs=_build_preprocessing_configs(
                    roi_3d=[6, 5, 4],
                    patches_per_volume=3,
                    pad_to_divisible=True,
                ),
            )

            # split=train => train subset receives 2 cases.
            self.assertEqual(len(dataset), 6)

    def test_getitem_returns_single_patch_with_expected_shape_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            datalist_path = self._prepare_datalist(base)
            dataset = ISLES26RandomPatches3D(
                directory=str(base),
                datalist_json=str(datalist_path),
                fold=0,
                subset_name="train",
                modalities=["T1_RAW"],
                test_flag=False,
                preprocessing_configs=_build_preprocessing_configs(
                    roi_3d=[6, 5, 4],
                    patches_per_volume=2,
                    pad_to_divisible=True,
                ),
            )

            image, label, patch_path = dataset[1]
            self.assertEqual(tuple(image.shape), (1, 6, 5, 4))
            self.assertEqual(tuple(label.shape), (1, 6, 5, 4))
            self.assertEqual(patch_path, "sub-r002s001_patch1")

    def test_random_patch_dataset_applies_3d_augmentation_before_cropping(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            case_a_t1 = base / "R001/sub-r001s001/ses-1/anat/t1_a.nii.gz"
            case_a_label = base / "R001/sub-r001s001/ses-1/anat/label_a.nii.gz"
            case_b_t1 = base / "R002/sub-r002s001/ses-1/anat/t1_b.nii.gz"
            case_b_label = base / "R002/sub-r002s001/ses-1/anat/label_b.nii.gz"

            case_a = np.zeros((3, 4, 2), dtype=np.float32)
            case_b = np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2) + 50.0
            label = np.zeros((3, 4, 2), dtype=np.float32)
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

            preprocessing_configs = _build_preprocessing_configs(
                roi_3d=[3, 4, 2],
                patches_per_volume=1,
                pad_to_divisible=False,
            )
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

            dataset = ISLES26RandomPatches3D(
                directory=str(base),
                datalist_json=str(datalist_path),
                fold=0,
                subset_name="train",
                modalities=["T1_RAW"],
                test_flag=False,
                preprocessing_configs=preprocessing_configs,
                aug_cfg=aug_cfg,
                is_training=True,
            )

            image, _label, _patch_path = dataset[0]
            self.assertEqual(tuple(image.shape), (1, 3, 4, 2))
            self.assertAlmostEqual(float(image[0, 0, 0, 0].item()), float(case_b[-1, 0, 0]))


if __name__ == "__main__":
    unittest.main()
