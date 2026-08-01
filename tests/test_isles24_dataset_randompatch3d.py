import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from src.data.loader_stack.contracts import LabelRequiredError
from src.data.loader_stack.isles24_loader import ISLES24RandomPatches3D


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
                "pixdim": [1.0, 1.0, 1.0],
                "interpolation": {"image": "bilinear", "label": "nearest"},
            },
        },
        "roi": {"volume_3d": roi_3d, "slice_2d": [roi_3d[0], roi_3d[1]]},
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


class TestIsles24RandomPatches3D(unittest.TestCase):
    def test_label_free_random_patch_construction_fails_immediately(self):
        with self.assertRaisesRegex(LabelRequiredError, "requires labels"):
            ISLES24RandomPatches3D(
                directory=".",
                datalist_json="unused.json",
                modalities=["CBF_min_0_max_70"],
                load_labels=False,
                preprocessing_configs={},
            )

    def _prepare_datalist(self, base: Path) -> Path:
        case_a = {
            "cbf": base / "R001/sub-r001s001/ses-1/anat/cbf_a.nii.gz",
            "tmax": base / "R001/sub-r001s001/ses-1/anat/tmax_a.nii.gz",
            "label": base / "R001/sub-r001s001/ses-1/anat/label_a.nii.gz",
        }
        case_b = {
            "cbf": base / "R002/sub-r002s001/ses-1/anat/cbf_b.nii.gz",
            "tmax": base / "R002/sub-r002s001/ses-1/anat/tmax_b.nii.gz",
            "label": base / "R002/sub-r002s001/ses-1/anat/label_b.nii.gz",
        }
        case_c = {
            "cbf": base / "R003/sub-r003s001/ses-1/anat/cbf_c.nii.gz",
            "tmax": base / "R003/sub-r003s001/ses-1/anat/tmax_c.nii.gz",
            "label": base / "R003/sub-r003s001/ses-1/anat/label_c.nii.gz",
        }

        vol_shape = (3, 4, 2)
        cbf_base = np.ones(vol_shape, dtype=np.float32) * 20.0
        tmax_base = np.ones(vol_shape, dtype=np.float32) * 10.0
        label = np.zeros(vol_shape, dtype=np.float32)
        label[1, 1, 1] = 1.0

        _write_nifti(case_a["cbf"], cbf_base)
        _write_nifti(case_a["tmax"], tmax_base)
        _write_nifti(case_a["label"], label)

        _write_nifti(case_b["cbf"], cbf_base * 1.5)
        _write_nifti(case_b["tmax"], tmax_base * 1.2)
        _write_nifti(case_b["label"], label)

        _write_nifti(case_c["cbf"], cbf_base * 2.0)
        _write_nifti(case_c["tmax"], tmax_base * 1.4)
        _write_nifti(case_c["label"], label)

        datalist = {
            "training": [
                {
                    "fold": 0,
                    "caseID": "sub-r001s001",
                    "CBF": ["R001/sub-r001s001/ses-1/anat/cbf_a.nii.gz"],
                    "TMAX": ["R001/sub-r001s001/ses-1/anat/tmax_a.nii.gz"],
                    "label": "R001/sub-r001s001/ses-1/anat/label_a.nii.gz",
                },
                {
                    "fold": 1,
                    "caseID": "sub-r002s001",
                    "CBF": ["R002/sub-r002s001/ses-1/anat/cbf_b.nii.gz"],
                    "TMAX": ["R002/sub-r002s001/ses-1/anat/tmax_b.nii.gz"],
                    "label": "R002/sub-r002s001/ses-1/anat/label_b.nii.gz",
                },
                {
                    "fold": 2,
                    "caseID": "sub-r003s001",
                    "CBF": ["R003/sub-r003s001/ses-1/anat/cbf_c.nii.gz"],
                    "TMAX": ["R003/sub-r003s001/ses-1/anat/tmax_c.nii.gz"],
                    "label": "R003/sub-r003s001/ses-1/anat/label_c.nii.gz",
                },
            ]
        }
        datalist_path = base / "isles24.json"
        datalist_path.write_text(json.dumps(datalist), encoding="utf-8")
        return datalist_path

    def test_flatten_policy_length_multiplies_cases_by_patch_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            datalist_path = self._prepare_datalist(base)
            dataset = ISLES24RandomPatches3D(
                directory=str(base),
                datalist_json=str(datalist_path),
                fold=0,
                modalities=["CBF_min_0_max_70", "TMAX_min_4_max_30"],
                test_flag=False,
                preprocessing_configs=_build_preprocessing_configs(
                    roi_3d=[6, 5, 4],
                    patches_per_volume=3,
                    pad_to_divisible=True,
                ),
            )

            # fold=0 => train receives cases with fold 1 and 2.
            self.assertEqual(len(dataset), 6)

    def test_getitem_returns_single_patch_with_expected_shape_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            datalist_path = self._prepare_datalist(base)
            dataset = ISLES24RandomPatches3D(
                directory=str(base),
                datalist_json=str(datalist_path),
                fold=0,
                modalities=["CBF_min_0_max_70", "TMAX_min_4_max_30"],
                test_flag=False,
                preprocessing_configs=_build_preprocessing_configs(
                    roi_3d=[6, 5, 4],
                    patches_per_volume=2,
                    pad_to_divisible=True,
                ),
            )

            image, label, patch_path = dataset[1]
            self.assertEqual(tuple(image.shape), (2, 6, 5, 4))
            self.assertEqual(tuple(label.shape), (1, 6, 5, 4))
            self.assertEqual(patch_path, "sub-r002s001_patch1")


if __name__ == "__main__":
    unittest.main()
