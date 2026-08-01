import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from src.data.loader_stack.isles24_loader import (
    ISLES24Dataset2D,
    ISLES24Dataset3D,
    datafold_read,
)
from src.data.loader_stack.contracts import InvalidCaseRecordError


def _write_nifti(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array.astype(np.float32), np.eye(4)), str(path))


def _preprocessing_configs() -> dict:
    return {
        "common": {
            "orientation": {"enabled": False, "axcodes": "RAS"},
            "spacing": {
                "enabled": False,
                "pixdim": [1.0, 1.0, 1.0],
                "interpolation": {"image": "bilinear", "label": "nearest"},
            },
        },
        "roi": {"volume_3d": [3, 4, 2], "slice_2d": [4, 4]},
        "full_volumes_3d": {"pad_to_divisible": False},
    }


class TestIsles24LabelOptional(unittest.TestCase):
    def _prepare_case(self, base: Path) -> Path:
        _write_nifti(
            base / "case/cbf.nii.gz",
            np.linspace(0.0, 70.0, 24, dtype=np.float32).reshape(3, 4, 2),
        )
        datalist_path = base / "isles24.json"
        datalist_path.write_text(
            json.dumps(
                {
                    "training": [
                        {
                            "caseID": "case-label-free",
                            "fold": 0,
                            "split": "test",
                            "CBF": ["case/cbf.nii.gz"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return datalist_path

    def test_datafold_label_requirement_defaults_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            datalist_path = self._prepare_case(base)

            with self.assertRaisesRegex(InvalidCaseRecordError, "load_labels=True"):
                datafold_read(
                    datalist=str(datalist_path),
                    basedir=str(base),
                    fold=0,
                    subset_name="val",
                    partitioning="fold",
                )

            records = datafold_read(
                datalist=str(datalist_path),
                basedir=str(base),
                fold=0,
                subset_name="val",
                partitioning="fold",
                load_labels=False,
            )
            self.assertEqual(len(records), 1)
            self.assertNotIn("label", records[0])

    def test_full_volume_and_slice_loaders_return_no_dummy_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            datalist_path = self._prepare_case(base)

            volume_dataset = ISLES24Dataset3D(
                directory=str(base),
                datalist_json=str(datalist_path),
                fold=0,
                subset_name="val",
                partitioning="fold",
                modalities=["CBF_min_0_max_70"],
                test_flag=True,
                load_labels=False,
                preprocessing_configs=_preprocessing_configs(),
            )
            volume, case_id = volume_dataset[0]
            self.assertEqual(tuple(volume.shape), (1, 3, 4, 2))
            self.assertEqual(case_id, "case-label-free")

            slice_dataset = ISLES24Dataset2D(
                directory=str(base),
                datalist_json=str(datalist_path),
                fold=0,
                subset_name="val",
                partitioning="fold",
                modalities=["CBF_RAW"],
                test_flag=True,
                load_labels=False,
                image_size=4,
                preprocessing_configs=_preprocessing_configs(),
            )
            image, virtual_path = slice_dataset[0]
            self.assertEqual(tuple(image.shape), (1, 4, 4))
            self.assertEqual(virtual_path, "case-label-free_slice0")


if __name__ == "__main__":
    unittest.main()
