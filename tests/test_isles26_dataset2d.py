import json
import tempfile
import threading
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from src.data.loader_stack.isles26_loader import ISLES26Dataset2D


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


class TestIsles26Dataset2D(unittest.TestCase):
    def test_label_free_dataset_returns_image_and_identity_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            volume_path = base / "case/t1.nii.gz"
            _write_nifti(
                volume_path,
                np.arange(24, dtype=np.float32).reshape(3, 4, 2),
            )
            datalist_path = base / "isles26.json"
            datalist_path.write_text(
                json.dumps(
                    {
                        "training": [
                            {
                                "split": "test",
                                "caseID": "case-label-free",
                                "T1": ["case/t1.nii.gz"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            dataset = ISLES26Dataset2D(
                directory=str(base),
                datalist_json=str(datalist_path),
                subset_name="test",
                modalities=["T1_RAW"],
                test_flag=False,
                load_labels=False,
                image_size=4,
                preprocessing_configs=_build_preprocessing_configs(image_size=4),
            )

            image, virtual_path = dataset[0]
            self.assertEqual(tuple(image.shape), (1, 4, 4))
            self.assertEqual(virtual_path, "case-label-free_slice0")
            self.assertFalse(dataset.load_labels)
            self.assertFalse(dataset.test_flag)

    def test_dataset2d_train_split_returns_slice_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            case_a_t1 = np.stack(
                [
                    np.full((4, 4), 1.0, dtype=np.float32),
                    np.full((4, 4), 2.0, dtype=np.float32),
                    np.full((4, 4), 3.0, dtype=np.float32),
                ],
                axis=-1,
            )
            case_a_label = np.stack(
                [
                    np.full((4, 4), 0.0, dtype=np.float32),
                    np.full((4, 4), 1.0, dtype=np.float32),
                    np.full((4, 4), 0.0, dtype=np.float32),
                ],
                axis=-1,
            )
            case_b_t1 = np.stack(
                [
                    np.full((4, 4), 10.0, dtype=np.float32),
                    np.full((4, 4), 11.0, dtype=np.float32),
                    np.full((4, 4), 12.0, dtype=np.float32),
                ],
                axis=-1,
            )
            case_b_label = np.stack(
                [
                    np.full((4, 4), 1.0, dtype=np.float32),
                    np.full((4, 4), 2.0, dtype=np.float32),
                    np.full((4, 4), 3.0, dtype=np.float32),
                ],
                axis=-1,
            )

            _write_nifti(base / "R001/sub-r001s001/ses-1/anat/t1_a.nii.gz", case_a_t1)
            _write_nifti(base / "R001/sub-r001s001/ses-1/anat/label_a.nii.gz", case_a_label)
            _write_nifti(base / "R002/sub-r002s001/ses-1/anat/t1_b.nii.gz", case_b_t1)
            _write_nifti(base / "R002/sub-r002s001/ses-1/anat/label_b.nii.gz", case_b_label)

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
                        "siteID": "R002",
                        "metadata_csv": "R002/sub-r002s001/ses-1/anat/meta.csv",
                        "metadata": {"SITE": "R002"},
                        "T1": ["R002/sub-r002s001/ses-1/anat/t1_b.nii.gz"],
                        "label": "R002/sub-r002s001/ses-1/anat/label_b.nii.gz",
                    },
                ]
            }
            datalist_path = base / "isles26.json"
            datalist_path.write_text(json.dumps(datalist), encoding="utf-8")

            dataset = ISLES26Dataset2D(
                directory=str(base),
                datalist_json=str(datalist_path),
                fold=0,
                subset_name="train",
                modalities=["T1_RAW"],
                test_flag=False,
                image_size=4,
                use_caching=False,
                preprocessing_configs=_build_preprocessing_configs(image_size=4),
            )

            self.assertEqual(len(dataset), 3)
            image, label, virtual_path = dataset[1]
            self.assertEqual(tuple(image.shape), (1, 4, 4))
            self.assertEqual(tuple(label.shape), (1, 4, 4))
            self.assertEqual(virtual_path, "sub-r002s001_slice1")
            self.assertAlmostEqual(float(image[0, 0, 0].item()), 11.0)
            self.assertAlmostEqual(float(label[0, 0, 0].item()), 2.0)

    def test_dataset2d_metadata_passthrough_and_shared_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            volume = np.stack(
                [
                    np.full((3, 3), 5.0, dtype=np.float32),
                    np.full((3, 3), 6.0, dtype=np.float32),
                ],
                axis=-1,
            )
            label = np.stack(
                [
                    np.full((3, 3), 0.0, dtype=np.float32),
                    np.full((3, 3), 1.0, dtype=np.float32),
                ],
                axis=-1,
            )
            _write_nifti(base / "R001/sub-r001s001/ses-1/anat/t1.nii.gz", volume)
            _write_nifti(base / "R001/sub-r001s001/ses-1/anat/label.nii.gz", label)

            datalist = {
                "training": [
                    {
                        "split": "val_fast",
                        "caseID": "sub-r001s001",
                        "siteID": "R001",
                        "metadata_csv": "R001/sub-r001s001/ses-1/anat/meta.csv",
                        "metadata": {"SITE": "R001", "SESSION_ID": "sub-r001s001_ses-1"},
                        "T1": ["R001/sub-r001s001/ses-1/anat/t1.nii.gz"],
                        "label": "R001/sub-r001s001/ses-1/anat/label.nii.gz",
                    }
                ]
            }
            datalist_path = base / "isles26.json"
            datalist_path.write_text(json.dumps(datalist), encoding="utf-8")

            shared_cache = {}
            dataset = ISLES26Dataset2D(
                directory=str(base),
                datalist_json=str(datalist_path),
                fold=0,
                subset_name="val_fast",
                modalities=["T1_RAW"],
                test_flag=True,
                image_size=3,
                use_caching=True,
                shared_cache=shared_cache,
                cache_lock=threading.Lock(),
                preprocessing_configs=_build_preprocessing_configs(image_size=3),
            )
            dataset.return_metadata = True

            image, label_tensor, virtual_path, metadata = dataset[0]
            self.assertEqual(tuple(image.shape), (1, 3, 3))
            self.assertEqual(tuple(label_tensor.shape), (1, 3, 3))
            self.assertEqual(virtual_path, "sub-r001s001_slice0")
            self.assertEqual(metadata["siteID"], "R001")
            self.assertEqual(metadata["metadata"]["SESSION_ID"], "sub-r001s001_ses-1")

            # Shared cache stores preprocessed case-level tensors keyed per-case.
            cache_key = ("subset:val_fast", 0)
            self.assertIn(cache_key, shared_cache)
            self.assertIn("image", shared_cache[cache_key])
            self.assertIn("label", shared_cache[cache_key])


if __name__ == "__main__":
    unittest.main()
