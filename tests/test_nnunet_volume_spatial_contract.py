import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor
from torch.utils.data import Subset

from scripts.evaluation.core.evaluation_pipeline import _evaluate_volume_sample
from scripts.nnunet.core.exporters import VolumeExportStrategy
from scripts.nnunet.core.io_adapters import iter_nnunet_volume_samples


class _SyntheticVolumeDataset:
    def __init__(self, source_path: Path, image: MetaTensor, label: MetaTensor):
        self.database = [{"T1": str(source_path), "caseID": "case001"}]
        self.base_modalities = ["T1"]
        self.modalities = ["T1_RAW"]
        self.image = image
        self.label = label

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.image, self.label, "case001"


def _write_nifti(path: Path, shape, affine, value=0.0):
    data = np.full(shape, value, dtype=np.float32)
    image = nib.Nifti1Image(data, affine=np.asarray(affine, dtype=np.float64))
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, path)


class TestVolumeExportSpatialContract(unittest.TestCase):
    def test_model_preprocessed_export_writes_transformed_tensor_affine(self):
        source_affine = np.diag([2.0, 3.0, 4.0, 1.0])
        export_affine = np.array(
            [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 20.0],
                [0.0, 0.0, 1.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source_path = base / "source.nii.gz"
            images_dir = base / "images"
            labels_dir = base / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()
            _write_nifti(source_path, (4, 5, 6), source_affine)
            image = MetaTensor(torch.zeros((1, 3, 4, 5)), affine=export_affine)
            label = MetaTensor(torch.zeros((1, 3, 4, 5)), affine=export_affine)
            dataset = _SyntheticVolumeDataset(source_path, image, label)

            record = VolumeExportStrategy("model_preprocessed")._process_single_case(
                0, dataset, images_dir, labels_dir, 1, "test"
            )

            exported = nib.load(images_dir / "case001_0000.nii.gz")
            self.assertTrue(np.allclose(exported.affine, export_affine))
            self.assertEqual(exported.shape, (3, 4, 5))
            qform, qform_code = exported.get_qform(coded=True)
            sform, sform_code = exported.get_sform(coded=True)
            self.assertEqual(int(qform_code), 1)
            self.assertEqual(int(sform_code), 1)
            self.assertTrue(np.allclose(qform, export_affine))
            self.assertTrue(np.allclose(sform, export_affine))
            self.assertEqual(record["export_space"], "model_preprocessed")
            self.assertTrue(np.allclose(record["source_affine"], source_affine))
            self.assertTrue(np.allclose(record["export_affine"], export_affine))
            self.assertEqual(record["export_spacing_xyz"], [1.0, 1.0, 1.0])

    def test_native_export_preserves_source_grid(self):
        native_affine = np.array(
            [
                [1.5, 0.0, 0.0, 4.0],
                [0.0, 2.0, 0.0, 5.0],
                [0.0, 0.0, 3.0, 6.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source_path = base / "source.nii.gz"
            images_dir = base / "images"
            labels_dir = base / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()
            _write_nifti(source_path, (3, 4, 5), native_affine)
            image = MetaTensor(torch.zeros((1, 3, 4, 5)), affine=native_affine)
            label = MetaTensor(torch.zeros((1, 3, 4, 5)), affine=native_affine)
            dataset = _SyntheticVolumeDataset(source_path, image, label)

            record = VolumeExportStrategy("native_input")._process_single_case(
                0, dataset, images_dir, labels_dir, 1, "test"
            )

            exported = nib.load(labels_dir / "case001.nii.gz")
            self.assertTrue(np.allclose(exported.affine, native_affine))
            self.assertEqual(record["export_space"], "native_input")

    def test_source_geometry_resolves_through_subset_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source_path = base / "source.nii.gz"
            images_dir = base / "images"
            labels_dir = base / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()
            _write_nifti(source_path, (3, 4, 5), np.eye(4))
            image = MetaTensor(torch.zeros((1, 3, 4, 5)), affine=np.eye(4))
            label = MetaTensor(torch.zeros((1, 3, 4, 5)), affine=np.eye(4))
            wrapped = Subset(_SyntheticVolumeDataset(source_path, image, label), [0])

            record = VolumeExportStrategy("native_input")._process_single_case(
                0, wrapped, images_dir, labels_dir, 1, "test"
            )

            self.assertEqual(record["source_path"], str(source_path))

    def test_native_declaration_rejects_changed_grid(self):
        source_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        export_affine = np.eye(4)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source_path = base / "source.nii.gz"
            images_dir = base / "images"
            labels_dir = base / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()
            _write_nifti(source_path, (4, 5, 6), source_affine)
            image = MetaTensor(torch.zeros((1, 3, 4, 5)), affine=export_affine)
            label = MetaTensor(torch.zeros((1, 3, 4, 5)), affine=export_affine)
            dataset = _SyntheticVolumeDataset(source_path, image, label)

            with self.assertRaisesRegex(ValueError, "native_input.*inconsistent"):
                VolumeExportStrategy("native_input")._process_single_case(
                    0, dataset, images_dir, labels_dir, 1, "test"
                )

    def test_image_label_affine_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source_path = base / "source.nii.gz"
            images_dir = base / "images"
            labels_dir = base / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()
            _write_nifti(source_path, (3, 4, 5), np.eye(4))
            image = MetaTensor(torch.zeros((1, 3, 4, 5)), affine=np.eye(4))
            label_affine = np.eye(4)
            label_affine[0, 3] = 5.0
            label = MetaTensor(torch.zeros((1, 3, 4, 5)), affine=label_affine)
            dataset = _SyntheticVolumeDataset(source_path, image, label)

            with self.assertRaisesRegex(ValueError, "image/label affine mismatch"):
                VolumeExportStrategy("model_preprocessed")._process_single_case(
                    0, dataset, images_dir, labels_dir, 1, "test"
                )


class TestNnunetVolumeProducerSpatialContract(unittest.TestCase):
    def test_matching_pair_declares_space_and_geometry(self):
        affine = np.diag([1.0, 2.0, 3.0, 1.0])
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pred_dir = base / "pred"
            gt_dir = base / "gt"
            pred_dir.mkdir()
            gt_dir.mkdir()
            _write_nifti(pred_dir / "case001.nii.gz", (3, 4, 5), affine)
            _write_nifti(gt_dir / "case001.nii.gz", (3, 4, 5), affine)

            samples = list(
                iter_nnunet_volume_samples(
                    pred_dir, gt_dir, volume_space="model_preprocessed"
                )
            )

            self.assertEqual(len(samples), 1)
            sample = samples[0]
            self.assertEqual(sample.prediction_space, "model_preprocessed")
            self.assertEqual(sample.reference_space, "model_preprocessed")
            self.assertEqual(sample.prediction_geometry.spacing, (1.0, 2.0, 3.0))
            self.assertEqual(sample.metadata["source"], "nnunet_volumes")

            records = _evaluate_volume_sample(
                sample=sample,
                thresholds=[0.5],
                metric_names=["DiceNativeCoefficient"],
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].metadata["source"], "nnunet_volumes")
            self.assertEqual(
                records[0].metadata["prediction_space"],
                "model_preprocessed",
            )

    def test_equal_shape_different_affine_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pred_dir = base / "pred"
            gt_dir = base / "gt"
            pred_dir.mkdir()
            gt_dir.mkdir()
            _write_nifti(pred_dir / "case001.nii.gz", (3, 4, 5), np.eye(4))
            gt_affine = np.eye(4)
            gt_affine[0, 3] = 10.0
            _write_nifti(gt_dir / "case001.nii.gz", (3, 4, 5), gt_affine)

            with self.assertRaisesRegex(ValueError, "Affine mismatch"):
                list(
                    iter_nnunet_volume_samples(
                        pred_dir, gt_dir, volume_space="native_input"
                    )
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
