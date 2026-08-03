"""Grand Challenge NIfTI transport boundary tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from scripts.gc_submission_builder.runtime.image_io import (
    ImageTransportError,
    inspect_nifti_input,
    write_nifti_prediction,
)
from scripts.gc_submission_builder.runtime.interfaces import OutputBinding
from src.inference.contracts import (
    NativeImageMetadata,
    PredictionResult,
    SpatialGeometry,
    SpatialTrace,
)


def _native_result(shape=(4, 5, 6)) -> PredictionResult:
    affine = (
        (0.0, -1.5, 0.0, 12.0),
        (2.0, 0.0, 0.0, -4.0),
        (0.0, 0.0, 3.0, 8.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    geometry = SpatialGeometry(
        shape=shape,
        affine=affine,
        spacing=(2.0, 1.5, 3.0),
        orientation="ALS",
    )
    metadata = NativeImageMetadata(
        canonical_key="T1",
        shape=shape,
        dtype="float32",
        affine=affine,
        spacing=geometry.spacing,
        orientation=geometry.orientation,
        qform=affine,
        sform=affine,
        qform_code=1,
        sform_code=2,
        source_reference="fixture-source",
    )
    probability = torch.full((1, 1, *shape), 0.75, dtype=torch.float32)
    return PredictionResult(
        probability=probability,
        mask=(probability >= 0.5).to(torch.uint8),
        output_space="native_input",
        spatial_trace=SpatialTrace(original=geometry, model=geometry),
        native_reference=metadata,
    )


class TestGcImageIo(unittest.TestCase):
    def test_inspect_nifti_input_accepts_one_finite_3d_nii_gz(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.nii.gz"
            nib.save(
                nib.Nifti1Image(
                    np.arange(60, dtype=np.float32).reshape(3, 4, 5),
                    np.diag([2.0, 3.0, 4.0, 1.0]),
                ),
                path,
            )
            inspected = inspect_nifti_input(path)

        self.assertEqual(inspected.shape, (3, 4, 5))
        self.assertEqual(inspected.dtype, "float32")
        self.assertEqual(inspected.orientation, "RAS")
        self.assertEqual(inspected.spacing, (2.0, 3.0, 4.0))

    def test_inspect_nifti_input_rejects_corrupt_and_4d_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corrupt = root / "corrupt.nii.gz"
            corrupt.write_bytes(b"not a nifti")
            with self.assertRaisesRegex(ImageTransportError, "could not be opened"):
                inspect_nifti_input(corrupt)

            four_d = root / "four-d.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((2, 3, 4, 2)), np.eye(4)), four_d)
            with self.assertRaisesRegex(ImageTransportError, "3D"):
                inspect_nifti_input(four_d)

    def test_write_nifti_prediction_reuses_native_writer_and_output_binding(self):
        binding = OutputBinding(
            slug="opaque-output",
            relative_path="images/fixture-segmentation",
            file_type="nifti",
        )
        result = _native_result()
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            validation = write_nifti_prediction(
                result,
                output_root=output_root,
                binding=binding,
            )
            output_path = output_root / "images" / "fixture-segmentation" / "output.nii.gz"
            reopened = nib.load(output_path)
            values = set(np.unique(np.asarray(reopened.dataobj)).tolist())

        self.assertEqual(Path(validation["path"]), output_path)
        self.assertEqual(reopened.shape, result.native_reference.shape)
        self.assertEqual(reopened.get_data_dtype(), np.dtype(np.uint8))
        self.assertEqual(values, {1})
        np.testing.assert_allclose(reopened.affine, result.native_reference.affine)

    def test_write_nifti_prediction_rejects_non_native_and_unsafe_output(self):
        binding = OutputBinding(
            slug="opaque-output",
            relative_path="images/fixture-segmentation",
            file_type="nifti",
        )
        result = _native_result()
        model_result = PredictionResult(
            probability=result.probability,
            mask=result.mask,
            output_space="model_preprocessed",
            spatial_trace=result.spatial_trace,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ImageTransportError, "native_input"):
                write_nifti_prediction(
                    model_result,
                    output_root=Path(tmp),
                    binding=binding,
                )
            with self.assertRaisesRegex(ImageTransportError, "safe relative path"):
                write_nifti_prediction(
                    result,
                    output_root=Path(tmp),
                    binding=OutputBinding(
                        slug="opaque-output",
                        relative_path="../escape",
                        file_type="nifti",
                    ),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
