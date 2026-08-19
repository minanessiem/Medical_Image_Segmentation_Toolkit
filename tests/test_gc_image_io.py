"""Grand Challenge NIfTI transport boundary tests."""

from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch

from scripts.gc_submission_builder.runtime.image_io import (
    ImageTransportError,
    canonicalize_image_inputs,
    inspect_nifti_input,
    materialize_prediction_outputs,
    write_mha_prediction,
    write_nifti_prediction,
)
from scripts.gc_submission_builder.runtime.interfaces import (
    OutputBinding,
    ResolvedImageInput,
)
from src.inference.contracts import (
    NativeImageMetadata,
    PredictionResult,
    SpatialGeometry,
    SpatialTrace,
)


def _native_result(shape=(4, 5, 6), affine=None) -> PredictionResult:
    affine = affine or (
        (0.0, -1.5, 0.0, 12.0),
        (2.0, 0.0, 0.0, -4.0),
        (0.0, 0.0, 3.0, 8.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    affine_array = np.asarray(affine, dtype=np.float64)
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(affine_array))
    orientation = "".join(nib.aff2axcodes(affine_array))
    geometry = SpatialGeometry(
        shape=shape,
        affine=affine,
        spacing=spacing,
        orientation=orientation,
    )
    metadata = NativeImageMetadata(
        canonical_key="T1",
        shape=shape,
        dtype="float32",
        affine=affine,
        spacing=spacing,
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


def _resolved_image(path: Path, *, source_format: str) -> ResolvedImageInput:
    return ResolvedImageInput(
        slug="opaque-image-socket",
        dataset_key="T1",
        source_path=path.resolve(),
        source_format=source_format,
        source_size_bytes=path.stat().st_size,
        observed_format_counts=MappingProxyType({source_format: 1}),
        other_regular_file_count=0,
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

    def test_canonicalize_mha_preserves_voxels_and_oblique_physical_geometry(self):
        angle = np.deg2rad(17.0)
        direction = (
            float(np.cos(angle)),
            float(-np.sin(angle)),
            0.0,
            float(np.sin(angle)),
            float(np.cos(angle)),
            0.0,
            0.0,
            0.0,
            1.0,
        )
        array = np.arange(5 * 6 * 7, dtype=np.float32).reshape(7, 6, 5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "platform-generated.mha"
            source = sitk.GetImageFromArray(array)
            source.SetSpacing((1.25, 2.5, 3.75))
            source.SetOrigin((-14.0, 9.0, 3.5))
            source.SetDirection(direction)
            sitk.WriteImage(source, str(source_path), useCompression=True)

            with canonicalize_image_inputs(
                {"T1": _resolved_image(source_path, source_format="mha")},
                scratch_root=root,
            ) as canonicalized:
                canonical = canonicalized.inputs["T1"]
                canonical_path = canonical.canonical_path
                self.assertTrue(canonical_path.is_file())
                self.assertTrue(canonical.converted)
                self.assertEqual(canonical.source_inspection.shape, (5, 6, 7))
                self.assertEqual(canonical.canonical_inspection.shape, (5, 6, 7))
                reopened = sitk.ReadImage(str(canonical_path))
                np.testing.assert_array_equal(sitk.GetArrayFromImage(reopened), array)
                np.testing.assert_allclose(reopened.GetSpacing(), source.GetSpacing())
                np.testing.assert_allclose(reopened.GetOrigin(), source.GetOrigin())
                np.testing.assert_allclose(reopened.GetDirection(), source.GetDirection())

            self.assertFalse(canonical_path.exists())
            self.assertFalse(any(root.glob("gc-input-*")))

    def test_canonicalize_full_volume_mha_allows_nifti_header_rounding(self):
        angle = np.deg2rad(17.0)
        direction = (
            float(np.cos(angle)),
            float(-np.sin(angle)),
            0.0,
            float(np.sin(angle)),
            float(np.cos(angle)),
            0.0,
            0.0,
            0.0,
            1.0,
        )
        size = (256, 120, 256)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "full-volume-platform-generated.mha"
            source = sitk.Image(list(size), sitk.sitkUInt8)
            source.SetSpacing((0.9375037, 0.9374981, 3.0000047))
            source.SetOrigin((-126.4, -97.2, 42.1))
            source.SetDirection(direction)
            sitk.WriteImage(source, str(source_path), useCompression=True)

            with canonicalize_image_inputs(
                {"T1": _resolved_image(source_path, source_format="mha")},
                scratch_root=root,
            ) as canonicalized:
                canonical = canonicalized.inputs["T1"]
                reopened = sitk.ReadImage(str(canonical.canonical_path))
                far_corner = tuple(value - 1 for value in size)
                source_point = source.TransformIndexToPhysicalPoint(far_corner)
                canonical_point = reopened.TransformIndexToPhysicalPoint(far_corner)
                drift_mm = max(
                    abs(observed - expected)
                    for observed, expected in zip(canonical_point, source_point)
                )
                self.assertGreater(drift_mm, 1e-5)
                self.assertLess(drift_mm, 1e-3)
                self.assertEqual(sitk.Hash(reopened), sitk.Hash(source))

    def test_canonicalize_uncompressed_nifti_is_lossless_and_gz_input_passes_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            array = np.arange(60, dtype=np.int16).reshape(3, 4, 5)
            affine = np.diag([2.0, 3.0, 4.0, 1.0])
            nii_path = root / "case.nii"
            nii_gz_path = root / "case.nii.gz"
            nib.save(nib.Nifti1Image(array, affine), nii_path)
            nib.save(nib.Nifti1Image(array, affine), nii_gz_path)

            with canonicalize_image_inputs(
                {"T1": _resolved_image(nii_path, source_format="nii")},
                scratch_root=root,
            ) as canonicalized:
                canonical_path = canonicalized.inputs["T1"].canonical_path
                self.assertEqual(
                    gzip.decompress(canonical_path.read_bytes()),
                    nii_path.read_bytes(),
                )
            self.assertFalse(canonical_path.exists())

            with canonicalize_image_inputs(
                {"T1": _resolved_image(nii_gz_path, source_format="nii_gz")},
                scratch_root=root,
            ) as canonicalized:
                canonical = canonicalized.inputs["T1"]
                self.assertEqual(canonical.canonical_path, nii_gz_path.resolve())
                self.assertFalse(canonical.converted)

    def test_canonicalize_scalar_3d_tiff_preserves_grid_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "platform-generated.tif"
            array = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(6, 5, 4)
            source = sitk.GetImageFromArray(array)
            source.SetSpacing((1.5, 2.0, 2.5))
            source.SetOrigin((-3.0, 4.0, 5.0))
            sitk.WriteImage(source, str(source_path))
            platform_source = sitk.ReadImage(str(source_path))

            with canonicalize_image_inputs(
                {"T1": _resolved_image(source_path, source_format="tif")},
                scratch_root=root,
            ) as canonicalized:
                canonical = canonicalized.inputs["T1"]
                reopened = sitk.ReadImage(str(canonical.canonical_path))
                np.testing.assert_array_equal(sitk.GetArrayFromImage(reopened), array)
                np.testing.assert_allclose(
                    reopened.GetSpacing(), platform_source.GetSpacing()
                )
                np.testing.assert_allclose(reopened.GetOrigin(), platform_source.GetOrigin())

    def test_canonicalize_rejects_vector_or_non_3d_platform_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vector_path = root / "vector.mha"
            vector = sitk.GetImageFromArray(
                np.zeros((4, 5, 6, 2), dtype=np.float32),
                isVector=True,
            )
            sitk.WriteImage(vector, str(vector_path))
            with self.assertRaisesRegex(ImageTransportError, "scalar"):
                with canonicalize_image_inputs(
                    {"T1": _resolved_image(vector_path, source_format="mha")},
                    scratch_root=root,
                ):
                    pass

            two_d_path = root / "two-d.tif"
            sitk.WriteImage(sitk.Image([6, 5], sitk.sitkUInt8), str(two_d_path))
            with self.assertRaisesRegex(ImageTransportError, "must be 3D"):
                with canonicalize_image_inputs(
                    {"T1": _resolved_image(two_d_path, source_format="tif")},
                    scratch_root=root,
                ):
                    pass

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
            result_key="mask",
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
            result_key="mask",
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
                        result_key="mask",
                        relative_path="../escape",
                        file_type="nifti",
                    ),
                )

    def test_write_mha_outputs_preserves_oblique_native_geometry_and_values(self):
        angle = np.deg2rad(23.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = rotation @ np.diag([1.25, 2.5, 3.75])
        affine[:3, 3] = [17.0, -11.0, 5.5]
        result = _native_result(affine=tuple(tuple(row) for row in affine))
        probability = torch.linspace(
            0.0,
            1.0,
            result.probability.numel(),
            dtype=torch.float32,
        ).reshape_as(result.probability)
        result = PredictionResult(
            probability=probability,
            mask=(probability >= 0.5).to(torch.uint8),
            output_space=result.output_space,
            spatial_trace=result.spatial_trace,
            native_reference=result.native_reference,
        )
        bindings = (
            OutputBinding(
                slug="segmentation",
                result_key="mask",
                relative_path="images/segmentation",
                file_type="mha",
            ),
            OutputBinding(
                slug="probability",
                result_key="probability",
                relative_path="images/probability",
                file_type="mha",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            validations = materialize_prediction_outputs(
                result,
                output_root=output_root,
                bindings=bindings,
            )
            mask_path = output_root / "images" / "segmentation" / "output.mha"
            probability_path = output_root / "images" / "probability" / "output.mha"
            mask_image = sitk.ReadImage(str(mask_path))
            probability_image = sitk.ReadImage(str(probability_path))
            mask = sitk.GetArrayFromImage(mask_image).transpose(2, 1, 0)
            probability = sitk.GetArrayFromImage(probability_image).transpose(2, 1, 0)
            header = probability_path.read_bytes().split(b"ElementDataFile", 1)[0]

        self.assertEqual(tuple(validations), ("segmentation", "probability"))
        self.assertIn(b"CompressedData = True", header)
        self.assertEqual(mask.dtype, np.dtype(np.uint8))
        self.assertEqual(probability.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(mask, result.mask[0, 0].numpy())
        np.testing.assert_allclose(probability, result.probability[0, 0].numpy())
        np.testing.assert_array_equal(mask, probability >= 0.5)
        self.assertEqual(mask_image.GetSize(), result.native_reference.shape)
        self.assertEqual(mask_image.GetSize(), probability_image.GetSize())
        self.assertEqual(mask_image.GetSpacing(), probability_image.GetSpacing())
        self.assertEqual(mask_image.GetOrigin(), probability_image.GetOrigin())
        self.assertEqual(mask_image.GetDirection(), probability_image.GetDirection())

        lps_from_ras = np.diag([-1.0, -1.0, 1.0])
        direction = np.asarray(mask_image.GetDirection()).reshape(3, 3)
        reconstructed_lps = np.eye(4)
        reconstructed_lps[:3, :3] = direction @ np.diag(mask_image.GetSpacing())
        reconstructed_lps[:3, 3] = mask_image.GetOrigin()
        expected_lps = np.eye(4)
        expected_lps[:3, :] = lps_from_ras @ affine[:3, :]
        np.testing.assert_allclose(reconstructed_lps, expected_lps, atol=1e-5)

    def test_mha_writer_rejects_sheared_native_geometry(self):
        affine = np.eye(4, dtype=np.float64)
        affine[0, 1] = 0.25
        result = _native_result(affine=tuple(tuple(row) for row in affine))
        binding = OutputBinding(
            slug="probability",
            result_key="probability",
            relative_path="images/probability",
            file_type="mha",
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ImageTransportError, "shear|orthonormal"):
                write_mha_prediction(
                    result,
                    output_root=Path(tmp),
                    binding=binding,
                )

    def test_mha_writer_preserves_axis_flip_geometry(self):
        affine = np.array(
            [
                [-1.5, 0.0, 0.0, 21.0],
                [0.0, 2.25, 0.0, -13.0],
                [0.0, 0.0, 3.5, 4.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        result = _native_result(affine=tuple(tuple(row) for row in affine))
        binding = OutputBinding(
            slug="segmentation",
            result_key="mask",
            relative_path="images/segmentation",
            file_type="mha",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "images/segmentation/output.mha"
            write_mha_prediction(result, output_root=Path(tmp), binding=binding)
            image = sitk.ReadImage(str(path))

        lps_from_ras = np.diag([-1.0, -1.0, 1.0])
        direction = np.asarray(image.GetDirection()).reshape(3, 3)
        observed = np.eye(4)
        observed[:3, :3] = direction @ np.diag(image.GetSpacing())
        observed[:3, 3] = image.GetOrigin()
        expected = np.eye(4)
        expected[:3, :] = lps_from_ras @ affine[:3, :]
        np.testing.assert_allclose(observed, expected, atol=1e-5)
        self.assertLess(np.linalg.det(direction), 0)

    def test_complete_output_set_removes_partial_artifacts_on_failure(self):
        result = _native_result()
        bindings = (
            OutputBinding(
                slug="mask",
                result_key="mask",
                relative_path="images/mask",
                file_type="nifti",
            ),
            OutputBinding(
                slug="unsupported",
                result_key="probability",
                relative_path="images/probability",
                file_type="nifti",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            stale = output_root / "images" / "mask" / "output.nii.gz"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"stale")
            with self.assertRaisesRegex(ImageTransportError, "probability.*NIfTI"):
                materialize_prediction_outputs(
                    result,
                    output_root=output_root,
                    bindings=bindings,
                )

            self.assertFalse(stale.exists())
            self.assertFalse(any(output_root.glob(".gc-staging-*")))

    def test_repeated_materialization_replaces_stale_declared_outputs(self):
        binding = OutputBinding(
            slug="segmentation",
            result_key="mask",
            relative_path="images/segmentation",
            file_type="mha",
        )
        first = _native_result()
        second = PredictionResult(
            probability=torch.zeros_like(first.probability),
            mask=torch.zeros_like(first.mask),
            output_space=first.output_space,
            spatial_trace=first.spatial_trace,
            native_reference=first.native_reference,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            materialize_prediction_outputs(first, output_root=root, bindings=(binding,))
            materialize_prediction_outputs(second, output_root=root, bindings=(binding,))
            reopened = sitk.GetArrayFromImage(
                sitk.ReadImage(str(root / "images/segmentation/output.mha"))
            )

        self.assertEqual(set(np.unique(reopened)), {0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
