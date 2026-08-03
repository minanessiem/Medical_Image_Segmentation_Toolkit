"""Case-level output-space, thresholding, and native NIfTI contracts."""

from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor

from src.inference.case_producer import build_case_producer
from src.inference.contracts import (
    InvalidPredictionError,
    NativeImageMetadata,
    PreprocessedCase,
    SpatialGeometry,
    SpatialRestorationError,
    SpatialTrace,
)
from src.inference.output import write_native_prediction_mask
from src.inference.pipeline import predict_preprocessed_case


def _affine_tuple(affine):
    return tuple(tuple(float(value) for value in row) for row in np.asarray(affine))


def _geometry(shape, affine):
    affine = np.asarray(affine, dtype=np.float64)
    return SpatialGeometry(
        shape=tuple(shape),
        affine=_affine_tuple(affine),
        spacing=tuple(float(value) for value in nib.affines.voxel_sizes(affine)),
        orientation="".join(str(value) for value in nib.aff2axcodes(affine)),
    )


def _metadata(shape, affine, *, qform=None, sform=None, qform_code=1, sform_code=2):
    qform = affine if qform is None else qform
    sform = affine if sform is None else sform
    geometry = _geometry(shape, affine)
    return NativeImageMetadata(
        canonical_key="T1",
        shape=geometry.shape,
        dtype="float32",
        affine=geometry.affine,
        spacing=geometry.spacing,
        orientation=geometry.orientation,
        qform=_affine_tuple(qform),
        sform=_affine_tuple(sform),
        qform_code=qform_code,
        sform_code=sform_code,
        source_reference="synthetic-reference",
    )


def _case(model_shape, model_affine, native_shape, native_affine):
    native = _metadata(native_shape, native_affine)
    model_geometry = _geometry(model_shape, model_affine)
    image = MetaTensor(torch.zeros((1, 1, *model_shape)), affine=torch.as_tensor(model_affine))
    return PreprocessedCase(
        case_id="case-1",
        image=image,
        spatial_trace=SpatialTrace(
            original=native.geometry,
            model=model_geometry,
            transform_history=({"class": "SpatialResample", "do_transforms": True},),
        ),
        native_metadata={"T1": native},
        reference_key="T1",
    )


def _write_nifti(path, data, affine):
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=2)
    nib.save(image, str(path))


def _dataset_cfg():
    return {
        "modalities": ["T1_RAW"],
        "preprocessing_configs": {
            "common": {
                "orientation": {"enabled": True, "axcodes": "RAS"},
                "spacing": {
                    "enabled": True,
                    "allow_native_spacing": False,
                    "pixdim": [1.0, 1.0, 1.0],
                    "interpolation": {"image": "bilinear", "label": "nearest"},
                },
            },
            "roi": {"volume_3d": [5, 7, 3], "slice_2d": [5, 7]},
            "online_slices_3d_to_2d": {
                "slice_axis": 2,
                "slice_order": "sequential",
            },
            "full_volumes_3d": {"pad_to_divisible": False},
        },
    }


class DummyExecutor:
    def __init__(self, probability, *, output_space, threshold=0.5):
        self.probability = probability
        self.policy = SimpleNamespace(
            output_space=output_space,
            decision=SimpleNamespace(threshold=threshold),
        )
        self.policy_source = "explicit_top_level"

    def __call__(self, image, progress_label=None, show_window_progress=True):
        del image, progress_label, show_window_progress
        return self.probability.clone()


class TestInferenceNativeOutput(unittest.TestCase):
    def test_model_space_result_remains_on_model_grid(self):
        case = _case((3, 3, 3), np.eye(4), (5, 5, 5), np.diag([0.5, 0.5, 0.5, 1.0]))
        probability = torch.linspace(0.0, 1.0, 27).reshape(1, 1, 3, 3, 3)

        result = predict_preprocessed_case(
            DummyExecutor(probability, output_space="model_preprocessed"),
            case,
            show_window_progress=False,
        )

        self.assertEqual(result.output_space, "model_preprocessed")
        torch.testing.assert_close(result.probability, probability)
        self.assertEqual(tuple(result.mask.shape), tuple(probability.shape))
        self.assertFalse(result.provenance["spatial_restoration"]["applied"])
        self.assertEqual(result.provenance["output_space"], "model_preprocessed")
        self.assertTrue(
            result.provenance["spatial_validation"][
                "shape_matches_declared_geometry"
            ]
        )
        self.assertFalse(
            result.provenance["spatial_validation"][
                "native_world_coordinates_validated"
            ]
        )

    def test_native_result_restores_probability_then_thresholds(self):
        case = _case(
            (3, 1, 1),
            np.diag([2.0, 1.0, 1.0, 1.0]),
            (5, 1, 1),
            np.eye(4),
        )
        probability = torch.tensor([0.2, 0.4, 0.6]).reshape(1, 1, 3, 1, 1)

        result = predict_preprocessed_case(
            DummyExecutor(probability, output_space="native_input", threshold=0.5),
            case,
            show_window_progress=False,
        )

        expected = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6]).reshape(1, 1, 5, 1, 1)
        torch.testing.assert_close(result.probability, expected, rtol=0, atol=1e-5)
        torch.testing.assert_close(
            result.mask,
            torch.tensor([0, 0, 0, 1, 1], dtype=torch.uint8).reshape(1, 1, 5, 1, 1),
        )
        self.assertEqual(result.output_space, "native_input")
        self.assertIs(result.native_reference, case.native_metadata["T1"])
        self.assertTrue(result.provenance["spatial_restoration"]["applied"])
        self.assertTrue(
            result.provenance["spatial_validation"][
                "native_world_coordinates_validated"
            ]
        )

    def test_finalized_case_producer_trace_restores_and_writes_native_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image_path = base / "t1.nii.gz"
            label_path = base / "label.nii.gz"
            native_affine = np.array(
                [
                    [-1.4, 0.0, 0.0, 12.0],
                    [0.0, 2.1, 0.0, -3.0],
                    [0.0, 0.0, 3.2, 4.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            shape = (5, 7, 3)
            _write_nifti(
                image_path,
                np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
                native_affine,
            )
            _write_nifti(
                label_path,
                np.zeros(shape, dtype=np.uint8),
                native_affine,
            )
            labeled = build_case_producer(
                dataset_id="isles26",
                dataset_cfg=_dataset_cfg(),
                load_labels=True,
            )(
                {
                    "caseID": "producer-roundtrip",
                    "T1": [str(image_path)],
                    "label": str(label_path),
                }
            )
            self.assertNotEqual(
                labeled.case.spatial_trace.model,
                labeled.case.spatial_trace.original,
            )
            probability = torch.full_like(labeled.case.image, 0.75)

            result = predict_preprocessed_case(
                DummyExecutor(probability, output_space="native_input"),
                labeled.case,
                show_window_progress=False,
            )
            output_path = base / "restored.nii.gz"
            validation = write_native_prediction_mask(result, output_path)
            reopened = nib.load(str(output_path))

            self.assertEqual(tuple(result.probability.shape[2:]), shape)
            self.assertEqual(reopened.shape, shape)
            np.testing.assert_allclose(
                reopened.affine,
                native_affine,
                rtol=0,
                atol=1e-5,
            )
            self.assertEqual(validation["spatial_validation"], "passed")
            self.assertTrue(
                result.provenance["spatial_validation"][
                    "native_world_coordinates_validated"
                ]
            )

    def test_native_output_uses_each_cases_own_grid(self):
        first = _case((3, 3, 3), np.eye(4), (5, 5, 5), np.diag([0.5, 0.5, 0.5, 1.0]))
        second_native_affine = np.array(
            [[-1.0, 0.0, 0.0, 7.0], [0.0, 2.0, 0.0, -3.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]]
        )
        second = _case((3, 3, 3), np.eye(4), (4, 6, 8), second_native_affine)
        executor = DummyExecutor(torch.full((1, 1, 3, 3, 3), 0.75), output_space="native_input")

        first_result = predict_preprocessed_case(executor, first, show_window_progress=False)
        second_result = predict_preprocessed_case(executor, second, show_window_progress=False)

        self.assertEqual(tuple(first_result.probability.shape[2:]), (5, 5, 5))
        self.assertEqual(tuple(second_result.probability.shape[2:]), (4, 6, 8))
        self.assertNotEqual(
            first_result.spatial_trace.original.affine,
            second_result.spatial_trace.original.affine,
        )

    def test_preprocessed_case_rejects_trace_affine_mismatch(self):
        native = _metadata((3, 3, 3), np.eye(4))
        image = MetaTensor(torch.zeros((1, 1, 3, 3, 3)), affine=torch.eye(4))
        corrupted_affine = np.eye(4)
        corrupted_affine[0, 3] = 99.0

        with self.assertRaisesRegex(SpatialRestorationError, "image affine"):
            PreprocessedCase(
                case_id="corrupted",
                image=image,
                spatial_trace=SpatialTrace(
                    original=native.geometry,
                    model=_geometry((3, 3, 3), corrupted_affine),
                ),
                native_metadata={"T1": native},
                reference_key="T1",
            )

    def test_writer_preserves_native_header_and_binary_uint8_contract(self):
        native_affine = np.array(
            [[-1.25, 0.0, 0.0, 12.0], [0.0, 1.5, 0.0, -8.0], [0.0, 0.0, 2.0, 3.5], [0.0, 0.0, 0.0, 1.0]]
        )
        case = _case((3, 4, 5), native_affine, (3, 4, 5), native_affine)
        probability = torch.zeros((1, 1, 3, 4, 5))
        probability[..., 1, 2, 3] = 1.0
        result = predict_preprocessed_case(
            DummyExecutor(probability, output_space="native_input"),
            case,
            show_window_progress=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "segmentation.nii.gz"
            validation = write_native_prediction_mask(result, output_path)
            reopened = nib.load(str(output_path))
            qform, qform_code = reopened.get_qform(coded=True)
            sform, sform_code = reopened.get_sform(coded=True)

            self.assertEqual(reopened.shape, (3, 4, 5))
            self.assertEqual(reopened.get_data_dtype(), np.dtype(np.uint8))
            self.assertEqual(set(np.unique(np.asarray(reopened.dataobj)).tolist()), {0, 1})
            np.testing.assert_allclose(reopened.affine, native_affine, rtol=0, atol=1e-6)
            np.testing.assert_allclose(qform, native_affine, rtol=0, atol=1e-5)
            np.testing.assert_allclose(sform, native_affine, rtol=0, atol=1e-6)
            self.assertEqual(int(qform_code), 1)
            self.assertEqual(int(sform_code), 2)
            self.assertEqual(validation["dtype"], "uint8")
            self.assertEqual(validation["allowed_values"], [0, 1])
            self.assertEqual(validation["path"], str(output_path))

    def test_writer_preserves_distinct_qform_and_sform_semantics(self):
        affine = np.eye(4)
        qform = np.eye(4)
        qform[0, 3] = 0.125
        base_case = _case((2, 3, 4), affine, (2, 3, 4), affine)
        native = replace(
            base_case.native_metadata["T1"],
            qform=_affine_tuple(qform),
            qform_code=4,
            sform=_affine_tuple(affine),
            sform_code=2,
        )
        case = replace(base_case, native_metadata={"T1": native})
        result = predict_preprocessed_case(
            DummyExecutor(
                torch.ones((1, 1, 2, 3, 4)),
                output_space="native_input",
            ),
            case,
            show_window_progress=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "distinct-forms.nii.gz"
            write_native_prediction_mask(result, path)
            reopened = nib.load(str(path))
            observed_qform, observed_qform_code = reopened.get_qform(coded=True)
            observed_sform, observed_sform_code = reopened.get_sform(coded=True)

        np.testing.assert_allclose(observed_qform, qform, rtol=0, atol=1e-6)
        np.testing.assert_allclose(observed_sform, affine, rtol=0, atol=1e-6)
        self.assertEqual(int(observed_qform_code), 4)
        self.assertEqual(int(observed_sform_code), 2)

    def test_writer_refuses_model_space_mask(self):
        affine = np.eye(4)
        case = _case((2, 3, 4), affine, (2, 3, 4), affine)
        result = predict_preprocessed_case(
            DummyExecutor(
                torch.ones((1, 1, 2, 3, 4)),
                output_space="model_preprocessed",
            ),
            case,
            show_window_progress=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "must-not-exist.nii.gz"
            with self.assertRaisesRegex(InvalidPredictionError, "refusing"):
                write_native_prediction_mask(result, path)
            self.assertFalse(path.exists())

    def test_writer_accepts_empty_and_full_masks(self):
        affine = np.eye(4)
        case = _case((2, 3, 4), affine, (2, 3, 4), affine)
        with tempfile.TemporaryDirectory() as tmp:
            for value in (0.0, 1.0):
                with self.subTest(value=value):
                    result = predict_preprocessed_case(
                        DummyExecutor(
                            torch.full((1, 1, 2, 3, 4), value),
                            output_space="native_input",
                        ),
                        case,
                        show_window_progress=False,
                    )
                    path = Path(tmp) / f"mask-{int(value)}.nii.gz"
                    validation = write_native_prediction_mask(result, path)
                    self.assertEqual(validation["allowed_values"], [int(value)])


if __name__ == "__main__":
    unittest.main()
