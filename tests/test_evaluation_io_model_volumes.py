"""Tests for the unified typed-case repository-model volume producer."""

import inspect
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from monai.data import MetaTensor
from omegaconf import OmegaConf

from scripts.evaluation.core.contracts import VolumeSample
from scripts.evaluation.io import model_volumes
from scripts.evaluation.io.model_volumes import (
    iter_model_volume_samples,
    validate_model_evaluation_mode,
)
from src.inference.contracts import (
    LabeledPreprocessedCase,
    NativeImageMetadata,
    PredictionResult,
    PreprocessedCase,
    SpatialGeometry,
    SpatialTrace,
)


def _base_cfg(diffusion_type="Discriminative", dim="3d"):
    return OmegaConf.create(
        {
            "data_mode": {"dim": dim, "loader_mode": "full_volumes_3d"},
            "diffusion": {"type": diffusion_type},
            "model": {"spatial_dims": dim, "image_channels": 1, "out_channels": 1},
        }
    )


def _geometry(shape, affine):
    affine = np.asarray(affine, dtype=np.float64)
    spacing = tuple(float(value) for value in np.sqrt((affine[:3, :3] ** 2).sum(axis=0)))
    return SpatialGeometry(
        shape=tuple(shape),
        affine=tuple(tuple(float(value) for value in row) for row in affine),
        spacing=spacing,
        orientation="RAS",
    )


def _metadata(key, geometry):
    return NativeImageMetadata(
        canonical_key=key,
        shape=geometry.shape,
        dtype="float32",
        affine=geometry.affine,
        spacing=geometry.spacing,
        orientation=geometry.orientation,
        qform=geometry.affine,
        sform=geometry.affine,
        qform_code=1,
        sform_code=1,
        source_reference=f"synthetic-{key}",
    )


def _labeled_case(case_id="case-a"):
    model_geometry = _geometry((2, 2, 2), np.eye(4))
    native_affine = np.diag([1.5, 2.0, 2.5, 1.0])
    native_affine[:3, 3] = [4.0, -3.0, 7.0]
    native_geometry = _geometry((3, 4, 5), native_affine)
    image_metadata = _metadata("T1", native_geometry)
    label_metadata = _metadata("label", native_geometry)
    case = PreprocessedCase(
        case_id=case_id,
        image=MetaTensor(torch.zeros((1, 1, 2, 2, 2)), affine=torch.eye(4)),
        spatial_trace=SpatialTrace(
            original=native_geometry,
            model=model_geometry,
            transform_history=({"class": "SpatialResample", "do_transforms": True},),
        ),
        native_metadata={"T1": image_metadata},
        reference_key="T1",
        metadata={
            "dataset_id": "isles26",
            "processed_modalities": ("T1_RAW",),
            "record_metadata": {
                "split": "val",
                "siteID": "site-a",
                "output_space": "untrusted-record-value",
            },
        },
    )
    return LabeledPreprocessedCase(
        case=case,
        model_label=torch.full((1, 1, 2, 2, 2), 0.25),
        model_label_geometry=model_geometry,
        native_label=torch.full((1, 1, 3, 4, 5), 1.0),
        native_label_metadata=label_metadata,
    )


def _prediction(labeled, output_space):
    geometry = (
        labeled.case.spatial_trace.original
        if output_space == "native_input"
        else labeled.case.spatial_trace.model
    )
    probability = torch.full((1, 1, *geometry.shape), 0.75)
    return PredictionResult(
        probability=probability,
        output_space=output_space,
        spatial_trace=labeled.case.spatial_trace,
        native_reference=(
            labeled.case.native_metadata[labeled.case.reference_key]
            if output_space == "native_input"
            else None
        ),
        provenance={"spatial_restoration": {"applied": output_space == "native_input"}},
    )


def _executor(output_space="model_preprocessed"):
    return SimpleNamespace(
        policy=SimpleNamespace(output_space=output_space, precision="fp32"),
        policy_source="explicit_top_level",
    )


class TestModelVolumeIO(unittest.TestCase):
    def test_unsupported_current_3d_diffusion_raises(self):
        cfg = _base_cfg(diffusion_type="OpenAI_DDPM", dim="3d")

        with self.assertRaises(ValueError) as ctx:
            validate_model_evaluation_mode(cfg)

        message = str(ctx.exception)
        self.assertIn("3D live-model evaluation", message)
        self.assertIn("discriminative", message.lower())
        self.assertIn("OpenAI_DDPM", message)
        self.assertIn("ProbabilityPredictor", message)

    def test_2d_mode_is_rejected_until_reconstruction_contract_exists(self):
        cfg = _base_cfg(diffusion_type="OpenAI_DDPM", dim="2d")

        with self.assertRaisesRegex(ValueError, "deferred 2D reconstruction contract"):
            validate_model_evaluation_mode(cfg)

    def test_one_typed_loop_selects_reference_for_both_output_spaces(self):
        labeled = _labeled_case()
        expectations = {
            "model_preprocessed": (labeled.model_label_geometry, 0.25, False),
            "native_input": (labeled.native_label_geometry, 1.0, True),
        }

        for output_space, (expected_geometry, expected_label, restored) in expectations.items():
            with self.subTest(output_space=output_space), patch(
                "scripts.evaluation.io.model_volumes.predict_preprocessed_case",
                return_value=_prediction(labeled, output_space),
            ) as predict_mock:
                samples = list(
                    iter_model_volume_samples(
                        executor=_executor(output_space),
                        cases=[labeled],
                        device="cpu",
                        loader_mode="full_volumes_3d",
                        subset="val_full",
                        show_progress=False,
                    )
                )

            self.assertEqual(len(samples), 1)
            sample = samples[0]
            self.assertIsInstance(sample, VolumeSample)
            self.assertEqual(sample.prediction_space, output_space)
            self.assertEqual(sample.reference_space, output_space)
            self.assertEqual(sample.prediction_geometry, expected_geometry)
            self.assertEqual(sample.reference_geometry, expected_geometry)
            self.assertAlmostEqual(float(sample.ground_truth_volume.mean()), expected_label)
            self.assertEqual(sample.metadata["siteID"], "site-a")
            self.assertEqual(sample.metadata["output_space"], output_space)
            self.assertEqual(sample.metadata["subset"], "val_full")
            self.assertEqual(sample.metadata["spatial_restoration_applied"], restored)
            moved_case = predict_mock.call_args.args[1]
            self.assertEqual(moved_case.image.device.type, "cpu")

    def test_iter_model_volume_samples_respects_max_samples(self):
        first = _labeled_case("case-a")
        second = _labeled_case("case-b")

        def prediction_for_case(_executor_value, case, **_kwargs):
            labeled = first if case.case_id == "case-a" else second
            return _prediction(labeled, "model_preprocessed")

        with patch(
            "scripts.evaluation.io.model_volumes.predict_preprocessed_case",
            side_effect=prediction_for_case,
        ):
            samples = list(
                iter_model_volume_samples(
                    executor=_executor(),
                    cases=[first, second],
                    device="cpu",
                    show_progress=False,
                    max_samples=1,
                )
            )

        self.assertEqual([sample.case_id for sample in samples], ["case-a"])

    def test_unlabeled_case_fails_at_typed_evaluation_boundary(self):
        with self.assertRaisesRegex(TypeError, "LabeledPreprocessedCase"):
            list(
                iter_model_volume_samples(
                    executor=_executor(),
                    cases=[_labeled_case().case],
                    device="cpu",
                    show_progress=False,
                )
            )

    def test_native_only_record_resolution_loop_is_absent(self):
        source = inspect.getsource(model_volumes)

        self.assertNotIn("_iter_native_model_volume_samples", source)
        self.assertNotIn("_resolve_dataset_records", source)
        self.assertNotIn("dataset.database", source)
        self.assertNotIn("get_preprocessing_adapter", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
