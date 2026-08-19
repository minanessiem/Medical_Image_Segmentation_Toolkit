"""Spatial restoration contracts for model- and native-grid inference results."""

from __future__ import annotations

import unittest

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor

from src.inference.contracts import SpatialGeometry, SpatialRestorationError, SpatialTrace
from src.inference.spatial import (
    restore_probability_to_native,
    validate_output_geometry,
)


def _affine_tuple(affine: np.ndarray):
    return tuple(tuple(float(value) for value in row) for row in affine)


def _geometry(shape, affine) -> SpatialGeometry:
    affine = np.asarray(affine, dtype=np.float64)
    return SpatialGeometry(
        shape=tuple(int(value) for value in shape),
        affine=_affine_tuple(affine),
        spacing=tuple(float(value) for value in nib.affines.voxel_sizes(affine)),
        orientation="".join(str(value) for value in nib.aff2axcodes(affine)),
    )


def _trace(model_shape, model_affine, native_shape, native_affine, *, history=True):
    operations = (
        ({"class": "SpatialResample", "do_transforms": True},)
        if history
        else ()
    )
    return SpatialTrace(
        original=_geometry(native_shape, native_affine),
        model=_geometry(model_shape, model_affine),
        transform_history=operations,
    )


def _world_linear_probability(shape, affine) -> np.ndarray:
    grid = np.indices(shape, dtype=np.float64).reshape(3, -1)
    homogeneous = np.vstack([grid, np.ones((1, grid.shape[1]), dtype=np.float64)])
    world = np.asarray(affine, dtype=np.float64) @ homogeneous
    values = 0.5 + (0.002 * world[0]) + (0.0015 * world[1]) + (0.001 * world[2])
    return np.clip(values.reshape(shape), 0.05, 0.95).astype(np.float32)


class TestInferenceSpatialRoundTrip(unittest.TestCase):
    def test_identity_restoration_is_exact_without_transform_history(self):
        affine = np.array(
            [[1.0, 0.0, 0.0, -4.0], [0.0, 2.0, 0.0, 3.0], [0.0, 0.0, 1.5, 9.0], [0.0, 0.0, 0.0, 1.0]]
        )
        probability = torch.linspace(0.0, 1.0, 3 * 5 * 7).reshape(1, 1, 3, 5, 7)
        trace = _trace((3, 5, 7), affine, (3, 5, 7), affine, history=False)

        restored = restore_probability_to_native(probability, trace)

        torch.testing.assert_close(restored, probability, rtol=0, atol=0)

    def test_continuous_restoration_handles_flip_permutation_spacing_translation_and_oblique(self):
        cases = {
            "axis_flip": (
                (5, 6, 7),
                np.array([[-1.0, 0.0, 0.0, 4.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
                (5, 6, 7),
                np.eye(4),
            ),
            "axis_permutation": (
                (6, 5, 7),
                np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
                (5, 6, 7),
                np.eye(4),
            ),
            "anisotropic_spacing": (
                (9, 11, 13),
                np.diag([0.5, 0.5, 0.5, 1.0]),
                (5, 6, 7),
                np.eye(4),
            ),
            "translation_and_odd_shape": (
                (11, 15, 19),
                np.array([[0.5, 0.0, 0.0, -1.0], [0.0, 0.5, 0.0, -1.0], [0.0, 0.0, 0.5, -1.0], [0.0, 0.0, 0.0, 1.0]]),
                (5, 7, 9),
                np.eye(4),
            ),
            "oblique": (
                (9, 9, 9),
                np.array([[0.9961947, 0.0, 0.0871557, -2.0], [0.0, 1.0, 0.0, -2.0], [-0.0871557, 0.0, 0.9961947, -2.0], [0.0, 0.0, 0.0, 1.0]]),
                (5, 5, 5),
                np.eye(4),
            ),
        }
        for name, (model_shape, model_affine, native_shape, native_affine) in cases.items():
            with self.subTest(name=name):
                model_values = _world_linear_probability(model_shape, model_affine)
                probability = torch.from_numpy(model_values).unsqueeze(0).unsqueeze(0)
                trace = _trace(
                    model_shape,
                    model_affine,
                    native_shape,
                    native_affine,
                )

                restored = restore_probability_to_native(probability, trace)
                expected = torch.from_numpy(
                    _world_linear_probability(native_shape, native_affine)
                ).unsqueeze(0).unsqueeze(0)

                self.assertEqual(tuple(restored.shape[2:]), native_shape)
                torch.testing.assert_close(restored, expected, rtol=0, atol=2e-4)

    def test_probability_is_restored_before_thresholding(self):
        model_affine = np.diag([2.0, 1.0, 1.0, 1.0])
        native_affine = np.eye(4)
        probability = torch.tensor([0.2, 0.4, 0.6]).reshape(1, 1, 3, 1, 1)
        trace = _trace((3, 1, 1), model_affine, (5, 1, 1), native_affine)

        restored_probability = restore_probability_to_native(probability, trace)
        threshold_first = restore_probability_to_native(
            (probability >= 0.5).to(torch.float32),
            trace,
        )
        expected = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6]).reshape(1, 1, 5, 1, 1)

        probability_error = torch.mean((restored_probability - expected) ** 2)
        threshold_first_error = torch.mean((threshold_first - expected) ** 2)
        self.assertLess(float(probability_error), float(threshold_first_error))

    def test_world_space_landmark_survives_axis_flip(self):
        model_affine = np.array(
            [[-1.0, 0.0, 0.0, 4.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        probability = torch.zeros((1, 1, 5, 5, 5))
        probability[0, 0, 1, 2, 3] = 1.0
        trace = _trace((5, 5, 5), model_affine, (5, 5, 5), np.eye(4))

        restored = restore_probability_to_native(probability, trace)

        self.assertEqual(float(restored[0, 0, 3, 2, 3]), 1.0)
        self.assertEqual(int(torch.count_nonzero(restored)), 1)

    def test_recorded_spatial_padding_is_cropped_back_to_native_grid(self):
        model_affine = np.eye(4)
        model_affine[:3, 3] = -1.0
        model_values = _world_linear_probability((5, 5, 5), model_affine)
        trace = SpatialTrace(
            original=_geometry((3, 3, 3), np.eye(4)),
            model=_geometry((5, 5, 5), model_affine),
            transform_history=({"class": "SpatialPad", "do_transforms": True},),
        )

        restored = restore_probability_to_native(
            torch.from_numpy(model_values).unsqueeze(0).unsqueeze(0),
            trace,
        )
        expected = torch.from_numpy(
            _world_linear_probability((3, 3, 3), np.eye(4))
        ).unsqueeze(0).unsqueeze(0)

        torch.testing.assert_close(restored, expected, rtol=0, atol=1e-6)

    def test_near_equivalent_nifti_qform_sform_rounding_is_accepted(self):
        native_affine = np.array(
            [
                [0.998809814, 0.025174877, 0.041774701, -119.675789],
                [-0.027847558, 0.997482359, 0.064658277, -90.232483],
                [-0.040043227, -0.065742239, 0.997032702, -121.763542],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        model_affine = np.array(
            [
                [0.998809814, 0.025175797, 0.041774701, -119.675789],
                [-0.027847556, 0.997518837, 0.064658277, -90.232483],
                [-0.040043227, -0.065744646, 0.997032702, -121.763542],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        probability = torch.linspace(0.0, 1.0, 5**3).reshape(1, 1, 5, 5, 5)
        trace = _trace(
            (5, 5, 5),
            model_affine,
            (5, 5, 5),
            native_affine,
        )

        restored = restore_probability_to_native(probability, trace)

        torch.testing.assert_close(restored, probability, rtol=0, atol=0)

    def test_world_geometry_validation_rejects_real_mismatch(self):
        expected = _geometry((5, 5, 5), np.eye(4))
        shifted = np.eye(4)
        shifted[0, 3] = 0.02

        with self.assertRaisesRegex(SpatialRestorationError, "affine"):
            validate_output_geometry(
                observed_shape=(5, 5, 5),
                observed_affine=shifted,
                expected=expected,
            )

    def test_changed_geometry_requires_a_supported_spatial_trace(self):
        trace = _trace(
            (3, 3, 3),
            np.eye(4),
            (5, 5, 5),
            np.diag([0.5, 0.5, 0.5, 1.0]),
            history=False,
        )
        with self.assertRaisesRegex(SpatialRestorationError, "transform history"):
            restore_probability_to_native(torch.full((1, 1, 3, 3, 3), 0.5), trace)

    def test_probability_must_match_the_recorded_model_grid(self):
        trace = _trace((4, 4, 4), np.eye(4), (4, 4, 4), np.eye(4), history=False)
        with self.assertRaisesRegex(SpatialRestorationError, "model grid"):
            restore_probability_to_native(torch.full((1, 1, 3, 4, 4), 0.5), trace)

    def test_unknown_applied_spatial_operation_fails_closed(self):
        trace = SpatialTrace(
            original=_geometry((5, 5, 5), np.eye(4)),
            model=_geometry((3, 3, 3), np.eye(4)),
            transform_history=({"class": "UnregisteredWarp", "do_transforms": True},),
        )
        with self.assertRaisesRegex(SpatialRestorationError, "UnregisteredWarp"):
            restore_probability_to_native(torch.full((1, 1, 3, 3, 3), 0.5), trace)


if __name__ == "__main__":
    unittest.main()
