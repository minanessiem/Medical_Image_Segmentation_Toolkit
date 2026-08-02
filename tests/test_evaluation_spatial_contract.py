"""Spatial-contract tests for geometry-aware 3D evaluation."""

import unittest

import torch

from scripts.evaluation.core.contracts import VolumeSample
from src.inference.contracts import SpatialGeometry


def _geometry(*, translation_x: float = 0.0, shape=(3, 4, 5)) -> SpatialGeometry:
    return SpatialGeometry(
        shape=shape,
        affine=(
            (1.0, 0.0, 0.0, translation_x),
            (0.0, 2.0, 0.0, 20.0),
            (0.0, 0.0, 3.0, 30.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        spacing=(1.0, 2.0, 3.0),
        orientation="RAS",
    )


def _sample(**overrides) -> VolumeSample:
    values = {
        "case_id": "case001",
        "volume_id": "case001",
        "prediction_volume": torch.zeros((1, 3, 4, 5)),
        "ground_truth_volume": torch.zeros((1, 3, 4, 5)),
        "prediction_space": "model_preprocessed",
        "reference_space": "model_preprocessed",
        "prediction_geometry": _geometry(),
        "reference_geometry": _geometry(),
    }
    values.update(overrides)
    return VolumeSample(**values)


class TestEvaluationSpatialContract(unittest.TestCase):
    def test_matching_declared_3d_sample_passes(self):
        _sample().validate()

    def test_missing_geometry_fails(self):
        sample = _sample(prediction_geometry=None)

        with self.assertRaisesRegex(ValueError, "prediction_geometry.*mandatory"):
            sample.validate()

    def test_unknown_space_fails(self):
        sample = _sample(prediction_space="guessed_space")

        with self.assertRaisesRegex(ValueError, "prediction_space must be one of"):
            sample.validate()

    def test_cross_space_pairing_fails(self):
        sample = _sample(reference_space="native_input")

        with self.assertRaisesRegex(ValueError, "space mismatch"):
            sample.validate()

    def test_tensor_geometry_shape_mismatch_fails(self):
        sample = _sample(prediction_geometry=_geometry(shape=(3, 4, 6)))

        with self.assertRaisesRegex(ValueError, "tensor/geometry shape mismatch"):
            sample.validate()

    def test_equal_shape_different_affine_fails(self):
        sample = _sample(reference_geometry=_geometry(translation_x=10.0))

        with self.assertRaisesRegex(ValueError, "affine mismatch"):
            sample.validate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
