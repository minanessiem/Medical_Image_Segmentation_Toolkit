"""Arbitrary-N probability ensemble contracts."""

from __future__ import annotations

import unittest

import torch

from src.inference.contracts import (
    InvalidPredictionError,
    PredictorCapabilities,
    UnsupportedModelError,
)
from src.inference.ensemble import MeanProbabilityAccumulator, mean_probability_ensemble
from src.inference.pipeline import (
    EnsembleMemberExecutor,
    EnsembleProbabilityExecutor,
    ModelProbabilityExecutor,
)
from src.inference.policy import EnsemblePolicy, InferencePolicy, SlidingWindowPolicy


class _ConstantPredictor:
    capabilities = PredictorCapabilities(
        model_family="discriminative",
        spatial_dims=3,
        input_channels=1,
        output_channels=1,
        supported_precisions=("fp32",),
    )

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, conditioned_image: torch.Tensor) -> torch.Tensor:
        return torch.full_like(conditioned_image, self.value)


def _ensemble_policy() -> InferencePolicy:
    return InferencePolicy(
        sliding_window=SlidingWindowPolicy(
            roi_size=(2, 2, 2),
            enabled=False,
        ),
        precision="fp32",
        ensemble=EnsemblePolicy(enabled=True, method="mean"),
    )


class TestProbabilityEnsemble(unittest.TestCase):
    def test_three_members_are_equally_averaged_in_fp32_without_a_count(self):
        result = mean_probability_ensemble(
            (
                torch.full((1, 1, 2, 2, 2), 0.1, dtype=torch.float16),
                torch.full((1, 1, 2, 2, 2), 0.4, dtype=torch.float16),
                torch.full((1, 1, 2, 2, 2), 0.7, dtype=torch.float16),
            )
        )

        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.allclose(result, torch.full_like(result, 0.4), atol=2e-4))

    def test_accumulator_rejects_empty_mismatched_and_invalid_probabilities(self):
        with self.assertRaisesRegex(InvalidPredictionError, "without at least one"):
            MeanProbabilityAccumulator().mean()

        accumulator = MeanProbabilityAccumulator()
        accumulator.add(torch.full((1, 1, 2, 2, 2), 0.5))
        with self.assertRaisesRegex(InvalidPredictionError, "identical shapes"):
            accumulator.add(torch.full((1, 1, 3, 2, 2), 0.5))
        with self.assertRaisesRegex(InvalidPredictionError, "within \[0, 1\]"):
            MeanProbabilityAccumulator().add(torch.tensor([1.1]))
        with self.assertRaisesRegex(InvalidPredictionError, "non-finite"):
            MeanProbabilityAccumulator().add(torch.tensor([float("nan")]))

    def test_executor_discovers_its_effective_count_from_members(self):
        policy = _ensemble_policy()
        members = tuple(
            EnsembleMemberExecutor(
                member_id=f"fold{index}",
                executor=ModelProbabilityExecutor(
                    predictor=_ConstantPredictor(value),
                    policy=policy,
                    policy_source="explicit_top_level",
                ),
            )
            for index, value in enumerate((0.2, 0.5, 0.8), start=1)
        )
        executor = EnsembleProbabilityExecutor(
            members=members,
            policy=policy,
            policy_source="explicit_top_level",
        )

        result = executor(torch.zeros((1, 1, 2, 2, 2)))

        self.assertEqual(executor.member_ids, ("fold1", "fold2", "fold3"))
        self.assertTrue(torch.equal(result, torch.full_like(result, 0.5)))

    def test_executor_rejects_duplicate_ids_and_disabled_ensemble(self):
        policy = _ensemble_policy()
        member = EnsembleMemberExecutor(
            member_id="fold1",
            executor=ModelProbabilityExecutor(
                predictor=_ConstantPredictor(0.5),
                policy=policy,
                policy_source="explicit_top_level",
            ),
        )
        with self.assertRaisesRegex(UnsupportedModelError, "unique"):
            EnsembleProbabilityExecutor(
                members=(member, member),
                policy=policy,
                policy_source="explicit_top_level",
            )

        disabled = InferencePolicy(
            sliding_window=SlidingWindowPolicy(
                roi_size=(2, 2, 2),
                enabled=False,
            )
        )
        disabled_member = EnsembleMemberExecutor(
            member_id="fold1",
            executor=ModelProbabilityExecutor(
                predictor=_ConstantPredictor(0.5),
                policy=disabled,
                policy_source="explicit_top_level",
            ),
        )
        with self.assertRaisesRegex(UnsupportedModelError, "enabled=true"):
            EnsembleProbabilityExecutor(
                members=(disabled_member,),
                policy=disabled,
                policy_source="explicit_top_level",
            )


if __name__ == "__main__":
    unittest.main()
