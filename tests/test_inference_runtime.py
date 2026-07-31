import unittest
from dataclasses import replace
from pathlib import Path

from omegaconf import OmegaConf

from src.inference.policy import ArtifactPolicy, parse_inference_policy
from src.inference.runtime import (
    AssessmentContext,
    InvalidInferenceRuntimeError,
    parse_inference_runtime,
    validate_runtime_compatibility,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class TestInferenceRuntimeProfiles(unittest.TestCase):
    def _load_profile(self, name):
        path = REPOSITORY_ROOT / "configs" / "inference_runtime" / f"{name}.yaml"
        return parse_inference_runtime(OmegaConf.load(path))

    def _policy(self, **overrides):
        raw = {}
        raw.update(overrides)
        return parse_inference_policy(raw, model_roi=(32, 32, 32))

    def test_native_and_container_test_accept_both_output_spaces(self):
        for profile_name in ("native", "gc_container_test"):
            runtime = self._load_profile(profile_name)
            for output_space in ("model_preprocessed", "native_input"):
                with self.subTest(profile=profile_name, output_space=output_space):
                    policy = self._policy(output_space=output_space)
                    validated = validate_runtime_compatibility(policy, runtime)
                    self.assertIs(validated.inference, policy)
                    self.assertIs(validated.runtime, runtime)

    def test_canonical_profiles_declare_execution_constraints(self):
        native = self._load_profile("native")
        container_test = self._load_profile("gc_container_test")
        submission = self._load_profile("gc_submission")

        self.assertEqual(native.profile, "native")
        self.assertFalse(native.require_cuda)
        self.assertIsNone(native.timeout_seconds)

        self.assertTrue(container_test.require_cuda)
        self.assertEqual(container_test.case_batch_size, 1)
        self.assertEqual(container_test.num_workers, 0)
        self.assertEqual(container_test.timeout_seconds, 600)

        self.assertTrue(submission.require_cuda)
        self.assertEqual(submission.case_batch_size, 1)
        self.assertEqual(submission.num_workers, 0)
        self.assertEqual(submission.timeout_seconds, 600)

    def test_gc_submission_rejects_model_space(self):
        runtime = self._load_profile("gc_submission")
        policy = self._policy(output_space="model_preprocessed")

        with self.assertRaisesRegex(
            InvalidInferenceRuntimeError,
            "gc_submission.*native_input",
        ):
            validate_runtime_compatibility(policy, runtime)

    def test_gc_submission_rejects_batch_workers_labels_sweeps_and_artifacts(self):
        valid_policy = self._policy(output_space="native_input", precision="fp16")
        base = OmegaConf.to_container(
            OmegaConf.load(
                REPOSITORY_ROOT
                / "configs"
                / "inference_runtime"
                / "gc_submission.yaml"
            ),
            resolve=True,
        )

        for field, value, message in (
            ("case_batch_size", 2, "case_batch_size.*1"),
            ("num_workers", 1, "num_workers.*0"),
        ):
            with self.subTest(field=field):
                modified = dict(base)
                modified[field] = value
                runtime = parse_inference_runtime(modified)
                with self.assertRaisesRegex(InvalidInferenceRuntimeError, message):
                    validate_runtime_compatibility(valid_policy, runtime)

        runtime = self._load_profile("gc_submission")
        for assessment, message in (
            (AssessmentContext(requires_ground_truth=True), "ground truth"),
            (AssessmentContext(threshold_sweep=True), "threshold sweep"),
        ):
            with self.subTest(assessment=assessment):
                with self.assertRaisesRegex(InvalidInferenceRuntimeError, message):
                    validate_runtime_compatibility(valid_policy, runtime, assessment)

        artifact_policy = replace(
            valid_policy,
            artifacts=ArtifactPolicy(enabled=True),
        )
        with self.assertRaisesRegex(InvalidInferenceRuntimeError, "intermediate artifacts"):
            validate_runtime_compatibility(artifact_policy, runtime)

    def test_bf16_fails_on_uncertified_deployment_profile(self):
        policy = self._policy(output_space="native_input", precision="bf16")
        runtime = self._load_profile("gc_submission")

        with self.assertRaisesRegex(
            InvalidInferenceRuntimeError,
            "precision.*bf16.*not allowed",
        ):
            validate_runtime_compatibility(policy, runtime)

    def test_unknown_runtime_keys_fail(self):
        with self.assertRaisesRegex(InvalidInferenceRuntimeError, "unknown keys.*mystery"):
            parse_inference_runtime({"mystery": True})

    def test_gc_profile_cannot_relax_production_constraints(self):
        runtime = parse_inference_runtime(
            {
                "profile": "gc_submission",
                "case_batch_size": 1,
                "num_workers": 0,
                "require_cuda": True,
                "timeout_seconds": 600,
                "constraints": {
                    "allowed_output_spaces": ["model_preprocessed", "native_input"],
                    "allowed_precisions": ["fp16", "fp32", "bf16"],
                    "allow_ground_truth": True,
                    "allow_threshold_sweep": True,
                    "allow_intermediate_artifacts": True,
                },
            }
        )
        policy = self._policy(output_space="native_input", precision="fp16")

        with self.assertRaisesRegex(InvalidInferenceRuntimeError, "hard constraints"):
            validate_runtime_compatibility(policy, runtime)


if __name__ == "__main__":
    unittest.main()
