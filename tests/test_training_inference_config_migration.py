"""Cut 6 contracts for canonical training-validation config composition."""

import unittest
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from src.inference.policy import resolve_inference_policy


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class TestTrainingInferenceConfigMigration(unittest.TestCase):
    def _compose(self, config_name):
        with hydra.initialize_config_dir(
            config_dir=str(REPOSITORY_ROOT / "configs"),
            version_base=None,
        ):
            return hydra.compose(config_name=config_name)

    def test_base_training_profiles_compose_direct_native_inference(self):
        for config_name in ("local", "cluster"):
            with self.subTest(config_name=config_name):
                cfg = self._compose(config_name)
                resolved = resolve_inference_policy(cfg)

                self.assertEqual(resolved.source, "explicit_top_level")
                self.assertFalse(resolved.policy.sliding_window.enabled)
                self.assertEqual(resolved.policy.output_space, "model_preprocessed")
                self.assertEqual(cfg.inference_runtime.profile, "native")
                self.assertIsNone(
                    OmegaConf.select(cfg, "validation.inference", default=None)
                )

    def test_3d_training_profiles_compose_sliding_window_native_inference(self):
        expected_window_batches = {
            "cluster_isles26_3d_randompatch_dynunet": 4,
            "cluster_isles26_3d_randompatch_swinunetr": 4,
            "cluster_isles26_atlas30_3d_randompatch_dynunet": 4,
            "cluster_isles26_3d_randompatch_dynunet_params_26062026": 4,
            "local_isles24_3d_randompatch_swinunetr": 1,
            "local_isles26_3d_randompatch_swinunetr": 1,
            "local_isles26_3d_randompatch_dynunet": 1,
        }

        for config_name, expected_window_batch in expected_window_batches.items():
            with self.subTest(config_name=config_name):
                cfg = self._compose(config_name)
                resolved = resolve_inference_policy(cfg)

                self.assertEqual(resolved.source, "explicit_top_level")
                self.assertTrue(resolved.policy.sliding_window.enabled)
                self.assertEqual(
                    resolved.policy.sliding_window.sw_batch_size,
                    expected_window_batch,
                )
                self.assertEqual(resolved.policy.output_space, "model_preprocessed")
                self.assertEqual(cfg.inference_runtime.profile, "native")
                self.assertIsNone(
                    OmegaConf.select(cfg, "validation.inference", default=None)
                )

    def test_existing_2d_ddp_profiles_inherit_direct_native_inference(self):
        for config_name in ("cluster_1M_ddp", "cluster_500K_64b_4gpu_ddp"):
            with self.subTest(config_name=config_name):
                cfg = self._compose(config_name)
                resolved = resolve_inference_policy(cfg)

                self.assertEqual(cfg.data_mode.dim, "2d")
                self.assertEqual(cfg.distribution.strategy, "ddp")
                self.assertEqual(resolved.source, "explicit_top_level")
                self.assertFalse(resolved.policy.sliding_window.enabled)
                self.assertEqual(resolved.policy.output_space, "model_preprocessed")
                self.assertEqual(cfg.inference_runtime.profile, "native")
                self.assertIsNone(
                    OmegaConf.select(cfg, "validation.inference", default=None)
                )

    def test_validation_presets_own_no_prediction_policy_fields(self):
        config_dir = REPOSITORY_ROOT / "configs" / "validation"
        for config_path in config_dir.glob("*.yaml"):
            with self.subTest(config_path=config_path.name):
                raw = OmegaConf.load(config_path)
                self.assertNotIn("inference", raw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
