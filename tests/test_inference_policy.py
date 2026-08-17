import unittest
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from src.inference.policy import (
    InvalidInferencePolicyError,
    parse_inference_policy,
    resolve_inference_policy,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class TestInferencePolicyParsing(unittest.TestCase):
    def _parse(self, raw_policy, model_roi=(32, 32, 32)):
        return parse_inference_policy(raw_policy, model_roi=model_roi)

    def test_policy_receives_model_roi_and_defaults_to_sliding_window(self):
        policy = self._parse({})
        self.assertEqual(policy.output_space, "model_preprocessed")
        self.assertEqual(policy.precision, "fp32")
        self.assertTrue(policy.sliding_window.enabled)
        self.assertEqual(policy.sliding_window.roi_size, (32, 32, 32))
        self.assertEqual(policy.sliding_window.sw_batch_size, 1)
        self.assertEqual(policy.decision.threshold, 0.5)
        self.assertFalse(policy.tta.enabled)
        self.assertFalse(policy.ensemble.enabled)
        self.assertFalse(policy.postprocessing.enabled)
        self.assertFalse(policy.artifacts.enabled)

    def test_unknown_top_level_and_nested_keys_fail(self):
        with self.assertRaisesRegex(InvalidInferencePolicyError, "unknown keys.*mystery"):
            self._parse({"mystery": True})

        with self.assertRaisesRegex(
            InvalidInferencePolicyError,
            "inference.sliding_window.*unknown keys.*mystery",
        ):
            self._parse({"sliding_window": {"mystery": True}})

    def test_invalid_values_fail_without_coercion(self):
        invalid_policies = (
            (
                {"output_space": "native-ish"},
                "output_space",
            ),
            (
                {"precision": "tf32"},
                "precision",
            ),
            (
                {"sliding_window": {"sw_batch_size": 0}},
                "sw_batch_size",
            ),
            (
                {"sliding_window": {"overlap": 1.0}},
                "overlap",
            ),
            (
                {
                    "decision": {"threshold": 1.1},
                },
                "threshold",
            ),
            (
                {"sliding_window": {"roi_size": None}},
                "unknown keys.*roi_size",
            ),
        )
        for raw_policy, message in invalid_policies:
            with self.subTest(raw_policy=raw_policy):
                with self.assertRaisesRegex(InvalidInferencePolicyError, message):
                    self._parse(raw_policy)

        explicit_direct = self._parse(
            {
                "sliding_window": {
                    "enabled": False,
                }
            }
        )
        self.assertFalse(explicit_direct.sliding_window.enabled)

    def test_unimplemented_features_are_disabled_stubs(self):
        feature_names = ("tta", "postprocessing", "artifacts")
        for feature_name in feature_names:
            with self.subTest(feature=feature_name):
                policy = self._parse(
                    {
                        feature_name: {"enabled": False},
                    }
                )
                self.assertFalse(getattr(policy, feature_name).enabled)
                with self.assertRaisesRegex(
                    InvalidInferencePolicyError,
                    "not implemented yet.*enabled=false",
                ):
                    self._parse(
                        {
                            feature_name: {"enabled": True},
                        }
                    )

        premature_settings = (
            {"tta": {"enabled": False, "transforms": []}},
            {"postprocessing": {"enabled": False, "connectivity": 26}},
            {"artifacts": {"enabled": False, "retain_model_space_probability": True}},
        )
        for premature_setting in premature_settings:
            raw_policy = {
                **premature_setting,
            }
            with self.subTest(raw_policy=raw_policy):
                with self.assertRaisesRegex(InvalidInferencePolicyError, "unknown keys"):
                    self._parse(raw_policy)

    def test_ensemble_supports_mean_without_a_configured_member_count(self):
        policy = self._parse(
            {"ensemble": {"enabled": True, "method": "mean"}}
        )
        self.assertTrue(policy.ensemble.enabled)
        self.assertEqual(policy.ensemble.method, "mean")

        with self.assertRaisesRegex(
            InvalidInferencePolicyError,
            "unknown keys.*member_count",
        ):
            self._parse(
                {
                    "ensemble": {
                        "enabled": True,
                        "method": "mean",
                        "member_count": 3,
                    }
                }
            )

        with self.assertRaisesRegex(InvalidInferencePolicyError, "method"):
            self._parse(
                {"ensemble": {"enabled": True, "method": "median"}}
            )

    def test_removed_placeholder_fields_are_rejected(self):
        for field_name, value in (
            ("schema_version", 1),
            ("deterministic", True),
            ("output", {"dtype": "uint8"}),
        ):
            with self.subTest(field=field_name):
                with self.assertRaisesRegex(
                    InvalidInferencePolicyError,
                    f"unknown keys.*{field_name}",
                ):
                    self._parse({field_name: value})

    def test_explicit_top_level_policy_replaces_legacy_policy(self):
        cfg = OmegaConf.create(
            {
                "inference": {
                    "output_space": "native_input",
                    "precision": "fp16",
                    "sliding_window": {
                        "enabled": True,
                        "sw_batch_size": 1,
                    },
                },
                "model": {"spatial_dims": "3d"},
                "data_mode": {"dim": "3d"},
                "dataset": {
                    "preprocessing_configs": {
                        "roi": {"volume_3d": [64, 64, 64]}
                    }
                },
                "validation": {
                    "inference": {
                        "mode": "sliding_window",
                        "sliding_window": {
                            "sw_batch_size": 4,
                            "overlap": 0.25,
                        },
                    }
                },
            }
        )

        resolved = resolve_inference_policy(cfg)

        self.assertEqual(resolved.source, "explicit_top_level")
        self.assertEqual(resolved.policy.output_space, "native_input")
        self.assertEqual(resolved.policy.sliding_window.sw_batch_size, 1)
        self.assertEqual(resolved.policy.sliding_window.overlap, 0.5)

    def test_legacy_validation_policy_is_translated_when_top_level_is_absent(self):
        cfg = OmegaConf.create(
            {
                "model": {"spatial_dims": "3d"},
                "data_mode": {
                    "dim": "3d",
                    "loader_mode": "random_patches_3d",
                },
                "dataset": {
                    "preprocessing_configs": {
                        "roi": {"volume_3d": [48, 48, 48]}
                    }
                },
                "validation": {
                    "inference": {
                        "mode": "sliding_window",
                        "sliding_window": {
                            "roi_size": None,
                            "sw_batch_size": 4,
                            "overlap": 0.25,
                            "blend_mode": "constant",
                            "padding_mode": "reflect",
                        },
                    }
                },
            }
        )

        resolved = resolve_inference_policy(cfg)

        self.assertEqual(resolved.source, "legacy_validation")
        self.assertTrue(resolved.policy.sliding_window.enabled)
        self.assertEqual(resolved.policy.sliding_window.roi_size, (48, 48, 48))
        self.assertEqual(resolved.policy.sliding_window.sw_batch_size, 4)
        self.assertEqual(resolved.policy.sliding_window.overlap, 0.25)
        self.assertEqual(resolved.policy.sliding_window.blend_mode, "constant")
        self.assertEqual(resolved.policy.sliding_window.padding_mode, "reflect")

    def test_explicit_legacy_direct_mode_remains_compatible(self):
        cfg = OmegaConf.create(
            {
                "model": {"spatial_dims": "2d"},
                "data_mode": {
                    "dim": "2d",
                    "loader_mode": "online_slices_3d_to_2d",
                },
                "dataset": {
                    "preprocessing_configs": {"roi": {"slice_2d": [48, 48]}}
                },
                "validation": {
                    "inference": {
                        "mode": "direct",
                        "sliding_window": {"roi_size": None},
                    }
                },
            }
        )

        resolved = resolve_inference_policy(cfg)

        self.assertEqual(resolved.source, "legacy_validation")
        self.assertFalse(resolved.policy.sliding_window.enabled)
        self.assertEqual(resolved.policy.sliding_window.roi_size, (48, 48))

    def test_legacy_roi_override_must_match_model_contract(self):
        cfg = OmegaConf.create(
            {
                "model": {"spatial_dims": "3d"},
                "data_mode": {
                    "dim": "3d",
                    "loader_mode": "full_volumes_3d",
                },
                "dataset": {
                    "preprocessing_configs": {
                        "roi": {"volume_3d": [64, 64, 64]}
                    }
                },
                "validation": {
                    "inference": {
                        "mode": "sliding_window",
                        "sliding_window": {"roi_size": [32, 32, 32]},
                    }
                },
            }
        )

        with self.assertRaisesRegex(
            InvalidInferencePolicyError,
            "Historical validation inference requested.*requires",
        ):
            resolve_inference_policy(cfg)

    def test_canonical_policy_files_parse(self):
        config_dir = REPOSITORY_ROOT / "configs" / "inference"
        expected = {
            "direct_model_space": ("model_preprocessed", "fp32", False),
            "sliding_window_model_space": ("model_preprocessed", "fp32", True),
            "sliding_window_native": ("native_input", "fp32", True),
            "sliding_window_native_fp16": ("native_input", "fp16", True),
        }

        data_modes = {
            "online_slices_3d_to_2d": (64, 64),
            "full_volumes_3d": (64, 64, 64),
        }
        for config_name, expected_values in expected.items():
            for data_mode, expected_roi in data_modes.items():
                with self.subTest(config_name=config_name, data_mode=data_mode):
                    with hydra.initialize_config_dir(
                        config_dir=str(REPOSITORY_ROOT / "configs"),
                        version_base=None,
                    ):
                        composed = hydra.compose(
                            config_name=None,
                            overrides=[
                                f"+data_mode={data_mode}",
                                "+model=dynunet_base",
                                "+dataset=isles26_modalities_t1raw",
                                f"+inference={config_name}",
                            ],
                        )
                    policy = resolve_inference_policy(composed).policy
                    observed = (
                        policy.output_space,
                        policy.precision,
                        policy.sliding_window.enabled,
                    )
                    self.assertEqual(observed, expected_values)
                    self.assertEqual(policy.sliding_window.roi_size, expected_roi)
                    self.assertEqual(policy.sliding_window.sw_batch_size, 1)
                    merged_policy = composed.inference
                    self.assertNotIn("roi_size", merged_policy.sliding_window)
                    self.assertNotIn("schema_version", merged_policy)
                    self.assertNotIn("deterministic", merged_policy)
                    self.assertNotIn("output", merged_policy)
                    for feature_name in (
                        "tta",
                        "ensemble",
                        "postprocessing",
                        "artifacts",
                    ):
                        self.assertEqual(
                            set(merged_policy[feature_name].keys()),
                            {"enabled"},
                        )

    def test_canonical_policy_files_use_narrow_inheritance_overrides(self):
        config_dir = REPOSITORY_ROOT / "configs" / "inference"
        expected_keys = {
            "direct_model_space.yaml": {"defaults", "sliding_window"},
            "sliding_window_native.yaml": {"defaults", "output_space"},
            "sliding_window_native_fp16.yaml": {"defaults", "precision"},
            "sliding_window_native_ensemble.yaml": {"defaults", "ensemble"},
        }

        for filename, keys in expected_keys.items():
            with self.subTest(filename=filename):
                raw_policy = OmegaConf.load(config_dir / filename)
                self.assertEqual(set(raw_policy.keys()), keys)

        self.assertEqual(
            {path.name for path in config_dir.glob("*.yaml")},
            {
                "direct_model_space.yaml",
                "sliding_window_model_space.yaml",
                "sliding_window_native.yaml",
                "sliding_window_native_fp16.yaml",
                "sliding_window_native_ensemble.yaml",
            },
        )

    def test_new_policy_rejects_roi_field_and_invalid_model_roi(self):
        cfg = OmegaConf.create(
            {
                "model": {"spatial_dims": "3d"},
                "data_mode": {"dim": "3d"},
                "dataset": {
                    "preprocessing_configs": {
                        "roi": {"volume_3d": [64, 64, 64]}
                    }
                },
                "inference": {
                    "sliding_window": {"roi_size": [64, 64, 64]}
                },
            }
        )

        with self.assertRaisesRegex(
            InvalidInferencePolicyError,
            "unknown keys.*roi_size",
        ):
            resolve_inference_policy(cfg)

        with self.assertRaisesRegex(
            InvalidInferencePolicyError,
            "must contain two or three positive integers",
        ):
            self._parse({}, model_roi=())

    def test_missing_policy_is_derived_from_model_contract(self):
        cfg = OmegaConf.create(
            {
                "model": {"spatial_dims": "2d"},
                "data_mode": {"dim": "2d"},
                "dataset": {
                    "preprocessing_configs": {"roi": {"slice_2d": [40, 40]}}
                },
            }
        )

        resolved = resolve_inference_policy(cfg)

        self.assertEqual(resolved.source, "model_contract")
        self.assertTrue(resolved.policy.sliding_window.enabled)
        self.assertEqual(resolved.policy.sliding_window.roi_size, (40, 40))


if __name__ == "__main__":
    unittest.main()
