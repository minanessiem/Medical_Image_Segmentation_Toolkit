"""
Tests for repository-model evaluation config utilities.
"""

import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from scripts.evaluation.core.model_config import (
    load_evaluation_config,
    load_run_config,
    merge_evaluation_config,
    apply_evaluation_overrides,
    resolve_evaluation_output_dir,
    write_resolved_evaluation_config,
)


class TestEvaluationModelConfig(unittest.TestCase):
    def test_missing_run_config_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                load_run_config(Path(tmp))

        self.assertIn(".hydra", str(ctx.exception))
        self.assertIn("config.yaml", str(ctx.exception))

    def test_load_run_config_from_synthetic_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            hydra_dir = run_dir / ".hydra"
            hydra_dir.mkdir(parents=True)
            (hydra_dir / "config.yaml").write_text(
                "dataset:\n"
                "  active_subsets:\n"
                "    val: val_fast\n"
                "data_mode:\n"
                "  dim: 3d\n",
                encoding="utf-8",
            )

            cfg = load_run_config(run_dir)
            OmegaConf.update(cfg, "dataset.active_subsets.val", "val_full")

        self.assertEqual(cfg.dataset.active_subsets.val, "val_full")
        self.assertEqual(cfg.data_mode.dim, "3d")

    def test_load_default_evaluation_config(self):
        cfg = load_evaluation_config("default")

        self.assertEqual(cfg.evaluation.input_source, "live_model")
        self.assertEqual(cfg.inference.output_space, "model_preprocessed")
        self.assertEqual(cfg.inference.sliding_window.sw_batch_size, 1)
        self.assertEqual(cfg.inference_runtime.profile, "native")
        self.assertEqual(cfg.evaluation.threshold_protocol.mode, "fixed")
        self.assertEqual(
            cfg.evaluation.threshold_protocol.primary.metric,
            "DiceNativeCoefficient",
        )

    def test_merge_evaluation_config_preserves_run_config(self):
        run_cfg = OmegaConf.create(
            {
                "dataset": {"active_subsets": {"val": "val_fast"}},
                "data_mode": {"dim": "3d"},
            }
        )
        eval_cfg = OmegaConf.create(
            {
                "evaluation": {
                    "run_dir": "/runs/example",
                    "model_name": "best_model",
                }
            }
        )

        merged = merge_evaluation_config(run_cfg, eval_cfg)

        self.assertEqual(merged.dataset.active_subsets.val, "val_fast")
        self.assertEqual(merged.data_mode.dim, "3d")
        self.assertEqual(merged.evaluation.run_dir, "/runs/example")
        self.assertEqual(merged.evaluation.model_name, "best_model")

    def test_explicit_evaluation_inference_replaces_saved_policy_as_a_whole(self):
        run_cfg = OmegaConf.create(
            {
                "inference": {
                    "output_space": "native_input",
                    "stale_saved_field": "must_not_survive",
                    "sliding_window": {"sw_batch_size": 4},
                },
                "validation": {"inference": {"mode": "direct"}},
            }
        )
        eval_cfg = load_evaluation_config("default")

        merged = merge_evaluation_config(run_cfg, eval_cfg)

        self.assertEqual(merged.inference.output_space, "model_preprocessed")
        self.assertEqual(merged.inference.sliding_window.sw_batch_size, 1)
        self.assertIsNone(
            OmegaConf.select(merged, "inference.stale_saved_field", default=None)
        )
        self.assertEqual(merged.validation.inference.mode, "direct")

    def test_apply_dotted_override(self):
        cfg = OmegaConf.create(
            {
                "dataset": {"active_subsets": {"val": "val_fast"}},
                "evaluation": {"checkpoint": {"use_ema": False}},
                "validation": {"val_batch_size": 16},
            }
        )

        updated = apply_evaluation_overrides(
            cfg,
            [
                "dataset.active_subsets.val=val_full",
                "evaluation.checkpoint.use_ema=true",
                "validation.val_batch_size=1",
                "evaluation.metrics_3d.names=[DiceNativeCoefficient,SurfaceDiceMonai]",
            ],
        )

        self.assertEqual(updated.dataset.active_subsets.val, "val_full")
        self.assertTrue(updated.evaluation.checkpoint.use_ema)
        self.assertEqual(updated.validation.val_batch_size, 1)
        self.assertIsInstance(updated.validation.val_batch_size, int)
        self.assertEqual(
            list(updated.evaluation.metrics_3d.names),
            ["DiceNativeCoefficient", "SurfaceDiceMonai"],
        )

    def test_apply_validation_group_override_changes_assessment_not_inference(self):
        cfg = OmegaConf.create(
            {
                "inference": {
                    "output_space": "model_preprocessed",
                    "sliding_window": {"enabled": True},
                },
                "validation": {
                    "inference": {"mode": "direct"},
                    "metrics": [{"name": "SliceWiseMetricsAggregator"}],
                }
            }
        )

        updated = apply_evaluation_overrides(
            cfg,
            ["validation=sliding_window_3d_metrics_full"],
        )

        self.assertTrue(updated.inference.sliding_window.enabled)
        self.assertEqual(updated.validation.inference.mode, "direct")
        self.assertEqual(
            updated.validation.metrics[0].name,
            "ThreeDMetricsAggregator",
        )

    def test_apply_inference_group_override_replaces_complete_policy(self):
        cfg = OmegaConf.create(
            {
                "inference": {
                    "output_space": "model_preprocessed",
                    "stale_field": True,
                }
            }
        )

        updated = apply_evaluation_overrides(cfg, ["inference=sliding_window_native"])

        self.assertEqual(updated.inference.output_space, "native_input")
        self.assertIsNone(OmegaConf.select(updated, "inference.stale_field", default=None))

    def test_resolve_default_output_dir(self):
        cfg = OmegaConf.create(
            {
                "evaluation": {
                    "run_dir": "/tmp/run_a",
                    "model_name": "best_model",
                    "output_dir": None,
                }
            }
        )

        output_dir = resolve_evaluation_output_dir(cfg, timestamp="2026-06-12_14-00-00")

        self.assertEqual(
            output_dir,
            Path("/tmp/run_a")
            / "analysis"
            / "evaluation_v3"
            / "best_model_2026-06-12_14-00-00",
        )

    def test_explicit_output_dir_wins(self):
        cfg = OmegaConf.create(
            {
                "evaluation": {
                    "run_dir": "/tmp/run_a",
                    "model_name": "best_model",
                    "output_dir": "/tmp/custom_eval",
                }
            }
        )

        self.assertEqual(resolve_evaluation_output_dir(cfg), Path("/tmp/custom_eval"))

    def test_write_resolved_evaluation_config(self):
        cfg = OmegaConf.create(
            {
                "dataset": {"active_subsets": {"val": "val_full"}},
                "evaluation": {"model_name": "best_model"},
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = write_resolved_evaluation_config(cfg, Path(tmp))
            self.assertTrue(path.exists())
            loaded = OmegaConf.load(path)

        self.assertEqual(loaded.dataset.active_subsets.val, "val_full")
        self.assertEqual(loaded.evaluation.model_name, "best_model")


if __name__ == "__main__":
    unittest.main(verbosity=2)
