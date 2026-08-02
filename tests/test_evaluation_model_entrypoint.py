"""
Tests for the config-driven repository-model evaluation CLI.
"""

import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts.evaluation.evaluate_model import compose_evaluation_config, main


def _make_run_dir(tmp: str) -> Path:
    run_dir = Path(tmp) / "run"
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True)
    (hydra_dir / "config.yaml").write_text(
        "dataset:\n"
        "  active_subsets:\n"
        "    val: val_fast\n"
        "data_mode:\n"
        "  dim: 3d\n"
        "  loader_mode: full_volumes_3d\n"
        "diffusion:\n"
        "  type: Discriminative\n"
        "validation:\n"
        "  inference:\n"
        "    mode: direct\n",
        encoding="utf-8",
    )
    return run_dir


def _mock_results(tmp: str):
    output_dir = Path(tmp) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "output_dir": str(output_dir),
        "json_path": str(output_dir / "canonical_results.json"),
        "volume_csv_path": str(output_dir / "volume_metrics_per_threshold.csv"),
        "per_case_csv_path": str(output_dir / "per_case_threshold_metrics.csv"),
        "oracle_csv_path": None,
        "config_path": str(output_dir / "resolved_evaluation_config.yaml"),
        "summary_path": str(output_dir / "evaluation_summary.txt"),
        "summary_text": "summary body",
    }


class TestEvaluateModelEntrypoint(unittest.TestCase):
    def test_compose_minimal_invocation_merges_run_and_eval_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(tmp)
            cfg = compose_evaluation_config(
                evaluation_config_name="default",
                overrides=[
                    f"evaluation.run_dir={run_dir}",
                    "evaluation.model_name=best_model",
                ],
            )

        self.assertEqual(cfg.evaluation.run_dir, str(run_dir))
        self.assertEqual(cfg.evaluation.model_name, "best_model")
        self.assertEqual(cfg.evaluation.input_source, "live_model")
        self.assertEqual(cfg.inference.output_space, "model_preprocessed")
        self.assertEqual(cfg.inference_runtime.profile, "native")
        self.assertEqual(cfg.dataset.active_subsets.val, "val_fast")
        self.assertEqual(cfg.evaluation.threshold_protocol.mode, "fixed")

    def test_compose_evaluation_preset_override_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(tmp)
            cfg = compose_evaluation_config(
                evaluation_config_name="threshold_sweep_with_oracle",
                overrides=[
                    f"evaluation.run_dir={run_dir}",
                    "evaluation.model_name=best_model",
                ],
            )

        self.assertEqual(cfg.evaluation.threshold_protocol.mode, "sweep_with_oracle")
        self.assertEqual(cfg.evaluation.threshold_protocol.thresholds, "0.05:0.90:0.05")
        self.assertEqual(cfg.inference.output_space, "model_preprocessed")
        self.assertEqual(cfg.inference_runtime.profile, "native")

    def test_compose_trailing_overrides_are_applied_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(tmp)
            cfg = compose_evaluation_config(
                evaluation_config_name="default",
                overrides=[
                    f"evaluation.run_dir={run_dir}",
                    "evaluation.model_name=best_model",
                    "dataset.active_subsets.val=val_full",
                    "validation=sliding_window_3d_metrics_subset",
                    "evaluation.threshold_protocol.mode=sweep",
                ],
            )

        self.assertEqual(cfg.dataset.active_subsets.val, "val_full")
        self.assertTrue(cfg.inference.sliding_window.enabled)
        self.assertEqual(cfg.validation.inference.mode, "direct")
        self.assertEqual(cfg.evaluation.threshold_protocol.mode, "sweep")

    def test_main_calls_pipeline_and_prints_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _make_run_dir(tmp)
            stdout = StringIO()
            with patch(
                "scripts.evaluation.evaluate_model.run_model_evaluation",
                return_value=_mock_results(tmp),
            ) as run_mock, redirect_stdout(stdout):
                exit_code = main(
                    [
                        f"evaluation.run_dir={run_dir}",
                        "evaluation.model_name=best_model",
                    ]
                )

        self.assertEqual(exit_code, 0)
        run_mock.assert_called_once()
        output = stdout.getvalue()
        self.assertIn("Repository Model Evaluation", output)
        self.assertIn("canonical_results.json", output)
        self.assertIn("evaluation_summary.txt", output)
        self.assertIn("summary body", output)

    def test_main_missing_run_dir_returns_nonzero(self):
        stderr = StringIO()
        with redirect_stderr(stderr):
            exit_code = main(["evaluation.model_name=best_model"])

        self.assertEqual(exit_code, 1)
        self.assertIn("evaluation.run_dir is required", stderr.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
