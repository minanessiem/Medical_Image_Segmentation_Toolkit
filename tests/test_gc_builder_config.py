"""Configuration and CLI contracts for the Grand Challenge model builder."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from scripts.gc_submission_builder.build_config import (
    BuilderConfigError,
    compose_inference_policy_file,
    load_model_artifact_build_config,
)
from scripts.gc_submission_builder.cli import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestGcBuilderConfig(unittest.TestCase):
    def test_default_config_accepts_only_builder_owned_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "full_model_directory_name"
            output_dir = root / "out"
            config = load_model_artifact_build_config(
                PROJECT_ROOT
                / "scripts"
                / "gc_submission_builder"
                / "configs"
                / "default.yaml",
                overrides={
                    "run_dir": run_dir,
                    "checkpoint": "best_model",
                    "output_dir": output_dir,
                },
            )

        self.assertEqual(config.run_dir, run_dir.resolve())
        self.assertEqual(config.checkpoint, "best_model")
        self.assertEqual(config.output_dir, output_dir.resolve())
        self.assertEqual(config.archive_name, "full_model_directory_name.tar.gz")
        self.assertEqual(config.validation_device, "cpu")
        self.assertTrue(config.inference_policy_path.is_file())
        self.assertEqual(config.inference_policy_path.name, "sliding_window_native_fp16.yaml")

    def test_explicit_archive_name_overrides_run_directory_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_model_artifact_build_config(
                PROJECT_ROOT
                / "scripts"
                / "gc_submission_builder"
                / "configs"
                / "default.yaml",
                overrides={
                    "run_dir": root / "model_directory",
                    "checkpoint": "best_model",
                    "output_dir": root / "out",
                    "archive_name": "manually_selected_name.tar.gz",
                },
            )

        self.assertEqual(config.archive_name, "manually_selected_name.tar.gz")

    def test_unknown_or_model_owned_builder_fields_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "builder.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "run_dir": "run",
                        "checkpoint": "model",
                        "inference_policy": "policy.yaml",
                        "output_dir": "out",
                        "model": {"image_size": 128},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(BuilderConfigError, "unknown keys.*model"):
                load_model_artifact_build_config(path)

    def test_inference_policy_inheritance_is_materialized_without_defaults(self):
        policy = compose_inference_policy_file(
            PROJECT_ROOT
            / "configs"
            / "inference"
            / "sliding_window_native_fp16.yaml"
        )

        self.assertNotIn("defaults", policy)
        self.assertEqual(policy["output_space"], "native_input")
        self.assertEqual(policy["precision"], "fp16")
        self.assertEqual(policy["sliding_window"]["sw_batch_size"], 1)
        self.assertFalse(policy["tta"]["enabled"])
        self.assertFalse(policy["ensemble"]["enabled"])
        self.assertFalse(policy["postprocessing"]["enabled"])

    def test_policy_inheritance_cycle_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.yaml").write_text("defaults: [b]\n", encoding="utf-8")
            (root / "b.yaml").write_text("defaults: [a]\n", encoding="utf-8")

            with self.assertRaisesRegex(BuilderConfigError, "inheritance cycle"):
                compose_inference_policy_file(root / "a.yaml")

    def test_build_model_and_build_all_dispatch_to_the_same_cut9_core(self):
        result = SimpleNamespace(
            artifact_dir=Path("artifact"),
            archive_path=Path("model_directory.tar.gz"),
            report_path=Path("model_build_report.json"),
        )
        common = [
            "--run-dir",
            "run",
            "--checkpoint",
            "best_model",
            "--output-dir",
            "out",
        ]
        with patch(
            "scripts.gc_submission_builder.cli.load_model_artifact_build_config"
        ) as load_config, patch(
            "scripts.gc_submission_builder.cli.build_model_artifact",
            return_value=result,
        ) as build:
            load_config.return_value = object()
            self.assertEqual(main(["build-model", *common]), 0)
            self.assertEqual(main(["build-all", *common]), 0)

        self.assertEqual(load_config.call_count, 2)
        self.assertEqual(build.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
