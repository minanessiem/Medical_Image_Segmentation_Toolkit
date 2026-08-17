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
        self.assertEqual(config.archive_name, "algorithmmodel.tar.gz")
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

    def test_member_list_is_the_only_source_of_ensemble_cardinality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "policy.yaml").write_text(
                "ensemble: {enabled: true, method: mean}\n",
                encoding="utf-8",
            )
            path = root / "builder.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "members": [
                            {
                                "id": f"fold{index}",
                                "run_dir": f"runs/fold{index}",
                                "checkpoint": "best_model",
                            }
                            for index in range(1, 4)
                        ],
                        "inference_policy": "policy.yaml",
                        "output_dir": "out",
                    }
                ),
                encoding="utf-8",
            )

            config = load_model_artifact_build_config(path)

            self.assertEqual(config.run_dir, None)
            self.assertEqual(config.checkpoint, None)
            self.assertEqual(
                tuple(member.member_id for member in config.members),
                ("fold1", "fold2", "fold3"),
            )
            self.assertEqual(config.archive_name, "algorithmmodel.tar.gz")

            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            raw["num_models"] = 3
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaisesRegex(BuilderConfigError, "unknown keys.*num_models"):
                load_model_artifact_build_config(path)

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

    def test_xy_tta_ensemble_policy_materializes_three_views_per_model(self):
        policy = compose_inference_policy_file(
            PROJECT_ROOT
            / "configs"
            / "inference"
            / "sliding_window_native_ensemble_tta_xy.yaml"
        )

        self.assertNotIn("defaults", policy)
        self.assertTrue(policy["ensemble"]["enabled"])
        self.assertEqual(policy["ensemble"]["method"], "mean")
        self.assertTrue(policy["tta"]["enabled"])
        self.assertEqual(policy["tta"]["flip_axes"], ["x", "y"])
        self.assertNotIn("view_count", policy["tta"])

    def test_policy_inheritance_cycle_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.yaml").write_text("defaults: [b]\n", encoding="utf-8")
            (root / "b.yaml").write_text("defaults: [a]\n", encoding="utf-8")

            with self.assertRaisesRegex(BuilderConfigError, "inheritance cycle"):
                compose_inference_policy_file(root / "a.yaml")

    def test_build_model_dispatches_to_cut9_core(self):
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

        self.assertEqual(load_config.call_count, 1)
        self.assertEqual(build.call_count, 1)

    def test_build_image_save_and_build_all_dispatch_separate_artifacts(self):
        model_result = SimpleNamespace(
            artifact_dir=Path("artifact"),
            archive_path=Path("model.tar.gz"),
            report_path=Path("model_build_report.json"),
        )
        image_result = SimpleNamespace(
            inspection=SimpleNamespace(
                image_reference="fixture:cut10",
                image_id="sha256:image",
            ),
            report_path=Path("container_build_report.json"),
        )
        model_args = [
            "--run-dir", "run", "--checkpoint", "best_model", "--output-dir", "out"
        ]
        with patch(
            "scripts.gc_submission_builder.cli.load_model_artifact_build_config",
            return_value=object(),
        ), patch(
            "scripts.gc_submission_builder.cli.build_model_artifact",
            return_value=model_result,
        ) as build_model, patch(
            "scripts.gc_submission_builder.cli.load_container_build_config",
            return_value=object(),
        ) as load_container, patch(
            "scripts.gc_submission_builder.cli.build_container_image",
            return_value=image_result,
        ) as build_image, patch(
            "scripts.gc_submission_builder.cli.save_container_image",
            return_value=Path("image.tar.gz"),
        ) as save_image:
            self.assertEqual(main(["build-image"]), 0)
            self.assertEqual(main(["save"]), 0)
            self.assertEqual(main(["build-all", *model_args]), 0)

        self.assertEqual(build_model.call_count, 1)
        self.assertEqual(load_container.call_count, 3)
        self.assertEqual(build_image.call_count, 2)
        self.assertEqual(save_image.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
