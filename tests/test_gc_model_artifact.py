"""Deterministic and strict model-artifact contracts for Cut 9."""

from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from omegaconf import OmegaConf

from scripts.gc_submission_builder.build_config import ModelArtifactBuildConfig
from scripts.gc_submission_builder.model_artifact import (
    ARTIFACT_DIRECTORY_NAME,
    MODEL_BUILD_REPORT_NAME,
    ModelArtifactError,
    build_model_artifact,
    resolve_release_checkpoint,
    validate_model_artifact,
)
from scripts.gc_submission_builder.release_manifest import (
    ARTIFACT_FILENAMES,
    ArtifactManifestError,
    sha256_file,
    verify_artifact_manifest,
)
from src.inference.contracts import PredictorCapabilities


def _saved_config(*, diffusion_type: str = "Discriminative", spatial_dims: int = 3, dataset_id: str = "isles26") -> dict:
    return {
        "dataset": {
            "id": dataset_id,
            "modalities": ["T1_RAW"],
            "num_modalities": 1,
            "preprocessing_configs": {
                "common": {
                    "orientation": {"enabled": True, "axcodes": "RAS"},
                    "spacing": {
                        "enabled": True,
                        "allow_native_spacing": False,
                        "pixdim": [1.0, 1.0, 1.0],
                        "interpolation": {"image": "bilinear", "label": "nearest"},
                    },
                },
                "roi": {"volume_3d": [8, 8, 8], "slice_2d": [8, 8]},
                "full_volumes_3d": {"pad_to_divisible": False},
            },
        },
        "data_mode": {"dim": f"{spatial_dims}d", "loader_mode": "full_volumes_3d"},
        "model": {
            "image_size": 8,
            "spatial_dims": spatial_dims,
            "image_channels": 1,
            "out_channels": 1,
        },
        "diffusion": {"type": diffusion_type},
        "validation": {
            "inference": {
                "mode": "sliding_window",
                "sliding_window": {"roi_size": None, "sw_batch_size": 4},
            }
        },
    }


def _write_source_tree(root: Path, **config_options) -> tuple[Path, Path, Path]:
    run_dir = root / "selected_run"
    (run_dir / ".hydra").mkdir(parents=True)
    config_path = run_dir / ".hydra" / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_saved_config(**config_options), sort_keys=False),
        encoding="utf-8",
    )
    checkpoint_dir = run_dir / "models" / "best"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "selected_model.pth"
    checkpoint_path.write_bytes(b"exact selected checkpoint\n")
    policy_path = root / "policy.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "output_space": "native_input",
                "precision": "fp32",
                "sliding_window": {
                    "enabled": True,
                    "sw_batch_size": 1,
                    "overlap": 0.5,
                    "blend_mode": "gaussian",
                    "padding_mode": "constant",
                },
                "tta": {"enabled": False},
                "ensemble": {"enabled": False},
                "decision": {"threshold": 0.5},
                "postprocessing": {"enabled": False},
                "artifacts": {"enabled": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return run_dir, checkpoint_path, policy_path


def _build_config(run_dir: Path, policy_path: Path, output_dir: Path) -> ModelArtifactBuildConfig:
    return ModelArtifactBuildConfig(
        run_dir=run_dir,
        checkpoint="selected_model",
        use_ema=False,
        inference_policy_path=policy_path,
        output_dir=output_dir,
        archive_name="algorithmmodel.tar.gz",
        validation_device="cpu",
        code_commit="test-commit",
        code_dirty=False,
        created_at_utc="2000-01-01T00:00:00Z",
    )


class TestGcModelArtifact(unittest.TestCase):
    def test_release_checkpoint_selection_rejects_ambiguity_and_accepts_exact_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, expected, _policy = _write_source_tree(Path(tmp))
            duplicate = run_dir / "models" / "checkpoints" / expected.name
            duplicate.parent.mkdir(parents=True)
            duplicate.write_bytes(b"different")

            with self.assertRaisesRegex(ModelArtifactError, "ambiguous"):
                resolve_release_checkpoint(run_dir, "selected_model")

            selected = resolve_release_checkpoint(
                run_dir,
                "models/best/selected_model.pth",
            )
            self.assertEqual(selected, expected.resolve())

            with self.assertRaisesRegex(ModelArtifactError, "not found"):
                resolve_release_checkpoint(run_dir, "missing_model")

    def test_build_is_deterministic_and_archive_has_no_enclosing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, checkpoint, policy = _write_source_tree(root)
            outputs = [root / "out_a", root / "out_b"]
            validation = SimpleNamespace(as_dict=lambda: {"status": "passed"})

            with patch(
                "scripts.gc_submission_builder.model_artifact.validate_model_artifact",
                return_value=validation,
            ):
                results = [
                    build_model_artifact(_build_config(run_dir, policy, output))
                    for output in outputs
                ]

            self.assertEqual(
                sha256_file(results[0].archive_path),
                sha256_file(results[1].archive_path),
            )
            self.assertEqual(
                (results[0].artifact_dir / "config.yaml").read_bytes(),
                (run_dir / ".hydra" / "config.yaml").read_bytes(),
            )
            self.assertEqual(
                (results[0].artifact_dir / "weights.pth").read_bytes(),
                checkpoint.read_bytes(),
            )
            with tarfile.open(results[0].archive_path, "r:gz") as archive:
                self.assertEqual(sorted(archive.getnames()), sorted(ARTIFACT_FILENAMES))
                self.assertTrue(all(member.mtime == 0 for member in archive.getmembers()))
            manifest = json.loads(
                (results[0].artifact_dir / "artifact_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["source_run"], run_dir.name)
            self.assertNotIn(str(run_dir.resolve()), json.dumps(manifest))
            archived_policy = yaml.safe_load(
                (results[0].artifact_dir / "inference_policy.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotIn("defaults", archived_policy)
            self.assertEqual(archived_policy["sliding_window"]["sw_batch_size"], 1)
            report = json.loads(results[0].report_path.read_text(encoding="utf-8"))
            self.assertEqual(Path(report["source"]["run_dir"]), run_dir.resolve())
            self.assertEqual(results[0].report_path.name, MODEL_BUILD_REPORT_NAME)
            self.assertEqual(results[0].artifact_dir.name, ARTIFACT_DIRECTORY_NAME)

    def test_manifest_detects_tampered_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, _checkpoint, policy = _write_source_tree(root)
            with patch(
                "scripts.gc_submission_builder.model_artifact.validate_model_artifact"
            ) as validate:
                validate.return_value = SimpleNamespace(as_dict=lambda: {})
                result = build_model_artifact(_build_config(run_dir, policy, root / "out"))
            verify_artifact_manifest(result.artifact_dir)
            (result.artifact_dir / "weights.pth").write_bytes(b"tampered")
            with self.assertRaisesRegex(ArtifactManifestError, "weights.*SHA-256"):
                verify_artifact_manifest(result.artifact_dir)

    def test_builder_refuses_to_write_outputs_inside_source_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, _checkpoint, policy = _write_source_tree(root)
            config = _build_config(run_dir, policy, run_dir / "release")

            with self.assertRaisesRegex(ModelArtifactError, "outside.*training run"):
                build_model_artifact(config)

    def test_validation_uses_archived_policy_strict_loader_predictor_and_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, _checkpoint, policy = _write_source_tree(root)
            with patch(
                "scripts.gc_submission_builder.model_artifact.validate_model_artifact"
            ) as validate:
                validate.return_value = SimpleNamespace(as_dict=lambda: {})
                result = build_model_artifact(_build_config(run_dir, policy, root / "out"))

            capabilities = PredictorCapabilities(
                model_family="discriminative",
                spatial_dims=3,
                input_channels=1,
                output_channels=1,
                supported_precisions=("fp16", "fp32"),
            )
            predictor = SimpleNamespace(capabilities=capabilities)
            producer = SimpleNamespace(adapter=SimpleNamespace(dataset_id="isles26"))
            with patch(
                "scripts.gc_submission_builder.model_artifact.load_model_strict",
                return_value=object(),
            ) as strict_load, patch(
                "scripts.gc_submission_builder.model_artifact.build_probability_predictor",
                return_value=predictor,
            ) as build_predictor, patch(
                "scripts.gc_submission_builder.model_artifact.build_case_producer",
                return_value=producer,
            ) as build_producer:
                validation = validate_model_artifact(result.artifact_dir, device="cpu")

            self.assertTrue(validation.strict_checkpoint_load)
            self.assertEqual(validation.dataset_id, "isles26")
            self.assertEqual(validation.output_space, "native_input")
            self.assertEqual(validation.runtime_profile, "gc_submission")
            self.assertEqual(validation.sliding_window_batch_size, 1)
            composed = strict_load.call_args.args[0]
            self.assertEqual(
                OmegaConf.select(composed, "validation.inference.sliding_window.sw_batch_size"),
                4,
            )
            self.assertEqual(
                OmegaConf.select(composed, "inference.sliding_window.sw_batch_size"),
                1,
            )
            build_predictor.assert_called_once()
            build_producer.assert_called_once_with(
                dataset_id="isles26",
                dataset_cfg=composed.dataset,
                load_labels=False,
            )

    def test_initial_release_rejects_generative_2d_and_unknown_dataset(self):
        invalid_cases = (
            ({"diffusion_type": "OpenAI_DDPM"}, "generative"),
            ({"spatial_dims": 2}, "3D"),
            ({"dataset_id": "unknown"}, "registered preprocessing adapter"),
        )
        for options, message in invalid_cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run_dir, _checkpoint, policy = _write_source_tree(root, **options)
                with patch(
                    "scripts.gc_submission_builder.model_artifact.validate_model_artifact"
                ) as validate:
                    validate.return_value = SimpleNamespace(as_dict=lambda: {})
                    result = build_model_artifact(
                        _build_config(run_dir, policy, root / "out")
                    )
                with self.assertRaisesRegex(ModelArtifactError, message):
                    validate_model_artifact(result.artifact_dir, device="cpu")

    def test_registered_isles24_and_isles26_artifacts_resolve_adapters(self):
        capabilities = PredictorCapabilities(
            model_family="discriminative",
            spatial_dims=3,
            input_channels=1,
            output_channels=1,
            supported_precisions=("fp16", "fp32"),
        )
        for dataset_id in ("isles24", "isles26"):
            with self.subTest(dataset_id=dataset_id), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run_dir, _checkpoint, policy = _write_source_tree(
                    root,
                    dataset_id=dataset_id,
                )
                with patch(
                    "scripts.gc_submission_builder.model_artifact.validate_model_artifact"
                ) as validate:
                    validate.return_value = SimpleNamespace(as_dict=lambda: {})
                    result = build_model_artifact(
                        _build_config(run_dir, policy, root / "out")
                    )
                with patch(
                    "scripts.gc_submission_builder.model_artifact.load_model_strict",
                    return_value=object(),
                ), patch(
                    "scripts.gc_submission_builder.model_artifact.build_probability_predictor",
                    return_value=SimpleNamespace(capabilities=capabilities),
                ), patch(
                    "scripts.gc_submission_builder.model_artifact.build_case_producer",
                    return_value=SimpleNamespace(
                        adapter=SimpleNamespace(dataset_id=dataset_id)
                    ),
                ):
                    checked = validate_model_artifact(result.artifact_dir, device="cpu")
                self.assertEqual(checked.preprocessing_adapter, dataset_id)

    def test_extracted_archive_revalidates_at_mount_equivalent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir, _checkpoint, policy = _write_source_tree(root)
            validation = SimpleNamespace(as_dict=lambda: {})
            with patch(
                "scripts.gc_submission_builder.model_artifact.validate_model_artifact",
                return_value=validation,
            ):
                result = build_model_artifact(_build_config(run_dir, policy, root / "out"))
            mounted = root / "opt_ml_model"
            mounted.mkdir()
            with tarfile.open(result.archive_path, "r:gz") as archive:
                archive.extractall(mounted)
            with patch(
                "scripts.gc_submission_builder.model_artifact.load_model_strict",
                return_value=object(),
            ), patch(
                "scripts.gc_submission_builder.model_artifact.build_probability_predictor",
                return_value=SimpleNamespace(
                    capabilities=PredictorCapabilities(
                        model_family="discriminative",
                        spatial_dims=3,
                        input_channels=1,
                        output_channels=1,
                        supported_precisions=("fp16", "fp32"),
                    )
                ),
            ), patch(
                "scripts.gc_submission_builder.model_artifact.build_case_producer",
                return_value=SimpleNamespace(
                    adapter=SimpleNamespace(dataset_id="isles26")
                ),
            ):
                revalidated = validate_model_artifact(mounted, device="cpu")
            self.assertTrue(revalidated.strict_checkpoint_load)


if __name__ == "__main__":
    unittest.main(verbosity=2)
