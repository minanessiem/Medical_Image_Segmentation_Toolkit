"""
Tests for config-driven repository-model evaluation pipeline.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from scripts.evaluation.core.contracts import VolumeSample
from scripts.evaluation.core.evaluation_pipeline import (
    _evaluate_volume_sample,
    _resolve_volume_metric_names,
    build_model_evaluation_request,
    run_model_evaluation,
)
from src.inference.contracts import SpatialGeometry


class DummyModel(nn.Module):
    def forward(self, x):
        return x


def _make_cfg(tmp: str, mode: str = "fixed"):
    run_dir = Path(tmp) / "run"
    output_dir = Path(tmp) / "eval"
    checkpoint_dir = run_dir / "models" / "best"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best_model.pth").write_bytes(b"checkpoint")
    return OmegaConf.create(
        {
            "evaluation": {
                "input_source": "live_model",
                "run_dir": str(run_dir),
                "model_name": "best_model",
                "output_dir": str(output_dir),
                "device": "cpu",
                "levels": ["volume"],
                "checkpoint": {"use_ema": False},
                "threshold_protocol": {
                    "mode": mode,
                    "thresholds": "0.3,0.5",
                    "fixed_threshold": 0.5,
                    "primary": {
                        "level": "volume",
                        "metric": "DiceNativeCoefficient",
                        "statistic": "mean",
                        "direction": "max",
                    },
                },
                "metrics_3d": {
                    "names": [
                        "DiceNativeCoefficient",
                        "PredictedVolumeMm3",
                        "GroundTruthVolumeMm3",
                    ]
                },
                "reporting": {"write_config": True},
                "show_progress": False,
            },
            "data_mode": {
                "dim": "3d",
                "loader_mode": "full_volumes_3d",
            },
            "diffusion": {"type": "Discriminative"},
            "model": {
                "spatial_dims": 3,
                "image_channels": 1,
                "out_channels": 1,
            },
            "dataset": {
                "id": "isles26",
                "modalities": ["T1_RAW"],
                "active_subsets": {"val": "val_fast"},
                "preprocessing_configs": {
                    "roi": {
                        "slice_2d": [2, 2],
                        "volume_3d": [2, 2, 2],
                    }
                },
            },
            "inference": {
                "output_space": "model_preprocessed",
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
            "inference_runtime": {
                "profile": "native",
                "case_batch_size": 1,
                "num_workers": 0,
                "require_cuda": False,
                "timeout_seconds": None,
                "constraints": {
                    "allowed_output_spaces": ["model_preprocessed", "native_input"],
                    "allowed_precisions": ["fp16", "fp32", "bf16"],
                    "allow_ground_truth": True,
                    "allow_threshold_sweep": True,
                    "allow_intermediate_artifacts": True,
                },
            },
            "validation": {
                "val_batch_size": 1,
                "inference": {"mode": "direct"},
            },
        }
    )


def _volume_sample(case_id: str = "case_a"):
    geometry = SpatialGeometry(
        shape=(2, 2, 2),
        affine=(
            (1.0, 0.0, 0.0, 10.0),
            (0.0, 2.0, 0.0, 20.0),
            (0.0, 0.0, 3.0, 30.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        spacing=(1.0, 2.0, 3.0),
        orientation="RAS",
    )
    return VolumeSample(
        case_id=case_id,
        volume_id=case_id,
        prediction_volume=torch.ones(1, 2, 2, 2),
        ground_truth_volume=torch.ones(1, 2, 2, 2),
        prediction_space="model_preprocessed",
        reference_space="model_preprocessed",
        prediction_geometry=geometry,
        reference_geometry=geometry,
        metadata={"source": "test_volume"},
    )


def _mock_metric_values(pred, gt, threshold, metric_configs=None, metric_names=None):
    del pred, gt, metric_configs
    dice_by_threshold = {0.3: 0.4, 0.5: 0.8}
    values = {
        "DiceNativeCoefficient": dice_by_threshold[round(float(threshold), 1)],
        "SurfaceDiceMonai": 0.25,
        "HausdorffDistance95Native": 12.0,
        "PredictedVolumeMm3": 8.0,
        "GroundTruthVolumeMm3": 10.0,
    }
    if metric_names is None:
        return values
    return {name: values[name] for name in metric_names}




class TestEvaluationPipeline(unittest.TestCase):
    def test_build_request_validates_required_fields(self):
        cfg = OmegaConf.create(
            {
                "evaluation": {"model_name": "best_model"},
                "data_mode": {"dim": "3d"},
                "diffusion": {"type": "Discriminative"},
            }
        )

        with self.assertRaises(ValueError):
            build_model_evaluation_request(cfg)

    def test_build_request_resolves_checkpoint_and_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="sweep")
            request = build_model_evaluation_request(cfg)

        self.assertEqual(request.model_name, "best_model")
        self.assertEqual(request.device, "cpu")
        self.assertEqual(request.data_dim, "3d")
        self.assertEqual(request.diffusion_type, "Discriminative")
        self.assertEqual(request.threshold_protocol.mode, "sweep")
        self.assertEqual(request.inference_policy.output_space, "model_preprocessed")
        self.assertEqual(request.inference_policy_source, "explicit_top_level")
        self.assertEqual(request.inference_runtime.profile, "native")
        self.assertTrue(str(request.checkpoint_path).endswith("best_model.pth"))

    def test_input_source_does_not_select_runtime_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="fixed")
            cfg.inference_runtime.profile = "gc_container_test"

            request = build_model_evaluation_request(cfg)

        self.assertEqual(cfg.evaluation.input_source, "live_model")
        self.assertEqual(request.inference_runtime.profile, "gc_container_test")

    def test_native_output_request_is_accepted_after_spatial_restoration(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="fixed")
            cfg.inference.output_space = "native_input"

            request = build_model_evaluation_request(cfg)

        self.assertEqual(request.inference_policy.output_space, "native_input")
        self.assertEqual(request.inference_runtime.profile, "native")

    def test_complete_case_batch_must_be_one_for_both_output_spaces(self):
        for output_space in ("model_preprocessed", "native_input"):
            with self.subTest(output_space=output_space), tempfile.TemporaryDirectory() as tmp:
                cfg = _make_cfg(tmp, mode="fixed")
                cfg.inference.output_space = output_space
                cfg.validation.val_batch_size = 2

                with self.assertRaisesRegex(
                    ValueError,
                    "one complete case at a time.*many cases sequentially.*sw_batch_size",
                ):
                    build_model_evaluation_request(cfg)

    def test_sliding_window_batch_is_independent_of_complete_case_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="fixed")
            cfg.validation.val_batch_size = 1
            cfg.inference.sliding_window.sw_batch_size = 3

            request = build_model_evaluation_request(cfg)

        self.assertEqual(request.inference_policy.sliding_window.sw_batch_size, 3)

    def test_default_validation_config_uses_complete_case_batch_one(self):
        config_path = Path(__file__).parents[1] / "configs" / "validation" / "default.yaml"
        validation_cfg = OmegaConf.load(config_path)

        self.assertEqual(validation_cfg.val_batch_size, 1)

    def test_submission_runtime_rejects_labeled_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="fixed")
            cfg.inference.output_space = "native_input"
            cfg.inference_runtime = OmegaConf.create(
                {
                    "profile": "gc_submission",
                    "case_batch_size": 1,
                    "num_workers": 0,
                    "require_cuda": True,
                    "timeout_seconds": 600,
                    "constraints": {
                        "allowed_output_spaces": ["native_input"],
                        "allowed_precisions": ["fp16", "fp32"],
                        "allow_ground_truth": False,
                        "allow_threshold_sweep": False,
                        "allow_intermediate_artifacts": False,
                    },
                }
            )

            with self.assertRaisesRegex(RuntimeError, "does not permit ground truth"):
                build_model_evaluation_request(cfg)

    def test_build_request_normalizes_level_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="sweep")
            cfg.evaluation.levels = ["volumes"]
            cfg.evaluation.threshold_protocol.primary.level = "volumes"

            request = build_model_evaluation_request(cfg)

        self.assertEqual(list(request.levels), ["volume"])
        self.assertEqual(request.threshold_protocol.primary.level, "volume")

    def test_build_request_rejects_2d_until_reconstruction_contract_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="sweep")
            cfg.data_mode.dim = "2d"
            cfg.data_mode.loader_mode = "online_slices_3d_to_2d"
            cfg.evaluation.levels = ["slices"]
            cfg.evaluation.threshold_protocol.primary.level = "slices"
            cfg.evaluation.threshold_protocol.primary.metric = "Dice2DForegroundOnly"

            with self.assertRaisesRegex(ValueError, "deferred 2D reconstruction contract"):
                build_model_evaluation_request(cfg)

    def test_build_request_rejects_primary_level_not_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="sweep")
            cfg.evaluation.levels = ["slice"]
            cfg.evaluation.threshold_protocol.primary.level = "volume"

            with self.assertRaises(ValueError) as ctx:
                build_model_evaluation_request(cfg)

        self.assertIn("primary.level must be included", str(ctx.exception))

    def test_build_request_rejects_3d_slice_level_until_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="sweep")
            cfg.evaluation.levels = ["slice"]
            cfg.evaluation.threshold_protocol.primary.level = "slice"

            with self.assertRaises(ValueError) as ctx:
                build_model_evaluation_request(cfg)

        self.assertIn("3D live-model evaluation currently supports volume-level", str(ctx.exception))

    def test_unsupported_3d_diffusion_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="fixed")
            cfg.diffusion.type = "OpenAI_DDPM"
            with self.assertRaises(ValueError) as ctx:
                build_model_evaluation_request(cfg)

        self.assertIn("3D live-model evaluation", str(ctx.exception))

    def test_validation_metric_subset_resolves_to_class_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="fixed")
            del cfg.evaluation.metrics_3d
            cfg.validation.metrics = [
                {
                    "name": "ThreeDMetricsAggregator",
                    "params": {
                        "enabled_metrics": [
                            "dice_3d",
                            "surface_dice_monai_3d",
                            "hd95_3d",
                        ]
                    },
                }
            ]

            metric_names = _resolve_volume_metric_names(cfg)

        self.assertEqual(
            metric_names,
            (
                "DiceNativeCoefficient",
                "SurfaceDiceMonai",
                "HausdorffDistance95Native",
            ),
        )

    def test_evaluation_metric_override_rejects_validation_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="fixed")
            cfg.evaluation.metrics_3d.names = ["dice_3d"]

            with self.assertRaises(ValueError) as ctx:
                _resolve_volume_metric_names(cfg)

        self.assertIn("must use 3D metric class names", str(ctx.exception))

    def test_declared_spatial_fields_cannot_be_overridden_by_metadata(self):
        sample = _volume_sample()
        sample.metadata["prediction_space"] = "native_input"
        sample.metadata["prediction_geometry"] = {"shape": [99, 99, 99]}

        with patch(
            "scripts.evaluation.core.evaluation_pipeline.compute_metrics_3d_at_threshold",
            side_effect=_mock_metric_values,
        ):
            record = _evaluate_volume_sample(
                sample=sample,
                thresholds=[0.5],
                metric_names=["DiceNativeCoefficient"],
            )[0]

        self.assertEqual(record.metadata["prediction_space"], "model_preprocessed")
        self.assertEqual(record.metadata["prediction_geometry"]["shape"], [2, 2, 2])

    def test_fixed_protocol_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="fixed")
            result = self._run_mocked_pipeline(cfg)
            output_dir = Path(result["output_dir"])

            self.assertTrue((output_dir / "canonical_results.json").exists())
            self.assertTrue((output_dir / "evaluation_summary.txt").exists())
            self.assertTrue((output_dir / "resolved_evaluation_config.yaml").exists())
            self.assertTrue((output_dir / "volume_metrics_per_threshold.csv").exists())
            self.assertTrue((output_dir / "per_case_threshold_metrics.csv").exists())
            self.assertIsNone(result["oracle_csv_path"])

            payload = json.loads((output_dir / "canonical_results.json").read_text())

        self.assertEqual(payload["protocol"]["mode"], "fixed")
        self.assertEqual(payload["protocol"]["thresholds_evaluated"], [0.5])
        self.assertIsNone(payload["threshold_analysis"]["best_global_threshold"])
        self.assertEqual(payload["data_summary"]["total_volumes"], 1)
        self.assertEqual(payload["metadata"]["producer"], "repository_model_live")
        self.assertEqual(payload["metadata"]["runtime_profile"], "native")
        self.assertEqual(payload["metadata"]["inference_policy_source"], "explicit_top_level")
        self.assertEqual(payload["spatial_contract"]["prediction_space"], "model_preprocessed")
        self.assertEqual(
            payload["spatial_contract"]["samples"][0]["reference_geometry"]["spacing"],
            [1.0, 2.0, 3.0],
        )

    def test_pipeline_uses_validation_metric_subset_with_class_name_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="fixed")
            del cfg.evaluation.metrics_3d
            cfg.validation.metrics = [
                {
                    "name": "ThreeDMetricsAggregator",
                    "params": {
                        "enabled_metrics": [
                            "dice_3d",
                            "surface_dice_monai_3d",
                            "hd95_3d",
                        ]
                    },
                }
            ]
            result = self._run_mocked_pipeline(cfg)
            payload = json.loads(Path(result["json_path"]).read_text())

        metric_names = payload["metrics"]["volume_level"]["metric_names"]
        self.assertEqual(
            metric_names,
            [
                "DiceNativeCoefficient",
                "SurfaceDiceMonai",
                "HausdorffDistance95Native",
            ],
        )
        threshold_metrics = payload["metrics"]["volume_level"]["threshold_results"][0]["metrics"]
        self.assertIn("DiceNativeCoefficient", threshold_metrics)
        self.assertIn("SurfaceDiceMonai", threshold_metrics)
        self.assertIn("HausdorffDistance95Native", threshold_metrics)
        self.assertNotIn("dice_3d", threshold_metrics)

    def test_sweep_protocol_selects_global_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="sweep")
            result = self._run_mocked_pipeline(cfg)
            payload = json.loads(Path(result["json_path"]).read_text())

        best = payload["threshold_analysis"]["best_global_threshold"]
        self.assertEqual(best["threshold"], 0.5)
        self.assertAlmostEqual(best["selected_statistic_value"], 0.8)
        self.assertEqual(result["selected_global_threshold"], 0.5)

    def test_sweep_with_oracle_writes_oracle_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="sweep_with_oracle")
            result = self._run_mocked_pipeline(cfg)
            oracle_path = Path(result["oracle_csv_path"])
            self.assertTrue(oracle_path.exists())
            payload = json.loads(Path(result["json_path"]).read_text())
            with oracle_path.open("r", encoding="utf-8", newline="") as handle:
                oracle_rows = list(csv.DictReader(handle))

        self.assertEqual(len(oracle_rows), 1)
        self.assertEqual(oracle_rows[0]["case_id"], "case_a")
        self.assertEqual(float(oracle_rows[0]["threshold"]), 0.5)
        self.assertEqual(payload["threshold_analysis"]["oracle_per_case"]["case_count"], 1)

    def _run_mocked_pipeline(self, cfg):
        producer = Mock(name="labeled_case_producer")
        records = [{"caseID": "case_a"}]
        typed_case = object()
        producer.return_value = typed_case

        def iter_typed_cases(**kwargs):
            self.assertEqual(list(kwargs["cases"]), [typed_case])
            return iter([_volume_sample()])

        with patch(
            "scripts.evaluation.core.evaluation_pipeline.build_model_for_evaluation",
            return_value=DummyModel(),
        ), patch(
            "scripts.evaluation.core.evaluation_pipeline.build_model_probability_executor",
            return_value=Mock(name="executor"),
        ) as executor_mock, patch(
            "scripts.evaluation.core.evaluation_pipeline.load_case_records",
            return_value=records,
        ) as records_mock, patch(
            "scripts.evaluation.core.evaluation_pipeline.build_case_producer",
            return_value=producer,
        ) as producer_mock, patch(
            "scripts.evaluation.core.evaluation_pipeline.iter_model_volume_samples",
            side_effect=iter_typed_cases,
        ) as volume_samples_mock, patch(
            "scripts.evaluation.core.evaluation_pipeline.compute_metrics_3d_at_threshold",
            side_effect=_mock_metric_values,
        ):
            result = run_model_evaluation(cfg)
        records_mock.assert_called_once_with(cfg, subset_role="val", load_labels=True)
        producer_mock.assert_called_once_with(
            dataset_id="isles26",
            dataset_cfg=cfg.dataset,
            load_labels=True,
        )
        executor_mock.assert_called_once()
        call_kwargs = volume_samples_mock.call_args.kwargs
        self.assertNotIn("dataloader", call_kwargs)
        self.assertEqual(call_kwargs["total_cases"], 1)
        producer.assert_called_once_with(records[0])
        return result

if __name__ == "__main__":
    unittest.main(verbosity=2)
