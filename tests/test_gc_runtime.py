"""Grand Challenge runtime composition and invocation tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np
from omegaconf import OmegaConf
import torch

from scripts.gc_submission_builder.runtime.inference import (
    GcInferenceRuntime,
    GcRuntimeError,
    initialize_runtime,
)
from scripts.gc_submission_builder.runtime.interfaces import load_interface_manifest
from scripts.gc_submission_builder.runtime.interfaces import validate_dataset_bindings
from src.inference.case_producer import build_case_producer
from src.inference.contracts import (
    PredictorCapabilities,
    PreprocessedCase,
    SpatialGeometry,
    SpatialTrace,
)
from src.inference.policy import parse_inference_policy
from src.inference.runtime import parse_inference_runtime


def _write_manifest(root: Path, *, dataset_key: str = "T1") -> Path:
    path = root / "interfaces.yaml"
    path.write_text(
        """interfaces:
  - name: fixture-interface
    inputs:
      - slug: arbitrary-image-socket
        dataset_key: %s
        relative_path: images/input
        file_type: nifti
        cardinality: one
    technical_inputs: []
    outputs:
      - slug: arbitrary-output-socket
        result_key: mask
        relative_path: images/output
        file_type: nifti
""" % dataset_key,
        encoding="utf-8",
    )
    return path


def _write_runtime_profile(root: Path, *, output_space: str = "native_input") -> Path:
    path = root / "runtime.yaml"
    path.write_text(
        f"""profile: gc_container_test
case_batch_size: 1
num_workers: 0
require_cuda: false
timeout_seconds: 600
constraints:
  allowed_output_spaces: [{output_space}]
  allowed_precisions: [fp32]
  allow_ground_truth: false
  allow_threshold_sweep: false
  allow_intermediate_artifacts: false
""",
        encoding="utf-8",
    )
    return path


def _write_model_files(root: Path, *, output_space: str = "native_input") -> Path:
    model = root / "model"
    model.mkdir()
    OmegaConf.save(
        OmegaConf.create(
            {
                "dataset": {
                    "id": "isles26",
                    "modalities": ["t1raw"],
                    "preprocessing_configs": {
                        "roi": {
                            "volume_3d": [8, 8, 8],
                            "slices_2d": [8, 8],
                        }
                    },
                },
                "model": {
                    "image_size": 8,
                    "spatial_dims": 3,
                    "image_channels": 1,
                    "out_channels": 1,
                },
                "diffusion": {"type": "Discriminative"},
            }
        ),
        model / "config.yaml",
    )
    (model / "inference_policy.yaml").write_text(
        f"""output_space: {output_space}
precision: fp32
sliding_window:
  enabled: true
  sw_batch_size: 1
  overlap: 0.5
  blend_mode: gaussian
  padding_mode: constant
tta: {{enabled: false}}
ensemble: {{enabled: false}}
decision: {{threshold: 0.5}}
postprocessing: {{enabled: false}}
artifacts: {{enabled: false}}
""",
        encoding="utf-8",
    )
    (model / "weights.pth").write_bytes(b"fixture")
    (model / "artifact_manifest.json").write_text("{}", encoding="utf-8")
    return model


class TestGcRuntime(unittest.TestCase):
    def test_fixture_manifests_bind_registered_isles24_and_isles26_adapters(self):
        cases = (
            ("isles24", "NCCT", "NCCT"),
            ("isles26", "T1_RAW", "T1"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset_id, modality, raw_key in cases:
                dataset_cfg = OmegaConf.load(
                    Path("configs/dataset") / f"{dataset_id}_base.yaml"
                )
                OmegaConf.update(dataset_cfg, "modalities", [modality], merge=False)
                model_cfg = OmegaConf.create({"model": {"image_size": 8}})
                composed = OmegaConf.merge(model_cfg, {"dataset": dataset_cfg})
                OmegaConf.resolve(composed)
                producer = build_case_producer(
                    dataset_id=dataset_id,
                    dataset_cfg=composed.dataset,
                    load_labels=False,
                )
                (root / dataset_id).mkdir()
                manifest = load_interface_manifest(
                    _write_manifest(root / dataset_id, dataset_key=raw_key)
                )
                validate_dataset_bindings(
                    manifest,
                    required_raw_keys=producer.required_raw_keys,
                )

                self.assertEqual(producer.adapter.dataset_id, dataset_id)
                self.assertEqual(producer.required_raw_keys, (raw_key,))

    def test_initialization_composes_artifact_shared_inference_and_unlabeled_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = _write_model_files(root)
            interface_path = _write_manifest(root)
            runtime_path = _write_runtime_profile(root)
            policy = parse_inference_policy(
                OmegaConf.load(model_dir / "inference_policy.yaml"),
                model_roi=(8, 8, 8),
            )
            capabilities = PredictorCapabilities(
                model_family="discriminative",
                spatial_dims=3,
                input_channels=1,
                output_channels=1,
                supported_precisions=("fp32",),
            )
            executor = SimpleNamespace(
                policy=policy,
                policy_source="explicit_top_level",
                predictor=SimpleNamespace(capabilities=capabilities),
            )
            producer = SimpleNamespace(
                required_raw_keys=("T1",),
                adapter=SimpleNamespace(dataset_id="isles26"),
            )
            artifact = {
                "config_sha256": "config-hash",
                "inference_policy_sha256": "policy-hash",
            }
            with patch(
                "scripts.gc_submission_builder.runtime.inference.verify_artifact_manifest",
                return_value=artifact,
            ), patch(
                "scripts.gc_submission_builder.runtime.inference.load_model_strict",
                return_value=object(),
            ) as load_model, patch(
                "scripts.gc_submission_builder.runtime.inference.build_model_probability_executor",
                return_value=executor,
            ), patch(
                "scripts.gc_submission_builder.runtime.inference.build_case_producer",
                return_value=producer,
            ) as build_producer:
                initialized = initialize_runtime(
                    model_dir=model_dir,
                    interface_manifest_path=interface_path,
                    runtime_profile_path=runtime_path,
                    device="cpu",
                )

        self.assertEqual(initialized.runtime_profile.profile, "gc_container_test")
        self.assertEqual(initialized.executor.policy.output_space, "native_input")
        self.assertEqual(initialized.device, torch.device("cpu"))
        load_model.assert_called_once()
        build_producer.assert_called_once_with(
            dataset_id="isles26",
            dataset_cfg=initialized.config.dataset,
            load_labels=False,
        )

    def test_initialization_fails_before_readiness_for_binding_or_cuda_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = _write_model_files(root)
            wrong_interface = _write_manifest(root, dataset_key="ADC")
            runtime_path = _write_runtime_profile(root)
            capabilities = PredictorCapabilities(
                model_family="discriminative",
                spatial_dims=3,
                input_channels=1,
                output_channels=1,
                supported_precisions=("fp32",),
            )
            policy = parse_inference_policy(
                OmegaConf.load(model_dir / "inference_policy.yaml"),
                model_roi=(8, 8, 8),
            )
            executor = SimpleNamespace(
                policy=policy,
                policy_source="explicit_top_level",
                predictor=SimpleNamespace(capabilities=capabilities),
            )
            producer = SimpleNamespace(
                required_raw_keys=("T1",),
                adapter=SimpleNamespace(dataset_id="isles26"),
            )
            artifact = {
                "config_sha256": "config-hash",
                "inference_policy_sha256": "policy-hash",
            }
            with patch(
                "scripts.gc_submission_builder.runtime.inference.verify_artifact_manifest",
                return_value=artifact,
            ), patch(
                "scripts.gc_submission_builder.runtime.inference.load_model_strict",
                return_value=object(),
            ), patch(
                "scripts.gc_submission_builder.runtime.inference.build_model_probability_executor",
                return_value=executor,
            ), patch(
                "scripts.gc_submission_builder.runtime.inference.build_case_producer",
                return_value=producer,
            ):
                with self.assertRaisesRegex(GcRuntimeError, "missing=.*T1.*unsupported=.*ADC"):
                    initialize_runtime(
                        model_dir=model_dir,
                        interface_manifest_path=wrong_interface,
                        runtime_profile_path=runtime_path,
                        device="cpu",
                    )

            production = root / "production.yaml"
            production.write_text(
                Path("configs/inference_runtime/gc_submission.yaml").read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
            with patch(
                "scripts.gc_submission_builder.runtime.inference.verify_artifact_manifest",
                return_value=artifact,
            ):
                with self.assertRaisesRegex(GcRuntimeError, "requires a CUDA device"):
                    initialize_runtime(
                        model_dir=model_dir,
                        interface_manifest_path=wrong_interface,
                        runtime_profile_path=production,
                        device="cpu",
                    )

    def test_diagnostic_output_space_override_recomposes_only_for_container_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = _write_model_files(root, output_space="native_input")
            interface_path = _write_manifest(root)
            runtime_path = _write_runtime_profile(
                root,
                output_space="model_preprocessed",
            )
            capabilities = PredictorCapabilities(
                model_family="discriminative",
                spatial_dims=3,
                input_channels=1,
                output_channels=1,
                supported_precisions=("fp32",),
            )
            producer = SimpleNamespace(
                required_raw_keys=("T1",),
                adapter=SimpleNamespace(dataset_id="isles26"),
            )
            artifact = {
                "config_sha256": "config-hash",
                "inference_policy_sha256": "policy-hash",
            }

            def build_executor(*, backend, cfg):
                del backend
                policy = parse_inference_policy(
                    cfg.inference,
                    model_roi=(8, 8, 8),
                )
                return SimpleNamespace(
                    policy=policy,
                    policy_source="explicit_top_level",
                    predictor=SimpleNamespace(capabilities=capabilities),
                )

            with patch(
                "scripts.gc_submission_builder.runtime.inference.verify_artifact_manifest",
                return_value=artifact,
            ), patch(
                "scripts.gc_submission_builder.runtime.inference.load_model_strict",
                return_value=object(),
            ), patch(
                "scripts.gc_submission_builder.runtime.inference.build_model_probability_executor",
                side_effect=build_executor,
            ), patch(
                "scripts.gc_submission_builder.runtime.inference.build_case_producer",
                return_value=producer,
            ):
                initialized = initialize_runtime(
                    model_dir=model_dir,
                    interface_manifest_path=interface_path,
                    runtime_profile_path=runtime_path,
                    device="cpu",
                    output_space_override="model_preprocessed",
                )

                self.assertEqual(
                    initialized.executor.policy.output_space,
                    "model_preprocessed",
                )
                self.assertEqual(
                    initialized.inference_policy_origin,
                    "diagnostic_output_space_override",
                )

                with self.assertRaisesRegex(
                    GcRuntimeError,
                    "only permitted.*gc_container_test",
                ):
                    initialize_runtime(
                        model_dir=model_dir,
                        interface_manifest_path=interface_path,
                        runtime_profile_path=Path(
                            "configs/inference_runtime/gc_submission.yaml"
                        ),
                        device="cpu",
                        output_space_override="model_preprocessed",
                    )

    def test_invoke_passes_only_canonical_raw_keys_to_shared_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = load_interface_manifest(_write_manifest(root))
            input_root = root / "input"
            image_dir = input_root / "images" / "input"
            image_dir.mkdir(parents=True)
            nib.save(
                nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.float32), np.eye(4)),
                image_dir / "protected-file-name.nii.gz",
            )
            (input_root / "inputs.json").write_text(
                json.dumps(
                    [
                        {
                            "socket": {
                                "slug": "arbitrary-image-socket",
                                "relative_path": "images/input",
                            }
                        }
                    ]
                ),
                encoding="utf-8",
            )
            geometry = SpatialGeometry.identity((3, 4, 5))
            case = PreprocessedCase(
                case_id="gc-invocation",
                image=torch.zeros((1, 1, 3, 4, 5)),
                spatial_trace=SpatialTrace(original=geometry, model=geometry),
                native_metadata={},
                reference_key=None,
            )
            producer = Mock()
            producer.preprocess.return_value = case
            producer.adapter.dataset_id = "isles26"
            policy = parse_inference_policy(
                {"output_space": "native_input"},
                model_roi=(3, 4, 5),
            )
            runtime = GcInferenceRuntime(
                config=OmegaConf.create({}),
                executor=SimpleNamespace(policy=policy, policy_source="explicit_top_level"),
                case_producer=producer,
                interface_manifest=manifest,
                runtime_profile=parse_inference_runtime(
                    OmegaConf.load(_write_runtime_profile(root))
                ),
                artifact_manifest=MappingProxyType({}),
                device=torch.device("cpu"),
            )
            with patch(
                "scripts.gc_submission_builder.runtime.inference.predict_preprocessed_case",
                return_value=object(),
            ) as predict, patch(
                "scripts.gc_submission_builder.runtime.inference.materialize_prediction_outputs",
                return_value={
                    "arbitrary-output-socket": {
                        "path": "protected-output",
                        "dtype": "uint8",
                    }
                },
            ):
                report = runtime.invoke(
                    input_root=input_root,
                    output_root=root / "output",
                )

        record = producer.preprocess.call_args.args[0]
        self.assertEqual(set(record), {"caseID", "T1"})
        self.assertNotIn("arbitrary-image-socket", record)
        self.assertEqual(record["caseID"], "gc-invocation")
        self.assertEqual(
            report.output_validations,
            {"arbitrary-output-socket": {"dtype": "uint8"}},
        )
        self.assertNotIn(
            "path", report.output_validations["arbitrary-output-socket"]
        )
        predict.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
