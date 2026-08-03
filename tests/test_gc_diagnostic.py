"""Non-production container diagnostic contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.gc_submission_builder.runtime.diagnostic import main


class TestGcDiagnostic(unittest.TestCase):
    def test_diagnostic_can_retain_model_space_without_production_socket(self):
        result = SimpleNamespace(
            output_space="model_preprocessed",
            probability=torch.full((1, 1, 3, 4, 5), 0.25),
            mask=torch.zeros((1, 1, 3, 4, 5), dtype=torch.uint8),
        )
        prediction = SimpleNamespace(
            interface=SimpleNamespace(name="fixture-interface"),
            result=result,
            elapsed_seconds=1.25,
            peak_cuda_memory_bytes=1024,
        )
        runtime = SimpleNamespace(
            inference_policy_origin="diagnostic_output_space_override",
            runtime_profile=SimpleNamespace(
                profile="gc_container_test",
                constraints=SimpleNamespace(allow_intermediate_artifacts=True),
            ),
            predict=lambda **_kwargs: prediction,
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "scripts.gc_submission_builder.runtime.diagnostic.initialize_runtime",
            return_value=runtime,
        ) as initialize:
            output = Path(tmp) / "diagnostic"
            exit_code = main(
                [
                    "--model-dir", str(Path(tmp) / "model"),
                    "--interface-manifest", str(Path(tmp) / "interfaces.yaml"),
                    "--runtime-profile", str(Path(tmp) / "runtime.yaml"),
                    "--input-dir", str(Path(tmp) / "input"),
                    "--output-dir", str(output),
                    "--device", "cpu",
                    "--output-space", "model_preprocessed",
                    "--retain",
                ]
            )
            report = json.loads(
                (output / "diagnostic_report.json").read_text(encoding="utf-8")
            )
            probability = np.load(output / "probability.npy", allow_pickle=False)
            mask = np.load(output / "mask.npy", allow_pickle=False)

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["output_space"], "model_preprocessed")
        self.assertTrue(report["retained"])
        self.assertEqual(
            report["inference_policy_origin"],
            "diagnostic_output_space_override",
        )
        self.assertEqual(probability.shape, (1, 1, 3, 4, 5))
        self.assertEqual(mask.dtype, np.uint8)
        initialize.assert_called_once_with(
            model_dir=Path(tmp) / "model",
            interface_manifest_path=Path(tmp) / "interfaces.yaml",
            runtime_profile_path=Path(tmp) / "runtime.yaml",
            device="cpu",
            output_space_override="model_preprocessed",
        )

    def test_retain_fails_when_diagnostic_profile_forbids_artifacts(self):
        result = SimpleNamespace(
            output_space="model_preprocessed",
            probability=torch.zeros((1, 1, 2, 2, 2)),
            mask=torch.zeros((1, 1, 2, 2, 2), dtype=torch.uint8),
        )
        runtime = SimpleNamespace(
            inference_policy_origin="artifact",
            runtime_profile=SimpleNamespace(
                profile="gc_container_test",
                constraints=SimpleNamespace(allow_intermediate_artifacts=False),
            ),
            predict=lambda **_kwargs: SimpleNamespace(
                interface=SimpleNamespace(name="fixture-interface"),
                result=result,
                elapsed_seconds=0.1,
                peak_cuda_memory_bytes=None,
            ),
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "scripts.gc_submission_builder.runtime.diagnostic.initialize_runtime",
            return_value=runtime,
        ):
            with self.assertRaisesRegex(RuntimeError, "does not permit"):
                main(
                    [
                        "--output-dir", str(Path(tmp) / "diagnostic"),
                        "--retain",
                    ]
                )

    def test_diagnostic_rejects_production_output_root_and_descendants(self):
        for output in (Path("/output"), Path("/output") / "diagnostic"):
            with self.subTest(output=output), patch(
                "scripts.gc_submission_builder.runtime.diagnostic.initialize_runtime"
            ) as initialize:
                with self.assertRaisesRegex(RuntimeError, "production /output"):
                    main(["--output-dir", str(output)])
            initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
