"""Container configuration and Docker command contract tests."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from scripts.gc_submission_builder.container_builder import (
    ContainerBuildError,
    build_container_image,
    save_container_image,
    test_container_image as run_container_test,
    validate_single_input_nifti_output,
)
from scripts.gc_submission_builder.container_config import (
    ContainerConfigError,
    load_container_build_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "scripts" / "gc_submission_builder" / "configs" / "container.yaml"
)


def _completed(command, *, stdout="", returncode=0):
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")


def _inspection_payload(*, architecture="amd64", user="algorithm:algorithm", label="invoke"):
    return json.dumps(
        [
            {
                "Id": "sha256:fixture-image",
                "Architecture": architecture,
                "Config": {
                    "User": user,
                    "Labels": {"org.grand-challenge.api-method": label},
                },
            }
        ]
    )


def _content_audit_payload(*, model_files=(), weight_files=()):
    return json.dumps(
        {
            "model_directory_files": list(model_files),
            "weight_like_files": list(weight_files),
        }
    )


class TestGcContainerBuilder(unittest.TestCase):
    def test_default_config_is_linux_amd64_and_fixture_manifest_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_container_build_config(
                DEFAULT_CONFIG,
                overrides={"output_dir": Path(tmp)},
            )

        self.assertEqual(config.platform, "linux/amd64")
        self.assertEqual(config.image_reference, "medseg-diffusion-gc:cut10")
        self.assertEqual(config.interface_manifest_path.name, "interface_manifest.fixture.yaml")
        self.assertTrue(config.dockerfile.is_file())

    def test_config_rejects_model_fields_and_non_amd64_platform(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "container.yaml"
            path.write_text(
                "image_name: fixture\nimage_tag: test\nmodel: {architecture: dynunet}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ContainerConfigError, "unknown keys.*model"):
                load_container_build_config(path)

            with self.assertRaisesRegex(ContainerConfigError, "linux/amd64"):
                load_container_build_config(
                    DEFAULT_CONFIG,
                    overrides={"platform": "linux/arm64"},
                )

    def test_build_pins_platform_manifest_and_release_inspection(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            config = load_container_build_config(
                DEFAULT_CONFIG,
                overrides={"output_dir": output},
            )
            commands = []

            def fake_run(command, **kwargs):
                commands.append(list(command))
                if command[:3] == ["docker", "image", "inspect"]:
                    return _completed(command, stdout=_inspection_payload())
                if command[:2] == ["docker", "run"]:
                    return _completed(command, stdout=_content_audit_payload())
                return _completed(command)

            with patch(
                "scripts.gc_submission_builder.container_builder._run",
                side_effect=fake_run,
            ):
                result = build_container_image(config)

            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        build = commands[0]
        self.assertEqual(build[:2], ["docker", "build"])
        self.assertIn("linux/amd64", build)
        self.assertIn("INTERFACE_MANIFEST=scripts/gc_submission_builder/configs/interface_manifest.fixture.yaml", build)
        self.assertFalse(report["model_embedded"])
        self.assertTrue(report["image"]["model_payload_audited"])
        self.assertEqual(report["image"]["architecture"], "amd64")
        self.assertEqual(report["image"]["api_method"], "invoke")
        audit = next(command for command in commands if command[:2] == ["docker", "run"])
        self.assertIn("none", audit)
        self.assertIn("--read-only", audit)

    def test_build_rejects_model_or_weight_files_embedded_in_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_container_build_config(
                DEFAULT_CONFIG,
                overrides={"output_dir": Path(tmp)},
            )

            def fake_run(command, **kwargs):
                if command[:3] == ["docker", "image", "inspect"]:
                    return _completed(command, stdout=_inspection_payload())
                if command[:2] == ["docker", "run"]:
                    return _completed(
                        command,
                        stdout=_content_audit_payload(
                            model_files=("/opt/ml/model/weights.pth",),
                            weight_files=("/opt/ml/model/weights.pth",),
                        ),
                    )
                return _completed(command)

            with patch(
                "scripts.gc_submission_builder.container_builder._run",
                side_effect=fake_run,
            ), self.assertRaisesRegex(ContainerBuildError, "embedded model payload"):
                build_container_image(config)

    def test_smoke_command_enforces_gc_resources_and_read_only_mounts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            input_root = root / "input"
            image_dir = input_root / "images" / "fixture-input"
            image_dir.mkdir(parents=True)
            input_affine = np.array(
                [
                    [0.0, -1.5, 0.0, 12.0],
                    [2.0, 0.0, 0.0, -4.0],
                    [0.0, 0.0, 3.0, 8.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            input_image = nib.Nifti1Image(
                np.zeros((3, 4, 5), dtype=np.float32),
                input_affine,
            )
            input_image.set_qform(input_affine, code=1)
            input_image.set_sform(input_affine, code=2)
            nib.save(input_image, image_dir / "input.nii.gz")
            (input_root / "inputs.json").write_text(
                json.dumps(
                    [
                        {
                            "socket": {
                                "slug": "opaque-fixture-image",
                                "relative_path": "images/fixture-input",
                            }
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "output"
            config = load_container_build_config(
                DEFAULT_CONFIG,
                overrides={"output_dir": root / "artifacts"},
            )
            commands = []

            def fake_run(command, **kwargs):
                commands.append(list(command))
                if command[:2] == ["docker", "run"]:
                    return _completed(command, stdout="container-id\n")
                if command[:3] == ["docker", "exec", command[2]] and command[-2:] == ["id", "-u"]:
                    return _completed(command, stdout="1000\n")
                if command[:2] == ["docker", "exec"] and "/invoke" in command[-1]:
                    prediction = output / "images" / "fixture-output" / "output.nii.gz"
                    prediction.parent.mkdir(parents=True, exist_ok=True)
                    output_image = nib.Nifti1Image(
                        np.zeros((3, 4, 5), dtype=np.uint8),
                        input_affine,
                    )
                    output_image.set_qform(input_affine, code=1)
                    output_image.set_sform(input_affine, code=2)
                    nib.save(output_image, prediction)
                    return _completed(command)
                return _completed(command)

            with patch(
                "scripts.gc_submission_builder.container_builder._run",
                side_effect=fake_run,
            ), patch(
                "scripts.gc_submission_builder.container_builder._wait_for_health"
            ):
                result = run_container_test(
                    config,
                    model_dir=model,
                    input_dir=input_root,
                    output_dir=output,
                )

        run = next(command for command in commands if command[:2] == ["docker", "run"])
        self.assertIn("none", run)
        self.assertIn("--read-only", run)
        self.assertIn("32g", run)
        self.assertIn("8", run)
        self.assertIn("all", run)
        self.assertTrue(any(value.endswith("dst=/opt/ml/model,readonly") for value in run))
        self.assertTrue(any(value.endswith("dst=/input,readonly") for value in run))
        self.assertEqual(result.output_dtype, "uint8")
        self.assertNotEqual(result.non_root_uid, 0)
        self.assertTrue(result.geometry_matches_native_input)

    def test_external_nifti_validation_rejects_geometry_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.nii.gz"
            output_path = root / "output.nii.gz"
            nib.save(
                nib.Nifti1Image(
                    np.zeros((3, 4, 5), dtype=np.float32),
                    np.eye(4),
                ),
                input_path,
            )
            wrong_affine = np.eye(4)
            wrong_affine[0, 3] = 10.0
            nib.save(
                nib.Nifti1Image(
                    np.zeros((3, 4, 5), dtype=np.uint8),
                    wrong_affine,
                ),
                output_path,
            )

            with self.assertRaisesRegex(ContainerBuildError, "affine"):
                validate_single_input_nifti_output(
                    output_path=output_path,
                    input_path=input_path,
                )

    def test_save_refuses_overwrite_and_gzips_docker_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_container_build_config(
                DEFAULT_CONFIG,
                overrides={"output_dir": root, "archive_name": "fixture.tar.gz"},
            )

            def fake_run(command, **kwargs):
                if command[:3] == ["docker", "image", "inspect"]:
                    return _completed(command, stdout=_inspection_payload())
                if command[:2] == ["docker", "run"]:
                    return _completed(command, stdout=_content_audit_payload())
                if command[:2] == ["docker", "save"]:
                    Path(command[command.index("--output") + 1]).write_bytes(b"docker-tar")
                return _completed(command)

            with patch(
                "scripts.gc_submission_builder.container_builder._run",
                side_effect=fake_run,
            ):
                archive = save_container_image(config)
                with gzip.open(archive, "rb") as handle:
                    self.assertEqual(handle.read(), b"docker-tar")
                with self.assertRaisesRegex(ContainerBuildError, "overwrite"):
                    save_container_image(config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
