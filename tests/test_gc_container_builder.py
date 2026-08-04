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
import SimpleITK as sitk

from scripts.gc_submission_builder.container_builder import (
    LOCAL_INVOKE_TIMEOUT_SECONDS,
    ContainerBuildError,
    _run_http_sidecar,
    _wait_for_health,
    build_container_image,
    save_container_image,
    test_container_image as run_container_test,
    validate_single_input_nifti_output,
    validate_single_input_output_set,
)
from scripts.gc_submission_builder.container_config import (
    ContainerConfigError,
    load_container_build_config,
)
from scripts.gc_submission_builder.runtime.interfaces import OutputBinding


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
    def test_default_config_is_linux_amd64_and_official_manifest_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_container_build_config(
                DEFAULT_CONFIG,
                overrides={"output_dir": Path(tmp)},
            )

        self.assertEqual(config.platform, "linux/amd64")
        self.assertEqual(config.image_reference, "medseg-diffusion-gc:isles26")
        self.assertEqual(config.interface_manifest_path.name, "isles26.yaml")
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
        self.assertIn(
            "INTERFACE_MANIFEST=scripts/gc_submission_builder/configs/interfaces/isles26.yaml",
            build,
        )
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
                overrides={
                    "output_dir": root / "artifacts",
                    "interface_manifest": (
                        PROJECT_ROOT
                        / "scripts"
                        / "gc_submission_builder"
                        / "configs"
                        / "interfaces"
                        / "fixture_single_nifti.yaml"
                    ),
                },
            )
            commands = []

            def fake_run(command, **kwargs):
                commands.append(list(command))
                if command[:2] == ["docker", "run"]:
                    return _completed(command, stdout="container-id\n")
                if command[:2] == ["docker", "exec"] and command[-2:] == ["id", "-u"]:
                    return _completed(command, stdout="1000\n")
                return _completed(command)

            def fake_invoke(*args, **kwargs):
                prediction = output / "images" / "fixture-output" / "output.nii.gz"
                prediction.parent.mkdir(parents=True, exist_ok=True)
                output_image = nib.Nifti1Image(
                    np.zeros((3, 4, 5), dtype=np.uint8),
                    input_affine,
                )
                output_image.set_qform(input_affine, code=1)
                output_image.set_sform(input_affine, code=2)
                nib.save(output_image, prediction)

            with patch(
                "scripts.gc_submission_builder.container_builder._run",
                side_effect=fake_run,
            ), patch(
                "scripts.gc_submission_builder.container_builder._wait_for_health"
            ) as health, patch(
                "scripts.gc_submission_builder.container_builder._invoke_from_sidecar",
                side_effect=fake_invoke,
            ) as invoke:
                result = run_container_test(
                    config,
                    model_dir=model,
                    input_dir=input_root,
                    output_dir=output,
                )

        network_create = next(
            command for command in commands if command[:3] == ["docker", "network", "create"]
        )
        self.assertIn("--internal", network_create)
        network_name = network_create[-1]
        run = next(command for command in commands if command[:2] == ["docker", "run"])
        self.assertEqual(run[run.index("--network") + 1], network_name)
        self.assertIn("--read-only", run)
        self.assertIn("32g", run)
        self.assertIn("8", run)
        self.assertIn("all", run)
        self.assertTrue(any(value.endswith("dst=/opt/ml/model,readonly") for value in run))
        self.assertTrue(any(value.endswith("dst=/input,readonly") for value in run))
        health.assert_called_once()
        invoke.assert_called_once()
        self.assertEqual(invoke.call_args.kwargs["timeout_seconds"], 300)
        self.assertEqual(
            result.output_validations["opaque-fixture-segmentation"]["dtype"],
            "uint8",
        )
        self.assertTrue(result.external_http_tested)
        self.assertNotEqual(result.non_root_uid, 0)
        self.assertTrue(result.geometry_matches_native_input)

    def test_http_probe_uses_external_sidecar_and_exact_status_without_redirects(self):
        commands = []

        def fake_run(command, **kwargs):
            commands.append((list(command), dict(kwargs)))
            return _completed(command, stdout="200\n")

        with patch(
            "scripts.gc_submission_builder.container_builder._run",
            side_effect=fake_run,
        ):
            result = _run_http_sidecar(
                container_name="algorithm",
                image_reference="fixture:image",
                network_name="offline-network",
                method="POST",
                path="/invoke",
                expected_status=201,
                timeout_seconds=LOCAL_INVOKE_TIMEOUT_SECONDS,
            )

        command, kwargs = commands[0]
        self.assertEqual(result.returncode, 0)
        self.assertEqual(command[:2], ["docker", "run"])
        self.assertNotIn("exec", command)
        self.assertEqual(command[command.index("--network") + 1], "offline-network")
        rendered = " ".join(command)
        self.assertIn("http.client", rendered)
        self.assertNotIn("urllib.request", rendered)
        self.assertEqual(command[-5:], ["algorithm", "POST", "/invoke", "300", "201"])
        self.assertEqual(kwargs["timeout_seconds"], 305)

    def test_health_sidecar_rejects_redirect_status_immediately(self):
        redirected = _completed([], stdout="302\n", returncode=22)
        with patch(
            "scripts.gc_submission_builder.container_builder._run_http_sidecar",
            return_value=redirected,
        ), self.assertRaisesRegex(ContainerBuildError, "exact HTTP 200.*302"):
            _wait_for_health(
                "algorithm",
                image_reference="fixture:image",
                network_name="offline-network",
                timeout_seconds=300,
            )

    def test_external_mha_set_validation_requires_complete_native_grid(self):
        bindings = (
            OutputBinding(
                slug="segmentation",
                result_key="mask",
                relative_path="images/segmentation",
                file_type="mha",
            ),
            OutputBinding(
                slug="probability",
                result_key="probability",
                relative_path="images/probability",
                file_type="mha",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.nii.gz"
            affine = np.diag([1.25, 2.5, 3.75, 1.0])
            affine[:3, 3] = [9.0, -7.0, 3.0]
            nib.save(
                nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.float32), affine),
                input_path,
            )
            reference = sitk.ReadImage(str(input_path))
            arrays = {
                "segmentation": np.zeros((5, 4, 3), dtype=np.uint8),
                "probability": np.full((5, 4, 3), 0.25, dtype=np.float32),
            }
            for binding in bindings:
                directory = root / "output" / binding.relative_path
                directory.mkdir(parents=True)
                image = sitk.GetImageFromArray(arrays[binding.slug])
                image.CopyInformation(reference)
                sitk.WriteImage(image, str(directory / "output.mha"), True)

            paths, validations = validate_single_input_output_set(
                bindings=bindings,
                output_root=root / "output",
                input_path=input_path,
            )
            self.assertEqual(tuple(paths), ("segmentation", "probability"))
            self.assertTrue(all(item["compressed"] for item in validations.values()))

            (root / "output" / "images" / "probability" / "output.mha").unlink()
            with self.assertRaisesRegex(ContainerBuildError, "probability"):
                validate_single_input_output_set(
                    bindings=bindings,
                    output_root=root / "output",
                    input_path=input_path,
                )

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
