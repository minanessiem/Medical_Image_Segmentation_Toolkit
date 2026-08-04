"""Grand Challenge interface-manifest and invocation dispatch contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from scripts.gc_submission_builder.runtime.interfaces import (
    InterfaceManifestError,
    load_interface_manifest,
    resolve_invocation,
    validate_dataset_bindings,
)


class TestGcInterfaces(unittest.TestCase):
    def _write_manifest(
        self,
        root: Path,
        *,
        slug: str = "opaque-fixture-image",
        dataset_key: str = "T1",
    ) -> Path:
        path = root / f"{slug}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "interfaces": [
                        {
                            "name": "fixture-nifti-interface",
                            "inputs": [
                                {
                                    "slug": slug,
                                    "dataset_key": dataset_key,
                                    "relative_path": "images/fixture-input",
                                    "file_type": "nifti",
                                    "cardinality": "one",
                                }
                            ],
                            "technical_inputs": [],
                            "outputs": [
                                {
                                    "slug": "opaque-fixture-segmentation",
                                    "result_key": "mask",
                                    "relative_path": "images/fixture-output",
                                    "file_type": "nifti",
                                }
                            ],
                        }
                    ]
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _write_invocation(root: Path, *, slug: str) -> Path:
        input_root = root / "input"
        image_dir = input_root / "images" / "fixture-input"
        image_dir.mkdir(parents=True)
        image = nib.Nifti1Image(
            np.zeros((5, 6, 7), dtype=np.float32),
            affine=np.eye(4, dtype=np.float64),
        )
        nib.save(image, image_dir / "untrusted-platform-name.nii.gz")
        (input_root / "inputs.json").write_text(
            json.dumps(
                [
                    {
                        "socket": {
                            "slug": slug,
                            "relative_path": "images/fixture-input",
                            "is_image_kind": True,
                            "is_json_kind": False,
                            "is_file_kind": False,
                        },
                        "file": None,
                        "image": {"name": "protected-original-name"},
                        "value": None,
                    }
                ]
            ),
            encoding="utf-8",
        )
        return input_root

    def test_arbitrary_socket_slug_maps_only_to_canonical_dataset_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_manifest = load_interface_manifest(
                self._write_manifest(root, slug="first-unrelated-slug")
            )
            first = resolve_invocation(
                first_manifest,
                self._write_invocation(root / "first", slug="first-unrelated-slug"),
            )
            second_manifest = load_interface_manifest(
                self._write_manifest(root, slug="second-unrelated-slug")
            )
            second = resolve_invocation(
                second_manifest,
                self._write_invocation(root / "second", slug="second-unrelated-slug"),
            )

        self.assertEqual(tuple(first.raw_modalities), ("T1",))
        self.assertEqual(tuple(second.raw_modalities), ("T1",))
        self.assertNotIn("first-unrelated-slug", first.raw_modalities)
        self.assertNotIn("second-unrelated-slug", second.raw_modalities)
        self.assertEqual(first.interface.name, "fixture-nifti-interface")

    def test_interface_key_is_exact_and_rejects_missing_or_extra_sockets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = load_interface_manifest(self._write_manifest(root))
            input_root = self._write_invocation(root, slug="opaque-fixture-image")
            inputs_path = input_root / "inputs.json"

            inputs_path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(InterfaceManifestError, "No configured interface"):
                resolve_invocation(manifest, input_root)

            inputs_path.write_text(
                json.dumps(
                    [
                        {"socket": {"slug": "opaque-fixture-image", "relative_path": "images/fixture-input"}},
                        {"socket": {"slug": "unexpected", "relative_path": "unexpected.json"}},
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(InterfaceManifestError, "No configured interface"):
                resolve_invocation(manifest, input_root)

    def test_invocation_rejects_relative_path_disagreement_and_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = load_interface_manifest(self._write_manifest(root))
            input_root = self._write_invocation(root, slug="opaque-fixture-image")
            inputs_path = input_root / "inputs.json"
            payload = json.loads(inputs_path.read_text(encoding="utf-8"))
            payload[0]["socket"]["relative_path"] = "images/wrong"
            inputs_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(InterfaceManifestError, "relative_path"):
                resolve_invocation(manifest, input_root)

            payload[0]["socket"]["relative_path"] = "images/fixture-input"
            inputs_path.write_text(json.dumps(payload), encoding="utf-8")
            image_dir = input_root / "images" / "fixture-input"
            nib.save(
                nib.Nifti1Image(np.ones((5, 6, 7)), np.eye(4)),
                image_dir / "second.nii.gz",
            )
            with self.assertRaisesRegex(InterfaceManifestError, "exactly one"):
                resolve_invocation(manifest, input_root)

    def test_dataset_binding_contract_requires_exact_unique_raw_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = load_interface_manifest(self._write_manifest(root))

            validate_dataset_bindings(manifest, required_raw_keys=("T1",))
            with self.assertRaisesRegex(InterfaceManifestError, "missing.*ADC"):
                validate_dataset_bindings(
                    manifest,
                    required_raw_keys=("T1", "ADC"),
                )

            duplicate_path = self._write_manifest(
                root,
                slug="other-socket",
                dataset_key="T1",
            )
            payload = yaml.safe_load(duplicate_path.read_text(encoding="utf-8"))
            payload["interfaces"][0]["inputs"].append(
                {
                    "slug": "duplicate-key-socket",
                    "dataset_key": "T1",
                    "relative_path": "images/other",
                    "file_type": "nifti",
                    "cardinality": "one",
                }
            )
            duplicate_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(InterfaceManifestError, "dataset_key.*unique"):
                load_interface_manifest(duplicate_path)

    def test_manifest_rejects_unknown_model_fields_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_manifest(root)
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            payload["model"] = {"architecture": "dynunet"}
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(InterfaceManifestError, "unknown keys.*model"):
                load_interface_manifest(path)

            path = self._write_manifest(root, slug="unsafe-path")
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            payload["interfaces"][0]["inputs"][0]["relative_path"] = "../secret"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(InterfaceManifestError, "relative path"):
                load_interface_manifest(path)

    def test_required_technical_json_is_transport_metadata_not_a_modality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_manifest(root)
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            payload["interfaces"][0]["technical_inputs"] = [
                {
                    "slug": "opaque-technical-parameters",
                    "relative_path": "technical/parameters.json",
                    "file_type": "json",
                    "required": True,
                    "schema": {
                        "field_strength_t": {"type": "number", "nullable": False}
                    },
                }
            ]
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            manifest = load_interface_manifest(path)
            input_root = self._write_invocation(root, slug="opaque-fixture-image")
            technical_path = input_root / "technical" / "parameters.json"
            technical_path.parent.mkdir(parents=True)
            technical_path.write_text('{"field_strength_t": 3.0}\n', encoding="utf-8")
            inputs = json.loads((input_root / "inputs.json").read_text(encoding="utf-8"))
            inputs.append(
                {
                    "socket": {
                        "slug": "opaque-technical-parameters",
                        "relative_path": "technical/parameters.json",
                    }
                }
            )
            (input_root / "inputs.json").write_text(json.dumps(inputs), encoding="utf-8")

            invocation = resolve_invocation(manifest, input_root)

        self.assertEqual(tuple(invocation.raw_modalities), ("T1",))
        self.assertEqual(
            invocation.technical_inputs["opaque-technical-parameters"],
            {"field_strength_t": 3.0},
        )
        self.assertNotIn("opaque-technical-parameters", invocation.raw_modalities)

    def test_official_manifest_declares_exact_result_bindings_and_nullable_schema(self):
        manifest = load_interface_manifest(
            Path(
                "scripts/gc_submission_builder/configs/interfaces/isles26.yaml"
            )
        )

        self.assertEqual(len(manifest.interfaces), 1)
        interface = manifest.interfaces[0]
        self.assertEqual(interface.interface_key, ("stroke-metadata", "t1-brain-mri"))
        self.assertEqual(
            [(binding.slug, binding.dataset_key) for binding in interface.inputs],
            [("t1-brain-mri", "T1")],
        )
        self.assertEqual(
            [
                (binding.slug, binding.result_key, binding.file_type)
                for binding in interface.outputs
            ],
            [
                ("stroke-lesion-segmentation", "mask", "mha"),
                ("lesion-probability-map", "probability", "mha"),
            ],
        )
        schema = interface.technical_inputs[0].schema
        self.assertEqual(set(schema), {"CENTER", "CHRONICITY", "DAYS_POST_STROKE"})
        self.assertTrue(all(field.nullable for field in schema.values()))

    def test_output_bindings_are_explicit_unique_and_do_not_accept_legacy_singular_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write_manifest(root)
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            output = payload["interfaces"][0]["outputs"][0]

            for field, value, message in (
                ("slug", output["slug"], "output socket slugs"),
                ("result_key", output["result_key"], "result_key"),
                ("relative_path", output["relative_path"], "relative paths"),
            ):
                duplicate = dict(output)
                duplicate.update(
                    {
                        "slug": "second-output",
                        "result_key": "probability",
                        "relative_path": "images/second-output",
                        "file_type": "mha",
                    }
                )
                duplicate[field] = value
                payload["interfaces"][0]["outputs"].append(duplicate)
                path.write_text(yaml.safe_dump(payload), encoding="utf-8")
                with self.assertRaisesRegex(InterfaceManifestError, message):
                    load_interface_manifest(path)
                payload["interfaces"][0]["outputs"].pop()

            payload["interfaces"][0]["outputs"][0]["result_key"] = "logits"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(InterfaceManifestError, "result_key"):
                load_interface_manifest(path)

            legacy = self._write_manifest(root, slug="legacy-output")
            payload = yaml.safe_load(legacy.read_text(encoding="utf-8"))
            payload["interfaces"][0]["output"] = payload["interfaces"][0].pop(
                "outputs"
            )[0]
            legacy.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(InterfaceManifestError, "unknown keys.*output"):
                load_interface_manifest(legacy)

    def test_official_metadata_schema_accepts_nulls_and_ignores_unconsumed_extras(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = load_interface_manifest(
                Path(
                    "scripts/gc_submission_builder/configs/interfaces/isles26.yaml"
                )
            )
            input_root = root / "input"
            image_dir = input_root / "images" / "t1-brain-mri"
            image_dir.mkdir(parents=True)
            nib.save(
                nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.float32), np.eye(4)),
                image_dir / "case.nii.gz",
            )
            metadata_path = input_root / "stroke-metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "CENTER": None,
                        "CHRONICITY": None,
                        "DAYS_POST_STROKE": None,
                        "future_transport_field": "ignored",
                    }
                ),
                encoding="utf-8",
            )
            (input_root / "inputs.json").write_text(
                json.dumps(
                    [
                        {
                            "socket": {
                                "slug": "t1-brain-mri",
                                "relative_path": "images/t1-brain-mri",
                            }
                        },
                        {
                            "socket": {
                                "slug": "stroke-metadata",
                                "relative_path": "stroke-metadata.json",
                            }
                        },
                    ]
                ),
                encoding="utf-8",
            )

            invocation = resolve_invocation(manifest, input_root)

        self.assertEqual(tuple(invocation.raw_modalities), ("T1",))
        self.assertIsNone(invocation.technical_inputs["stroke-metadata"]["CENTER"])

    def test_official_metadata_schema_rejects_missing_wrong_and_nonfinite_values(self):
        manifest_path = Path(
            "scripts/gc_submission_builder/configs/interfaces/isles26.yaml"
        )
        invalid_values = (
            ({"CENTER": "R1", "CHRONICITY": 1}, "missing required fields"),
            (
                {
                    "CENTER": 7,
                    "CHRONICITY": 1,
                    "DAYS_POST_STROKE": 2.5,
                },
                "CENTER.*string",
            ),
            (
                {
                    "CENTER": "R1",
                    "CHRONICITY": True,
                    "DAYS_POST_STROKE": 2.5,
                },
                "CHRONICITY.*integer",
            ),
            (
                {
                    "CENTER": "R1",
                    "CHRONICITY": 1,
                    "DAYS_POST_STROKE": float("nan"),
                },
                "DAYS_POST_STROKE.*finite",
            ),
            (["not", "an", "object"], "must be a mapping"),
        )
        for metadata, message in invalid_values:
            with self.subTest(metadata=metadata):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    manifest = load_interface_manifest(manifest_path)
                    input_root = root / "input"
                    image_dir = input_root / "images" / "t1-brain-mri"
                    image_dir.mkdir(parents=True)
                    nib.save(
                        nib.Nifti1Image(np.zeros((3, 4, 5)), np.eye(4)),
                        image_dir / "case.nii.gz",
                    )
                    (input_root / "stroke-metadata.json").write_text(
                        json.dumps(metadata), encoding="utf-8"
                    )
                    (input_root / "inputs.json").write_text(
                        json.dumps(
                            [
                                {
                                    "socket": {
                                        "slug": "t1-brain-mri",
                                        "relative_path": "images/t1-brain-mri",
                                    }
                                },
                                {
                                    "socket": {
                                        "slug": "stroke-metadata",
                                        "relative_path": "stroke-metadata.json",
                                    }
                                },
                            ]
                        ),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(InterfaceManifestError, message):
                        resolve_invocation(manifest, input_root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
