"""Characterization tests for shared single-model loading."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.models.model_loader import (
    StrictModelLoadError,
    load_checkpoint_into_model,
    load_model,
    load_model_strict,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, value):
        return self.layer(value)


class TestSharedModelLoader(unittest.TestCase):
    def test_checkpoint_loader_preserves_existing_module_prefix_handling(self):
        source = TinyModel()
        target = TinyModel()
        checkpoint_state = {
            f"module.{key}": value.clone()
            for key, value in source.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pth"
            torch.save({"model_state_dict": checkpoint_state}, checkpoint_path)
            missing, unexpected = load_checkpoint_into_model(
                target,
                checkpoint_path,
                device="cpu",
            )

        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[key], value))

    def test_checkpoint_loader_preserves_partial_load_reporting(self):
        source = TinyModel()
        partial_state = dict(source.state_dict())
        partial_state.pop("layer.bias")

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pth"
            torch.save(partial_state, checkpoint_path)
            missing, unexpected = load_checkpoint_into_model(
                TinyModel(),
                checkpoint_path,
                device="cpu",
            )

        self.assertEqual(missing, ["layer.bias"])
        self.assertEqual(unexpected, [])

    def test_load_model_preserves_evaluation_construction_sequence(self):
        cfg = OmegaConf.create({"diffusion": {"type": "Discriminative"}})
        base_model = TinyModel()
        adapter = TinyModel()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pth"
            torch.save(adapter.state_dict(), checkpoint_path)

            with patch(
                "src.models.model_loader.build_model",
                return_value=base_model,
            ) as build_model_mock, patch(
                "src.models.model_loader.Diffusion.build_diffusion",
                return_value=adapter,
            ) as build_diffusion_mock:
                loaded, missing, unexpected = load_model(
                    cfg,
                    checkpoint_path,
                    device="cpu",
                )

        self.assertIs(loaded, adapter)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        self.assertFalse(loaded.training)
        self.assertTrue(all(parameter.requires_grad for parameter in loaded.parameters()))
        build_model_mock.assert_called_once_with(cfg)
        build_diffusion_mock.assert_called_once()
        self.assertIs(build_diffusion_mock.call_args.args[0], base_model)
        self.assertIs(build_diffusion_mock.call_args.args[1], cfg)
        self.assertEqual(str(build_diffusion_mock.call_args.args[2]), "cpu")

    def test_strict_release_loader_rejects_partial_compatibility_load(self):
        adapter = TinyModel()
        partial_state = dict(adapter.state_dict())
        partial_state.pop("layer.bias")
        cfg = OmegaConf.create({"diffusion": {"type": "Discriminative"}})

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pth"
            torch.save(partial_state, checkpoint_path)
            with patch(
                "src.models.model_loader.build_model",
                return_value=TinyModel(),
            ), patch(
                "src.models.model_loader.Diffusion.build_diffusion",
                return_value=adapter,
            ):
                with self.assertRaisesRegex(
                    StrictModelLoadError,
                    "missing keys.*layer.bias",
                ):
                    load_model_strict(cfg, checkpoint_path, device="cpu")


if __name__ == "__main__":
    unittest.main()
