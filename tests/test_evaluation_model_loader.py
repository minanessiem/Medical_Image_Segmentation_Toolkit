"""
Tests for repository-model evaluation model loading utilities.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from scripts.evaluation.core.model_loader import (
    build_model_for_evaluation,
    find_checkpoint,
    is_discriminative_config,
    load_checkpoint_into_model,
    resolve_diffusion_type,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


class TestEvaluationModelLoader(unittest.TestCase):
    def test_find_checkpoint_in_best_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            checkpoint_dir = run_dir / "models" / "best"
            checkpoint_dir.mkdir(parents=True)
            expected = checkpoint_dir / "best_model.pth"
            expected.write_bytes(b"checkpoint")

            found = find_checkpoint(run_dir, "best_model")

        self.assertEqual(found, expected)

    def test_find_checkpoint_in_checkpoints_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            checkpoint_dir = run_dir / "models" / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            expected = checkpoint_dir / "step_100.pth"
            expected.write_bytes(b"checkpoint")

            found = find_checkpoint(run_dir, "step_100")

        self.assertEqual(found, expected)

    def test_find_checkpoint_in_singular_checkpoint_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            checkpoint_dir = run_dir / "models" / "checkpoint"
            checkpoint_dir.mkdir(parents=True)
            expected = checkpoint_dir / "step_100.pth"
            expected.write_bytes(b"checkpoint")

            found = find_checkpoint(run_dir, "step_100")

        self.assertEqual(found, expected)

    def test_find_checkpoint_in_models_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            checkpoint_dir = run_dir / "models"
            checkpoint_dir.mkdir(parents=True)
            expected = checkpoint_dir / "final.pth"
            expected.write_bytes(b"checkpoint")

            found = find_checkpoint(run_dir, "final")

        self.assertEqual(found, expected)

    def test_find_checkpoint_strips_pth_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            checkpoint_dir = run_dir / "models" / "best"
            checkpoint_dir.mkdir(parents=True)
            expected = checkpoint_dir / "best_model.pth"
            expected.write_bytes(b"checkpoint")

            found = find_checkpoint(run_dir, "best_model.pth")

        self.assertEqual(found, expected)

    def test_find_ema_checkpoint_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            checkpoint_dir = run_dir / "models" / "best"
            checkpoint_dir.mkdir(parents=True)
            expected = checkpoint_dir / "best_model_ema_0.9999.pth"
            expected.write_bytes(b"checkpoint")

            found = find_checkpoint(run_dir, "best_model", use_ema=True)

        self.assertEqual(found, expected)

    def test_missing_checkpoint_error_lists_searched_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with self.assertRaises(FileNotFoundError) as ctx:
                find_checkpoint(run_dir, "missing_model")

        message = str(ctx.exception)
        self.assertIn("missing_model", message)
        self.assertIn("models/best", message.replace("\\", "/"))
        self.assertIn("models/checkpoints", message.replace("\\", "/"))

    def test_load_checkpoint_into_model_strips_module_prefix(self):
        model = TinyModel()
        source = TinyModel()
        checkpoint_state = {
            f"module.{key}": value.clone()
            for key, value in source.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pth"
            torch.save({"model_state_dict": checkpoint_state}, checkpoint_path)
            missing, unexpected = load_checkpoint_into_model(
                model=model,
                checkpoint_path=checkpoint_path,
                device="cpu",
            )

        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(model.state_dict()[key], value))

    def test_resolve_diffusion_type_defaults_to_discriminative(self):
        cfg = OmegaConf.create({})

        self.assertEqual(resolve_diffusion_type(cfg), "Discriminative")
        self.assertTrue(is_discriminative_config(cfg))

    def test_is_discriminative_config_false_for_openai_diffusion(self):
        cfg = OmegaConf.create({"diffusion": {"type": "OpenAI_DDPM"}})

        self.assertEqual(resolve_diffusion_type(cfg), "OpenAI_DDPM")
        self.assertFalse(is_discriminative_config(cfg))

    def test_build_model_for_evaluation_delegates_to_shared_loader(self):
        cfg = OmegaConf.create({"diffusion": {"type": "Discriminative"}})
        adapter = TinyModel()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pth"
            torch.save(adapter.state_dict(), checkpoint_path)

            with patch(
                "scripts.evaluation.core.model_loader.load_model",
                return_value=(adapter, [], []),
            ) as load_model_mock:
                loaded = build_model_for_evaluation(
                    cfg=cfg,
                    checkpoint_path=checkpoint_path,
                    device="cpu",
                )

        self.assertIs(loaded, adapter)
        load_model_mock.assert_called_once_with(
            cfg,
            checkpoint_path=checkpoint_path,
            device="cpu",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
