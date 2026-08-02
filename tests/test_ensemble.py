"""
Unit tests for the ensemble validation module.

Tests cover:
- mean_ensemble function
- soft_staple function
- ensemble_predictions dispatch
- should_ensemble config detection
- should_log_ensembled_image config detection

Run with: python -m unittest tests.test_ensemble -v
"""

import unittest
import torch
from omegaconf import OmegaConf

from src.utils.ensemble import (
    mean_ensemble,
    soft_staple,
    ensemble_predictions,
    should_ensemble,
    should_log_ensembled_image,
)


class TestMeanEnsemble(unittest.TestCase):
    """Tests for mean_ensemble function."""
    
    def test_basic_shape(self):
        """Output shape should be [B, C, H, W] from [N, B, C, H, W] input."""
        samples = torch.rand(5, 2, 1, 64, 64)
        result = mean_ensemble(samples)
        self.assertEqual(result.shape, (2, 1, 64, 64))
    
    def test_single_sample(self):
        """Single sample should return the sample unchanged."""
        samples = torch.rand(1, 2, 1, 64, 64)
        result = mean_ensemble(samples)
        self.assertTrue(torch.allclose(result, samples[0]))
    
    def test_averaging_correct(self):
        """Mean should correctly average across samples."""
        samples = torch.zeros(3, 1, 1, 2, 2)
        samples[0] = 0.0
        samples[1] = 0.5
        samples[2] = 1.0
        result = mean_ensemble(samples)
        expected = torch.full((1, 1, 2, 2), 0.5)
        self.assertTrue(torch.allclose(result, expected))

    def test_3d_probabilities_average_across_only_the_member_axis(self):
        samples = torch.stack(
            (
                torch.zeros((1, 1, 2, 3, 4)),
                torch.ones((1, 1, 2, 3, 4)),
            ),
            dim=0,
        )

        result = mean_ensemble(samples)

        self.assertEqual(result.shape, (1, 1, 2, 3, 4))
        torch.testing.assert_close(result, torch.full_like(result, 0.5))


class TestSoftStaple(unittest.TestCase):
    """Tests for soft_staple function."""
    
    def test_basic_shape(self):
        """Output shape should be [B, C, H, W] from [N, B, C, H, W] input."""
        samples = torch.rand(5, 2, 1, 64, 64)
        result = soft_staple(samples, max_iters=3, tolerance=0.02)
        self.assertEqual(result.shape, (2, 1, 64, 64))
    
    def test_output_in_valid_range(self):
        """Output should be in [0, 1] range (probabilities)."""
        samples = torch.rand(5, 2, 1, 64, 64)
        result = soft_staple(samples, max_iters=5, tolerance=0.02)
        self.assertGreaterEqual(result.min().item(), 0.0)
        self.assertLessEqual(result.max().item(), 1.0)
    
    def test_unanimous_consensus_ones(self):
        """When all samples are 1.0, output should be close to 1.0."""
        samples = torch.ones(5, 1, 1, 4, 4)
        result = soft_staple(samples, max_iters=5, tolerance=0.02)
        self.assertGreater(result.mean().item(), 0.9)
    
    def test_unanimous_consensus_zeros(self):
        """When all samples are 0.0, output should be close to 0.0."""
        samples = torch.zeros(5, 1, 1, 4, 4)
        result = soft_staple(samples, max_iters=5, tolerance=0.02)
        self.assertLess(result.mean().item(), 0.1)
    
    def test_convergence(self):
        """Should converge within max_iters for consistent samples."""
        samples = torch.rand(5, 1, 1, 16, 16)
        result = soft_staple(samples, max_iters=10, tolerance=0.01)
        self.assertIsNotNone(result)


class TestEnsemblePredictions(unittest.TestCase):
    """Tests for ensemble_predictions dispatch function."""
    
    def test_mean_dispatch(self):
        """Should dispatch to mean_ensemble when method='mean'."""
        samples = torch.rand(5, 2, 1, 64, 64)
        cfg = OmegaConf.create({'method': 'mean'})
        result = ensemble_predictions(samples, cfg)
        expected = mean_ensemble(samples)
        self.assertTrue(torch.allclose(result, expected))
    
    def test_soft_staple_dispatch(self):
        """Should dispatch to soft_staple when method='soft_staple'."""
        samples = torch.rand(5, 2, 1, 64, 64)
        cfg = OmegaConf.create({
            'method': 'soft_staple',
            'soft_staple': {'max_iters': 3, 'tolerance': 0.02}
        })
        result = ensemble_predictions(samples, cfg)
        self.assertEqual(result.shape, (2, 1, 64, 64))
    
    def test_unknown_method_raises(self):
        """Should raise ValueError for unknown method."""
        samples = torch.rand(5, 2, 1, 64, 64)
        cfg = OmegaConf.create({'method': 'unknown_method'})
        with self.assertRaises(ValueError) as context:
            ensemble_predictions(samples, cfg)
        self.assertIn("Unknown ensemble method", str(context.exception))

    def test_soft_staple_rejects_3d_probabilities(self):
        samples = torch.rand(3, 1, 1, 2, 3, 4)
        cfg = OmegaConf.create({
            'method': 'soft_staple',
            'soft_staple': {'max_iters': 3, 'tolerance': 0.02}
        })

        with self.assertRaisesRegex(ValueError, "soft_staple.*2D"):
            ensemble_predictions(samples, cfg)


class TestShouldEnsemble(unittest.TestCase):
    """Tests for should_ensemble config detection function."""
    
    def test_empty_config(self):
        """Should return False for empty config."""
        cfg = OmegaConf.create({})
        self.assertFalse(should_ensemble(cfg))
    
    def test_missing_validation_section(self):
        """Should return False when validation section is missing."""
        cfg = OmegaConf.create({'training': {}})
        self.assertFalse(should_ensemble(cfg))
    
    def test_missing_ensemble_section(self):
        """Should return False when ensemble section is missing."""
        cfg = OmegaConf.create({'validation': {'validation_interval': 5000}})
        self.assertFalse(should_ensemble(cfg))
    
    def test_missing_enabled_flag(self):
        """Should return False when enabled flag is missing."""
        cfg = OmegaConf.create({'validation': {'ensemble': {'num_samples': 5}}})
        self.assertFalse(should_ensemble(cfg))
    
    def test_enabled_true(self):
        """Should return True when enabled is True."""
        cfg = OmegaConf.create({'validation': {'ensemble': {'enabled': True}}})
        self.assertTrue(should_ensemble(cfg))
    
    def test_enabled_false(self):
        """Should return False when enabled is False."""
        cfg = OmegaConf.create({'validation': {'ensemble': {'enabled': False}}})
        self.assertFalse(should_ensemble(cfg))


class TestShouldLogEnsembledImage(unittest.TestCase):
    """Tests for should_log_ensembled_image function."""
    
    def test_step_zero(self):
        """Should return False at step 0."""
        cfg = OmegaConf.create({
            'validation': {'ensembled_image': {'enabled': True, 'interval': 5000}}
        })
        self.assertFalse(should_log_ensembled_image(cfg, 0))
    
    def test_empty_config(self):
        """Should return False for empty config."""
        cfg = OmegaConf.create({})
        self.assertFalse(should_log_ensembled_image(cfg, 5000))
    
    def test_missing_ensembled_image_section(self):
        """Should return False when ensembled_image section is missing."""
        cfg = OmegaConf.create({'validation': {'validation_interval': 5000}})
        self.assertFalse(should_log_ensembled_image(cfg, 5000))
    
    def test_disabled(self):
        """Should return False when enabled is False."""
        cfg = OmegaConf.create({
            'validation': {'ensembled_image': {'enabled': False, 'interval': 5000}}
        })
        self.assertFalse(should_log_ensembled_image(cfg, 5000))
    
    def test_interval_match(self):
        """Should return True when step matches interval."""
        cfg = OmegaConf.create({
            'validation': {'ensembled_image': {'enabled': True, 'interval': 5000}}
        })
        self.assertTrue(should_log_ensembled_image(cfg, 5000))
        self.assertTrue(should_log_ensembled_image(cfg, 10000))
        self.assertTrue(should_log_ensembled_image(cfg, 15000))
    
    def test_interval_no_match(self):
        """Should return False when step doesn't match interval."""
        cfg = OmegaConf.create({
            'validation': {'ensembled_image': {'enabled': True, 'interval': 5000}}
        })
        self.assertFalse(should_log_ensembled_image(cfg, 3000))
        self.assertFalse(should_log_ensembled_image(cfg, 7500))
        self.assertFalse(should_log_ensembled_image(cfg, 12345))


class TestConfigLoading(unittest.TestCase):
    """Tests for ensemble.yaml config file loading."""
    
    def test_config_loads(self):
        """Config file should load without errors."""
        cfg = OmegaConf.load('configs/validation/ensemble.yaml')
        self.assertIsNotNone(cfg)
    
    def test_config_has_ensemble_section(self):
        """Config should have ensemble section with expected keys."""
        cfg = OmegaConf.load('configs/validation/ensemble.yaml')
        self.assertTrue(hasattr(cfg, 'ensemble'))
        self.assertTrue(cfg.ensemble.enabled)
        self.assertEqual(cfg.ensemble.num_samples, 5)
        self.assertEqual(cfg.ensemble.method, 'soft_staple')
    
    def test_config_has_ensembled_image_section(self):
        """Config should have ensembled_image section with expected keys."""
        cfg = OmegaConf.load('configs/validation/ensemble.yaml')
        self.assertTrue(hasattr(cfg, 'ensembled_image'))
        self.assertTrue(cfg.ensembled_image.enabled)
        self.assertEqual(cfg.ensembled_image.num_samples, 4)
        self.assertEqual(cfg.ensembled_image.interval, 5000)
    
    def test_config_has_soft_staple_params(self):
        """Config should have soft_staple parameters."""
        cfg = OmegaConf.load('configs/validation/ensemble.yaml')
        self.assertTrue(hasattr(cfg.ensemble, 'soft_staple'))
        self.assertEqual(cfg.ensemble.soft_staple.max_iters, 5)
        self.assertEqual(cfg.ensemble.soft_staple.tolerance, 0.02)


if __name__ == '__main__':
    unittest.main(verbosity=2)
