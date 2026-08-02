import unittest

from omegaconf import OmegaConf

from src.utils.train_utils import validate_training_runtime_contract


def _base_cfg():
    return OmegaConf.create(
        {
            "data_mode": {"dim": "3d"},
            "diffusion": {"type": "OpenAI_DDPM"},
            "logging": {"enable_sampling_snapshots": False},
            "validation": {
                "ensemble": {"enabled": False, "method": "soft_staple"},
                "ensembled_image": {"enabled": False},
            },
        }
    )


class TestTrainingRuntimeContracts(unittest.TestCase):
    def test_rejects_3d_diffusion_sampling_snapshots(self):
        cfg = _base_cfg()
        cfg.logging.enable_sampling_snapshots = True
        with self.assertRaisesRegex(ValueError, "enable_sampling_snapshots=true"):
            validate_training_runtime_contract(cfg)

    def test_rejects_3d_diffusion_validation_ensemble(self):
        cfg = _base_cfg()
        cfg.validation.ensemble.enabled = True
        cfg.validation.ensemble.method = "mean"
        with self.assertRaisesRegex(ValueError, "validation.ensemble.enabled=true"):
            validate_training_runtime_contract(cfg)

    def test_rejects_3d_diffusion_ensembled_image_logging(self):
        cfg = _base_cfg()
        cfg.validation.ensembled_image.enabled = True
        with self.assertRaisesRegex(
            ValueError, "validation.ensembled_image.enabled=true"
        ):
            validate_training_runtime_contract(cfg)

    def test_allows_3d_discriminative(self):
        cfg = _base_cfg()
        cfg.diffusion.type = "Discriminative"
        cfg.validation.ensemble.method = "mean"
        cfg.logging.enable_sampling_snapshots = True
        cfg.validation.ensemble.enabled = True
        cfg.validation.ensembled_image.enabled = True
        # Discriminative is explicitly allowed by this guard.
        validate_training_runtime_contract(cfg)

    def test_rejects_3d_soft_staple_for_discriminative_models(self):
        cfg = _base_cfg()
        cfg.diffusion.type = "Discriminative"
        cfg.validation.ensemble.enabled = True

        with self.assertRaisesRegex(ValueError, "soft_staple.*2D"):
            validate_training_runtime_contract(cfg)

    def test_allows_2d_diffusion(self):
        cfg = _base_cfg()
        cfg.data_mode.dim = "2d"
        cfg.logging.enable_sampling_snapshots = True
        cfg.validation.ensemble.enabled = True
        cfg.validation.ensembled_image.enabled = True
        validate_training_runtime_contract(cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
