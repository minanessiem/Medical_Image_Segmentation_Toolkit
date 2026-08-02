import unittest

from omegaconf import OmegaConf

from scripts.nnunet.core.conversion_core import validate_converter_contract
from scripts.nnunet.core.evaluation_pipeline import build_evaluation_request
from scripts.slurm.single_job_runner import load_config


class TestNnunetThreeDimensionalConversionPresets(unittest.TestCase):
    def test_current_presets_declare_expected_export_space(self):
        expected = {
            "nnunet/convert/isles24_cluster_3d_baseline": "native_input",
            "nnunet/convert/isles24_local_3d_baseline": "native_input",
            "nnunet/convert/isles26_cluster_3d_t1raw": "model_preprocessed",
            "nnunet/convert/isles26_local_3d_t1raw": "model_preprocessed",
            "nnunet/convert/isles26_local_3d_t1raw_native": "native_input",
            "nnunet/convert/isles26_atlas30_cluster_3d_t1raw": "model_preprocessed",
        }
        for config_name, export_space in expected.items():
            with self.subTest(config_name=config_name):
                cfg = load_config(config_name, [], resolve_final=True)
                self.assertEqual(cfg["nnunet"]["export_space"], export_space)

    def test_isles26_native_preset_disables_only_spatial_preprocessing(self):
        cfg = load_config(
            "nnunet/convert/isles26_local_3d_t1raw_native",
            [],
            resolve_final=True,
        )

        self.assertEqual(cfg["dataset"]["modalities"], ["T1_RAW"])
        self.assertEqual(cfg["dataset"]["nnunet"]["dataset_id"], "264")
        self.assertEqual(
            cfg["dataset"]["nnunet"]["dataset_name"],
            "isles26_t1_raw_native",
        )
        common = cfg["dataset"]["preprocessing_configs"]["common"]
        self.assertFalse(common["orientation"]["enabled"])
        self.assertFalse(common["spacing"]["enabled"])
        self.assertTrue(common["spacing"]["allow_native_spacing"])
        self.assertFalse(
            cfg["dataset"]["preprocessing_configs"]["full_volumes_3d"][
                "pad_to_divisible"
            ]
        )
        validate_converter_contract(OmegaConf.create(cfg))


def _evaluation_cfg(input_format="volumes_3d", export_space="model_preprocessed"):
    return OmegaConf.create(
        {
            "dataset": {
                "nnunet": {"dataset_id": "501", "dataset_name": "synthetic"}
            },
            "nnunet": {
                "dataset_id": "501",
                "dataset_name": "synthetic",
                "export_space": export_space,
            },
            "nnunet_eval": {
                "pred_dir": "/tmp/pred",
                "gt_dir": "/tmp/gt",
                "output_dir": "/tmp/output",
                "input_format": input_format,
                "levels": ["volume"],
                "threshold": 0.5,
                "allow_shape_mismatch": False,
                "foreground_only_all_metrics": False,
            },
        }
    )


class TestNnunetEvaluationSpaceConfig(unittest.TestCase):
    def test_volume_request_consumes_conversion_owned_space(self):
        request = build_evaluation_request(_evaluation_cfg())
        self.assertEqual(request.volume_space, "model_preprocessed")

    def test_volume_request_rejects_missing_conversion_space(self):
        cfg = _evaluation_cfg()
        del cfg.nnunet["export_space"]
        with self.assertRaisesRegex(ValueError, "from the composed conversion preset"):
            build_evaluation_request(cfg)

    def test_slice_request_fails_before_geometry_aware_evaluation(self):
        with self.assertRaisesRegex(ValueError, "parent-volume geometry"):
            build_evaluation_request(_evaluation_cfg(input_format="slices_2d"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
