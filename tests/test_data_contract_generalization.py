import unittest
from typing import List, Optional

from omegaconf import OmegaConf

from scripts.nnunet.convert_to_nnunet import (
    validate_converter_contract,
)
from src.data.loaders import validate_dataset_contract


def _base_cfg(
    *,
    dataset_id: str,
    dataset_name: Optional[str] = None,
    modalities: Optional[List[str]] = None,
    loader_mode: str = "online_slices_3d_to_2d",
    dim: str = "2d",
):
    modalities = modalities or ["T1_RAW"]
    dataset_name = dataset_name or dataset_id
    dataset_key = str(dataset_id).strip().lower()
    if dataset_key == "isles24":
        partitioning = "fold"
        subsets = {
            "train": {"fold_not_in": [0]},
            "val": {"fold_in": [0]},
        }
        active_subsets = {"train": "train", "val": "val", "sample": "val"}
    else:
        partitioning = "split"
        subsets = {
            "train": {"split_in": ["train"]},
            "val_fast": {"split_in": ["val_fast"]},
            "val_rest": {"split_in": ["val_rest"]},
            "val_full": {"split_in": ["val_fast", "val_rest"]},
        }
        active_subsets = {
            "train": "train",
            "val": "val_fast",
            "sample": "val_fast",
        }
    return OmegaConf.create(
        {
            "dataset": {
                "id": dataset_id,
                "name": dataset_name,
                "fold": 0,
                "partitioning": partitioning,
                "subsets": subsets,
                "active_subsets": active_subsets,
                "modalities": modalities,
                "num_modalities": len(modalities),
                "preprocessing_configs": {},
                "nnunet": {"dataset_id": "501", "dataset_name": "testset"},
            },
            "data_mode": {
                "loader_mode": loader_mode,
                "dim": dim,
                "per_side_context_slices": 0,
                "channel_layout": None,
            },
            "data_io": {
                "paths": {
                    "data_root": "/tmp/data",
                    "split_file": "/tmp/split.json",
                    "nnunet_root": "/tmp/nnunet",
                }
            },
            "data_runtime": {
                "train_batch_size": 1,
                "test_batch_size": 1,
                "num_train_workers": 0,
                "num_valid_workers": 0,
                "num_test_workers": 0,
                "use_caching": False,
                "use_shared_cache": False,
                # Contract requires explicit (non-None) values for these keys.
                "train_prefetch_factor": 2,
                "test_prefetch_factor": 2,
                "loader_smoke_testing": False,
            },
            "nnunet": {
                "dataset_id": "501",
                "dataset_name": "testset",
                "output_dir": "/tmp/output",
                "test": True,
                "test_max_slices": 10,
            },
        }
    )


class TestDataContractGeneralization(unittest.TestCase):
    def test_validate_dataset_contract_accepts_isles26_online(self):
        cfg = _base_cfg(dataset_id="isles26", modalities=["T1_RAW"])
        validate_dataset_contract(cfg)

    def test_validate_dataset_contract_accepts_isles24_random_patches_3d(self):
        cfg = _base_cfg(
            dataset_id="isles24",
            modalities=["CBF_min_0_max_70", "TMAX_min_4_max_30"],
            loader_mode="random_patches_3d",
            dim="3d",
        )
        validate_dataset_contract(cfg)

    def test_validate_dataset_contract_accepts_isles26_random_patches_3d(self):
        cfg = _base_cfg(
            dataset_id="isles26",
            modalities=["T1_RAW"],
            loader_mode="random_patches_3d",
            dim="3d",
        )
        validate_dataset_contract(cfg)

    def test_validate_dataset_contract_rejects_isles26_random_patches_wrong_dim(self):
        cfg = _base_cfg(
            dataset_id="isles26",
            modalities=["T1_RAW"],
            loader_mode="random_patches_3d",
            dim="2d",
        )
        with self.assertRaisesRegex(
            ValueError,
            "random_patches_3d requires data_mode.dim='3d'",
        ):
            validate_dataset_contract(cfg)

    def test_validate_dataset_contract_accepts_isles26_nnunet_mode(self):
        cfg = _base_cfg(
            dataset_id="isles26",
            modalities=["T1_RAW"],
            loader_mode="nnunet_slices_2d",
            dim="2d",
        )
        cfg.data_mode.per_side_context_slices = 1
        cfg.data_mode.channel_layout = "slice_major"
        validate_dataset_contract(cfg)

    def test_validate_converter_contract_accepts_isles24_route(self):
        cfg = _base_cfg(
            dataset_id="isles24",
            modalities=["NCCT"],
            loader_mode="online_slices_3d_to_2d",
            dim="2d",
        )
        validate_converter_contract(cfg)

    def test_validate_converter_contract_accepts_isles26_loader_module(self):
        cfg = _base_cfg(
            dataset_id="isles26",
            modalities=["T1_RAW"],
            loader_mode="online_slices_3d_to_2d",
            dim="2d",
        )
        validate_converter_contract(cfg)

    def test_validate_converter_contract_accepts_isles26_without_fold_key(self):
        cfg = _base_cfg(
            dataset_id="isles26",
            modalities=["T1_RAW"],
            loader_mode="online_slices_3d_to_2d",
            dim="2d",
        )
        del cfg.dataset["fold"]
        validate_converter_contract(cfg)

    def test_validate_converter_contract_requires_explicit_3d_export_space(self):
        cfg = _base_cfg(
            dataset_id="isles26",
            modalities=["T1_RAW"],
            loader_mode="full_volumes_3d",
            dim="3d",
        )
        with self.assertRaisesRegex(ValueError, "nnunet.export_space"):
            validate_converter_contract(cfg)

    def test_validate_converter_contract_accepts_known_3d_export_space(self):
        cfg = _base_cfg(
            dataset_id="isles26",
            modalities=["T1_RAW"],
            loader_mode="full_volumes_3d",
            dim="3d",
        )
        cfg.nnunet.export_space = "model_preprocessed"
        validate_converter_contract(cfg)

    def test_validate_converter_contract_rejects_unknown_3d_export_space(self):
        cfg = _base_cfg(
            dataset_id="isles26",
            modalities=["T1_RAW"],
            loader_mode="full_volumes_3d",
            dim="3d",
        )
        cfg.nnunet.export_space = "probably_native"
        with self.assertRaisesRegex(ValueError, "must be one of"):
            validate_converter_contract(cfg)


if __name__ == "__main__":
    unittest.main()
