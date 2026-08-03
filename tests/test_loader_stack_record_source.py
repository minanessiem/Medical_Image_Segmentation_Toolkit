import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from src.data.loader_stack.contracts import CaseRecordSourceError
from src.data.loader_stack.isles24_loader import datafold_read as read_isles24_records
from src.data.loader_stack.isles26_loader import datafold_read as read_isles26_records
from src.data.loader_stack.record_source import load_case_records


def _config(
    *,
    dataset_id: str,
    data_root: Path,
    split_file: Path,
    partitioning: str,
    subsets: dict,
    active_subsets: dict,
    fold: int | None = None,
):
    dataset = {
        "id": dataset_id,
        "name": dataset_id,
        "partitioning": partitioning,
        "subsets": subsets,
        "active_subsets": active_subsets,
    }
    if fold is not None:
        dataset["fold"] = fold
    return OmegaConf.create(
        {
            "dataset": dataset,
            "data_io": {
                "paths": {
                    "data_root": str(data_root),
                    "split_file": str(split_file),
                }
            },
        }
    )


def _write_datalist(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps({"training": records}), encoding="utf-8")


class TestLoaderStackRecordSource(unittest.TestCase):
    def test_isles26_validation_union_matches_registered_dataset_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            split_file = base / "isles26.json"
            _write_datalist(
                split_file,
                [
                    {
                        "caseID": "train-case",
                        "split": "train",
                        "T1": "train/t1.nii.gz",
                        "label": "train/label.nii.gz",
                    },
                    {
                        "caseID": "fast-case",
                        "split": "val_fast",
                        "T1": ["fast/t1.nii.gz"],
                        "label": "fast/label.nii.gz",
                    },
                    {
                        "caseID": "rest-case",
                        "split": "val_rest",
                        "T1": "rest/t1.nii.gz",
                        "label": "rest/label.nii.gz",
                    },
                ],
            )
            subsets = {
                "train": {"split_in": ["train"]},
                "val_full": {"split_in": ["val_fast", "val_rest"]},
            }
            cfg = _config(
                dataset_id="isles26",
                data_root=base,
                split_file=split_file,
                partitioning="split",
                subsets=subsets,
                active_subsets={"train": "train", "val": "val_full", "sample": "val_full"},
            )

            actual = load_case_records(cfg, subset_role="val", load_labels=True)
            expected = read_isles26_records(
                datalist=str(split_file),
                basedir=str(base),
                subset_name="val_full",
                partitioning="split",
                subset_definitions={
                    "train": {"split_in": ("train",)},
                    "val_full": {"split_in": ("val_fast", "val_rest")},
                },
                load_labels=True,
            )

            self.assertEqual(actual, expected)
            self.assertEqual([record["caseID"] for record in actual], ["fast-case", "rest-case"])
            self.assertEqual(actual[0]["T1"], [str(base / "fast/t1.nii.gz")])
            self.assertEqual(actual[0]["label"], str(base / "fast/label.nii.gz"))

    def test_isles24_fold_validation_matches_registered_dataset_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            split_file = base / "isles24.json"
            _write_datalist(
                split_file,
                [
                    {
                        "caseID": "train-case",
                        "fold": 1,
                        "CBF": "train/cbf.nii.gz",
                        "label": "train/label.nii.gz",
                    },
                    {
                        "caseID": "val-case",
                        "fold": 0,
                        "CBF": ["val/cbf.nii.gz"],
                        "label": "val/label.nii.gz",
                    },
                ],
            )
            subsets = {
                "train": {"fold_not_in": [0]},
                "val": {"fold_in": [0]},
            }
            cfg = _config(
                dataset_id="isles24",
                data_root=base,
                split_file=split_file,
                partitioning="fold",
                subsets=subsets,
                active_subsets={"train": "train", "val": "val", "sample": "val"},
                fold=0,
            )

            actual = load_case_records(cfg, subset_role="val", load_labels=True)
            train_records = load_case_records(cfg, subset_role="train", load_labels=True)
            sample_records = load_case_records(cfg, subset_role="sample", load_labels=False)
            expected = read_isles24_records(
                datalist=str(split_file),
                basedir=str(base),
                fold=0,
                subset_name="val",
                partitioning="fold",
                subset_definitions={
                    "train": {"fold_not_in": (0,)},
                    "val": {"fold_in": (0,)},
                },
                load_labels=True,
            )

            self.assertEqual(actual, expected)
            self.assertEqual([record["caseID"] for record in train_records], ["train-case"])
            self.assertEqual([record["caseID"] for record in sample_records], ["val-case"])
            self.assertNotIn("label", sample_records[0])
            self.assertEqual([record["caseID"] for record in actual], ["val-case"])
            self.assertEqual(actual[0]["CBF"], [str(base / "val/cbf.nii.gz")])

    def test_validation_only_config_does_not_require_train_or_sample_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            split_file = base / "isles26.json"
            _write_datalist(
                split_file,
                [{"caseID": "blind-case", "split": "val", "T1": "t1.nii.gz"}],
            )
            cfg = _config(
                dataset_id="isles26",
                data_root=base,
                split_file=split_file,
                partitioning="split",
                subsets={"only_validation": {"split_in": ["val"]}},
                active_subsets={"val": "only_validation"},
            )

            records = load_case_records(cfg, subset_role="val", load_labels=False)

            self.assertEqual([record["caseID"] for record in records], ["blind-case"])
            self.assertNotIn("label", records[0])
            with self.assertRaisesRegex(
                CaseRecordSourceError,
                "dataset='isles26'.*role='val'.*subset='only_validation'.*blind-case.*label",
            ):
                load_case_records(cfg, subset_role="val", load_labels=True)

    def test_requested_roles_route_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            split_file = base / "isles26.json"
            _write_datalist(
                split_file,
                [
                    {"caseID": "train-case", "split": "train", "T1": "a.nii.gz", "label": "a-label.nii.gz"},
                    {"caseID": "val-case", "split": "val", "T1": "b.nii.gz", "label": "b-label.nii.gz"},
                    {"caseID": "sample-case", "split": "sample", "T1": "c.nii.gz", "label": "c-label.nii.gz"},
                ],
            )
            cfg = _config(
                dataset_id="isles26",
                data_root=base,
                split_file=split_file,
                partitioning="split",
                subsets={
                    "training": {"split_in": ["train"]},
                    "validation": {"split_in": ["val"]},
                    "sampling": {"split_in": ["sample"]},
                },
                active_subsets={
                    "train": "training",
                    "val": "validation",
                    "sample": "sampling",
                },
            )

            self.assertEqual(load_case_records(cfg, "train")[0]["caseID"], "train-case")
            self.assertEqual(load_case_records(cfg, "val")[0]["caseID"], "val-case")
            self.assertEqual(load_case_records(cfg, "sample")[0]["caseID"], "sample-case")

    def test_record_loading_constructs_no_dataset_dataloader_or_transform_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            split_file = base / "isles26.json"
            _write_datalist(
                split_file,
                [{"caseID": "val-case", "split": "val", "T1": "t1.nii.gz", "label": "label.nii.gz"}],
            )
            cfg = _config(
                dataset_id="isles26",
                data_root=base,
                split_file=split_file,
                partitioning="split",
                subsets={"validation": {"split_in": ["val"]}},
                active_subsets={"val": "validation"},
            )

            with (
                patch("torch.utils.data.DataLoader") as dataloader,
                patch("src.data.loader_stack.isles26_loader.ISLES26Dataset3D") as dataset,
                patch("src.data.loader_stack.isles26_loader.build_full_volumes_3d_pipeline") as pipeline,
            ):
                records = load_case_records(cfg)

            self.assertEqual(len(records), 1)
            dataloader.assert_not_called()
            dataset.assert_not_called()
            pipeline.assert_not_called()

    def test_unknown_dataset_and_missing_requested_role_fail_with_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            split_file = base / "dataset.json"
            _write_datalist(split_file, [])
            unknown_cfg = _config(
                dataset_id="not-registered",
                data_root=base,
                split_file=split_file,
                partitioning="split",
                subsets={"validation": {"split_in": ["val"]}},
                active_subsets={"val": "validation"},
            )
            with self.assertRaisesRegex(CaseRecordSourceError, "not-registered"):
                load_case_records(unknown_cfg)

            missing_role_cfg = _config(
                dataset_id="isles26",
                data_root=base,
                split_file=split_file,
                partitioning="split",
                subsets={"training": {"split_in": ["train"]}},
                active_subsets={"train": "training"},
            )
            with self.assertRaisesRegex(CaseRecordSourceError, "active_subsets.*val"):
                load_case_records(missing_role_cfg)


if __name__ == "__main__":
    unittest.main()
