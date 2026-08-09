import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd

try:
    import nibabel  # noqa: F401
except ImportError:
    sys.modules["nibabel"] = types.ModuleType("nibabel")

try:
    import tqdm  # noqa: F401
except ImportError:
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, **_kwargs: iterable
    sys.modules["tqdm"] = tqdm_module

from scripts.dataset_setup.ISLES26_json_creator import (
    assign_site_balanced_validation_pool,
    build_case_entry,
    build_metadata_dataframe,
    discover_cases,
    resolve_training_root,
)


def _metadata_with_singleton_site() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "caseID": ["a1", "a2", "a3", "b1", "c1", "c2"],
            "SITE": ["A", "A", "A", "B", "C", "C"],
        }
    )


def _make_subject(training_root: Path, center: str, case_id: str) -> Path:
    anat_dir = training_root / center / case_id / "ses-1" / "anat"
    anat_dir.mkdir(parents=True)
    return anat_dir


class TestCaseDiscovery(unittest.TestCase):
    def test_discovers_structurally_valid_centers_regardless_of_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp)
            training_root = dataset_root / "Training_Raw"
            _make_subject(training_root, "R001", "sub-r001s001")
            _make_subject(training_root, "SOOP", "sub-soop0003")
            _make_subject(training_root, "Clinic-West", "sub-clinic001")
            (training_root / "not-a-center").mkdir()

            self.assertEqual(resolve_training_root(dataset_root), training_root)
            cases = discover_cases(training_root)

            self.assertEqual(
                {(case.site_id, case.case_id) for case in cases},
                {
                    ("R001", "sub-r001s001"),
                    ("SOOP", "sub-soop0003"),
                    ("Clinic-West", "sub-clinic001"),
                },
            )

    def test_resolves_training_root_with_only_non_r_center(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp)
            training_root = dataset_root / "Training_Raw"
            _make_subject(training_root, "SOOP", "sub-soop0003")

            self.assertEqual(resolve_training_root(dataset_root), training_root)

    def test_blank_metadata_site_falls_back_to_directory_center(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            training_root = Path(tmp) / "Training_Raw"
            case_id = "sub-soop0910"
            anat_dir = _make_subject(training_root, "SOOP", case_id)
            metadata_path = anat_dir / f"{case_id}_ses-1_metadata.csv"
            metadata_path.write_text(
                "ATLAS2_DATASET,SESSION_ID,DAYS_POST_STROKE,CHRONICITY,SITE\n",
                encoding="utf-8",
            )
            case = discover_cases(training_root)[0]

            metadata_df = build_metadata_dataframe([case], training_root)
            entry = build_case_entry(case, training_root, fold=0)

            self.assertEqual(metadata_df.loc[0, "SITE"], "SOOP")
            self.assertEqual(entry["metadata"]["SITE"], "SOOP")
            self.assertIsNone(entry["metadata"]["ATLAS2_DATASET"])
            self.assertIsNone(entry["metadata"]["SESSION_ID"])
            json.dumps(entry, allow_nan=False)


class TestSiteBalancedValidationPool(unittest.TestCase):
    def test_singleton_site_stays_in_training(self) -> None:
        result = assign_site_balanced_validation_pool(
            _metadata_with_singleton_site(),
            target_val_full_count=2,
            seed=42,
        )

        self.assertEqual(result.loc[result["SITE"] == "B", "split"].tolist(), ["train"])
        self.assertEqual(result.attrs["singleton_training_sites"], ["B"])

        for site in ("A", "C"):
            site_splits = set(result.loc[result["SITE"] == site, "split"])
            self.assertEqual(site_splits, {"train", "validation_pool"})

    def test_minimum_validation_size_counts_only_multi_case_sites(self) -> None:
        with self.assertRaisesRegex(ValueError, "every multi-case site in validation"):
            assign_site_balanced_validation_pool(
                _metadata_with_singleton_site(),
                target_val_full_count=1,
            )

    def test_maximum_validation_size_reserves_training_cases_and_singletons(self) -> None:
        with self.assertRaisesRegex(ValueError, "singleton sites assigned to training"):
            assign_site_balanced_validation_pool(
                _metadata_with_singleton_site(),
                target_val_full_count=4,
            )


if __name__ == "__main__":
    unittest.main()
