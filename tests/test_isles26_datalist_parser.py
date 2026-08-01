import json
import tempfile
import unittest
from pathlib import Path

from src.data.loader_stack.isles26_loader import datafold_read


class TestIsles26DatalistParser(unittest.TestCase):
    def test_datafold_read_split_subset_and_union(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            datalist_path = base / "dataset.json"
            payload = {
                "training": [
                    {
                        "caseID": "sub-r001s001",
                        "split": "val_fast",
                        "siteID": "R001",
                        "T1": ["R001/sub-r001s001/ses-1/anat/t1.nii.gz"],
                        "label": "R001/sub-r001s001/ses-1/anat/label.nii.gz",
                        "metadata_csv": "R001/sub-r001s001/ses-1/anat/meta.csv",
                        "metadata": {"SITE": "R001", "SESSION_ID": "sub-r001s001_ses-1"},
                    },
                    {
                        "split": "train",
                        "caseID": ["sub-r002s001"],
                        "siteID": "R002",
                        "T1": "R002/sub-r002s001/ses-1/anat/t1.nii.gz",
                        "label": "R002/sub-r002s001/ses-1/anat/label.nii.gz",
                    },
                    {
                        "split": "val_rest",
                        "caseID": "sub-r003s001",
                        "siteID": "R003",
                        "T1": "R003/sub-r003s001/ses-1/anat/t1.nii.gz",
                        "label": "R003/sub-r003s001/ses-1/anat/label.nii.gz",
                    },
                ]
            }
            datalist_path.write_text(json.dumps(payload), encoding="utf-8")

            train_records = datafold_read(
                datalist=str(datalist_path),
                basedir=str(base),
                subset_name="train",
            )
            val_fast_records = datafold_read(
                datalist=str(datalist_path),
                basedir=str(base),
                subset_name="val_fast",
            )
            subset_definitions = {
                "val_full": {"split_in": ("val_fast", "val_rest")},
            }
            val_full_records = datafold_read(
                datalist=str(datalist_path),
                basedir=str(base),
                subset_name="val_full",
                subset_definitions=subset_definitions,
            )

            self.assertEqual(len(train_records), 1)
            self.assertEqual(len(val_fast_records), 1)
            self.assertEqual(len(val_full_records), 2)

            val_case = val_fast_records[0]
            self.assertEqual(val_case["caseID"], "sub-r001s001")
            self.assertEqual(
                val_case["T1"][0],
                str(base / "R001/sub-r001s001/ses-1/anat/t1.nii.gz"),
            )
            self.assertEqual(
                val_case["label"],
                str(base / "R001/sub-r001s001/ses-1/anat/label.nii.gz"),
            )
            self.assertEqual(
                val_case["metadata_csv"],
                str(base / "R001/sub-r001s001/ses-1/anat/meta.csv"),
            )
            self.assertEqual(val_case["metadata"]["SITE"], "R001")

            train_case = train_records[0]
            self.assertEqual(train_case["caseID"], "sub-r002s001")
            self.assertEqual(len(train_case["T1"]), 1)
            self.assertEqual(
                train_case["T1"][0],
                str(base / "R002/sub-r002s001/ses-1/anat/t1.nii.gz"),
            )

    def test_datafold_read_missing_required_key_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            datalist_path = base / "dataset.json"
            payload = {
                "training": [
                    {
                        "caseID": "sub-r001s001",
                        "split": "train",
                        # Missing T1 and label.
                    }
                ]
            }
            datalist_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required keys"):
                datafold_read(datalist=str(datalist_path), basedir=str(base), subset_name="train")

    def test_datafold_read_requires_training_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            datalist_path = base / "dataset.json"
            payload = {"training": {"split": "train"}}
            datalist_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must contain a list"):
                datafold_read(datalist=str(datalist_path), basedir=str(base), subset_name="train")

    def test_label_requirement_is_explicit_and_defaults_to_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            datalist_path = base / "dataset.json"
            datalist_path.write_text(
                json.dumps(
                    {
                        "training": [
                            {
                                "caseID": "case-label-free",
                                "split": "val",
                                "T1": "t1.nii.gz",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "label"):
                datafold_read(
                    datalist=str(datalist_path),
                    basedir=str(base),
                    subset_name="val",
                )

            records = datafold_read(
                datalist=str(datalist_path),
                basedir=str(base),
                subset_name="val",
                load_labels=False,
            )
            self.assertEqual(len(records), 1)
            self.assertNotIn("label", records[0])


if __name__ == "__main__":
    unittest.main()
