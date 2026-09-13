import os
import tempfile
import unittest

from run_internal_cv import (
    _export_internal_split_artifacts,
    _resolve_split_targets,
    _split_rows_patient_grouped,
)


class TestInternalSplit(unittest.TestCase):
    def test_resolve_split_targets_prefers_joint_label_center(self):
        rows = []
        for center in ("ISPY1", "ISPY2"):
            for label in ("0", "1"):
                for idx in range(3):
                    rows.append(
                        {
                            "id": f"{center}_{label}_{idx}",
                            "label": label,
                            "center": center,
                        }
                    )

        targets, mode = _resolve_split_targets(rows, "label", "center", test_size=0.5)
        self.assertEqual(mode, "joint label-center")
        self.assertEqual(len(targets), len(rows))

    def test_export_internal_split_artifacts_writes_excel_outputs(self):
        train_rows = [
            {"id": "A1", "label": "0", "center": "ISPY1"},
            {"id": "A2", "label": "1", "center": "ISPY2"},
        ]
        test_rows = [
            {"id": "B1", "label": "0", "center": "ISPY1"},
        ]
        fieldnames = ["id", "label", "center"]
        resolved_columns = {"id": "id", "label": "label", "center": "center"}

        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = _export_internal_split_artifacts(
                tmpdir,
                train_rows,
                test_rows,
                fieldnames,
                resolved_columns,
                seed=2026,
                test_size=0.2,
                stratify_mode="label",
            )
            self.assertTrue(os.path.isdir(split_dir))
            self.assertTrue(os.path.isfile(os.path.join(split_dir, "internal_split.xlsx")))
            self.assertTrue(os.path.isfile(os.path.join(split_dir, "train_pool.xlsx")))
            self.assertTrue(os.path.isfile(os.path.join(split_dir, "internal_test.xlsx")))

    def test_patient_grouped_internal_split_has_no_group_overlap(self):
        rows = []
        for group_idx in range(20):
            label = str(group_idx % 2)
            center = "ISPY1" if group_idx % 4 < 2 else "ISPY2"
            for scan_idx in range(2):
                rows.append(
                    {
                        "id": f"P{group_idx}_scan{scan_idx}",
                        "pid": f"P{group_idx}",
                        "label": label,
                        "center": center,
                    }
                )

        train_rows, test_rows, mode, group_summary = _split_rows_patient_grouped(
            rows,
            ["id", "pid", "label", "center"],
            {"id": "id", "label": "label", "center": "center"},
            test_size=0.25,
            seed=2026,
        )

        train_groups = {row["pid"] for row in train_rows}
        test_groups = {row["pid"] for row in test_rows}
        self.assertEqual(train_groups & test_groups, set())
        self.assertEqual(group_summary["patient_group_overlap"], 0)
        self.assertIn("patient-group", mode)

    def test_patient_grouped_internal_split_rejects_conflicting_labels(self):
        rows = [
            {"id": "a", "pid": "P1", "label": "0", "center": "ISPY1"},
            {"id": "b", "pid": "P1", "label": "1", "center": "ISPY1"},
            {"id": "c", "pid": "P2", "label": "0", "center": "ISPY2"},
            {"id": "d", "pid": "P3", "label": "1", "center": "ISPY2"},
        ]

        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            _split_rows_patient_grouped(
                rows,
                ["id", "pid", "label", "center"],
                {"id": "id", "label": "label", "center": "center"},
                test_size=0.25,
                seed=2026,
            )


if __name__ == "__main__":
    unittest.main()
