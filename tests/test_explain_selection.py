import os
import tempfile
import unittest

from utils.explain_selection import load_cv_threshold, select_explain_records


class TestExplainSelection(unittest.TestCase):
    def test_internal_center_is_restricted_to_fold_validation_ids(self):
        all_data = [
            {"id": "A", "center": "C1"},
            {"id": "B", "center": "C1"},
            {"id": "X", "center": "C2"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            fold_dir = os.path.join(tmpdir, "fold_1")
            os.makedirs(fold_dir, exist_ok=True)
            with open(os.path.join(fold_dir, "val_ids.txt"), "w", encoding="utf-8") as f:
                f.write("B\n")

            rows, mode = select_explain_records(
                all_data, "C1", tmpdir, 1, strict_fold_match=True
            )
            self.assertEqual(mode, "fold_validation_subset")
            self.assertEqual([row["id"] for row in rows], ["B"])

    def test_internal_center_without_overlap_raises(self):
        all_data = [
            {"id": "A", "center": "C1"},
            {"id": "B", "center": "C1"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            fold_dir = os.path.join(tmpdir, "fold_1")
            os.makedirs(fold_dir, exist_ok=True)
            with open(os.path.join(fold_dir, "val_ids.txt"), "w", encoding="utf-8") as f:
                f.write("Z\n")

            with self.assertRaisesRegex(ValueError, "refusing to mix in training samples"):
                select_explain_records(all_data, "C1", tmpdir, 1, strict_fold_match=True)

    def test_external_center_falls_back_to_center_only_when_no_overlap(self):
        all_data = [
            {"id": "A", "center": "C1"},
            {"id": "X", "center": "EXT"},
            {"id": "Y", "center": "EXT"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            fold_dir = os.path.join(tmpdir, "fold_1")
            os.makedirs(fold_dir, exist_ok=True)
            with open(os.path.join(fold_dir, "val_ids.txt"), "w", encoding="utf-8") as f:
                f.write("A\n")

            rows, mode = select_explain_records(all_data, "EXT", tmpdir, 1)
            self.assertEqual(mode, "center_only_no_fold_overlap")
            self.assertEqual([row["id"] for row in rows], ["X", "Y"])

    def test_load_cv_threshold_reads_exported_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "best_threshold.json"), "w", encoding="utf-8") as f:
                f.write('{"best_threshold": 0.37}')
            threshold, source = load_cv_threshold(tmpdir)
            self.assertAlmostEqual(threshold, 0.37)
            self.assertEqual(source, "pooled_oof_threshold")

    def test_load_cv_threshold_falls_back_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            threshold, source = load_cv_threshold(tmpdir)
            self.assertAlmostEqual(threshold, 0.5)
            self.assertEqual(source, "default_missing_threshold")


if __name__ == "__main__":
    unittest.main()
