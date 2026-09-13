import unittest
import tempfile
import csv

from utils.metadata import load_metadata_records, resolve_required_metadata_columns


class TestMetadata(unittest.TestCase):
    def test_load_metadata_records(self):
        """Test standard case with id, label, center."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label", "center", "ihc_cd8"])
            writer.writerow(["P1", "0", "C1", "1.5"])
            writer.writerow(["P2", "1", "C2", "0.0"])
            temp_path = f.name

        records = load_metadata_records(temp_path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["id"], "P1")
        self.assertEqual(records[0]["label"], 0)
        self.assertEqual(records[0]["center"], "C1")
        self.assertEqual(records[0]["ihc_cd8"], 1.5)

    def test_non_numeric_ihc_is_dropped_without_dropping_row(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label", "center", "ihc_cd8", "ihc_cd34"])
            writer.writerow(["P1", "0", "C1", "pos", "2.0"])
            temp_path = f.name

        records = load_metadata_records(temp_path)
        self.assertEqual(len(records), 1)
        self.assertNotIn("ihc_cd8", records[0])
        self.assertEqual(records[0]["ihc_cd34"], 2.0)

    def test_missing_required_column(self):
        """Test exception raising when missing required columns."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "center"])  # Missing label
            writer.writerow(["P1", "C1"])
            temp_path = f.name

        with self.assertRaisesRegex(ValueError, "Missing required metadata columns"):
            load_metadata_records(temp_path)

    def test_duplicate_ids(self):
        """Test duplicate detection."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label", "center"])
            writer.writerow(["P1", "0", "C1"])
            writer.writerow(["P1", "1", "C2"])  # Duplicate ID
            temp_path = f.name

        with self.assertRaisesRegex(ValueError, "Duplicate sample IDs detected"):
            load_metadata_records(temp_path, require_unique_ids=True)

    def test_invalid_required_row_is_reported(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label", "center"])
            writer.writerow(["P1", "not-an-int", "C1"])
            temp_path = f.name

        with self.assertRaisesRegex(ValueError, "invalid metadata row"):
            load_metadata_records(temp_path)

    def test_non_binary_label_is_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label", "center"])
            writer.writerow(["P1", "2", "C1"])
            temp_path = f.name

        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            load_metadata_records(temp_path)

    def test_missing_label_row_is_skipped(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label", "center"])
            writer.writerow(["P1", "0", "C1"])
            writer.writerow(["P2", "", "C1"])
            writer.writerow(["P3", "1", "C2"])
            temp_path = f.name

        records = load_metadata_records(temp_path)
        self.assertEqual([record["id"] for record in records], ["P1", "P3"])
        self.assertEqual([record["label"] for record in records], [0, 1])

    def test_latin1_metadata_is_read_when_utf8_fails(self):
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".csv") as f:
            f.write(b"id,label,center,note\n")
            f.write(b"P1,0,C1,\xe5\xe5\n")
            temp_path = f.name

        records = load_metadata_records(temp_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], "P1")

    def test_pid_alias_is_supported(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            writer = csv.writer(f)
            writer.writerow(["pid", "pCR", "center"])
            writer.writerow(["P1", "1", "C1"])
            temp_path = f.name

        records = load_metadata_records(temp_path)
        self.assertEqual(records[0]["id"], "P1")
        self.assertEqual(records[0]["label"], 1)

    def test_required_column_aliases_are_supported(self):
        resolved = resolve_required_metadata_columns(["ID", "PCR", "site"])
        self.assertEqual(resolved["id"], "ID")
        self.assertEqual(resolved["label"], "PCR")
        self.assertEqual(resolved["center"], "site")


if __name__ == "__main__":
    unittest.main()
