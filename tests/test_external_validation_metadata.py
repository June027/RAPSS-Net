import csv
import os
import tempfile
import unittest

import pandas as pd

from run_external_validation import (
    _coerce_binary_label,
    _standardize_external_metadata,
)


class TestExternalValidationMetadata(unittest.TestCase):
    def test_unspecified_center_keeps_existing_centers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.csv")
            output_path = os.path.join(tmpdir, "standardized.csv")
            with open(source_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "center", "label"])
                writer.writerow(["P1", "C1", "0"])
                writer.writerow(["P2", "C2", "1"])

            info = _standardize_external_metadata(
                source_path,
                output_path,
                test_center=None,
                filter_center=False,
                default_center="External",
            )

            df = pd.read_csv(output_path)
            self.assertEqual(info["num_records"], 2)
            self.assertEqual(df["center"].tolist(), ["C1", "C2"])

    def test_requested_label_column_overrides_label_alias(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.csv")
            output_path = os.path.join(tmpdir, "standardized.csv")
            with open(source_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "center", "label", "PCR"])
                writer.writerow(["P1", "External", "0", "1"])

            info = _standardize_external_metadata(
                source_path,
                output_path,
                test_center="External",
                label_column="PCR",
            )

            df = pd.read_csv(output_path)
            self.assertEqual(info["label_column"], "PCR")
            self.assertEqual(df["label"].tolist(), [1])

    def test_binary_label_parser_rejects_unstructured_text(self):
        self.assertEqual(_coerce_binary_label("G5=1"), 1)
        with self.assertRaisesRegex(ValueError, "Cannot parse binary label"):
            _coerce_binary_label("grade 10")


if __name__ == "__main__":
    unittest.main()
