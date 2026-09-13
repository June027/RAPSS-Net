import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.check_split_consistency import summarize_consistency


class TestSplitConsistency(unittest.TestCase):
    def test_low_risk_when_distributions_match(self):
        train_df = pd.DataFrame(
            {
                "id": ["a", "b", "c", "d"],
                "label": [0, 1, 0, 1],
                "center": ["C1", "C1", "C2", "C2"],
                "ihc_cd8": [1.0, 2.0, 1.1, 1.9],
            }
        )
        val_df = pd.DataFrame(
            {
                "id": ["e", "f", "g", "h"],
                "label": [0, 1, 0, 1],
                "center": ["C1", "C1", "C2", "C2"],
                "ihc_cd8": [1.0, 2.0, 1.1, 1.9],
            }
        )
        report = summarize_consistency(train_df, val_df)
        self.assertEqual(report["overall_risk"], "low")

    def test_high_risk_when_center_distribution_is_shifted(self):
        train_df = pd.DataFrame(
            {
                "id": ["a", "b", "c", "d"],
                "label": [0, 1, 0, 1],
                "center": ["C1", "C1", "C1", "C1"],
            }
        )
        val_df = pd.DataFrame(
            {
                "id": ["e", "f", "g", "h"],
                "label": [0, 1, 0, 1],
                "center": ["C2", "C2", "C2", "C2"],
            }
        )
        report = summarize_consistency(train_df, val_df)
        self.assertEqual(report["overall_risk"], "high")
        self.assertTrue(any("center" in reason for reason in report["high_risk_reasons"]))


if __name__ == "__main__":
    unittest.main()
