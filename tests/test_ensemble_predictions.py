import csv
import tempfile
import unittest
from pathlib import Path

from tools.ensemble_predictions import (
    align_prediction_runs,
    evaluate_combinations,
    load_prediction_run,
)


def _write_prediction_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "prob"])
        writer.writerows(rows)


class TestEnsemblePredictions(unittest.TestCase):
    def test_pairwise_soft_voting_can_outperform_each_single_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows_a = [
                ["s1", 0, 0.10],
                ["s2", 0, 0.90],
                ["s3", 1, 0.60],
                ["s4", 1, 0.70],
                ["s5", 1, 0.80],
                ["s6", 0, 0.20],
            ]
            rows_b = [
                ["s1", 0, 0.20],
                ["s2", 0, 0.30],
                ["s3", 1, 0.55],
                ["s4", 1, 0.65],
                ["s5", 1, 0.75],
                ["s6", 0, 0.85],
            ]
            rows_c = [
                ["s1", 0, 0.05],
                ["s2", 0, 0.95],
                ["s3", 1, 0.40],
                ["s4", 1, 0.45],
                ["s5", 1, 0.50],
                ["s6", 0, 0.90],
            ]
            path_a = root / "a.csv"
            path_b = root / "b.csv"
            path_c = root / "c.csv"
            _write_prediction_csv(path_a, rows_a)
            _write_prediction_csv(path_b, rows_b)
            _write_prediction_csv(path_c, rows_c)

            runs = [
                load_prediction_run(str(path_a), name="A"),
                load_prediction_run(str(path_b), name="B"),
                load_prediction_run(str(path_c), name="C"),
            ]
            ids, labels, aligned_probs = align_prediction_runs(runs)
            rows = evaluate_combinations(
                ids=ids,
                y_true=labels,
                aligned_probs=aligned_probs,
                min_size=1,
                max_size=3,
            )

            self.assertEqual(rows[0]["combo_key"], "A + B")
            self.assertAlmostEqual(rows[0]["auc"], 8.0 / 9.0, places=6)
            single_aucs = {row["combo_key"]: row["auc"] for row in rows if row["combo_size"] == 1}
            self.assertLess(single_aucs["A"], rows[0]["auc"])
            self.assertLess(single_aucs["B"], rows[0]["auc"])

    def test_alignment_rejects_mismatched_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path_a = root / "a.csv"
            path_b = root / "b.csv"
            _write_prediction_csv(
                path_a,
                [["s1", 0, 0.10], ["s2", 1, 0.90]],
            )
            _write_prediction_csv(
                path_b,
                [["s1", 0, 0.20], ["s3", 1, 0.80]],
            )

            runs = [
                load_prediction_run(str(path_a), name="A"),
                load_prediction_run(str(path_b), name="B"),
            ]
            with self.assertRaises(ValueError):
                align_prediction_runs(runs)


if __name__ == "__main__":
    unittest.main()
