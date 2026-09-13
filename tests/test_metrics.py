import unittest
import numpy as np

from utils.metrics import (
    calculate_fast_metrics,
    binary_metrics_from_probs,
    bootstrap_auc_ci,
)


class TestMetrics(unittest.TestCase):
    def test_calculate_fast_metrics_normal(self):
        y_true = [0, 0, 1, 1]
        y_prob = [0.1, 0.4, 0.6, 0.9]
        metrics = calculate_fast_metrics(y_true, y_prob)
        self.assertEqual(metrics["AUC"], 1.0)
        self.assertGreater(metrics["Sens"], 0.99)
        self.assertGreater(metrics["Spec"], 0.99)
        self.assertIn("Thresh", metrics)

    def test_calculate_fast_metrics_single_class(self):
        """Test robustness when there is only one class (e.g. edge case in small batches)"""
        y_true = [0, 0, 0, 0]
        y_prob = [0.1, 0.2, 0.3, 0.4]
        metrics = calculate_fast_metrics(y_true, y_prob)
        self.assertTrue(np.isnan(metrics["AUC"]))
        self.assertTrue(np.isnan(metrics["Sens"]))
        self.assertTrue(np.isnan(metrics["Spec"]))

    def test_binary_metrics_from_probs(self):
        y_true = [0, 1, 0, 1, 0]
        y_prob = [0.1, 0.9, 0.8, 0.7, 0.2]
        metrics = binary_metrics_from_probs(y_true, y_prob, threshold=0.5)

        self.assertTrue(0.0 <= metrics["auc"] <= 1.0)
        self.assertTrue(0.0 <= metrics["pr_auc"] <= 1.0)
        self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)
        self.assertTrue(0.0 <= metrics["sensitivity"] <= 1.0)
        self.assertTrue(0.0 <= metrics["specificity"] <= 1.0)
        self.assertTrue(0.0 <= metrics["ppv"] <= 1.0)
        self.assertTrue(0.0 <= metrics["npv"] <= 1.0)
        self.assertGreaterEqual(metrics["brier_score"], 0.0)

    def test_bootstrap_auc_ci(self):
        y_true = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10
        y_prob = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5] * 10

        ci = bootstrap_auc_ci(y_true, y_prob, seed=42, n_boot=100)
        self.assertEqual(len(ci), 2)
        self.assertLessEqual(ci[0], ci[1])
        self.assertGreater(ci[0], 0.5)


if __name__ == "__main__":
    unittest.main()
