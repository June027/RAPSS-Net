import unittest

from utils.calibration import expected_calibration_error, reliability_bin_stats


class TestCalibration(unittest.TestCase):
    def test_expected_calibration_error_uses_empirical_positive_rate(self):
        y_true = [0, 1]
        y_prob = [0.1, 0.9]
        self.assertAlmostEqual(
            expected_calibration_error(y_true, y_prob, n_bins=1), 0.0
        )

    def test_reliability_bin_stats_reports_positive_fraction(self):
        empirical_rates, confidence_means, counts = reliability_bin_stats(
            [0, 1], [0.1, 0.9], n_bins=1
        )
        self.assertEqual(counts, [2])
        self.assertAlmostEqual(empirical_rates[0], 0.5)
        self.assertAlmostEqual(confidence_means[0], 0.5)


if __name__ == "__main__":
    unittest.main()
