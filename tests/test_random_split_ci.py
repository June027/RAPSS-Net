import unittest

try:
    from run_random_split import _fmt_ci, _with_auc_ci
except ModuleNotFoundError:
    _fmt_ci = None
    _with_auc_ci = None


@unittest.skipIf(_with_auc_ci is None, "Random split dependencies are not installed.")
class TestRandomSplitAucCi(unittest.TestCase):
    def test_metric_payload_includes_auc_ci(self):
        metrics = {
            "PCR_Loss": 0.1,
            "AUC": 0.8,
            "PR-AUC": 0.7,
            "Precision": 0.6,
            "Recall": 0.5,
            "F1": 0.55,
            "ACC": 0.75,
            "Micro-ACC": 0.75,
            "NPV": 0.8,
            "Kappa": 0.4,
            "Sens": 0.5,
            "Spec": 0.9,
            "Thresh": 0.45,
        }

        payload = _with_auc_ci(metrics, [0.61, 0.93])

        self.assertEqual(payload["auc_95ci"], [0.61, 0.93])
        self.assertEqual(payload["auc"], 0.8)

    def test_fmt_ci_handles_missing_bounds(self):
        self.assertEqual(_fmt_ci([0.61, 0.93]), "[0.6100, 0.9300]")
        self.assertEqual(_fmt_ci([None, None]), "n/a")


if __name__ == "__main__":
    unittest.main()
