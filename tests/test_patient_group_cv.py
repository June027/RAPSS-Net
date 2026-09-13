import unittest

import numpy as np

from train import _patient_group_id, validate_cv_feasibility


class TestPatientGroupCV(unittest.TestCase):
    def test_patient_group_id_uses_explicit_group(self):
        self.assertEqual(
            _patient_group_id({"id": "scan_1", "patient_group": "patient_a"}),
            "patient_a",
        )

    def test_validate_cv_feasibility_counts_patient_groups(self):
        labels = np.array([0, 0, 1, 1])
        groups = np.array(["P1", "P1", "P2", "P2"])

        with self.assertRaisesRegex(ValueError, "minority class"):
            validate_cv_feasibility(labels, n_splits=2, groups=groups)

    def test_validate_cv_feasibility_rejects_group_label_conflict(self):
        labels = np.array([0, 1, 0, 1])
        groups = np.array(["P1", "P1", "P2", "P3"])

        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            validate_cv_feasibility(labels, n_splits=2, groups=groups)


if __name__ == "__main__":
    unittest.main()
