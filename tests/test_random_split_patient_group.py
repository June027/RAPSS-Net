import unittest

from run_random_split import _split_patient_group_records


class TestRandomSplitPatientGroup(unittest.TestCase):
    def test_default_random_split_keeps_patient_groups_disjoint(self):
        records = [
            {"id": "p1_a", "patient_group": "P1", "label": 0, "center": "ISPY1"},
            {"id": "p1_b", "patient_group": "P1", "label": 0, "center": "ISPY1"},
            {"id": "p2_a", "patient_group": "P2", "label": 0, "center": "ISPY2"},
            {"id": "p3_a", "patient_group": "P3", "label": 1, "center": "ISPY1"},
            {"id": "p3_b", "patient_group": "P3", "label": 1, "center": "ISPY1"},
            {"id": "p4_a", "patient_group": "P4", "label": 1, "center": "ISPY2"},
        ]

        train_records, val_records, stratify_mode = _split_patient_group_records(
            records,
            val_size=0.5,
            seed=2026,
        )

        train_groups = {row["patient_group"] for row in train_records}
        val_groups = {row["patient_group"] for row in val_records}
        self.assertEqual(train_groups & val_groups, set())
        self.assertTrue(stratify_mode.startswith("patient_group-"))
        for group_id in ["P1", "P3"]:
            locations = {
                "train" if row in train_records else "val"
                for row in records
                if row["patient_group"] == group_id
            }
            self.assertEqual(len(locations), 1)

    def test_random_split_rejects_conflicting_patient_group_labels(self):
        records = [
            {"id": "p1_a", "patient_group": "P1", "label": 0, "center": "ISPY1"},
            {"id": "p1_b", "patient_group": "P1", "label": 1, "center": "ISPY1"},
            {"id": "p2_a", "patient_group": "P2", "label": 0, "center": "ISPY2"},
            {"id": "p3_a", "patient_group": "P3", "label": 1, "center": "ISPY2"},
        ]

        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            _split_patient_group_records(records, val_size=0.5, seed=2026)


if __name__ == "__main__":
    unittest.main()
