import csv
import importlib.util
import os
import tempfile
import unittest

import numpy as np

from utils.metadata import load_metadata_records


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    from datasets.mri_dataset import MRIDataset


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for dataset integration tests")
class TestDatasetChain(unittest.TestCase):
    def test_metadata_to_dataset_chain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = os.path.join(tmpdir, "images")
            mask_dir = os.path.join(tmpdir, "masks")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            np.save(
                os.path.join(image_dir, "P1.npy"),
                np.ones((4, 6, 8, 10), dtype=np.float32),
            )
            np.save(
                os.path.join(mask_dir, "P1.npy"),
                np.ones((1, 6, 8, 10), dtype=np.float32),
            )

            csv_path = os.path.join(tmpdir, "metadata.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["id", "label", "center", "ihc_cd8", "ihc_cd34", "ihc_ki67"]
                )
                writer.writerow(["P1", "1", "C1", "1.5", "2.5", "3.5"])

            records = load_metadata_records(csv_path, include_ihc=True, require_unique_ids=True)
            dataset = MRIDataset(
                records,
                tmpdir,
                {"in_channels": 4, "use_kinetics_channel": False},
                is_train=False,
                strict_file_check=True,
                split_name="test",
            )

            sample = dataset[0]
            self.assertEqual(sample["id"], "P1")
            self.assertEqual(tuple(sample["image"].shape), (4, 6, 8, 10))
            self.assertEqual(tuple(sample["mask"].shape), (1, 6, 8, 10))
            self.assertEqual(int(sample["label"].item()), 1)
            self.assertEqual(float(sample["has_ihc"].item()), 1.0)
            self.assertTrue(np.allclose(sample["ihc"].numpy()[:3], np.array([1.5, 2.5, 3.5])))


if __name__ == "__main__":
    unittest.main()
