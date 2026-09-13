import json
import os
import tempfile
import unittest
from unittest import mock

from run_internal_cv import _evaluate_internal_test_set


class TestInternalTestEvaluation(unittest.TestCase):
    def test_internal_test_eval_uses_cv_ensemble_and_oof_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = os.path.join(tmpdir, "InternalCV")
            os.makedirs(run_dir, exist_ok=True)
            for fold_id in (1, 2):
                fold_dir = os.path.join(run_dir, f"fold_{fold_id}")
                os.makedirs(fold_dir, exist_ok=True)
                with open(
                    os.path.join(fold_dir, f"best_model_fold_{fold_id}.pth"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write("checkpoint")
            with open(os.path.join(run_dir, "best_threshold.json"), "w", encoding="utf-8") as f:
                json.dump({"best_threshold": 0.37}, f)
            with open(os.path.join(run_dir, "cv_results.json"), "w", encoding="utf-8") as f:
                json.dump({"folds": [{"fold": 1}, {"fold": 2}]}, f)

            metadata_path = os.path.join(tmpdir, "metadata_internal_test.csv")
            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write("id,label,center\nP1,1,ISPY1\n")
            base_config = {
                "paths": {
                    "data_dir": tmpdir,
                    "metadata_path": os.path.join(tmpdir, "metadata_train.csv"),
                    "log_dir": run_dir,
                },
                "data": {"cache_in_memory": True},
                "model": {"type": "tridual_3d", "in_channels": 2},
                "train": {"batch_size": 1, "amp": False},
                "loss_weights": {"pcr": 1.0},
            }

            with mock.patch("run_internal_cv.run_logged_command", return_value=0) as run_cmd:
                result = _evaluate_internal_test_set(
                    project_dir=tmpdir,
                    run_output_dir=run_dir,
                    base_config=base_config,
                    test_metadata_path=metadata_path,
                    test_rows=[{"id": "P1", "label": "1", "center": "ISPY1"}],
                )

            self.assertIsNotNone(result)
            command = run_cmd.call_args.args[0]
            self.assertIn("test.py", command)
            self.assertIn("--threshold", command)
            self.assertIn("0.37", command)
            self.assertEqual(command.count("--model_path"), 2)
            self.assertTrue(os.path.isfile(result["manifest_path"]))
            with open(result["manifest_path"], "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["mode"], "cv_ensemble_internal_test")
            self.assertEqual(float(manifest["threshold"]), 0.37)


if __name__ == "__main__":
    unittest.main()
