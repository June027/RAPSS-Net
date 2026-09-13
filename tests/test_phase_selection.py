import importlib.util
import os
import pathlib
import sys
import tempfile
import types
import unittest


if "nibabel" not in sys.modules:
    nibabel_stub = types.ModuleType("nibabel")
    nibabel_stub.io_orientation = lambda affine: affine
    nibabel_stub.orientations = types.SimpleNamespace(
        axcodes2ornt=lambda code: code,
        ornt_transform=lambda orig, target: (orig, target),
    )
    sys.modules["nibabel"] = nibabel_stub

if "SimpleITK" not in sys.modules:
    sitk_stub = types.ModuleType("SimpleITK")

    class _ProcessObject:
        @staticmethod
        def SetGlobalDefaultNumberOfThreads(_value):
            return None

    sitk_stub.ProcessObject = _ProcessObject
    sys.modules["SimpleITK"] = sitk_stub


MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "pre" / "prepare_all_to_npy_mask.py"
SPEC = importlib.util.spec_from_file_location("prepare_all_to_npy_mask", MODULE_PATH)
PREPARE_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREPARE_MODULE)


class TestPhaseSelection(unittest.TestCase):
    def setUp(self):
        self.all_nii = [
            r"E:\mock\case_001\case_001_acq1.nii.gz",
            r"E:\mock\case_001\case_001_acq3.nii.gz",
            r"E:\mock\case_001\case_001_mask.nii.gz",
        ]
        self.phase_cfg = {
            "enabled": True,
            "phase_columns": ["post_early", "post_late"],
            "filename_phase_regex": r"acq(?P<phase>\d+)",
        }

    def test_resolves_metadata_by_numeric_phase_index(self):
        row_data = {"post_early": 1, "post_late": 3}
        selection, error = PREPARE_MODULE.resolve_phase_selected_images(
            self.all_nii, row_data, self.phase_cfg
        )
        self.assertIsNone(error)
        self.assertEqual(
            [pathlib.Path(path).name for path in selection["paths"]],
            ["case_001_acq1.nii.gz", "case_001_acq3.nii.gz"],
        )

    def test_resolves_metadata_by_phase_suffix(self):
        row_data = {"post_early": "acq1", "post_late": "acq3"}
        selection, error = PREPARE_MODULE.resolve_phase_selected_images(
            self.all_nii, row_data, self.phase_cfg
        )
        self.assertIsNone(error)
        self.assertEqual(selection["phase_indices"], ["acq1", "acq3"])

    def test_resolves_metadata_by_full_filename(self):
        row_data = {
            "post_early": "case_001_acq1.nii.gz",
            "post_late": r"E:\tmp\case_001_acq3.nii.gz",
        }
        selection, error = PREPARE_MODULE.resolve_phase_selected_images(
            self.all_nii, row_data, self.phase_cfg
        )
        self.assertIsNone(error)
        self.assertEqual(
            [pathlib.Path(path).name for path in selection["paths"]],
            ["case_001_acq1.nii.gz", "case_001_acq3.nii.gz"],
        )

    def test_resolves_private_phases_by_filename_keywords(self):
        all_nii = [
            r"E:\mock\LZ_001\LZ_001_ZQ.nii.gz",
            r"E:\mock\LZ_001\LZ_001_Fz.nii.gz",
            r"E:\mock\LZ_001\LZ_001_mask.nii.gz",
        ]
        phase_cfg = {
            "enabled": True,
            "use_metadata_columns": False,
            "phase_columns": ["zq", "fz"],
            "phase_filename_keywords": {"zq": ["zq"], "fz": ["fz"]},
        }
        selection, error = PREPARE_MODULE.resolve_phase_selected_images(
            all_nii, {}, phase_cfg
        )
        self.assertIsNone(error)
        self.assertEqual(
            [pathlib.Path(path).name for path in selection["paths"]],
            ["LZ_001_ZQ.nii.gz", "LZ_001_Fz.nii.gz"],
        )

    def test_ispy2_falls_back_to_phase_groups_when_metadata_missing(self):
        all_nii = [
            r"E:\mock\ISPY2\ISPY2-001_spy2_aqc_2.nii.gz",
            r"E:\mock\ISPY2\ISPY2-001_spy2_aqc_5.nii.gz",
            r"E:\mock\ISPY2\ISPY2-001_mask.nii.gz",
        ]
        phase_cfg = {
            "enabled": True,
            "phase_columns": ["post_early", "post_late"],
            "filename_phase_regex": r"(?:aqc|acq)[_-]?(?P<phase>\d+)",
            "fallback_phase_groups": {
                "post_early": [2, 3, 4],
                "post_late": [5, 6],
            },
        }
        selection, error = PREPARE_MODULE.resolve_phase_selected_images(
            all_nii, {}, phase_cfg
        )
        self.assertIsNone(error)
        self.assertEqual(selection["phase_indices"], [2, 5])
        self.assertEqual(
            selection["phase_sources"],
            ["phase_group_fallback", "phase_group_fallback"],
        )

    def test_failed_case_is_copied_without_moving_original_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            patient_dir = os.path.join(tmpdir, "case_001")
            problematic_dir = os.path.join(tmpdir, "problematic")
            out_root = os.path.join(tmpdir, "out")
            os.makedirs(patient_dir, exist_ok=True)
            os.makedirs(os.path.join(out_root, "images"), exist_ok=True)
            os.makedirs(os.path.join(out_root, "masks"), exist_ok=True)
            with open(
                os.path.join(patient_dir, "case_001_acq1.nii.gz"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("dummy")

            report = PREPARE_MODULE.process_single_patient(
                patient_dir=patient_dir,
                ds_name="TEST",
                row_data={"pCR": "1"},
                pcr_col="pCR",
                out_root=out_root,
                problematic_raw_dir=problematic_dir,
                target_shape=(32, 128, 128),
                interp_order=3,
                n4_iters=[1, 1, 1, 1],
                n4_enabled=True,
                standardization_mode="per_phase",
                phase_selection_cfg={"enabled": False},
            )

            archived_dir = os.path.join(problematic_dir, "TEST_case_001")
            self.assertEqual(report["status"], "failed")
            self.assertTrue(os.path.isdir(patient_dir))
            self.assertTrue(os.path.isdir(archived_dir))
            self.assertTrue(
                os.path.isfile(os.path.join(archived_dir, "case_001_acq1.nii.gz"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(archived_dir, "_failure_reason.txt"))
            )


if __name__ == "__main__":
    unittest.main()
