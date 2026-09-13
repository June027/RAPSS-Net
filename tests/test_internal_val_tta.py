import unittest

try:
    import torch
    from engine.trainer import KFoldTrainer
except ModuleNotFoundError:
    torch = None
    KFoldTrainer = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this test environment.")
class TestInternalValTTA(unittest.TestCase):
    def test_resolve_internal_val_tta_views_from_train_config(self):
        trainer = KFoldTrainer.__new__(KFoldTrainer)
        trainer.config = {
            "train": {
                "internal_val_tta_enabled": True,
                "internal_val_tta_flip_axes": [["D"], ["H", "W"], "D"],
            }
        }

        self.assertEqual(trainer._resolve_eval_tta_views(), [(), (2,), (3, 4)])

    def test_internal_val_tta_disabled_by_default(self):
        trainer = KFoldTrainer.__new__(KFoldTrainer)
        trainer.config = {"train": {}, "test": {"tta_enabled": True, "tta_flip_axes": [["D"]]}}

        self.assertEqual(trainer._resolve_eval_tta_views(), [()])

    def test_apply_tta_volume_uses_spatial_dims(self):
        tensor = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        flipped = KFoldTrainer._apply_tta_volume(tensor, (2, 4))

        self.assertTrue(torch.equal(flipped, torch.flip(tensor, dims=[1, 3])))


if __name__ == "__main__":
    unittest.main()
