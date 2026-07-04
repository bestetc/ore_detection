import unittest
from pathlib import Path

from ore_detection.models.cs_unet import create_cs_unet
from ore_detection.training.torch_dataset import SourceTorchDataset
from ore_detection.models.simple_unet import create_simple_unet


class TestTorchOptionalPipeline(unittest.TestCase):
    def test_torch_dataset_reports_missing_torch_cleanly(self):
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            with self.assertRaisesRegex(RuntimeError, "PyTorch is required"):
                SourceTorchDataset([])

    def test_simple_unet_factory_reports_missing_torch_cleanly(self):
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            with self.assertRaisesRegex(RuntimeError, "PyTorch is required"):
                create_simple_unet()

    def test_cs_unet_factory_output_shape(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")

        model = create_cs_unet(
            out_channels=5,
            base_channels=4,
            num_heads=2,
            transformer_layers=1,
            token_grid_size=2,
        )
        logits = model(torch.zeros((2, 3, 32, 40), dtype=torch.float32))

        self.assertEqual(tuple(logits.shape), (2, 5, 32, 40))


if __name__ == "__main__":
    unittest.main()
