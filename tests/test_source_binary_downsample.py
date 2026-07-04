import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.training.source_binary_downsample import prepare_downsampled_source_binary_dataset
from ore_detection.training.source_ore_downsample import model_compatible_downsample_size


class TestSourceBinaryDownsample(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")
        self.torch = torch

    def test_prepare_downsamples_binary_images_masks_and_uses_train_stats_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_image = root / "datasets" / "set_1" / "imgs" / "train" / "train_01.png"
            test_image = root / "datasets" / "set_1" / "imgs" / "test" / "test_01.png"
            train_mask = root / "binary_masks" / "set_1" / "train" / "train_01.png"
            test_mask = root / "binary_masks" / "set_1" / "test" / "test_01.png"
            train_image.parent.mkdir(parents=True)
            test_image.parent.mkdir(parents=True)
            train_mask.parent.mkdir(parents=True)
            test_mask.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), color=(0, 255, 0)).save(train_image)
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(test_image)
            mask = Image.new("L", (8, 8), color=0)
            for x in range(4, 8):
                for y in range(8):
                    mask.putpixel((x, y), 255)
            mask.save(train_mask)
            mask.save(test_mask)

            summary = prepare_downsampled_source_binary_dataset(
                datasets_root=root / "datasets",
                binary_masks_root=root / "binary_masks",
                output_root=root / "downsampled",
                datasets=("set_1",),
                factor=2,
                size_divisor=2,
                const_path=root / "source_binary_const.py",
            )

            self.assertEqual(summary["sample_count"], 2)
            output_image = root / "downsampled" / "images" / "set_1" / "train" / "train_01.png"
            output_mask = root / "downsampled" / "masks" / "set_1" / "train" / "train_01.pt"
            with Image.open(output_image) as image:
                self.assertEqual(image.size, (4, 4))
            mask_tensor = self.torch.load(output_mask, map_location="cpu", weights_only=True)
            self.assertEqual(tuple(mask_tensor.shape), (4, 4))
            self.assertEqual(set(int(value) for value in mask_tensor.flatten()), {0, 1})
            self.assertAlmostEqual(summary["stats"]["mean"][0], 0.0)
            self.assertAlmostEqual(summary["stats"]["mean"][1], 1.0)
            self.assertAlmostEqual(summary["stats"]["mean"][2], 0.0)
            self.assertIn("SOURCE_BINARY_TRAIN_RGB_MEAN", (root / "source_binary_const.py").read_text(encoding="utf-8"))

    def test_model_compatible_size_matches_binary_dataset_expectation(self):
        self.assertEqual(model_compatible_downsample_size(3396, 2547, factor=4, size_divisor=4), (848, 636))


if __name__ == "__main__":
    unittest.main()
