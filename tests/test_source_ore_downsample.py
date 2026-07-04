import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.data.ore_type_legend import load_legend_config
from ore_detection.training.source_ore_downsample import (
    model_compatible_downsample_size,
    prepare_downsampled_source_ore_dataset,
)


class TestSourceOreDownsample(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")
        self.torch = torch
        self.legend = load_legend_config()

    def _save_one_hot_mask(self, path: Path, *, class_name: str, width: int = 8, height: int = 8):
        mask = self.torch.zeros((self.legend.class_count, height, width), dtype=self.torch.uint8)
        mask[self.legend.background_index] = 1
        class_index = self.legend.class_index(class_name)
        mask[self.legend.background_index, :, width // 2 :] = 0
        mask[class_index, :, width // 2 :] = 1
        path.parent.mkdir(parents=True)
        self.torch.save(mask, path)

    def test_model_compatible_downsample_size_rounds_to_divisible_dimensions(self):
        self.assertEqual(model_compatible_downsample_size(3396, 2547, factor=4, size_divisor=4), (848, 636))

    def test_prepare_downsamples_images_masks_and_uses_train_stats_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_image = root / "datasets" / "set_1" / "imgs" / "train" / "train_01.png"
            test_image = root / "datasets" / "set_1" / "imgs" / "test" / "test_01.png"
            train_mask = root / "masks" / "set_1" / "train" / "train_01.pt"
            test_mask = root / "masks" / "set_1" / "test" / "test_01.pt"
            train_image.parent.mkdir(parents=True)
            test_image.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(train_image)
            Image.new("RGB", (8, 8), color=(0, 0, 255)).save(test_image)
            self._save_one_hot_mask(train_mask, class_name="chalcopyrite")
            self._save_one_hot_mask(test_mask, class_name="pyrite")

            summary = prepare_downsampled_source_ore_dataset(
                datasets_root=root / "datasets",
                masks_root=root / "masks",
                output_root=root / "downsampled",
                datasets=("set_1",),
                factor=2,
                size_divisor=2,
                const_path=root / "const.py",
            )

            self.assertEqual(summary["sample_count"], 2)
            output_image = root / "downsampled" / "images" / "set_1" / "train" / "train_01.png"
            output_mask = root / "downsampled" / "masks" / "set_1" / "train" / "train_01.pt"
            with Image.open(output_image) as image:
                self.assertEqual(image.size, (4, 4))
            mask = self.torch.load(output_mask, map_location="cpu", weights_only=True)
            self.assertEqual(tuple(mask.shape), (4, 4))
            self.assertIn(self.legend.class_index("chalcopyrite"), set(int(value) for value in mask.flatten()))
            self.assertAlmostEqual(summary["stats"]["mean"][0], 1.0)
            self.assertAlmostEqual(summary["stats"]["mean"][1], 0.0)
            self.assertAlmostEqual(summary["stats"]["mean"][2], 0.0)
            self.assertIn("SOURCE_ORE_TRAIN_RGB_MEAN", (root / "const.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
