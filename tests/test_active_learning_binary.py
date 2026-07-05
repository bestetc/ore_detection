import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.training.active_learning_binary import (
    balanced_class_weights_from_counts,
    list_active_learning_binary_samples,
    load_active_learning_binary_sample,
)


class TestActiveLearningBinary(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")
        self.torch = torch

    def _classes(self):
        return [
            {"id": 0, "name": "background", "color": [0, 0, 0]},
            {"id": 1, "name": "sulfide_ore", "color": [0, 220, 0]},
            {"id": 3, "name": "talc", "color": [255, 255, 255]},
            {"id": 4, "name": "normal_ore", "color": [0, 120, 255]},
            {"id": 255, "name": "ignore", "color": [255, 0, 255]},
        ]

    def test_balanced_class_weights_use_sqrt_inverse_frequency_with_clamp(self):
        weights = balanced_class_weights_from_counts([100, 25, 4])

        self.assertEqual(weights, (1.0, 2.0, 5.0))
        self.assertEqual(balanced_class_weights_from_counts([100, 0]), (1.0, 8.0))

    def test_full_edited_mask_converts_sulfide_and_talc_to_trainable_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "active" / "sample"
            sample_dir.mkdir(parents=True)
            image_path = root / "raw.png"
            mask_path = sample_dir / "class_index_mask.png"
            Image.new("RGB", (2, 2), (10, 20, 30)).save(image_path)
            mask = Image.new("L", (2, 2))
            mask.putdata([0, 1, 3, 255])
            mask.save(mask_path)
            metadata = {
                "format": "single_class_index_png_and_torch_one_hot_chw",
                "source_image_path": str(image_path),
                "class_index_mask": str(mask_path),
                "classes": self._classes(),
            }
            (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            samples = list_active_learning_binary_samples(root / "active")
            loaded = load_active_learning_binary_sample(samples[0])

            self.assertEqual(len(samples), 1)
            self.assertEqual(loaded["mask"].flatten().tolist(), [0, 1, 2, 0])
            self.assertEqual(loaded["weight"].flatten().tolist(), [1, 1, 1, 0])

    def test_panorama_review_uses_reviewed_tile_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "active" / "panorama"
            tile_dir = sample_dir / "reviewed_tiles"
            tile_dir.mkdir(parents=True)
            image_path = root / "raw.png"
            Image.new("RGB", (8, 8), (10, 20, 30)).save(image_path)
            tile_path = tile_dir / "patch_00001.pt"
            self.torch.save(
                {
                    "class_index": self.torch.tensor([[1, 4], [3, 255]], dtype=self.torch.uint8),
                    "crop_box": (2, 2, 4, 4),
                    "source_image_path": str(image_path),
                },
                tile_path,
            )
            metadata = {
                "format": "single_class_index_png_patch_log_and_patch_crop_tensors",
                "source_image_path": str(image_path),
                "class_index_mask": str(sample_dir / "class_index_mask.png"),
                "tensor_tiles": [str(tile_path)],
                "classes": self._classes(),
            }
            Image.new("L", (8, 8), 0).save(sample_dir / "class_index_mask.png")
            (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            samples = list_active_learning_binary_samples(root / "active")
            loaded = load_active_learning_binary_sample(samples[0], output_size=4)

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].crop_box, (2, 2, 4, 4))
            self.assertEqual(tuple(loaded["image"].shape), (3, 4, 4))
            self.assertEqual(tuple(loaded["mask"].shape), (4, 4))
            self.assertEqual(int(loaded["mask"][2, 0]), 2)
            self.assertEqual(int(loaded["weight"][2, 0]), 1)
            self.assertEqual(int(loaded["weight"][2, 2]), 0)


if __name__ == "__main__":
    unittest.main()
