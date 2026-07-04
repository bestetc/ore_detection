import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.training.source_dataset import SourceSegmentationSample, list_source_samples, load_sample_images


class TestSourceDataset(unittest.TestCase):
    def test_list_source_samples_pairs_images_with_binary_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "datasets" / "set_1" / "imgs" / "train" / "train_01.jpg"
            mask_path = root / "binary_masks" / "set_1" / "train" / "train_01.png"
            image_path.parent.mkdir(parents=True)
            mask_path.parent.mkdir(parents=True)
            Image.new("RGB", (4, 3), color=(1, 2, 3)).save(image_path)
            Image.new("L", (4, 3), color=1).save(mask_path)

            samples = list_source_samples(
                datasets_root=root / "datasets",
                binary_masks_root=root / "binary_masks",
                datasets=("set_1",),
            )

            self.assertEqual(len(samples), 1)
            self.assertIsInstance(samples[0], SourceSegmentationSample)
            self.assertEqual(samples[0].dataset, "set_1")
            self.assertEqual(samples[0].split, "train")
            self.assertEqual(samples[0].stem, "train_01")
            self.assertEqual(samples[0].width, 4)
            self.assertEqual(samples[0].height, 3)
            self.assertEqual(samples[0].mask_path, mask_path)

    def test_list_source_samples_can_filter_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "test"):
                image_path = root / "datasets" / "set_2" / "imgs" / split / f"{split}_01.jpg"
                mask_path = root / "binary_masks" / "set_2" / split / f"{split}_01.png"
                image_path.parent.mkdir(parents=True)
                mask_path.parent.mkdir(parents=True)
                Image.new("RGB", (2, 2)).save(image_path)
                Image.new("L", (2, 2)).save(mask_path)

            samples = list_source_samples(
                datasets_root=root / "datasets",
                binary_masks_root=root / "binary_masks",
                datasets=("set_2",),
                split="test",
            )

            self.assertEqual([sample.split for sample in samples], ["test"])

    def test_load_sample_images_returns_rgb_image_and_l_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.jpg"
            mask_path = root / "mask.png"
            Image.new("RGB", (2, 2), color=(1, 2, 3)).save(image_path)
            Image.new("L", (2, 2), color=1).save(mask_path)
            sample = SourceSegmentationSample(
                dataset="set_1",
                split="train",
                stem="image",
                image_path=image_path,
                mask_path=mask_path,
                width=2,
                height=2,
            )

            image, mask = load_sample_images(sample)

            self.assertEqual(image.mode, "RGB")
            self.assertEqual(mask.mode, "L")
            self.assertEqual(image.size, mask.size)


if __name__ == "__main__":
    unittest.main()
