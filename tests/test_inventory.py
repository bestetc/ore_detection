import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from ore_detection.data.inventory import inventory_baseline_images, inventory_source_dataset


class TestInventory(unittest.TestCase):
    def test_inventory_baseline_parses_part_and_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "baseline" / "Part 2" / "Hard ore" / "sample.JPG"
            panorama = root / "baseline" / "panoramas" / "unlabeled"
            nested_panorama = root / "baseline" / "panoramas" / "Part 2" / "Hard ore" / "unlabeled"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (3, 2), color=(1, 2, 3)).save(image_path)

            with patch("ore_detection.data.inventory._iter_files", return_value=[image_path, panorama, nested_panorama]):
                records = inventory_baseline_images(root / "baseline")

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["dataset"], "baseline")
            self.assertEqual(records[0]["part"], "Part 2")
            self.assertEqual(records[0]["label"], "Hard ore")
            self.assertEqual(records[0]["width"], 3)
            self.assertEqual(records[0]["height"], 2)
            self.assertEqual(records[0]["magnification"], "10x")

    def test_inventory_source_dataset_pairs_images_and_masks_by_split_and_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "set_1" / "imgs" / "train" / "train_01.jpg"
            mask = root / "set_1" / "masks_colored" / "train" / "train_01.png"
            skipped_mask = root / "set_1" / "masks" / "train" / "train_01.png"
            skipped_human_mask = root / "set_1" / "masks_human" / "train" / "train_01.png"
            img.parent.mkdir(parents=True)
            mask.parent.mkdir(parents=True)
            skipped_mask.parent.mkdir(parents=True)
            skipped_human_mask.parent.mkdir(parents=True)
            Image.new("RGB", (4, 3), color=(10, 20, 30)).save(img)
            Image.new("RGB", (4, 3), color=(0, 0, 0)).save(mask)
            Image.new("RGB", (4, 3), color=(255, 0, 0)).save(skipped_mask)
            Image.new("RGB", (4, 3), color=(0, 255, 0)).save(skipped_human_mask)

            records = inventory_source_dataset(root / "set_1", dataset_name="set_1")

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["dataset"], "set_1")
            self.assertEqual(records[0]["split"], "train")
            self.assertIn("masks_colored", records[0]["mask_colored_path"])
            self.assertNotIn("masks_human", records[0]["mask_colored_path"])
            self.assertEqual(records[0]["mask_width"], 4)
            self.assertEqual(records[0]["mask_height"], 3)
            self.assertEqual(records[0]["magnification"], "50x")
            self.assertEqual(records[0]["unique_mask_colors_count"], 1)


if __name__ == "__main__":
    unittest.main()
