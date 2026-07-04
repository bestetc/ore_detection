import unittest
import tempfile
from pathlib import Path

from PIL import Image

from ore_detection.talc.hsv_candidates import (
    calculate_rgb_mean_std,
    erode_binary_mask,
    iter_baseline_crop_paths,
    overlay_mask,
    rgb_sum_threshold_mask,
    standardize_image_to_uint8,
)


class TestTalcHsvCandidates(unittest.TestCase):
    def test_calculate_rgb_mean_std_uses_all_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.png"
            second = root / "second.png"
            Image.new("RGB", (1, 1), color=(0, 10, 20)).save(first)
            Image.new("RGB", (1, 1), color=(10, 20, 30)).save(second)

            stats = calculate_rgb_mean_std([first, second])

            self.assertEqual(stats["image_count"], 2)
            self.assertEqual(stats["pixel_count"], 2)
            self.assertEqual(stats["mean"], (5.0, 15.0, 25.0))
            self.assertEqual(stats["std"], (5.0, 5.0, 5.0))

    def test_standardize_image_to_uint8_maps_mean_to_mid_gray(self):
        image = Image.new("RGB", (1, 1), color=(10, 20, 30))
        stats = {"mean": (10.0, 20.0, 30.0), "std": (5.0, 5.0, 5.0)}

        scaled = standardize_image_to_uint8(image, stats, clip_sigma=1.0)

        self.assertEqual(scaled.getpixel((0, 0)), (128, 128, 128))

    def test_standardize_image_to_uint8_can_run_without_sigma_clip(self):
        image = Image.new("RGB", (3, 1))
        image.putdata([(0, 0, 0), (10, 10, 10), (20, 20, 20)])
        stats = {"mean": (10.0, 10.0, 10.0), "std": (5.0, 5.0, 5.0)}

        scaled = standardize_image_to_uint8(image, stats, clip_sigma=None, output_std=10.0)

        self.assertEqual(list(scaled.getdata()), [(108, 108, 108), (128, 128, 128), (148, 148, 148)])

    def test_rgb_sum_threshold_mask_marks_pixels_strictly_below_threshold(self):
        image = Image.new("RGB", (4, 1))
        image.putdata([(0, 0, 0), (49, 50, 50), (50, 50, 50), (255, 255, 255)])

        mask = rgb_sum_threshold_mask(image, threshold=150)

        self.assertEqual(list(mask.tobytes()), [255, 255, 0, 0])

    def test_erode_binary_mask_shrinks_foreground(self):
        mask = Image.new("L", (5, 5), color=0)
        for y in range(1, 4):
            for x in range(1, 4):
                mask.putpixel((x, y), 255)

        eroded = erode_binary_mask(mask, kernel_size=3, iterations=1)

        self.assertEqual(sum(1 for value in eroded.tobytes() if value > 0), 1)
        self.assertEqual(eroded.getpixel((2, 2)), 255)

    def test_erode_binary_mask_requires_odd_kernel(self):
        mask = Image.new("L", (3, 3), color=255)

        with self.assertRaises(ValueError):
            erode_binary_mask(mask, kernel_size=2, iterations=1)

    def test_iter_baseline_crop_paths_skips_panoramas_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panorama = root / "panoramas" / "1.jpg"
            crop = root / "Part 1" / "Hard ore" / "crop.jpg"
            panorama.parent.mkdir(parents=True)
            crop.parent.mkdir(parents=True)
            Image.new("RGB", (1, 1)).save(panorama)
            Image.new("RGB", (1, 1)).save(crop)

            paths = iter_baseline_crop_paths(root)

            self.assertEqual(paths, [crop])

    def test_overlay_preserves_size_and_mode(self):
        image = Image.new("RGB", (2, 2), color=(100, 100, 100))
        mask = Image.new("L", (2, 2), color=255)

        overlay = overlay_mask(image, mask)

        self.assertEqual(overlay.mode, "RGB")
        self.assertEqual(overlay.size, image.size)


if __name__ == "__main__":
    unittest.main()
