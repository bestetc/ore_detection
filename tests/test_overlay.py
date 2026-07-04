import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.visualization.overlay import colorize_mask, overlay_mask_on_image, save_overlay


class TestOverlay(unittest.TestCase):
    def test_colorize_mask_uses_palette_and_transparent_background(self):
        mask = Image.new("L", (3, 1))
        mask.putdata([0, 1, 2])

        colorized = colorize_mask(mask, alpha=128)

        self.assertEqual(colorized.mode, "RGBA")
        self.assertEqual(colorized.getpixel((0, 0))[3], 0)
        self.assertEqual(colorized.getpixel((1, 0)), (0, 255, 0, 128))
        self.assertEqual(colorized.getpixel((2, 0)), (255, 0, 0, 128))

    def test_overlay_mask_on_image_preserves_size_and_changes_foreground_pixel(self):
        image = Image.new("RGB", (2, 1), color=(100, 100, 100))
        mask = Image.new("L", (2, 1))
        mask.putdata([0, 1])

        overlay = overlay_mask_on_image(image, mask, alpha=128)

        self.assertEqual(overlay.size, image.size)
        self.assertEqual(overlay.getpixel((0, 0)), (100, 100, 100, 255))
        self.assertNotEqual(overlay.getpixel((1, 0)), (100, 100, 100, 255))

    def test_save_overlay_writes_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "overlay.png"
            image = Image.new("RGB", (2, 1), color=(100, 100, 100))
            mask = Image.new("L", (2, 1), color=1)

            save_overlay(image, mask, output)

            self.assertTrue(output.exists())
            with Image.open(output) as saved:
                self.assertEqual(saved.size, (2, 1))


if __name__ == "__main__":
    unittest.main()
