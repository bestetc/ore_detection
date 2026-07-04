import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.data.color_mask import color_mask_to_binary, convert_color_mask_file, unique_rgb_colors


class TestColorMask(unittest.TestCase):
    def test_color_mask_to_binary_maps_black_to_background_and_colors_to_ore(self):
        mask = Image.new("RGB", (3, 1))
        mask.putdata([(0, 0, 0), (255, 0, 0), (12, 34, 56)])

        binary = color_mask_to_binary(mask)

        self.assertEqual(binary.mode, "L")
        self.assertEqual(list(binary.tobytes()), [0, 1, 1])

    def test_color_mask_to_binary_treats_alpha_zero_as_background(self):
        mask = Image.new("RGBA", (2, 1))
        mask.putdata([(255, 0, 0, 0), (255, 0, 0, 255)])

        binary = color_mask_to_binary(mask)

        self.assertEqual(list(binary.tobytes()), [0, 1])

    def test_convert_color_mask_file_writes_only_zero_and_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "mask.png"
            target = root / "out" / "mask.png"
            mask = Image.new("RGB", (2, 2))
            mask.putdata([(0, 0, 0), (1, 2, 3), (0, 0, 0), (10, 20, 30)])
            mask.save(source)

            stats = convert_color_mask_file(source, target)

            self.assertEqual(stats["ore_pixels"], 2)
            self.assertEqual(stats["background_pixels"], 2)
            self.assertTrue(target.exists())
            with Image.open(target) as saved:
                self.assertEqual(set(saved.tobytes()), {0, 1})

    def test_unique_rgb_colors_counts_colors(self):
        mask = Image.new("RGB", (3, 1))
        mask.putdata([(0, 0, 0), (1, 2, 3), (1, 2, 3)])

        colors = unique_rgb_colors(mask)

        self.assertEqual(colors, {(0, 0, 0): 1, (1, 2, 3): 2})


if __name__ == "__main__":
    unittest.main()
