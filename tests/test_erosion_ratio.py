import unittest

from PIL import Image

from ore_detection.descriptors.erosion_ratio import (
    ErosionRatioConfig,
    choose_erosion_ratio_threshold,
    classify_erosion_ratio_intergrowth,
    erosion_ratio_score_map,
)
from ore_detection.descriptors.intergrowth_classification import HARD_ORE_ID, NORMAL_ORE_ID


def compact_ore(size: int = 32) -> Image.Image:
    image = Image.new("L", (size, size), 0)
    pixels = image.load()
    for y in range(4, size - 4):
        for x in range(4, size - 4):
            pixels[x, y] = 1
    return image


def fragmented_ore(size: int = 32) -> Image.Image:
    image = Image.new("L", (size, size), 0)
    pixels = image.load()
    for y in range(2, size, 4):
        for x in range(2, size, 4):
            pixels[x, y] = 1
    return image


class TestErosionRatio(unittest.TestCase):
    def test_compact_mask_has_higher_erosion_ratio_than_fragmented_mask(self):
        config = ErosionRatioConfig(erosion_kernel_size=3, erosion_iterations=1, window_size=32, min_ore_fraction=0.0)

        compact_score, _ = erosion_ratio_score_map(compact_ore(), config=config)
        fragmented_score, _ = erosion_ratio_score_map(fragmented_ore(), config=config)

        self.assertGreater(compact_score.getpixel((16, 16)), fragmented_score.getpixel((16, 16)))
        self.assertEqual(fragmented_score.getpixel((16, 16)), 0)

    def test_low_ore_fraction_window_score_is_zero(self):
        image = Image.new("L", (32, 32), 0)
        image.putpixel((16, 16), 1)
        config = ErosionRatioConfig(erosion_kernel_size=3, erosion_iterations=0, window_size=32, min_ore_fraction=0.05)

        score, summaries = erosion_ratio_score_map(image, config=config)

        self.assertEqual(score.getpixel((16, 16)), 0)
        self.assertEqual(summaries[0]["ratio"], 0.0)

    def test_threshold_classifies_high_ratio_as_normal_and_low_ratio_as_hard(self):
        config = ErosionRatioConfig(
            erosion_kernel_size=3,
            erosion_iterations=1,
            window_size=32,
            min_ore_fraction=0.0,
            normal_threshold=0.5,
        )

        compact = classify_erosion_ratio_intergrowth(compact_ore(), config=config)
        fragmented = classify_erosion_ratio_intergrowth(fragmented_ore(), config=config)

        self.assertIn(NORMAL_ORE_ID, set(compact.intergrowth_mask.tobytes()))
        self.assertNotIn(HARD_ORE_ID, set(compact.intergrowth_mask.tobytes()))
        self.assertIn(HARD_ORE_ID, set(fragmented.intergrowth_mask.tobytes()))
        self.assertNotIn(NORMAL_ORE_ID, set(fragmented.intergrowth_mask.tobytes()))

    def test_choose_threshold_uses_normal_high_hard_low_policy(self):
        result = choose_erosion_ratio_threshold([(0.8, "Normal ore"), (0.2, "Hard ore")])

        self.assertGreater(result["threshold"], 0.2)
        self.assertLess(result["threshold"], 0.8)
        self.assertEqual(result["balanced_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
