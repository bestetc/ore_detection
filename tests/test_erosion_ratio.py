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


def split_touching_mineral_block() -> tuple[Image.Image, Image.Image]:
    ore = Image.new("L", (8, 8), 0)
    mineral = Image.new("L", (8, 8), 0)
    ore_pixels = ore.load()
    pixels = mineral.load()
    for y in range(1, 7):
        for x in range(1, 7):
            ore_pixels[x, y] = 1
            pixels[x, y] = 1 if x < 4 else 2
    return ore, mineral


class TestErosionRatio(unittest.TestCase):
    def test_compact_mask_has_higher_erosion_ratio_than_fragmented_mask(self):
        config = ErosionRatioConfig(erosion_kernel_size=3, erosion_iterations=1, window_size=32, min_ore_fraction=0.0)

        compact_score, _ = erosion_ratio_score_map(compact_ore(), config=config)
        fragmented_score, _ = erosion_ratio_score_map(fragmented_ore(), config=config)

        self.assertGreater(compact_score.getpixel((16, 16)), fragmented_score.getpixel((16, 16)))
        self.assertEqual(fragmented_score.getpixel((16, 16)), 0)

    def test_multiclass_erosion_erodes_each_touching_mineral_class_separately(self):
        ore, mineral = split_touching_mineral_block()
        config = ErosionRatioConfig(erosion_kernel_size=3, erosion_iterations=1, window_size=8, min_ore_fraction=0.0)

        binary_score, binary_summaries = erosion_ratio_score_map(ore, config=config)
        class_score, class_summaries = erosion_ratio_score_map(ore, multiclass_mask=mineral, config=config)

        self.assertLess(class_score.getpixel((3, 3)), binary_score.getpixel((3, 3)))
        self.assertEqual(binary_summaries[0]["eroded_ore_area"], 16)
        self.assertEqual(class_summaries[0]["eroded_ore_area"], 8)
        self.assertEqual(class_summaries[0]["class_count"], 2)
        self.assertTrue(class_summaries[0]["class_aware_erosion"])

    def test_multiclass_erosion_can_turn_close_mixed_block_to_hard(self):
        ore, mineral = split_touching_mineral_block()
        config = ErosionRatioConfig(
            erosion_kernel_size=3,
            erosion_iterations=1,
            window_size=8,
            min_ore_fraction=0.0,
            normal_threshold=0.30,
        )

        binary = classify_erosion_ratio_intergrowth(ore, config=config)
        class_aware = classify_erosion_ratio_intergrowth(ore, multiclass_mask=mineral, config=config)

        self.assertIn(NORMAL_ORE_ID, set(binary.intergrowth_mask.tobytes()))
        self.assertIn(HARD_ORE_ID, set(class_aware.intergrowth_mask.tobytes()))
        self.assertNotIn(NORMAL_ORE_ID, set(class_aware.intergrowth_mask.tobytes()))

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
