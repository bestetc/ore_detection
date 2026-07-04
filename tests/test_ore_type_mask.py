import unittest

from PIL import Image

from ore_detection.data.ore_type_legend import load_legend_config
from ore_detection.data.ore_type_mask import (
    color_mask_to_class_image,
    color_mask_to_one_hot_tensor,
    one_hot_to_binary_ore,
    unknown_colors,
    validate_one_hot_tensor,
)


class TestOreTypeMask(unittest.TestCase):
    def setUp(self):
        self.legend = load_legend_config()

    def test_color_mask_to_class_image_uses_legend_targets(self):
        mask = Image.new("RGB", (3, 1))
        mask.putdata([(0, 0, 0), (255, 165, 0), (0, 191, 255)])

        class_image = color_mask_to_class_image(mask, dataset="set_1", legend=self.legend)

        self.assertEqual(
            list(class_image.tobytes()),
            [
                self.legend.class_index("background"),
                self.legend.class_index("chalcopyrite"),
                self.legend.class_index("bornite"),
            ],
        )

    def test_unknown_colors_are_reported_and_rejected(self):
        mask = Image.new("RGB", (2, 1))
        mask.putdata([(0, 0, 0), (1, 2, 3)])

        self.assertEqual(unknown_colors(mask, dataset="set_1", legend=self.legend), {(1, 2, 3): 1})
        with self.assertRaisesRegex(ValueError, "missing from source ore-type legend"):
            color_mask_to_class_image(mask, dataset="set_1", legend=self.legend)

    def test_one_hot_tensor_and_binary_ore_exclude_background(self):
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")

        mask = Image.new("RGB", (3, 1))
        mask.putdata([(0, 0, 0), (255, 165, 0), (0, 191, 255)])

        one_hot = color_mask_to_one_hot_tensor(mask, dataset="set_1", legend=self.legend)
        validate_one_hot_tensor(one_hot, legend=self.legend)
        binary = one_hot_to_binary_ore(one_hot, legend=self.legend)

        self.assertEqual(tuple(one_hot.shape), (self.legend.class_count, 1, 3))
        self.assertEqual(binary.squeeze(0).tolist(), [[0, 1, 1]])


if __name__ == "__main__":
    unittest.main()
