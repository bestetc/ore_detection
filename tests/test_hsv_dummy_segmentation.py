import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.segmentation.hsv_dummy import HsvDummyConfig, hsv_value_binary_mask, hsv_value_confidence


class TestHsvDummySegmentation(unittest.TestCase):
    def test_bright_foreground_uses_hsv_value_threshold(self):
        image = Image.new("RGB", (4, 1))
        image.putdata([(0, 0, 0), (49, 49, 49), (50, 1, 1), (255, 255, 255)])

        mask = hsv_value_binary_mask(image, HsvDummyConfig(value_threshold=50, foreground="bright"))

        self.assertEqual(mask.mode, "L")
        self.assertEqual(list(mask.tobytes()), [0, 0, 1, 1])

    def test_dark_foreground_uses_hsv_value_threshold(self):
        image = Image.new("RGB", (3, 1))
        image.putdata([(0, 0, 0), (50, 1, 1), (255, 255, 255)])

        mask = hsv_value_binary_mask(image, HsvDummyConfig(value_threshold=50, foreground="dark"))

        self.assertEqual(list(mask.tobytes()), [1, 0, 0])

    def test_standard_scaling_runs_before_value_threshold(self):
        image = Image.new("RGB", (2, 1))
        image.putdata([(10, 10, 10), (20, 20, 20)])
        stats = {"mean": (10.0, 10.0, 10.0), "std": (10.0, 10.0, 10.0)}

        mask = hsv_value_binary_mask(
            image,
            HsvDummyConfig(value_threshold=150, foreground="bright", standardize=True, standardize_stats=stats, clip_sigma=1.0),
        )

        self.assertEqual(list(mask.tobytes()), [0, 1])

    def test_confidence_is_hsv_value_channel(self):
        image = Image.new("RGB", (2, 1))
        image.putdata([(10, 0, 0), (200, 1, 1)])

        confidence = hsv_value_confidence(image)

        self.assertEqual(list(confidence.tobytes()), [10, 200])


if __name__ == "__main__":
    unittest.main()
