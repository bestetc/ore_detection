import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.descriptors.intergrowth_classification import (
    HARD_ORE_ID,
    IGNORE_ID,
    NORMAL_ORE_ID,
    TALC_ID,
    IntergrowthClassifierConfig,
    choose_hard_threshold,
    classify_intergrowth_mask,
    local_windows,
    save_intergrowth_artifacts,
)


def scattered_mask(size: int = 16) -> Image.Image:
    image = Image.new("L", (size, size), 0)
    pixels = image.load()
    for y in range(1, size, 3):
        for x in range(1, size, 3):
            pixels[x, y] = 1
    return image


def compact_mask(size: int = 16) -> Image.Image:
    image = Image.new("L", (size, size), 0)
    pixels = image.load()
    for y in range(3, 13):
        for x in range(3, 13):
            pixels[x, y] = 1
    return image


class TestIntergrowthClassification(unittest.TestCase):
    def test_default_config_uses_base_local_window_params(self):
        config = IntergrowthClassifierConfig()

        self.assertEqual(config.window_size, 128)
        self.assertEqual(config.stride, 64)

    def test_local_windows_stable_regions_cover_image(self):
        windows = local_windows(11, 7, window_size=5, stride=3)
        covered = set()
        for window in windows:
            left, top, right, bottom = window.stable_box
            for y in range(top, bottom):
                for x in range(left, right):
                    covered.add((x, y))

        self.assertEqual(len(covered), 11 * 7)

    def test_local_windows_cover_image_when_stride_exceeds_window(self):
        windows = local_windows(32, 24, window_size=8, stride=16)
        covered = set()
        for window in windows:
            left, top, right, bottom = window.stable_box
            for y in range(top, bottom):
                for x in range(left, right):
                    covered.add((x, y))

        self.assertEqual(len(covered), 32 * 24)

    def test_fragmented_mask_classifies_as_hard_and_compact_as_normal(self):
        config = IntergrowthClassifierConfig(window_size=16, stride=16, hard_threshold=0.5, min_ore_fraction=0)

        hard = classify_intergrowth_mask(scattered_mask(), config=config)
        normal = classify_intergrowth_mask(compact_mask(), config=config)

        self.assertIn(HARD_ORE_ID, set(hard.intergrowth_mask.tobytes()))
        self.assertNotIn(NORMAL_ORE_ID, set(hard.intergrowth_mask.tobytes()))
        self.assertIn(NORMAL_ORE_ID, set(normal.intergrowth_mask.tobytes()))
        self.assertNotIn(HARD_ORE_ID, set(normal.intergrowth_mask.tobytes()))
        self.assertEqual(hard.metrics["image_label"], "hard_ore")
        self.assertEqual(normal.metrics["image_label"], "normal_ore")

    def test_label_precedence_preserves_talc_and_ignore_over_ore(self):
        ore = Image.new("L", (4, 4), 1)
        talc = Image.new("L", (4, 4), 0)
        ignore = Image.new("L", (4, 4), 0)
        talc.putpixel((1, 1), 1)
        ignore.putpixel((2, 2), 1)
        config = IntergrowthClassifierConfig(window_size=4, stride=4, hard_threshold=1.0, min_ore_fraction=0)

        result = classify_intergrowth_mask(ore, talc_mask=talc, ignore_mask=ignore, config=config)

        self.assertEqual(result.intergrowth_mask.getpixel((1, 1)), TALC_ID)
        self.assertEqual(result.intergrowth_mask.getpixel((2, 2)), IGNORE_ID)
        self.assertEqual(result.intergrowth_mask.getpixel((0, 0)), NORMAL_ORE_ID)

    def test_save_intergrowth_artifacts_updates_prediction_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "sample"
            sample_dir.mkdir()
            mask_path = sample_dir / "ore_mask.png"
            compact_mask().save(mask_path)
            metadata_path = sample_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample",
                        "image_path": str(root / "raw.png"),
                        "image_width": 16,
                        "image_height": 16,
                        "artifacts": {
                            "ore_mask": str(mask_path),
                            "review_mask": str(mask_path),
                        },
                    }
                ),
                encoding="utf-8",
            )

            metrics = save_intergrowth_artifacts(
                sample_dir,
                config=IntergrowthClassifierConfig(window_size=16, stride=16, hard_threshold=0.5, min_ore_fraction=0),
            )
            updated = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertTrue(Path(metrics["artifacts"]["intergrowth_mask"]).exists())
            self.assertTrue(Path(metrics["artifacts"]["intergrowth_score"]).exists())
            self.assertEqual(metrics["area_metrics"]["image_label"], "normal_ore")
            self.assertIn("intergrowth_mask", updated["artifacts"])
            self.assertIn("intergrowth", updated)

    def test_choose_hard_threshold_uses_weak_labels(self):
        result = choose_hard_threshold([(0.1, "Normal ore"), (0.2, "Normal ore"), (0.8, "Hard ore")])

        self.assertGreaterEqual(result["balanced_accuracy"], 1.0)
        self.assertGreater(result["threshold"], 0.2)
        self.assertLessEqual(result["threshold"], 0.8)


if __name__ == "__main__":
    unittest.main()
