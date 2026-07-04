import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.inference.prediction_store import save_hsv_dummy_prediction
from ore_detection.segmentation.hsv_dummy import HsvDummyConfig


class TestPredictionStore(unittest.TestCase):
    def test_save_hsv_dummy_prediction_writes_ui_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "sample.jpg"
            Image.new("RGB", (3, 1)).save(image_path)
            with Image.open(image_path) as image:
                image.putdata([(0, 0, 0), (80, 80, 80), (255, 255, 255)])
                image.save(image_path)

            artifacts = save_hsv_dummy_prediction(
                image_path,
                output_root=root / "predictions",
                config=HsvDummyConfig(value_threshold=50, foreground="bright"),
                sample_id="sample-1",
            )

            self.assertTrue(artifacts.sample_dir.exists())
            self.assertTrue(artifacts.ore_mask_path.exists())
            self.assertTrue(artifacts.ore_confidence_path.exists())
            self.assertTrue(artifacts.overlay_path.exists())
            self.assertTrue(artifacts.metadata_path.exists())
            with Image.open(artifacts.ore_mask_path) as mask:
                self.assertEqual(list(mask.tobytes()), [0, 1, 1])
            metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["method"], "hsv_value_dummy")
            self.assertEqual(metadata["sample_id"], "sample-1")
            self.assertEqual(metadata["config"]["value_threshold"], 50)


if __name__ == "__main__":
    unittest.main()
