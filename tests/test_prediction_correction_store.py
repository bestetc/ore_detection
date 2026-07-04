import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.inference.prediction_store import accept_prediction_as_correction, save_hsv_dummy_prediction
from ore_detection.segmentation.hsv_dummy import HsvDummyConfig


class TestPredictionCorrectionStore(unittest.TestCase):
    def test_accept_prediction_as_correction_copies_mask_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "sample.jpg"
            Image.new("RGB", (2, 1), color=(100, 100, 100)).save(image_path)
            artifacts = save_hsv_dummy_prediction(
                image_path,
                output_root=root / "predictions",
                config=HsvDummyConfig(value_threshold=50),
                sample_id="sample-1",
            )

            correction = accept_prediction_as_correction(artifacts.sample_dir, label="ore")

            self.assertTrue(correction.mask_path.exists())
            self.assertTrue(correction.metadata_path.exists())
            with Image.open(correction.mask_path) as mask:
                self.assertEqual(mask.size, (2, 1))
            metadata = json.loads(correction.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["label"], "ore")
            self.assertEqual(metadata["status"], "accepted_dummy_prediction")
            self.assertIn("created_at", metadata)


if __name__ == "__main__":
    unittest.main()
