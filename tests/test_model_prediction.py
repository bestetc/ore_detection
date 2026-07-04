import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.inference.model_prediction import (
    binary_prediction_from_logits,
    clip_class_image_to_binary,
    clip_class_index_to_binary,
    load_simple_unet_checkpoint,
    multiclass_prediction_from_logits,
    read_checkpoint_metadata,
    save_segmentation_prediction,
)
from ore_detection.models.simple_unet import create_simple_unet


class TestModelPrediction(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")
        self.torch = torch

    def _save_checkpoint(self, path: Path, *, out_channels: int, **metadata):
        model = create_simple_unet(out_channels=out_channels)
        checkpoint = {
            "model": model.state_dict(),
            "image_size": 16,
            "normalization": {"mean": (0.1, 0.2, 0.3), "std": (0.4, 0.5, 0.6)},
            "epoch": 7,
            "test_metrics": {"loss": 1.25},
        }
        checkpoint.update(metadata)
        self.torch.save(checkpoint, path)

    def test_checkpoint_metadata_loads_binary_and_ore_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary_path = root / "binary.pt"
            ore_path = root / "ore.pt"
            self._save_checkpoint(binary_path, out_channels=1)
            self._save_checkpoint(
                ore_path,
                out_channels=3,
                class_names=("background", "pyrite", "chalcopyrite"),
                background_index=0,
            )

            binary = read_checkpoint_metadata(binary_path)
            ore = read_checkpoint_metadata(ore_path)

            self.assertEqual(binary.task, "binary")
            self.assertEqual(binary.out_channels, 1)
            self.assertEqual(binary.normalization_mean, (0.1, 0.2, 0.3))
            self.assertEqual(ore.task, "multiclass")
            self.assertEqual(ore.class_names, ("background", "pyrite", "chalcopyrite"))
            self.assertEqual(ore.background_index, 0)

    def test_binary_logits_to_mask_probability_and_confidence(self):
        logits = self.torch.tensor([[[[-20.0, 20.0], [0.0, 2.0]]]])

        prediction = binary_prediction_from_logits(logits, threshold=0.5)

        self.assertEqual(prediction.mask.tolist(), [[[[0, 1], [1, 1]]]])
        self.assertAlmostEqual(float(prediction.probability[0, 0, 1, 0]), 0.5)
        self.assertAlmostEqual(float(prediction.confidence[0, 0, 1, 0]), 0.5)

    def test_multiclass_logits_to_class_mask_and_probability(self):
        logits = self.torch.tensor(
            [
                [
                    [[4.0, 0.0], [0.0, 0.0]],
                    [[0.0, 5.0], [0.0, 2.0]],
                    [[0.0, 0.0], [6.0, 0.0]],
                ]
            ]
        )

        prediction = multiclass_prediction_from_logits(
            logits,
            class_names=("background", "pyrite", "galena"),
            background_index=0,
        )

        self.assertEqual(prediction.class_index.tolist(), [[[0, 1], [2, 1]]])
        self.assertGreater(float(prediction.class_probability[0, 0, 1]), 0.9)

    def test_class_index_clips_to_binary_ore_mask(self):
        class_index = self.torch.tensor([[[1, 2], [2, 1]]])
        binary = self.torch.tensor([[[1, 0], [1, 0]]])

        clipped = clip_class_index_to_binary(class_index, binary, background_index=0)

        self.assertEqual(clipped.tolist(), [[[1, 0], [2, 0]]])

    def test_class_image_clips_to_binary_ore_mask(self):
        class_image = Image.new("L", (2, 2))
        class_image.putdata([1, 2, 2, 1])
        binary_mask = Image.new("L", (2, 2))
        binary_mask.putdata([1, 0, 1, 0])

        clipped = clip_class_image_to_binary(class_image, binary_mask, background_index=0)

        self.assertEqual(list(clipped.tobytes()), [1, 0, 2, 0])

    def test_save_segmentation_prediction_writes_no_talc_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "sample.png"
            Image.new("RGB", (8, 8), (100, 110, 120)).save(image_path)
            binary_path = root / "binary.pt"
            ore_path = root / "ore.pt"
            self._save_checkpoint(binary_path, out_channels=1)
            self._save_checkpoint(
                ore_path,
                out_channels=3,
                class_names=("background", "pyrite", "chalcopyrite"),
                background_index=0,
            )
            binary_model = load_simple_unet_checkpoint(binary_path)
            ore_model = load_simple_unet_checkpoint(ore_path)

            artifacts = save_segmentation_prediction(
                image_path,
                binary_model=binary_model,
                ore_model=ore_model,
                output_root=root / "predictions",
                sample_id="sample-1",
            )

            self.assertTrue(artifacts.ore_mask_path.exists())
            self.assertTrue(artifacts.ore_probability_path.exists())
            self.assertTrue(artifacts.ore_confidence_path.exists())
            self.assertTrue(artifacts.multiclass_mask_path.exists())
            metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
            self.assertFalse(metadata["policy"]["talc_prediction_enabled"])
            self.assertEqual(metadata["policy"]["talc_policy"], "manual_ui_annotation_only")


if __name__ == "__main__":
    unittest.main()
