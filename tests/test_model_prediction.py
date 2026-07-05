import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.inference.model_prediction import (
    CheckpointMetadata,
    LoadedSegmentationModel,
    binary_mask_from_class_image,
    binary_prediction_from_logits,
    clip_class_image_to_binary,
    clip_class_index_to_binary,
    load_simple_unet_checkpoint,
    multiclass_prediction_from_logits,
    predict_segmentation_image,
    predict_multiclass_segmentation_image,
    read_checkpoint_metadata,
    save_segmentation_prediction,
)
from ore_detection.models.ct_unet import create_ct_unet
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

    def _loaded_constant_multiclass_model(self, *, class_index: int = 2):
        torch = self.torch

        class ConstantMulticlassModel(torch.nn.Module):
            def forward(self, inputs):
                logits = torch.zeros(
                    (int(inputs.shape[0]), 3, int(inputs.shape[2]), int(inputs.shape[3])),
                    dtype=inputs.dtype,
                    device=inputs.device,
                )
                logits[:, int(class_index)] = 8.0
                return logits

        metadata = CheckpointMetadata(
            path=Path("synthetic_ore_talc.pt"),
            task="multiclass",
            architecture="ct_unet",
            out_channels=3,
            model_kwargs={},
            class_names=("background", "ore", "talc"),
            background_index=0,
            image_size=8,
            epoch=1,
            notebook="synthetic",
            best_test_loss=None,
            train_metrics={},
            test_metrics={},
            normalization_mean=(0.0, 0.0, 0.0),
            normalization_std=(1.0, 1.0, 1.0),
        )
        return LoadedSegmentationModel(
            model=ConstantMulticlassModel().eval(),
            metadata=metadata,
            device=torch.device("cpu"),
        )

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
            self.assertEqual(binary.architecture, "simple_unet")
            self.assertEqual(binary.out_channels, 1)
            self.assertEqual(binary.normalization_mean, (0.1, 0.2, 0.3))
            self.assertEqual(ore.task, "multiclass")
            self.assertEqual(ore.class_names, ("background", "pyrite", "chalcopyrite"))
            self.assertEqual(ore.background_index, 0)

    def test_ct_unet_checkpoint_loads_from_architecture_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "ct_unet.pt"
            model_kwargs = {
                "base_channels": 4,
                "num_heads": 2,
                "transformer_layers": 0,
                "token_grid_size": 2,
            }
            model = create_ct_unet(out_channels=1, **model_kwargs)
            self.torch.save(
                {
                    "model": model.state_dict(),
                    "architecture": "ct_unet",
                    "model_kwargs": model_kwargs,
                    "image_size": 16,
                    "normalization": {"mean": (0.1, 0.2, 0.3), "std": (0.4, 0.5, 0.6)},
                },
                path,
            )

            loaded = load_simple_unet_checkpoint(path)

            self.assertEqual(loaded.metadata.architecture, "ct_unet")
            logits = loaded.model(self.torch.zeros((1, 3, 16, 16), dtype=self.torch.float32))
            self.assertEqual(tuple(logits.shape), (1, 1, 16, 16))

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

    def test_class_image_converts_non_background_to_binary_ore_mask(self):
        class_image = Image.new("L", (3, 1))
        class_image.putdata([0, 2, 1])

        binary = binary_mask_from_class_image(class_image, background_index=0)

        self.assertEqual(list(binary.tobytes()), [0, 255, 255])

    def test_predict_multiclass_segmentation_image_uses_ore_model_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ore_path = root / "ore.pt"
            self._save_checkpoint(
                ore_path,
                out_channels=3,
                class_names=("background", "pyrite", "chalcopyrite"),
                background_index=0,
            )
            ore_model = load_simple_unet_checkpoint(ore_path)
            image = Image.new("RGB", (8, 6), (100, 110, 120))

            prediction = predict_multiclass_segmentation_image(image, ore_model=ore_model)

            self.assertEqual(prediction.multiclass_mask.size, image.size)
            self.assertEqual(prediction.multiclass_confidence.size, image.size)
            self.assertEqual(prediction.metadata["method"], "trained_ore_multiclass_segmentation")
            self.assertTrue(prediction.metadata["policy"]["ore_mask_from_multiclass_non_background"])
            self.assertFalse(prediction.metadata["policy"]["talc_prediction_enabled"])

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

    def test_multiclass_ctunet_prediction_saves_talc_class_two_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "sample.png"
            Image.new("RGB", (8, 8), (100, 110, 120)).save(image_path)
            model = self._loaded_constant_multiclass_model(class_index=2)

            prediction = predict_segmentation_image(Image.open(image_path), binary_model=model)

            self.assertEqual(set(prediction.multiclass_mask.tobytes()), {2})
            self.assertEqual(set(prediction.talc_mask.tobytes()), {1})
            self.assertTrue(prediction.metadata["policy"]["talc_prediction_enabled"])

            artifacts = save_segmentation_prediction(
                image_path,
                binary_model=model,
                output_root=root / "predictions",
                sample_id="sample-talc",
            )

            self.assertTrue(artifacts.ore_mask_path.exists())
            self.assertTrue(artifacts.multiclass_mask_path.exists())
            self.assertTrue(artifacts.talc_mask_path.exists())
            self.assertTrue(artifacts.talc_probability_path.exists())
            with Image.open(artifacts.multiclass_mask_path) as saved_multiclass:
                self.assertEqual(set(saved_multiclass.tobytes()), {2})
            with Image.open(artifacts.talc_mask_path) as saved_talc:
                self.assertEqual(set(saved_talc.tobytes()), {1})
            metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(metadata["policy"]["talc_prediction_enabled"])
            self.assertEqual(metadata["policy"]["talc_policy"], "checkpoint_class")
            self.assertEqual(metadata["ore_checkpoint"]["class_names"], ["background", "ore", "talc"])


if __name__ == "__main__":
    unittest.main()
