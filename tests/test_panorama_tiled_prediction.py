import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.inference.model_prediction import CheckpointMetadata, LoadedSegmentationModel
from ore_detection.inference.tiled_prediction import make_tile_grid, save_tiled_segmentation_prediction


class TestPanoramaTiledPrediction(unittest.TestCase):
    def test_tile_grid_stable_regions_cover_every_pixel_once(self):
        width, height = 30, 18
        tiles = make_tile_grid(width, height, tile_size=16, overlap=4)
        coverage = Image.new("L", (width, height), 0)
        for tile in tiles:
            crop = tile.stable_crop_box
            x, y = tile.stable_paste_xy
            coverage.paste(1, (x, y, x + (crop[2] - crop[0]), y + (crop[3] - crop[1])))

        self.assertEqual(set(coverage.tobytes()), {1})

    def test_tiled_prediction_writes_full_resolution_masks(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")

        class ConstantBinaryModel(torch.nn.Module):
            def forward(self, x):
                return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), 10.0, device=x.device)

        metadata = CheckpointMetadata(
            path=Path("binary.pt"),
            task="binary",
            out_channels=1,
            class_names=(),
            background_index=None,
            image_size=16,
            epoch=1,
            notebook=None,
            best_test_loss=None,
            train_metrics={},
            test_metrics={},
            normalization_mean=(0.0, 0.0, 0.0),
            normalization_std=(1.0, 1.0, 1.0),
        )
        loaded = LoadedSegmentationModel(model=ConstantBinaryModel(), metadata=metadata, device=torch.device("cpu"))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "source.png"
            Image.new("RGB", (30, 18), (100, 110, 120)).save(image_path)

            artifacts = save_tiled_segmentation_prediction(
                image_path,
                binary_model=loaded,
                output_root=root / "predictions",
                sample_id="job-1",
                tile_size=16,
                overlap=4,
                batch_size=2,
                preview_max_size=64,
            )

            with Image.open(artifacts.ore_mask_path) as saved_mask:
                self.assertEqual(saved_mask.size, (30, 18))
                self.assertEqual(set(saved_mask.tobytes()), {1})
            self.assertTrue(artifacts.raw_preview_path.exists())
            self.assertTrue(artifacts.overlay_preview_path.exists())


if __name__ == "__main__":
    unittest.main()
