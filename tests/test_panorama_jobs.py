import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.backend.panorama_jobs import PanoramaJobManager, PanoramaPredictionRequest
from ore_detection.models.simple_unet import create_simple_unet


class TestPanoramaJobs(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")
        self.torch = torch

    def test_panorama_job_progress_reaches_completed_with_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "panorama.png"
            checkpoint_path = root / "binary.pt"
            Image.new("RGB", (16, 16), (100, 110, 120)).save(image_path)
            model = create_simple_unet(out_channels=1)
            self.torch.save(
                {
                    "model": model.state_dict(),
                    "image_size": 16,
                    "normalization": {"mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
                },
                checkpoint_path,
            )
            manager = PanoramaJobManager()

            status = manager.start(
                request=PanoramaPredictionRequest(
                    image_path=image_path,
                    binary_model_path=checkpoint_path,
                    device="cpu",
                    tile_size=16,
                    overlap=0,
                    batch_size=1,
                ),
                jobs_root=root / "jobs",
                predictions_root=root / "predictions",
            )
            job_id = status["job_id"]
            deadline = time.time() + 10
            while time.time() < deadline:
                status = manager.status(job_id, jobs_root=root / "jobs")
                if status["status"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.05)

            self.assertEqual(status["status"], "completed", status.get("error"))
            self.assertEqual(status["total_tiles"], 1)
            self.assertEqual(status["processed_tiles"], 1)
            self.assertIn("ore_mask", status["artifacts"])
            self.assertTrue((root / "jobs" / job_id / "progress.json").exists())

    def test_panorama_job_can_run_ore_model_without_binary_chaining(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "panorama.png"
            checkpoint_path = root / "ore.pt"
            Image.new("RGB", (16, 16), (100, 110, 120)).save(image_path)
            model = create_simple_unet(out_channels=3)
            self.torch.save(
                {
                    "model": model.state_dict(),
                    "image_size": 16,
                    "class_names": ("background", "pyrite", "chalcopyrite"),
                    "background_index": 0,
                    "normalization": {"mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
                },
                checkpoint_path,
            )
            manager = PanoramaJobManager()

            status = manager.start(
                request=PanoramaPredictionRequest(
                    image_path=image_path,
                    binary_model_path=root / "unused-binary.pt",
                    ore_model_path=checkpoint_path,
                    model_kind="ore",
                    device="cpu",
                    tile_size=16,
                    overlap=0,
                    batch_size=1,
                ),
                jobs_root=root / "jobs",
                predictions_root=root / "predictions",
            )
            job_id = status["job_id"]
            deadline = time.time() + 10
            while time.time() < deadline:
                status = manager.status(job_id, jobs_root=root / "jobs")
                if status["status"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.05)

            self.assertEqual(status["status"], "completed", status.get("error"))
            self.assertEqual(status["model_kind"], "ore")
            self.assertIsNone(status["artifacts"]["ore_mask"])
            self.assertIn("ore_multiclass_mask", status["artifacts"])
            metadata = self.torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(tuple(metadata["class_names"]), ("background", "pyrite", "chalcopyrite"))


if __name__ == "__main__":
    unittest.main()
