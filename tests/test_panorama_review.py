import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.backend.panorama_review import (
    append_brush_patch,
    class_area_metrics,
    render_panorama_tile,
    restore_base_prediction,
    save_panorama_review,
)


def write_sample_metadata(sample_dir: Path, image_path: Path, mask_path: Path) -> None:
    metadata = {
        "sample_id": "job-1",
        "image_path": str(image_path),
        "image_width": 8,
        "image_height": 8,
        "total_tiles": 1,
        "ore_checkpoint": None,
        "artifacts": {
            "ore_mask": str(mask_path),
            "ore_confidence": str(mask_path),
            "ore_probability": str(mask_path),
            "review_mask": str(mask_path),
        },
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def add_intergrowth_artifacts(sample_dir: Path) -> Path:
    path = sample_dir / "intergrowth_mask.png"
    score = sample_dir / "intergrowth_score.png"
    confidence = sample_dir / "intergrowth_confidence.png"
    mask = Image.new("L", (8, 8), 0)
    mask.putpixel((2, 2), 4)
    mask.putpixel((3, 3), 5)
    mask.save(path)
    Image.new("L", (8, 8), 128).save(score)
    Image.new("L", (8, 8), 200).save(confidence)
    metadata_path = sample_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["artifacts"].update(
        {
            "intergrowth_mask": str(path),
            "intergrowth_score": str(score),
            "intergrowth_confidence": str(confidence),
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


class TestPanoramaReview(unittest.TestCase):
    def test_tile_rendering_applies_patch_log_without_full_browser_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "sample"
            sample_dir.mkdir()
            image_path = root / "raw.png"
            mask_path = sample_dir / "ore_mask.png"
            Image.new("RGB", (8, 8), (10, 20, 30)).save(image_path)
            Image.new("L", (8, 8), 0).save(mask_path)
            write_sample_metadata(sample_dir, image_path, mask_path)

            append_brush_patch(sample_dir, x=4, y=4, radius=2, class_id=3)
            tile = render_panorama_tile(sample_dir, layer="class_index", x=0, y=0, width=8, height=8)

            self.assertEqual(tile.getpixel((4, 4)), 3)
            self.assertEqual(tile.size, (8, 8))

            metrics = class_area_metrics(sample_dir)
            self.assertEqual(metrics["classes"][0]["id"], 3)
            self.assertGreater(metrics["classes"][0]["fraction"], 0)

            restore_base_prediction(sample_dir)
            restored = render_panorama_tile(sample_dir, layer="class_index", x=0, y=0, width=8, height=8)
            self.assertEqual(set(restored.tobytes()), {0})

    def test_intergrowth_layers_render_and_report_metrics_without_edit_patches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "sample"
            sample_dir.mkdir()
            image_path = root / "raw.png"
            mask_path = sample_dir / "ore_mask.png"
            Image.new("RGB", (8, 8), (10, 20, 30)).save(image_path)
            Image.new("L", (8, 8), 0).save(mask_path)
            write_sample_metadata(sample_dir, image_path, mask_path)
            add_intergrowth_artifacts(sample_dir)
            append_brush_patch(sample_dir, x=4, y=4, radius=2, class_id=3)

            class_tile = render_panorama_tile(sample_dir, layer="intergrowth_class_index", x=0, y=0, width=8, height=8)
            score_tile = render_panorama_tile(sample_dir, layer="intergrowth_score", x=0, y=0, width=8, height=8)
            metrics = class_area_metrics(sample_dir, layer="intergrowth")

            self.assertEqual(class_tile.getpixel((2, 2)), 4)
            self.assertEqual(class_tile.getpixel((3, 3)), 5)
            self.assertEqual(score_tile.getpixel((0, 0)), 128)
            self.assertEqual({row["id"] for row in metrics["classes"]}, {4, 5})

    def test_save_panorama_review_writes_patch_tensors_not_full_one_hot(self):
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "sample"
            active_root = root / "active"
            sample_dir.mkdir()
            image_path = root / "raw.png"
            mask_path = sample_dir / "ore_mask.png"
            Image.new("RGB", (8, 8), (10, 20, 30)).save(image_path)
            Image.new("L", (8, 8), 0).save(mask_path)
            write_sample_metadata(sample_dir, image_path, mask_path)
            append_brush_patch(sample_dir, x=4, y=4, radius=2, class_id=3)

            metadata = save_panorama_review(sample_dir, output_root=active_root, crop_size=8)

            self.assertTrue(Path(metadata["class_index_mask"]).exists())
            self.assertFalse(metadata["full_one_hot_tensor_saved"])
            self.assertEqual(metadata["patch_count"], 1)
            self.assertEqual(len(metadata["tensor_tiles"]), 1)
            self.assertTrue(Path(metadata["patch_log"]).exists())


if __name__ == "__main__":
    unittest.main()
