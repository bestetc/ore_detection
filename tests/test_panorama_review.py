import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from ore_detection.backend.panorama_review import (
    append_brush_patch,
    apply_talc_threshold_to_panorama,
    class_area_metrics,
    render_panorama_tile,
    restore_base_prediction,
    save_erosion_ratio_intergrowth_artifacts,
    save_panorama_review,
    talc_histograms_for_panorama,
)
from ore_detection.descriptors.erosion_ratio import ErosionRatioConfig


def write_sample_metadata(sample_dir: Path, image_path: Path, mask_path: Path) -> None:
    with Image.open(image_path) as image:
        width, height = image.size
    metadata = {
        "sample_id": "job-1",
        "image_path": str(image_path),
        "image_width": width,
        "image_height": height,
        "total_tiles": 1,
        "ore_checkpoint": None,
        "artifacts": {
            "ore_mask": str(mask_path),
            "ore_confidence": str(mask_path),
            "ore_probability": str(mask_path),
            "review_mask": str(mask_path),
            "base_prediction_mask": str(mask_path),
        },
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def add_intergrowth_artifacts(sample_dir: Path) -> Path:
    path = sample_dir / "intergrowth_mask.png"
    score = sample_dir / "intergrowth_score.png"
    hard_score = sample_dir / "intergrowth_hard_score.png"
    confidence = sample_dir / "intergrowth_confidence.png"
    mask = Image.new("L", (8, 8), 0)
    mask.putpixel((2, 2), 4)
    mask.putpixel((3, 3), 5)
    mask.save(path)
    Image.new("L", (8, 8), 128).save(score)
    Image.new("L", (8, 8), 127).save(hard_score)
    Image.new("L", (8, 8), 200).save(confidence)
    metadata_path = sample_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["artifacts"].update(
        {
            "intergrowth_mask": str(path),
            "intergrowth_score": str(score),
            "intergrowth_hard_score": str(hard_score),
            "intergrowth_confidence": str(confidence),
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


class TestPanoramaReview(unittest.TestCase):
    def test_class_area_metrics_visible_crop_uses_crop_mask_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "sample"
            sample_dir.mkdir()
            image_path = root / "raw.png"
            mask_path = sample_dir / "ore_mask.png"
            Image.new("RGB", (4, 4), (10, 20, 30)).save(image_path)
            mask = Image.new("L", (4, 4), 4)
            for y in range(2):
                for x in range(2):
                    mask.putpixel((x, y), 3)
            mask.save(mask_path)
            write_sample_metadata(sample_dir, image_path, mask_path)

            full = class_area_metrics(sample_dir)
            visible = class_area_metrics(sample_dir, x=0, y=0, width=2, height=2)

            self.assertEqual(full["total_pixels"], 16)
            self.assertEqual(visible["box"], {"x": 0, "y": 0, "width": 2, "height": 2})
            self.assertEqual(visible["total_pixels"], 4)
            self.assertEqual({row["id"]: row["pixels"] for row in visible["classes"]}, {3: 4})

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

    def test_talc_histograms_and_hsv_threshold_update_full_review_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "sample"
            sample_dir.mkdir()
            image_path = root / "raw.png"
            mask_path = sample_dir / "ore_mask.png"
            image = Image.new("RGB", (5, 5), (200, 200, 200))
            image.putpixel((0, 0), (5, 5, 5))
            image.putpixel((2, 0), (5, 5, 5))
            image.save(image_path)
            mask = Image.new("L", (5, 5), 0)
            mask.putpixel((1, 0), 3)
            mask.putpixel((2, 0), 1)
            mask.save(mask_path)
            write_sample_metadata(sample_dir, image_path, mask_path)
            append_brush_patch(sample_dir, x=4, y=4, radius=1, class_id=4)

            histograms = talc_histograms_for_panorama(sample_dir)
            self.assertEqual(histograms["histograms"]["hsv_value"]["histogram"][5], 2)
            self.assertEqual(histograms["histograms"]["rgb_sum"]["histogram"][15], 2)

            result = apply_talc_threshold_to_panorama(sample_dir, metric="hsv_value", threshold=10)
            tile = render_panorama_tile(sample_dir, layer="class_index", x=0, y=0, width=5, height=5)

            self.assertTrue(result["ok"])
            self.assertEqual(tile.getpixel((0, 0)), 3)
            self.assertEqual(tile.getpixel((1, 0)), 0)
            self.assertEqual(tile.getpixel((2, 0)), 1)
            self.assertEqual(tile.getpixel((4, 4)), 4)
            self.assertFalse((sample_dir / "patch_log.jsonl").exists())
            metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["talc_threshold"]["metric"], "hsv_value")

            restore_base_prediction(sample_dir)
            restored = render_panorama_tile(sample_dir, layer="class_index", x=0, y=0, width=5, height=5)
            self.assertEqual(restored.getpixel((0, 0)), 0)
            self.assertEqual(restored.getpixel((1, 0)), 3)
            self.assertEqual(restored.getpixel((2, 0)), 1)
            self.assertEqual(restored.getpixel((4, 4)), 0)
            restored_metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertNotIn("talc_threshold", restored_metadata)

    def test_rgb_sum_talc_threshold_uses_765_range_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "sample"
            sample_dir.mkdir()
            image_path = root / "raw.png"
            mask_path = sample_dir / "ore_mask.png"
            image = Image.new("RGB", (2, 1), (20, 20, 20))
            image.putpixel((0, 0), (10, 10, 10))
            image.save(image_path)
            Image.new("L", (2, 1), 0).save(mask_path)
            write_sample_metadata(sample_dir, image_path, mask_path)

            apply_talc_threshold_to_panorama(sample_dir, metric="rgb_sum", threshold=40)
            tile = render_panorama_tile(sample_dir, layer="class_index", x=0, y=0, width=2, height=1)

            self.assertEqual(tile.getpixel((0, 0)), 3)
            self.assertEqual(tile.getpixel((1, 0)), 0)

    def test_ore_only_review_uses_multiclass_review_mask_without_binary_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "sample"
            active_root = root / "active"
            sample_dir.mkdir()
            image_path = root / "raw.png"
            mask_path = sample_dir / "ore_multiclass_mask.png"
            confidence_path = sample_dir / "ore_multiclass_confidence.png"
            Image.new("RGB", (16, 16), (10, 20, 30)).save(image_path)
            mask = Image.new("L", (16, 16), 0)
            ImageDraw.Draw(mask).rectangle((2, 2, 13, 13), fill=1)
            mask.save(mask_path)
            Image.new("L", (16, 16), 200).save(confidence_path)
            metadata = {
                "sample_id": "ore-job",
                "image_path": str(image_path),
                "image_width": 16,
                "image_height": 16,
                "total_tiles": 1,
                "ore_checkpoint": {"class_names": ["background", "pyrite"], "background_index": 0},
                "artifacts": {
                    "ore_mask": None,
                    "ore_multiclass_mask": str(mask_path),
                    "ore_multiclass_confidence": str(confidence_path),
                    "review_mask": str(mask_path),
                    "base_prediction_mask": str(mask_path),
                },
            }
            (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            tile = render_panorama_tile(sample_dir, layer="class_index", x=0, y=0, width=16, height=16)
            metrics = class_area_metrics(sample_dir)
            intergrowth = save_erosion_ratio_intergrowth_artifacts(
                sample_dir,
                config=ErosionRatioConfig(
                    erosion_kernel_size=3,
                    erosion_iterations=0,
                    window_size=16,
                    normal_threshold=0.5,
                    min_ore_fraction=0,
                ),
            )
            intergrowth_metrics = class_area_metrics(sample_dir, layer="intergrowth")

            self.assertEqual(tile.getpixel((8, 8)), 1)
            self.assertEqual(metrics["classes"][0]["name"], "pyrite")
            self.assertEqual(intergrowth["source_ore_mask"], "ore_multiclass_mask_non_background")
            self.assertIn("intergrowth_score", intergrowth["artifacts"])
            self.assertIn("intergrowth_hard_score", intergrowth["artifacts"])
            self.assertIn("score_metrics", intergrowth)
            self.assertGreater(intergrowth_metrics["score_metrics"]["mean_erosion_ratio_score"], 0)
            try:
                import torch  # noqa: F401
            except ModuleNotFoundError:
                return
            saved = save_panorama_review(sample_dir, output_root=active_root, crop_size=8)
            self.assertTrue(Path(saved["class_index_mask"]).exists())

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
            normal_soft = render_panorama_tile(sample_dir, layer="intergrowth_normal_soft_mask", x=0, y=0, width=8, height=8)
            hard_soft = render_panorama_tile(sample_dir, layer="intergrowth_hard_soft_mask", x=0, y=0, width=8, height=8)
            normal_overlay = render_panorama_tile(sample_dir, layer="intergrowth_normal_soft_overlay", x=0, y=0, width=8, height=8)
            metrics = class_area_metrics(sample_dir, layer="intergrowth")

            self.assertEqual(class_tile.getpixel((2, 2)), 4)
            self.assertEqual(class_tile.getpixel((3, 3)), 5)
            self.assertEqual(normal_soft.mode, "RGB")
            self.assertEqual(hard_soft.mode, "RGB")
            self.assertEqual(normal_overlay.mode, "RGB")
            self.assertNotEqual(normal_soft.getpixel((0, 0))[0], normal_soft.getpixel((0, 0))[1])
            self.assertEqual({row["id"] for row in metrics["classes"]}, {4, 5})
            self.assertEqual(metrics["intergrowth_metrics"]["ore_pixels"], 2)
            self.assertEqual(metrics["intergrowth_metrics"]["normal_ore_pixels"], 1)
            self.assertEqual(metrics["intergrowth_metrics"]["hard_ore_pixels"], 1)
            self.assertAlmostEqual(metrics["intergrowth_metrics"]["normal_ore_fraction_of_ore"], 0.5)
            self.assertIn("legend", metrics)
            self.assertIn("color", metrics["classes"][0])
            self.assertEqual(metrics["score_metrics"]["ore_pixels"], 2)
            self.assertAlmostEqual(metrics["score_metrics"]["mean_erosion_ratio_score"], 128 / 255)
            self.assertAlmostEqual(metrics["score_metrics"]["mean_hard_score"], 127 / 255)

    def test_intergrowth_metrics_before_artifacts_returns_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "sample"
            sample_dir.mkdir()
            image_path = root / "raw.png"
            mask_path = sample_dir / "ore_mask.png"
            Image.new("RGB", (8, 8), (10, 20, 30)).save(image_path)
            Image.new("L", (8, 8), 0).save(mask_path)
            write_sample_metadata(sample_dir, image_path, mask_path)

            metrics = class_area_metrics(sample_dir, layer="intergrowth")
            tile = render_panorama_tile(
                sample_dir,
                layer="intergrowth_normal_soft_mask",
                x=0,
                y=0,
                width=8,
                height=8,
                output_width=4,
                output_height=4,
            )

            self.assertFalse(metrics["intergrowth_ready"])
            self.assertEqual(metrics["message"], "intergrowth not ready")
            self.assertEqual(metrics["classes"], [])
            self.assertEqual(tile.size, (4, 4))
            self.assertEqual(tile.mode, "RGB")

    def test_large_intergrowth_uses_score_grid_artifacts_tiles_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "sample"
            sample_dir.mkdir()
            image_path = root / "raw.png"
            mask_path = sample_dir / "ore_mask.png"
            Image.new("RGB", (1200, 1000), (10, 20, 30)).save(image_path)
            mask = Image.new("L", (1200, 1000), 0)
            ImageDraw.Draw(mask).rectangle((100, 100, 1099, 899), fill=1)
            mask.save(mask_path)
            write_sample_metadata(sample_dir, image_path, mask_path)

            result = save_erosion_ratio_intergrowth_artifacts(
                sample_dir,
                config=ErosionRatioConfig(
                    erosion_kernel_size=3,
                    erosion_iterations=0,
                    window_size=200,
                    normal_threshold=0.5,
                    min_ore_fraction=0,
                ),
                large_image_pixel_threshold=1,
            )
            metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
            normal_tile = render_panorama_tile(
                sample_dir,
                layer="intergrowth_normal_soft_mask",
                x=0,
                y=0,
                width=1200,
                height=1000,
                output_width=120,
                output_height=100,
            )
            hard_overlay = render_panorama_tile(
                sample_dir,
                layer="intergrowth_hard_soft_overlay",
                x=0,
                y=0,
                width=1200,
                height=1000,
                output_width=120,
                output_height=100,
            )
            class_tile = render_panorama_tile(
                sample_dir,
                layer="intergrowth_class_index",
                x=0,
                y=0,
                width=1200,
                height=1000,
                output_width=120,
                output_height=100,
            )
            full_metrics = class_area_metrics(sample_dir, layer="intergrowth")
            crop_metrics = class_area_metrics(sample_dir, layer="intergrowth", x=0, y=0, width=1200, height=900)

            self.assertEqual(result["mode"], "score_grid")
            self.assertTrue(Path(result["artifacts"]["intergrowth_score_grid"]).exists())
            self.assertTrue(Path(result["artifacts"]["intergrowth_hard_score_grid"]).exists())
            self.assertNotIn("intergrowth_mask", metadata["artifacts"])
            self.assertEqual(metadata["intergrowth"]["mode"], "score_grid")
            self.assertEqual(normal_tile.size, (120, 100))
            self.assertEqual(hard_overlay.size, (120, 100))
            self.assertEqual(class_tile.mode, "L")
            self.assertTrue(full_metrics["approximate"])
            self.assertEqual(full_metrics["mode"], "score_grid")
            self.assertGreater(full_metrics["intergrowth_metrics"]["ore_pixels"], 0)
            self.assertTrue(crop_metrics["approximate"])
            self.assertEqual(crop_metrics["mode"], "score_grid")
            self.assertGreater(crop_metrics["intergrowth_metrics"]["normal_ore_pixels"], 0)

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
