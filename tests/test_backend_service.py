import base64
from io import BytesIO
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.backend.service import (
    BackendConfig,
    create_prediction_from_request,
    list_saved_class_index_masks,
    list_ui_images,
    render_active_learning_html,
    render_index_html,
    render_inference_html,
    render_panorama_review_html,
    render_prediction_html,
    save_edited_mask_from_request,
    save_uploaded_image_from_request,
)
from ore_detection.backend.ui_annotation import ui_class_metadata, ui_classes_for_model


def image_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class TestBackendService(unittest.TestCase):
    def test_list_ui_images_finds_baseline_and_source_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "datasets" / "baseline" / "Part 1" / "Normal ore" / "a.jpg"
            skipped = root / "datasets" / "set_1" / "masks_human" / "train" / "mask.png"
            source = root / "datasets" / "set_1" / "imgs" / "train" / "b.jpg"
            first.parent.mkdir(parents=True)
            skipped.parent.mkdir(parents=True)
            source.parent.mkdir(parents=True)
            Image.new("RGB", (1, 1)).save(first)
            Image.new("RGB", (1, 1)).save(skipped)
            Image.new("RGB", (1, 1)).save(source)

            images = list_ui_images(root / "datasets")

            self.assertEqual([p.name for p in images], ["a.jpg", "b.jpg"])

    def test_repo_demo_images_are_available_to_ui_listing(self):
        images = list_ui_images(Path("datasets"))

        self.assertIn(Path("datasets/demo/hard_ore_demo.jpg"), images)
        self.assertIn(Path("datasets/demo/normal_ore_demo.jpg"), images)

    def test_default_runtime_artifact_paths_exist(self):
        config = BackendConfig().resolve()

        self.assertTrue(config.binary_model_path.exists())
        self.assertTrue(config.ore_model_path.exists())
        self.assertTrue(config.intergrowth_classifier_path.exists())

    def test_create_prediction_from_request_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "img.jpg"
            Image.new("RGB", (2, 1)).save(image_path)
            config = BackendConfig(project_root=root, predictions_root=root / "predictions")

            artifacts = create_prediction_from_request(
                image_path=str(image_path),
                value_threshold="1",
                foreground="bright",
                standardize="off",
                config=config,
            )

            self.assertTrue(artifacts.ore_mask_path.exists())
            self.assertIn("hsv_dummy", str(artifacts.sample_dir))

    def test_render_index_html_shows_only_workflow_choices(self):
        html = render_index_html([], default_threshold=90)

        self.assertIn("Inference", html)
        self.assertIn("/inference", html)
        self.assertIn("Active Learning", html)
        self.assertIn("/active-learning", html)
        self.assertNotIn("Panorama inference job", html)
        self.assertNotIn("Talc mask creation", html)

    def test_render_inference_html_has_prediction_controls_without_brush_tools(self):
        html = render_inference_html([])

        self.assertIn("/jobs/panorama-predict", html)
        self.assertIn("binary segmentation", html)
        self.assertIn("ore segmentation", html)
        self.assertIn("Crop area", html)
        self.assertIn("Metrics", html)
        self.assertIn("Select new image", html)
        self.assertIn("Run intergrowth classification", html)
        self.assertIn("intergrowth score", html)
        self.assertIn("/intergrowth", html)
        self.assertIn("Drag and drop an image here", html)
        self.assertIn("/upload-image", html)
        self.assertIn("fileInput", html)
        self.assertNotIn("Brush", html)
        self.assertNotIn("Talc mask creation", html)

    def test_render_active_learning_html_has_image_next_and_prediction_launcher(self):
        html = render_active_learning_html([])

        self.assertIn("Active Learning", html)
        self.assertIn("Next image", html)
        self.assertIn("Run prediction and open editor", html)
        self.assertIn("/active-learning?job_id=", html)
        self.assertIn("/jobs/panorama-predict", html)
        self.assertIn("binary segmentation", html)
        self.assertIn("ore segmentation", html)

    def test_panorama_review_hover_redraws_guides_without_tile_refetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_id = "job1"
            sample_dir = root / "predictions" / "panorama" / job_id
            job_dir = root / "jobs" / job_id
            sample_dir.mkdir(parents=True)
            job_dir.mkdir(parents=True)
            image_path = root / "raw.png"
            mask_path = sample_dir / "ore_mask.png"
            Image.new("RGB", (8, 8), (10, 20, 30)).save(image_path)
            Image.new("L", (8, 8), 0).save(mask_path)
            metadata = {
                "sample_id": job_id,
                "image_path": str(image_path),
                "image_width": 8,
                "image_height": 8,
                "total_tiles": 1,
                "artifacts": {
                    "ore_mask": str(mask_path),
                    "ore_confidence": str(mask_path),
                    "ore_probability": str(mask_path),
                    "review_mask": str(mask_path),
                },
            }
            (sample_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            progress = {
                "job_id": job_id,
                "status": "completed",
                "phase": "completed",
                "sample_dir": str(sample_dir),
                "total_tiles": 1,
            }
            (job_dir / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
            config = BackendConfig(
                project_root=root,
                predictions_root=root / "predictions",
                panorama_jobs_root=root / "jobs",
            )

            html = render_panorama_review_html(job_id, config=config)

            self.assertIn("const canvasImages = {raw: null, overlay: null, mask: null};", html)
            self.assertIn("function redrawGuides()", html)
            self.assertIn("if (cropSelecting && cropStart) { cropCurrent = p; redrawGuides(); return; }", html)
            self.assertIn("canvas.addEventListener('mouseleave', () => { hoverPoint = null; redrawGuides(); });", html)
            self.assertIn("t: String(tileRevision)", html)
            self.assertNotIn("t: String(Date.now())", html)

    def test_render_prediction_html_contains_editor_and_talc_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "img.jpg"
            Image.new("RGB", (2, 1), (10, 20, 30)).save(image_path)
            config = BackendConfig(project_root=root, predictions_root=root / "predictions")
            artifacts = create_prediction_from_request(
                image_path=str(image_path), value_threshold="1", foreground="bright", config=config
            )

            html = render_prediction_html(artifacts, config=config)

            self.assertIn("Ore mask review instrument", html)
            self.assertIn("Current model", html)
            self.assertIn("Image name/address", html)
            self.assertIn("Color → class description", html)
            self.assertIn("Instrument", html)
            self.assertIn("View — no new artifacts", html)
            self.assertIn("Scale/crop all three images together only", html)
            self.assertIn("Active learning brush", html)
            self.assertIn("Mask class currently edited", html)
            self.assertIn("Add selected class", html)
            self.assertIn("Remove to background", html)
            self.assertIn("Save active-learning mask", html)
            self.assertIn("Restore prediction mask", html)
            self.assertIn("Raw image + mask", html)
            self.assertIn("Talc mask creation", html)
            self.assertIn("Histogram: HSV Value", html)
            self.assertIn("R + G + B", html)
            self.assertIn("Metrics for all non-zero classes", html)
            self.assertIn("/save-mask", html)
            self.assertIn("one-hot mask tensor", html)

    def test_model_ui_classes_append_talc_normal_hard(self):
        classes = ui_classes_for_model(("background", "pyrite", "chalcopyrite"))
        self.assertEqual([item.name for item in classes[:3]], ["background", "pyrite", "chalcopyrite"])
        self.assertIn("talc", [item.name for item in classes])
        self.assertIn("normal_ore", [item.name for item in classes])
        self.assertIn("hard_ore", [item.name for item in classes])
        talc = next(item for item in classes if item.name == "talc")
        self.assertEqual(talc.color, (255, 255, 255))

    def test_save_edited_mask_from_request_writes_png_metadata_and_torch_tensor(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "datasets" / "sample.png"
            source.parent.mkdir(parents=True)
            Image.new("RGB", (2, 2)).save(source)
            mask = Image.new("L", (2, 2))
            mask.putdata([0, 1, 3, 4])
            config = BackendConfig(project_root=root, active_learning_root=root / "active")

            metadata = save_edited_mask_from_request(
                image_path=str(source),
                mask_data_url=image_data_url(mask),
                classes_json=json.dumps(ui_class_metadata()),
                config=config,
            )

            self.assertTrue(Path(metadata["class_index_mask"]).exists())
            self.assertTrue(Path(metadata["one_hot_tensor"]).exists())
            saved = torch.load(metadata["one_hot_tensor"], map_location="cpu", weights_only=False)
            self.assertEqual(tuple(saved["one_hot"].shape), (6, 2, 2))
            self.assertEqual(saved["channel_class_names"][:4], ["background", "sulfide_ore", "oxide_magnetite_hematite", "talc"])
            self.assertAlmostEqual(metadata["metrics"]["talc"], 0.25)

    def test_create_prediction_from_request_can_reload_saved_class_index_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "img.jpg"
            Image.new("RGB", (2, 2), (10, 20, 30)).save(image_path)
            mask_path = root / "active" / "sample" / "class_index_mask.png"
            mask_path.parent.mkdir(parents=True)
            mask = Image.new("L", (2, 2))
            mask.putdata([0, 3, 4, 5])
            mask.save(mask_path)
            config = BackendConfig(project_root=root, predictions_root=root / "predictions", active_learning_root=root / "active")

            artifacts = create_prediction_from_request(
                image_path=str(image_path),
                value_threshold="1",
                foreground="bright",
                saved_mask_path=str(mask_path),
                config=config,
            )
            html = render_prediction_html(artifacts, config=config)
            metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))

            self.assertTrue((artifacts.sample_dir / "loaded_class_index_mask.png").exists())
            self.assertEqual(list(Image.open(artifacts.sample_dir / "loaded_class_index_mask.png").tobytes()), [0, 3, 4, 5])
            self.assertEqual(Path(metadata["loaded_active_learning_mask"]), mask_path)
            self.assertIn("loaded_class_index_mask.png", html)

    def test_list_saved_class_index_masks_finds_only_saved_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved = root / "active" / "a" / "class_index_mask.png"
            other = root / "active" / "b" / "mask_preview.png"
            saved.parent.mkdir(parents=True)
            other.parent.mkdir(parents=True)
            Image.new("L", (1, 1)).save(saved)
            Image.new("RGB", (1, 1)).save(other)

            self.assertEqual(list_saved_class_index_masks(root / "active"), [saved])

    def test_save_uploaded_image_from_request_persists_drag_drop_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = BackendConfig(project_root=root, uploads_root=root / "uploads")

            result = save_uploaded_image_from_request(
                file_name="Dropped Image.png",
                image_data_url=image_data_url(Image.new("RGB", (1, 1), (1, 2, 3))),
                config=config,
            )

            self.assertTrue(Path(result["path"]).exists())
            self.assertTrue(result["relative_path"].startswith("uploads/"))


if __name__ == "__main__":
    unittest.main()
