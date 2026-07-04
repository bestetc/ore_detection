import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ore_detection.backend.service import (
    BackendConfig,
    create_prediction_from_request,
    render_prediction_html,
    resolve_artifact_path,
)


class TestBackendArtifacts(unittest.TestCase):
    def test_render_prediction_html_shows_overlay_and_accept_form(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "img.jpg"
            Image.new("RGB", (2, 1)).save(image_path)
            config = BackendConfig(project_root=root, predictions_root=root / "predictions")
            artifacts = create_prediction_from_request(
                image_path=str(image_path), value_threshold="1", foreground="bright", config=config
            )

            html = render_prediction_html(artifacts, config=config)

            self.assertIn("overlay.png", html)
            self.assertIn("ore_mask.png", html)
            self.assertIn("/accept", html)
            self.assertIn("Accept original prediction mask", html)

    def test_resolve_artifact_path_keeps_requests_inside_prediction_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = BackendConfig(project_root=root, predictions_root=root / "predictions").resolve()
            artifact = config.predictions_root / "sample" / "overlay.png"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"x")

            resolved = resolve_artifact_path("sample/overlay.png", config=config)

            self.assertEqual(resolved, artifact)
            with self.assertRaisesRegex(ValueError, "outside prediction root"):
                resolve_artifact_path("../secret.txt", config=config)


if __name__ == "__main__":
    unittest.main()
