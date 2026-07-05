"""Stdlib HTTP server for the local ore segmentation review UI."""

from __future__ import annotations

import json
import mimetypes
from io import BytesIO
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ore_detection.backend.service import (
    BackendConfig,
    accept_prediction_from_request,
    add_panorama_brush_patch_from_request,
    apply_panorama_talc_threshold_from_request,
    cancel_panorama_job,
    create_prediction_from_request,
    get_panorama_job_status,
    list_saved_class_index_masks,
    list_ui_images,
    panorama_metrics_from_request,
    panorama_talc_histograms_from_request,
    render_active_learning_html,
    render_index_html,
    render_inference_html,
    render_panorama_review_html,
    render_panorama_tile_from_request,
    render_prediction_html,
    resolve_artifact_path,
    resolve_source_image_path,
    restore_panorama_prediction_from_request,
    run_intergrowth_from_request,
    save_panorama_review_from_request,
    save_edited_mask_from_request,
    save_uploaded_image_from_request,
    start_panorama_prediction_from_request,
)


CLIENT_DISCONNECT_WINERRORS = {10053, 10054}


def is_client_disconnect(exc: BaseException) -> bool:
    """Return True for browser/client disconnects while a response is being sent."""
    if isinstance(exc, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)):
        return True
    return isinstance(exc, OSError) and getattr(exc, "winerror", None) in CLIENT_DISCONNECT_WINERRORS


class OreDetectionHandler(SimpleHTTPRequestHandler):
    backend_config = BackendConfig().resolve()

    def _end_headers_safely(self) -> bool:
        try:
            self.end_headers()
            return True
        except OSError as exc:
            if is_client_disconnect(exc):
                self.close_connection = True
                return False
            raise

    def _write_body_safely(self, body: bytes) -> bool:
        try:
            self.wfile.write(body)
            return True
        except OSError as exc:
            if is_client_disconnect(exc):
                self.close_connection = True
                return False
            raise

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if self._end_headers_safely():
            self._write_body_safely(body)

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if self._end_headers_safely():
            self._write_body_safely(body)

    def _send_file(self, path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        if not self._end_headers_safely():
            return
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                if not self._write_body_safely(chunk):
                    return

    def _send_image(self, image, *, fmt: str = "PNG") -> None:
        buffer = BytesIO()
        image.save(buffer, format=fmt)
        body = buffer.getvalue()
        content_type = "image/jpeg" if fmt.upper() in {"JPEG", "JPG"} else "image/png"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if self._end_headers_safely():
            self._write_body_safely(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            images = list_ui_images(self.backend_config.datasets_root, limit=500)
            masks = list_saved_class_index_masks(self.backend_config.active_learning_root, limit=500)
            self._send_html(render_index_html(images, saved_masks=masks, config=self.backend_config))
            return
        if parsed.path == "/inference":
            images = list_ui_images(self.backend_config.datasets_root, limit=500)
            self._send_html(render_inference_html(images, config=self.backend_config))
            return
        if parsed.path == "/active-learning":
            query = parse_qs(parsed.query)
            job_id = query.get("job_id", [""])[0].strip()
            if job_id:
                try:
                    self._send_html(render_panorama_review_html(job_id, config=self.backend_config))
                except Exception as exc:
                    images = list_ui_images(self.backend_config.datasets_root, limit=500)
                    self._send_html(render_active_learning_html(images, message=str(exc), config=self.backend_config))
                return
            images = list_ui_images(self.backend_config.datasets_root, limit=500)
            self._send_html(render_active_learning_html(images, config=self.backend_config))
            return
        if parsed.path == "/artifact":
            query = parse_qs(parsed.query)
            try:
                artifact = resolve_artifact_path(query.get("path", [""])[0], config=self.backend_config)
            except ValueError as exc:
                self.send_error(400, str(exc))
                return
            self._send_file(artifact)
            return
        if parsed.path == "/source-image":
            query = parse_qs(parsed.query)
            try:
                source = resolve_source_image_path(query.get("path", [""])[0], config=self.backend_config)
            except ValueError as exc:
                self.send_error(400, str(exc))
                return
            self._send_file(source)
            return
        if parsed.path.startswith("/jobs/"):
            self._handle_job_get(parsed)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length).decode("utf-8")
        form = {key: values[0] for key, values in parse_qs(data).items()}
        if parsed.path == "/jobs/panorama-predict":
            self._handle_panorama_predict(form)
            return
        if parsed.path.startswith("/jobs/"):
            self._handle_job_post(parsed, form)
            return
        if parsed.path == "/predict":
            self._handle_predict(form)
            return
        if parsed.path == "/accept":
            self._handle_accept(form)
            return
        if parsed.path == "/save-mask":
            self._handle_save_mask(form)
            return
        if parsed.path == "/upload-image":
            self._handle_upload_image(form)
            return
        self.send_error(404)

    def _handle_predict(self, form: dict[str, str]) -> None:
        try:
            artifacts = create_prediction_from_request(config=self.backend_config, **form)
            self._send_html(render_prediction_html(artifacts, config=self.backend_config))
        except Exception as exc:  # keep UI alive and show actionable error
            images = list_ui_images(self.backend_config.datasets_root, limit=500)
            masks = list_saved_class_index_masks(self.backend_config.active_learning_root, limit=500)
            self._send_html(render_index_html(images, message=f"Error: {exc}", saved_masks=masks, config=self.backend_config), status=400)

    def _handle_accept(self, form: dict[str, str]) -> None:
        try:
            correction = accept_prediction_from_request(config=self.backend_config, **form)
            message = f"Accepted correction saved to {correction.correction_dir}"
            status = 200
        except Exception as exc:
            message = f"Error: {exc}"
            status = 400
        images = list_ui_images(self.backend_config.datasets_root, limit=500)
        masks = list_saved_class_index_masks(self.backend_config.active_learning_root, limit=500)
        self._send_html(render_index_html(images, message=message, saved_masks=masks, config=self.backend_config), status=status)

    def _handle_save_mask(self, form: dict[str, str]) -> None:
        try:
            metadata = save_edited_mask_from_request(config=self.backend_config, **form)
            self._send_json(metadata)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _handle_upload_image(self, form: dict[str, str]) -> None:
        try:
            result = save_uploaded_image_from_request(config=self.backend_config, **form)
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _handle_panorama_predict(self, form: dict[str, str]) -> None:
        try:
            status = start_panorama_prediction_from_request(config=self.backend_config, **form)
            self._send_json(status)
        except Exception as exc:
            model_kind = form.get("model_kind") if form.get("model_kind") in {"binary", "ore", "ct_unet"} else "binary"
            selected_field = {
                "binary": "binary_model_path",
                "ore": "ore_model_path",
                "ct_unet": "ct_unet_model_path",
            }[model_kind]
            selected_path = form.get(selected_field, "")
            self._send_json(
                {
                    "error": str(exc),
                    "model_kind": model_kind,
                    "selected_model_path": selected_path,
                    "exception_type": type(exc).__name__,
                },
                status=400,
            )

    def _handle_job_get(self, parsed) -> None:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            self.send_error(404)
            return
        job_id = parts[1]
        try:
            if len(parts) == 2:
                self._send_json(get_panorama_job_status(job_id, config=self.backend_config))
                return
            if len(parts) == 3 and parts[2] == "review":
                self._send_html(render_panorama_review_html(job_id, config=self.backend_config))
                return
            if len(parts) == 3 and parts[2] == "metrics":
                query = parse_qs(parsed.query)
                self._send_json(
                    panorama_metrics_from_request(
                        job_id=job_id,
                        x=query.get("x", [""])[0],
                        y=query.get("y", [""])[0],
                        width=query.get("width", [""])[0],
                        height=query.get("height", [""])[0],
                        layer=query.get("layer", ["prediction"])[0],
                        config=self.backend_config,
                    )
                )
                return
            if len(parts) == 3 and parts[2] == "talc-histograms":
                self._send_json(panorama_talc_histograms_from_request(job_id=job_id, config=self.backend_config))
                return
            if len(parts) == 3 and parts[2] == "tile":
                query = parse_qs(parsed.query)
                image = render_panorama_tile_from_request(
                    job_id=job_id,
                    layer=query.get("layer", ["raw"])[0],
                    x=query.get("x", ["0"])[0],
                    y=query.get("y", ["0"])[0],
                    width=query.get("width", ["1024"])[0],
                    height=query.get("height", ["768"])[0],
                    output_width=query.get("output_width", [""])[0],
                    output_height=query.get("output_height", [""])[0],
                    config=self.backend_config,
                )
                self._send_image(image, fmt="PNG")
                return
        except Exception as exc:
            if is_client_disconnect(exc):
                self.close_connection = True
                return
            self._send_json({"error": str(exc)}, status=400)
            return
        self.send_error(404)

    def _handle_job_post(self, parsed, form: dict[str, str]) -> None:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) != 3:
            self.send_error(404)
            return
        job_id = parts[1]
        action = parts[2]
        try:
            if action == "cancel":
                self._send_json(cancel_panorama_job(job_id, config=self.backend_config))
                return
            if action == "brush":
                self._send_json(add_panorama_brush_patch_from_request(job_id=job_id, config=self.backend_config, **form))
                return
            if action == "restore":
                self._send_json(restore_panorama_prediction_from_request(job_id=job_id, config=self.backend_config))
                return
            if action == "talc-threshold":
                self._send_json(
                    apply_panorama_talc_threshold_from_request(job_id=job_id, config=self.backend_config, **form)
                )
                return
            if action == "intergrowth":
                self._send_json(run_intergrowth_from_request(job_id=job_id, config=self.backend_config, **form))
                return
            if action == "save-review":
                self._send_json(save_panorama_review_from_request(job_id=job_id, config=self.backend_config, **form))
                return
        except Exception as exc:
            if is_client_disconnect(exc):
                self.close_connection = True
                return
            self._send_json({"error": str(exc)}, status=400)
            return
        self.send_error(404)


def run_server(*, host: str = "127.0.0.1", port: int = 7860, config: BackendConfig | None = None) -> None:
    handler = OreDetectionHandler
    handler.backend_config = (config or BackendConfig()).resolve()
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Ore Detection UI: http://{host}:{port}")
    server.serve_forever()
