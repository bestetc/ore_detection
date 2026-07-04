"""Backend helpers for the local segmentation review and active-learning UI."""

from __future__ import annotations

import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from ore_detection.backend.ui_annotation import (
    save_edited_mask_from_data_url,
    save_uploaded_image_from_data_url,
    ui_class_metadata,
)
from ore_detection.backend.panorama_jobs import PANORAMA_JOB_MANAGER, PanoramaPredictionRequest
from ore_detection.backend.panorama_review import (
    append_brush_patch,
    class_area_metrics,
    read_panorama_metadata,
    render_panorama_tile,
    restore_base_prediction,
    save_panorama_review,
)
from ore_detection.descriptors.intergrowth_classification import (
    IntergrowthClassifierConfig,
    load_intergrowth_classifier_config,
    save_intergrowth_artifacts,
)
from ore_detection.inference.model_prediction import (
    SegmentationPredictionArtifacts,
    load_simple_unet_checkpoint,
    save_segmentation_prediction,
)
from ore_detection.inference.prediction_store import (
    CorrectionArtifacts,
    PredictionArtifacts,
    accept_prediction_as_correction,
    save_hsv_dummy_prediction,
)
from ore_detection.segmentation.hsv_dummy import HsvDummyConfig
from ore_detection.talc.hsv_candidates import IMAGE_SUFFIXES, calculate_rgb_mean_std

ArtifactLike = PredictionArtifacts | SegmentationPredictionArtifacts


@dataclass(frozen=True)
class BackendConfig:
    project_root: Path = Path(".")
    datasets_root: Path = Path("datasets")
    predictions_root: Path = Path("data_work/predictions/ui")
    active_learning_root: Path = Path("data_work/active_learning_masks")
    panorama_jobs_root: Path = Path("data_work/panorama_jobs")
    uploads_root: Path = Path("data_work/ui_uploads")
    binary_model_path: Path = Path("models/source_binary_segmentation/001/best.pt")
    ore_model_path: Path = Path("models/source_ore_segmentation/001/best.pt")
    intergrowth_classifier_path: Path = Path("models/intergrowth_classifier/001/classifier.json")
    stats_sample_max_image_size: int = 512
    panorama_tile_size: int = 512
    panorama_overlap: int = 128
    panorama_batch_size: int = 4

    def resolve(self) -> "BackendConfig":
        root = Path(self.project_root).resolve()

        def under_root(path: Path) -> Path:
            return (root / path).resolve() if not path.is_absolute() else path.resolve()

        return BackendConfig(
            project_root=root,
            datasets_root=under_root(Path(self.datasets_root)),
            predictions_root=under_root(Path(self.predictions_root)),
            active_learning_root=under_root(Path(self.active_learning_root)),
            panorama_jobs_root=under_root(Path(self.panorama_jobs_root)),
            uploads_root=under_root(Path(self.uploads_root)),
            binary_model_path=under_root(Path(self.binary_model_path)),
            ore_model_path=under_root(Path(self.ore_model_path)),
            intergrowth_classifier_path=under_root(Path(self.intergrowth_classifier_path)),
            stats_sample_max_image_size=self.stats_sample_max_image_size,
            panorama_tile_size=self.panorama_tile_size,
            panorama_overlap=self.panorama_overlap,
            panorama_batch_size=self.panorama_batch_size,
        )


def list_ui_images(root: str | Path, *, limit: int | None = None) -> list[Path]:
    """List source/baseline images while skipping mask folders."""
    root = Path(root)
    suffixes = {suffix.lower() for suffix in IMAGE_SUFFIXES}
    skipped = {"masks", "masks_human", "masks_colored"}
    images = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in suffixes
        and not (skipped & {part.lower() for part in path.relative_to(root).parts})
    ]
    images = sorted(images)
    return images[:limit] if limit is not None else images


def list_saved_class_index_masks(root: str | Path, *, limit: int | None = None) -> list[Path]:
    """List saved active-learning class-index masks in stable order."""
    root = Path(root)
    if not root.exists():
        return []
    masks = sorted(path for path in root.rglob("class_index_mask.png") if path.is_file())
    return masks[:limit] if limit is not None else masks


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").lower() in {"1", "true", "on", "yes"}


def _parse_int(value: str | int | None, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _parse_float(value: str | float | None, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _stats_for_request(image_path: Path, config: BackendConfig, *, standardize: bool) -> dict[str, object] | None:
    if not standardize:
        return None
    return calculate_rgb_mean_std([image_path], max_image_size=config.stats_sample_max_image_size)


def _resolve_request_path(path_value: str | Path, *, config: BackendConfig) -> Path:
    cfg = config.resolve()
    path = Path(path_value)
    if not path.is_absolute():
        path = (cfg.project_root / path).resolve()
    else:
        path = path.resolve()
    try:
        path.relative_to(cfg.project_root)
    except ValueError as exc:
        raise ValueError("path is outside project root") from exc
    return path


def _resolve_active_learning_mask(path_value: str | Path, *, config: BackendConfig) -> Path:
    cfg = config.resolve()
    path = Path(path_value)
    if not path.is_absolute():
        path = (cfg.project_root / path).resolve()
    else:
        path = path.resolve()
    try:
        path.relative_to(cfg.active_learning_root)
    except ValueError as exc:
        raise ValueError("saved mask path is outside active-learning root") from exc
    if path.name != "class_index_mask.png":
        raise ValueError("saved mask path must point to class_index_mask.png")
    if not path.exists():
        raise FileNotFoundError(f"saved mask does not exist: {path}")
    return path


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_prediction_from_request(
    *,
    image_path: str,
    model_kind: str = "hsv_dummy",
    value_threshold: str = "90",
    foreground: str = "bright",
    standardize: str | bool = "off",
    binary_threshold: str = "0.5",
    binary_model_path: str = "",
    ore_model_path: str = "",
    device: str = "auto",
    saved_mask_path: str = "",
    config: BackendConfig | None = None,
) -> ArtifactLike:
    """Create one UI prediction from form values."""
    cfg = (config or BackendConfig()).resolve()
    path = _resolve_request_path(image_path, config=cfg)
    if not path.exists():
        raise FileNotFoundError(f"image does not exist: {path}")

    if model_kind == "trained_binary_ore":
        resolved_binary_model = _resolve_request_path(binary_model_path or cfg.binary_model_path, config=cfg)
        resolved_ore_model = _resolve_request_path(ore_model_path or cfg.ore_model_path, config=cfg)
        resolved_device = _resolve_device(device)
        binary_model = load_simple_unet_checkpoint(resolved_binary_model, device=resolved_device)
        ore_model = load_simple_unet_checkpoint(resolved_ore_model, device=resolved_device) if resolved_ore_model.exists() else None
        artifacts = save_segmentation_prediction(
            path,
            binary_model=binary_model,
            ore_model=ore_model,
            output_root=cfg.predictions_root / "trained_binary_ore",
            binary_threshold=float(binary_threshold),
        )
        if saved_mask_path:
            _attach_saved_mask(artifacts, _resolve_active_learning_mask(saved_mask_path, config=cfg))
        return artifacts

    use_standardize = _parse_bool(standardize)
    stats = _stats_for_request(path, cfg, standardize=use_standardize)
    hsv_config = HsvDummyConfig(
        value_threshold=int(value_threshold),
        foreground="dark" if foreground == "dark" else "bright",
        standardize=use_standardize,
        standardize_stats=stats,
    )
    artifacts = save_hsv_dummy_prediction(path, output_root=cfg.predictions_root / "hsv_dummy", config=hsv_config)
    if saved_mask_path:
        _attach_saved_mask(artifacts, _resolve_active_learning_mask(saved_mask_path, config=cfg))
    return artifacts


def resolve_artifact_path(requested_path: str, *, config: BackendConfig) -> Path:
    """Resolve a UI artifact path under the configured prediction root."""
    cfg = config.resolve()
    candidate = (cfg.predictions_root / requested_path).resolve()
    try:
        candidate.relative_to(cfg.predictions_root)
    except ValueError as exc:
        raise ValueError("artifact path is outside prediction root") from exc
    return candidate


def resolve_source_image_path(requested_path: str, *, config: BackendConfig) -> Path:
    """Resolve source images under the project root for safe UI serving."""
    return _resolve_request_path(requested_path, config=config)


def artifact_url(path: Path, *, config: BackendConfig) -> str:
    """Return an `/artifact` URL for a prediction artifact."""
    cfg = config.resolve()
    rel = path.resolve().relative_to(cfg.predictions_root)
    return f"/artifact?path={quote(rel.as_posix())}"


def source_image_url(path: Path, *, config: BackendConfig) -> str:
    cfg = config.resolve()
    rel = path.resolve().relative_to(cfg.project_root)
    return f"/source-image?path={quote(rel.as_posix())}"


def accept_prediction_from_request(
    *,
    sample_dir: str,
    label: str = "ore",
    config: BackendConfig | None = None,
) -> CorrectionArtifacts:
    """Accept a prediction as a reviewed correction from UI form data."""
    cfg = (config or BackendConfig()).resolve()
    resolved = resolve_artifact_path(sample_dir, config=cfg)
    return accept_prediction_as_correction(resolved, label=label)


def save_edited_mask_from_request(
    *,
    image_path: str,
    mask_data_url: str,
    classes_json: str,
    config: BackendConfig | None = None,
) -> dict[str, Any]:
    """Persist a browser-edited class-index mask and one-hot tensor."""
    cfg = (config or BackendConfig()).resolve()
    classes = json.loads(classes_json)
    return save_edited_mask_from_data_url(
        source_image_path=_resolve_request_path(image_path, config=cfg),
        mask_data_url=mask_data_url,
        output_root=cfg.active_learning_root,
        classes=classes,
    )


def save_uploaded_image_from_request(
    *,
    file_name: str,
    image_data_url: str,
    config: BackendConfig | None = None,
) -> dict[str, str]:
    """Persist a drag-and-dropped image for server-side prediction."""
    cfg = (config or BackendConfig()).resolve()
    path = save_uploaded_image_from_data_url(file_name=file_name, image_data_url=image_data_url, output_root=cfg.uploads_root)
    rel = path.resolve().relative_to(cfg.project_root).as_posix()
    return {"path": str(path), "relative_path": rel}


def start_panorama_prediction_from_request(
    *,
    image_path: str,
    model_kind: str = "binary",
    binary_model_path: str = "",
    ore_model_path: str = "",
    include_ore_model: str | bool = "off",
    device: str = "auto",
    binary_threshold: str | float = "0.5",
    tile_size: str | int = "",
    overlap: str | int = "",
    batch_size: str | int = "",
    config: BackendConfig | None = None,
) -> dict[str, Any]:
    """Start a background panorama prediction job and return its status."""
    cfg = (config or BackendConfig()).resolve()
    path = _resolve_request_path(image_path, config=cfg)
    if not path.exists():
        raise FileNotFoundError(f"image does not exist: {path}")
    resolved_binary_model = _resolve_request_path(binary_model_path or cfg.binary_model_path, config=cfg)
    resolved_ore_model = _resolve_request_path(ore_model_path or cfg.ore_model_path, config=cfg)
    request = PanoramaPredictionRequest(
        image_path=path,
        binary_model_path=resolved_binary_model,
        ore_model_path=resolved_ore_model,
        model_kind="ore" if model_kind == "ore" else "binary",
        include_ore_model=_parse_bool(include_ore_model),
        device=device,
        binary_threshold=_parse_float(binary_threshold, 0.5),
        tile_size=_parse_int(tile_size, cfg.panorama_tile_size),
        overlap=_parse_int(overlap, cfg.panorama_overlap),
        batch_size=_parse_int(batch_size, cfg.panorama_batch_size),
    )
    return PANORAMA_JOB_MANAGER.start(
        request=request,
        jobs_root=cfg.panorama_jobs_root,
        predictions_root=cfg.predictions_root,
    )


def get_panorama_job_status(job_id: str, *, config: BackendConfig | None = None) -> dict[str, Any]:
    cfg = (config or BackendConfig()).resolve()
    return PANORAMA_JOB_MANAGER.status(job_id, jobs_root=cfg.panorama_jobs_root)


def cancel_panorama_job(job_id: str, *, config: BackendConfig | None = None) -> dict[str, Any]:
    cfg = (config or BackendConfig()).resolve()
    return PANORAMA_JOB_MANAGER.cancel(job_id, jobs_root=cfg.panorama_jobs_root)


def resolve_panorama_sample_dir(job_id: str, *, config: BackendConfig | None = None) -> Path:
    """Resolve a completed panorama sample directory from its job id."""
    cfg = (config or BackendConfig()).resolve()
    status = get_panorama_job_status(job_id, config=cfg)
    candidate = Path(str(status.get("sample_dir") or cfg.predictions_root / "panorama" / job_id)).resolve()
    try:
        candidate.relative_to(cfg.predictions_root)
    except ValueError as exc:
        raise ValueError("panorama sample directory is outside predictions root") from exc
    if not candidate.exists():
        raise FileNotFoundError(f"panorama sample directory does not exist: {candidate}")
    return candidate


def render_panorama_tile_from_request(
    *,
    job_id: str,
    layer: str,
    x: str = "0",
    y: str = "0",
    width: str = "1024",
    height: str = "768",
    output_width: str = "",
    output_height: str = "",
    config: BackendConfig | None = None,
):
    sample_dir = resolve_panorama_sample_dir(job_id, config=config)
    return render_panorama_tile(
        sample_dir,
        layer=layer,
        x=int(float(x or 0)),
        y=int(float(y or 0)),
        width=max(1, int(float(width or 1))),
        height=max(1, int(float(height or 1))),
        output_width=int(float(output_width)) if output_width else None,
        output_height=int(float(output_height)) if output_height else None,
    )


def add_panorama_brush_patch_from_request(
    *,
    job_id: str,
    x: str,
    y: str,
    radius: str,
    class_id: str,
    config: BackendConfig | None = None,
) -> dict[str, Any]:
    sample_dir = resolve_panorama_sample_dir(job_id, config=config)
    patch = append_brush_patch(
        sample_dir,
        x=int(float(x)),
        y=int(float(y)),
        radius=max(1, int(float(radius))),
        class_id=int(float(class_id)),
    )
    return {"ok": True, "patch": patch.as_dict(), "patch_log": str(sample_dir / "patch_log.jsonl")}


def restore_panorama_prediction_from_request(
    *,
    job_id: str,
    config: BackendConfig | None = None,
) -> dict[str, Any]:
    sample_dir = resolve_panorama_sample_dir(job_id, config=config)
    return restore_base_prediction(sample_dir)


def panorama_metrics_from_request(
    *,
    job_id: str,
    x: str = "",
    y: str = "",
    width: str = "",
    height: str = "",
    layer: str = "prediction",
    config: BackendConfig | None = None,
) -> dict[str, Any]:
    sample_dir = resolve_panorama_sample_dir(job_id, config=config)
    metric_layer = "intergrowth" if layer == "intergrowth" else "prediction"
    if x == "" or y == "" or width == "" or height == "":
        return class_area_metrics(sample_dir, layer=metric_layer)
    return class_area_metrics(
        sample_dir,
        x=int(float(x)),
        y=int(float(y)),
        width=max(1, int(float(width))),
        height=max(1, int(float(height))),
        layer=metric_layer,
    )


def run_intergrowth_from_request(
    *,
    job_id: str,
    window_size: str | int = "",
    stride: str | int = "",
    hard_threshold: str | float = "",
    config: BackendConfig | None = None,
) -> dict[str, Any]:
    """Run hard/normal morphology classification for a completed prediction job."""
    cfg = (config or BackendConfig()).resolve()
    sample_dir = resolve_panorama_sample_dir(job_id, config=cfg)
    classifier = load_intergrowth_classifier_config(cfg.intergrowth_classifier_path)
    if window_size != "":
        classifier = IntergrowthClassifierConfig.from_dict({**classifier.to_dict(), "window_size": int(float(window_size))})
    if stride != "":
        classifier = IntergrowthClassifierConfig.from_dict({**classifier.to_dict(), "stride": int(float(stride))})
    if hard_threshold != "":
        classifier = IntergrowthClassifierConfig.from_dict({**classifier.to_dict(), "hard_threshold": float(hard_threshold)})
    return save_intergrowth_artifacts(
        sample_dir,
        config=classifier,
        classifier_config_path=cfg.intergrowth_classifier_path,
    )


def save_panorama_review_from_request(
    *,
    job_id: str,
    classes_json: str = "",
    config: BackendConfig | None = None,
) -> dict[str, Any]:
    cfg = (config or BackendConfig()).resolve()
    sample_dir = resolve_panorama_sample_dir(job_id, config=cfg)
    classes = json.loads(classes_json) if classes_json else None
    return save_panorama_review(sample_dir, output_root=cfg.active_learning_root, classes=classes)


def _metadata_for_artifacts(artifacts: ArtifactLike) -> dict[str, Any]:
    try:
        return json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _attach_saved_mask(artifacts: ArtifactLike, saved_mask_path: Path) -> None:
    """Copy a reviewed class-index mask beside prediction artifacts and record it in metadata."""
    target = artifacts.sample_dir / "loaded_class_index_mask.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(saved_mask_path, target)
    metadata = _metadata_for_artifacts(artifacts)
    metadata["ui_initial_mask"] = str(target)
    metadata["loaded_active_learning_mask"] = str(saved_mask_path)
    artifact_map = metadata.setdefault("artifacts", {})
    if isinstance(artifact_map, dict):
        artifact_map["ui_initial_mask"] = str(target)
    artifacts.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _class_names_from_metadata(metadata: dict[str, Any]) -> tuple[str, ...]:
    checkpoint = metadata.get("ore_checkpoint")
    if isinstance(checkpoint, dict):
        names = checkpoint.get("class_names")
        if isinstance(names, list) and names:
            return tuple(str(name) for name in names)
    return ()


def _initial_mask_path(artifacts: ArtifactLike, metadata: dict[str, Any] | None = None) -> Path:
    if metadata and metadata.get("ui_initial_mask"):
        path = Path(str(metadata["ui_initial_mask"]))
        if path.exists():
            return path
    multiclass = getattr(artifacts, "multiclass_mask_path", None)
    if multiclass is not None and Path(multiclass).exists():
        return Path(multiclass)
    return artifacts.ore_mask_path


def _optional_artifact(path: Path | None, *, config: BackendConfig) -> str | None:
    if path is None or not path.exists():
        return None
    return artifact_url(path, config=config)


def _json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def render_prediction_html(artifacts: ArtifactLike, *, config: BackendConfig) -> str:
    """Render a model review page with live mask/talc editing tools."""
    cfg = config.resolve()
    metadata = _metadata_for_artifacts(artifacts)
    class_names = _class_names_from_metadata(metadata)
    classes = ui_class_metadata(class_names)
    image_path = metadata.get("image_path", "")
    if not image_path:
        image_path = str(metadata.get("source_image_path", ""))
    if not image_path:
        image_path = ""
    raw_url = source_image_url(Path(image_path), config=cfg) if image_path else ""
    mask_path = _initial_mask_path(artifacts, metadata)
    rel_sample_dir = artifacts.sample_dir.resolve().relative_to(cfg.predictions_root).as_posix()
    initial = {
        "imagePath": str(image_path),
        "modelName": str(metadata.get("method", "unknown")),
        "rawUrl": raw_url,
        "maskUrl": artifact_url(mask_path, config=cfg),
        "overlayUrl": _optional_artifact(getattr(artifacts, "overlay_path", None), config=cfg),
        "confidenceUrl": _optional_artifact(getattr(artifacts, "ore_confidence_path", None), config=cfg),
        "probabilityUrl": _optional_artifact(getattr(artifacts, "ore_probability_path", None), config=cfg),
        "multiclassConfidenceUrl": _optional_artifact(getattr(artifacts, "multiclass_confidence_path", None), config=cfg),
        "classes": classes,
    }
    initial_json = _json_for_script(initial)
    classes_json = _json_for_script(classes)
    escaped_sample_dir = html.escape(rel_sample_dir)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Ore Mask Review Instrument</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1rem; line-height: 1.35; background: #f7f7f7; color: #222; }}
    .topbar, .panel, .tools {{ background: white; border: 1px solid #ddd; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }}
    .topbar {{ position: sticky; top: 0; z-index: 10; box-shadow: 0 2px 10px rgba(0,0,0,.06); }}
    .top-grid {{ display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: .5rem 1rem; }}
    .canvas-row {{ display: grid; grid-template-columns: repeat(3, minmax(260px, 1fr)); gap: 1rem; align-items: start; }}
    .canvas-box {{ background: #111; border-radius: 8px; padding: .5rem; overflow: auto; max-height: 70vh; }}
    canvas {{ background: #222; image-rendering: pixelated; cursor: crosshair; max-width: none; }}
    .hist-row {{ display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 1rem; }}
    .instrument-grid {{ display: grid; grid-template-columns: repeat(3, minmax(260px, 1fr)); gap: 1rem; align-items: start; }}
    .instrument-card {{ border: 1px solid #e0e0e0; border-radius: 8px; padding: .75rem; background: #fbfbfb; }}
    .instrument-card h2 {{ margin-top: 0; font-size: 1.05rem; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: .5rem 1rem; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: .35rem; }}
    .swatch {{ display: inline-block; width: 1rem; height: 1rem; border: 1px solid #555; }}
    label {{ margin-right: 1rem; white-space: nowrap; }}
    input, select, button {{ padding: .35rem; margin: .2rem; }}
    .metrics span {{ display: inline-block; min-width: 12rem; }}
    .metrics table {{ border-collapse: collapse; margin-top: .5rem; }}
    .metrics th, .metrics td {{ border: 1px solid #ddd; padding: .25rem .45rem; text-align: right; }}
    .metrics th:first-child, .metrics td:first-child {{ text-align: left; }}
    .message {{ font-weight: 600; }}
  </style>
</head>
<body>
  <div class="topbar">
    <h1>Ore mask review instrument</h1>
    <p><a href="/">Back to model/image selection</a></p>
    <div class="top-grid">
      <div><strong>Current model:</strong> <code id="currentModel"></code></div>
      <div><strong>Image name/address:</strong> <code>{html.escape(str(image_path))}</code></div>
    </div>
    <p><strong>Color → class description</strong></p>
    <div class="legend" id="legend"></div>
  </div>

  <div class="tools">
    <h2>Instrument</h2>
    <div class="instrument-grid">
      <section class="instrument-card">
        <h2>View — no new artifacts</h2>
        <p>Scale/crop all three images together only.</p>
        <button id="zoomIn" type="button">+ scale</button>
        <button id="zoomOut" type="button">- scale</button>
        <button id="cropMode" type="button">Crop visible area</button>
        <button id="fullView" type="button">Return to full view</button>
        <p class="message" id="viewStatus"></p>
      </section>
      <section class="instrument-card">
        <h2>Active learning brush</h2>
        <label>Mask class currently edited <select id="classSelect"></select></label>
        <label>Brush size <input id="brushSize" type="range" min="1" max="80" value="12"><span id="brushSizeText">12</span> px</label>
        <button id="brushAddMode" type="button">Add selected class</button>
        <button id="brushRemoveMode" type="button">Remove to background</button>
        <label><input id="eraseMode" type="checkbox"> remove brushed pixels to background</label>
        <p>Choosing class <code>background</code> also moves brushed pixels to background.</p>
      </section>
      <section class="instrument-card">
        <h2>Save active-learning mask</h2>
        <p>Saves full class-index mask and torch one-hot tensor. Non-edited classes are saved too.</p>
        <button id="restorePrediction" type="button">Restore prediction mask</button>
        <button id="saveMask" type="button">Save full one-hot mask tensor</button>
        <p class="message" id="saveStatus"></p>
      </section>
    </div>
  </div>

  <div class="panel metrics">
    <strong>Metrics for all non-zero classes:</strong>
    <div id="allClassMetrics"></div>
  </div>

  <div class="canvas-row panel">
    <div><h2>Raw image</h2><div class="canvas-box"><canvas id="rawCanvas"></canvas></div></div>
    <div><h2>Raw image + mask</h2><div class="canvas-box"><canvas id="overlayCanvas"></canvas></div></div>
    <div><h2>Mask only</h2><div class="canvas-box"><canvas id="maskCanvas"></canvas></div></div>
  </div>

  <div class="tools">
    <h2>Talc mask creation</h2>
    <label>Metric
      <select id="talcMetricSelect">
        <option value="hsv_value">HSV Value</option>
        <option value="rgb_sum">R + G + B</option>
      </select>
    </label>
    <label>Threshold <input id="talcThreshold" type="range" min="0" max="255" value="50"><span id="talcThresholdText">50</span></label>
    <button id="applyTalc" type="button">Apply talc threshold to mask</button>
    <p>Pixels with selected metric below threshold become the editable <code>talc</code> class. Use the brush to add/remove talc regions after thresholding.</p>
  </div>

  <div class="hist-row panel">
    <div><h2>Histogram: HSV Value</h2><canvas id="histV" width="512" height="180"></canvas></div>
    <div><h2>Histogram: R + G + B</h2><canvas id="histRgb" width="512" height="180"></canvas></div>
  </div>

  <form method="post" action="/accept" class="tools">
    <input type="hidden" name="sample_dir" value="{escaped_sample_dir}">
    <input type="hidden" name="label" value="ore">
    <button type="submit">Accept original prediction mask as reviewed ore correction</button>
  </form>

<script>
const INITIAL = {initial_json};
const CLASSES_JSON = {classes_json};
const classes = INITIAL.classes;
let rawImage = new Image();
let maskImage = new Image();
let sourceWidth = 0, sourceHeight = 0;
let maskData = null;
let baseMaskData = null;
let zoom = 1.0;
let crop = null;
let cropSelecting = false;
let cropStart = null;
let cropCurrent = null;
let hoverPoint = null;
let drawing = false;
const canvases = {{ raw: document.getElementById('rawCanvas'), overlay: document.getElementById('overlayCanvas'), mask: document.getElementById('maskCanvas') }};
const classByName = Object.fromEntries(classes.map(c => [c.name, c]));
const colorById = Object.fromEntries(classes.map(c => [String(c.id), c.color]));
const backgroundId = classes.find(c => c.name === 'background' || c.name === 'background_matrix')?.id ?? 0;
const talcId = classByName.talc?.id ?? 3;
const normalId = classByName.normal_ore?.id ?? -1;
const hardId = classByName.hard_ore?.id ?? -1;

function setupControls() {{
  document.getElementById('currentModel').textContent = INITIAL.modelName || 'unknown';
  const select = document.getElementById('classSelect');
  for (const cls of classes) {{
    const opt = document.createElement('option'); opt.value = cls.id; opt.textContent = `${{cls.id}} — ${{cls.name}}`; select.appendChild(opt);
  }}
  const legend = document.getElementById('legend');
  for (const cls of classes) {{
    const item = document.createElement('span'); item.className = 'legend-item';
    const sw = document.createElement('span'); sw.className = 'swatch'; sw.style.background = `rgb(${{cls.color.join(',')}})`;
    item.appendChild(sw); item.appendChild(document.createTextNode(`${{cls.id}}: ${{cls.name}}`)); legend.appendChild(item);
  }}
  document.getElementById('brushSize').addEventListener('input', e => document.getElementById('brushSizeText').textContent = e.target.value);
  document.getElementById('brushAddMode').onclick = () => {{ document.getElementById('eraseMode').checked = false; }};
  document.getElementById('brushRemoveMode').onclick = () => {{ document.getElementById('eraseMode').checked = true; }};
  document.getElementById('zoomIn').onclick = () => {{ zoom = Math.min(8, zoom * 1.25); drawAll(); }};
  document.getElementById('zoomOut').onclick = () => {{ zoom = Math.max(0.1, zoom / 1.25); drawAll(); }};
  document.getElementById('fullView').onclick = () => {{ crop = null; cropSelecting = false; cropStart = null; cropCurrent = null; drawAll(); }};
  document.getElementById('cropMode').onclick = () => {{ cropSelecting = true; cropStart = null; document.getElementById('viewStatus').textContent = 'Crop mode: click two corners on any image.'; }};
  document.getElementById('talcMetricSelect').onchange = updateTalcSliderRange;
  document.getElementById('talcThreshold').oninput = e => document.getElementById('talcThresholdText').textContent = e.target.value;
  document.getElementById('applyTalc').onclick = applyTalcThreshold;
  document.getElementById('saveMask').onclick = saveMask;
  document.getElementById('restorePrediction').onclick = restorePredictionMask;
}}

function loadImages() {{
  rawImage.onload = () => {{
    sourceWidth = rawImage.naturalWidth; sourceHeight = rawImage.naturalHeight;
    maskImage.onload = () => {{ initMaskFromImage(); computeHistograms(); drawAll(); }};
    maskImage.src = INITIAL.maskUrl;
  }};
  rawImage.src = INITIAL.rawUrl;
}}

function initMaskFromImage() {{
  const c = document.createElement('canvas'); c.width = sourceWidth; c.height = sourceHeight;
  const ctx = c.getContext('2d', {{willReadFrequently: true}});
  ctx.drawImage(maskImage, 0, 0, sourceWidth, sourceHeight);
  const data = ctx.getImageData(0, 0, sourceWidth, sourceHeight).data;
  maskData = new Uint8ClampedArray(sourceWidth * sourceHeight);
  for (let i = 0; i < maskData.length; i++) maskData[i] = data[i * 4];
  baseMaskData = new Uint8ClampedArray(maskData);
}}

function visibleRect() {{ return crop || {{x: 0, y: 0, w: sourceWidth, h: sourceHeight}}; }}
function prepareCanvas(canvas) {{ const r = visibleRect(); canvas.width = Math.max(1, Math.round(r.w * zoom)); canvas.height = Math.max(1, Math.round(r.h * zoom)); return r; }}

function drawAll() {{
  if (!maskData) return;
  const r = visibleRect();
  for (const canvas of Object.values(canvases)) prepareCanvas(canvas);
  const rawCtx = canvases.raw.getContext('2d');
  rawCtx.imageSmoothingEnabled = zoom < 1;
  rawCtx.drawImage(rawImage, r.x, r.y, r.w, r.h, 0, 0, canvases.raw.width, canvases.raw.height);

  const maskCanvasFull = colorMaskCanvas();
  const maskCtx = canvases.mask.getContext('2d'); maskCtx.imageSmoothingEnabled = false;
  maskCtx.drawImage(maskCanvasFull, r.x, r.y, r.w, r.h, 0, 0, canvases.mask.width, canvases.mask.height);

  const overCtx = canvases.overlay.getContext('2d'); overCtx.imageSmoothingEnabled = zoom < 1;
  overCtx.drawImage(rawImage, r.x, r.y, r.w, r.h, 0, 0, canvases.overlay.width, canvases.overlay.height);
  overCtx.globalAlpha = 0.45; overCtx.imageSmoothingEnabled = false;
  overCtx.drawImage(maskCanvasFull, r.x, r.y, r.w, r.h, 0, 0, canvases.overlay.width, canvases.overlay.height);
  overCtx.globalAlpha = 1.0;
  drawInteractionGuides();
  updateMetrics();
}}

function drawInteractionGuides() {{
  for (const canvas of [canvases.overlay, canvases.mask]) {{
    const ctx = canvas.getContext('2d');
    if (hoverPoint) {{
      const r = visibleRect();
      const radius = Math.max(1, parseInt(document.getElementById('brushSize').value, 10) / 2);
      const cx = ((hoverPoint.x - r.x) / r.w) * canvas.width;
      const cy = ((hoverPoint.y - r.y) / r.h) * canvas.height;
      const cr = radius / r.w * canvas.width;
      ctx.save(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx, cy, cr, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
    }}
    if (cropSelecting && cropStart && cropCurrent) {{
      const r = visibleRect();
      const sx = ((cropStart.x - r.x) / r.w) * canvas.width;
      const sy = ((cropStart.y - r.y) / r.h) * canvas.height;
      const ex = ((cropCurrent.x - r.x) / r.w) * canvas.width;
      const ey = ((cropCurrent.y - r.y) / r.h) * canvas.height;
      ctx.save(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.setLineDash([6, 4]); ctx.strokeRect(Math.min(sx, ex), Math.min(sy, ey), Math.abs(ex - sx), Math.abs(ey - sy)); ctx.restore();
    }}
  }}
}}

function colorMaskCanvas() {{
  const c = document.createElement('canvas'); c.width = sourceWidth; c.height = sourceHeight;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(sourceWidth, sourceHeight);
  for (let i = 0; i < maskData.length; i++) {{
    const color = colorById[String(maskData[i])] || [255,255,255];
    img.data[i*4] = color[0]; img.data[i*4+1] = color[1]; img.data[i*4+2] = color[2]; img.data[i*4+3] = maskData[i] === backgroundId ? 40 : 255;
  }}
  ctx.putImageData(img, 0, 0); return c;
}}

function canvasToImageXY(canvas, evt) {{
  const rect = canvas.getBoundingClientRect(); const r = visibleRect();
  const x = Math.floor(r.x + ((evt.clientX - rect.left) / rect.width) * r.w);
  const y = Math.floor(r.y + ((evt.clientY - rect.top) / rect.height) * r.h);
  return {{x: Math.max(0, Math.min(sourceWidth - 1, x)), y: Math.max(0, Math.min(sourceHeight - 1, y))}};
}}

function applyBrush(x, y) {{
  const size = parseInt(document.getElementById('brushSize').value, 10);
  const selected = parseInt(document.getElementById('classSelect').value, 10);
  const value = document.getElementById('eraseMode').checked ? backgroundId : selected;
  const radius = Math.max(1, Math.floor(size / 2));
  for (let yy = y - radius; yy <= y + radius; yy++) for (let xx = x - radius; xx <= x + radius; xx++) {{
    if (xx < 0 || yy < 0 || xx >= sourceWidth || yy >= sourceHeight) continue;
    const dx = xx - x, dy = yy - y; if (dx*dx + dy*dy <= radius*radius) maskData[yy * sourceWidth + xx] = value;
  }}
}}

function attachCanvasEvents(canvas) {{
  canvas.addEventListener('mousedown', evt => {{
    const p = canvasToImageXY(canvas, evt);
    if (cropSelecting) {{
      cropStart = p; cropCurrent = p; document.getElementById('viewStatus').textContent = 'Crop mode: drag to select area.';
      return;
    }}
    drawing = true; applyBrush(p.x, p.y); drawAll();
  }});
  canvas.addEventListener('mousemove', evt => {{ const p = canvasToImageXY(canvas, evt); hoverPoint = p; if (cropSelecting && cropStart) {{ cropCurrent = p; drawAll(); return; }} if (drawing) {{ applyBrush(p.x, p.y); }} drawAll(); }});
  canvas.addEventListener('mouseup', evt => {{ if (cropSelecting && cropStart) {{ const p = canvasToImageXY(canvas, evt); const x = Math.min(cropStart.x, p.x), y = Math.min(cropStart.y, p.y); crop = {{x, y, w: Math.max(1, Math.abs(cropStart.x - p.x)), h: Math.max(1, Math.abs(cropStart.y - p.y))}}; cropSelecting = false; cropStart = null; cropCurrent = null; document.getElementById('viewStatus').textContent = 'Cropped view.'; drawAll(); }} drawing = false; }});
  canvas.addEventListener('mouseleave', () => {{ drawing = false; hoverPoint = null; drawAll(); }});
}}

let histV = [], histRgb = [];
function computeHistograms() {{
  const c = document.createElement('canvas'); c.width = sourceWidth; c.height = sourceHeight;
  const ctx = c.getContext('2d', {{willReadFrequently: true}}); ctx.drawImage(rawImage, 0, 0);
  const data = ctx.getImageData(0,0,sourceWidth,sourceHeight).data;
  histV = new Array(256).fill(0); histRgb = new Array(766).fill(0);
  for (let i=0;i<data.length;i+=4) {{ const r=data[i], g=data[i+1], b=data[i+2]; histV[Math.max(r,g,b)]++; histRgb[r+g+b]++; }}
  drawHistogram('histV', histV); drawHistogram('histRgb', histRgb); updateTalcSliderRange();
}}
function drawHistogram(id, hist) {{
  const c = document.getElementById(id), ctx = c.getContext('2d'); ctx.clearRect(0,0,c.width,c.height);
  const axisH = 22; const plotH = c.height - axisH; const max = Math.max(...hist); ctx.fillStyle = '#4a76d1';
  for (let x=0;x<c.width;x++) {{ const start=Math.floor(x*hist.length/c.width), end=Math.floor((x+1)*hist.length/c.width); let sum=0; for(let i=start;i<end;i++) sum+=hist[i]; const h=Math.log1p(sum)/Math.log1p(max)*plotH; ctx.fillRect(x,plotH-h,1,h); }}
  ctx.strokeStyle = '#333'; ctx.beginPath(); ctx.moveTo(0, plotH + .5); ctx.lineTo(c.width, plotH + .5); ctx.stroke();
  ctx.fillStyle = '#222'; ctx.font = '12px system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.fillText('0', 0, c.height - 4);
  ctx.textAlign = 'center'; ctx.fillText(String(Math.floor((hist.length - 1) / 2)), c.width / 2, c.height - 4);
  ctx.textAlign = 'right'; ctx.fillText(String(hist.length - 1), c.width - 2, c.height - 4);
}}
function updateTalcSliderRange() {{
  const metric = document.getElementById('talcMetricSelect').value;
  const slider = document.getElementById('talcThreshold'); slider.max = metric === 'rgb_sum' ? 765 : 255;
  if (parseInt(slider.value,10) > parseInt(slider.max,10)) slider.value = slider.max;
  document.getElementById('talcThresholdText').textContent = slider.value;
}}
function applyTalcThreshold() {{
  const metric = document.getElementById('talcMetricSelect').value; const threshold = parseInt(document.getElementById('talcThreshold').value,10);
  const c = document.createElement('canvas'); c.width = sourceWidth; c.height = sourceHeight;
  const ctx = c.getContext('2d', {{willReadFrequently: true}}); ctx.drawImage(rawImage, 0, 0);
  const data = ctx.getImageData(0,0,sourceWidth,sourceHeight).data;
  for (let i=0, p=0;i<data.length;i+=4,p++) {{ const r=data[i], g=data[i+1], b=data[i+2]; const v = metric === 'rgb_sum' ? r+g+b : Math.max(r,g,b); if (v < threshold) maskData[p] = talcId; else if (maskData[p] === talcId) maskData[p] = backgroundId; }}
  drawAll();
}}
function updateMetrics() {{
  const total = Math.max(1, maskData.length);
  const counts = new Map();
  for (const v of maskData) {{ if (v !== 0) counts.set(v, (counts.get(v) || 0) + 1); }}
  const classById = Object.fromEntries(classes.map(c => [String(c.id), c.name]));
  let rows = Array.from(counts.entries()).sort((a,b)=>a[0]-b[0]).map(([id,count]) => `<tr><td>${{id}}: ${{classById[String(id)] || 'class_' + id}}</td><td>${{count}}</td><td>${{(count/total).toFixed(5)}}</td></tr>`).join('');
  if (!rows) rows = '<tr><td colspan="3">no non-zero classes</td></tr>';
  document.getElementById('allClassMetrics').innerHTML = `<table><tr><th>Class</th><th>Pixels</th><th>Fraction</th></tr>${{rows}}</table>`;
}}
function restorePredictionMask() {{
  if (!baseMaskData) return;
  maskData = new Uint8ClampedArray(baseMaskData);
  drawAll();
}}
function maskDataUrl() {{
  const c = document.createElement('canvas'); c.width = sourceWidth; c.height = sourceHeight;
  const ctx = c.getContext('2d'); const img = ctx.createImageData(sourceWidth, sourceHeight);
  for (let i=0;i<maskData.length;i++) {{ const v = maskData[i]; img.data[i*4]=v; img.data[i*4+1]=v; img.data[i*4+2]=v; img.data[i*4+3]=255; }}
  ctx.putImageData(img,0,0); return c.toDataURL('image/png');
}}
async function saveMask() {{
  const status = document.getElementById('saveStatus'); status.textContent = 'Saving...';
  const body = new URLSearchParams(); body.set('image_path', INITIAL.imagePath); body.set('mask_data_url', maskDataUrl()); body.set('classes_json', JSON.stringify(classes));
  const response = await fetch('/save-mask', {{method: 'POST', headers: {{'Content-Type': 'application/x-www-form-urlencoded'}}, body}});
  const result = await response.json();
  status.textContent = response.ok ? `Saved: ${{result.one_hot_tensor}}` : `Error: ${{result.error}}`;
}}
setupControls(); Object.values(canvases).forEach(attachCanvasEvents); loadImages();
</script>
</body>
</html>"""


def render_panorama_review_html(job_id: str, *, config: BackendConfig) -> str:
    """Render a tile-based panorama review page for a completed job."""
    cfg = config.resolve()
    status = get_panorama_job_status(job_id, config=cfg)
    if status.get("status") != "completed":
        status_json = _json_for_script(status)
        return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Panorama job status</title></head>
<body>
  <h1>Panorama job status</h1>
  <p><a href="/active-learning">Back to Active Learning</a></p>
  <pre id="status"></pre>
  <script>
    const STATUS = {status_json};
    document.getElementById('status').textContent = JSON.stringify(STATUS, null, 2);
  </script>
</body>
</html>"""

    sample_dir = resolve_panorama_sample_dir(job_id, config=cfg)
    metadata = read_panorama_metadata(sample_dir)
    class_names = _class_names_from_metadata(metadata)
    classes = ui_class_metadata(class_names)
    initial = {
        "jobId": job_id,
        "imagePath": metadata.get("image_path", ""),
        "imageWidth": int(metadata["image_width"]),
        "imageHeight": int(metadata["image_height"]),
        "classes": classes,
        "status": status,
    }
    initial_json = _json_for_script(initial)
    classes_json = _json_for_script(classes)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Panorama Mask Review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1rem; line-height: 1.35; background: #f7f7f7; color: #222; }}
    .topbar, .panel, .tools {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }}
    .topbar {{ position: sticky; top: 0; z-index: 10; box-shadow: 0 2px 10px rgba(0,0,0,.06); }}
    .canvas-row {{ display: grid; grid-template-columns: repeat(3, minmax(280px, 1fr)); gap: 1rem; align-items: start; }}
    .canvas-box {{ background: #111; border-radius: 8px; padding: .5rem; overflow: auto; }}
    canvas {{ background: #222; image-rendering: pixelated; cursor: crosshair; max-width: 100%; }}
    label {{ margin-right: 1rem; white-space: nowrap; }}
    input, select, button {{ padding: .35rem; margin: .2rem; }}
    table {{ border-collapse: collapse; margin-top: .5rem; }}
    th, td {{ border: 1px solid #ddd; padding: .25rem .45rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: .5rem 1rem; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: .35rem; }}
    .swatch {{ display: inline-block; width: 1rem; height: 1rem; border: 1px solid #555; }}
    .progress-line span {{ display: inline-block; min-width: 9rem; }}
    .message {{ font-weight: 600; }}
  </style>
</head>
<body>
  <div class="topbar">
    <h1>Panorama mask review</h1>
    <p><a href="/active-learning">Back to Active Learning</a></p>
    <p><strong>Image:</strong> <code>{html.escape(str(metadata.get("image_path", "")))}</code></p>
    <p class="progress-line">
      <span>job: <b>{html.escape(job_id)}</b></span>
      <span>size: <b>{int(metadata["image_width"])} x {int(metadata["image_height"])}</b></span>
      <span>tiles: <b>{int(metadata["total_tiles"])}</b></span>
    </p>
    <div class="legend" id="legend"></div>
  </div>

  <div class="tools">
    <button id="zoomIn" type="button">+ scale</button>
    <button id="zoomOut" type="button">- scale</button>
    <button id="left" type="button">left</button>
    <button id="right" type="button">right</button>
    <button id="up" type="button">up</button>
    <button id="down" type="button">down</button>
    <button id="fullView" type="button">full view</button>
    <button id="cropMode" type="button">Crop area</button>
    <button id="runIntergrowth" type="button">Run intergrowth classification</button>
    <label>View layer
      <select id="viewMode">
        <option value="review">editable prediction mask</option>
        <option value="intergrowth">intergrowth mask</option>
        <option value="intergrowth_score">intergrowth score</option>
        <option value="intergrowth_confidence">intergrowth confidence</option>
      </select>
    </label>
    <label>Class <select id="classSelect"></select></label>
    <label>Brush <input id="brushSize" type="range" min="2" max="160" value="24"><span id="brushSizeText">24</span> px</label>
    <button id="restorePrediction" type="button">Restore prediction mask</button>
    <button id="saveReview" type="button">Save reviewed panorama mask</button>
    <span class="message" id="message"></span>
  </div>

  <div class="panel">
    <p id="viewport"></p>
    <div class="canvas-row">
      <div><h2>Raw image</h2><div class="canvas-box"><canvas id="rawCanvas"></canvas></div></div>
      <div><h2>Raw image + mask</h2><div class="canvas-box"><canvas id="overlayCanvas"></canvas></div></div>
      <div><h2>Mask only</h2><div class="canvas-box"><canvas id="maskCanvas"></canvas></div></div>
    </div>
    <h2>Metrics</h2>
    <div id="metrics"></div>
  </div>

<script>
const INITIAL = {initial_json};
const CLASSES_JSON = {classes_json};
const classes = INITIAL.classes;
const jobId = INITIAL.jobId;
let view = {{
  x: 0,
  y: 0,
  w: INITIAL.imageWidth,
  h: INITIAL.imageHeight
}};
let cropSelecting = false, cropStart = null, cropCurrent = null, hoverPoint = null;
const displayW = 520;
const canvases = {{
  raw: document.getElementById('rawCanvas'),
  overlay: document.getElementById('overlayCanvas'),
  mask: document.getElementById('maskCanvas')
}};
const canvasImages = {{raw: null, overlay: null, mask: null}};
const canvasImageKeys = {{raw: '', overlay: '', mask: ''}};
let tileRevision = 0;

function setup() {{
  const select = document.getElementById('classSelect');
  const legend = document.getElementById('legend');
  for (const cls of classes) {{
    const opt = document.createElement('option');
    opt.value = cls.id; opt.textContent = `${{cls.id}} - ${{cls.name}}`; select.appendChild(opt);
    const item = document.createElement('span'); item.className = 'legend-item';
    const sw = document.createElement('span'); sw.className = 'swatch'; sw.style.background = `rgb(${{cls.color.join(',')}})`;
    item.appendChild(sw); item.appendChild(document.createTextNode(`${{cls.id}}: ${{cls.name}}`)); legend.appendChild(item);
  }}
  document.getElementById('brushSize').addEventListener('input', e => document.getElementById('brushSizeText').textContent = e.target.value);
  document.getElementById('zoomIn').onclick = () => zoom(1 / 1.25);
  document.getElementById('zoomOut').onclick = () => zoom(1.25);
  document.getElementById('fullView').onclick = () => {{ view = {{x: 0, y: 0, w: INITIAL.imageWidth, h: INITIAL.imageHeight}}; cropSelecting = false; cropStart = null; cropCurrent = null; drawAll(); }};
  document.getElementById('cropMode').onclick = () => {{ cropSelecting = true; cropStart = null; cropCurrent = null; message.textContent = 'drag crop area on any view'; }};
  document.getElementById('left').onclick = () => pan(-0.25, 0);
  document.getElementById('right').onclick = () => pan(0.25, 0);
  document.getElementById('up').onclick = () => pan(0, -0.25);
  document.getElementById('down').onclick = () => pan(0, 0.25);
  document.getElementById('viewMode').onchange = drawAll;
  document.getElementById('runIntergrowth').onclick = runIntergrowth;
  document.getElementById('restorePrediction').onclick = restorePrediction;
  document.getElementById('saveReview').onclick = saveReview;
  Object.values(canvases).forEach(attachCanvasEvents);
  drawAll();
}}

function clampView() {{
  view.w = Math.max(64, Math.min(INITIAL.imageWidth, view.w));
  view.h = Math.max(64, Math.min(INITIAL.imageHeight, view.h));
  view.x = Math.max(0, Math.min(INITIAL.imageWidth - view.w, view.x));
  view.y = Math.max(0, Math.min(INITIAL.imageHeight - view.h, view.y));
}}
function pan(dx, dy) {{ view.x += view.w * dx; view.y += view.h * dy; clampView(); drawAll(); }}
function zoom(factor) {{
  const cx = view.x + view.w / 2, cy = view.y + view.h / 2;
  view.w *= factor; view.h *= factor;
  view.x = cx - view.w / 2; view.y = cy - view.h / 2;
  clampView(); drawAll();
}}
function canvasSize() {{
  const h = Math.max(1, Math.round(displayW * view.h / view.w));
  for (const canvas of Object.values(canvases)) {{ canvas.width = displayW; canvas.height = h; }}
}}
function tileUrl(layer) {{
  const params = new URLSearchParams({{
    layer,
    x: String(Math.round(view.x)),
    y: String(Math.round(view.y)),
    width: String(Math.round(view.w)),
    height: String(Math.round(view.h)),
    output_width: String(displayW),
    output_height: String(Math.max(1, Math.round(displayW * view.h / view.w))),
    t: String(tileRevision)
  }});
  return `/jobs/${{jobId}}/tile?${{params.toString()}}`;
}}
function panelLayer(panel) {{
  const mode = document.getElementById('viewMode').value;
  if (panel === 'raw') return 'raw';
  if (mode === 'review') return panel === 'overlay' ? 'overlay' : 'mask';
  if (mode === 'intergrowth') return panel === 'overlay' ? 'intergrowth_overlay' : 'intergrowth_mask';
  if (mode === 'intergrowth_score') return 'intergrowth_score';
  if (mode === 'intergrowth_confidence') return 'intergrowth_confidence';
  return panel === 'overlay' ? 'overlay' : 'mask';
}}
function metricLayer() {{
  return document.getElementById('viewMode').value === 'review' ? 'prediction' : 'intergrowth';
}}
function drawLayer(layer, canvasKey) {{
  const url = tileUrl(layer);
  canvasImageKeys[canvasKey] = url;
  canvasImages[canvasKey] = null;
  renderCanvas(canvasKey);
  const img = new Image();
  img.onload = () => {{
    if (canvasImageKeys[canvasKey] !== url) return;
    canvasImages[canvasKey] = img;
    renderCanvas(canvasKey);
  }};
  img.src = url;
}}
function drawAll() {{
  canvasSize();
  document.getElementById('viewport').textContent = `visible x=${{Math.round(view.x)}}, y=${{Math.round(view.y)}}, w=${{Math.round(view.w)}}, h=${{Math.round(view.h)}}`;
  drawLayer(panelLayer('raw'), 'raw');
  drawLayer(panelLayer('overlay'), 'overlay');
  drawLayer(panelLayer('mask'), 'mask');
  updateMetrics();
}}
function renderCanvas(canvasKey) {{
  const canvas = canvases[canvasKey];
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const img = canvasImages[canvasKey];
  if (img) ctx.drawImage(img, 0, 0);
  drawInteractionGuides(ctx, canvas);
}}
function redrawGuides() {{
  for (const key of Object.keys(canvases)) renderCanvas(key);
}}
function drawInteractionGuides(ctx, canvas) {{
  if (hoverPoint) {{
    const cx = ((hoverPoint.x - view.x) / view.w) * canvas.width;
    const cy = ((hoverPoint.y - view.y) / view.h) * canvas.height;
    const radius = parseInt(document.getElementById('brushSize').value, 10) / 2 / view.w * canvas.width;
    ctx.save(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
  }}
  if (cropSelecting && cropStart && cropCurrent) {{
    const sx = ((cropStart.x - view.x) / view.w) * canvas.width, sy = ((cropStart.y - view.y) / view.h) * canvas.height;
    const ex = ((cropCurrent.x - view.x) / view.w) * canvas.width, ey = ((cropCurrent.y - view.y) / view.h) * canvas.height;
    ctx.save(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.setLineDash([6,4]); ctx.strokeRect(Math.min(sx,ex), Math.min(sy,ey), Math.abs(ex-sx), Math.abs(ey-sy)); ctx.restore();
  }}
}}
function eventToImageXY(canvas, event) {{
  const rect = canvas.getBoundingClientRect();
  return {{
    x: Math.round(view.x + ((event.clientX - rect.left) / rect.width) * view.w),
    y: Math.round(view.y + ((event.clientY - rect.top) / rect.height) * view.h)
  }};
}}
function attachCanvasEvents(canvas) {{
  canvas.addEventListener('mousedown', event => {{
    const p = eventToImageXY(canvas, event);
    if (cropSelecting) {{ cropStart = p; cropCurrent = p; redrawGuides(); return; }}
    brushAtPoint(p);
  }});
  canvas.addEventListener('mousemove', event => {{
    const p = eventToImageXY(canvas, event); hoverPoint = p;
    if (cropSelecting && cropStart) {{ cropCurrent = p; redrawGuides(); return; }}
    redrawGuides();
  }});
  canvas.addEventListener('mouseup', event => {{
    if (!cropSelecting || !cropStart) return;
    const p = eventToImageXY(canvas, event);
    const x = Math.min(cropStart.x, p.x), y = Math.min(cropStart.y, p.y);
    view = {{x, y, w: Math.max(1, Math.abs(cropStart.x - p.x)), h: Math.max(1, Math.abs(cropStart.y - p.y))}};
    cropSelecting = false; cropStart = null; cropCurrent = null; clampView(); drawAll();
  }});
  canvas.addEventListener('mouseleave', () => {{ hoverPoint = null; redrawGuides(); }});
}}
async function brushAtEvent(event) {{
  const p = eventToImageXY(event.currentTarget, event);
  await brushAtPoint(p);
}}
async function brushAtPoint(p) {{
  if (document.getElementById('viewMode').value !== 'review') {{
    document.getElementById('message').textContent = 'switch to editable prediction mask before brushing';
    return;
  }}
  const body = new URLSearchParams();
  body.set('x', String(p.x)); body.set('y', String(p.y));
  body.set('radius', document.getElementById('brushSize').value);
  body.set('class_id', document.getElementById('classSelect').value);
  const response = await fetch(`/jobs/${{jobId}}/brush`, {{method: 'POST', headers: {{'Content-Type': 'application/x-www-form-urlencoded'}}, body}});
  const result = await response.json();
  document.getElementById('message').textContent = response.ok ? `patch saved at ${{p.x}}, ${{p.y}}` : `error: ${{result.error}}`;
  if (response.ok) tileRevision += 1;
  drawAll();
}}
async function restorePrediction() {{
  const response = await fetch(`/jobs/${{jobId}}/restore`, {{method: 'POST'}});
  const result = await response.json();
  document.getElementById('message').textContent = response.ok ? 'prediction mask restored' : `error: ${{result.error}}`;
  if (response.ok) tileRevision += 1;
  drawAll();
}}
async function runIntergrowth() {{
  const response = await fetch(`/jobs/${{jobId}}/intergrowth`, {{method: 'POST'}});
  const result = await response.json();
  document.getElementById('message').textContent = response.ok ? `intergrowth: ${{result.area_metrics?.image_label || 'done'}}` : `error: ${{result.error}}`;
  if (response.ok) {{
    document.getElementById('viewMode').value = 'intergrowth';
    tileRevision += 1;
  }}
  drawAll();
}}
async function updateMetrics() {{
  const p = new URLSearchParams({{layer: metricLayer(), x: Math.round(view.x), y: Math.round(view.y), width: Math.round(view.w), height: Math.round(view.h)}});
  const full = await (await fetch(`/jobs/${{jobId}}/metrics?layer=${{metricLayer()}}`)).json();
  const visible = await (await fetch(`/jobs/${{jobId}}/metrics?${{p.toString()}}`)).json();
  metrics.innerHTML = metricTable('Full image', full) + metricTable('Visible crop', visible);
}}
function metricTable(title, data) {{
  let rows = data.classes.map(c => `<tr><td>${{c.id}}: ${{c.name}}</td><td>${{c.pixels}}</td><td>${{Number(c.fraction).toFixed(5)}}</td></tr>`).join('');
  if (!rows) rows = '<tr><td colspan="3">no non-zero classes</td></tr>';
  return `<h3>${{title}}</h3><table><tr><th>Class</th><th>Pixels</th><th>Fraction</th></tr>${{rows}}</table>`;
}}
async function saveReview() {{
  const message = document.getElementById('message'); message.textContent = 'saving review...';
  const body = new URLSearchParams(); body.set('classes_json', JSON.stringify(classes));
  const response = await fetch(`/jobs/${{jobId}}/save-review`, {{method: 'POST', headers: {{'Content-Type': 'application/x-www-form-urlencoded'}}, body}});
  const result = await response.json();
  message.textContent = response.ok ? `saved: ${{result.metadata_path}}` : `error: ${{result.error}}`;
}}
setup();
</script>
</body>
</html>"""


def _legacy_render_mixed_index_html(
    images: Iterable[Path],
    *,
    default_threshold: int = 90,
    message: str = "",
    config: BackendConfig | None = None,
    saved_masks: Iterable[Path] | None = None,
) -> str:
    """Render the legacy mixed model/image selection UI."""
    cfg = (config or BackendConfig()).resolve()
    options = "\n".join(
        f'<option value="{html.escape(str(path))}">{html.escape(str(path))}</option>' for path in images
    )
    saved_mask_options = "\n".join(
        f'<option value="{html.escape(str(path))}">{html.escape(str(path))}</option>'
        for path in (saved_masks if saved_masks is not None else list_saved_class_index_masks(cfg.active_learning_root, limit=500))
    )
    escaped_message = html.escape(message)
    binary_path = html.escape(str(cfg.binary_model_path))
    ore_path = html.escape(str(cfg.ore_model_path))
    panorama_tile_size = int(cfg.panorama_tile_size)
    panorama_overlap = int(cfg.panorama_overlap)
    panorama_batch_size = int(cfg.panorama_batch_size)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Ore Detection UI</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; line-height: 1.45; }}
    label {{ display: block; margin-top: 1rem; font-weight: 600; }}
    input, select, button {{ padding: 0.4rem; min-width: 18rem; }}
    .note, .dropzone {{ background: #f6f8fa; padding: 1rem; border-radius: 8px; border: 1px solid #ddd; }}
    .dropzone {{ border-style: dashed; margin-top: 1rem; }}
  </style>
</head>
<body>
  <h1>Ore Detection UI</h1>
  <p class="note">Use trained binary/ore models or the HSV dummy baseline, review predictions, create a talc mask from HSV Value or R+G+B threshold, brush-edit classes, then save a one-hot tensor for active learning.</p>
  <p><strong>{escaped_message}</strong></p>

  <section class="note">
    <h2>Panorama inference job</h2>
    <form id="panoramaForm">
      <label>Panorama image path/name</label>
      <input id="panoramaImagePath" name="image_path" list="imageOptions" value="" placeholder="datasets/baseline/panoramas/...jpg">
      <label>Binary model checkpoint</label>
      <input name="binary_model_path" value="{binary_path}">
      <label>Ore segmentation checkpoint</label>
      <input name="ore_model_path" value="{ore_path}">
      <label><input name="include_ore_model" type="checkbox"> run ore segmentation after binary outline</label>
      <label>Device</label>
      <select name="device"><option value="auto">auto</option><option value="cuda">cuda</option><option value="cpu">cpu</option></select>
      <label>Binary threshold</label>
      <input name="binary_threshold" type="number" min="0" max="1" step="0.01" value="0.5">
      <label>Tile size</label>
      <input name="tile_size" type="number" min="64" step="32" value="{panorama_tile_size}">
      <label>Overlap</label>
      <input name="overlap" type="number" min="0" step="16" value="{panorama_overlap}">
      <label>Batch size</label>
      <input name="batch_size" type="number" min="1" step="1" value="{panorama_batch_size}">
      <p><button type="submit">Start panorama job</button> <button type="button" id="cancelPanoramaJob">Cancel job</button></p>
    </form>
    <progress id="panoramaProgress" value="0" max="1" style="width:100%;"></progress>
    <pre id="panoramaStatus"></pre>
    <p id="panoramaReviewLink"></p>
  </section>

  <form method="post" action="/predict" id="predictForm">
    <label>Current model</label>
    <select name="model_kind">
      <option value="trained_binary_ore" selected>trained binary + ore segmentation models</option>
      <option value="hsv_dummy">HSV Value dummy baseline</option>
    </select>
    <label>Image path/name</label>
    <input id="imagePath" name="image_path" list="imageOptions" value="" placeholder="datasets/.../image.jpg">
    <datalist id="imageOptions">{options}</datalist>
    <label>Or choose from indexed images</label>
    <select id="imageSelect"><option value="">-- choose image --</option>{options}</select>
    <div class="dropzone" id="dropzone">Drag and drop an image here, or <input type="file" id="fileInput" accept="image/*"></div>
    <p id="uploadStatus"></p>

    <label>Reload previously saved active-learning mask</label>
    <select name="saved_mask_path">
      <option value="">-- start from model prediction --</option>
      {saved_mask_options}
    </select>
    <p class="note">Optional: load a saved <code>class_index_mask.png</code> from <code>data_work/active_learning_masks</code> as the editable starting mask.</p>

    <label>Binary model checkpoint</label>
    <input name="binary_model_path" value="{binary_path}">
    <label>Ore segmentation checkpoint</label>
    <input name="ore_model_path" value="{ore_path}">
    <label>Device</label>
    <select name="device"><option value="auto">auto</option><option value="cuda">cuda</option><option value="cpu">cpu</option></select>
    <label>Binary threshold</label>
    <input name="binary_threshold" type="number" min="0" max="1" step="0.01" value="0.5">

    <fieldset>
      <legend>HSV dummy fallback settings</legend>
      <label>HSV Value threshold</label>
      <input name="value_threshold" type="number" min="0" max="255" value="{default_threshold}">
      <label>Foreground mode</label>
      <select name="foreground"><option value="bright">bright ore-like regions</option><option value="dark">dark/talc-like regions</option></select>
      <label><input name="standardize" type="checkbox"> apply standard scaling before HSV threshold</label>
    </fieldset>
    <p><button type="submit">Create prediction / open editor</button></p>
  </form>
<script>
const imageSelect = document.getElementById('imageSelect');
const imagePath = document.getElementById('imagePath');
const panoramaImagePath = document.getElementById('panoramaImagePath');
imageSelect.addEventListener('change', () => {{ if (imageSelect.value) {{ imagePath.value = imageSelect.value; panoramaImagePath.value = imageSelect.value; }} }});
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
function uploadFile(file) {{
  const reader = new FileReader();
  reader.onload = async () => {{
    document.getElementById('uploadStatus').textContent = 'Uploading dropped image...';
    const body = new URLSearchParams(); body.set('file_name', file.name); body.set('image_data_url', reader.result);
    const response = await fetch('/upload-image', {{method:'POST', headers:{{'Content-Type':'application/x-www-form-urlencoded'}}, body}});
    const result = await response.json();
    if (response.ok) {{ imagePath.value = result.relative_path; document.getElementById('uploadStatus').textContent = 'Uploaded: ' + result.relative_path; }}
    else {{ document.getElementById('uploadStatus').textContent = 'Upload error: ' + result.error; }}
  }};
  reader.readAsDataURL(file);
}}
fileInput.addEventListener('change', () => {{ if (fileInput.files[0]) uploadFile(fileInput.files[0]); }});
dropzone.addEventListener('dragover', event => {{ event.preventDefault(); dropzone.style.background = '#e8f0fe'; }});
dropzone.addEventListener('dragleave', () => {{ dropzone.style.background = '#f6f8fa'; }});
dropzone.addEventListener('drop', event => {{ event.preventDefault(); dropzone.style.background = '#f6f8fa'; if (event.dataTransfer.files[0]) uploadFile(event.dataTransfer.files[0]); }});
let currentPanoramaJobId = null;
let panoramaTimer = null;
const panoramaForm = document.getElementById('panoramaForm');
const panoramaProgress = document.getElementById('panoramaProgress');
const panoramaStatus = document.getElementById('panoramaStatus');
const panoramaReviewLink = document.getElementById('panoramaReviewLink');
async function pollPanoramaJob(jobId) {{
  const response = await fetch(`/jobs/${{jobId}}`);
  const status = await response.json();
  const processed = status.processed_tiles || 0;
  const total = status.total_tiles || 0;
  panoramaProgress.max = Math.max(1, total);
  panoramaProgress.value = processed;
  const elapsed = Number(status.elapsed_sec || 0).toFixed(1);
  const eta = status.eta_sec === null || status.eta_sec === undefined ? 'unknown' : Number(status.eta_sec).toFixed(1) + 's';
  panoramaStatus.textContent = `status=${{status.status}} phase=${{status.phase}} tiles=${{processed}}/${{total}} elapsed=${{elapsed}}s eta=${{eta}} tile/s=${{Number(status.tile_per_sec || 0).toFixed(2)}} batch/s=${{Number(status.batch_per_sec || 0).toFixed(2)}}\\n` + JSON.stringify(status.timings || {{}}, null, 2);
  if (status.status === 'completed') {{
    clearInterval(panoramaTimer);
    panoramaReviewLink.innerHTML = `<a href="/active-learning?job_id=${{jobId}}">Open panorama mask review</a>`;
  }}
  if (status.status === 'failed' || status.status === 'cancelled') clearInterval(panoramaTimer);
}}
panoramaForm.addEventListener('submit', async event => {{
  event.preventDefault();
  const body = new URLSearchParams(new FormData(panoramaForm));
  const response = await fetch('/jobs/panorama-predict', {{method:'POST', headers:{{'Content-Type':'application/x-www-form-urlencoded'}}, body}});
  const status = await response.json();
  if (!response.ok) {{ panoramaStatus.textContent = 'Error: ' + status.error; return; }}
  currentPanoramaJobId = status.job_id;
  panoramaReviewLink.textContent = '';
  if (panoramaTimer) clearInterval(panoramaTimer);
  await pollPanoramaJob(currentPanoramaJobId);
  panoramaTimer = setInterval(() => pollPanoramaJob(currentPanoramaJobId), 1000);
}});
document.getElementById('cancelPanoramaJob').addEventListener('click', async () => {{
  if (!currentPanoramaJobId) return;
  await fetch(`/jobs/${{currentPanoramaJobId}}/cancel`, {{method:'POST'}});
  await pollPanoramaJob(currentPanoramaJobId);
}});
</script>
</body>
</html>"""


def _render_image_options(images: Iterable[Path]) -> str:
    return "\n".join(
        f'<option value="{html.escape(str(path))}">{html.escape(str(path))}</option>' for path in images
    )


def render_index_html(
    images: Iterable[Path],
    *,
    default_threshold: int = 90,
    message: str = "",
    config: BackendConfig | None = None,
    saved_masks: Iterable[Path] | None = None,
) -> str:
    """Render the start page with the two main UI workflows."""
    escaped_message = html.escape(message)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Ore Detection</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 980px; margin: 2rem auto; line-height: 1.45; color: #222; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 1rem; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem; background: #fff; }}
    .card h2 {{ margin-top: 0; }}
    a.button {{ display: inline-block; padding: .55rem .8rem; border: 1px solid #333; border-radius: 6px; color: #111; text-decoration: none; }}
    .message {{ font-weight: 600; }}
  </style>
</head>
<body>
  <h1>Ore Detection</h1>
  <p class="message">{escaped_message}</p>
  <div class="grid">
    <section class="card">
      <h2>Inference</h2>
      <p>Run one selected model, inspect prediction, crop/zoom the same region across raw, overlay, and mask panels, and review class-area metrics.</p>
      <p><a class="button" href="/inference">Open Inference</a></p>
    </section>
    <section class="card">
      <h2>Active Learning</h2>
      <p>Run prediction for annotation, edit masks with brush and talc tools, restore the original prediction, and save reviewed masks.</p>
      <p><a class="button" href="/active-learning">Open Active Learning</a></p>
    </section>
  </div>
</body>
</html>"""


def render_inference_html(
    images: Iterable[Path],
    *,
    message: str = "",
    config: BackendConfig | None = None,
) -> str:
    """Render prediction-only panorama inference UI."""
    cfg = (config or BackendConfig()).resolve()
    options = _render_image_options(images)
    binary_path = html.escape(str(cfg.binary_model_path))
    ore_path = html.escape(str(cfg.ore_model_path))
    escaped_message = html.escape(message)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Ore Inference</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1rem; line-height: 1.35; background: #f7f7f7; color: #222; }}
    .topbar, .panel {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }}
    .canvas-row {{ display: grid; grid-template-columns: repeat(3, minmax(280px, 1fr)); gap: 1rem; }}
    .canvas-box {{ background: #111; border-radius: 8px; padding: .5rem; overflow: auto; }}
    canvas {{ background: #222; image-rendering: pixelated; cursor: crosshair; max-width: 100%; }}
    label {{ display: inline-block; margin: .25rem .75rem .25rem 0; }}
    input, select, button {{ padding: .35rem; margin: .2rem; }}
    progress {{ width: 100%; }}
    table {{ border-collapse: collapse; margin-top: .5rem; }}
    th, td {{ border: 1px solid #ddd; padding: .25rem .45rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .message {{ font-weight: 600; }}
  </style>
</head>
<body>
  <div class="topbar">
    <h1>Inference</h1>
    <p><a href="/">Start</a> | <a href="/active-learning">Active Learning</a></p>
    <p class="message">{escaped_message}</p>
    <form id="inferenceForm">
      <label>Image <input id="imagePath" name="image_path" list="imageOptions" placeholder="datasets/.../image.jpg"></label>
      <datalist id="imageOptions">{options}</datalist>
      <label>Choose indexed image <select id="imageSelect"><option value="">-- choose image --</option>{options}</select></label>
      <label>Model
        <select name="model_kind" id="modelKind">
          <option value="binary">binary segmentation</option>
          <option value="ore">ore segmentation</option>
        </select>
      </label>
      <label>Binary checkpoint <input name="binary_model_path" value="{binary_path}"></label>
      <label>Ore checkpoint <input name="ore_model_path" value="{ore_path}"></label>
      <label>Device <select name="device"><option value="auto">auto</option><option value="cuda">cuda</option><option value="cpu">cpu</option></select></label>
      <label>Binary threshold <input name="binary_threshold" type="number" min="0" max="1" step="0.01" value="0.5"></label>
      <label>Tile size <input name="tile_size" type="number" min="64" step="32" value="{int(cfg.panorama_tile_size)}"></label>
      <label>Overlap <input name="overlap" type="number" min="0" step="16" value="{int(cfg.panorama_overlap)}"></label>
      <label>Batch size <input name="batch_size" type="number" min="1" step="1" value="{int(cfg.panorama_batch_size)}"></label>
      <button type="submit">Run prediction</button>
      <button type="button" id="cancelJob">Cancel</button>
      <button type="button" id="selectNewImage">Select new image</button>
      <button type="button" id="selectNewModel">Select new model prediction</button>
    </form>
  </div>

  <div class="panel">
    <progress id="progress" value="0" max="1"></progress>
    <pre id="status"></pre>
  </div>

  <div class="panel" id="viewer" style="display:none;">
    <p>
      <button id="zoomIn" type="button">+ scale</button>
      <button id="zoomOut" type="button">- scale</button>
      <button id="cropMode" type="button">Crop area</button>
      <button id="fullView" type="button">Full image</button>
      <button id="runIntergrowth" type="button">Run intergrowth classification</button>
      <label>View layer
        <select id="viewMode">
          <option value="prediction">prediction mask</option>
          <option value="intergrowth">intergrowth mask</option>
          <option value="intergrowth_score">intergrowth score</option>
          <option value="intergrowth_confidence">intergrowth confidence</option>
        </select>
      </label>
      <span id="viewportText"></span>
    </p>
    <div class="canvas-row">
      <div><h2>Raw image</h2><div class="canvas-box"><canvas id="rawCanvas"></canvas></div></div>
      <div><h2>Raw image + mask</h2><div class="canvas-box"><canvas id="overlayCanvas"></canvas></div></div>
      <div><h2>Mask only</h2><div class="canvas-box"><canvas id="maskCanvas"></canvas></div></div>
    </div>
    <h2>Metrics</h2>
    <div id="metrics"></div>
  </div>

<script>
let currentJobId = null, timer = null, statusData = null;
let view = {{x:0,y:0,w:1,h:1}};
let cropSelecting = false, cropStart = null, cropCurrent = null;
const displayW = 520;
const canvases = {{raw: rawCanvas, overlay: overlayCanvas, mask: maskCanvas}};
imageSelect.addEventListener('change', () => {{ if (imageSelect.value) imagePath.value = imageSelect.value; }});
selectNewImage.onclick = () => imagePath.focus();
selectNewModel.onclick = () => modelKind.focus();
inferenceForm.addEventListener('submit', async event => {{
  event.preventDefault();
  const body = new URLSearchParams(new FormData(inferenceForm));
  const response = await fetch('/jobs/panorama-predict', {{method:'POST', headers:{{'Content-Type':'application/x-www-form-urlencoded'}}, body}});
  const result = await response.json();
  if (!response.ok) {{ status.textContent = 'Error: ' + result.error; return; }}
  currentJobId = result.job_id; viewer.style.display = 'none';
  if (timer) clearInterval(timer);
  await pollJob();
  timer = setInterval(pollJob, 1000);
}});
cancelJob.onclick = async () => {{ if (currentJobId) await fetch(`/jobs/${{currentJobId}}/cancel`, {{method:'POST'}}); }};
async function pollJob() {{
  const response = await fetch(`/jobs/${{currentJobId}}`);
  statusData = await response.json();
  progress.max = Math.max(1, statusData.total_tiles || 0);
  progress.value = statusData.processed_tiles || 0;
  const eta = statusData.eta_sec == null ? 'unknown' : Number(statusData.eta_sec).toFixed(1) + 's';
  status.textContent = `status=${{statusData.status}} phase=${{statusData.phase}} model=${{statusData.model_kind}} tiles=${{statusData.processed_tiles||0}}/${{statusData.total_tiles||0}} elapsed=${{Number(statusData.elapsed_sec||0).toFixed(1)}}s eta=${{eta}} tile/s=${{Number(statusData.tile_per_sec||0).toFixed(2)}} batch/s=${{Number(statusData.batch_per_sec||0).toFixed(2)}}`;
  if (statusData.status === 'completed') {{
    clearInterval(timer);
    view = {{x:0, y:0, w:statusData.image_width || 1, h:statusData.image_height || 1}};
    viewer.style.display = 'block';
    drawAll();
  }}
  if (statusData.status === 'failed' || statusData.status === 'cancelled') clearInterval(timer);
}}
function canvasSize() {{ const h = Math.max(1, Math.round(displayW * view.h / view.w)); for (const c of Object.values(canvases)) {{ c.width=displayW; c.height=h; }} }}
function tileUrl(layer) {{
  const h = Math.max(1, Math.round(displayW * view.h / view.w));
  const p = new URLSearchParams({{layer, x:Math.round(view.x), y:Math.round(view.y), width:Math.round(view.w), height:Math.round(view.h), output_width:displayW, output_height:h, t:Date.now()}});
  return `/jobs/${{currentJobId}}/tile?${{p.toString()}}`;
}}
function drawLayer(layer, canvas) {{ const img = new Image(); img.onload = () => {{ const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(img,0,0); drawCropBorder(ctx, canvas); }}; img.src = tileUrl(layer); }}
function panelLayer(panel) {{
  const mode = viewMode.value;
  if (panel === 'raw') return 'raw';
  if (mode === 'prediction') return panel === 'overlay' ? 'overlay' : 'mask';
  if (mode === 'intergrowth') return panel === 'overlay' ? 'intergrowth_overlay' : 'intergrowth_mask';
  if (mode === 'intergrowth_score') return 'intergrowth_score';
  if (mode === 'intergrowth_confidence') return 'intergrowth_confidence';
  return panel === 'overlay' ? 'overlay' : 'mask';
}}
function metricLayer() {{ return viewMode.value === 'prediction' ? 'prediction' : 'intergrowth'; }}
function drawAll() {{ if (!currentJobId) return; canvasSize(); viewportText.textContent = `x=${{Math.round(view.x)}} y=${{Math.round(view.y)}} w=${{Math.round(view.w)}} h=${{Math.round(view.h)}}`; drawLayer(panelLayer('raw'), rawCanvas); drawLayer(panelLayer('overlay'), overlayCanvas); drawLayer(panelLayer('mask'), maskCanvas); updateMetrics(); }}
function zoom(f) {{ const cx=view.x+view.w/2, cy=view.y+view.h/2; view.w*=f; view.h*=f; view.x=cx-view.w/2; view.y=cy-view.h/2; clampView(); drawAll(); }}
function clampView() {{ view.w=Math.max(1,Math.min(statusData.image_width,view.w)); view.h=Math.max(1,Math.min(statusData.image_height,view.h)); view.x=Math.max(0,Math.min(statusData.image_width-view.w,view.x)); view.y=Math.max(0,Math.min(statusData.image_height-view.h,view.y)); }}
zoomIn.onclick = () => zoom(1/1.25); zoomOut.onclick = () => zoom(1.25); fullView.onclick = () => {{ view={{x:0,y:0,w:statusData.image_width,h:statusData.image_height}}; drawAll(); }}; cropMode.onclick = () => {{ cropSelecting=true; cropStart=null; cropCurrent=null; }};
viewMode.onchange = drawAll;
runIntergrowth.onclick = async () => {{
  if (!currentJobId) {{ status.textContent = 'Run prediction before intergrowth classification.'; return; }}
  const body = new URLSearchParams();
  const response = await fetch(`/jobs/${{currentJobId}}/intergrowth`, {{method:'POST', headers:{{'Content-Type':'application/x-www-form-urlencoded'}}, body}});
  const result = await response.json();
  if (!response.ok) {{ status.textContent = 'Intergrowth error: ' + result.error; return; }}
  viewMode.value = 'intergrowth';
  status.textContent += `\\nintergrowth=${{result.area_metrics?.image_label || 'done'}} hard_fraction=${{Number(result.area_metrics?.hard_fraction_of_metallic_ore || 0).toFixed(4)}}`;
  drawAll();
}};
function toImageXY(canvas, event) {{ const r=canvas.getBoundingClientRect(); return {{x:view.x+((event.clientX-r.left)/r.width)*view.w, y:view.y+((event.clientY-r.top)/r.height)*view.h}}; }}
function attachCrop(canvas) {{
  canvas.addEventListener('mousedown', e => {{ if (!cropSelecting) return; cropStart=toImageXY(canvas,e); cropCurrent=cropStart; drawAll(); }});
  canvas.addEventListener('mousemove', e => {{ if (!cropSelecting || !cropStart) return; cropCurrent=toImageXY(canvas,e); drawAll(); }});
  canvas.addEventListener('mouseup', e => {{ if (!cropSelecting || !cropStart) return; cropCurrent=toImageXY(canvas,e); const x=Math.min(cropStart.x,cropCurrent.x), y=Math.min(cropStart.y,cropCurrent.y); view={{x,y,w:Math.max(1,Math.abs(cropStart.x-cropCurrent.x)),h:Math.max(1,Math.abs(cropStart.y-cropCurrent.y))}}; cropSelecting=false; cropStart=null; cropCurrent=null; clampView(); drawAll(); }});
}}
function drawCropBorder(ctx, canvas) {{ if (!cropSelecting || !cropStart || !cropCurrent) return; const sx=(cropStart.x-view.x)/view.w*canvas.width, sy=(cropStart.y-view.y)/view.h*canvas.height; const ex=(cropCurrent.x-view.x)/view.w*canvas.width, ey=(cropCurrent.y-view.y)/view.h*canvas.height; ctx.save(); ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.setLineDash([6,4]); ctx.strokeRect(Math.min(sx,ex),Math.min(sy,ey),Math.abs(ex-sx),Math.abs(ey-sy)); ctx.restore(); }}
Object.values(canvases).forEach(attachCrop);
async function updateMetrics() {{
  const full = await (await fetch(`/jobs/${{currentJobId}}/metrics?layer=${{metricLayer()}}`)).json();
  const p = new URLSearchParams({{layer:metricLayer(), x:Math.round(view.x), y:Math.round(view.y), width:Math.round(view.w), height:Math.round(view.h)}});
  const crop = await (await fetch(`/jobs/${{currentJobId}}/metrics?${{p.toString()}}`)).json();
  metrics.innerHTML = metricTable('Full image', full) + metricTable('Visible crop', crop);
}}
function metricTable(title, data) {{ let rows = data.classes.map(c => `<tr><td>${{c.id}}: ${{c.name}}</td><td>${{c.pixels}}</td><td>${{Number(c.fraction).toFixed(5)}}</td></tr>`).join(''); if (!rows) rows='<tr><td colspan="3">no non-zero classes</td></tr>'; return `<h3>${{title}}</h3><table><tr><th>Class</th><th>Pixels</th><th>Fraction</th></tr>${{rows}}</table>`; }}
</script>
</body>
</html>"""


def render_active_learning_html(
    images: Iterable[Path],
    *,
    message: str = "",
    config: BackendConfig | None = None,
) -> str:
    """Render active-learning job launcher; editing happens on the review page."""
    cfg = (config or BackendConfig()).resolve()
    options = _render_image_options(images)
    binary_path = html.escape(str(cfg.binary_model_path))
    ore_path = html.escape(str(cfg.ore_model_path))
    escaped_message = html.escape(message)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Active Learning</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 1.5rem auto; line-height: 1.4; }}
    .panel {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; background: #fff; }}
    label {{ display: block; margin-top: .75rem; font-weight: 600; }}
    input, select, button {{ padding: .4rem; min-width: 18rem; }}
    progress {{ width: 100%; }}
    .message {{ font-weight: 600; }}
  </style>
</head>
<body>
  <h1>Active Learning</h1>
  <p><a href="/">Start</a> | <a href="/inference">Inference</a></p>
  <p class="message">{escaped_message}</p>
  <section class="panel">
    <h2>Predict and edit mask</h2>
    <form id="activeForm">
      <label>Image</label>
      <input id="activeImagePath" name="image_path" list="imageOptions" placeholder="datasets/.../image.jpg">
      <datalist id="imageOptions">{options}</datalist>
      <label>Choose indexed image</label>
      <select id="activeImageSelect"><option value="">-- choose image --</option>{options}</select>
      <button id="nextImage" type="button">Next image</button>
      <label>Model</label>
      <select name="model_kind"><option value="binary">binary segmentation</option><option value="ore">ore segmentation</option></select>
      <label>Binary checkpoint</label><input name="binary_model_path" value="{binary_path}">
      <label>Ore checkpoint</label><input name="ore_model_path" value="{ore_path}">
      <label>Device</label><select name="device"><option value="auto">auto</option><option value="cuda">cuda</option><option value="cpu">cpu</option></select>
      <label>Binary threshold</label><input name="binary_threshold" type="number" min="0" max="1" step="0.01" value="0.5">
      <label>Tile size</label><input name="tile_size" type="number" min="64" step="32" value="{int(cfg.panorama_tile_size)}">
      <label>Overlap</label><input name="overlap" type="number" min="0" step="16" value="{int(cfg.panorama_overlap)}">
      <label>Batch size</label><input name="batch_size" type="number" min="1" step="1" value="{int(cfg.panorama_batch_size)}">
      <p><button type="submit">Run prediction and open editor</button></p>
    </form>
    <progress id="progress" value="0" max="1"></progress>
    <pre id="status"></pre>
    <p id="reviewLink"></p>
  </section>
<script>
const imageValues = Array.from(document.querySelectorAll('#imageOptions option')).map(o => o.value);
activeImageSelect.addEventListener('change', () => {{ if (activeImageSelect.value) activeImagePath.value = activeImageSelect.value; }});
nextImage.onclick = () => {{ const current = activeImagePath.value; let i = imageValues.indexOf(current); i = (i + 1) % Math.max(1, imageValues.length); activeImagePath.value = imageValues[i] || ''; activeImageSelect.value = activeImagePath.value; }};
let jobId = null, timer = null;
activeForm.addEventListener('submit', async event => {{
  event.preventDefault();
  const body = new URLSearchParams(new FormData(activeForm));
  const response = await fetch('/jobs/panorama-predict', {{method:'POST', headers:{{'Content-Type':'application/x-www-form-urlencoded'}}, body}});
  const result = await response.json();
  if (!response.ok) {{ status.textContent = 'Error: ' + result.error; return; }}
  jobId = result.job_id; reviewLink.textContent = '';
  if (timer) clearInterval(timer);
  await poll();
  timer = setInterval(poll, 1000);
}});
async function poll() {{
  const s = await (await fetch(`/jobs/${{jobId}}`)).json();
  progress.max = Math.max(1, s.total_tiles || 0); progress.value = s.processed_tiles || 0;
  const eta = s.eta_sec == null ? 'unknown' : Number(s.eta_sec).toFixed(1) + 's';
  status.textContent = `status=${{s.status}} phase=${{s.phase}} model=${{s.model_kind}} tiles=${{s.processed_tiles||0}}/${{s.total_tiles||0}} elapsed=${{Number(s.elapsed_sec||0).toFixed(1)}}s eta=${{eta}}`;
  if (s.status === 'completed') {{ clearInterval(timer); reviewLink.innerHTML = `<a href="/active-learning?job_id=${{jobId}}">Open mask editor</a>`; }}
  if (s.status === 'failed' || s.status === 'cancelled') clearInterval(timer);
}}
</script>
</body>
</html>"""
