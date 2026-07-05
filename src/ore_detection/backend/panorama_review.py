"""Server-side panorama tile rendering and mask-edit persistence."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from ore_detection.backend.ui_annotation import (
    BASE_UI_CLASSES,
    UiClass,
    class_index_to_color_image,
    palette_from_classes,
    ui_classes_for_model,
)
from ore_detection.descriptors.erosion_ratio import (
    ErosionRatioConfig,
    classify_erosion_ratio_intergrowth,
    erosion_ratio_score_grid,
    load_erosion_ratio_config,
    mean_ore_ratio_score,
)
from ore_detection.inference.tiled_prediction import allow_large_pillow_images
from ore_detection.visualization.overlay import overlay_mask_on_image

TALC_THRESHOLD_METRICS: dict[str, dict[str, int | str]] = {
    "hsv_value": {"label": "HSV Value", "min": 0, "max": 255},
    "rgb_sum": {"label": "R + G + B", "min": 0, "max": 765},
}
INTERGROWTH_GRID_PIXEL_THRESHOLD = 25_000_000
INTERGROWTH_METRIC_SAMPLE_MAX_PIXELS = 1_000_000


@dataclass(frozen=True)
class BrushPatch:
    """One circular brush edit in source-image coordinates."""

    x: int
    y: int
    radius: int
    class_id: int
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "radius": self.radius,
            "class_id": self.class_id,
            "created_at": self.created_at,
        }


def read_panorama_metadata(sample_dir: str | Path) -> dict[str, Any]:
    metadata_path = Path(sample_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"panorama metadata does not exist: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _write_panorama_metadata(sample_dir: str | Path, metadata: dict[str, Any]) -> None:
    metadata_path = Path(sample_dir) / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _class_names_from_metadata(metadata: dict[str, Any]) -> tuple[str, ...]:
    checkpoint = metadata.get("ore_checkpoint")
    if isinstance(checkpoint, dict):
        names = checkpoint.get("class_names")
        if isinstance(names, list) and names:
            return tuple(str(name) for name in names)
    return ()


def _classes_from_metadata(metadata: dict[str, Any], classes: Iterable[dict[str, Any]] | None = None) -> tuple[UiClass, ...]:
    if classes is None:
        return ui_classes_for_model(_class_names_from_metadata(metadata))
    result: list[UiClass] = []
    for item in classes:
        result.append(
            UiClass(
                int(item["id"]),
                str(item["name"]),
                tuple(int(value) for value in item["color"]),
                str(item.get("meaning", "")),
                bool(item.get("editable", True)),
            )
        )
    return tuple(result)


def _metadata_artifact_path(metadata: dict[str, Any], name: str) -> Path:
    artifacts = metadata.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("metadata artifacts block is missing")
    value = artifacts.get(name)
    if not value:
        raise FileNotFoundError(f"metadata artifact `{name}` is missing")
    return Path(str(value))


def _optional_metadata_artifact_path(metadata: dict[str, Any], name: str) -> Path | None:
    artifacts = metadata.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return None
    value = artifacts.get(name)
    return Path(str(value)) if value else None


def review_mask_path(sample_dir: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    metadata = metadata or read_panorama_metadata(sample_dir)
    artifacts = metadata.get("artifacts", {})
    if isinstance(artifacts, dict) and artifacts.get("review_mask"):
        return Path(str(artifacts["review_mask"]))
    if isinstance(artifacts, dict) and artifacts.get("ore_multiclass_mask"):
        return Path(str(artifacts["ore_multiclass_mask"]))
    return _metadata_artifact_path(metadata, "ore_mask")


def patch_log_path(sample_dir: str | Path) -> Path:
    return Path(sample_dir) / "patch_log.jsonl"


def _mutable_review_mask_path(sample_dir: str | Path) -> Path:
    return Path(sample_dir) / "review_mask.png"


def restore_base_prediction(sample_dir: str | Path) -> dict[str, Any]:
    """Reset the rendered mask to the immutable base prediction and clear patches."""
    sample_dir = Path(sample_dir)
    path = patch_log_path(sample_dir)
    if path.exists():
        path.unlink()

    metadata = read_panorama_metadata(sample_dir)
    artifacts = metadata.get("artifacts", {})
    review_mask = None
    base_prediction = None
    if isinstance(artifacts, dict):
        base_value = artifacts.get("base_prediction_mask")
        if base_value:
            base_prediction = Path(str(base_value))
            review_mask = _mutable_review_mask_path(sample_dir)
            if base_prediction.exists():
                if base_prediction.resolve() != review_mask.resolve():
                    shutil.copy2(base_prediction, review_mask)
                artifacts["review_mask"] = str(review_mask)
                metadata["artifacts"] = artifacts
        metadata.pop("talc_threshold", None)
        metadata["review_updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_panorama_metadata(sample_dir, metadata)

    return {
        "ok": True,
        "patch_log_cleared": True,
        "sample_dir": str(sample_dir),
        "review_mask": str(review_mask) if review_mask is not None else str(review_mask_path(sample_dir, metadata)),
        "base_prediction_mask": str(base_prediction) if base_prediction is not None else None,
    }


def _class_id_for_name(classes: Iterable[UiClass], name: str) -> int:
    for item in classes:
        if item.name == name:
            return int(item.id)
    raise ValueError(f"UI class `{name}` is missing")


def _background_class_id(classes: Iterable[UiClass]) -> int:
    for item in classes:
        if item.name in {"background", "background_matrix"}:
            return int(item.id)
    return 0


def talc_histograms_for_panorama(sample_dir: str | Path) -> dict[str, Any]:
    """Return full-image histograms used by the panorama talc threshold UI."""
    sample_dir = Path(sample_dir)
    metadata = read_panorama_metadata(sample_dir)
    hsv_histogram = np.zeros(256, dtype=np.int64)
    rgb_sum_histogram = np.zeros(766, dtype=np.int64)
    with allow_large_pillow_images():
        with Image.open(Path(str(metadata["image_path"]))) as opened:
            height = int(opened.height)
            width = int(opened.width)
            for top in range(0, height, 512):
                bottom = min(height, top + 512)
                tile = np.asarray(opened.crop((0, top, width, bottom)).convert("RGB"), dtype=np.uint8)
                hsv_histogram += np.bincount(tile.max(axis=2).ravel(), minlength=256)
                rgb_sum = tile.sum(axis=2, dtype=np.uint16)
                rgb_sum_histogram += np.bincount(rgb_sum.ravel(), minlength=766)
    histograms = {
        "hsv_value": hsv_histogram.astype(int).tolist(),
        "rgb_sum": rgb_sum_histogram.astype(int).tolist(),
    }
    return {
        "image_path": str(metadata["image_path"]),
        "image_width": int(metadata["image_width"]),
        "image_height": int(metadata["image_height"]),
        "histograms": {
            name: {
                **dict(TALC_THRESHOLD_METRICS[name]),
                "histogram": values,
                "total": sum(values),
            }
            for name, values in histograms.items()
        },
    }


def _metric_value(metric: str, r: int, g: int, b: int) -> int:
    if metric == "hsv_value":
        return max(r, g, b)
    if metric == "rgb_sum":
        return r + g + b
    raise ValueError(f"unknown talc threshold metric: {metric}")


def apply_talc_threshold_to_panorama(
    sample_dir: str | Path,
    *,
    metric: str,
    threshold: int,
) -> dict[str, Any]:
    """Apply a full-panorama dark-pixel talc threshold to the editable review mask."""
    sample_dir = Path(sample_dir)
    if metric not in TALC_THRESHOLD_METRICS:
        raise ValueError("metric must be `hsv_value` or `rgb_sum`")
    metric_info = TALC_THRESHOLD_METRICS[metric]
    min_threshold = int(metric_info["min"])
    max_threshold = int(metric_info["max"])
    threshold = int(threshold)
    if threshold < min_threshold or threshold > max_threshold:
        raise ValueError(f"threshold for {metric} must be in [{min_threshold}, {max_threshold}]")

    metadata = read_panorama_metadata(sample_dir)
    classes = _classes_from_metadata(metadata)
    talc_id = _class_id_for_name(classes, "talc")
    background_id = _background_class_id(classes)
    patches = load_brush_patches(sample_dir)

    with allow_large_pillow_images():
        with Image.open(review_mask_path(sample_dir, metadata)) as opened_mask:
            mask = opened_mask.convert("L")
        if patches:
            mask = apply_patches_to_mask_tile(mask, origin_x=0, origin_y=0, patches=patches)
        talc_pixels = 0
        cleared_talc_pixels = 0
        with Image.open(Path(str(metadata["image_path"]))) as opened_image:
            if opened_image.size != mask.size:
                raise ValueError(f"source image size {opened_image.size} does not match review mask size {mask.size}")
            height = int(mask.height)
            width = int(mask.width)
            for top in range(0, height, 512):
                bottom = min(height, top + 512)
                box = (0, top, width, bottom)
                image_tile = np.asarray(opened_image.crop(box).convert("RGB"), dtype=np.uint8)
                mask_tile = np.array(mask.crop(box), dtype=np.uint8, copy=True)
                if metric == "hsv_value":
                    metric_values = image_tile.max(axis=2)
                elif metric == "rgb_sum":
                    metric_values = image_tile.sum(axis=2, dtype=np.uint16)
                else:
                    raise ValueError(f"unknown talc threshold metric: {metric}")
                background = mask_tile == int(background_id)
                talc = mask_tile == int(talc_id)
                below = metric_values < int(threshold)
                new_talc = background & below
                cleared_talc = talc & ~below
                talc_pixels += int(np.count_nonzero(new_talc))
                cleared_talc_pixels += int(np.count_nonzero(cleared_talc))
                mask_tile[new_talc] = int(talc_id)
                mask_tile[cleared_talc] = int(background_id)
                mask.paste(Image.fromarray(mask_tile), (0, top))

        updated = mask
        review_mask = _mutable_review_mask_path(sample_dir)
        updated.save(review_mask, compress_level=1)

    patch_path = patch_log_path(sample_dir)
    if patch_path.exists():
        patch_path.unlink()

    artifacts = metadata.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("metadata artifacts block is missing")
    artifacts["review_mask"] = str(review_mask)
    metadata["artifacts"] = artifacts
    metadata["talc_threshold"] = {
        "metric": metric,
        "threshold": threshold,
        "talc_class_id": talc_id,
        "background_class_id": background_id,
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "materialized_patch_count": len(patches),
    }
    metadata["review_updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_panorama_metadata(sample_dir, metadata)

    return {
        "ok": True,
        "metric": metric,
        "threshold": threshold,
        "review_mask": str(review_mask),
        "patch_log_cleared": True,
        "materialized_patch_count": len(patches),
        "new_talc_pixels": talc_pixels,
        "cleared_talc_pixels": cleared_talc_pixels,
    }


def _binary_mask_from_class(mask: Image.Image, class_id: int) -> Image.Image:
    values = np.asarray(mask.convert("L"), dtype=np.uint8)
    return Image.fromarray(np.where(values == int(class_id), 255, 0).astype(np.uint8))


def _binary_from_multiclass(multiclass: Image.Image, *, background_index: int) -> Image.Image:
    values = np.asarray(multiclass.convert("L"), dtype=np.uint8)
    return Image.fromarray(np.where(values != int(background_index), 255, 0).astype(np.uint8))


def _mean_score_on_ore(score: Image.Image, ore_mask: Image.Image) -> float:
    return mean_ore_ratio_score(score, ore_mask)


_SCORE_COLORMAPS: dict[str, tuple[tuple[float, tuple[int, int, int]], ...]] = {
    "viridis": (
        (0.0, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.0, (253, 231, 37)),
    ),
    "magma": (
        (0.0, (0, 0, 4)),
        (0.25, (80, 18, 123)),
        (0.50, (182, 54, 121)),
        (0.75, (251, 136, 97)),
        (1.0, (252, 253, 191)),
    ),
}


def _score_palette(name: str) -> list[tuple[int, int, int]]:
    stops = _SCORE_COLORMAPS[name]
    palette: list[tuple[int, int, int]] = []
    for value in range(256):
        point = value / 255
        left = stops[0]
        right = stops[-1]
        for index in range(len(stops) - 1):
            if stops[index][0] <= point <= stops[index + 1][0]:
                left = stops[index]
                right = stops[index + 1]
                break
        span = max(1e-9, right[0] - left[0])
        t = (point - left[0]) / span
        palette.append(
            tuple(
                int(round(left[1][channel] + (right[1][channel] - left[1][channel]) * t))
                for channel in range(3)
            )
        )
    return palette


def _colorize_soft_score(score: Image.Image, *, colormap: str) -> Image.Image:
    score_array = np.asarray(score.convert("L"), dtype=np.uint8)
    palette = np.asarray(_score_palette(colormap), dtype=np.uint8)
    return Image.fromarray(palette[score_array])


def _blend_soft_score_on_image(
    raw: Image.Image,
    score: Image.Image,
    intergrowth_mask: Image.Image,
    *,
    colormap: str,
) -> Image.Image:
    raw_array = np.asarray(raw.convert("RGB"), dtype=np.uint8)
    soft_array = np.asarray(_colorize_soft_score(score, colormap=colormap), dtype=np.uint8)
    mask_values = np.asarray(intergrowth_mask.convert("L"), dtype=np.uint8)
    normal_id = next(item.id for item in BASE_UI_CLASSES if item.name == "normal_ore")
    hard_id = next(item.id for item in BASE_UI_CLASSES if item.name == "hard_ore")
    blend_mask = np.isin(mask_values, np.array([normal_id, hard_id], dtype=np.uint8))
    output = raw_array.copy()
    blended = ((raw_array.astype(np.uint16) * 35 + soft_array.astype(np.uint16) * 65 + 50) // 100).astype(np.uint8)
    output[blend_mask] = blended[blend_mask]
    return Image.fromarray(output)


def _optional_class_id_for_name(classes: Iterable[UiClass], name: str) -> int | None:
    for item in classes:
        if item.name == name:
            return int(item.id)
    return None


def _intergrowth_grid_mode(metadata: dict[str, Any]) -> bool:
    intergrowth = metadata.get("intergrowth", {})
    artifacts = metadata.get("artifacts", {})
    return (
        isinstance(intergrowth, dict)
        and intergrowth.get("mode") == "score_grid"
        or isinstance(artifacts, dict)
        and bool(artifacts.get("intergrowth_score_grid"))
    )


def _intergrowth_ready(metadata: dict[str, Any]) -> bool:
    artifacts = metadata.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return False
    if _intergrowth_grid_mode(metadata):
        return bool(artifacts.get("intergrowth_score_grid") and artifacts.get("intergrowth_hard_score_grid"))
    return bool(
        artifacts.get("intergrowth_mask")
        and artifacts.get("intergrowth_score")
        and artifacts.get("intergrowth_hard_score")
    )


def _intergrowth_config_from_metadata(metadata: dict[str, Any]) -> ErosionRatioConfig:
    intergrowth = metadata.get("intergrowth", {})
    if isinstance(intergrowth, dict) and isinstance(intergrowth.get("classifier_config"), dict):
        return ErosionRatioConfig.from_dict(intergrowth["classifier_config"])
    return ErosionRatioConfig()


def _grid_box_for_source_box(
    metadata: dict[str, Any],
    grid: Image.Image,
    source_box: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    image_width = max(1, int(metadata["image_width"]))
    image_height = max(1, int(metadata["image_height"]))
    left, top, right, bottom = source_box
    return (
        left / image_width * grid.width,
        top / image_height * grid.height,
        right / image_width * grid.width,
        bottom / image_height * grid.height,
    )


def _resized_grid_tile(
    metadata: dict[str, Any],
    artifact_name: str,
    source_box: tuple[int, int, int, int],
    target_size: tuple[int, int] | None,
    *,
    resampling: Image.Resampling,
) -> Image.Image:
    with Image.open(_metadata_artifact_path(metadata, artifact_name)) as grid:
        output_size = target_size or (source_box[2] - source_box[0], source_box[3] - source_box[1])
        grid_box = _grid_box_for_source_box(metadata, grid, source_box)
        return grid.resize(output_size, resampling, box=grid_box).convert("L")


def _review_mask_tile_for_box(
    sample_dir: Path,
    metadata: dict[str, Any],
    source_box: tuple[int, int, int, int],
    target_size: tuple[int, int] | None,
) -> Image.Image:
    with Image.open(review_mask_path(sample_dir, metadata)) as mask:
        if target_size is None:
            return mask.crop(source_box).convert("L")
        return mask.resize(target_size, Image.Resampling.NEAREST, box=source_box).convert("L")


def _blend_soft_score_on_review_mask(
    raw: Image.Image,
    score: Image.Image,
    review_mask: Image.Image,
    classes: Iterable[UiClass],
    *,
    colormap: str,
) -> Image.Image:
    raw_array = np.asarray(raw.convert("RGB"), dtype=np.uint8)
    soft_array = np.asarray(_colorize_soft_score(score, colormap=colormap), dtype=np.uint8)
    class_objects = tuple(classes)
    background_id = _background_class_id(class_objects)
    talc_id = _optional_class_id_for_name(class_objects, "talc")
    ignore_id = _optional_class_id_for_name(class_objects, "ignore")
    excluded = [background_id]
    if talc_id is not None:
        excluded.append(talc_id)
    if ignore_id is not None:
        excluded.append(ignore_id)
    mask_values = np.asarray(review_mask.convert("L"), dtype=np.uint8)
    blend_mask = ~np.isin(mask_values, np.asarray(excluded, dtype=np.uint8))
    output = raw_array.copy()
    blended = ((raw_array.astype(np.uint16) * 35 + soft_array.astype(np.uint16) * 65 + 50) // 100).astype(np.uint8)
    output[blend_mask] = blended[blend_mask]
    return Image.fromarray(output)


def _grid_intergrowth_class_tile(
    sample_dir: Path,
    metadata: dict[str, Any],
    source_box: tuple[int, int, int, int],
    target_size: tuple[int, int] | None,
) -> Image.Image:
    classes = _classes_from_metadata(metadata)
    cfg = _intergrowth_config_from_metadata(metadata)
    review_tile = _review_mask_tile_for_box(sample_dir, metadata, source_box, target_size)
    score_tile = _resized_grid_tile(
        metadata,
        "intergrowth_score_grid",
        source_box,
        target_size,
        resampling=Image.Resampling.BILINEAR,
    )
    background_id = _background_class_id(classes)
    talc_id = _optional_class_id_for_name(classes, "talc")
    ignore_id = _optional_class_id_for_name(classes, "ignore")
    excluded = [background_id]
    if talc_id is not None:
        excluded.append(talc_id)
    if ignore_id is not None:
        excluded.append(ignore_id)
    threshold_u8 = int(round(cfg.normal_threshold * 255))
    review_values = np.asarray(review_tile.convert("L"), dtype=np.uint8)
    score_values = np.asarray(score_tile.convert("L"), dtype=np.uint8)
    output = np.full(review_values.shape, int(cfg.background_id), dtype=np.uint8)
    ore_mask = ~np.isin(review_values, np.asarray(excluded, dtype=np.uint8))
    output[ore_mask] = np.where(score_values[ore_mask] > threshold_u8, int(cfg.normal_ore_id), int(cfg.hard_ore_id))
    if talc_id is not None:
        output[review_values == int(talc_id)] = int(cfg.talc_id)
    if ignore_id is not None:
        output[review_values == int(ignore_id)] = int(cfg.ignore_id)
    return Image.fromarray(output)


def _area_metrics_from_counts(counts: dict[str, int], total_pixels: int) -> dict[str, Any]:
    total = max(1, int(total_pixels))
    metallic_area = int(counts.get("normal_ore", 0)) + int(counts.get("hard_ore", 0))
    hard_fraction = counts.get("hard_ore", 0) / metallic_area if metallic_area else 0.0
    normal_fraction = counts.get("normal_ore", 0) / metallic_area if metallic_area else 0.0
    talc_fraction = counts.get("talc", 0) / total
    if talc_fraction > 0.10:
        image_label = "talc"
    elif metallic_area == 0:
        image_label = "background"
    elif hard_fraction >= 0.50:
        image_label = "hard_ore"
    else:
        image_label = "normal_ore"
    normalized_counts = {
        "background": int(counts.get("background", 0)),
        "talc": int(counts.get("talc", 0)),
        "normal_ore": int(counts.get("normal_ore", 0)),
        "hard_ore": int(counts.get("hard_ore", 0)),
        "ignore": int(counts.get("ignore", 0)),
    }
    return {
        "total_pixels": total_pixels,
        "counts": normalized_counts,
        "fractions": {name: count / total for name, count in normalized_counts.items()},
        "metallic_ore_pixels": metallic_area,
        "hard_fraction_of_metallic_ore": hard_fraction,
        "normal_fraction_of_metallic_ore": normal_fraction,
        "image_label": image_label,
    }


def _grid_metrics_from_summaries(
    summaries: list[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
    config: ErosionRatioConfig,
    talc_pixels: int = 0,
    ignore_pixels: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normal_pixels = 0
    hard_pixels = 0
    normal_score_sum = 0.0
    hard_score_sum = 0.0
    ore_pixels = 0
    for row in summaries:
        row_ore_pixels = int(row.get("ore_area", 0) or 0)
        if row_ore_pixels <= 0:
            continue
        normal_score = float(row.get("normal_score", 0.0) or 0.0)
        hard_score = float(row.get("hard_score", 1.0 - normal_score) or 0.0)
        ore_pixels += row_ore_pixels
        normal_score_sum += normal_score * row_ore_pixels
        hard_score_sum += hard_score * row_ore_pixels
        if normal_score > config.normal_threshold:
            normal_pixels += row_ore_pixels
        else:
            hard_pixels += row_ore_pixels
    total_pixels = int(image_width) * int(image_height)
    background_pixels = max(0, total_pixels - normal_pixels - hard_pixels - int(talc_pixels) - int(ignore_pixels))
    area_metrics = _area_metrics_from_counts(
        {
            "background": background_pixels,
            "talc": int(talc_pixels),
            "normal_ore": normal_pixels,
            "hard_ore": hard_pixels,
            "ignore": int(ignore_pixels),
        },
        total_pixels,
    )
    score_metrics = {
        "ore_pixels": ore_pixels,
        "mean_erosion_ratio_score": (normal_score_sum / ore_pixels) if ore_pixels else 0.0,
        "mean_hard_score": (hard_score_sum / ore_pixels) if ore_pixels else 0.0,
    }
    return area_metrics, score_metrics


def _save_erosion_ratio_intergrowth_grid_artifacts(
    sample_dir: Path,
    *,
    metadata: dict[str, Any],
    config: ErosionRatioConfig,
    classifier_config_path: str | Path | None = None,
) -> dict[str, Any]:
    metadata_path = sample_dir / "metadata.json"
    classes = _classes_from_metadata(metadata)
    talc_id = _optional_class_id_for_name(classes, "talc")
    ignore_id = _optional_class_id_for_name(classes, "ignore")
    ore_checkpoint = metadata.get("ore_checkpoint") if isinstance(metadata.get("ore_checkpoint"), dict) else {}
    background_index = int(ore_checkpoint.get("background_index", 0)) if isinstance(ore_checkpoint, dict) else 0
    artifacts = metadata.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("metadata artifacts block is missing")

    ore_mask_path = _optional_metadata_artifact_path(metadata, "ore_mask")
    multiclass_mask_path = _optional_metadata_artifact_path(metadata, "ore_multiclass_mask")
    if ore_mask_path is None and multiclass_mask_path is None:
        raise FileNotFoundError("prediction metadata has neither ore_mask nor ore_multiclass_mask")

    with allow_large_pillow_images():
        multiclass_image = Image.open(multiclass_mask_path).convert("L") if multiclass_mask_path is not None else None
        if ore_mask_path is not None and ore_mask_path.exists():
            ore_image = Image.open(ore_mask_path).convert("L")
            ore_source = "ore_mask"
        elif multiclass_image is not None:
            ore_image = _binary_from_multiclass(multiclass_image, background_index=background_index)
            ore_source = "ore_multiclass_mask_non_background"
        else:
            raise FileNotFoundError("usable ore mask artifact is missing")

        normal_grid, summaries = erosion_ratio_score_grid(
            ore_image,
            multiclass_mask=multiclass_image,
            background_index=background_index,
            config=config,
        )
        hard_grid = Image.new("L", normal_grid.size)
        normal_grid_values = np.asarray(normal_grid, dtype=np.uint8)
        ore_area_values = np.asarray([int(row.get("ore_area", 0) or 0) for row in summaries], dtype=np.int64).reshape(
            normal_grid_values.shape
        )
        hard_grid = Image.fromarray(np.where(ore_area_values > 0, 255 - normal_grid_values, 0).astype(np.uint8))

        talc_pixels = 0
        ignore_pixels = 0
        with Image.open(review_mask_path(sample_dir, metadata)) as review:
            review_hist = review.convert("L").histogram()
            if talc_id is not None:
                talc_pixels = int(review_hist[talc_id])
            if ignore_id is not None:
                ignore_pixels = int(review_hist[ignore_id])

    intergrowth_score_grid_path = sample_dir / "intergrowth_score_grid.png"
    intergrowth_hard_score_grid_path = sample_dir / "intergrowth_hard_score_grid.png"
    intergrowth_metrics_path = sample_dir / "intergrowth_metrics.json"
    normal_grid.save(intergrowth_score_grid_path, compress_level=1)
    hard_grid.save(intergrowth_hard_score_grid_path, compress_level=1)

    image_width = int(metadata["image_width"])
    image_height = int(metadata["image_height"])
    area_metrics, score_metrics = _grid_metrics_from_summaries(
        summaries,
        image_width=image_width,
        image_height=image_height,
        config=config,
        talc_pixels=talc_pixels,
        ignore_pixels=ignore_pixels,
    )
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_prediction_metadata": str(metadata_path),
        "source_ore_mask": ore_source,
        "mode": "score_grid",
        "approximate": True,
        "classifier_config": config.to_dict(),
        "classifier_config_path": str(classifier_config_path) if classifier_config_path is not None else None,
        "grid_size": [normal_grid.width, normal_grid.height],
        "image_size": [image_width, image_height],
        "window_count": len(summaries),
        "window_scores": {
            "min": min((row["normal_score"] for row in summaries), default=0.0),
            "max": max((row["normal_score"] for row in summaries), default=0.0),
            "mean": (
                sum(float(row["normal_score"]) for row in summaries) / len(summaries)
                if summaries
                else 0.0
            ),
        },
        "area_metrics": area_metrics,
        "score_metrics": score_metrics,
        "artifacts": {
            "intergrowth_score_grid": str(intergrowth_score_grid_path),
            "intergrowth_hard_score_grid": str(intergrowth_hard_score_grid_path),
            "intergrowth_metrics": str(intergrowth_metrics_path),
        },
    }
    intergrowth_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    artifacts.update(metrics["artifacts"])
    for legacy_key in ("intergrowth_mask", "intergrowth_score", "intergrowth_hard_score", "intergrowth_confidence"):
        artifacts.pop(legacy_key, None)
    metadata["artifacts"] = artifacts
    metadata["intergrowth"] = {
        "method": "local_erosion_ratio_score",
        "mode": "score_grid",
        "source_ore_mask": ore_source,
        "classifier_config": config.to_dict(),
        "metrics_path": str(intergrowth_metrics_path),
        "area_metrics": area_metrics,
        "score_metrics": score_metrics,
        "approximate": True,
        "grid_size": [normal_grid.width, normal_grid.height],
    }
    _write_panorama_metadata(sample_dir, metadata)
    metrics["metadata_path"] = str(metadata_path)
    return metrics


def save_erosion_ratio_intergrowth_artifacts(
    sample_dir: str | Path,
    *,
    config: ErosionRatioConfig | None = None,
    classifier_config_path: str | Path | None = None,
    large_image_pixel_threshold: int = INTERGROWTH_GRID_PIXEL_THRESHOLD,
) -> dict[str, Any]:
    """Create hard/normal intergrowth artifacts using the calibrated erosion-ratio metric."""
    sample_dir = Path(sample_dir)
    metadata_path = sample_dir / "metadata.json"
    metadata = read_panorama_metadata(sample_dir)
    cfg = config or load_erosion_ratio_config(classifier_config_path)
    cfg.validate()
    pixel_count = int(metadata["image_width"]) * int(metadata["image_height"])
    if pixel_count > int(large_image_pixel_threshold):
        return _save_erosion_ratio_intergrowth_grid_artifacts(
            sample_dir,
            metadata=metadata,
            config=cfg,
            classifier_config_path=classifier_config_path,
        )
    classes = _classes_from_metadata(metadata)
    talc_id = _class_id_for_name(classes, "talc")
    ignore_id = _class_id_for_name(classes, "ignore")
    patches = load_brush_patches(sample_dir)

    ore_checkpoint = metadata.get("ore_checkpoint") if isinstance(metadata.get("ore_checkpoint"), dict) else {}
    background_index = int(ore_checkpoint.get("background_index", 0)) if isinstance(ore_checkpoint, dict) else 0
    artifacts = metadata.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("metadata artifacts block is missing")

    ore_mask_path = _optional_metadata_artifact_path(metadata, "ore_mask")
    multiclass_mask_path = _optional_metadata_artifact_path(metadata, "ore_multiclass_mask")
    if ore_mask_path is None and multiclass_mask_path is None:
        raise FileNotFoundError("prediction metadata has neither ore_mask nor ore_multiclass_mask")

    with allow_large_pillow_images():
        multiclass_image = Image.open(multiclass_mask_path).convert("L") if multiclass_mask_path is not None else None
        if ore_mask_path is not None and ore_mask_path.exists():
            ore_image = Image.open(ore_mask_path).convert("L")
            ore_source = "ore_mask"
        elif multiclass_image is not None:
            ore_image = _binary_from_multiclass(multiclass_image, background_index=background_index)
            ore_source = "ore_multiclass_mask_non_background"
        else:
            raise FileNotFoundError("usable ore mask artifact is missing")

        with Image.open(review_mask_path(sample_dir, metadata)) as opened_review:
            review_mask = opened_review.convert("L")
        if patches:
            review_mask = apply_patches_to_mask_tile(review_mask, origin_x=0, origin_y=0, patches=patches)
        if review_mask.size != ore_image.size:
            raise ValueError(f"review mask size {review_mask.size} does not match ore mask size {ore_image.size}")

        talc_mask = _binary_mask_from_class(review_mask, talc_id)
        ignore_mask = _binary_mask_from_class(review_mask, ignore_id)
        result = classify_erosion_ratio_intergrowth(
            ore_image,
            multiclass_mask=multiclass_image,
            talc_mask=talc_mask,
            ignore_mask=ignore_mask,
            background_index=background_index,
            config=cfg,
        )

    intergrowth_mask_path = sample_dir / "intergrowth_mask.png"
    intergrowth_score_path = sample_dir / "intergrowth_score.png"
    intergrowth_hard_score_path = sample_dir / "intergrowth_hard_score.png"
    intergrowth_metrics_path = sample_dir / "intergrowth_metrics.json"
    result.intergrowth_mask.save(intergrowth_mask_path, compress_level=1)
    result.normal_score.save(intergrowth_score_path, compress_level=1)
    result.hard_score.save(intergrowth_hard_score_path, compress_level=1)

    score_metrics = {
        "mean_erosion_ratio_score": _mean_score_on_ore(result.normal_score, ore_image),
        "mean_hard_score": _mean_score_on_ore(result.hard_score, ore_image),
    }
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_prediction_metadata": str(metadata_path),
        "source_ore_mask": ore_source,
        "classifier_config": cfg.to_dict(),
        "window_count": len(result.window_summaries),
        "window_scores": {
            "min": min((row["normal_score"] for row in result.window_summaries), default=0.0),
            "max": max((row["normal_score"] for row in result.window_summaries), default=0.0),
            "mean": (
                sum(float(row["normal_score"]) for row in result.window_summaries) / len(result.window_summaries)
                if result.window_summaries
                else 0.0
            ),
        },
        "area_metrics": result.metrics,
        "score_metrics": score_metrics,
        "artifacts": {
            "intergrowth_mask": str(intergrowth_mask_path),
            "intergrowth_score": str(intergrowth_score_path),
            "intergrowth_hard_score": str(intergrowth_hard_score_path),
            "intergrowth_metrics": str(intergrowth_metrics_path),
        },
    }
    intergrowth_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    artifacts.update(metrics["artifacts"])
    artifacts.pop("intergrowth_confidence", None)
    metadata["artifacts"] = artifacts
    metadata["intergrowth"] = {
        "method": "local_erosion_ratio_score",
        "source_ore_mask": ore_source,
        "classifier_config": cfg.to_dict(),
        "metrics_path": str(intergrowth_metrics_path),
        "area_metrics": result.metrics,
        "score_metrics": score_metrics,
    }
    _write_panorama_metadata(sample_dir, metadata)
    metrics["metadata_path"] = str(metadata_path)
    return metrics


def append_brush_patch(
    sample_dir: str | Path,
    *,
    x: int,
    y: int,
    radius: int,
    class_id: int,
) -> BrushPatch:
    """Append one brush edit to the panorama patch log."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    patch = BrushPatch(
        x=int(x),
        y=int(y),
        radius=int(radius),
        class_id=int(class_id),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    path = patch_log_path(sample_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(patch.as_dict()) + "\n")
    return patch


def load_brush_patches(sample_dir: str | Path) -> list[BrushPatch]:
    path = patch_log_path(sample_dir)
    if not path.exists():
        return []
    patches: list[BrushPatch] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        patches.append(
            BrushPatch(
                x=int(data["x"]),
                y=int(data["y"]),
                radius=int(data["radius"]),
                class_id=int(data["class_id"]),
                created_at=str(data.get("created_at", "")),
            )
        )
    return patches


def _patch_intersects_box(patch: BrushPatch, box: tuple[int, int, int, int]) -> bool:
    left, top, right, bottom = box
    return not (
        patch.x + patch.radius < left
        or patch.x - patch.radius > right
        or patch.y + patch.radius < top
        or patch.y - patch.radius > bottom
    )


def apply_patches_to_mask_tile(
    mask_tile: Image.Image,
    *,
    origin_x: int,
    origin_y: int,
    patches: Iterable[BrushPatch],
) -> Image.Image:
    """Apply relevant brush patches to a cropped mask tile."""
    tile = mask_tile.convert("L")
    draw = ImageDraw.Draw(tile)
    box = (origin_x, origin_y, origin_x + tile.width, origin_y + tile.height)
    for patch in patches:
        if not _patch_intersects_box(patch, box):
            continue
        left = patch.x - patch.radius - origin_x
        top = patch.y - patch.radius - origin_y
        right = patch.x + patch.radius - origin_x
        bottom = patch.y + patch.radius - origin_y
        draw.ellipse((left, top, right, bottom), fill=int(patch.class_id))
    return tile


def _apply_patches_to_resized_mask_tile(
    mask_tile: Image.Image,
    *,
    source_box: tuple[int, int, int, int],
    patches: Iterable[BrushPatch],
) -> Image.Image:
    tile = mask_tile.convert("L")
    draw = ImageDraw.Draw(tile)
    left, top, right, bottom = source_box
    source_width = max(1, right - left)
    source_height = max(1, bottom - top)
    for patch in patches:
        if not _patch_intersects_box(patch, source_box):
            continue
        cx = (patch.x - left) / source_width * tile.width
        cy = (patch.y - top) / source_height * tile.height
        rx = patch.radius / source_width * tile.width
        ry = patch.radius / source_height * tile.height
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=int(patch.class_id))
    return tile


def _clamped_box(metadata: dict[str, Any], *, x: int, y: int, width: int, height: int) -> tuple[int, int, int, int]:
    image_width = int(metadata["image_width"])
    image_height = int(metadata["image_height"])
    left = max(0, min(int(x), image_width - 1))
    top = max(0, min(int(y), image_height - 1))
    right = max(left + 1, min(image_width, left + max(1, int(width))))
    bottom = max(top + 1, min(image_height, top + max(1, int(height))))
    return (left, top, right, bottom)


def render_panorama_tile(
    sample_dir: str | Path,
    *,
    layer: str,
    x: int,
    y: int,
    width: int,
    height: int,
    output_width: int | None = None,
    output_height: int | None = None,
) -> Image.Image:
    """Render one panorama viewport tile for raw/mask/overlay/confidence layers."""
    sample_dir = Path(sample_dir)
    metadata = read_panorama_metadata(sample_dir)
    box = _clamped_box(metadata, x=x, y=y, width=width, height=height)
    patches = load_brush_patches(sample_dir)
    classes = _classes_from_metadata(metadata)
    intergrowth_classes = BASE_UI_CLASSES
    palette = palette_from_classes(classes)
    intergrowth_palette = palette_from_classes(intergrowth_classes)
    grid_mode = _intergrowth_grid_mode(metadata)
    target_size = None
    if output_width is not None or output_height is not None:
        target_size = (max(1, int(output_width or width)), max(1, int(output_height or height)))

    with allow_large_pillow_images():
        if layer.startswith("intergrowth") and not _intergrowth_ready(metadata):
            output_size = target_size or (box[2] - box[0], box[3] - box[1])
            if "overlay" in layer:
                with Image.open(Path(str(metadata["image_path"]))) as image:
                    output = image.resize(output_size, Image.Resampling.BILINEAR, box=box).convert("RGB")
            elif layer.endswith("class_index"):
                output = Image.new("L", output_size, 0)
            else:
                output = Image.new("RGB", output_size, (0, 0, 0))
        elif layer == "raw":
            with Image.open(Path(str(metadata["image_path"]))) as image:
                if target_size is None:
                    output = image.crop(box).convert("RGB")
                else:
                    output = image.resize(target_size, Image.Resampling.BILINEAR, box=box).convert("RGB")
        elif layer in {"mask", "class_index"}:
            with Image.open(review_mask_path(sample_dir, metadata)) as mask:
                if target_size is None:
                    class_tile = apply_patches_to_mask_tile(mask.crop(box), origin_x=box[0], origin_y=box[1], patches=patches)
                else:
                    resized = mask.resize(target_size, Image.Resampling.NEAREST, box=box)
                    class_tile = _apply_patches_to_resized_mask_tile(resized, source_box=box, patches=patches)
            output = class_tile if layer == "class_index" else class_index_to_color_image(class_tile, classes=classes)
        elif layer in {"intergrowth_mask", "intergrowth_class_index"}:
            if grid_mode:
                class_tile = _grid_intergrowth_class_tile(sample_dir, metadata, box, target_size)
            else:
                with Image.open(_metadata_artifact_path(metadata, "intergrowth_mask")) as mask:
                    if target_size is None:
                        class_tile = mask.crop(box).convert("L")
                    else:
                        class_tile = mask.resize(target_size, Image.Resampling.NEAREST, box=box).convert("L")
            output = (
                class_tile
                if layer == "intergrowth_class_index"
                else class_index_to_color_image(class_tile, classes=intergrowth_classes)
            )
        elif layer in {"intergrowth_normal_soft_mask", "intergrowth_hard_soft_mask"}:
            if grid_mode:
                artifact_name = (
                    "intergrowth_score_grid"
                    if layer == "intergrowth_normal_soft_mask"
                    else "intergrowth_hard_score_grid"
                )
            else:
                artifact_name = "intergrowth_score" if layer == "intergrowth_normal_soft_mask" else "intergrowth_hard_score"
            colormap = "viridis" if layer == "intergrowth_normal_soft_mask" else "magma"
            if grid_mode:
                score_tile = _resized_grid_tile(
                    metadata,
                    artifact_name,
                    box,
                    target_size,
                    resampling=Image.Resampling.BILINEAR,
                )
            else:
                with Image.open(_metadata_artifact_path(metadata, artifact_name)) as score:
                    if target_size is None:
                        score_tile = score.crop(box).convert("L")
                    else:
                        score_tile = score.resize(target_size, Image.Resampling.BILINEAR, box=box).convert("L")
            output = _colorize_soft_score(score_tile, colormap=colormap)
        elif layer == "overlay":
            with Image.open(Path(str(metadata["image_path"]))) as image:
                if target_size is None:
                    raw_tile = image.crop(box).convert("RGB")
                else:
                    raw_tile = image.resize(target_size, Image.Resampling.BILINEAR, box=box).convert("RGB")
            with Image.open(review_mask_path(sample_dir, metadata)) as mask:
                if target_size is None:
                    class_tile = apply_patches_to_mask_tile(mask.crop(box), origin_x=box[0], origin_y=box[1], patches=patches)
                else:
                    resized = mask.resize(target_size, Image.Resampling.NEAREST, box=box)
                    class_tile = _apply_patches_to_resized_mask_tile(resized, source_box=box, patches=patches)
            output = overlay_mask_on_image(raw_tile, class_tile, palette=palette).convert("RGB")
        elif layer == "intergrowth_overlay":
            with Image.open(Path(str(metadata["image_path"]))) as image:
                if target_size is None:
                    raw_tile = image.crop(box).convert("RGB")
                else:
                    raw_tile = image.resize(target_size, Image.Resampling.BILINEAR, box=box).convert("RGB")
            if grid_mode:
                class_tile = _grid_intergrowth_class_tile(sample_dir, metadata, box, target_size)
            else:
                with Image.open(_metadata_artifact_path(metadata, "intergrowth_mask")) as mask:
                    if target_size is None:
                        class_tile = mask.crop(box).convert("L")
                    else:
                        class_tile = mask.resize(target_size, Image.Resampling.NEAREST, box=box).convert("L")
            output = overlay_mask_on_image(raw_tile, class_tile, palette=intergrowth_palette).convert("RGB")
        elif layer in {"intergrowth_normal_soft_overlay", "intergrowth_hard_soft_overlay"}:
            if grid_mode:
                artifact_name = (
                    "intergrowth_score_grid"
                    if layer == "intergrowth_normal_soft_overlay"
                    else "intergrowth_hard_score_grid"
                )
            else:
                artifact_name = "intergrowth_score" if layer == "intergrowth_normal_soft_overlay" else "intergrowth_hard_score"
            colormap = "viridis" if layer == "intergrowth_normal_soft_overlay" else "magma"
            with Image.open(Path(str(metadata["image_path"]))) as image:
                if target_size is None:
                    raw_tile = image.crop(box).convert("RGB")
                else:
                    raw_tile = image.resize(target_size, Image.Resampling.BILINEAR, box=box).convert("RGB")
            if grid_mode:
                score_tile = _resized_grid_tile(
                    metadata,
                    artifact_name,
                    box,
                    target_size,
                    resampling=Image.Resampling.BILINEAR,
                )
                review_tile = _review_mask_tile_for_box(sample_dir, metadata, box, target_size)
                output = _blend_soft_score_on_review_mask(raw_tile, score_tile, review_tile, classes, colormap=colormap)
            else:
                with Image.open(_metadata_artifact_path(metadata, artifact_name)) as score:
                    if target_size is None:
                        score_tile = score.crop(box).convert("L")
                    else:
                        score_tile = score.resize(target_size, Image.Resampling.BILINEAR, box=box).convert("L")
                with Image.open(_metadata_artifact_path(metadata, "intergrowth_mask")) as mask:
                    if target_size is None:
                        class_tile = mask.crop(box).convert("L")
                    else:
                        class_tile = mask.resize(target_size, Image.Resampling.NEAREST, box=box).convert("L")
                output = _blend_soft_score_on_image(raw_tile, score_tile, class_tile, colormap=colormap)
        elif layer in {"confidence", "probability", "multiclass_confidence"}:
            artifact_name = {
                "confidence": "ore_confidence",
                "probability": "ore_probability",
                "multiclass_confidence": "ore_multiclass_confidence",
            }[layer]
            with Image.open(_metadata_artifact_path(metadata, artifact_name)) as image:
                if target_size is None:
                    output = image.crop(box).convert("L")
                else:
                    output = image.resize(target_size, Image.Resampling.BILINEAR, box=box).convert("L")
        elif layer in {"intergrowth_score", "intergrowth_confidence"}:
            artifact_name = "intergrowth_score_grid" if grid_mode and layer == "intergrowth_score" else layer
            if grid_mode and layer == "intergrowth_score":
                output = _resized_grid_tile(
                    metadata,
                    artifact_name,
                    box,
                    target_size,
                    resampling=Image.Resampling.BILINEAR,
                )
            else:
                with Image.open(_metadata_artifact_path(metadata, artifact_name)) as image:
                    if target_size is None:
                        output = image.crop(box).convert("L")
                    else:
                        output = image.resize(target_size, Image.Resampling.BILINEAR, box=box).convert("L")
        else:
            raise ValueError(f"unknown panorama tile layer: {layer}")

    return output


def _metrics_rows_from_area_metrics(area_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    counts = area_metrics.get("counts", {}) if isinstance(area_metrics.get("counts"), dict) else {}
    fractions = area_metrics.get("fractions", {}) if isinstance(area_metrics.get("fractions"), dict) else {}
    color_by_name = {item.name: list(item.color) for item in BASE_UI_CLASSES}
    rows: list[dict[str, Any]] = []
    for item in BASE_UI_CLASSES:
        count = int(counts.get(item.name, 0) or 0)
        if count <= 0:
            continue
        rows.append(
            {
                "id": item.id,
                "name": item.name,
                "color": color_by_name.get(item.name, [255, 255, 255]),
                "pixels": count,
                "fraction": float(fractions.get(item.name, 0.0) or 0.0),
            }
        )
    return rows


def _legend_rows(classes: Iterable[UiClass]) -> list[dict[str, Any]]:
    return [
        {
            "id": int(item.id),
            "name": item.name,
            "color": [int(channel) for channel in item.color],
        }
        for item in classes
    ]


def _intergrowth_not_ready_metrics(box: tuple[int, int, int, int]) -> dict[str, Any]:
    total = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    return {
        "ready": False,
        "intergrowth_ready": False,
        "message": "intergrowth not ready",
        "box": {"x": box[0], "y": box[1], "width": box[2] - box[0], "height": box[3] - box[1]},
        "total_pixels": total,
        "legend": _legend_rows(BASE_UI_CLASSES),
        "classes": [],
        "intergrowth_metrics": {
            "ore_pixels": 0,
            "normal_ore_pixels": 0,
            "hard_ore_pixels": 0,
            "normal_ore_fraction_of_ore": 0.0,
        },
        "approximate": False,
    }


def _sample_size_for_metrics(width: int, height: int) -> tuple[tuple[int, int], bool]:
    pixels = max(1, int(width) * int(height))
    if pixels <= INTERGROWTH_METRIC_SAMPLE_MAX_PIXELS:
        return (max(1, int(width)), max(1, int(height))), False
    scale = math.sqrt(INTERGROWTH_METRIC_SAMPLE_MAX_PIXELS / pixels)
    return (max(1, int(round(width * scale))), max(1, int(round(height * scale)))), True


def _grid_full_intergrowth_metrics(sample_dir: Path, metadata: dict[str, Any], box: tuple[int, int, int, int]) -> dict[str, Any]:
    metrics_path = _metadata_artifact_path(metadata, "intergrowth_metrics")
    cached = json.loads(metrics_path.read_text(encoding="utf-8"))
    area_metrics = cached.get("area_metrics", {})
    intergrowth_metrics = {
        "ore_pixels": int(area_metrics.get("metallic_ore_pixels", 0) or 0),
        "normal_ore_pixels": int(area_metrics.get("counts", {}).get("normal_ore", 0) or 0),
        "hard_ore_pixels": int(area_metrics.get("counts", {}).get("hard_ore", 0) or 0),
        "normal_ore_fraction_of_ore": float(area_metrics.get("normal_fraction_of_metallic_ore", 0.0) or 0.0),
    }
    return {
        "ready": True,
        "intergrowth_ready": True,
        "mode": "score_grid",
        "approximate": True,
        "box": {"x": box[0], "y": box[1], "width": box[2] - box[0], "height": box[3] - box[1]},
        "total_pixels": int(area_metrics.get("total_pixels", (box[2] - box[0]) * (box[3] - box[1])) or 0),
        "legend": _legend_rows(BASE_UI_CLASSES),
        "classes": _metrics_rows_from_area_metrics(area_metrics),
        "intergrowth_metrics": intergrowth_metrics,
        "score_metrics": cached.get("score_metrics", {}),
        "metrics_path": str(metrics_path),
        "sample_dir": str(sample_dir),
    }


def _grid_crop_intergrowth_metrics(
    sample_dir: Path,
    metadata: dict[str, Any],
    box: tuple[int, int, int, int],
) -> dict[str, Any]:
    width = box[2] - box[0]
    height = box[3] - box[1]
    sample_size, approximate = _sample_size_for_metrics(width, height)
    classes = _classes_from_metadata(metadata)
    cfg = _intergrowth_config_from_metadata(metadata)
    review_tile = _review_mask_tile_for_box(sample_dir, metadata, box, sample_size)
    normal_score_tile = _resized_grid_tile(
        metadata,
        "intergrowth_score_grid",
        box,
        sample_size,
        resampling=Image.Resampling.BILINEAR,
    )
    hard_score_tile = _resized_grid_tile(
        metadata,
        "intergrowth_hard_score_grid",
        box,
        sample_size,
        resampling=Image.Resampling.BILINEAR,
    )
    background_id = _background_class_id(classes)
    talc_id = _optional_class_id_for_name(classes, "talc")
    ignore_id = _optional_class_id_for_name(classes, "ignore")
    excluded = [background_id]
    if talc_id is not None:
        excluded.append(talc_id)
    if ignore_id is not None:
        excluded.append(ignore_id)
    threshold_u8 = int(round(cfg.normal_threshold * 255))
    review_values = np.asarray(review_tile.convert("L"), dtype=np.uint8)
    normal_score_values = np.asarray(normal_score_tile.convert("L"), dtype=np.uint8)
    hard_score_values = np.asarray(hard_score_tile.convert("L"), dtype=np.uint8)
    ignore_mask = (review_values == int(ignore_id)) if ignore_id is not None else np.zeros(review_values.shape, dtype=bool)
    talc_mask = (review_values == int(talc_id)) if talc_id is not None else np.zeros(review_values.shape, dtype=bool)
    ore_mask = ~np.isin(review_values, np.asarray(excluded, dtype=np.uint8))
    normal_mask = ore_mask & (normal_score_values > threshold_u8)
    hard_mask = ore_mask & ~normal_mask
    counts = {
        "background": int(np.count_nonzero(~ore_mask & ~talc_mask & ~ignore_mask)),
        "talc": int(np.count_nonzero(talc_mask)),
        "normal_ore": int(np.count_nonzero(normal_mask)),
        "hard_ore": int(np.count_nonzero(hard_mask)),
        "ignore": int(np.count_nonzero(ignore_mask)),
    }
    ore_sample_pixels = int(np.count_nonzero(ore_mask))
    normal_score_sum = float(normal_score_values[ore_mask].sum(dtype=np.uint64) / 255) if ore_sample_pixels else 0.0
    hard_score_sum = float(hard_score_values[ore_mask].sum(dtype=np.uint64) / 255) if ore_sample_pixels else 0.0

    sample_pixels = max(1, sample_size[0] * sample_size[1])
    crop_pixels = max(1, width * height)
    scale = crop_pixels / sample_pixels
    scaled_counts = {name: int(round(count * scale)) for name, count in counts.items()}
    area_metrics = _area_metrics_from_counts(scaled_counts, crop_pixels)
    ore_pixels = int(area_metrics["metallic_ore_pixels"])
    intergrowth_metrics = {
        "ore_pixels": ore_pixels,
        "normal_ore_pixels": int(area_metrics["counts"]["normal_ore"]),
        "hard_ore_pixels": int(area_metrics["counts"]["hard_ore"]),
        "normal_ore_fraction_of_ore": float(area_metrics["normal_fraction_of_metallic_ore"]),
    }
    return {
        "ready": True,
        "intergrowth_ready": True,
        "mode": "score_grid",
        "approximate": approximate,
        "sample_size": {"width": sample_size[0], "height": sample_size[1]},
        "box": {"x": box[0], "y": box[1], "width": width, "height": height},
        "total_pixels": crop_pixels,
        "legend": _legend_rows(BASE_UI_CLASSES),
        "classes": _metrics_rows_from_area_metrics(area_metrics),
        "intergrowth_metrics": intergrowth_metrics,
        "score_metrics": {
            "ore_pixels": int(round(ore_sample_pixels * scale)),
            "mean_erosion_ratio_score": (normal_score_sum / ore_sample_pixels) if ore_sample_pixels else 0.0,
            "mean_hard_score": (hard_score_sum / ore_sample_pixels) if ore_sample_pixels else 0.0,
        },
    }


def class_area_metrics(
    sample_dir: str | Path,
    *,
    x: int | None = None,
    y: int | None = None,
    width: int | None = None,
    height: int | None = None,
    layer: str = "prediction",
) -> dict[str, Any]:
    """Return area fractions for every non-zero class in a full mask or viewport."""
    sample_dir = Path(sample_dir)
    metadata = read_panorama_metadata(sample_dir)
    if x is None or y is None or width is None or height is None:
        box = (0, 0, int(metadata["image_width"]), int(metadata["image_height"]))
    else:
        box = _clamped_box(metadata, x=x, y=y, width=width, height=height)
    use_intergrowth = layer == "intergrowth"
    patches = [] if use_intergrowth else load_brush_patches(sample_dir)
    classes = tuple(BASE_UI_CLASSES) if use_intergrowth else _classes_from_metadata(metadata)
    name_by_id = {item.id: item.name for item in classes}
    full_image_box = (
        box[0] == 0
        and box[1] == 0
        and box[2] == int(metadata["image_width"])
        and box[3] == int(metadata["image_height"])
    )
    if use_intergrowth:
        if not _intergrowth_ready(metadata):
            return _intergrowth_not_ready_metrics(box)
        if _intergrowth_grid_mode(metadata):
            if full_image_box:
                return _grid_full_intergrowth_metrics(sample_dir, metadata, box)
            return _grid_crop_intergrowth_metrics(sample_dir, metadata, box)
    with allow_large_pillow_images():
        mask_path = _metadata_artifact_path(metadata, "intergrowth_mask") if use_intergrowth else review_mask_path(sample_dir, metadata)
        with Image.open(mask_path) as mask:
            if patches:
                class_tile = apply_patches_to_mask_tile(mask.crop(box), origin_x=box[0], origin_y=box[1], patches=patches)
            else:
                class_tile = mask.crop(box).convert("L")
    total = max(1, class_tile.width * class_tile.height)
    class_array = np.asarray(class_tile.convert("L"), dtype=np.uint8)
    histogram = np.bincount(class_array.ravel(), minlength=256)
    counts = {class_id: int(count) for class_id, count in enumerate(histogram) if class_id != 0 and int(count) > 0}
    result = {
        "box": {"x": box[0], "y": box[1], "width": box[2] - box[0], "height": box[3] - box[1]},
        "total_pixels": total,
        "legend": _legend_rows(classes),
        "classes": [
            {
                "id": class_id,
                "name": name_by_id.get(class_id, f"class_{class_id}"),
                "color": list(next((item.color for item in classes if item.id == class_id), (255, 255, 255))),
                "pixels": count,
                "fraction": count / total,
            }
            for class_id, count in sorted(counts.items())
        ],
    }
    if use_intergrowth:
        score_path = _optional_metadata_artifact_path(metadata, "intergrowth_score")
        hard_score_path = _optional_metadata_artifact_path(metadata, "intergrowth_hard_score")
        normal_id = next(item.id for item in BASE_UI_CLASSES if item.name == "normal_ore")
        hard_id = next(item.id for item in BASE_UI_CLASSES if item.name == "hard_ore")
        normal_pixels = counts.get(normal_id, 0)
        hard_pixels = counts.get(hard_id, 0)
        ore_pixels = normal_pixels + hard_pixels
        result["intergrowth_metrics"] = {
            "ore_pixels": ore_pixels,
            "normal_ore_pixels": normal_pixels,
            "hard_ore_pixels": hard_pixels,
            "normal_ore_fraction_of_ore": (normal_pixels / ore_pixels) if ore_pixels else 0.0,
        }
        if score_path is not None and score_path.exists():
            ore_mask = np.isin(class_array, np.asarray([normal_id, hard_id], dtype=np.uint8))
            ore_index_count = int(np.count_nonzero(ore_mask))
            with Image.open(score_path) as opened_score:
                normal_scores = np.asarray(opened_score.crop(box).convert("L"), dtype=np.uint8)
            if hard_score_path is not None and hard_score_path.exists():
                with Image.open(hard_score_path) as opened_hard:
                    hard_scores = np.asarray(opened_hard.crop(box).convert("L"), dtype=np.uint8)
            else:
                hard_scores = (255 - normal_scores).astype(np.uint8)
            if ore_index_count:
                result["score_metrics"] = {
                    "ore_pixels": ore_index_count,
                    "mean_erosion_ratio_score": float(normal_scores[ore_mask].mean() / 255),
                    "mean_hard_score": float(hard_scores[ore_mask].mean() / 255),
                }
            else:
                result["score_metrics"] = {
                    "ore_pixels": 0,
                    "mean_erosion_ratio_score": 0.0,
                    "mean_hard_score": 0.0,
                }
    return result


def _patch_crop_box(metadata: dict[str, Any], patch: BrushPatch, crop_size: int) -> tuple[int, int, int, int]:
    image_width = int(metadata["image_width"])
    image_height = int(metadata["image_height"])
    half = max(1, crop_size // 2)
    left = max(0, min(image_width - 1, patch.x - half))
    top = max(0, min(image_height - 1, patch.y - half))
    right = min(image_width, left + crop_size)
    bottom = min(image_height, top + crop_size)
    left = max(0, right - crop_size)
    top = max(0, bottom - crop_size)
    return (left, top, right, bottom)


def _save_patch_tensors(
    *,
    source_mask: Image.Image,
    metadata: dict[str, Any],
    output_dir: Path,
    patches: list[BrushPatch],
    classes: tuple[UiClass, ...],
    crop_size: int,
) -> list[str]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required to save active-learning panorama crop tensors") from exc

    tensor_dir = output_dir / "reviewed_tiles"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    channel_classes = [item for item in classes if int(item.id) != 255]
    channel_ids = [int(item.id) for item in channel_classes]
    written: list[str] = []
    for index, patch in enumerate(patches):
        crop_box = _patch_crop_box(metadata, patch, crop_size)
        crop = source_mask.crop(crop_box).convert("L")
        values = torch.from_numpy(np.array(crop, dtype=np.uint8, copy=True)).to(dtype=torch.int64).view(crop.height, crop.width)
        one_hot = torch.stack([(values == class_id).to(dtype=torch.uint8) for class_id in channel_ids], dim=0)
        path = tensor_dir / f"patch_{index + 1:05d}.pt"
        torch.save(
            {
                "class_index": values.to(dtype=torch.uint8),
                "one_hot": one_hot,
                "channel_class_ids": channel_ids,
                "channel_class_names": [item.name for item in channel_classes],
                "crop_box": crop_box,
                "patch": patch.as_dict(),
                "source_image_path": str(metadata["image_path"]),
                "source_panorama_metadata": str(Path(output_dir) / "metadata.json"),
            },
            path,
        )
        written.append(str(path))
    return written


def save_panorama_review(
    sample_dir: str | Path,
    *,
    output_root: str | Path,
    classes: Iterable[dict[str, Any]] | None = None,
    crop_size: int = 512,
) -> dict[str, Any]:
    """Apply brush patches once and save active-learning review artifacts."""
    sample_dir = Path(sample_dir)
    metadata = read_panorama_metadata(sample_dir)
    patches = load_brush_patches(sample_dir)
    class_objects = _classes_from_metadata(metadata, classes)
    output_root = Path(output_root)
    sample_id = str(metadata.get("sample_id") or sample_dir.name)
    output_dir = output_root / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)

    class_index_path = output_dir / "class_index_mask.png"
    metadata_path = output_dir / "metadata.json"
    patch_log_copy_path = output_dir / "patch_log.jsonl"
    preview_path = output_dir / "mask_preview.png"

    with allow_large_pillow_images():
        with Image.open(review_mask_path(sample_dir, metadata)) as opened:
            mask = opened.convert("L")
            if patches:
                mask = apply_patches_to_mask_tile(mask, origin_x=0, origin_y=0, patches=patches)
            mask.save(class_index_path, compress_level=1)
            preview_size = 1600
            scale = preview_size / max(1, max(mask.width, mask.height))
            if scale < 1:
                preview = mask.resize(
                    (max(1, int(math.floor(mask.width * scale))), max(1, int(math.floor(mask.height * scale)))),
                    Image.Resampling.NEAREST,
                )
            else:
                preview = mask.copy()
            class_index_to_color_image(preview, classes=class_objects).save(preview_path)
            tensor_tiles = _save_patch_tensors(
                source_mask=mask,
                metadata=metadata,
                output_dir=output_dir,
                patches=patches,
                classes=class_objects,
                crop_size=crop_size,
            )

    if patch_log_path(sample_dir).exists():
        shutil.copy2(patch_log_path(sample_dir), patch_log_copy_path)
    else:
        patch_log_copy_path.write_text("", encoding="utf-8")

    review_metadata = {
        "source_panorama_sample_dir": str(sample_dir),
        "source_image_path": str(metadata["image_path"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format": "single_class_index_png_patch_log_and_patch_crop_tensors",
        "full_one_hot_tensor_saved": False,
        "class_index_mask": str(class_index_path),
        "mask_preview": str(preview_path),
        "patch_log": str(patch_log_copy_path),
        "patch_count": len(patches),
        "tensor_tiles": tensor_tiles,
        "classes": [item.as_dict() for item in class_objects],
        "source_prediction_metadata": str(sample_dir / "metadata.json"),
    }
    metadata_path.write_text(json.dumps(review_metadata, indent=2), encoding="utf-8")
    review_metadata["metadata_path"] = str(metadata_path)
    return review_metadata
