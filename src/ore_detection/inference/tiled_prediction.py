"""Tiled panorama inference for large optical microscopy images."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable, Iterable

from PIL import Image

from ore_detection.inference.model_prediction import (
    LoadedSegmentationModel,
    _jsonable,
    _require_torch,
    binary_prediction_from_logits,
    multiclass_prediction_from_logits,
)
from ore_detection.visualization.overlay import colorize_mask, overlay_mask_on_image


ProgressCallback = Callable[[dict[str, Any]], None]


class PanoramaPredictionCancelled(RuntimeError):
    """Raised when a long-running panorama prediction job is cancelled."""


@dataclass(frozen=True)
class TileGridConfig:
    """Panorama tiling parameters."""

    tile_size: int = 512
    overlap: int = 128

    @property
    def stride(self) -> int:
        return self.tile_size - self.overlap

    def validate(self) -> None:
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.tile_size:
            raise ValueError("overlap must be smaller than tile_size")


@dataclass(frozen=True)
class Tile:
    """One source crop and its stable paste region."""

    index: int
    x: int
    y: int
    width: int
    height: int
    image_width: int
    image_height: int
    overlap: int
    stable_left: int
    stable_top: int
    stable_right: int
    stable_bottom: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def stable_crop_box(self) -> tuple[int, int, int, int]:
        return (
            self.stable_left,
            self.stable_top,
            max(self.stable_left + 1, self.stable_right),
            max(self.stable_top + 1, self.stable_bottom),
        )

    @property
    def stable_paste_xy(self) -> tuple[int, int]:
        left, top, _, _ = self.stable_crop_box
        return (self.x + left, self.y + top)


@dataclass(frozen=True)
class PanoramaPredictionArtifacts:
    """Artifacts written by a tiled panorama prediction."""

    sample_dir: Path
    ore_mask_path: Path
    ore_probability_path: Path
    ore_confidence_path: Path
    raw_preview_path: Path
    mask_preview_path: Path
    overlay_preview_path: Path
    metadata_path: Path
    multiclass_mask_path: Path | None = None
    multiclass_confidence_path: Path | None = None


def _axis_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= 0:
        raise ValueError("image dimensions must be positive")
    if length <= tile_size:
        return [0]
    last = length - tile_size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_tile_grid(width: int, height: int, *, tile_size: int = 512, overlap: int = 128) -> list[Tile]:
    """Return tiles that cover an image with overlap and stable paste regions."""
    config = TileGridConfig(tile_size=tile_size, overlap=overlap)
    config.validate()
    xs = _axis_starts(width, config.tile_size, config.stride)
    ys = _axis_starts(height, config.tile_size, config.stride)

    def stable_bounds(starts: list[int], length: int, index: int) -> tuple[int, int]:
        start = starts[index]
        tile_length = min(config.tile_size, length - start)
        if index == 0:
            left_abs = start
        else:
            prev_start = starts[index - 1]
            prev_length = min(config.tile_size, length - prev_start)
            left_abs = (start + prev_start + prev_length) // 2
        if index == len(starts) - 1:
            right_abs = start + tile_length
        else:
            next_start = starts[index + 1]
            right_abs = (next_start + start + tile_length) // 2
        return (max(0, left_abs - start), min(tile_length, right_abs - start))

    tiles: list[Tile] = []
    for y_index, y in enumerate(ys):
        for x_index, x in enumerate(xs):
            stable_left, stable_right = stable_bounds(xs, width, x_index)
            stable_top, stable_bottom = stable_bounds(ys, height, y_index)
            tiles.append(
                Tile(
                    index=len(tiles),
                    x=x,
                    y=y,
                    width=min(config.tile_size, width - x),
                    height=min(config.tile_size, height - y),
                    image_width=width,
                    image_height=height,
                    overlap=overlap,
                    stable_left=stable_left,
                    stable_top=stable_top,
                    stable_right=stable_right,
                    stable_bottom=stable_bottom,
                )
            )
    return tiles


def batched(items: Iterable[Tile], batch_size: int) -> Iterable[list[Tile]]:
    """Yield fixed-size batches from an iterable."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: list[Tile] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@contextmanager
def allow_large_pillow_images():
    """Temporarily disable PIL's decompression-bomb limit for trusted lab panoramas."""
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def _safe_sample_id(image_path: str | Path) -> str:
    path = Path(image_path)
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{path.stem}-{digest}"


def _fit_size(width: int, height: int, max_size: int) -> tuple[int, int]:
    if max(width, height) <= max_size:
        return (width, height)
    scale = max_size / max(width, height)
    return (max(1, int(round(width * scale))), max(1, int(round(height * scale))))


def _crop_padded_tile(image: Image.Image, tile: Tile, *, tile_size: int) -> Image.Image:
    crop = image.crop(tile.box).convert("RGB")
    if crop.size == (tile_size, tile_size):
        return crop
    padded = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
    padded.paste(crop, (0, 0))
    return padded


def _batch_to_tensor(tile_images: list[Image.Image], metadata: Any, *, device: Any, input_size: int):
    torch, _ = _require_torch()
    tensors = []
    for image in tile_images:
        rgb = image.convert("RGB")
        if rgb.size != (input_size, input_size):
            rgb = rgb.resize((input_size, input_size), Image.Resampling.BILINEAR)
        data = torch.tensor(list(rgb.tobytes()), dtype=torch.float32, device=device)
        tensor = data.view(input_size, input_size, 3).permute(2, 0, 1) / 255.0
        tensors.append(tensor)
    batch = torch.stack(tensors, dim=0)
    mean = torch.tensor(metadata.normalization_mean, dtype=batch.dtype, device=device).view(1, 3, 1, 1)
    std = torch.tensor(metadata.normalization_std, dtype=batch.dtype, device=device).view(1, 3, 1, 1).clamp_min(1e-6)
    return (batch - mean) / std


def _resize_logits_if_needed(logits: Any, *, tile_size: int):
    _, F = _require_torch()
    if int(logits.shape[-2]) == tile_size and int(logits.shape[-1]) == tile_size:
        return logits
    return F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)


def _single_channel_tile_images(tensor: Any, *, scale: int = 255) -> list[Image.Image]:
    torch, _ = _require_torch()
    if tensor.ndim == 4:
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(f"expected [B,H,W] or [B,1,H,W] tensor, got {tuple(tensor.shape)}")
    images: list[Image.Image] = []
    values = (tensor.detach().cpu().float().clamp(0.0, 1.0) * scale).round().to(dtype=torch.uint8)
    for index in range(int(values.shape[0])):
        tile = values[index]
        image = Image.new("L", (int(tile.shape[1]), int(tile.shape[0])))
        image.putdata(tile.flatten().tolist())
        images.append(image)
    return images


def _class_index_tile_images(class_index: Any) -> list[Image.Image]:
    torch, _ = _require_torch()
    if class_index.ndim != 3:
        raise ValueError(f"expected class_index shaped [B,H,W], got {tuple(class_index.shape)}")
    if int(class_index.max().detach().cpu()) > 255:
        raise ValueError("class_index contains values above 255 and cannot be stored as an 8-bit image")
    values = class_index.detach().cpu().to(dtype=torch.uint8)
    images: list[Image.Image] = []
    for index in range(int(values.shape[0])):
        tile = values[index]
        image = Image.new("L", (int(tile.shape[1]), int(tile.shape[0])))
        image.putdata(tile.flatten().tolist())
        images.append(image)
    return images


def _paste_stable_region(target: Image.Image, tile_image: Image.Image, tile: Tile) -> None:
    crop_box = tile.stable_crop_box
    paste_xy = tile.stable_paste_xy
    target.paste(tile_image.crop(crop_box), paste_xy)


def _save_png_l(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, compress_level=1)


def _make_previews(
    *,
    image: Image.Image,
    review_mask: Image.Image,
    sample_dir: Path,
    preview_max_size: int,
) -> tuple[Path, Path, Path]:
    raw_preview_path = sample_dir / "raw_preview.jpg"
    mask_preview_path = sample_dir / "mask_preview.png"
    overlay_preview_path = sample_dir / "overlay_preview.png"
    preview_size = _fit_size(image.width, image.height, preview_max_size)
    raw_preview = image.resize(preview_size, Image.Resampling.BILINEAR).convert("RGB")
    mask_preview = review_mask.resize(preview_size, Image.Resampling.NEAREST)
    raw_preview.save(raw_preview_path, quality=85)
    colorize_mask(mask_preview).save(mask_preview_path)
    overlay_mask_on_image(raw_preview, mask_preview).save(overlay_preview_path)
    return raw_preview_path, mask_preview_path, overlay_preview_path


def _update_progress(callback: ProgressCallback | None, **values: Any) -> None:
    if callback is not None:
        callback(values)


def _check_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise PanoramaPredictionCancelled("panorama prediction was cancelled")


def save_tiled_segmentation_prediction(
    image_path: str | Path,
    *,
    binary_model: LoadedSegmentationModel,
    ore_model: LoadedSegmentationModel | None = None,
    output_root: str | Path = "data_work/predictions/panorama",
    binary_threshold: float = 0.5,
    tile_size: int = 512,
    overlap: int = 128,
    batch_size: int = 4,
    sample_id: str | None = None,
    preview_max_size: int = 1600,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> PanoramaPredictionArtifacts:
    """Run tiled binary and optional ore segmentation on a large image."""
    torch, _ = _require_torch()
    TileGridConfig(tile_size=tile_size, overlap=overlap).validate()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    image_path = Path(image_path)
    output_root = Path(output_root)
    resolved_sample_id = sample_id or _safe_sample_id(image_path)
    sample_dir = output_root / resolved_sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    _update_progress(progress_callback, phase="opening_image", processed_tiles=0, timings={})
    timings: dict[str, float] = {"cpu_prepare_sec": 0.0, "gpu_inference_sec": 0.0, "stitch_write_sec": 0.0}

    with allow_large_pillow_images():
        with Image.open(image_path) as opened:
            width, height = opened.size
            tiles = make_tile_grid(width, height, tile_size=tile_size, overlap=overlap)
            total_batches = math.ceil(len(tiles) / batch_size)
            _update_progress(
                progress_callback,
                phase="initializing_outputs",
                image_width=width,
                image_height=height,
                total_tiles=len(tiles),
                total_batches=total_batches,
            )
            ore_mask = Image.new("L", (width, height), 0)
            ore_probability = Image.new("L", (width, height), 0)
            ore_confidence = Image.new("L", (width, height), 0)
            multiclass_mask = Image.new("L", (width, height), 0) if ore_model is not None else None
            multiclass_confidence = Image.new("L", (width, height), 0) if ore_model is not None else None

            processed_tiles = 0
            processed_batches = 0
            _update_progress(progress_callback, phase="binary_inference", processed_tiles=0, processed_batches=0)

            for tile_batch in batched(tiles, batch_size):
                _check_cancelled(cancel_event)
                cpu_start = time.perf_counter()
                tile_images = [_crop_padded_tile(opened, tile, tile_size=tile_size) for tile in tile_batch]
                input_size = binary_model.metadata.image_size or tile_size
                binary_input = _batch_to_tensor(
                    tile_images,
                    binary_model.metadata,
                    device=binary_model.device,
                    input_size=input_size,
                )
                timings["cpu_prepare_sec"] += time.perf_counter() - cpu_start

                gpu_start = time.perf_counter()
                with torch.no_grad():
                    binary_logits = _resize_logits_if_needed(binary_model.model(binary_input), tile_size=tile_size)
                    binary_prediction = binary_prediction_from_logits(binary_logits, threshold=binary_threshold)
                    mask_tiles = _single_channel_tile_images(binary_prediction.mask, scale=1)
                    probability_tiles = _single_channel_tile_images(binary_prediction.probability, scale=255)
                    confidence_tiles = _single_channel_tile_images(binary_prediction.confidence, scale=255)

                    multiclass_tiles = None
                    multiclass_confidence_tiles = None
                    if ore_model is not None:
                        ore_input_size = ore_model.metadata.image_size or tile_size
                        ore_input = (
                            binary_input
                            if ore_input_size == input_size and ore_model.device == binary_model.device
                            else _batch_to_tensor(tile_images, ore_model.metadata, device=ore_model.device, input_size=ore_input_size)
                        )
                        ore_logits = _resize_logits_if_needed(ore_model.model(ore_input), tile_size=tile_size)
                        ore_prediction = multiclass_prediction_from_logits(
                            ore_logits,
                            class_names=ore_model.metadata.class_names,
                            background_index=ore_model.metadata.background_index or 0,
                            clip_binary_mask=binary_prediction.mask,
                        )
                        multiclass_tiles = _class_index_tile_images(ore_prediction.class_index)
                        multiclass_confidence_tiles = _single_channel_tile_images(ore_prediction.class_probability, scale=255)
                timings["gpu_inference_sec"] += time.perf_counter() - gpu_start

                stitch_start = time.perf_counter()
                for index, tile in enumerate(tile_batch):
                    _paste_stable_region(ore_mask, mask_tiles[index], tile)
                    _paste_stable_region(ore_probability, probability_tiles[index], tile)
                    _paste_stable_region(ore_confidence, confidence_tiles[index], tile)
                    if multiclass_mask is not None and multiclass_confidence is not None:
                        assert multiclass_tiles is not None
                        assert multiclass_confidence_tiles is not None
                        _paste_stable_region(multiclass_mask, multiclass_tiles[index], tile)
                        _paste_stable_region(multiclass_confidence, multiclass_confidence_tiles[index], tile)
                timings["stitch_write_sec"] += time.perf_counter() - stitch_start

                processed_tiles += len(tile_batch)
                processed_batches += 1
                _update_progress(
                    progress_callback,
                    phase="binary_ore_inference" if ore_model is not None else "binary_inference",
                    processed_tiles=processed_tiles,
                    processed_batches=processed_batches,
                    timings=timings.copy(),
                )

            _check_cancelled(cancel_event)
            _update_progress(progress_callback, phase="writing_artifacts", processed_tiles=processed_tiles)
            write_start = time.perf_counter()
            ore_mask_path = sample_dir / "ore_mask.png"
            ore_probability_path = sample_dir / "ore_probability.png"
            ore_confidence_path = sample_dir / "ore_confidence.png"
            multiclass_mask_path = sample_dir / "ore_multiclass_mask.png" if multiclass_mask is not None else None
            multiclass_confidence_path = (
                sample_dir / "ore_multiclass_confidence.png" if multiclass_confidence is not None else None
            )
            _save_png_l(ore_mask, ore_mask_path)
            _save_png_l(ore_probability, ore_probability_path)
            _save_png_l(ore_confidence, ore_confidence_path)
            if multiclass_mask is not None and multiclass_mask_path is not None:
                _save_png_l(multiclass_mask, multiclass_mask_path)
            if multiclass_confidence is not None and multiclass_confidence_path is not None:
                _save_png_l(multiclass_confidence, multiclass_confidence_path)

            review_mask = multiclass_mask if multiclass_mask is not None else ore_mask
            raw_preview_path, mask_preview_path, overlay_preview_path = _make_previews(
                image=opened,
                review_mask=review_mask,
                sample_dir=sample_dir,
                preview_max_size=preview_max_size,
            )
            timings["stitch_write_sec"] += time.perf_counter() - write_start

    metadata_path = sample_dir / "metadata.json"
    metadata = {
        "method": "tiled_trained_binary_ore_segmentation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": resolved_sample_id,
        "image_path": str(image_path),
        "image_width": width,
        "image_height": height,
        "tile_size": tile_size,
        "overlap": overlap,
        "stride": tile_size - overlap,
        "batch_size": batch_size,
        "total_tiles": len(tiles),
        "total_batches": total_batches,
        "binary_threshold": binary_threshold,
        "policy": {
            "binary_outline_is_primary": True,
            "multiclass_clipped_to_binary_mask": ore_model is not None,
            "talc_prediction_enabled": False,
            "talc_policy": "manual_ui_annotation_only",
        },
        "timings": timings,
        "binary_checkpoint": binary_model.metadata.as_dict(),
        "ore_checkpoint": ore_model.metadata.as_dict() if ore_model is not None else None,
        "artifacts": {
            "ore_mask": str(ore_mask_path),
            "ore_probability": str(ore_probability_path),
            "ore_confidence": str(ore_confidence_path),
            "ore_multiclass_mask": str(multiclass_mask_path) if multiclass_mask_path is not None else None,
            "ore_multiclass_confidence": str(multiclass_confidence_path) if multiclass_confidence_path is not None else None,
            "raw_preview": str(raw_preview_path),
            "mask_preview": str(mask_preview_path),
            "overlay_preview": str(overlay_preview_path),
            "review_mask": str(multiclass_mask_path or ore_mask_path),
        },
    }
    metadata_path.write_text(json.dumps(_jsonable(metadata), indent=2), encoding="utf-8")
    _update_progress(
        progress_callback,
        phase="completed",
        processed_tiles=len(tiles),
        processed_batches=total_batches,
        timings=timings.copy(),
        artifacts=metadata["artifacts"],
    )

    return PanoramaPredictionArtifacts(
        sample_dir=sample_dir,
        ore_mask_path=ore_mask_path,
        ore_probability_path=ore_probability_path,
        ore_confidence_path=ore_confidence_path,
        raw_preview_path=raw_preview_path,
        mask_preview_path=mask_preview_path,
        overlay_preview_path=overlay_preview_path,
        metadata_path=metadata_path,
        multiclass_mask_path=multiclass_mask_path,
        multiclass_confidence_path=multiclass_confidence_path,
    )


def save_tiled_selected_model_prediction(
    image_path: str | Path,
    *,
    model: LoadedSegmentationModel,
    model_kind: str,
    output_root: str | Path = "data_work/predictions/panorama",
    binary_threshold: float = 0.5,
    tile_size: int = 512,
    overlap: int = 128,
    batch_size: int = 4,
    sample_id: str | None = None,
    preview_max_size: int = 1600,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> PanoramaPredictionArtifacts:
    """Run one selected model over a panorama without chaining models."""
    torch, _ = _require_torch()
    if model_kind not in {"binary", "ore"}:
        raise ValueError("model_kind must be `binary` or `ore`")
    TileGridConfig(tile_size=tile_size, overlap=overlap).validate()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    image_path = Path(image_path)
    output_root = Path(output_root)
    resolved_sample_id = sample_id or _safe_sample_id(image_path)
    sample_dir = output_root / resolved_sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {"cpu_prepare_sec": 0.0, "gpu_inference_sec": 0.0, "stitch_write_sec": 0.0}

    _update_progress(progress_callback, phase="opening_image", processed_tiles=0, timings={})
    with allow_large_pillow_images():
        with Image.open(image_path) as opened:
            width, height = opened.size
            tiles = make_tile_grid(width, height, tile_size=tile_size, overlap=overlap)
            total_batches = math.ceil(len(tiles) / batch_size)
            _update_progress(
                progress_callback,
                phase="initializing_outputs",
                image_width=width,
                image_height=height,
                total_tiles=len(tiles),
                total_batches=total_batches,
            )

            if model_kind == "binary":
                mask_image = Image.new("L", (width, height), 0)
                probability_image = Image.new("L", (width, height), 0)
                confidence_image = Image.new("L", (width, height), 0)
                multiclass_mask = None
                multiclass_confidence = None
            else:
                mask_image = Image.new("L", (width, height), 0)
                probability_image = None
                confidence_image = Image.new("L", (width, height), 0)
                multiclass_mask = mask_image
                multiclass_confidence = confidence_image

            processed_tiles = 0
            processed_batches = 0
            _update_progress(progress_callback, phase=f"{model_kind}_inference", processed_tiles=0, processed_batches=0)

            for tile_batch in batched(tiles, batch_size):
                _check_cancelled(cancel_event)
                cpu_start = time.perf_counter()
                tile_images = [_crop_padded_tile(opened, tile, tile_size=tile_size) for tile in tile_batch]
                input_size = model.metadata.image_size or tile_size
                model_input = _batch_to_tensor(tile_images, model.metadata, device=model.device, input_size=input_size)
                timings["cpu_prepare_sec"] += time.perf_counter() - cpu_start

                gpu_start = time.perf_counter()
                with torch.no_grad():
                    logits = _resize_logits_if_needed(model.model(model_input), tile_size=tile_size)
                    if model_kind == "binary":
                        prediction = binary_prediction_from_logits(logits, threshold=binary_threshold)
                        mask_tiles = _single_channel_tile_images(prediction.mask, scale=1)
                        probability_tiles = _single_channel_tile_images(prediction.probability, scale=255)
                        confidence_tiles = _single_channel_tile_images(prediction.confidence, scale=255)
                    else:
                        prediction = multiclass_prediction_from_logits(
                            logits,
                            class_names=model.metadata.class_names,
                            background_index=model.metadata.background_index or 0,
                        )
                        mask_tiles = _class_index_tile_images(prediction.class_index)
                        probability_tiles = None
                        confidence_tiles = _single_channel_tile_images(prediction.class_probability, scale=255)
                timings["gpu_inference_sec"] += time.perf_counter() - gpu_start

                stitch_start = time.perf_counter()
                for index, tile in enumerate(tile_batch):
                    _paste_stable_region(mask_image, mask_tiles[index], tile)
                    if probability_image is not None and probability_tiles is not None:
                        _paste_stable_region(probability_image, probability_tiles[index], tile)
                    _paste_stable_region(confidence_image, confidence_tiles[index], tile)
                timings["stitch_write_sec"] += time.perf_counter() - stitch_start

                processed_tiles += len(tile_batch)
                processed_batches += 1
                _update_progress(
                    progress_callback,
                    phase=f"{model_kind}_inference",
                    processed_tiles=processed_tiles,
                    processed_batches=processed_batches,
                    timings=timings.copy(),
                )

            _check_cancelled(cancel_event)
            _update_progress(progress_callback, phase="writing_artifacts", processed_tiles=processed_tiles)
            write_start = time.perf_counter()
            if model_kind == "binary":
                ore_mask_path = sample_dir / "ore_mask.png"
                ore_probability_path = sample_dir / "ore_probability.png"
                ore_confidence_path = sample_dir / "ore_confidence.png"
                multiclass_mask_path = None
                multiclass_confidence_path = None
                _save_png_l(mask_image, ore_mask_path)
                assert probability_image is not None
                _save_png_l(probability_image, ore_probability_path)
                _save_png_l(confidence_image, ore_confidence_path)
                review_mask_path_value = ore_mask_path
            else:
                ore_mask_path = sample_dir / "ore_multiclass_mask.png"
                ore_probability_path = sample_dir / "ore_multiclass_confidence.png"
                ore_confidence_path = sample_dir / "ore_multiclass_confidence.png"
                multiclass_mask_path = ore_mask_path
                multiclass_confidence_path = ore_confidence_path
                _save_png_l(mask_image, multiclass_mask_path)
                _save_png_l(confidence_image, multiclass_confidence_path)
                review_mask_path_value = multiclass_mask_path

            base_prediction_mask_path = sample_dir / "base_prediction_mask.png"
            shutil.copy2(review_mask_path_value, base_prediction_mask_path)
            raw_preview_path, mask_preview_path, overlay_preview_path = _make_previews(
                image=opened,
                review_mask=mask_image,
                sample_dir=sample_dir,
                preview_max_size=preview_max_size,
            )
            timings["stitch_write_sec"] += time.perf_counter() - write_start

    metadata_path = sample_dir / "metadata.json"
    talc_prediction_enabled = model_kind == "ore" and any(
        str(name).strip().lower() == "talc" for name in model.metadata.class_names
    )
    artifacts = {
        "ore_mask": str(ore_mask_path) if model_kind == "binary" else None,
        "ore_probability": str(ore_probability_path) if model_kind == "binary" else None,
        "ore_confidence": str(ore_confidence_path) if model_kind == "binary" else None,
        "ore_multiclass_mask": str(multiclass_mask_path) if multiclass_mask_path is not None else None,
        "ore_multiclass_confidence": str(multiclass_confidence_path) if multiclass_confidence_path is not None else None,
        "raw_preview": str(raw_preview_path),
        "mask_preview": str(mask_preview_path),
        "overlay_preview": str(overlay_preview_path),
        "review_mask": str(review_mask_path_value),
        "base_prediction_mask": str(base_prediction_mask_path),
    }
    metadata = {
        "method": f"tiled_selected_{model_kind}_segmentation",
        "model_kind": model_kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": resolved_sample_id,
        "image_path": str(image_path),
        "image_width": width,
        "image_height": height,
        "tile_size": tile_size,
        "overlap": overlap,
        "stride": tile_size - overlap,
        "batch_size": batch_size,
        "total_tiles": len(tiles),
        "total_batches": total_batches,
        "binary_threshold": binary_threshold if model_kind == "binary" else None,
        "policy": {
            "selected_model_only": True,
            "binary_outline_is_primary": model_kind == "binary",
            "multiclass_clipped_to_binary_mask": False,
            "talc_prediction_enabled": talc_prediction_enabled,
            "talc_policy": "checkpoint_class" if talc_prediction_enabled else "manual_ui_annotation_only",
        },
        "timings": timings,
        "binary_checkpoint": model.metadata.as_dict() if model_kind == "binary" else None,
        "ore_checkpoint": model.metadata.as_dict() if model_kind == "ore" else None,
        "artifacts": artifacts,
    }
    metadata_path.write_text(json.dumps(_jsonable(metadata), indent=2), encoding="utf-8")
    _update_progress(
        progress_callback,
        phase="completed",
        processed_tiles=len(tiles),
        processed_batches=total_batches,
        timings=timings.copy(),
        artifacts=artifacts,
    )

    return PanoramaPredictionArtifacts(
        sample_dir=sample_dir,
        ore_mask_path=ore_mask_path,
        ore_probability_path=ore_probability_path,
        ore_confidence_path=ore_confidence_path,
        raw_preview_path=raw_preview_path,
        mask_preview_path=mask_preview_path,
        overlay_preview_path=overlay_preview_path,
        metadata_path=metadata_path,
        multiclass_mask_path=multiclass_mask_path,
        multiclass_confidence_path=multiclass_confidence_path,
    )
