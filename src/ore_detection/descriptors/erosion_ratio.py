"""Erosion-ratio hard/normal ore metric."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from ore_detection.descriptors.intergrowth_classification import (
    BACKGROUND_ID,
    HARD_ORE_ID,
    IGNORE_ID,
    NORMAL_ORE_ID,
    TALC_ID,
    intergrowth_area_metrics,
)


@dataclass(frozen=True)
class ErosionRatioConfig:
    """Configuration for local erosion-ratio scoring."""

    erosion_kernel_size: int = 5
    erosion_iterations: int = 1
    window_size: int = 128
    normal_threshold: float = 0.5
    min_ore_fraction: float = 0.05
    background_id: int = BACKGROUND_ID
    talc_id: int = TALC_ID
    normal_ore_id: int = NORMAL_ORE_ID
    hard_ore_id: int = HARD_ORE_ID
    ignore_id: int = IGNORE_ID

    def validate(self) -> None:
        if self.erosion_kernel_size <= 0 or self.erosion_kernel_size % 2 == 0:
            raise ValueError("erosion_kernel_size must be a positive odd integer")
        if self.erosion_iterations < 0:
            raise ValueError("erosion_iterations must be non-negative")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 <= self.normal_threshold <= 1.0:
            raise ValueError("normal_threshold must be in [0, 1]")
        if not 0.0 <= self.min_ore_fraction <= 1.0:
            raise ValueError("min_ore_fraction must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "local_erosion_ratio_score",
            "version": 1,
            "erosion_kernel_size": self.erosion_kernel_size,
            "erosion_iterations": self.erosion_iterations,
            "window_size": self.window_size,
            "normal_threshold": self.normal_threshold,
            "min_ore_fraction": self.min_ore_fraction,
            "class_ids": {
                "background": self.background_id,
                "talc": self.talc_id,
                "normal_ore": self.normal_ore_id,
                "hard_ore": self.hard_ore_id,
                "ignore": self.ignore_id,
            },
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ErosionRatioConfig":
        class_ids = values.get("class_ids", {}) if isinstance(values.get("class_ids"), dict) else {}
        return cls(
            erosion_kernel_size=int(values.get("erosion_kernel_size", 5)),
            erosion_iterations=int(values.get("erosion_iterations", 1)),
            window_size=int(values.get("window_size", 128)),
            normal_threshold=float(values.get("normal_threshold", 0.5)),
            min_ore_fraction=float(values.get("min_ore_fraction", 0.05)),
            background_id=int(class_ids.get("background", values.get("background_id", BACKGROUND_ID))),
            talc_id=int(class_ids.get("talc", values.get("talc_id", TALC_ID))),
            normal_ore_id=int(class_ids.get("normal_ore", values.get("normal_ore_id", NORMAL_ORE_ID))),
            hard_ore_id=int(class_ids.get("hard_ore", values.get("hard_ore_id", HARD_ORE_ID))),
            ignore_id=int(class_ids.get("ignore", values.get("ignore_id", IGNORE_ID))),
        )


def load_erosion_ratio_config(path: str | Path | None = None) -> ErosionRatioConfig:
    """Load erosion-ratio calibration config, or return defaults when missing."""
    if path is None:
        config = ErosionRatioConfig()
        config.validate()
        return config
    config_path = Path(path)
    if not config_path.exists():
        config = ErosionRatioConfig()
        config.validate()
        return config
    config = ErosionRatioConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    config.validate()
    return config


@dataclass(frozen=True)
class ErosionRatioResult:
    """Erosion-ratio score maps and class mask."""

    intergrowth_mask: Image.Image
    normal_score: Image.Image
    hard_score: Image.Image
    ratio_score: Image.Image
    metrics: dict[str, Any]
    window_summaries: list[dict[str, Any]]


def erode_binary_mask(mask: Image.Image, *, kernel_size: int = 5, iterations: int = 1) -> Image.Image:
    """Return a binary mask eroded by a square min filter."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    eroded = mask.convert("L").point(lambda value: 255 if value > 0 else 0)
    radius = kernel_size // 2
    for _ in range(iterations):
        padded = ImageOps.expand(eroded, border=radius, fill=0)
        eroded = padded.filter(ImageFilter.MinFilter(kernel_size)).crop(
            (radius, radius, radius + eroded.width, radius + eroded.height)
        )
    return eroded


def _axis_windows(length: int, window_size: int) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    window = min(length, window_size)
    if length <= window:
        return [(0, length)]
    last = length - window
    starts = list(range(0, last + 1, window))
    if starts[-1] != last:
        starts.append(last)
    return [(start, min(length, start + window)) for start in starts]


def _window_erosion_ratio(
    mask: Image.Image,
    *,
    config: ErosionRatioConfig,
    class_mask: Image.Image | None = None,
    background_index: int = 0,
) -> tuple[float, float, int, int, int]:
    if class_mask is not None:
        class_l = class_mask.convert("L")
        class_hist = class_l.histogram()
        pixel_count = sum(class_hist)
        class_ids = [class_id for class_id, count in enumerate(class_hist) if count and class_id != int(background_index)]
        ore_area = sum(count for class_id, count in enumerate(class_hist) if class_id != int(background_index))
    else:
        mask_l = mask.convert("L")
        mask_hist = mask_l.histogram()
        pixel_count = sum(mask_hist)
        class_ids = []
        ore_area = sum(mask_hist[1:])
    ore_fraction = ore_area / pixel_count if pixel_count else 0.0
    if pixel_count == 0 or ore_area == 0 or ore_fraction < config.min_ore_fraction:
        return 0.0, ore_fraction, ore_area, 0, 0

    if class_mask is None:
        eroded = erode_binary_mask(mask, kernel_size=config.erosion_kernel_size, iterations=config.erosion_iterations)
        eroded_area = sum(eroded.histogram()[1:])
        return eroded_area / ore_area, ore_fraction, ore_area, eroded_area, 1

    if not class_ids:
        return 0.0, ore_fraction, ore_area, 0, 0

    eroded_area = 0
    for class_id in class_ids:
        class_binary = class_l.point([255 if value == class_id else 0 for value in range(256)])
        eroded = erode_binary_mask(class_binary, kernel_size=config.erosion_kernel_size, iterations=config.erosion_iterations)
        eroded_area += sum(eroded.histogram()[1:])
    return eroded_area / ore_area, ore_fraction, ore_area, eroded_area, len(class_ids)


def erosion_ratio_score_grid(
    ore_mask: Image.Image,
    *,
    multiclass_mask: Image.Image | None = None,
    background_index: int = 0,
    config: ErosionRatioConfig | None = None,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Return a coarse 0..255 normal-ore score grid from local erosion ratios."""
    cfg = config or ErosionRatioConfig()
    cfg.validate()
    ore = ore_mask.convert("L").point(lambda value: 255 if value > 0 else 0)
    classes = multiclass_mask.convert("L") if multiclass_mask is not None else None
    if classes is not None and classes.size != ore.size:
        raise ValueError("multiclass_mask size must match ore_mask size")
    x_windows = _axis_windows(ore.width, cfg.window_size)
    y_windows = _axis_windows(ore.height, cfg.window_size)
    if not x_windows or not y_windows:
        return Image.new("L", (1, 1), 0), []

    score_values: list[int] = []
    summaries: list[dict[str, Any]] = []
    for top, bottom in y_windows:
        for left, right in x_windows:
            box = (left, top, right, bottom)
            ratio, ore_fraction, ore_area, eroded_area, class_count = _window_erosion_ratio(
                ore.crop(box),
                config=cfg,
                class_mask=classes.crop(box) if classes is not None else None,
                background_index=background_index,
            )
            score_values.append(int(round(max(0.0, min(1.0, ratio)) * 255)))
            summaries.append(
                {
                    "box": box,
                    "ratio": ratio,
                    "normal_score": ratio,
                    "hard_score": 1.0 - ratio,
                    "ore_area_fraction": ore_fraction,
                    "ore_area": ore_area,
                    "eroded_ore_area": eroded_area,
                    "class_count": class_count,
                    "class_aware_erosion": classes is not None,
                }
            )

    grid = Image.new("L", (len(x_windows), len(y_windows)))
    grid.putdata(score_values)
    return grid, summaries


def erosion_ratio_score_map(
    ore_mask: Image.Image,
    *,
    multiclass_mask: Image.Image | None = None,
    background_index: int = 0,
    config: ErosionRatioConfig | None = None,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Return a 0..255 normal-ore score image from local erosion ratios.

    The local window value is ``eroded_ore_area / ore_area``. When a multiclass
    mineral mask is provided, each non-background class is eroded separately and
    eroded areas are summed. Windows with ore area below ``min_ore_fraction``
    receive score 0. The coarse window grid is resized with bilinear
    interpolation to create a full-resolution score map.
    """
    cfg = config or ErosionRatioConfig()
    cfg.validate()
    ore = ore_mask.convert("L").point(lambda value: 255 if value > 0 else 0)
    grid, summaries = erosion_ratio_score_grid(
        ore,
        multiclass_mask=multiclass_mask,
        background_index=background_index,
        config=cfg,
    )
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    score = grid.resize(ore.size, resampling).convert("L")
    return score, summaries


def invert_score_on_ore(score: Image.Image, ore_mask: Image.Image) -> Image.Image:
    """Return ``255 - score`` for ore pixels and 0 for background pixels."""
    score_values = np.asarray(score.convert("L"), dtype=np.uint8)
    ore_values = np.asarray(ore_mask.convert("L"), dtype=np.uint8)
    return Image.fromarray(np.where(ore_values > 0, 255 - score_values, 0).astype(np.uint8))


def classify_erosion_ratio_intergrowth(
    ore_mask: Image.Image,
    *,
    multiclass_mask: Image.Image | None = None,
    background_index: int = 0,
    config: ErosionRatioConfig | None = None,
    talc_mask: Image.Image | None = None,
    ignore_mask: Image.Image | None = None,
) -> ErosionRatioResult:
    """Classify ore pixels with the interpolated erosion-ratio metric."""
    cfg = config or ErosionRatioConfig()
    cfg.validate()
    ore = ore_mask.convert("L")
    ratio_score, summaries = erosion_ratio_score_map(
        ore,
        multiclass_mask=multiclass_mask,
        background_index=background_index,
        config=cfg,
    )
    normal_threshold_u8 = int(round(cfg.normal_threshold * 255))
    ore_values = np.asarray(ore, dtype=np.uint8)
    score_values = np.asarray(ratio_score.convert("L"), dtype=np.uint8)
    talc_values = (
        np.asarray(talc_mask.convert("L"), dtype=np.uint8)
        if talc_mask is not None
        else np.zeros(ore_values.shape, dtype=np.uint8)
    )
    ignore_values = (
        np.asarray(ignore_mask.convert("L"), dtype=np.uint8)
        if ignore_mask is not None
        else np.zeros(ore_values.shape, dtype=np.uint8)
    )
    ore_pixels = ore_values > 0
    ignore_pixels = ignore_values > 0
    talc_pixels = (talc_values > 0) & ~ignore_pixels
    normal_pixels = ore_pixels & ~ignore_pixels & ~talc_pixels & (score_values > normal_threshold_u8)
    hard_pixels = ore_pixels & ~ignore_pixels & ~talc_pixels & ~normal_pixels

    class_values = np.full(ore_values.shape, cfg.background_id, dtype=np.uint8)
    class_values[normal_pixels] = cfg.normal_ore_id
    class_values[hard_pixels] = cfg.hard_ore_id
    class_values[talc_pixels] = cfg.talc_id
    class_values[ignore_pixels] = cfg.ignore_id
    normal_values = np.where(ore_pixels & ~ignore_pixels & ~talc_pixels, score_values, 0).astype(np.uint8)
    hard_values = np.where(ore_pixels & ~ignore_pixels & ~talc_pixels, 255 - score_values, 0).astype(np.uint8)

    intergrowth = Image.fromarray(class_values)
    normal_score = Image.fromarray(normal_values)
    hard_score = Image.fromarray(hard_values)
    return ErosionRatioResult(
        intergrowth_mask=intergrowth,
        normal_score=normal_score,
        hard_score=hard_score,
        ratio_score=ratio_score,
        metrics=intergrowth_area_metrics(intergrowth),
        window_summaries=summaries,
    )


def mean_ore_ratio_score(score: Image.Image, ore_mask: Image.Image) -> float:
    """Return mean 0..1 erosion-ratio score over ore pixels."""
    score_values = np.asarray(score.convert("L"), dtype=np.uint8)
    ore_values = np.asarray(ore_mask.convert("L"), dtype=np.uint8)
    ore_pixels = ore_values > 0
    if not bool(np.any(ore_pixels)):
        return 0.0
    return float(score_values[ore_pixels].mean() / 255)


def choose_erosion_ratio_threshold(labeled_scores: Iterable[tuple[float, str]]) -> dict[str, Any]:
    """Choose a normal/hard threshold maximizing balanced accuracy."""
    rows = [(float(score), str(label).lower()) for score, label in labeled_scores]
    rows = [(score, "hard" if "hard" in label else "normal") for score, label in rows if "hard" in label or "normal" in label]
    if not rows:
        return {"threshold": 0.5, "balanced_accuracy": 0.0, "sample_count": 0}
    candidates = sorted({score for score, _ in rows})
    thresholds = [0.0, 1.0]
    thresholds.extend(candidates)
    thresholds.extend((left + right) / 2.0 for left, right in zip(candidates, candidates[1:]))
    best = {"threshold": 0.5, "balanced_accuracy": -1.0, "sample_count": len(rows)}
    for threshold in thresholds:
        hard_total = sum(1 for _, label in rows if label == "hard")
        normal_total = sum(1 for _, label in rows if label == "normal")
        hard_correct = sum(1 for score, label in rows if label == "hard" and score < threshold)
        normal_correct = sum(1 for score, label in rows if label == "normal" and score > threshold)
        hard_recall = hard_correct / hard_total if hard_total else 0.0
        normal_recall = normal_correct / normal_total if normal_total else 0.0
        balanced = (hard_recall + normal_recall) / 2.0
        if balanced > best["balanced_accuracy"]:
            best = {"threshold": threshold, "balanced_accuracy": balanced, "sample_count": len(rows)}
    return best


def save_erosion_ratio_config(config: ErosionRatioConfig, path: str | Path) -> Path:
    """Persist the erosion-ratio calibration config."""
    config.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return output
