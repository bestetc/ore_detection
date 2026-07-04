"""Erosion-ratio hard/normal ore metric."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
    class_values = class_mask.convert("L").tobytes() if class_mask is not None else None
    values = class_values if class_values is not None else mask.convert("L").tobytes()
    pixel_count = len(values)
    if class_values is not None:
        class_ids = sorted({int(value) for value in class_values if int(value) != int(background_index)})
        ore_area = sum(1 for value in class_values if int(value) != int(background_index))
    else:
        class_ids = []
        ore_area = sum(1 for value in values if value > 0)
    ore_fraction = ore_area / pixel_count if pixel_count else 0.0
    if pixel_count == 0 or ore_area == 0 or ore_fraction < config.min_ore_fraction:
        return 0.0, ore_fraction, ore_area, 0, 0

    if class_mask is None:
        eroded = erode_binary_mask(mask, kernel_size=config.erosion_kernel_size, iterations=config.erosion_iterations)
        eroded_area = sum(1 for value in eroded.tobytes() if value > 0)
        return eroded_area / ore_area, ore_fraction, ore_area, eroded_area, 1

    if not class_ids:
        return 0.0, ore_fraction, ore_area, 0, 0

    eroded_area = 0
    for class_id in class_ids:
        class_binary = Image.new("L", class_mask.size)
        class_binary.putdata([255 if int(value) == class_id else 0 for value in class_values])
        eroded = erode_binary_mask(class_binary, kernel_size=config.erosion_kernel_size, iterations=config.erosion_iterations)
        eroded_area += sum(1 for value in eroded.tobytes() if value > 0)
    return eroded_area / ore_area, ore_fraction, ore_area, eroded_area, len(class_ids)


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
    classes = multiclass_mask.convert("L") if multiclass_mask is not None else None
    if classes is not None and classes.size != ore.size:
        raise ValueError("multiclass_mask size must match ore_mask size")
    x_windows = _axis_windows(ore.width, cfg.window_size)
    y_windows = _axis_windows(ore.height, cfg.window_size)
    if not x_windows or not y_windows:
        return Image.new("L", ore.size, 0), []

    ratios: list[float] = []
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
            ratios.append(ratio)
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

    coarse = Image.new("F", (len(x_windows), len(y_windows)))
    coarse.putdata(ratios)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    interpolated = coarse.resize(ore.size, resampling)
    score = interpolated.point(lambda value: value * 255 + 0.5).convert("L")
    return score, summaries


def invert_score_on_ore(score: Image.Image, ore_mask: Image.Image) -> Image.Image:
    """Return ``255 - score`` for ore pixels and 0 for background pixels."""
    score_values = score.convert("L").tobytes()
    ore_values = ore_mask.convert("L").tobytes()
    out = Image.new("L", score.size)
    out.putdata([255 - int(value) if ore > 0 else 0 for value, ore in zip(score_values, ore_values)])
    return out


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
    talc_values = talc_mask.convert("L").tobytes() if talc_mask is not None else None
    ignore_values = ignore_mask.convert("L").tobytes() if ignore_mask is not None else None
    ore_values = ore.tobytes()
    score_values = ratio_score.tobytes()
    class_values: list[int] = []
    normal_values: list[int] = []
    hard_values: list[int] = []
    for index, (ore_value, score_value) in enumerate(zip(ore_values, score_values)):
        talc_value = talc_values[index] if talc_values is not None else 0
        ignore_value = ignore_values[index] if ignore_values is not None else 0
        if ignore_value > 0:
            class_values.append(cfg.ignore_id)
            normal_values.append(0)
            hard_values.append(0)
        elif talc_value > 0:
            class_values.append(cfg.talc_id)
            normal_values.append(0)
            hard_values.append(0)
        elif ore_value > 0:
            class_values.append(cfg.normal_ore_id if score_value > normal_threshold_u8 else cfg.hard_ore_id)
            normal_values.append(int(score_value))
            hard_values.append(255 - int(score_value))
        else:
            class_values.append(cfg.background_id)
            normal_values.append(0)
            hard_values.append(0)

    intergrowth = Image.new("L", ore.size)
    intergrowth.putdata(class_values)
    normal_score = Image.new("L", ore.size)
    normal_score.putdata(normal_values)
    hard_score = Image.new("L", ore.size)
    hard_score.putdata(hard_values)
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
    score_values = score.convert("L").tobytes()
    ore_values = ore_mask.convert("L").tobytes()
    values = [int(score) / 255 for score, ore in zip(score_values, ore_values) if ore > 0]
    return sum(values) / len(values) if values else 0.0


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
    import json

    config.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return output
