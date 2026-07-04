"""Hard/normal intergrowth classification from metallic ore mask morphology."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image

from ore_detection.descriptors.contacts import contact_lengths, hetero_sulfide_contact_length
from ore_detection.descriptors.morphology import component_stats
from ore_detection.inference.tiled_prediction import allow_large_pillow_images

BACKGROUND_ID = 0
TALC_ID = 3
NORMAL_ORE_ID = 4
HARD_ORE_ID = 5
IGNORE_ID = 255

DEFAULT_FEATURE_WEIGHTS: dict[str, float] = {
    "small_object_fraction": 0.30,
    "fragmentation_score": 0.25,
    "dominant_component_inverse": 0.25,
    "boundary_contact_score": 0.20,
    "mineral_contact_score": 0.10,
}


@dataclass(frozen=True)
class LocalWindow:
    """One morphology scoring window and the stable sub-region it owns."""

    box: tuple[int, int, int, int]
    stable_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class IntergrowthClassifierConfig:
    """Config for dependency-light morphology scoring."""

    window_size: int = 128
    stride: int = 64
    small_area_threshold: int = 25
    hard_threshold: float = 0.5
    min_ore_fraction: float = 0.001
    background_id: int = BACKGROUND_ID
    talc_id: int = TALC_ID
    normal_ore_id: int = NORMAL_ORE_ID
    hard_ore_id: int = HARD_ORE_ID
    ignore_id: int = IGNORE_ID
    feature_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FEATURE_WEIGHTS))

    def validate(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.small_area_threshold <= 0:
            raise ValueError("small_area_threshold must be positive")
        if not 0.0 <= self.hard_threshold <= 1.0:
            raise ValueError("hard_threshold must be in [0, 1]")
        if self.min_ore_fraction < 0:
            raise ValueError("min_ore_fraction must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "local_morphology_linear_score",
            "version": 1,
            "window_size": self.window_size,
            "stride": self.stride,
            "small_area_threshold": self.small_area_threshold,
            "hard_threshold": self.hard_threshold,
            "min_ore_fraction": self.min_ore_fraction,
            "class_ids": {
                "background": self.background_id,
                "talc": self.talc_id,
                "normal_ore": self.normal_ore_id,
                "hard_ore": self.hard_ore_id,
                "ignore": self.ignore_id,
            },
            "feature_weights": dict(self.feature_weights),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "IntergrowthClassifierConfig":
        class_ids = values.get("class_ids", {}) if isinstance(values.get("class_ids"), dict) else {}
        return cls(
            window_size=int(values.get("window_size", 128)),
            stride=int(values.get("stride", 64)),
            small_area_threshold=int(values.get("small_area_threshold", 25)),
            hard_threshold=float(values.get("hard_threshold", 0.5)),
            min_ore_fraction=float(values.get("min_ore_fraction", 0.001)),
            background_id=int(class_ids.get("background", values.get("background_id", BACKGROUND_ID))),
            talc_id=int(class_ids.get("talc", values.get("talc_id", TALC_ID))),
            normal_ore_id=int(class_ids.get("normal_ore", values.get("normal_ore_id", NORMAL_ORE_ID))),
            hard_ore_id=int(class_ids.get("hard_ore", values.get("hard_ore_id", HARD_ORE_ID))),
            ignore_id=int(class_ids.get("ignore", values.get("ignore_id", IGNORE_ID))),
            feature_weights={str(k): float(v) for k, v in dict(values.get("feature_weights", DEFAULT_FEATURE_WEIGHTS)).items()},
        )


@dataclass(frozen=True)
class IntergrowthClassificationResult:
    """In-memory intergrowth masks and summary metadata."""

    intergrowth_mask: Image.Image
    score: Image.Image
    confidence: Image.Image
    metrics: dict[str, Any]
    window_summaries: list[dict[str, Any]]


def load_intergrowth_classifier_config(path: str | Path | None = None) -> IntergrowthClassifierConfig:
    """Load classifier config if present, otherwise return the default config."""
    if path is None:
        config = IntergrowthClassifierConfig()
        config.validate()
        return config
    config_path = Path(path)
    if not config_path.exists():
        config = IntergrowthClassifierConfig()
        config.validate()
        return config
    config = IntergrowthClassifierConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    config.validate()
    return config


def save_intergrowth_classifier_config(config: IntergrowthClassifierConfig, path: str | Path) -> Path:
    """Persist a calibrated morphology classifier config."""
    config.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return output


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _rows_from_l_image(mask: Image.Image, *, binary: bool) -> list[list[int]]:
    image = mask.convert("L")
    raw = list(image.tobytes())
    rows: list[list[int]] = []
    for row in range(image.height):
        start = row * image.width
        values = raw[start : start + image.width]
        if binary:
            rows.append([1 if value > 0 else 0 for value in values])
        else:
            rows.append([int(value) for value in values])
    return rows


def _positions(length: int, window_size: int, stride: int) -> list[int]:
    if length <= 0:
        return []
    window = min(length, window_size)
    if length <= window:
        return [0]
    last = length - window
    effective_stride = max(1, min(stride, window))
    positions = list(range(0, last + 1, effective_stride))
    if positions[-1] != last:
        positions.append(last)
    return positions


def local_windows(width: int, height: int, *, window_size: int = 128, stride: int = 64) -> list[LocalWindow]:
    """Return local morphology windows with stable sub-regions covering the image."""
    if width <= 0 or height <= 0:
        return []
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    window_w = min(width, window_size)
    window_h = min(height, window_size)
    effective_stride_x = min(stride, window_w)
    effective_stride_y = min(stride, window_h)
    margin_x = max(0, (window_w - effective_stride_x) // 2)
    margin_y = max(0, (window_h - effective_stride_y) // 2)
    windows: list[LocalWindow] = []
    for top in _positions(height, window_h, stride):
        for left in _positions(width, window_w, stride):
            right = min(width, left + window_w)
            bottom = min(height, top + window_h)
            stable_left = left if left == 0 else min(right - 1, left + margin_x)
            stable_top = top if top == 0 else min(bottom - 1, top + margin_y)
            stable_right = right if right == width else max(stable_left + 1, right - margin_x)
            stable_bottom = bottom if bottom == height else max(stable_top + 1, bottom - margin_y)
            windows.append(LocalWindow((left, top, right, bottom), (stable_left, stable_top, stable_right, stable_bottom)))
    return windows


def _quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = fraction * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def extract_intergrowth_features(
    binary_mask: Image.Image | Sequence[Sequence[Any]],
    *,
    multiclass_mask: Image.Image | Sequence[Sequence[Any]] | None = None,
    background_index: int = 0,
    small_area_threshold: int = 25,
) -> dict[str, float]:
    """Compute morphology features for one local window."""
    if isinstance(binary_mask, Image.Image):
        binary_rows = _rows_from_l_image(binary_mask, binary=True)
        width, height = binary_mask.size
    else:
        binary_rows = [[1 if int(value) > 0 else 0 for value in row] for row in binary_mask]
        height = len(binary_rows)
        width = len(binary_rows[0]) if height else 0

    pixel_count = width * height
    ore_area = float(sum(sum(row) for row in binary_rows))
    if pixel_count == 0 or ore_area == 0:
        return {
            "pixel_count": float(pixel_count),
            "ore_area": 0.0,
            "ore_area_fraction": 0.0,
            "component_count": 0.0,
            "component_count_density": 0.0,
            "component_density_per_10k": 0.0,
            "component_area_p50": 0.0,
            "component_area_p90": 0.0,
            "component_area_max": 0.0,
            "dominant_component_fraction": 0.0,
            "dominant_component_inverse": 0.0,
            "small_object_fraction": 0.0,
            "perimeter": 0.0,
            "perimeter_over_sqrt_area_weighted_mean": 0.0,
            "perimeter2_over_area_weighted_mean": 0.0,
            "compactness": 0.0,
            "solidity_proxy": 0.0,
            "ore_background_contact_length": 0.0,
            "ore_background_contact_density": 0.0,
            "ore_background_contact_per_ore_area": 0.0,
            "mineral_contact_length": 0.0,
            "mineral_contact_density": 0.0,
        }

    stats = component_stats(binary_rows, foreground_values={1})
    areas = sorted(float(item["area"]) for item in stats)
    perimeter = float(sum(item["perimeter"] for item in stats))
    small_area = float(sum(item["area"] for item in stats if item["area"] <= small_area_threshold))

    def weighted_mean(key: str) -> float:
        return sum(float(item[key]) * float(item["area"]) for item in stats) / ore_area

    perimeter_over_sqrt_area = (
        sum((float(item["perimeter"]) / sqrt(max(1.0, float(item["area"])))) * float(item["area"]) for item in stats)
        / ore_area
    )
    ore_background_contact = float(contact_lengths(binary_rows, classes={0, 1}).get((0, 1), 0))

    mineral_contact = 0.0
    if multiclass_mask is not None:
        if isinstance(multiclass_mask, Image.Image):
            class_rows = _rows_from_l_image(multiclass_mask, binary=False)
        else:
            class_rows = [[int(value) for value in row] for row in multiclass_mask]
        mineral_classes = {value for row in class_rows for value in row if int(value) != int(background_index)}
        if mineral_classes:
            mineral_contact = float(hetero_sulfide_contact_length(class_rows, sulfide_classes=mineral_classes))

    return {
        "pixel_count": float(pixel_count),
        "ore_area": ore_area,
        "ore_area_fraction": ore_area / pixel_count if pixel_count else 0.0,
        "component_count": float(len(stats)),
        "component_count_density": float(len(stats) / ore_area),
        "component_density_per_10k": float(len(stats) * 10000 / pixel_count) if pixel_count else 0.0,
        "component_area_p50": _quantile(areas, 0.50),
        "component_area_p90": _quantile(areas, 0.90),
        "component_area_max": areas[-1] if areas else 0.0,
        "dominant_component_fraction": areas[-1] / ore_area if areas else 0.0,
        "dominant_component_inverse": 1.0 - (areas[-1] / ore_area if areas else 0.0),
        "small_object_fraction": small_area / ore_area,
        "perimeter": perimeter,
        "perimeter_over_sqrt_area_weighted_mean": float(perimeter_over_sqrt_area),
        "perimeter2_over_area_weighted_mean": float(weighted_mean("perimeter2_over_area")),
        "compactness": float(weighted_mean("circularity")),
        "solidity_proxy": float(weighted_mean("bbox_fill")),
        "ore_background_contact_length": ore_background_contact,
        "ore_background_contact_density": ore_background_contact / pixel_count if pixel_count else 0.0,
        "ore_background_contact_per_ore_area": ore_background_contact / ore_area if ore_area else 0.0,
        "mineral_contact_length": mineral_contact,
        "mineral_contact_density": mineral_contact / pixel_count if pixel_count else 0.0,
    }


def hard_score_from_features(
    features: dict[str, float],
    *,
    feature_weights: dict[str, float] | None = None,
) -> float:
    """Convert morphology features into a 0..1 hard-ore score."""
    weights = dict(feature_weights or DEFAULT_FEATURE_WEIGHTS)
    feature_scores = {
        "small_object_fraction": _clamp01(features.get("small_object_fraction", 0.0)),
        "fragmentation_score": _clamp01(features.get("component_count_density", 0.0) * 25.0),
        "dominant_component_inverse": _clamp01(features.get("dominant_component_inverse", 0.0)),
        "boundary_contact_score": _clamp01((features.get("ore_background_contact_per_ore_area", 0.0) - 0.25) / 1.25),
        "mineral_contact_score": _clamp01(features.get("mineral_contact_length", 0.0) / max(1.0, features.get("ore_area", 0.0))),
    }
    total_weight = sum(max(0.0, weight) for weight in weights.values())
    if total_weight <= 0:
        return 0.0
    return _clamp01(sum(feature_scores.get(name, 0.0) * max(0.0, weight) for name, weight in weights.items()) / total_weight)


def confidence_from_score(score: float, *, threshold: float) -> float:
    """Return confidence as normalized distance from the hard/normal threshold."""
    denominator = max(threshold, 1.0 - threshold, 1e-6)
    return _clamp01(abs(float(score) - float(threshold)) / denominator)


def _blank_like(mask: Image.Image, value: int = 0) -> Image.Image:
    return Image.new("L", mask.size, int(value))


def _ensure_same_size(name: str, image: Image.Image, expected: tuple[int, int]) -> None:
    if image.size != expected:
        raise ValueError(f"{name} size {image.size} does not match expected size {expected}")


def _classify_stable_region(
    *,
    ore_tile: Image.Image,
    score: float,
    confidence: float,
    config: IntergrowthClassifierConfig,
    talc_tile: Image.Image | None = None,
    ignore_tile: Image.Image | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    ore_values = list(ore_tile.convert("L").tobytes())
    count = len(ore_values)
    talc_values = list(talc_tile.convert("L").tobytes()) if talc_tile is not None else [0] * count
    ignore_values = list(ignore_tile.convert("L").tobytes()) if ignore_tile is not None else [0] * count
    class_id = config.hard_ore_id if score >= config.hard_threshold else config.normal_ore_id
    score_value = int(round(_clamp01(score) * 255))
    confidence_value = int(round(_clamp01(confidence) * 255))
    class_values: list[int] = []
    score_values: list[int] = []
    confidence_values: list[int] = []
    for ore, talc, ignore in zip(ore_values, talc_values, ignore_values):
        if ignore > 0:
            class_values.append(config.ignore_id)
            score_values.append(0)
            confidence_values.append(0)
        elif talc > 0:
            class_values.append(config.talc_id)
            score_values.append(0)
            confidence_values.append(0)
        elif ore > 0:
            class_values.append(class_id)
            score_values.append(score_value)
            confidence_values.append(confidence_value)
        else:
            class_values.append(config.background_id)
            score_values.append(0)
            confidence_values.append(0)

    class_tile = Image.new("L", ore_tile.size)
    score_tile = Image.new("L", ore_tile.size)
    confidence_tile = Image.new("L", ore_tile.size)
    class_tile.putdata(class_values)
    score_tile.putdata(score_values)
    confidence_tile.putdata(confidence_values)
    return class_tile, score_tile, confidence_tile


def intergrowth_area_metrics(mask: Image.Image, *, config: IntergrowthClassifierConfig | None = None) -> dict[str, Any]:
    """Return area metrics for a hard/normal/talc/background intergrowth mask."""
    cfg = config or IntergrowthClassifierConfig()
    values = list(mask.convert("L").tobytes())
    total = max(1, len(values))
    counts = {
        "background": sum(1 for value in values if value == cfg.background_id),
        "talc": sum(1 for value in values if value == cfg.talc_id),
        "normal_ore": sum(1 for value in values if value == cfg.normal_ore_id),
        "hard_ore": sum(1 for value in values if value == cfg.hard_ore_id),
        "ignore": sum(1 for value in values if value == cfg.ignore_id),
    }
    metallic_area = counts["normal_ore"] + counts["hard_ore"]
    hard_fraction = counts["hard_ore"] / metallic_area if metallic_area else 0.0
    normal_fraction = counts["normal_ore"] / metallic_area if metallic_area else 0.0
    talc_fraction = counts["talc"] / total
    if talc_fraction > 0.10:
        image_label = "talc"
    elif metallic_area == 0:
        image_label = "background"
    elif hard_fraction >= 0.50:
        image_label = "hard_ore"
    else:
        image_label = "normal_ore"
    return {
        "total_pixels": total,
        "counts": counts,
        "fractions": {name: count / total for name, count in counts.items()},
        "metallic_ore_pixels": metallic_area,
        "hard_fraction_of_metallic_ore": hard_fraction,
        "normal_fraction_of_metallic_ore": normal_fraction,
        "image_label": image_label,
    }


def classify_intergrowth_mask(
    ore_mask: Image.Image,
    *,
    multiclass_mask: Image.Image | None = None,
    talc_mask: Image.Image | None = None,
    ignore_mask: Image.Image | None = None,
    background_index: int = 0,
    config: IntergrowthClassifierConfig | None = None,
) -> IntergrowthClassificationResult:
    """Classify ore pixels as hard or normal from local morphology."""
    cfg = config or IntergrowthClassifierConfig()
    cfg.validate()
    ore = ore_mask.convert("L")
    size = ore.size
    if multiclass_mask is not None:
        _ensure_same_size("multiclass_mask", multiclass_mask, size)
        multiclass_mask = multiclass_mask.convert("L")
    if talc_mask is not None:
        _ensure_same_size("talc_mask", talc_mask, size)
        talc_mask = talc_mask.convert("L")
    if ignore_mask is not None:
        _ensure_same_size("ignore_mask", ignore_mask, size)
        ignore_mask = ignore_mask.convert("L")

    intergrowth = Image.new("L", size, cfg.background_id)
    score_image = Image.new("L", size, 0)
    confidence_image = Image.new("L", size, 0)
    summaries: list[dict[str, Any]] = []

    for window in local_windows(ore.width, ore.height, window_size=cfg.window_size, stride=cfg.stride):
        crop = ore.crop(window.box)
        multiclass_crop = multiclass_mask.crop(window.box) if multiclass_mask is not None else None
        features = extract_intergrowth_features(
            crop,
            multiclass_mask=multiclass_crop,
            background_index=background_index,
            small_area_threshold=cfg.small_area_threshold,
        )
        score = hard_score_from_features(features, feature_weights=cfg.feature_weights)
        if features["ore_area_fraction"] < cfg.min_ore_fraction:
            score = 0.0
        confidence = confidence_from_score(score, threshold=cfg.hard_threshold)
        stable = window.stable_box
        class_tile, score_tile, confidence_tile = _classify_stable_region(
            ore_tile=ore.crop(stable),
            score=score,
            confidence=confidence,
            config=cfg,
            talc_tile=talc_mask.crop(stable) if talc_mask is not None else None,
            ignore_tile=ignore_mask.crop(stable) if ignore_mask is not None else None,
        )
        intergrowth.paste(class_tile, stable)
        score_image.paste(score_tile, stable)
        confidence_image.paste(confidence_tile, stable)
        summaries.append(
            {
                "box": window.box,
                "stable_box": stable,
                "score": score,
                "confidence": confidence,
                "hard": score >= cfg.hard_threshold,
                "ore_area_fraction": features["ore_area_fraction"],
                "component_count": features["component_count"],
                "small_object_fraction": features["small_object_fraction"],
                "dominant_component_fraction": features["dominant_component_fraction"],
                "ore_background_contact_per_ore_area": features["ore_background_contact_per_ore_area"],
            }
        )

    return IntergrowthClassificationResult(
        intergrowth_mask=intergrowth,
        score=score_image,
        confidence=confidence_image,
        metrics=intergrowth_area_metrics(intergrowth, config=cfg),
        window_summaries=summaries,
    )


def _binary_from_multiclass(multiclass: Image.Image, *, background_index: int) -> Image.Image:
    values = [1 if int(value) != int(background_index) else 0 for value in multiclass.convert("L").tobytes()]
    binary = Image.new("L", multiclass.size)
    binary.putdata(values)
    return binary


def _metadata_artifact_path(metadata: dict[str, Any], name: str) -> Path | None:
    artifacts = metadata.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return None
    value = artifacts.get(name)
    return Path(str(value)) if value else None


def save_intergrowth_artifacts(
    sample_dir: str | Path,
    *,
    config: IntergrowthClassifierConfig | None = None,
    classifier_config_path: str | Path | None = None,
    talc_mask_path: str | Path | None = None,
    ignore_mask_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create and save intergrowth hard/normal artifacts for one prediction directory."""
    sample_dir = Path(sample_dir)
    metadata_path = sample_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"prediction metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cfg = config or load_intergrowth_classifier_config(classifier_config_path)
    cfg.validate()

    ore_checkpoint = metadata.get("ore_checkpoint") if isinstance(metadata.get("ore_checkpoint"), dict) else {}
    background_index = int(ore_checkpoint.get("background_index", 0)) if isinstance(ore_checkpoint, dict) else 0
    ore_mask_path = _metadata_artifact_path(metadata, "ore_mask")
    multiclass_mask_path = _metadata_artifact_path(metadata, "ore_multiclass_mask")
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
        talc_image = Image.open(talc_mask_path).convert("L") if talc_mask_path else None
        ignore_image = Image.open(ignore_mask_path).convert("L") if ignore_mask_path else None
        result = classify_intergrowth_mask(
            ore_image,
            multiclass_mask=multiclass_image,
            talc_mask=talc_image,
            ignore_mask=ignore_image,
            background_index=background_index,
            config=cfg,
        )
        if multiclass_image is not None:
            multiclass_image.close()
        ore_image.close()
        if talc_image is not None:
            talc_image.close()
        if ignore_image is not None:
            ignore_image.close()

    intergrowth_mask_path = sample_dir / "intergrowth_mask.png"
    intergrowth_score_path = sample_dir / "intergrowth_score.png"
    intergrowth_confidence_path = sample_dir / "intergrowth_confidence.png"
    intergrowth_metrics_path = sample_dir / "intergrowth_metrics.json"
    result.intergrowth_mask.save(intergrowth_mask_path, compress_level=1)
    result.score.save(intergrowth_score_path, compress_level=1)
    result.confidence.save(intergrowth_confidence_path, compress_level=1)

    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_prediction_metadata": str(metadata_path),
        "source_ore_mask": ore_source,
        "classifier_config": cfg.to_dict(),
        "window_count": len(result.window_summaries),
        "window_scores": {
            "min": min((row["score"] for row in result.window_summaries), default=0.0),
            "max": max((row["score"] for row in result.window_summaries), default=0.0),
            "mean": (
                sum(float(row["score"]) for row in result.window_summaries) / len(result.window_summaries)
                if result.window_summaries
                else 0.0
            ),
        },
        "area_metrics": result.metrics,
        "artifacts": {
            "intergrowth_mask": str(intergrowth_mask_path),
            "intergrowth_score": str(intergrowth_score_path),
            "intergrowth_confidence": str(intergrowth_confidence_path),
            "intergrowth_metrics": str(intergrowth_metrics_path),
        },
    }
    intergrowth_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    artifacts = metadata.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
        metadata["artifacts"] = artifacts
    artifacts.update(metrics["artifacts"])
    metadata["intergrowth"] = {
        "source_ore_mask": ore_source,
        "classifier_config": cfg.to_dict(),
        "metrics_path": str(intergrowth_metrics_path),
        "area_metrics": result.metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics["metadata_path"] = str(metadata_path)
    return metrics


def choose_hard_threshold(labeled_scores: Iterable[tuple[float, str]]) -> dict[str, Any]:
    """Choose a threshold maximizing balanced accuracy for weak hard/normal labels."""
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
        hard_correct = sum(1 for score, label in rows if label == "hard" and score >= threshold)
        normal_correct = sum(1 for score, label in rows if label == "normal" and score < threshold)
        hard_recall = hard_correct / hard_total if hard_total else 0.0
        normal_recall = normal_correct / normal_total if normal_total else 0.0
        balanced = (hard_recall + normal_recall) / 2.0
        if balanced > best["balanced_accuracy"]:
            best = {"threshold": threshold, "balanced_accuracy": balanced, "sample_count": len(rows)}
    return best
