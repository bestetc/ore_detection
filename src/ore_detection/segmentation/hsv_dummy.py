"""HSV-Value dummy binary segmentation.

This is a deterministic placeholder for UI/backend development. It is not a
trained model: foreground is selected from the HSV Value channel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from PIL import Image

from ore_detection.talc.hsv_candidates import standardize_image_to_uint8, value_channel

ForegroundMode = Literal["bright", "dark"]


@dataclass(frozen=True)
class HsvDummyConfig:
    value_threshold: int = 90
    foreground: ForegroundMode = "bright"
    standardize: bool = False
    standardize_stats: dict[str, object] | None = None
    clip_sigma: float = 3.0

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        if data["standardize_stats"] is not None:
            stats = data["standardize_stats"]
            data["standardize_stats"] = {
                "mean": list(stats["mean"]),
                "std": list(stats["std"]),
            }
        return data


def _validate_config(config: HsvDummyConfig) -> None:
    if config.value_threshold < 0 or config.value_threshold > 255:
        raise ValueError(f"value_threshold must be in [0, 255], got {config.value_threshold}")
    if config.foreground not in {"bright", "dark"}:
        raise ValueError(f"foreground must be 'bright' or 'dark', got {config.foreground!r}")
    if config.standardize and config.standardize_stats is None:
        raise ValueError("standardize_stats are required when standardize=True")


def preprocess_for_hsv(image: Image.Image, config: HsvDummyConfig) -> Image.Image:
    """Apply optional standard scaling before HSV thresholding."""
    _validate_config(config)
    rgb = image.convert("RGB")
    if not config.standardize:
        return rgb
    return standardize_image_to_uint8(rgb, config.standardize_stats or {}, clip_sigma=config.clip_sigma)


def hsv_value_confidence(image: Image.Image, config: HsvDummyConfig | None = None) -> Image.Image:
    """Return HSV Value channel, after optional preprocessing, as 8-bit confidence."""
    cfg = config or HsvDummyConfig()
    return value_channel(preprocess_for_hsv(image, cfg))


def hsv_value_binary_mask(image: Image.Image, config: HsvDummyConfig | None = None) -> Image.Image:
    """Create a 0/1 binary mask from HSV Value thresholding."""
    cfg = config or HsvDummyConfig()
    value = hsv_value_confidence(image, cfg)
    threshold = cfg.value_threshold
    if cfg.foreground == "bright":
        return value.point(lambda pixel: 1 if pixel >= threshold else 0, mode="L")
    return value.point(lambda pixel: 1 if pixel < threshold else 0, mode="L")
