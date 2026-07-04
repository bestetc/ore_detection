"""HSV Value-channel utilities for talc candidate masks."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ImageFilter

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEFAULT_VALUE_PERCENTILES = (1, 2, 5, 10, 15, 20)
DEFAULT_VALUE_THRESHOLD = 50
DEFAULT_RGB_SUM_THRESHOLD = 150
DEFAULT_HSV_SV_SUM_THRESHOLD = 120
DEFAULT_HSV_SV_PRODUCT_THRESHOLD = 5000
DEFAULT_STATS_MAX_IMAGE_SIZE = 512
DEFAULT_STANDARDIZE_CLIP_SIGMA = None
DEFAULT_STANDARDIZE_OUTPUT_STD = 64.0
DEFAULT_STATISTIC_QUANTILES = (0, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 100)
DEFAULT_EROSION_KERNEL_SIZE = 3
DEFAULT_EROSION_ITERATIONS = 1


def _pixels(image: Image.Image):
    return image.get_flattened_data() if hasattr(image, "get_flattened_data") else image.getdata()


def iter_image_paths(root: str | Path, *, suffixes: Iterable[str] = IMAGE_SUFFIXES) -> list[Path]:
    """List image files below a root directory in stable order."""
    root = Path(root)
    suffix_set = {suffix.lower() for suffix in suffixes}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffix_set)


def iter_baseline_crop_paths(
    root: str | Path,
    *,
    panorama_folder_name: str = "panoramas",
    suffixes: Iterable[str] = IMAGE_SUFFIXES,
) -> list[Path]:
    """List baseline crop images, excluding panorama images."""
    root = Path(root)
    return [
        path
        for path in iter_image_paths(root, suffixes=suffixes)
        if panorama_folder_name.lower() not in {part.lower() for part in path.relative_to(root).parts}
    ]


def value_channel(image: Image.Image) -> Image.Image:
    """Return the HSV Value channel as an 8-bit grayscale image."""
    return image.convert("RGB").convert("HSV").getchannel("V")


def calculate_rgb_mean_std(
    image_paths: Iterable[str | Path],
    *,
    max_image_size: int | None = None,
) -> dict[str, object]:
    """Calculate RGB mean/std over a set of images.

    Pass ``max_image_size`` to compute stats from resized copies of every image,
    which keeps notebook startup practical while still sampling every crop.
    """
    channel_sum = [0.0, 0.0, 0.0]
    channel_sum_sq = [0.0, 0.0, 0.0]
    channel_count = 0
    image_count = 0

    for image_path in image_paths:
        path = Path(image_path)
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            if max_image_size is not None and max(image.size) > max_image_size:
                image = image.copy()
                image.thumbnail((max_image_size, max_image_size), Image.Resampling.BILINEAR)
            histogram = image.histogram()

        pixel_count = image.width * image.height
        channel_count += pixel_count
        image_count += 1
        values = range(256)
        for channel_index in range(3):
            counts = histogram[channel_index * 256 : (channel_index + 1) * 256]
            channel_sum[channel_index] += sum(value * count for value, count in zip(values, counts))
            channel_sum_sq[channel_index] += sum((value * value) * count for value, count in zip(values, counts))

    if channel_count == 0:
        raise ValueError("cannot calculate RGB stats from an empty image set")

    mean = tuple(value / channel_count for value in channel_sum)
    std = []
    for channel_index in range(3):
        variance = (channel_sum_sq[channel_index] / channel_count) - (mean[channel_index] * mean[channel_index])
        std.append(max(variance, 0.0) ** 0.5)
    return {
        "mean": mean,
        "std": tuple(value if value > 0 else 1.0 for value in std),
        "pixel_count": channel_count,
        "image_count": image_count,
        "max_image_size": max_image_size,
    }


def standardize_image_to_uint8(
    image: Image.Image,
    stats: dict[str, object],
    *,
    clip_sigma: Optional[float] = DEFAULT_STANDARDIZE_CLIP_SIGMA,
    output_std: float = DEFAULT_STANDARDIZE_OUTPUT_STD,
) -> Image.Image:
    """Apply RGB mean/std scaling and return an 8-bit RGB preview image.

    When ``clip_sigma`` is ``None``, z-scores are not clipped before conversion
    to the 8-bit preview range. Values outside the displayable range are only
    bounded at the final 0..255 image boundary.
    """
    if clip_sigma is not None and clip_sigma <= 0:
        raise ValueError(f"clip_sigma must be positive, got {clip_sigma}")
    if output_std <= 0:
        raise ValueError(f"output_std must be positive, got {output_std}")
    mean = tuple(float(value) for value in stats["mean"])  # type: ignore[index]
    std = tuple(float(value) if float(value) > 0 else 1.0 for value in stats["std"])  # type: ignore[index]
    rgb = image.convert("RGB")
    scaled_channels = []
    for channel, channel_mean, channel_std in zip(rgb.split(), mean, std):
        lookup = []
        for value in range(256):
            z_score = (value - channel_mean) / channel_std
            if clip_sigma is None:
                scaled = int(round((z_score * output_std) + 128))
            else:
                clipped = min(clip_sigma, max(-clip_sigma, z_score))
                scaled = int(round(((clipped + clip_sigma) / (2 * clip_sigma)) * 255))
            lookup.append(min(255, max(0, scaled)))
        scaled_channels.append(channel.point(lookup))
    return Image.merge("RGB", scaled_channels)


def statistic_histogram(image: Image.Image, *, statistic: str) -> list[int]:
    """Return an integer histogram for a supported talc-threshold statistic."""
    rgb = image.convert("RGB")
    if statistic == "hsv_value":
        return value_channel(rgb).histogram()
    if statistic == "rgb_sum":
        histogram = [0] * 766
        for r, g, b in _pixels(rgb):
            histogram[int(r) + int(g) + int(b)] += 1
        return histogram
    hsv = rgb.convert("HSV")
    if statistic == "hsv_s_plus_v":
        histogram = [0] * 511
        for _, s, v in _pixels(hsv):
            histogram[int(s) + int(v)] += 1
        return histogram
    if statistic == "hsv_s_times_v":
        histogram = [0] * 65026
        for _, s, v in _pixels(hsv):
            histogram[int(s) * int(v)] += 1
        return histogram
    raise ValueError(f"unknown statistic: {statistic}")


def quantiles_from_histogram(histogram: list[int], quantiles: Iterable[float]) -> dict[float, int]:
    """Calculate integer quantiles from a bounded integer histogram."""
    total = sum(histogram)
    if total == 0:
        raise ValueError("cannot calculate quantiles for an empty histogram")
    sorted_quantiles = [float(quantile) for quantile in quantiles]
    result: dict[float, int] = {}
    targets = [(quantile, max(1, ceil(total * (quantile / 100.0)))) for quantile in sorted_quantiles]
    cumulative = 0
    target_index = 0
    for value, count in enumerate(histogram):
        cumulative += count
        while target_index < len(targets) and cumulative >= targets[target_index][1]:
            result[targets[target_index][0]] = value
            target_index += 1
    for quantile, _ in targets[target_index:]:
        result[quantile] = len(histogram) - 1
    return result


def talc_statistic_quantile_rows(
    image: Image.Image,
    *,
    quantiles: Iterable[float] = DEFAULT_STATISTIC_QUANTILES,
) -> list[dict[str, object]]:
    """Return one quantile-table row per talc candidate statistic."""
    rows: list[dict[str, object]] = []
    for statistic, label in (
        ("hsv_value", "HSV V"),
        ("rgb_sum", "RGB sum"),
        ("hsv_s_plus_v", "HSV S+V"),
        ("hsv_s_times_v", "HSV S*V"),
    ):
        values = quantiles_from_histogram(statistic_histogram(image, statistic=statistic), quantiles)
        row: dict[str, object] = {"metric": label}
        row.update({f"q{quantile:g}": value for quantile, value in values.items()})
        rows.append(row)
    return rows


def percentile_from_histogram(histogram: list[int], percentile: float) -> int:
    """Return the 0-255 intensity threshold at a percentile from a histogram."""
    if percentile < 0 or percentile > 100:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    total = sum(histogram)
    if total == 0:
        raise ValueError("cannot calculate percentile for an empty histogram")
    target = total * (percentile / 100.0)
    cumulative = 0
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= target:
            return value
    return 255


def value_percentiles(image: Image.Image, percentiles: Iterable[float] = DEFAULT_VALUE_PERCENTILES) -> dict[float, int]:
    """Calculate HSV Value-channel percentile thresholds for one image."""
    histogram = value_channel(image).histogram()
    return {float(percentile): percentile_from_histogram(histogram, float(percentile)) for percentile in percentiles}


def dark_value_mask(image: Image.Image, *, percentile: float) -> Image.Image:
    """Create a candidate talc mask from dark Value-channel pixels."""
    threshold = value_percentiles(image, (percentile,))[float(percentile)]
    return value_channel(image).point(lambda value: 255 if value <= threshold else 0, mode="L")


def value_threshold_mask(image: Image.Image, *, threshold: int = DEFAULT_VALUE_THRESHOLD) -> Image.Image:
    """Create a talc candidate mask from pixels with HSV Value below a threshold."""
    if threshold < 0 or threshold > 255:
        raise ValueError(f"threshold must be in [0, 255], got {threshold}")
    return value_channel(image).point(lambda value: 255 if value < threshold else 0, mode="L")


def rgb_sum_threshold_mask(image: Image.Image, *, threshold: int = DEFAULT_RGB_SUM_THRESHOLD) -> Image.Image:
    """Create a talc candidate mask from pixels where R + G + B is below a threshold."""
    if threshold < 0 or threshold > 765:
        raise ValueError(f"threshold must be in [0, 765], got {threshold}")
    rgb = image.convert("RGB")
    values = [255 if int(r) + int(g) + int(b) < threshold else 0 for r, g, b in _pixels(rgb)]
    mask = Image.new("L", rgb.size)
    mask.putdata(values)
    return mask


def erode_binary_mask(
    mask: Image.Image,
    *,
    kernel_size: int = DEFAULT_EROSION_KERNEL_SIZE,
    iterations: int = DEFAULT_EROSION_ITERATIONS,
) -> Image.Image:
    """Erode white foreground regions in a binary mask."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative, got {iterations}")
    eroded = mask.convert("L").point(lambda value: 255 if value > 0 else 0, mode="L")
    for _ in range(iterations):
        eroded = eroded.filter(ImageFilter.MinFilter(kernel_size))
    return eroded


def hsv_sv_sum_threshold_mask(image: Image.Image, *, threshold: int = DEFAULT_HSV_SV_SUM_THRESHOLD) -> Image.Image:
    """Create a mask where HSV saturation plus value is below a threshold."""
    if threshold < 0 or threshold > 510:
        raise ValueError(f"threshold must be in [0, 510], got {threshold}")
    hsv = image.convert("RGB").convert("HSV")
    values = [255 if int(s) + int(v) < threshold else 0 for _, s, v in _pixels(hsv)]
    mask = Image.new("L", hsv.size)
    mask.putdata(values)
    return mask


def hsv_sv_product_threshold_mask(
    image: Image.Image,
    *,
    threshold: int = DEFAULT_HSV_SV_PRODUCT_THRESHOLD,
    value_ceiling: int = DEFAULT_VALUE_THRESHOLD,
) -> Image.Image:
    """Create a mask where HSV S*V is low and V remains dark."""
    if threshold < 0 or threshold > 65025:
        raise ValueError(f"threshold must be in [0, 65025], got {threshold}")
    if value_ceiling < 0 or value_ceiling > 255:
        raise ValueError(f"value_ceiling must be in [0, 255], got {value_ceiling}")
    hsv = image.convert("RGB").convert("HSV")
    values = [255 if int(v) < value_ceiling and int(s) * int(v) < threshold else 0 for _, s, v in _pixels(hsv)]
    mask = Image.new("L", hsv.size)
    mask.putdata(values)
    return mask


def candidate_area_fraction(mask: Image.Image) -> float:
    """Return the fraction of pixels selected by a binary candidate mask."""
    mask_l = mask.convert("L")
    total = mask_l.width * mask_l.height
    if total == 0:
        return 0.0
    selected = sum(1 for value in _pixels(mask_l) if value > 0)
    return selected / total


def mask_iou(mask_a: Image.Image, mask_b: Image.Image) -> float:
    """Return IoU between two binary masks where non-zero pixels are foreground."""
    a = mask_a.convert("L")
    b = mask_b.convert("L")
    if a.size != b.size:
        raise ValueError("mask_a and mask_b must have the same size")
    intersection = 0
    union = 0
    for value_a, value_b in zip(_pixels(a), _pixels(b)):
        foreground_a = value_a > 0
        foreground_b = value_b > 0
        if foreground_a and foreground_b:
            intersection += 1
        if foreground_a or foreground_b:
            union += 1
    return 1.0 if union == 0 else intersection / union


def overlay_mask(
    image: Image.Image,
    mask: Image.Image,
    *,
    color: tuple[int, int, int] = (0, 96, 255),
    alpha: int = 115,
) -> Image.Image:
    """Overlay a single-channel candidate mask on an RGB image."""
    base = image.convert("RGBA")
    mask_l = mask.convert("L")
    overlay = Image.new("RGBA", base.size, (*color, 0))
    overlay.putalpha(mask_l.point(lambda value: alpha if value > 0 else 0, mode="L"))
    return Image.alpha_composite(base, overlay).convert("RGB")


def summarize_value_percentiles(
    image_paths: Iterable[str | Path],
    *,
    percentiles: Iterable[float] = DEFAULT_VALUE_PERCENTILES,
) -> list[dict[str, object]]:
    """Calculate Value-channel percentile thresholds for many images."""
    rows: list[dict[str, object]] = []
    for image_path in image_paths:
        path = Path(image_path)
        with Image.open(path) as image:
            thresholds = value_percentiles(image, percentiles)
        row: dict[str, object] = {"image_path": str(path)}
        row.update({f"v_p{percentile:g}": value for percentile, value in thresholds.items()})
        rows.append(row)
    return rows
