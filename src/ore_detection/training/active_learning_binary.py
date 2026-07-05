"""Active-learning class-index masks as ore/talc finetuning samples."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image

TARGET_CLASS_NAMES = ("background", "ore", "talc")
TARGET_BACKGROUND_INDEX = 0
TARGET_ORE_INDEX = 1
TARGET_TALC_INDEX = 2

BACKGROUND_CLASS_NAMES = frozenset({"background", "background_matrix"})
TALC_CLASS_NAMES = frozenset({"talc"})
IGNORE_CLASS_NAMES = frozenset({"ignore"})


def balanced_class_weights_from_counts(
    class_counts: Sequence[int | float],
    *,
    min_weight: float = 1.0,
    max_weight: float = 8.0,
) -> tuple[float, ...]:
    """Return sqrt-inverse-frequency weights clamped to a stable range."""
    if min_weight <= 0:
        raise ValueError(f"min_weight must be positive, got {min_weight}")
    if max_weight < min_weight:
        raise ValueError(f"max_weight must be >= min_weight, got {max_weight} < {min_weight}")
    counts = [float(value) for value in class_counts]
    if not counts:
        raise ValueError("class_counts must not be empty")
    if any(value < 0 for value in counts):
        raise ValueError(f"class_counts must be non-negative, got {class_counts}")
    positive_counts = [value for value in counts if value > 0]
    if not positive_counts:
        raise ValueError("at least one class count must be positive")
    max_count = max(positive_counts)
    weights = []
    for count in counts:
        raw = max_weight if count <= 0 else math.sqrt(max_count / count)
        weights.append(min(max(raw, min_weight), max_weight))
    return tuple(weights)


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for active-learning binary training samples.") from exc
    return torch


@dataclass(frozen=True)
class ActiveLearningBinarySample:
    """One active-learning full mask or reviewed crop for ore/talc finetuning."""

    sample_id: str
    source_image_path: Path
    metadata_path: Path
    class_index_mask_path: Path | None
    tensor_tile_path: Path | None
    crop_box: tuple[int, int, int, int] | None
    width: int
    height: int
    classes: tuple[dict[str, Any], ...]


def _resolve_metadata_path(value: str | Path | None, *, metadata_path: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (metadata_path.parent / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _torch_load(path: Path) -> Any:
    torch = _require_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _classes_by_id(classes: Iterable[dict[str, Any]]) -> dict[int, str]:
    by_id: dict[int, str] = {}
    for item in classes:
        try:
            class_id = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        by_id[class_id] = str(item.get("name", class_id)).strip().lower()
    return by_id


def _sample_from_tensor_tile(
    *,
    metadata_path: Path,
    metadata: dict[str, Any],
    tile_path: Path,
    tile_index: int,
) -> ActiveLearningBinarySample | None:
    payload = _torch_load(tile_path)
    class_index = payload.get("class_index") if isinstance(payload, dict) else None
    if class_index is None or getattr(class_index, "ndim", None) != 2:
        return None
    crop_box_value = payload.get("crop_box")
    if not crop_box_value or len(crop_box_value) != 4:
        return None
    source_image = _resolve_metadata_path(
        payload.get("source_image_path") or metadata.get("source_image_path"),
        metadata_path=metadata_path,
    )
    if source_image is None or not source_image.exists():
        return None
    height, width = int(class_index.shape[0]), int(class_index.shape[1])
    return ActiveLearningBinarySample(
        sample_id=f"{metadata_path.parent.name}_patch_{tile_index:05d}",
        source_image_path=source_image,
        metadata_path=metadata_path,
        class_index_mask_path=_resolve_metadata_path(metadata.get("class_index_mask"), metadata_path=metadata_path),
        tensor_tile_path=tile_path,
        crop_box=tuple(int(value) for value in crop_box_value),
        width=width,
        height=height,
        classes=tuple(dict(item) for item in metadata.get("classes", ())),
    )


def _sample_from_full_mask(metadata_path: Path, metadata: dict[str, Any]) -> ActiveLearningBinarySample | None:
    class_index_path = _resolve_metadata_path(metadata.get("class_index_mask"), metadata_path=metadata_path)
    source_image = _resolve_metadata_path(metadata.get("source_image_path"), metadata_path=metadata_path)
    if class_index_path is None or source_image is None:
        return None
    if not class_index_path.exists() or not source_image.exists():
        return None
    with Image.open(class_index_path) as mask:
        width, height = mask.size
    return ActiveLearningBinarySample(
        sample_id=metadata_path.parent.name,
        source_image_path=source_image,
        metadata_path=metadata_path,
        class_index_mask_path=class_index_path,
        tensor_tile_path=None,
        crop_box=None,
        width=width,
        height=height,
        classes=tuple(dict(item) for item in metadata.get("classes", ())),
    )


def list_active_learning_binary_samples(
    root: str | Path = "data_work/active_learning_masks",
    *,
    use_reviewed_tiles: bool = True,
    include_full_edited_masks: bool = True,
) -> list[ActiveLearningBinarySample]:
    """List active-learning masks usable for ore/talc finetuning.

    Panorama reviews with saved ``reviewed_tiles`` contribute only reviewed crop
    tensors. Full class-index masks are included for direct browser-edited masks
    that do not have panorama patch tensors.
    """
    root = Path(root)
    if not root.exists():
        return []
    samples: list[ActiveLearningBinarySample] = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        metadata = _load_json(metadata_path)
        tensor_tiles = [
            path
            for value in metadata.get("tensor_tiles", ())
            if (path := _resolve_metadata_path(value, metadata_path=metadata_path)) is not None and path.exists()
        ]
        if use_reviewed_tiles and tensor_tiles:
            for index, tile_path in enumerate(tensor_tiles, start=1):
                sample = _sample_from_tensor_tile(
                    metadata_path=metadata_path,
                    metadata=metadata,
                    tile_path=tile_path,
                    tile_index=index,
                )
                if sample is not None:
                    samples.append(sample)
            continue
        if include_full_edited_masks and metadata.get("format") == "single_class_index_png_and_torch_one_hot_chw":
            sample = _sample_from_full_mask(metadata_path, metadata)
            if sample is not None:
                samples.append(sample)
    return samples


def _class_index_image_from_tensor(class_index: Any) -> Image.Image:
    torch = _require_torch()
    values = class_index.detach().cpu().to(dtype=torch.uint8)
    image = Image.new("L", (int(values.shape[1]), int(values.shape[0])))
    image.putdata(values.flatten().tolist())
    return image


def _target_mask_and_weight(
    class_index: Any,
    *,
    classes: Sequence[dict[str, Any]],
    background_class_names: Iterable[str] = BACKGROUND_CLASS_NAMES,
    talc_class_names: Iterable[str] = TALC_CLASS_NAMES,
    ignore_class_names: Iterable[str] = IGNORE_CLASS_NAMES,
):
    torch = _require_torch()
    values = class_index.to(dtype=torch.int64)
    names = _classes_by_id(classes)
    background_names = {str(name).lower() for name in background_class_names}
    talc_names = {str(name).lower() for name in talc_class_names}
    ignore_names = {str(name).lower() for name in ignore_class_names}
    known_ids = set(names)
    ore_ids = {
        class_id
        for class_id, class_name in names.items()
        if class_name not in background_names and class_name not in talc_names and class_name not in ignore_names
    }
    talc_ids = {class_id for class_id, class_name in names.items() if class_name in talc_names}
    ignored_ids = {class_id for class_id, class_name in names.items() if class_name in ignore_names}

    target = torch.full_like(values, TARGET_BACKGROUND_INDEX, dtype=torch.int64)
    valid = torch.zeros_like(values, dtype=torch.bool)
    for class_id in known_ids:
        valid |= values == int(class_id)
    for class_id in ignored_ids:
        valid &= values != int(class_id)
    for class_id in ore_ids:
        target = torch.where(values == int(class_id), torch.full_like(target, TARGET_ORE_INDEX), target)
    for class_id in talc_ids:
        target = torch.where(values == int(class_id), torch.full_like(target, TARGET_TALC_INDEX), target)
    return target.to(dtype=torch.uint8), valid.to(dtype=torch.uint8)


def _image_to_tensor(image: Image.Image):
    torch = _require_torch()
    rgb = image.convert("RGB")
    data = torch.tensor(list(rgb.tobytes()), dtype=torch.float32)
    return data.view(rgb.height, rgb.width, 3).permute(2, 0, 1) / 255.0


def load_active_learning_binary_sample(
    sample: ActiveLearningBinarySample,
    *,
    output_size: int | None = None,
):
    """Load one active-learning sample as image, class-index mask, and loss weight."""
    torch = _require_torch()
    if sample.tensor_tile_path is not None:
        payload = _torch_load(sample.tensor_tile_path)
        class_image = _class_index_image_from_tensor(payload["class_index"])
    elif sample.class_index_mask_path is not None:
        with Image.open(sample.class_index_mask_path) as opened:
            class_image = opened.convert("L")
    else:
        raise ValueError(f"active-learning sample has no mask source: {sample}")

    with Image.open(sample.source_image_path) as opened:
        image = opened.convert("RGB")
        if sample.crop_box is not None:
            image = image.crop(sample.crop_box)

    if image.size != class_image.size:
        image = image.resize(class_image.size, Image.Resampling.BILINEAR)
    if output_size is not None:
        size = (int(output_size), int(output_size))
        image = image.resize(size, Image.Resampling.BILINEAR)
        class_image = class_image.resize(size, Image.Resampling.NEAREST)

    class_values = torch.tensor(list(class_image.tobytes()), dtype=torch.int64).view(class_image.height, class_image.width)
    mask, weight = _target_mask_and_weight(class_values, classes=sample.classes)
    return {
        "image": _image_to_tensor(image),
        "mask": mask,
        "weight": weight,
        "sample": sample,
    }


def load_active_learning_binary_tensors(
    samples: Sequence[ActiveLearningBinarySample],
    *,
    device: str | object,
    output_size: int | None = None,
    image_dtype=None,
    pin_memory: bool = False,
) -> dict[str, object]:
    """Load active-learning ore/talc samples into stacked tensors."""
    torch = _require_torch()
    if not samples:
        raise ValueError("cannot cache an empty active-learning sample list")
    device = torch.device(device)
    if image_dtype is None:
        image_dtype = torch.float16 if device.type == "cuda" else torch.float32
    rows = [load_active_learning_binary_sample(sample, output_size=output_size) for sample in samples]
    images = torch.stack([row["image"] for row in rows]).to(device=device, dtype=image_dtype)
    masks = torch.stack([row["mask"] for row in rows]).to(device=device)
    weights = torch.stack([row["weight"] for row in rows]).to(device=device)
    if pin_memory and device.type == "cpu" and torch.cuda.is_available():
        images = images.pin_memory()
        masks = masks.pin_memory()
        weights = weights.pin_memory()
    return {"image": images, "mask": masks, "weight": weights, "samples": list(samples)}
