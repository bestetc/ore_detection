"""UI annotation taxonomy, mask rendering, and active-learning persistence."""

from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class UiClass:
    """One editable UI mask class."""

    id: int
    name: str
    color: tuple[int, int, int]
    meaning: str
    editable: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "color": list(self.color),
            "meaning": self.meaning,
            "editable": self.editable,
        }


BASE_UI_CLASSES: tuple[UiClass, ...] = (
    UiClass(0, "background", (0, 0, 0), "background / non-selected matrix"),
    UiClass(1, "sulfide_ore", (0, 220, 0), "binary ore or generic sulfide ore proposal"),
    UiClass(2, "oxide_magnetite_hematite", (255, 64, 64), "oxide / magnetite / hematite ore class"),
    UiClass(3, "talc", (255, 255, 255), "reviewed talc mask; candidate created from dark-pixel threshold"),
    UiClass(4, "normal_ore", (0, 120, 255), "reviewed normal/coarse intergrowth region"),
    UiClass(5, "hard_ore", (255, 220, 0), "reviewed hard/thin/ragged intergrowth region"),
    UiClass(255, "ignore", (255, 0, 255), "unknown / invalid / do not train"),
)

MODEL_COLOR_CYCLE: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (0, 220, 0),
    (255, 180, 0),
    (255, 80, 80),
    (80, 160, 255),
    (200, 120, 255),
    (0, 220, 220),
    (180, 220, 80),
    (255, 120, 200),
    (160, 120, 60),
    (120, 255, 120),
    (120, 120, 255),
    (255, 255, 120),
    (255, 140, 80),
    (160, 160, 160),
)

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[-\w.]+/[-\w.+]+);base64,(?P<data>.+)$", re.DOTALL)


def ui_classes_for_model(class_names: Iterable[str] | None = None) -> tuple[UiClass, ...]:
    """Return editable classes for the current model plus talc/normal/hard extras.

    If a multiclass model exposes class names, preserve its class indices for the
    initial class-index mask. Extra UI-only classes are appended after the model
    channels. For the HSV/binary path, use the stable base UI taxonomy.
    """
    names = tuple(str(name) for name in (class_names or ()) if str(name))
    if not names:
        return BASE_UI_CLASSES

    classes: list[UiClass] = []
    for index, name in enumerate(names):
        color = MODEL_COLOR_CYCLE[index % len(MODEL_COLOR_CYCLE)]
        normalized_name = name.lower()
        if index == 0 or normalized_name in {"background", "background_matrix"}:
            color = (0, 0, 0)
        elif normalized_name == "talc":
            color = (255, 255, 255)
        classes.append(UiClass(index, name, color, f"model class `{name}`"))

    next_id = len(classes)
    reserved_extras = (
        ("talc", (255, 255, 255), "reviewed talc mask; candidate created from dark-pixel threshold"),
        ("normal_ore", (0, 120, 255), "reviewed normal/coarse intergrowth region"),
        ("hard_ore", (255, 220, 0), "reviewed hard/thin/ragged intergrowth region"),
    )
    existing = {item.name.lower() for item in classes}
    for name, color, meaning in reserved_extras:
        if name not in existing:
            classes.append(UiClass(next_id, name, color, meaning))
            next_id += 1
    classes.append(UiClass(255, "ignore", (255, 0, 255), "unknown / invalid / do not train"))
    return tuple(classes)


def ui_class_metadata(class_names: Iterable[str] | None = None) -> list[dict[str, Any]]:
    return [item.as_dict() for item in ui_classes_for_model(class_names)]


def palette_from_classes(classes: Iterable[UiClass]) -> dict[int, tuple[int, int, int]]:
    return {item.id: item.color for item in classes}


def class_index_to_color_image(mask: Image.Image, *, classes: Iterable[UiClass] = BASE_UI_CLASSES) -> Image.Image:
    """Colorize a class-index mask for UI display."""
    mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)
    lookup = np.full((256, 3), 255, dtype=np.uint8)
    for class_id, color in palette_from_classes(classes).items():
        lookup[int(class_id) & 255] = color
    return Image.fromarray(lookup[mask_array])


def _decode_data_url(data_url: str) -> tuple[str, bytes]:
    match = _DATA_URL_RE.match(data_url.strip())
    if not match:
        raise ValueError("expected a base64 data URL")
    try:
        return match.group("mime"), base64.b64decode(match.group("data"), validate=True)
    except binascii.Error as exc:
        raise ValueError("data URL contains invalid base64") from exc


def image_from_data_url(data_url: str) -> Image.Image:
    """Decode an image data URL into a PIL image."""
    mime, raw = _decode_data_url(data_url)
    if not mime.startswith("image/"):
        raise ValueError(f"expected image data URL, got {mime}")
    with Image.open(BytesIO(raw)) as opened:
        return opened.copy()


def save_uploaded_image_from_data_url(
    *,
    file_name: str,
    image_data_url: str,
    output_root: str | Path,
) -> Path:
    """Persist a drag-and-dropped UI image and return its server path."""
    safe_name = Path(file_name or "uploaded.png").name
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(safe_name).stem).strip("._") or "uploaded"
    suffix = Path(safe_name).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        suffix = ".png"
    image = image_from_data_url(image_data_url).convert("RGB")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"{stem}{suffix}"
    counter = 1
    while path.exists():
        path = output_root / f"{stem}-{counter}{suffix}"
        counter += 1
    image.save(path)
    return path


def mask_metrics(mask: Image.Image, *, classes: Iterable[UiClass]) -> dict[str, float]:
    """Return area fractions for requested UI metrics."""
    values = np.asarray(mask.convert("L"), dtype=np.uint8)
    total = max(1, int(values.size))
    id_by_name = {item.name: item.id for item in classes}
    result: dict[str, float] = {}
    for name in ("hard_ore", "normal_ore", "talc"):
        class_id = id_by_name.get(name)
        result[name] = (int(np.count_nonzero(values == int(class_id))) / total) if class_id is not None else 0.0
    return result


def save_edited_mask_from_data_url(
    *,
    source_image_path: str | Path,
    mask_data_url: str,
    output_root: str | Path,
    classes: Iterable[dict[str, Any]] | Iterable[UiClass],
) -> dict[str, Any]:
    """Save edited class-index mask plus a torch one-hot tensor for training loops."""
    source = Path(source_image_path)
    image = image_from_data_url(mask_data_url).convert("L")
    sample_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem).strip("._") or "sample"
    output_dir = Path(output_root) / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)

    class_items: list[dict[str, Any]] = []
    for item in classes:
        if isinstance(item, UiClass):
            class_items.append(item.as_dict())
        else:
            class_items.append(dict(item))
    channel_classes = [item for item in class_items if int(item["id"]) != 255]
    channel_ids = [int(item["id"]) for item in channel_classes]

    class_index_path = output_dir / "class_index_mask.png"
    one_hot_path = output_dir / "one_hot_mask.pt"
    metadata_path = output_dir / "metadata.json"
    color_preview_path = output_dir / "mask_preview.png"

    image.save(class_index_path)
    class_objects = [
        UiClass(int(item["id"]), str(item["name"]), tuple(int(v) for v in item["color"]), str(item.get("meaning", "")))
        for item in class_items
    ]
    class_index_to_color_image(image, classes=class_objects).save(color_preview_path)

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required to save active-learning one-hot tensors") from exc

    height, width = image.height, image.width
    values = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).to(dtype=torch.int64).view(height, width)
    one_hot = torch.stack([(values == class_id).to(dtype=torch.uint8) for class_id in channel_ids], dim=0)
    torch.save(
        {
            "one_hot": one_hot,
            "class_index": values.to(dtype=torch.uint8),
            "channel_class_ids": channel_ids,
            "channel_class_names": [str(item["name"]) for item in channel_classes],
            "source_image_path": str(source_image_path),
        },
        one_hot_path,
    )

    metadata = {
        "source_image_path": str(source_image_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format": "single_class_index_png_and_torch_one_hot_chw",
        "class_index_mask": str(class_index_path),
        "one_hot_tensor": str(one_hot_path),
        "mask_preview": str(color_preview_path),
        "classes": class_items,
        "channel_class_ids": channel_ids,
        "metrics": mask_metrics(image, classes=class_objects),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    return metadata
