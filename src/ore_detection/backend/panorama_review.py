"""Server-side panorama tile rendering and mask-edit persistence."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from ore_detection.backend.ui_annotation import (
    BASE_UI_CLASSES,
    UiClass,
    class_index_to_color_image,
    palette_from_classes,
    ui_classes_for_model,
)
from ore_detection.inference.tiled_prediction import allow_large_pillow_images
from ore_detection.visualization.overlay import overlay_mask_on_image


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


def restore_base_prediction(sample_dir: str | Path) -> dict[str, Any]:
    """Clear review patches so the rendered mask returns to the immutable prediction."""
    sample_dir = Path(sample_dir)
    path = patch_log_path(sample_dir)
    if path.exists():
        path.unlink()
    return {"ok": True, "patch_log_cleared": True, "sample_dir": str(sample_dir)}


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
    target_size = None
    if output_width is not None or output_height is not None:
        target_size = (max(1, int(output_width or width)), max(1, int(output_height or height)))

    with allow_large_pillow_images():
        if layer == "raw":
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
            with Image.open(_metadata_artifact_path(metadata, "intergrowth_mask")) as mask:
                if target_size is None:
                    class_tile = mask.crop(box).convert("L")
                else:
                    class_tile = mask.resize(target_size, Image.Resampling.NEAREST, box=box).convert("L")
            output = overlay_mask_on_image(raw_tile, class_tile, palette=intergrowth_palette).convert("RGB")
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
            artifact_name = layer
            with Image.open(_metadata_artifact_path(metadata, artifact_name)) as image:
                if target_size is None:
                    output = image.crop(box).convert("L")
                else:
                    output = image.resize(target_size, Image.Resampling.BILINEAR, box=box).convert("L")
        else:
            raise ValueError(f"unknown panorama tile layer: {layer}")

    return output


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
    with allow_large_pillow_images():
        mask_path = _metadata_artifact_path(metadata, "intergrowth_mask") if use_intergrowth else review_mask_path(sample_dir, metadata)
        with Image.open(mask_path) as mask:
            if patches:
                class_tile = apply_patches_to_mask_tile(mask.crop(box), origin_x=box[0], origin_y=box[1], patches=patches)
            else:
                class_tile = mask.crop(box).convert("L")
    total = max(1, class_tile.width * class_tile.height)
    counts: dict[int, int] = {}
    for value in class_tile.tobytes():
        if value == 0:
            continue
        counts[value] = counts.get(value, 0) + 1
    return {
        "box": {"x": box[0], "y": box[1], "width": box[2] - box[0], "height": box[3] - box[1]},
        "total_pixels": total,
        "classes": [
            {
                "id": class_id,
                "name": name_by_id.get(class_id, f"class_{class_id}"),
                "pixels": count,
                "fraction": count / total,
            }
            for class_id, count in sorted(counts.items())
        ],
    }


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
        values = torch.tensor(list(crop.tobytes()), dtype=torch.int64).view(crop.height, crop.width)
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
