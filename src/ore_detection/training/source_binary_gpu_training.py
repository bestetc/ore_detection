"""GPU-centered helpers for source binary segmentation training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from PIL import Image

from ore_detection.training.source_binary_const import SOURCE_BINARY_TRAIN_RGB_MEAN, SOURCE_BINARY_TRAIN_RGB_STD


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for source binary GPU training helpers.") from exc
    return torch, F


@dataclass(frozen=True)
class DownsampledBinarySample:
    dataset: str
    split: str
    stem: str
    image_path: Path
    mask_path: Path
    width: int
    height: int


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def list_downsampled_source_binary_samples(
    root: str | Path = "data_work/source_binary_downsampled",
    *,
    split: str | None = None,
) -> list[DownsampledBinarySample]:
    """List paired downsampled source binary images and masks."""
    root = Path(root)
    image_root = root / "images"
    mask_root = root / "masks"
    samples: list[DownsampledBinarySample] = []
    for mask_path in sorted(mask_root.rglob("*.pt")):
        rel = mask_path.relative_to(mask_root)
        if len(rel.parts) != 3:
            continue
        dataset, sample_split, filename = rel.parts
        if split is not None and sample_split != split:
            continue
        stem = Path(filename).stem
        image_path = image_root / dataset / sample_split / f"{stem}.png"
        if not image_path.exists():
            continue
        width, height = _image_size(image_path)
        samples.append(
            DownsampledBinarySample(
                dataset=dataset,
                split=sample_split,
                stem=stem,
                image_path=image_path,
                mask_path=mask_path,
                width=width,
                height=height,
            )
        )
    return samples


def load_image_tensor(path: str | Path):
    """Load an RGB image as a float tensor shaped ``[3,H,W]`` in the 0..1 scale."""
    torch, _ = _require_torch()
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        data = torch.tensor(list(image.tobytes()), dtype=torch.float32)
        return data.view(image.height, image.width, 3).permute(2, 0, 1) / 255.0


def load_binary_mask(path: str | Path):
    """Load a compact binary mask tensor shaped ``[H,W]``."""
    torch, _ = _require_torch()
    try:
        mask = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        mask = torch.load(path, map_location="cpu")
    if mask.ndim != 2:
        raise ValueError(f"binary mask must be shaped [H,W], got {tuple(mask.shape)} for {path}")
    return (mask > 0).to(dtype=torch.uint8)


def load_cached_binary_tensors(
    samples: Sequence[DownsampledBinarySample],
    *,
    device: str | object,
    image_dtype=None,
    pin_memory: bool = False,
) -> dict[str, object]:
    """Load downsampled binary samples into stacked tensors."""
    torch, _ = _require_torch()
    if not samples:
        raise ValueError("cannot cache an empty sample list")
    device = torch.device(device)
    if image_dtype is None:
        image_dtype = torch.float16 if device.type == "cuda" else torch.float32

    images = []
    masks = []
    for sample in samples:
        image = load_image_tensor(sample.image_path)
        mask = load_binary_mask(sample.mask_path)
        if image.shape[1:] != mask.shape:
            raise ValueError(f"image/mask size mismatch for {sample.image_path} and {sample.mask_path}")
        images.append(image)
        masks.append(mask)
    image_tensor = torch.stack(images).to(device=device, dtype=image_dtype)
    mask_tensor = torch.stack(masks).to(device=device)
    if pin_memory and device.type == "cpu" and torch.cuda.is_available():
        image_tensor = image_tensor.pin_memory()
        mask_tensor = mask_tensor.pin_memory()
    return {"image": image_tensor, "mask": mask_tensor, "samples": list(samples)}


def iter_cached_binary_batches(
    cached: dict[str, object],
    *,
    batch_size: int,
    shuffle: bool,
    generator=None,
) -> Iterator[dict[str, object]]:
    """Yield batches from tensors returned by ``load_cached_binary_tensors``."""
    torch, _ = _require_torch()
    images = cached["image"]
    masks = cached["mask"]
    sample_count = int(images.shape[0])
    if shuffle:
        order = torch.randperm(sample_count, device=images.device, generator=generator)
    else:
        order = torch.arange(sample_count, device=images.device)
    samples = cached.get("samples", [])
    for start in range(0, sample_count, batch_size):
        index = order[start : start + batch_size]
        yield {
            "image": images.index_select(0, index).float(),
            "mask": masks.index_select(0, index).float().unsqueeze(1),
            "samples": [samples[int(i.detach().cpu())] for i in index] if samples else [],
        }


def normalize_binary_images(
    images,
    *,
    mean: tuple[float, float, float] = SOURCE_BINARY_TRAIN_RGB_MEAN,
    std: tuple[float, float, float] = SOURCE_BINARY_TRAIN_RGB_STD,
):
    """Apply train-set standard scaling to image tensors."""
    torch, _ = _require_torch()
    mean_tensor = torch.tensor(mean, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=images.dtype, device=images.device).view(1, 3, 1, 1).clamp_min(1e-6)
    return (images - mean_tensor) / std_tensor


def binary_iou_from_logits(logits, target, threshold: float = 0.5, eps: float = 1e-7):
    """Calculate binary IoU from logits and binary targets."""
    torch, _ = _require_torch()
    probs = torch.sigmoid(logits)
    pred = probs > threshold
    target_bool = target > 0.5
    intersection = (pred & target_bool).sum(dim=(1, 2, 3)).float()
    union = (pred | target_bool).sum(dim=(1, 2, 3)).float()
    return ((intersection + eps) / (union + eps)).mean()


def binary_dice_loss_from_logits(logits, target, eps: float = 1e-6):
    """Calculate soft binary Dice loss from logits and binary targets."""
    torch, _ = _require_torch()
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1 - dice.mean()


def _pad_to_minimum(image, mask, *, target_size: int):
    torch, F = _require_torch()
    _, height, width = image.shape
    pad_height = max(0, target_size - height)
    pad_width = max(0, target_size - width)
    if pad_height == 0 and pad_width == 0:
        return image, mask
    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top
    image = F.pad(image.unsqueeze(0), (left, right, top, bottom), value=0.0).squeeze(0)
    mask = F.pad(mask.unsqueeze(0), (left, right, top, bottom), value=0.0).squeeze(0)
    return image, mask


def _random_crop(image, mask, *, output_size: int):
    torch, _ = _require_torch()
    _, height, width = image.shape
    top = 0 if height == output_size else int(torch.randint(0, height - output_size + 1, (), device=image.device).item())
    left = 0 if width == output_size else int(torch.randint(0, width - output_size + 1, (), device=image.device).item())
    return image[:, top : top + output_size, left : left + output_size], mask[:, top : top + output_size, left : left + output_size]


def augment_source_binary_batch(
    images,
    masks,
    *,
    output_size: int,
    hflip_p: float = 0.5,
    vflip_p: float = 0.5,
    scale_range: tuple[float, float] = (0.5, 2.0),
    brightness_range: tuple[float, float] = (0.6, 1.4),
    contrast_range: tuple[float, float] = (0.6, 1.4),
):
    """Apply train-only GPU augmentation to images and binary masks."""
    torch, F = _require_torch()
    if images.ndim != 4:
        raise ValueError(f"images must be shaped [B,3,H,W], got {tuple(images.shape)}")
    if masks.ndim != 4:
        raise ValueError(f"masks must be shaped [B,1,H,W], got {tuple(masks.shape)}")
    scale_min, scale_max = scale_range
    if output_size < 1:
        raise ValueError(f"output_size must be positive, got {output_size}")
    if scale_min <= 0 or scale_max < scale_min:
        raise ValueError(f"invalid scale_range: {scale_range}")

    augmented_images = []
    augmented_masks = []
    for sample_index in range(int(images.shape[0])):
        image = images[sample_index]
        mask = masks[sample_index]

        if float(torch.rand((), device=images.device).item()) < hflip_p:
            image = torch.flip(image, dims=(2,))
            mask = torch.flip(mask, dims=(2,))
        if float(torch.rand((), device=images.device).item()) < vflip_p:
            image = torch.flip(image, dims=(1,))
            mask = torch.flip(mask, dims=(1,))

        rotation = int(torch.randint(0, 4, (), device=images.device).item())
        if rotation:
            image = torch.rot90(image, rotation, dims=(1, 2))
            mask = torch.rot90(mask, rotation, dims=(1, 2))

        scale = scale_min + (scale_max - scale_min) * float(torch.rand((), device=images.device).item())
        scaled_height = max(1, int(round(image.shape[1] * scale)))
        scaled_width = max(1, int(round(image.shape[2] * scale)))
        image = F.interpolate(
            image.unsqueeze(0),
            size=(scaled_height, scaled_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mask = F.interpolate(
            mask.unsqueeze(0),
            size=(scaled_height, scaled_width),
            mode="nearest",
        ).squeeze(0)

        image, mask = _pad_to_minimum(image, mask, target_size=output_size)
        image, mask = _random_crop(image, mask, output_size=output_size)

        brightness = brightness_range[0] + (brightness_range[1] - brightness_range[0]) * float(
            torch.rand((), device=images.device).item()
        )
        contrast = contrast_range[0] + (contrast_range[1] - contrast_range[0]) * float(
            torch.rand((), device=images.device).item()
        )
        channel_mean = image.mean(dim=(1, 2), keepdim=True)
        image = ((image - channel_mean) * contrast + channel_mean) * brightness
        image = image.clamp(0.0, 1.0)
        mask = (mask > 0.5).float()

        augmented_images.append(image)
        augmented_masks.append(mask)

    return torch.stack(augmented_images), torch.stack(augmented_masks)
