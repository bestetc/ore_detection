"""Dataset indexing and PyTorch wrapper for multiclass source ore masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

from ore_detection.data.ore_type_legend import OreTypeLegend, load_legend_config

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for SourceOreTorchDataset. Install the `ml` optional dependencies.") from exc
    return torch


@dataclass(frozen=True)
class SourceOreSegmentationSample:
    dataset: str
    split: str
    stem: str
    image_path: Path
    mask_path: Path
    width: int
    height: int


def _find_image(image_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        for candidate in (image_dir / f"{stem}{suffix}", image_dir / f"{stem}{suffix.upper()}"):
            if candidate.exists():
                return candidate

    sample_dir = image_dir / stem
    if sample_dir.exists():
        for suffix in IMAGE_SUFFIXES:
            for candidate in (sample_dir / f"{stem}_r000{suffix}", sample_dir / f"{stem}_r000{suffix.upper()}"):
                if candidate.exists():
                    return candidate
        recursive_matches = sorted(path for path in sample_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        if recursive_matches:
            return recursive_matches[0]

    matches = sorted(path for path in image_dir.rglob(f"{stem}*") if path.suffix.lower() in IMAGE_SUFFIXES)
    return matches[0] if matches else None


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def list_source_ore_samples(
    *,
    datasets_root: str | Path = "datasets",
    masks_root: str | Path = "data_work/source_ore_type_masks",
    datasets: Iterable[str] = ("set_1", "set_2", "set_3"),
    split: str | None = None,
) -> list[SourceOreSegmentationSample]:
    """List source samples paired with saved one-hot `.pt` masks."""
    datasets_root = Path(datasets_root)
    masks_root = Path(masks_root)
    samples: list[SourceOreSegmentationSample] = []

    for dataset in datasets:
        dataset_mask_root = masks_root / dataset
        if not dataset_mask_root.exists():
            continue
        for mask_path in sorted(dataset_mask_root.rglob("*.pt")):
            rel = mask_path.relative_to(dataset_mask_root)
            if len(rel.parts) < 2:
                continue
            sample_split = rel.parts[0]
            if split is not None and sample_split != split:
                continue
            image_path = _find_image(datasets_root / dataset / "imgs" / sample_split, mask_path.stem)
            if image_path is None:
                continue
            width, height = _image_size(image_path)
            samples.append(
                SourceOreSegmentationSample(
                    dataset=dataset,
                    split=sample_split,
                    stem=mask_path.stem,
                    image_path=image_path,
                    mask_path=mask_path,
                    width=width,
                    height=height,
                )
            )
    return samples


def load_sample_image_and_mask(sample: SourceOreSegmentationSample):
    """Load a source image and one-hot mask tensor."""
    torch = _require_torch()
    image = Image.open(sample.image_path).convert("RGB")
    try:
        mask = torch.load(sample.mask_path, map_location="cpu", weights_only=True)
    except TypeError:
        mask = torch.load(sample.mask_path, map_location="cpu")
    if mask.ndim != 3:
        raise ValueError(f"mask tensor must be shaped [C,H,W], got {tuple(mask.shape)} for {sample.mask_path}")
    if image.size != (int(mask.shape[2]), int(mask.shape[1])):
        raise ValueError(f"image/mask size mismatch for {sample.image_path} and {sample.mask_path}")
    return image, mask


class SourceOreTorchDataset:
    """PyTorch Dataset wrapper around multiclass source ore samples."""

    def __init__(
        self,
        samples: Sequence[SourceOreSegmentationSample],
        *,
        legend: OreTypeLegend | None = None,
        image_size: int | tuple[int, int] | None = 512,
    ):
        self.torch = _require_torch()
        self.samples = list(samples)
        self.legend = legend if legend is not None else load_legend_config()
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image, mask = load_sample_image_and_mask(sample)
        if int(mask.shape[0]) != self.legend.class_count:
            raise ValueError(
                f"{sample.mask_path} has {mask.shape[0]} channels, expected {self.legend.class_count}"
            )

        if self.image_size is not None:
            image_size = self._resolve_image_size(self.image_size)
            image = image.resize(image_size, Image.Resampling.BILINEAR)
            mask = self._resize_mask(mask, image_size)

        image_tensor = self._image_to_tensor(image)
        mask_tensor = mask.float()
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "class_index": mask_tensor.argmax(dim=0).long(),
            "dataset": sample.dataset,
            "split": sample.split,
            "stem": sample.stem,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
        }

    @staticmethod
    def _resolve_image_size(image_size: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(image_size, int):
            return (image_size, image_size)
        return image_size

    def _resize_mask(self, mask, image_size: tuple[int, int]):
        height_width = (image_size[1], image_size[0])
        return self.torch.nn.functional.interpolate(
            mask.float().unsqueeze(0),
            size=height_width,
            mode="nearest",
        ).squeeze(0)

    def _image_to_tensor(self, image: Image.Image):
        data = list(image.convert("RGB").tobytes())
        tensor = self.torch.tensor(data, dtype=self.torch.float32).view(image.height, image.width, 3)
        return tensor.permute(2, 0, 1) / 255.0
