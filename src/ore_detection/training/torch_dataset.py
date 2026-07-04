"""Optional PyTorch dataset for source binary segmentation samples."""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

from ore_detection.training.source_dataset import SourceSegmentationSample, load_sample_images


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for SourceTorchDataset. Install the `ml` optional dependencies.") from exc
    return torch


class SourceTorchDataset:
    """PyTorch Dataset wrapper around framework-neutral source samples."""

    def __init__(self, samples: Sequence[SourceSegmentationSample], *, image_size: int | None = 512):
        self.torch = _require_torch()
        self.samples = list(samples)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image, mask = load_sample_images(sample)
        if self.image_size is not None:
            size = (self.image_size, self.image_size)
            image = image.resize(size, Image.Resampling.BILINEAR)
            mask = mask.resize(size, Image.Resampling.NEAREST)
        image_tensor = self._image_to_tensor(image)
        mask_tensor = self._mask_to_tensor(mask)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "dataset": sample.dataset,
            "split": sample.split,
            "stem": sample.stem,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
        }

    def _image_to_tensor(self, image: Image.Image):
        data = list(image.convert("RGB").tobytes())
        tensor = self.torch.tensor(data, dtype=self.torch.float32).view(image.height, image.width, 3)
        return tensor.permute(2, 0, 1) / 255.0

    def _mask_to_tensor(self, mask: Image.Image):
        data = list(mask.convert("L").tobytes())
        tensor = self.torch.tensor(data, dtype=self.torch.float32).view(1, mask.height, mask.width)
        return (tensor > 0).float()
