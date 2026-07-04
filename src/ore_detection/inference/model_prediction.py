"""Inference helpers for trained binary and multiclass ore segmentation models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from ore_detection.models.simple_unet import create_simple_unet
from ore_detection.visualization.overlay import save_overlay


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for model prediction. Install the `ml` optional dependencies.") from exc
    return torch, F


def _torch_load(path: Path, *, map_location: str | object = "cpu") -> Any:
    torch, _ = _require_torch()
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


@dataclass(frozen=True)
class CheckpointMetadata:
    """Serializable metadata resolved from a trained segmentation checkpoint."""

    path: Path
    task: str
    out_channels: int
    class_names: tuple[str, ...]
    background_index: int | None
    image_size: int | None
    epoch: int | None
    notebook: str | None
    best_test_loss: float | None
    train_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "task": self.task,
            "out_channels": self.out_channels,
            "class_names": list(self.class_names),
            "background_index": self.background_index,
            "image_size": self.image_size,
            "epoch": self.epoch,
            "notebook": self.notebook,
            "best_test_loss": self.best_test_loss,
            "train_metrics": _jsonable(self.train_metrics),
            "test_metrics": _jsonable(self.test_metrics),
            "normalization": {
                "mean": list(self.normalization_mean),
                "std": list(self.normalization_std),
            },
        }


@dataclass(frozen=True)
class LoadedSegmentationModel:
    """A loaded model plus checkpoint metadata."""

    model: Any
    metadata: CheckpointMetadata
    device: Any


@dataclass(frozen=True)
class BinaryLogitPrediction:
    probability: Any
    mask: Any
    confidence: Any
    threshold: float


@dataclass(frozen=True)
class MulticlassLogitPrediction:
    probabilities: Any
    class_index: Any
    class_probability: Any
    class_names: tuple[str, ...]
    background_index: int


@dataclass(frozen=True)
class SegmentationPrediction:
    ore_mask: Image.Image
    ore_probability: Image.Image
    ore_confidence: Image.Image
    multiclass_mask: Image.Image | None
    multiclass_confidence: Image.Image | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SegmentationPredictionArtifacts:
    sample_dir: Path
    ore_mask_path: Path
    ore_probability_path: Path
    ore_confidence_path: Path
    overlay_path: Path
    metadata_path: Path
    multiclass_mask_path: Path | None = None
    multiclass_confidence_path: Path | None = None


def _state_dict_from_checkpoint(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and "head.weight" in checkpoint:
        return checkpoint
    raise ValueError("checkpoint does not contain a SimpleUNet state_dict under key 'model'")


def _out_channels_from_state_dict(state_dict: dict[str, Any]) -> int:
    head_weight = state_dict.get("head.weight")
    if head_weight is None:
        raise ValueError("checkpoint state_dict is missing 'head.weight'")
    return int(head_weight.shape[0])


def read_checkpoint_metadata(path: str | Path) -> CheckpointMetadata:
    """Read checkpoint metadata without constructing a model."""
    path = Path(path)
    checkpoint = _torch_load(path, map_location="cpu")
    state_dict = _state_dict_from_checkpoint(checkpoint)
    out_channels = _out_channels_from_state_dict(state_dict)

    class_names = tuple(str(name) for name in checkpoint.get("class_names", ())) if isinstance(checkpoint, dict) else ()
    if class_names and len(class_names) != out_channels:
        raise ValueError(
            f"checkpoint class_names length ({len(class_names)}) does not match out_channels ({out_channels})"
        )
    if not class_names and out_channels > 1:
        class_names = tuple(f"class_{index}" for index in range(out_channels))

    task = "binary" if out_channels == 1 else "multiclass"
    normalization = checkpoint.get("normalization", {}) if isinstance(checkpoint, dict) else {}
    mean = tuple(float(value) for value in normalization.get("mean", (0.0, 0.0, 0.0)))
    std = tuple(float(value) if float(value) > 0 else 1.0 for value in normalization.get("std", (1.0, 1.0, 1.0)))
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("checkpoint normalization mean/std must each contain three RGB values")

    background_index = checkpoint.get("background_index") if isinstance(checkpoint, dict) else None
    return CheckpointMetadata(
        path=path,
        task=task,
        out_channels=out_channels,
        class_names=class_names,
        background_index=int(background_index) if background_index is not None else (0 if task == "multiclass" else None),
        image_size=int(checkpoint["image_size"]) if isinstance(checkpoint, dict) and checkpoint.get("image_size") else None,
        epoch=int(checkpoint["epoch"]) if isinstance(checkpoint, dict) and checkpoint.get("epoch") is not None else None,
        notebook=str(checkpoint["notebook"]) if isinstance(checkpoint, dict) and checkpoint.get("notebook") else None,
        best_test_loss=float(checkpoint["best_test_loss"])
        if isinstance(checkpoint, dict) and checkpoint.get("best_test_loss") is not None
        else None,
        train_metrics=dict(checkpoint.get("train_metrics", {})) if isinstance(checkpoint, dict) else {},
        test_metrics=dict(checkpoint.get("test_metrics", {})) if isinstance(checkpoint, dict) else {},
        normalization_mean=mean,  # type: ignore[arg-type]
        normalization_std=std,  # type: ignore[arg-type]
    )


def load_simple_unet_checkpoint(path: str | Path, *, device: str | object = "cpu") -> LoadedSegmentationModel:
    """Load an existing SimpleUNet checkpoint for inference."""
    torch, _ = _require_torch()
    device = torch.device(device)
    path = Path(path)
    checkpoint = _torch_load(path, map_location=device)
    state_dict = _state_dict_from_checkpoint(checkpoint)
    metadata = read_checkpoint_metadata(path)
    model = create_simple_unet(out_channels=metadata.out_channels).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return LoadedSegmentationModel(model=model, metadata=metadata, device=device)


def binary_prediction_from_logits(logits: Any, *, threshold: float = 0.5) -> BinaryLogitPrediction:
    """Convert binary logits to ore probability, mask, and confidence tensors."""
    torch, _ = _require_torch()
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if logits.ndim != 4 or int(logits.shape[1]) != 1:
        raise ValueError(f"binary logits must be shaped [B,1,H,W], got {tuple(logits.shape)}")
    probability = torch.sigmoid(logits)
    mask = (probability >= threshold).to(dtype=torch.uint8)
    confidence = torch.maximum(probability, 1.0 - probability)
    return BinaryLogitPrediction(probability=probability, mask=mask, confidence=confidence, threshold=threshold)


def _binary_mask_without_channel(binary_mask: Any):
    if binary_mask.ndim == 4 and int(binary_mask.shape[1]) == 1:
        return binary_mask[:, 0]
    if binary_mask.ndim == 3:
        return binary_mask
    raise ValueError(f"binary mask must be shaped [B,1,H,W] or [B,H,W], got {tuple(binary_mask.shape)}")


def clip_class_index_to_binary(class_index: Any, binary_mask: Any, *, background_index: int = 0):
    """Set multiclass labels outside the binary ore mask to background."""
    torch, _ = _require_torch()
    if class_index.ndim != 3:
        raise ValueError(f"class_index must be shaped [B,H,W], got {tuple(class_index.shape)}")
    mask = _binary_mask_without_channel(binary_mask).to(device=class_index.device)
    if tuple(mask.shape) != tuple(class_index.shape):
        raise ValueError(f"class_index and binary_mask spatial shapes differ: {tuple(class_index.shape)} vs {tuple(mask.shape)}")
    background = torch.full_like(class_index, int(background_index))
    return torch.where(mask.bool(), class_index, background)


def multiclass_prediction_from_logits(
    logits: Any,
    *,
    class_names: tuple[str, ...] = (),
    background_index: int = 0,
    clip_binary_mask: Any | None = None,
) -> MulticlassLogitPrediction:
    """Convert multiclass logits to class labels and probabilities."""
    torch, _ = _require_torch()
    if logits.ndim != 4:
        raise ValueError(f"multiclass logits must be shaped [B,C,H,W], got {tuple(logits.shape)}")
    class_count = int(logits.shape[1])
    if class_count < 2:
        raise ValueError("multiclass logits must have at least two channels")
    if background_index < 0 or background_index >= class_count:
        raise ValueError(f"background_index must be in [0,{class_count}), got {background_index}")
    if class_names and len(class_names) != class_count:
        raise ValueError(f"class_names length ({len(class_names)}) does not match class count ({class_count})")

    probabilities = torch.softmax(logits, dim=1)
    class_probability, class_index = probabilities.max(dim=1)
    if clip_binary_mask is not None:
        mask = _binary_mask_without_channel(clip_binary_mask).to(device=class_index.device)
        class_index = clip_class_index_to_binary(class_index, mask, background_index=background_index)
        class_probability = torch.where(mask.bool(), class_probability, probabilities[:, background_index])
    resolved_names = class_names or tuple(f"class_{index}" for index in range(class_count))
    return MulticlassLogitPrediction(
        probabilities=probabilities,
        class_index=class_index,
        class_probability=class_probability,
        class_names=resolved_names,
        background_index=background_index,
    )


def _image_to_tensor(image: Image.Image, metadata: CheckpointMetadata, *, device: Any):
    torch, _ = _require_torch()
    rgb = image.convert("RGB")
    target_size = metadata.image_size or max(1, max(rgb.size))
    resized = rgb.resize((target_size, target_size), Image.Resampling.BILINEAR)
    data = torch.tensor(list(resized.tobytes()), dtype=torch.float32, device=device)
    tensor = data.view(resized.height, resized.width, 3).permute(2, 0, 1).unsqueeze(0) / 255.0
    mean = torch.tensor(metadata.normalization_mean, dtype=tensor.dtype, device=device).view(1, 3, 1, 1)
    std = torch.tensor(metadata.normalization_std, dtype=tensor.dtype, device=device).view(1, 3, 1, 1).clamp_min(1e-6)
    return (tensor - mean) / std


def _single_channel_to_image(tensor: Any, *, scale: int = 255) -> Image.Image:
    if tensor.ndim == 4:
        tensor = tensor[0, 0]
    elif tensor.ndim == 3:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"expected a single-channel tensor, got shape {tuple(tensor.shape)}")
    values = (tensor.detach().cpu().float().clamp(0.0, 1.0) * scale).round().to(dtype=_require_torch()[0].uint8)
    image = Image.new("L", (int(values.shape[1]), int(values.shape[0])))
    image.putdata(values.flatten().tolist())
    return image


def _class_index_to_image(class_index: Any) -> Image.Image:
    if class_index.ndim == 3:
        class_index = class_index[0]
    if class_index.ndim != 2:
        raise ValueError(f"expected class_index shaped [H,W] or [1,H,W], got {tuple(class_index.shape)}")
    if int(class_index.max().detach().cpu()) > 255:
        raise ValueError("class_index contains values above 255 and cannot be stored as an 8-bit PNG")
    values = class_index.detach().cpu().to(dtype=_require_torch()[0].uint8)
    image = Image.new("L", (int(values.shape[1]), int(values.shape[0])))
    image.putdata(values.flatten().tolist())
    return image


def clip_class_image_to_binary(class_image: Image.Image, binary_mask: Image.Image, *, background_index: int = 0) -> Image.Image:
    """Clip a class-index PIL mask to a binary ore mask."""
    class_l = class_image.convert("L")
    mask_l = binary_mask.convert("L")
    if class_l.size != mask_l.size:
        raise ValueError("class_image and binary_mask must have the same size")
    values = [
        int(label) if int(mask_value) > 0 else int(background_index)
        for label, mask_value in zip(class_l.tobytes(), mask_l.tobytes())
    ]
    clipped = Image.new("L", class_l.size)
    clipped.putdata(values)
    return clipped


def predict_segmentation_image(
    image: Image.Image,
    *,
    binary_model: LoadedSegmentationModel,
    ore_model: LoadedSegmentationModel | None = None,
    binary_threshold: float = 0.5,
) -> SegmentationPrediction:
    """Predict binary ore and optional clipped multiclass ore masks for one image."""
    torch, _ = _require_torch()
    original_size = image.size
    with torch.no_grad():
        binary_input = _image_to_tensor(image, binary_model.metadata, device=binary_model.device)
        binary_logits = binary_model.model(binary_input)
        binary_prediction = binary_prediction_from_logits(binary_logits, threshold=binary_threshold)

        ore_mask = _single_channel_to_image(binary_prediction.mask, scale=1)
        ore_probability = _single_channel_to_image(binary_prediction.probability, scale=255)
        ore_confidence = _single_channel_to_image(binary_prediction.confidence, scale=255)

        if ore_mask.size != original_size:
            ore_mask = ore_mask.resize(original_size, Image.Resampling.NEAREST)
            ore_probability = ore_probability.resize(original_size, Image.Resampling.BILINEAR)
            ore_confidence = ore_confidence.resize(original_size, Image.Resampling.BILINEAR)

        multiclass_mask = None
        multiclass_confidence = None
        if ore_model is not None:
            ore_input = _image_to_tensor(image, ore_model.metadata, device=ore_model.device)
            ore_logits = ore_model.model(ore_input)
            ore_prediction = multiclass_prediction_from_logits(
                ore_logits,
                class_names=ore_model.metadata.class_names,
                background_index=ore_model.metadata.background_index or 0,
            )
            multiclass_mask = _class_index_to_image(ore_prediction.class_index)
            multiclass_confidence = _single_channel_to_image(ore_prediction.class_probability, scale=255)
            if multiclass_mask.size != original_size:
                multiclass_mask = multiclass_mask.resize(original_size, Image.Resampling.NEAREST)
                multiclass_confidence = multiclass_confidence.resize(original_size, Image.Resampling.BILINEAR)
            multiclass_mask = clip_class_image_to_binary(
                multiclass_mask,
                ore_mask,
                background_index=ore_model.metadata.background_index or 0,
            )

    metadata = {
        "method": "trained_binary_ore_segmentation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "binary_outline_is_primary": True,
            "multiclass_clipped_to_binary_mask": ore_model is not None,
            "talc_prediction_enabled": False,
            "talc_policy": "manual_ui_annotation_only",
        },
        "binary_threshold": binary_threshold,
        "binary_checkpoint": binary_model.metadata.as_dict(),
        "ore_checkpoint": ore_model.metadata.as_dict() if ore_model is not None else None,
    }
    return SegmentationPrediction(
        ore_mask=ore_mask,
        ore_probability=ore_probability,
        ore_confidence=ore_confidence,
        multiclass_mask=multiclass_mask,
        multiclass_confidence=multiclass_confidence,
        metadata=metadata,
    )


def _safe_sample_id(image_path: str | Path) -> str:
    path = Path(image_path)
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{path.stem}-{digest}"


def save_segmentation_prediction(
    image_path: str | Path,
    *,
    binary_model: LoadedSegmentationModel,
    ore_model: LoadedSegmentationModel | None = None,
    output_root: str | Path = "data_work/predictions/model_segmentation",
    binary_threshold: float = 0.5,
    sample_id: str | None = None,
) -> SegmentationPredictionArtifacts:
    """Run trained-model prediction and save disk artifacts for review/descriptors."""
    image_path = Path(image_path)
    output_root = Path(output_root)
    resolved_sample_id = sample_id or _safe_sample_id(image_path)
    sample_dir = output_root / resolved_sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    prediction = predict_segmentation_image(
        image,
        binary_model=binary_model,
        ore_model=ore_model,
        binary_threshold=binary_threshold,
    )
    prediction.metadata["sample_id"] = resolved_sample_id
    prediction.metadata["image_path"] = str(image_path)

    ore_mask_path = sample_dir / "ore_mask.png"
    ore_probability_path = sample_dir / "ore_probability.png"
    ore_confidence_path = sample_dir / "ore_confidence.png"
    overlay_path = sample_dir / "overlay.png"
    metadata_path = sample_dir / "metadata.json"
    multiclass_mask_path = sample_dir / "ore_multiclass_mask.png" if prediction.multiclass_mask is not None else None
    multiclass_confidence_path = (
        sample_dir / "ore_multiclass_confidence.png" if prediction.multiclass_confidence is not None else None
    )

    prediction.ore_mask.save(ore_mask_path)
    prediction.ore_probability.save(ore_probability_path)
    prediction.ore_confidence.save(ore_confidence_path)
    save_overlay(image, prediction.ore_mask, overlay_path)
    if prediction.multiclass_mask is not None and multiclass_mask_path is not None:
        prediction.multiclass_mask.save(multiclass_mask_path)
    if prediction.multiclass_confidence is not None and multiclass_confidence_path is not None:
        prediction.multiclass_confidence.save(multiclass_confidence_path)

    prediction.metadata["artifacts"] = {
        "ore_mask": str(ore_mask_path),
        "ore_probability": str(ore_probability_path),
        "ore_confidence": str(ore_confidence_path),
        "overlay": str(overlay_path),
        "ore_multiclass_mask": str(multiclass_mask_path) if multiclass_mask_path is not None else None,
        "ore_multiclass_confidence": str(multiclass_confidence_path) if multiclass_confidence_path is not None else None,
    }
    metadata_path.write_text(json.dumps(_jsonable(prediction.metadata), indent=2), encoding="utf-8")

    return SegmentationPredictionArtifacts(
        sample_dir=sample_dir,
        ore_mask_path=ore_mask_path,
        ore_probability_path=ore_probability_path,
        ore_confidence_path=ore_confidence_path,
        overlay_path=overlay_path,
        metadata_path=metadata_path,
        multiclass_mask_path=multiclass_mask_path,
        multiclass_confidence_path=multiclass_confidence_path,
    )
