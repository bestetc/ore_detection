"""Store UI-readable prediction and correction artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from ore_detection.segmentation.hsv_dummy import HsvDummyConfig, hsv_value_binary_mask, hsv_value_confidence
from ore_detection.visualization.overlay import save_overlay


@dataclass(frozen=True)
class PredictionArtifacts:
    sample_dir: Path
    ore_mask_path: Path
    ore_confidence_path: Path
    overlay_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class CorrectionArtifacts:
    correction_dir: Path
    mask_path: Path
    metadata_path: Path


def safe_sample_id(image_path: str | Path) -> str:
    """Return a stable filesystem-safe sample id for an image path."""
    path = Path(image_path)
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{path.stem}-{digest}"


def save_hsv_dummy_prediction(
    image_path: str | Path,
    *,
    output_root: str | Path = "data_work/predictions/hsv_dummy",
    config: HsvDummyConfig | None = None,
    sample_id: str | None = None,
) -> PredictionArtifacts:
    """Generate HSV dummy segmentation and save UI-ready artifacts."""
    image_path = Path(image_path)
    output_root = Path(output_root)
    cfg = config or HsvDummyConfig()
    resolved_sample_id = sample_id or safe_sample_id(image_path)
    sample_dir = output_root / resolved_sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    mask = hsv_value_binary_mask(image, cfg)
    confidence = hsv_value_confidence(image, cfg)

    ore_mask_path = sample_dir / "ore_mask.png"
    ore_confidence_path = sample_dir / "ore_confidence.png"
    overlay_path = sample_dir / "overlay.png"
    metadata_path = sample_dir / "metadata.json"

    mask.save(ore_mask_path)
    confidence.save(ore_confidence_path)
    save_overlay(image, mask, overlay_path)
    metadata_path.write_text(
        json.dumps(
            {
                "method": "hsv_value_dummy",
                "sample_id": resolved_sample_id,
                "image_path": str(image_path),
                "config": cfg.to_dict(),
                "artifacts": {
                    "ore_mask": str(ore_mask_path),
                    "ore_confidence": str(ore_confidence_path),
                    "overlay": str(overlay_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return PredictionArtifacts(
        sample_dir=sample_dir,
        ore_mask_path=ore_mask_path,
        ore_confidence_path=ore_confidence_path,
        overlay_path=overlay_path,
        metadata_path=metadata_path,
    )


def accept_prediction_as_correction(sample_dir: str | Path, *, label: str = "ore") -> CorrectionArtifacts:
    """Persist the current dummy prediction mask as an accepted correction.

    This is the first active-learning persistence path: it lets a reviewer mark
    the deterministic dummy mask as accepted so later stages can distinguish
    reviewed masks from raw HSV outputs.
    """
    sample_dir = Path(sample_dir)
    source_mask = sample_dir / "ore_mask.png"
    source_metadata = sample_dir / "metadata.json"
    if not source_mask.exists():
        raise FileNotFoundError(f"missing prediction mask: {source_mask}")

    correction_dir = sample_dir / "corrections"
    correction_dir.mkdir(parents=True, exist_ok=True)
    mask_path = correction_dir / f"{label}_mask.png"
    metadata_path = correction_dir / "correction_metadata.json"
    shutil.copy2(source_mask, mask_path)

    source = json.loads(source_metadata.read_text(encoding="utf-8")) if source_metadata.exists() else {}
    metadata_path.write_text(
        json.dumps(
            {
                "label": label,
                "status": "accepted_dummy_prediction",
                "source_prediction": str(sample_dir),
                "source_metadata": source,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return CorrectionArtifacts(correction_dir=correction_dir, mask_path=mask_path, metadata_path=metadata_path)
