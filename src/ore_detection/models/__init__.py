"""Model definitions for ore segmentation."""

from ore_detection.models.cs_unet import create_cs_unet
from ore_detection.models.simple_unet import create_simple_unet

__all__ = ["create_cs_unet", "create_simple_unet"]
