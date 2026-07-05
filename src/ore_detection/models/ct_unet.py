"""CT-UNet segmentation model factory.

The project-local CT-UNet uses the existing convolution-transformer U-Net
implementation. This module gives training notebooks a stable CTUnet entry
point without duplicating the architecture code.
"""

from __future__ import annotations

from ore_detection.models.cs_unet import create_cs_unet


def create_ct_unet(
    *,
    in_channels: int = 3,
    out_channels: int = 1,
    base_channels: int = 32,
    num_heads: int = 4,
    transformer_layers: int = 2,
    token_grid_size: int = 16,
):
    """Create a compact convolution-transformer U-Net segmentation model."""
    return create_cs_unet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        num_heads=num_heads,
        transformer_layers=transformer_layers,
        token_grid_size=token_grid_size,
    )
