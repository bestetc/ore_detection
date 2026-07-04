"""CS-UNet-style segmentation model.

This is a compact project-local implementation inspired by hybrid
CNN-transformer U-Net designs. It keeps CNN blocks for local ore boundaries and
adds a transformer context block at the bottleneck for wider field context.
"""

from __future__ import annotations


def _require_torch_modules():
    try:
        import torch
        import torch.nn.functional as F
        from torch import nn
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for CS-UNet. Install the `ml` optional dependencies.") from exc
    return torch, F, nn


def create_cs_unet(
    *,
    in_channels: int = 3,
    out_channels: int = 1,
    base_channels: int = 32,
    num_heads: int = 4,
    transformer_layers: int = 2,
    token_grid_size: int = 16,
):
    """Create a CS-UNet-style model returning logits shaped ``[B,C,H,W]``.

    ``token_grid_size`` bounds transformer memory by applying attention to a
    pooled bottleneck grid, then upsampling the context back to the bottleneck
    feature map.
    """
    torch, F, nn = _require_torch_modules()
    if in_channels < 1:
        raise ValueError(f"in_channels must be positive, got {in_channels}")
    if out_channels < 1:
        raise ValueError(f"out_channels must be positive, got {out_channels}")
    if base_channels < 1:
        raise ValueError(f"base_channels must be positive, got {base_channels}")
    if num_heads < 1:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if transformer_layers < 0:
        raise ValueError(f"transformer_layers must be non-negative, got {transformer_layers}")
    if token_grid_size < 1:
        raise ValueError(f"token_grid_size must be positive, got {token_grid_size}")

    bottleneck_channels = base_channels * 8
    if bottleneck_channels % num_heads != 0:
        raise ValueError(
            f"base_channels * 8 must be divisible by num_heads, got {bottleneck_channels} and {num_heads}"
        )

    def conv_block(cin: int, cout: int):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class BottleneckTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            if transformer_layers == 0:
                self.encoder = nn.Identity()
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=bottleneck_channels,
                    nhead=num_heads,
                    dim_feedforward=bottleneck_channels * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
            self.norm = nn.LayerNorm(bottleneck_channels)

        def forward(self, x):
            height, width = x.shape[-2:]
            pooled = F.adaptive_avg_pool2d(x, (token_grid_size, token_grid_size))
            tokens = pooled.flatten(2).transpose(1, 2)
            tokens = self.norm(tokens)
            tokens = self.encoder(tokens)
            context = tokens.transpose(1, 2).reshape(
                x.shape[0],
                bottleneck_channels,
                token_grid_size,
                token_grid_size,
            )
            context = F.interpolate(context, size=(height, width), mode="bilinear", align_corners=False)
            return x + context

    class CSUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = conv_block(in_channels, base_channels)
            self.enc2 = conv_block(base_channels, base_channels * 2)
            self.enc3 = conv_block(base_channels * 2, base_channels * 4)
            self.bottleneck = conv_block(base_channels * 4, bottleneck_channels)
            self.context = BottleneckTransformer()
            self.dec3 = conv_block(bottleneck_channels + base_channels * 4, base_channels * 4)
            self.dec2 = conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
            self.dec1 = conv_block(base_channels * 2 + base_channels, base_channels)
            self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(F.max_pool2d(e1, kernel_size=2))
            e3 = self.enc3(F.max_pool2d(e2, kernel_size=2))
            bottleneck = self.bottleneck(F.max_pool2d(e3, kernel_size=2))
            bottleneck = self.context(bottleneck)

            d3 = F.interpolate(bottleneck, size=e3.shape[-2:], mode="bilinear", align_corners=False)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            return self.head(d1)

    return CSUNet()
