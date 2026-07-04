"""Small baseline U-Net for ore segmentation.

This is intentionally simple: enough to overfit a small source subset and prove
that the data pipeline is wired correctly before switching to a stronger model.
"""

from __future__ import annotations


def _require_torch_modules():
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for the segmentation model. Install the `ml` optional dependencies.") from exc
    return torch, nn


def create_simple_unet(*, in_channels: int = 3, out_channels: int = 1, base_channels: int = 16):
    """Create a compact U-Net-like model for binary or multiclass logits."""
    torch, nn = _require_torch_modules()

    def block(cin: int, cout: int):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    class SimpleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = block(in_channels, base_channels)
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = block(base_channels, base_channels * 2)
            self.pool2 = nn.MaxPool2d(2)
            self.bottleneck = block(base_channels * 2, base_channels * 4)
            self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
            self.dec2 = block(base_channels * 4, base_channels * 2)
            self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
            self.dec1 = block(base_channels * 2, base_channels)
            self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            b = self.bottleneck(self.pool2(e2))
            d2 = self.up2(b)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            return self.head(d1)

    return SimpleUNet()
