"""Spectral CNN Architecture for Anti-Spoofing and Voice Clone Detection."""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # type: ignore


if TORCH_AVAILABLE:
    class ResidualBlock(nn.Module):
        """2D Residual Convolutional block with batch norm and LeakyReLU."""

        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.LeakyReLU(0.2, inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = self.shortcut(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            out = self.relu(out)
            return out


    class SpectralResNet(nn.Module):
        """Lightweight ResNet model tailored for Spectral Voice Clone Detection."""

        def __init__(self, in_channels: int = 1, num_classes: int = 1):
            super().__init__()
            self.in_channels = 32
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu = nn.LeakyReLU(0.2, inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.layer1 = self._make_layer(32, 2, stride=1)
            self.layer2 = self._make_layer(64, 2, stride=2)
            self.layer3 = self._make_layer(128, 2, stride=2)

            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(128, 64)
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(64, num_classes)

        def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
            layers = [ResidualBlock(self.in_channels, out_channels, stride)]
            self.in_channels = out_channels
            for _ in range(1, blocks):
                layers.append(ResidualBlock(out_channels, out_channels))
            return nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Input shape: (batch_size, 1, n_mels, time_steps) -> Output logit shape: (batch_size, 1)"""
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.global_pool(x)
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            logits = self.fc2(x)
            return logits
else:
    class SpectralResNet:  # type: ignore
        """Stub when PyTorch is not installed."""
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to instantiate SpectralResNet.")
