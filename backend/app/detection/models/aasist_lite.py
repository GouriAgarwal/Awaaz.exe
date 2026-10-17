"""AASIST-Lite Raw Waveform Anti-Spoofing Architecture."""

import math
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
    class SincConv(nn.Module):
        """Learnable Sinc-based bandpass filter frontend operating on raw waveforms."""

        def __init__(self, out_channels: int = 70, kernel_size: int = 129, sample_rate: int = 16000):
            super().__init__()
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.sample_rate = sample_rate

            # Linear initialization of bandpass frequencies
            f_min = 30.0
            f_max = sample_rate / 2.0 - 100.0
            hz_points = np.linspace(f_min, f_max, out_channels + 1)
            
            # Learnable filter parameters (f1: low edge, f2: band width)
            self.f1 = nn.Parameter(torch.tensor(hz_points[:-1], dtype=torch.float32).view(-1, 1))
            self.band = nn.Parameter(torch.tensor(np.diff(hz_points), dtype=torch.float32).view(-1, 1))

            # Fixed time grid
            t = np.linspace(0, (kernel_size - 1) / sample_rate, kernel_size)
            t = 2 * np.pi * (t - np.median(t))
            self.register_buffer("t", torch.tensor(t, dtype=torch.float32).view(1, -1))
            self.register_buffer("window", torch.hamming_window(kernel_size).view(1, -1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (batch, 1, time_samples) -> (batch, out_channels, time_samples)"""
            f1 = torch.abs(self.f1)
            f2 = f1 + torch.abs(self.band)

            # Bandpass sinc filter calculation
            low = 2 * f1 / self.sample_rate
            high = 2 * f2 / self.sample_rate

            # Sinc bandpass kernel
            f1_t = 2 * np.pi * f1 * self.t / (2 * np.pi)
            f2_t = 2 * np.pi * f2 * self.t / (2 * np.pi)

            # Safe sinc: sin(x)/x
            sinc_high = torch.sin(f2_t) / (self.t + 1e-8)
            sinc_low = torch.sin(f1_t) / (self.t + 1e-8)
            bandpass = (sinc_high - sinc_low) * self.window
            bandpass = bandpass / (torch.max(torch.abs(bandpass), dim=1, keepdim=True)[0] + 1e-8)

            filters = bandpass.view(self.out_channels, 1, self.kernel_size)
            return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)


    class AASISTLite(nn.Module):
        """AASIST-Lite graph / residual architecture for 16 kHz raw waveform voice clone detection."""

        def __init__(self, sample_rate: int = 16000):
            super().__init__()
            self.frontend = SincConv(out_channels=32, kernel_size=129, sample_rate=sample_rate)
            self.bn0 = nn.BatchNorm1d(32)
            self.relu = nn.LeakyReLU(0.2)

            self.conv1 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
            self.bn1 = nn.BatchNorm1d(64)
            self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

            self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
            self.bn2 = nn.BatchNorm1d(128)
            self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

            self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)
            self.bn3 = nn.BatchNorm1d(128)

            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(128, 64)
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(64, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Input x: (batch, 1, n_samples) -> logit: (batch, 1)"""
            feat = self.relu(self.bn0(torch.abs(self.frontend(x))))
            
            feat = self.maxpool1(self.relu(self.bn1(self.conv1(feat))))
            feat = self.maxpool2(self.relu(self.bn2(self.conv2(feat))))
            feat = self.relu(self.bn3(self.conv3(feat)))

            pooled = self.global_pool(feat).squeeze(-1)
            dense = self.relu(self.fc1(pooled))
            dense = self.dropout(dense)
            logits = self.fc2(dense)
            return logits
else:
    class AASISTLite:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to instantiate AASISTLite.")
