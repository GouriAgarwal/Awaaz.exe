"""Temporal windowing and preprocessing for synthetic voice detection models."""

import math
from typing import List, Tuple
import numpy as np

from app.audio.schemas import AudioData
from app.audio.validator import AudioSignalError
from app.core.config import settings


class AudioWindowPreprocessor:
    """Chunks 16 kHz mono float32 AudioData into fixed temporal windows with overlap."""

    def __init__(
        self,
        window_size_sec: float = settings.DETECTION_WINDOW_SECONDS,
        window_hop_sec: float = settings.DETECTION_HOP_SECONDS,
        sample_rate: int = settings.TARGET_SAMPLE_RATE,
    ):
        if window_size_sec <= 0:
            raise ValueError(f"window_size_sec must be positive, got {window_size_sec}")
        if window_hop_sec <= 0:
            raise ValueError(f"window_hop_sec must be positive, got {window_hop_sec}")
        if window_hop_sec > window_size_sec:
            raise ValueError(f"window_hop_sec ({window_hop_sec}) cannot exceed window_size_sec ({window_size_sec})")

        self.window_size_sec = float(window_size_sec)
        self.window_hop_sec = float(window_hop_sec)
        self.sample_rate = int(sample_rate)

        self.window_samples = int(round(self.window_size_sec * self.sample_rate))
        self.hop_samples = int(round(self.window_hop_sec * self.sample_rate))

    def process(self, audio_data: AudioData) -> List[Tuple[int, float, float, np.ndarray]]:
        """Extract sliding temporal windows from AudioData.
        
        Args:
            audio_data: Validated 16 kHz mono float32 AudioData.
            
        Returns:
            List of tuples: (window_index, start_time_sec, end_time_sec, window_samples_array)
        """
        samples = audio_data.samples
        if samples is None or len(samples) == 0:
            raise AudioSignalError("Cannot extract detection windows from empty audio data.")

        # Sanity check for non-finite values
        if not np.all(np.isfinite(samples)):
            raise AudioSignalError("Audio data contains NaN or infinite values.")

        num_samples = len(samples)
        duration_sec = num_samples / self.sample_rate

        # Case 1: Audio shorter than window size -> pad to window size
        if num_samples < self.window_samples:
            pad_amount = self.window_samples - num_samples
            # Use reflect padding if possible (at least 2 samples), else zero pad
            if num_samples > 1:
                padded = np.pad(samples, (0, pad_amount), mode="reflect")
            else:
                padded = np.pad(samples, (0, pad_amount), mode="constant", constant_values=0.0)

            return [(0, 0.0, duration_sec, padded.astype(np.float32))]

        # Case 2: Audio >= window size -> sliding window extraction
        windows: List[Tuple[int, float, float, np.ndarray]] = []
        start_idx = 0
        window_idx = 0

        while start_idx + self.window_samples <= num_samples:
            end_idx = start_idx + self.window_samples
            chunk = samples[start_idx:end_idx].copy().astype(np.float32)
            start_sec = start_idx / self.sample_rate
            end_sec = end_idx / self.sample_rate
            windows.append((window_idx, start_sec, end_sec, chunk))
            
            start_idx += self.hop_samples
            window_idx += 1

        # Handle remaining tail if it represents a meaningful duration (e.g. >= 0.5s) and wasn't fully covered
        if start_idx < num_samples:
            remaining_samples = num_samples - start_idx
            # If the remaining segment has not been covered by the last window
            last_end = windows[-1][2] * self.sample_rate if windows else 0
            if num_samples > last_end:
                tail_chunk = samples[-self.window_samples:].copy().astype(np.float32)
                start_sec = max(0.0, (num_samples - self.window_samples) / self.sample_rate)
                end_sec = duration_sec
                windows.append((window_idx, start_sec, end_sec, tail_chunk))

        return windows
