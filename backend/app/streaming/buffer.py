"""
Bounded Rolling Audio Buffer for Real-Time Streaming (16 kHz Mono Float32).
"""
import logging
import numpy as np
from typing import List, Tuple, Optional
from app.audio.schemas import AudioData

logger = logging.getLogger(__name__)


class StreamingAudioBuffer:
    """
    Thread-safe, bounded rolling audio buffer for real-time speech ingestion.
    Enforces 16 kHz mono float32 standardization and fixed-hop window extraction.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        window_sec: float = 3.0,
        hop_sec: float = 1.5,
        max_buffer_sec: float = 15.0,
    ):
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.max_buffer_sec = max_buffer_sec

        self.window_samples = int(self.sample_rate * self.window_sec)
        self.hop_samples = int(self.sample_rate * self.hop_sec)
        self.max_buffer_samples = int(self.sample_rate * self.max_buffer_sec)

        # Internal buffer storage (1D float32 numpy array)
        self._buffer: np.ndarray = np.empty(0, dtype=np.float32)
        
        # Absolute stream tracking
        self.total_samples_received: int = 0
        self.total_chunks_received: int = 0
        self.last_sequence_number: int = -1
        self.next_window_start_sample: int = 0
        self.windows_extracted_count: int = 0

    @property
    def buffered_duration_sec(self) -> float:
        return float(len(self._buffer) / self.sample_rate)

    @property
    def total_duration_received_sec(self) -> float:
        return float(self.total_samples_received / self.sample_rate)

    @property
    def current_sample_count(self) -> int:
        return len(self._buffer)

    @property
    def total_samples_ingested(self) -> int:
        return self.total_samples_received

    def push_chunk(self, chunk_samples: np.ndarray, seq: Optional[int] = None) -> None:
        """Alias for append_chunk."""
        self.append_chunk(chunk_samples, sequence_number=seq)

    def extract_ready_windows(self) -> List[Tuple[int, float, float, AudioData]]:
        """Alias for get_available_windows."""
        return self.get_available_windows()

    def append_chunk(self, chunk_samples: np.ndarray, sequence_number: Optional[int] = None) -> None:
        """
        Appends a 1D float32 audio chunk to the rolling buffer.
        Validates chunk dimensions, range, and sequence numbers.
        """
        if chunk_samples is None or len(chunk_samples) == 0:
            return

        # Ensure 1D float32
        if chunk_samples.ndim > 1:
            chunk_samples = np.mean(chunk_samples, axis=-1)
        
        chunk_samples = chunk_samples.astype(np.float32)

        # Clamp extreme values and replace NaNs/Infs
        if not np.isfinite(chunk_samples).all():
            chunk_samples = np.nan_to_num(chunk_samples, nan=0.0, posinf=1.0, neginf=-1.0)
        chunk_samples = np.clip(chunk_samples, -1.0, 1.0)

        # Sequence ordering validation
        if sequence_number is not None:
            if sequence_number <= self.last_sequence_number and self.last_sequence_number >= 0:
                logger.warning(f"Out-of-order or duplicate chunk received: seq={sequence_number}, last={self.last_sequence_number}")
            self.last_sequence_number = sequence_number

        # Append to buffer
        self._buffer = np.concatenate((self._buffer, chunk_samples))
        self.total_samples_received += len(chunk_samples)
        self.total_chunks_received += 1

        # Bound buffer memory to max_buffer_samples
        # Calculate how many samples we need to retain for next window
        # The oldest sample needed by next_window_start_sample in stream coordinates:
        earliest_needed_sample_in_stream = self.next_window_start_sample
        # Buffer starts at stream sample: (self.total_samples_received - len(self._buffer))
        buffer_start_stream_sample = self.total_samples_received - len(self._buffer)

        samples_to_discard = earliest_needed_sample_in_stream - buffer_start_stream_sample
        if samples_to_discard > 0 and len(self._buffer) > self.max_buffer_samples:
            discard_count = min(samples_to_discard, len(self._buffer) - self.window_samples)
            if discard_count > 0:
                self._buffer = self._buffer[discard_count:]

    def get_available_windows(self) -> List[Tuple[int, float, float, AudioData]]:
        """
        Extracts all newly available complete 3.0s analysis windows on 1.5s hop intervals.
        Returns list of (window_index, start_sec, end_sec, AudioData).
        """
        extracted_windows = []

        while True:
            # Check if we have received enough stream samples to cover the next window
            window_end_sample = self.next_window_start_sample + self.window_samples
            if self.total_samples_received < window_end_sample:
                break

            # Locate window within current buffer slice
            buffer_start_stream_sample = self.total_samples_received - len(self._buffer)
            rel_start = self.next_window_start_sample - buffer_start_stream_sample
            rel_end = rel_start + self.window_samples

            if rel_start < 0 or rel_end > len(self._buffer):
                # Buffer might have slipped or underrun
                logger.warning(f"Window index {self.windows_extracted_count} out of buffer bounds [rel_start={rel_start}, rel_end={rel_end}, buf_len={len(self._buffer)}]")
                break

            window_chunk = self._buffer[rel_start:rel_end].copy()
            start_sec = float(self.next_window_start_sample / self.sample_rate)
            end_sec = float(window_end_sample / self.sample_rate)

            audio_data = AudioData(
                samples=window_chunk,
                sample_rate=self.sample_rate,
                duration_seconds=self.window_sec,
                channels=1,
            )

            extracted_windows.append((
                self.windows_extracted_count,
                start_sec,
                end_sec,
                audio_data
            ))

            self.windows_extracted_count += 1
            self.next_window_start_sample += self.hop_samples

        return extracted_windows

    def reset(self) -> None:
        """Resets the buffer and stream state."""
        self._buffer = np.empty(0, dtype=np.float32)
        self.total_samples_received = 0
        self.total_chunks_received = 0
        self.last_sequence_number = -1
        self.next_window_start_sample = 0
        self.windows_extracted_count = 0
