"""
Base interface for Speaker Verification Encoders.
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseSpeakerEncoder(ABC):
    """
    Abstract base class for speaker embedding encoders.
    """
    model_id: str
    model_name: str
    embedding_dimension: int
    expected_sample_rate: int = 16000
    is_l2_normalized: bool = True

    @abstractmethod
    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract a normalized 1D speaker embedding vector from raw 1D float32 audio.

        Args:
            audio: 1D float32 numpy array normalized to [-1.0, 1.0].
            sample_rate: Sample rate in Hz (default 16000).

        Returns:
            1D float32 numpy array of shape (embedding_dimension,), L2-normalized.
        """
        pass
