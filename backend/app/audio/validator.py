"""Audio validation and error checking logic."""

import numpy as np
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class AudioValidationError(Exception):
    """Base exception for audio validation errors."""

    def __init__(self, message: str, code: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code


class AudioEmptyError(AudioValidationError):
    """Raised when uploaded audio data is empty."""

    def __init__(self, message: str = "Uploaded audio file is empty."):
        super().__init__(message, code=400)


class AudioSizeError(AudioValidationError):
    """Raised when audio file exceeds size limit."""

    def __init__(self, message: str = f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB."):
        super().__init__(message, code=413)


class AudioFormatError(AudioValidationError):
    """Raised when audio format or container is unsupported or corrupted."""

    def __init__(self, message: str = "Unsupported or corrupted audio format."):
        super().__init__(message, code=415)


class AudioDurationError(AudioValidationError):
    """Raised when audio duration falls outside acceptable limits."""

    def __init__(self, message: str = "Audio duration outside allowed limits."):
        super().__init__(message, code=400)


class AudioSignalError(AudioValidationError):
    """Raised when audio samples contain invalid numbers (NaN, Inf) or unusable signal."""

    def __init__(self, message: str = "Invalid numeric samples (NaN/Inf) detected in audio stream."):
        super().__init__(message, code=400)


class AudioValidator:
    """Validator ensuring decoded audio complies with security and model constraints."""

    @staticmethod
    def validate_raw_bytes(data: bytes) -> None:
        """Validate raw audio bytes before decoding."""
        if not data or len(data) == 0:
            raise AudioEmptyError("Audio payload is empty (0 bytes).")

        if len(data) > settings.max_file_size_bytes:
            raise AudioSizeError(
                f"Audio file size ({round(len(data)/(1024*1024), 2)}MB) exceeds limit of {settings.MAX_FILE_SIZE_MB}MB."
            )

    @staticmethod
    def validate_samples(
        samples: np.ndarray,
        sample_rate: int = 16000,
        min_duration: float = settings.MIN_DURATION_SECONDS,
        max_duration: float = settings.MAX_DURATION_SECONDS,
    ) -> None:
        """Validate decoded NumPy samples for duration and numeric integrity."""
        if samples is None or len(samples) == 0:
            raise AudioEmptyError("Decoded audio stream contains 0 samples.")

        # Check for NaN or Inf
        if not np.all(np.isfinite(samples)):
            raise AudioSignalError("Decoded audio contains NaN or infinite values.")

        duration = len(samples) / sample_rate
        if duration < min_duration:
            raise AudioDurationError(
                f"Audio duration ({duration:.2f}s) is too short. Minimum duration is {min_duration:.1f}s."
            )

        if duration > max_duration:
            raise AudioDurationError(
                f"Audio duration ({duration:.2f}s) exceeds maximum allowed duration of {max_duration:.1f}s."
            )
