"""Audio processing, decoding, validation, and standardization package."""

from app.audio.decoder import AudioDecoder
from app.audio.normalizer import AudioNormalizer
from app.audio.processor import AudioProcessor
from app.audio.schemas import (
    AudioData,
    AudioQualityMetrics,
    AudioMetadataSchema,
    AudioQualitySchema,
    AudioInspectResponse,
)
from app.audio.validator import (
    AudioValidator,
    AudioValidationError,
    AudioEmptyError,
    AudioFormatError,
    AudioSizeError,
    AudioDurationError,
    AudioSignalError,
)

__all__ = [
    "AudioDecoder",
    "AudioNormalizer",
    "AudioProcessor",
    "AudioData",
    "AudioQualityMetrics",
    "AudioMetadataSchema",
    "AudioQualitySchema",
    "AudioInspectResponse",
    "AudioValidator",
    "AudioValidationError",
    "AudioEmptyError",
    "AudioFormatError",
    "AudioSizeError",
    "AudioDurationError",
    "AudioSignalError",
]
