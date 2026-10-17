"""Audio service for coordinating audio ingestion and inspection."""

from typing import Optional
from app.audio.processor import AudioProcessor
from app.audio.schemas import (
    AudioInspectResponse,
    AudioMetadataSchema,
    AudioQualitySchema,
    AudioData,
    AudioQualityMetrics,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


class AudioService:
    """Service layer coordinating audio decoding, preprocessing, and inspection."""

    @classmethod
    def inspect_audio(cls, file_bytes: bytes, filename: Optional[str] = None) -> AudioInspectResponse:
        """
        Inspect uploaded or recorded audio file and return standardization and quality metadata.

        Args:
            file_bytes: Raw binary content of the audio file.
            filename: Optional original filename for diagnostic logging.

        Returns:
            AudioInspectResponse containing audio metadata and quality metrics.
        """
        logger.debug("Received audio inspection request for: %s (size=%d bytes)", filename or "stream", len(file_bytes))

        audio_data, quality_metrics = AudioProcessor.process(file_bytes)

        return AudioInspectResponse(
            success=True,
            audio=AudioMetadataSchema(
                duration_seconds=audio_data.duration_seconds,
                sample_rate=audio_data.sample_rate,
                channels=audio_data.channels,
                original_format=audio_data.original_format,
                original_sample_rate=audio_data.original_sample_rate,
                original_channels=audio_data.original_channels,
            ),
            quality=AudioQualitySchema(
                rms_db=quality_metrics.rms_db,
                peak_db=quality_metrics.peak_db,
                clipping_ratio=quality_metrics.clipping_ratio,
                silence_ratio=quality_metrics.silence_ratio,
                quality=quality_metrics.quality,
                notes=quality_metrics.notes,
            ),
        )

    @classmethod
    def get_processed_audio_data(cls, file_bytes: bytes) -> AudioData:
        """Internal helper for future detection modules to retrieve the standardized AudioData."""
        audio_data, _ = AudioProcessor.process(file_bytes)
        return audio_data
