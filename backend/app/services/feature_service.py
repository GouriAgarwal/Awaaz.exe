"""Feature extraction service coordinating audio ingestion with feature analysis."""

import time
from typing import Optional
from app.audio.processor import AudioProcessor
from app.features.extractor import FeatureExtractor
from app.features.schemas import FeatureExtractionResponse, FEATURE_EXPLAINABILITY_METADATA
from app.core.logging import get_logger

logger = get_logger(__name__)


class FeatureService:
    """Service coordinating end-to-end feature extraction from raw audio payloads."""

    @classmethod
    def extract_features(cls, file_bytes: bytes, filename: Optional[str] = None) -> FeatureExtractionResponse:
        """
        Ingest, standardize, and extract comprehensive acoustic features from audio payload.

        Args:
            file_bytes: Raw binary content of uploaded audio file or microphone recording.
            filename: Optional filename for logging.

        Returns:
            FeatureExtractionResponse containing acoustic features, latency, and explainability metadata.
        """
        start_time = time.perf_counter()

        # Step 1: Decode and standardize audio (reusing AudioProcessor)
        audio_data, _ = AudioProcessor.process(file_bytes)

        # Step 2: Extract acoustic, spectral, harmonic, and prosodic features
        features = FeatureExtractor.extract_from_audio_data(audio_data)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        logger.info(
            "Extracted features for %s in %.2f ms (duration=%.2fs)",
            filename or "audio",
            elapsed_ms,
            audio_data.duration_seconds,
        )

        return FeatureExtractionResponse(
            success=True,
            processing_time_ms=elapsed_ms,
            features=features,
            explainability=FEATURE_EXPLAINABILITY_METADATA,
        )
