"""Detection service coordinating audio ingestion, preprocessing, and synthetic voice inference."""

from typing import List, Optional
from app.audio.processor import AudioProcessor
from app.audio.schemas import AudioData, AudioQualityMetrics
from app.core.logging import get_logger
from app.detection.base import BaseSyntheticVoiceDetector
from app.detection.registry import detector_registry
from app.detection.schemas import DetectionResult, DetectorMetadata

logger = get_logger("services.detection")


class DetectionService:
    """High-level service coordinating synthetic voice detection pipeline."""

    def __init__(self, audio_processor: Optional[AudioProcessor] = None):
        self.audio_processor = audio_processor or AudioProcessor()

    def analyze_audio_data(
        self,
        audio_data: AudioData,
        detector_name: Optional[str] = None,
    ) -> DetectionResult:
        """Run synthetic voice detection directly on preprocessed AudioData.
        
        Args:
            audio_data: Standardized 16 kHz mono float32 AudioData container.
            detector_name: Optional explicit detector backend to use.
            
        Returns:
            DetectionResult with classification, scores, and window breakdown.
        """
        detector: BaseSyntheticVoiceDetector = detector_registry.get_detector(detector_name)
        logger.info(f"Running synthetic voice detection using '{detector.metadata().model_name}'")
        return detector.predict(audio_data)

    def analyze_raw_audio(
        self,
        file_bytes: bytes,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        detector_name: Optional[str] = None,
    ) -> tuple[DetectionResult, AudioQualityMetrics]:
        """Process raw uploaded audio bytes through the ingestion pipeline and run detection.
        
        Args:
            file_bytes: Raw audio binary data.
            filename: Optional original filename.
            content_type: Optional MIME content type.
            detector_name: Optional detector backend identifier.
            
        Returns:
            Tuple of (DetectionResult, AudioQualityMetrics).
        """
        audio_data, quality_metrics = self.audio_processor.process(file_bytes=file_bytes)
        result = self.analyze_audio_data(audio_data=audio_data, detector_name=detector_name)
        return result, quality_metrics

    def get_active_metadata(self) -> DetectorMetadata:
        """Retrieve metadata of the currently active detector."""
        return detector_registry.get_detector().metadata()

    def list_available_detectors(self) -> List[DetectorMetadata]:
        """List metadata of all registered detectors."""
        return detector_registry.list_detectors()


detection_service = DetectionService()
