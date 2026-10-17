"""Abstract Base Class for Synthetic Voice Detectors."""

from abc import ABC, abstractmethod
from app.audio.schemas import AudioData
from app.detection.schemas import DetectionResult, DetectorMetadata


class BaseSyntheticVoiceDetector(ABC):
    """Abstract interface that all voice clone / synthetic detectors must implement."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights and initialize compute graphs."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if model is loaded and ready for inference, False otherwise."""
        pass

    @abstractmethod
    def predict(self, audio_data: AudioData) -> DetectionResult:
        """Run synthetic voice detection on standardized 16 kHz mono AudioData.
        
        Args:
            audio_data: Validated 16 kHz mono float32 audio container.
            
        Returns:
            DetectionResult with classification, scores, window breakdowns, and metadata.
        """
        pass

    @abstractmethod
    def metadata(self) -> DetectorMetadata:
        """Return architectural and provenance metadata for this detector."""
        pass
