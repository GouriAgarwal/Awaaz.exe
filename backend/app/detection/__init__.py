"""VoiceShield Synthetic / Cloned Voice Detection Subsystem (Step 4)."""

from app.detection.base import BaseSyntheticVoiceDetector
from app.detection.calibration import ScoreCalibrator
from app.detection.fallback import FallbackSyntheticDetector
from app.detection.inference import DeepLearningSyntheticDetector
from app.detection.metadata import (
    get_deep_learning_metadata,
    get_fallback_metadata,
    get_spectral_cnn_metadata,
)
from app.detection.preprocessing import AudioWindowPreprocessor
from app.detection.registry import DetectorRegistry, detector_registry
from app.detection.schemas import (
    ClassificationLabel,
    DetectionResponse,
    DetectionResult,
    DetectorMetadata,
    ScoreType,
    WindowScore,
)

__all__ = [
    "BaseSyntheticVoiceDetector",
    "ScoreCalibrator",
    "FallbackSyntheticDetector",
    "DeepLearningSyntheticDetector",
    "get_deep_learning_metadata",
    "get_fallback_metadata",
    "get_spectral_cnn_metadata",
    "AudioWindowPreprocessor",
    "DetectorRegistry",
    "detector_registry",
    "ClassificationLabel",
    "DetectionResponse",
    "DetectionResult",
    "DetectorMetadata",
    "ScoreType",
    "WindowScore",
]
