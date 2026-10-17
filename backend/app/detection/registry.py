"""Registry and factory for synthetic voice detectors."""

from typing import Dict, List, Optional
from app.core.logging import get_logger
from app.detection.base import BaseSyntheticVoiceDetector
from app.detection.fallback import FallbackSyntheticDetector
from app.detection.inference import DeepLearningSyntheticDetector
from app.detection.pretrained.adapter import AASISTPretrainedAdapter
from app.detection.pretrained.loader import PretrainedModelLoader
from app.detection.schemas import DetectorMetadata

logger = get_logger("detection.registry")


class DetectorRegistry:
    """Registry maintaining active and registered synthetic voice detectors."""

    _instance: Optional["DetectorRegistry"] = None

    def __new__(cls) -> "DetectorRegistry":
        if cls._instance is None:
            cls._instance = super(DetectorRegistry, cls).__new__(cls)
            cls._instance._detectors = {}
            cls._instance._active_name = "aasist_pretrained"
            cls._instance._initialize_default_detectors()
        return cls._instance

    def _initialize_default_detectors(self) -> None:
        """Initialize registered pretrained models and reference baselines."""
        # 1. Official Genuine Pretrained Model
        is_valid, _, _ = PretrainedModelLoader.verify_checkpoint_integrity()
        pretrained_adapter = AASISTPretrainedAdapter()
        self._detectors["aasist_pretrained"] = pretrained_adapter
        self._detectors["aasist"] = pretrained_adapter  # default alias

        # 2. Reference Untrained Baselines & Fallbacks
        self._detectors["aasist_untrained"] = DeepLearningSyntheticDetector(model_type="aasist")
        self._detectors["spec_cnn"] = DeepLearningSyntheticDetector(model_type="spectral_cnn")
        self._detectors["spec_cnn_untrained"] = self._detectors["spec_cnn"]
        self._detectors["fallback"] = FallbackSyntheticDetector()

        if is_valid:
            self._active_name = "aasist_pretrained"
            logger.info("DetectorRegistry initialized with active model: VoiceShield-AASIST-Pretrained-v1")
        else:
            self._active_name = "fallback"
            logger.warning("Pretrained checkpoint unverified; DetectorRegistry active model set to fallback.")

    def register(self, name: str, detector: BaseSyntheticVoiceDetector) -> None:
        """Register a custom detector instance."""
        self._detectors[name.lower()] = detector

    def get_detector(self, name: Optional[str] = None) -> BaseSyntheticVoiceDetector:
        """Retrieve a detector by name or get the current active detector."""
        target_name = (name or self._active_name).lower()
        if target_name not in self._detectors:
            logger.warning(f"Requested detector '{target_name}' not found; falling back to '{self._active_name}'.")
            return self._detectors.get(self._active_name, self._detectors.get("fallback", FallbackSyntheticDetector()))
        return self._detectors[target_name]

    def set_active_detector(self, name: str) -> None:
        """Set the active default detector."""
        target_name = name.lower()
        if target_name not in self._detectors:
            raise KeyError(f"Detector '{target_name}' is not registered.")
        self._active_name = target_name
        logger.info(f"Active detector set to: '{target_name}'")

    def get_active_detector_name(self) -> str:
        """Get the identifier of the active detector."""
        return self._active_name

    def list_detectors(self) -> List[DetectorMetadata]:
        """List metadata for all distinct registered detectors."""
        seen_names = set()
        metadata_list = []
        for det in self._detectors.values():
            meta = det.metadata()
            if meta.model_name not in seen_names:
                seen_names.add(meta.model_name)
                metadata_list.append(meta)
        return metadata_list


# Global registry instance
detector_registry = DetectorRegistry()
