"""Score calibration and decision threshold mapping for synthetic voice detection."""

from typing import Dict, Tuple
from app.core.config import settings
from app.detection.schemas import ClassificationLabel, ScoreType


class ScoreCalibrator:
    """Manages score transformation, threshold classification, and confidence assignment.
    
    Adheres strictly to scientific honesty principles:
    - Raw model logits or continuous outputs without empirical isotonic/Platt validation
      are marked strictly as `uncalibrated_model_score`.
    - Thresholds define explicit bounds for NATURAL, SYNTHETIC, and UNCERTAIN regions.
    """

    def __init__(
        self,
        threshold_natural: float = settings.DETECTION_THRESHOLD_NATURAL,
        threshold_synthetic: float = settings.DETECTION_THRESHOLD_SYNTHETIC,
        score_type: ScoreType = ScoreType.UNCALIBRATED_MODEL_SCORE,
    ):
        if threshold_natural >= threshold_synthetic:
            raise ValueError(
                f"threshold_natural ({threshold_natural}) must be strictly less than threshold_synthetic ({threshold_synthetic})"
            )
        self.threshold_natural = float(threshold_natural)
        self.threshold_synthetic = float(threshold_synthetic)
        self.score_type = score_type

    def classify_score(self, synthetic_score: float) -> Tuple[ClassificationLabel, str]:
        """Map a normalized synthetic score [0.0, 1.0] to a decision label and confidence band.
        
        Args:
            synthetic_score: Normalized synthetic likelihood in [0.0, 1.0].
            
        Returns:
            Tuple of (ClassificationLabel, confidence_band_string)
        """
        clamped_score = max(0.0, min(1.0, float(synthetic_score)))

        if clamped_score < self.threshold_natural:
            label = ClassificationLabel.NATURAL
            if clamped_score <= 0.15:
                confidence = "HIGH"
            else:
                confidence = "MEDIUM"
        elif clamped_score > self.threshold_synthetic:
            label = ClassificationLabel.SYNTHETIC
            if clamped_score >= 0.85:
                confidence = "HIGH"
            else:
                confidence = "MEDIUM"
        else:
            label = ClassificationLabel.UNCERTAIN
            confidence = "UNCERTAIN"

        return label, confidence

    def get_thresholds_dict(self) -> Dict[str, float]:
        """Return applied threshold boundaries."""
        return {
            "natural_threshold": self.threshold_natural,
            "synthetic_threshold": self.threshold_synthetic,
            "uncertain_lower": self.threshold_natural,
            "uncertain_upper": self.threshold_synthetic,
        }
