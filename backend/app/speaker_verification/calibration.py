"""
Calibration and Threshold Configuration for Speaker Verification.
"""
from dataclasses import dataclass
from typing import Optional

from app.speaker_verification.schemas import CalibrationStatus, ConfidenceBand, SpeakerDecision


@dataclass
class SpeakerVerificationThreshold:
    """
    Encapsulates decision threshold metadata.
    """
    threshold: float = 0.65
    threshold_version: str = "v1.0-provisional"
    calibration_status: CalibrationStatus = CalibrationStatus.NOT_CALIBRATED
    operating_point: str = "PROVISIONAL_UNVALIDATED"
    dataset_id: str = "NONE_PENDING_BENCHMARK"
    uncertainty_margin: float = 0.08
    notes: str = (
        "Provisional cosine similarity threshold. "
        "Not calibrated on empirical target/non-target verification trials."
    )


class SpeakerScoreCalibrator:
    """
    Manages probability calibration states for speaker verification scores.
    """
    def __init__(self, threshold_config: Optional[SpeakerVerificationThreshold] = None):
        self.config = threshold_config or SpeakerVerificationThreshold()

    def determine_confidence(self, similarity: float) -> ConfidenceBand:
        """
        Determines the confidence band based on distance from the threshold boundary.
        """
        margin = self.config.uncertainty_margin
        thresh = self.config.threshold

        if similarity >= thresh + margin:
            return ConfidenceBand.HIGH
        elif similarity <= thresh - margin:
            return ConfidenceBand.HIGH if similarity < (thresh - 2 * margin) else ConfidenceBand.MEDIUM
        else:
            return ConfidenceBand.LOW
