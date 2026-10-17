"""
Decision Engine for Speaker Identity Verification.
"""
from typing import Tuple

from app.speaker_verification.schemas import SpeakerDecision, ConfidenceBand
from app.speaker_verification.calibration import SpeakerScoreCalibrator, SpeakerVerificationThreshold


class SpeakerDecisionEngine:
    """
    Renders MATCH / NON_MATCH / UNCERTAIN decisions with confidence bands.
    """
    def __init__(self, calibrator: SpeakerScoreCalibrator):
        self.calibrator = calibrator

    def evaluate(self, similarity: float) -> Tuple[SpeakerDecision, ConfidenceBand]:
        """
        Evaluates similarity against threshold.
        """
        thresh = self.calibrator.config.threshold
        margin = self.calibrator.config.uncertainty_margin
        confidence = self.calibrator.determine_confidence(similarity)

        if similarity >= (thresh + margin / 2):
            decision = SpeakerDecision.MATCH
        elif similarity < (thresh - margin / 2):
            decision = SpeakerDecision.NON_MATCH
        else:
            decision = SpeakerDecision.UNCERTAIN

        return decision, confidence
