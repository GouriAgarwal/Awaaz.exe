"""
Evidence Confidence Evaluator for VoiceShield Fusion Engine.
"""
from typing import Dict, Any


class EvidenceConfidenceEvaluator:
    """
    Evaluates evidence reliability factoring audio quality, duration, and model confidence states.
    """

    @staticmethod
    def evaluate_confidence(
        audio_quality_status: str,
        duration_seconds: float,
        synthetic_confidence: str,
        speaker_confidence: str,
    ) -> str:
        """
        Determines evidence confidence level: 'LOW', 'MEDIUM', or 'HIGH'.
        """
        if audio_quality_status == "invalid":
            return "LOW"

        score = 3 # Start with HIGH (3 points)

        # Penalty for degraded audio
        if audio_quality_status == "warning":
            score -= 1

        # Penalty for short duration (< 2.0s)
        if duration_seconds < 2.0:
            score -= 1

        # Penalty for low model confidence
        if synthetic_confidence == "LOW" or speaker_confidence == "LOW":
            score -= 1

        if score >= 3:
            return "HIGH"
        elif score == 2:
            return "MEDIUM"
        else:
            return "LOW"
