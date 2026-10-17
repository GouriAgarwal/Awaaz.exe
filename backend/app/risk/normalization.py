"""
Score Normalization and Band Categorization for Fusion Engine.
Maps continuous model outputs into discrete bands for transparent, auditable rule evaluation.
"""
from enum import Enum
from typing import Tuple


class SyntheticBand(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNCERTAIN = "UNCERTAIN"


class SpeakerSimilarityBand(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNCERTAIN = "UNCERTAIN"


class SignalNormalizer:
    """
    Categorizes raw synthetic scores and cosine similarities into documented evaluation bands.
    """

    @staticmethod
    def categorize_synthetic_score(score: float, classification: str = "") -> SyntheticBand:
        """
        Categorizes AASIST synthetic score into LOW, MEDIUM, HIGH, or UNCERTAIN.
        """
        if classification in ["MODEL_UNAVAILABLE", "UNAVAILABLE"]:
            return SyntheticBand.UNCERTAIN

        if score < 0.35:
            return SyntheticBand.LOW
        elif score < 0.65:
            return SyntheticBand.MEDIUM
        else:
            return SyntheticBand.HIGH

    @staticmethod
    def categorize_speaker_similarity(similarity: float, decision: str = "") -> SpeakerSimilarityBand:
        """
        Categorizes ECAPA-TDNN cosine similarity into VERY_LOW, LOW, MEDIUM, HIGH, or UNCERTAIN.
        """
        if decision in ["UNAVAILABLE", "PROFILE_NOT_FOUND"]:
            return SpeakerSimilarityBand.UNCERTAIN

        if similarity < 0.25:
            return SpeakerSimilarityBand.VERY_LOW
        elif similarity < 0.55:
            return SpeakerSimilarityBand.LOW
        elif similarity < 0.65:
            return SpeakerSimilarityBand.MEDIUM
        else:
            return SpeakerSimilarityBand.HIGH

