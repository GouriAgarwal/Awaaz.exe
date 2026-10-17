"""
Speaker Identity Verification Subsystem for VoiceShield.
"""
from app.speaker_verification.base import BaseSpeakerEncoder
from app.speaker_verification.embedding import SpeechBrainECAPAEncoder
from app.speaker_verification.similarity import CosineSimilarityMetric
from app.speaker_verification.calibration import SpeakerScoreCalibrator, SpeakerVerificationThreshold
from app.speaker_verification.decision import SpeakerDecisionEngine
from app.speaker_verification.enrollment import SpeakerEnrollmentManager
from app.speaker_verification.registry import SpeakerVerificationService
from app.speaker_verification.provenance import get_speaker_model_provenance
from app.speaker_verification.privacy import BiometricPrivacyPolicy
from app.speaker_verification.schemas import (
    SpeakerDecision,
    ConfidenceBand,
    CalibrationStatus,
    ScoreType,
    EnrollmentResponse,
    VerificationResponse,
    SpeakerProfileSummary,
    SpeakerModelProvenance,
)

__all__ = [
    "BaseSpeakerEncoder",
    "SpeechBrainECAPAEncoder",
    "CosineSimilarityMetric",
    "SpeakerScoreCalibrator",
    "SpeakerVerificationThreshold",
    "SpeakerDecisionEngine",
    "SpeakerEnrollmentManager",
    "SpeakerVerificationService",
    "get_speaker_model_provenance",
    "BiometricPrivacyPolicy",
    "SpeakerDecision",
    "ConfidenceBand",
    "CalibrationStatus",
    "ScoreType",
    "EnrollmentResponse",
    "VerificationResponse",
    "SpeakerProfileSummary",
    "SpeakerModelProvenance",
]
