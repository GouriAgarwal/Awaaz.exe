"""
Pydantic schemas for VoiceShield Speaker Identity Verification Subsystem.
"""
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class SpeakerDecision(str, Enum):
    MATCH = "MATCH"
    NON_MATCH = "NON_MATCH"
    UNCERTAIN = "UNCERTAIN"


class ConfidenceBand(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class CalibrationStatus(str, Enum):
    NOT_CALIBRATED = "NOT_CALIBRATED"
    CALIBRATED_PLATT = "CALIBRATED_PLATT"
    CALIBRATED_ISOTONIC = "CALIBRATED_ISOTONIC"


class ScoreType(str, Enum):
    COSINE_SIMILARITY = "cosine_similarity"


class PrivacyMetadata(BaseModel):
    raw_audio_persisted: bool = Field(False, description="Whether raw audio was saved to disk or database.")
    embeddings_logged: bool = Field(False, description="Whether biometric embedding vectors were logged.")
    in_memory_only: bool = Field(True, description="Whether profile is held strictly in volatile RAM.")
    policy: str = Field(
        "VoiceShield Biometric Privacy: Raw audio is discarded immediately after feature extraction. Embeddings are never logged in plaintext.",
        description="Privacy policy statement."
    )


class AudioQualitySummary(BaseModel):
    duration_seconds: float
    sample_rate: int = 16000
    rms_db: float
    peak_db: float
    snr_estimate_db: float
    clipping_detected: bool = False
    silence_ratio: float = 0.0


class SpeakerProfileSummary(BaseModel):
    profile_id: str
    model_id: str
    embedding_dimension: int
    sample_count: int
    total_audio_duration_seconds: float
    created_at: str
    updated_at: str
    in_memory_only: bool = True


class EnrollmentResponse(BaseModel):
    success: bool
    profile_id: str
    sample_count: int
    total_audio_duration_seconds: float
    audio_quality: AudioQualitySummary
    privacy: PrivacyMetadata
    message: str
    warning: Optional[str] = None


class VerificationResponse(BaseModel):
    profile_id: str
    model_id: str
    similarity_score: float = Field(..., ge=-1.0, le=1.0, description="Cosine similarity between enrollment centroid and verification sample.")
    score_type: ScoreType = ScoreType.COSINE_SIMILARITY
    decision: SpeakerDecision
    confidence_band: ConfidenceBand
    threshold: float
    threshold_version: str
    calibration_status: CalibrationStatus
    provisional_warning: Optional[str] = Field(
        "PROVISIONAL — NOT SCIENTIFICALLY VALIDATED: Cosine similarity is not an identity probability. Threshold is uncalibrated.",
        description="Scientific disclosure notice."
    )
    duration_seconds: float
    latency_ms: float
    audio_quality: AudioQualitySummary
    privacy: PrivacyMetadata


class SpeakerModelProvenance(BaseModel):
    model_id: str
    model_name: str
    version: str
    architecture: str
    source_repository: str
    checkpoint_url: str
    checkpoint_sha256: str
    license: str
    embedding_dimension: int
    expected_sample_rate: int
    parameter_count: int
    training_dataset: str
    is_l2_normalized: bool
    calibration_status: CalibrationStatus
    scientific_status: str
    disclaimer: str


# Future-compatible schemas for anti-spoofing + identity verification fusion
class SpeakerVerificationResult(BaseModel):
    profile_id: str
    similarity_score: float
    decision: SpeakerDecision
    is_enrolled: bool = True


class SyntheticDetectionResult(BaseModel):
    synthetic_score: float
    classification: str
    detector_id: str


class ImpersonationRiskResult(BaseModel):
    status: str = "FUSION_NOT_YET_IMPLEMENTED"
    speaker_verification: Optional[SpeakerVerificationResult] = None
    synthetic_detection: Optional[SyntheticDetectionResult] = None
    note: str = "Step 8 implements Speaker Identity independently. Cross-modal fusion engine is planned for subsequent steps."
