"""
Pydantic schemas and enums for Prosody & Behavioral Analysis Subsystem.
Provides explainable acoustic variance indicators, pitch statistics, and pause dynamics.
"""
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ProsodyClassification(str, Enum):
    """Scientifically honest classification of speech prosodic dynamics."""
    NATURAL_VARIATION = "NATURAL_VARIATION"
    LOW_VARIATION = "LOW_VARIATION"
    UNUSUAL_PROSODY = "UNUSUAL_PROSODY"
    INSUFFICIENT_AUDIO = "INSUFFICIENT_AUDIO"


class ProsodySeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ProsodyEvidenceItem(BaseModel):
    """Structured evidence explaining prosodic observations."""
    code: str = Field(..., description="Machine-readable evidence code, e.g. PROSODY_FLAT_PITCH.")
    severity: ProsodySeverity = Field(..., description="Evidence severity rating.")
    message: str = Field(..., description="Human-readable explanation of the finding.")


class ProsodyFeatureSummary(BaseModel):
    """Explainable, deterministic prosodic feature summary extracted from speech audio."""
    pitch_mean_hz: Optional[float] = Field(None, description="Voiced fundamental frequency mean (Hz).")
    pitch_std_hz: Optional[float] = Field(None, description="Voiced fundamental frequency standard deviation (Hz).")
    pitch_min_hz: Optional[float] = Field(None, description="Voiced fundamental frequency minimum (Hz).")
    pitch_max_hz: Optional[float] = Field(None, description="Voiced fundamental frequency maximum (Hz).")
    pitch_range_hz: Optional[float] = Field(None, description="F0 range: max - min (Hz).")
    pitch_variation_coef: Optional[float] = Field(
        None, description="Pitch variation coefficient: std / mean. Values < 0.05 often indicate monotone delivery."
    )
    energy_rms_mean: float = Field(..., description="Mean root-mean-square frame energy.")
    energy_rms_std: float = Field(..., description="Standard deviation of frame RMS energy.")
    energy_dynamic_range_db: float = Field(..., description="Dynamic range in decibels (max RMS / min active RMS).")
    voiced_unvoiced_ratio: float = Field(..., description="Ratio of voiced frames to unvoiced speech frames.")
    speech_activity_ratio: float = Field(..., description="Fraction of duration classified as active speech.")
    pause_ratio: float = Field(..., description="Fraction of utterance containing silent/unvoiced pauses.")
    pause_count: int = Field(..., description="Number of detected pause intervals (>150ms).")
    pause_mean_duration_ms: float = Field(..., description="Mean pause interval duration in milliseconds.")
    pause_max_duration_ms: float = Field(..., description="Longest pause interval duration in milliseconds.")
    speech_rhythm_proxy: float = Field(
        ..., description="Proxy for speaking rhythm cadence (voiced syllabic burst rate per second)."
    )
    microvariation_jitter_proxy: Optional[float] = Field(
        None, description="Mean frame-to-frame relative pitch jitter proxy."
    )
    energy_delta_mean: float = Field(..., description="Mean frame-to-frame energy transition delta.")


class ProsodyAnalysisResult(BaseModel):
    """Complete prosodic and behavioral integrity assessment."""
    classification: ProsodyClassification = Field(..., description="Honest prosodic classification.")
    features: ProsodyFeatureSummary = Field(..., description="Detailed extracted prosodic feature summary.")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in prosodic extraction reliability.")
    confidence: str = Field(..., description="Confidence rating: HIGH, MEDIUM, LOW, or UNRELIABLE.")
    evidence: List[ProsodyEvidenceItem] = Field(default_factory=list, description="Structured explanatory evidence items.")
    duration_seconds: float = Field(..., description="Processed audio duration in seconds.")
    voiced_frames_count: int = Field(..., description="Number of detected voiced speech frames.")
    total_frames_count: int = Field(..., description="Total analyzed frames.")
    limitations_disclosure: str = Field(
        default=(
            "SCIENTIFIC DISCLOSURE: Prosody and behavioral analysis provides acoustic variance indicators; "
            "it does not prove speech synthesis on its own and can be affected by emotional state, speaking style, "
            "and telecommunication codecs."
        ),
        description="Mandatory scientific limitations disclosure."
    )
    ephemeral_privacy: str = Field(
        default="RAM_ONLY_NO_AUDIO_PERSISTENCE",
        description="Biometric privacy guarantee: raw waveforms are discarded immediately after feature extraction."
    )
