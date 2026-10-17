"""
Pydantic schemas for VoiceShield Controlled Fusion & Impersonation Risk Engine.
"""
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RecommendedAction(str, Enum):
    ALLOW = "ALLOW"
    MONITOR = "MONITOR"
    STEP_UP_VERIFICATION = "STEP_UP_VERIFICATION"
    BLOCK_OR_ESCALATE = "BLOCK_OR_ESCALATE"


class EvidenceSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EvidenceItem(BaseModel):
    code: str
    severity: EvidenceSeverity
    message: str


class SyntheticSignalSummary(BaseModel):
    synthetic_score: float
    synthetic_score_type: str = "uncalibrated_model_score"
    classification: str
    detector_id: str
    confidence_band: str


class SpeakerSignalSummary(BaseModel):
    speaker_similarity: float
    speaker_score_type: str = "cosine_similarity"
    decision: str
    confidence_band: str
    profile_id: str
    model_id: str


class ContextualSignals(BaseModel):
    call_duration_seconds: Optional[float] = None
    caller_number_reputation: Optional[str] = None
    known_contact: Optional[bool] = None
    transaction_amount: Optional[float] = None
    new_device: Optional[bool] = None
    geographic_anomaly: Optional[bool] = None
    time_anomaly: Optional[bool] = None
    recent_password_reset: Optional[bool] = None
    recent_account_change: Optional[bool] = None


class RiskAnalysisResult(BaseModel):
    risk_score: int = Field(..., ge=0, le=100, description="Heuristic impersonation risk score from 0 to 100.")
    risk_score_type: str = "heuristic_fusion_score"
    risk_level: RiskLevel
    decision: str
    evidence_confidence: str
    signals: Dict[str, Any]
    evidence: List[EvidenceItem]
    recommended_action: RecommendedAction
    action_reason: str
    calibration_status: str = "NOT_CALIBRATED"
    fusion_version: str = "v1.0-rule-matrix"
    audio_quality_status: str = "good"
    latency_ms: float
    provisional_disclaimer: str = (
        "PROVISIONAL HEURISTIC FUSION: Risk score is a transparent decision-support metric "
        "and is not a calibrated probability of fraud or impersonation."
    )
    privacy: Dict[str, Any]


class RiskSimulationRequest(BaseModel):
    synthetic_score: float = Field(..., ge=0.0, le=1.0, description="Simulated synthetic voice score [0.0, 1.0].")
    speaker_similarity: float = Field(..., ge=-1.0, le=1.0, description="Simulated speaker cosine similarity [-1.0, 1.0].")
    audio_quality: Optional[str] = Field("good", description="Audio quality condition: 'good', 'warning', or 'invalid'.")
    profile_id: Optional[str] = Field("simulated_profile", description="Optional simulated profile ID.")


class RiskConfigResponse(BaseModel):
    fusion_version: str
    ruleset_version: str
    calibration_status: str
    risk_bands: Dict[str, Dict[str, Any]]
    speaker_similarity_bands: Dict[str, Dict[str, Any]]
    synthetic_score_bands: Dict[str, Dict[str, Any]]
    action_policies: Dict[str, str]


class RiskProvenanceResponse(BaseModel):
    fusion_engine: str
    fusion_version: str
    calibration_status: str
    ruleset_version: str
    synthetic_detector: Dict[str, Any]
    speaker_verifier: Dict[str, Any]
    disclaimer: str


class ExtendedRecommendedAction(str, Enum):
    ALLOW = "ALLOW"
    MONITOR = "MONITOR"
    STEP_UP_VERIFICATION = "STEP_UP_VERIFICATION"
    CALL_BACK_REQUIRED = "CALL_BACK_REQUIRED"
    MFA_REQUIRED = "MFA_REQUIRED"
    BLOCK_OR_ESCALATE = "BLOCK_OR_ESCALATE"
    BLOCK_TRANSACTION_AND_ESCALATE = "BLOCK_TRANSACTION_AND_ESCALATE"


class MultiLayerEvidenceItem(BaseModel):
    layer: str = Field(..., description="Intelligence layer: SYNTHETIC, SPEAKER, PROSODY, CONTEXT, or FUSION.")
    code: str = Field(..., description="Machine-readable evidence code.")
    severity: str = Field(..., description="Severity level: INFO, LOW, MEDIUM, HIGH, CRITICAL.")
    message: str = Field(..., description="Human-readable explanation of the finding.")


class MultiLayerRiskAnalysisResult(BaseModel):
    """
    Step 13: Multi-Layer Voice Integrity & Contextual Risk Intelligence Assessment.
    Combines AASIST synthetic voice detection, ECAPA speaker verification, prosodic dynamics, and context.
    """
    overall_risk_score: int = Field(..., ge=0, le=100, description="Heuristic contextual impersonation risk score (0-100).")
    risk_level: RiskLevel = Field(..., description="Categorical risk level: LOW, MEDIUM, HIGH, CRITICAL.")
    risk_score_type: str = Field(
        default="heuristic_contextual_impersonation_risk",
        description="Explicit label indicating score is heuristic and not a calibrated probability."
    )
    recommended_action: ExtendedRecommendedAction = Field(
        ..., description="Context-aware operational prevention recommendation."
    )
    action_reason: str = Field(..., description="Primary explanatory rationale for recommended action.")
    enforcement_disclaimer: str = Field(
        default=(
            "ADVISORY RECOMMENDATION ONLY: VoiceShield provides policy recommendations for downstream orchestration; "
            "actual external transaction blocking or telephony disconnection is not directly enforced by this endpoint."
        ),
        description="Clear distinction between advisory recommendation and actual external enforcement."
    )
    synthetic_signal: Dict[str, Any] = Field(..., description="AASIST detector output summary.")
    speaker_signal: Dict[str, Any] = Field(..., description="SpeechBrain ECAPA-TDNN speaker verification output summary.")
    prosody_signal: Dict[str, Any] = Field(..., description="Prosody and behavioral dynamics output summary.")
    context_signal: Dict[str, Any] = Field(..., description="Contextual risk intelligence output summary.")
    evidence: List[MultiLayerEvidenceItem] = Field(default_factory=list, description="Aggregated explainable evidence items.")
    primary_rationale: str = Field(..., description="High-level human summary of the risk finding.")
    scientific_disclosure: str = Field(
        default=(
            "SCIENTIFIC DISCLOSURE: Overall risk is computed via non-linear multi-signal heuristic rules. "
            "Scores are not statistically calibrated empirical probabilities."
        ),
        description="Mandatory scientific status disclosure."
    )
    latency_ms: float = Field(..., description="Total processing latency in milliseconds.")
    privacy_policy: str = Field(
        default="RAM_ONLY_NO_RAW_AUDIO_OR_EMBEDDINGS_PERSISTED",
        description="Ephemeral in-memory privacy guarantee."
    )


class MultiLayerRiskSimulationRequest(BaseModel):
    synthetic_score: float = Field(..., ge=0.0, le=1.0, description="Simulated synthetic voice score [0.0, 1.0].")
    speaker_similarity: float = Field(..., ge=-1.0, le=1.0, description="Simulated speaker cosine similarity [-1.0, 1.0].")
    prosody_classification: str = Field(
        default="NATURAL_VARIATION",
        description="Simulated prosody class: NATURAL_VARIATION, LOW_VARIATION, UNUSUAL_PROSODY, INSUFFICIENT_AUDIO."
    )
    call_type: str = Field(default="NORMAL_CALL", description="Call type.")
    caller_trust: str = Field(default="UNKNOWN_CALLER", description="Caller trust level.")
    requested_action: str = Field(default="INFORMATION_ONLY", description="Requested action.")
    transaction_amount: Optional[float] = Field(None, ge=0.0, description="Transaction amount in USD.")
    historical_risk: str = Field(default="NONE", description="Historical risk level.")
    profile_id: Optional[str] = Field("simulated_profile", description="Optional simulated profile ID.")
