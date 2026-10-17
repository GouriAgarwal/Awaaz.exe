"""
Pydantic schemas and enums for VoiceShield Automated Prevention & Response Subsystem.
Defines prevention actions, statuses, verification methods, policy profiles, and timeline events.
"""
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class PreventionAction(str, Enum):
    """Actionable prevention response prescribed by the prevention policy engine."""
    ALLOW = "ALLOW"
    MONITOR = "MONITOR"
    SHOW_WARNING = "SHOW_WARNING"
    PAUSE_SENSITIVE_ACTION = "PAUSE_SENSITIVE_ACTION"
    REQUIRE_CALLBACK = "REQUIRE_CALLBACK"
    REQUIRE_MFA = "REQUIRE_MFA"
    REQUIRE_SECONDARY_VERIFICATION = "REQUIRE_SECONDARY_VERIFICATION"
    ESCALATE_TO_SUPERVISOR = "ESCALATE_TO_SUPERVISOR"
    ESCALATE_TO_SOC = "ESCALATE_TO_SOC"
    BLOCK_TRANSACTION = "BLOCK_TRANSACTION"


class PreventionStatus(str, Enum):
    """Workflow state lifecycle indicator."""
    ALLOWED = "ALLOWED"
    MONITORING = "MONITORING"
    WARNING = "WARNING"
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"
    PAUSED = "PAUSED"
    ESCALATED = "ESCALATED"
    BLOCKED = "BLOCKED"
    RESOLVED = "RESOLVED"


class SensitiveActionType(str, Enum):
    """Classification of the operational or business action attempted during the call."""
    FUND_TRANSFER = "FUND_TRANSFER"
    PAYMENT_APPROVAL = "PAYMENT_APPROVAL"
    ACCOUNT_CHANGE = "ACCOUNT_CHANGE"
    PRIVILEGED_ACCESS = "PRIVILEGED_ACCESS"
    CONFIDENTIAL_DISCLOSURE = "CONFIDENTIAL_DISCLOSURE"
    EXECUTIVE_INSTRUCTION = "EXECUTIVE_INSTRUCTION"
    GENERAL_CALL = "GENERAL_CALL"


class VerificationMethod(str, Enum):
    """Secondary authentication mechanism recommended to clear a paused or challenged action."""
    CALLBACK_TO_REGISTERED_NUMBER = "CALLBACK_TO_REGISTERED_NUMBER"
    MULTI_FACTOR_AUTHENTICATION = "MULTI_FACTOR_AUTHENTICATION"
    SUPERVISOR_APPROVAL = "SUPERVISOR_APPROVAL"
    OUT_OF_BAND_VERIFICATION = "OUT_OF_BAND_VERIFICATION"
    SECURITY_QUESTION = "SECURITY_QUESTION"
    BIOMETRIC_REAUTHENTICATION = "BIOMETRIC_REAUTHENTICATION"


class PolicyProfile(str, Enum):
    """Organizational risk tolerance profile."""
    BANKING = "BANKING"
    ENTERPRISE = "ENTERPRISE"
    GOVERNMENT = "GOVERNMENT"
    TELECOM = "TELECOM"
    DEFAULT = "DEFAULT"


class TimelineEvent(BaseModel):
    """Safe, privacy-preserving incident timeline record."""
    event_id: str = Field(..., description="Unique event identifier.")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of occurrence.")
    event_type: str = Field(..., description="Machine-readable event classification code.")
    risk_level: Optional[str] = Field(None, description="Impersonation risk level associated with event.")
    risk_score: Optional[int] = Field(None, ge=0, le=100, description="Risk score (0-100).")
    prevention_status: PreventionStatus = Field(..., description="Workflow state at time of event.")
    prevention_action: PreventionAction = Field(..., description="Action taken or recommended.")
    details: str = Field(..., description="Human-readable audit log message.")
    correlation_id: Optional[str] = Field(None, description="Request or session correlation ID.")


class PreventionEvaluationRequest(BaseModel):
    """Input payload for evaluating prevention policies against voice and contextual risk."""
    risk_score: int = Field(..., ge=0, le=100, description="Overall impersonation risk score (0-100).")
    risk_level: str = Field(..., description="Impersonation risk band: LOW, MEDIUM, HIGH, CRITICAL.")
    recommended_action: Optional[str] = Field(None, description="Recommendation from upstream fusion.")
    synthetic_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="AASIST synthetic score.")
    speaker_similarity: Optional[float] = Field(None, ge=-1.0, le=1.0, description="ECAPA speaker cosine similarity.")
    context_risk_score: Optional[int] = Field(None, ge=0, le=100, description="Step 13 context risk score.")
    sensitive_action_type: SensitiveActionType = Field(
        default=SensitiveActionType.GENERAL_CALL, description="Attempted sensitive action."
    )
    transaction_amount: Optional[float] = Field(None, ge=0.0, description="Transaction monetary amount in currency units.")
    policy_profile: PolicyProfile = Field(
        default=PolicyProfile.DEFAULT, description="Active organizational policy profile."
    )
    caller_trust: Optional[str] = Field(None, description="Trust band of caller identity.")
    caller_id: Optional[str] = Field(None, description="Sanitized caller telephone or endpoint ID.")
    target_profile_id: Optional[str] = Field(None, description="Claimed enrolled speaker profile ID.")
    session_id: Optional[str] = Field(None, description="Active streaming or call session ID.")


class PreventionEvaluationResponse(BaseModel):
    """Complete prevention evaluation outcome and active workflow state."""
    workflow_id: str = Field(..., description="Unique prevention workflow tracking ID.")
    prevention_status: PreventionStatus = Field(..., description="Current lifecycle state.")
    primary_action: PreventionAction = Field(..., description="Immediate prescribed action.")
    required_actions: List[PreventionAction] = Field(default_factory=list, description="All required mitigation steps.")
    is_blocked: bool = Field(..., description="Whether sensitive action execution is halted.")
    is_paused: bool = Field(..., description="Whether action is temporarily suspended awaiting verification.")
    requires_verification: bool = Field(..., description="Whether secondary verification is required.")
    recommended_verification_methods: List[VerificationMethod] = Field(
        default_factory=list, description="Available verification pathways."
    )
    policy_profile: PolicyProfile = Field(..., description="Policy profile utilized.")
    explanation: str = Field(..., description="Clear explanatory rationale for prevention response.")
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Contributing risk signals and policy rules.")
    timeline: List[TimelineEvent] = Field(default_factory=list, description="Safe chronological incident event history.")
    created_at: str = Field(..., description="Workflow initialization timestamp.")
    updated_at: str = Field(..., description="Last workflow modification timestamp.")
    enforcement_disclaimer: str = Field(
        default=(
            "ADVISORY FRAUD PREVENTION WORKFLOW: VoiceShield provides automated decision-support and workflow controls. "
            "Downstream integration with core banking, IAM, or telecom APIs is required for automated physical enforcement."
        ),
        description="Mandatory scientific and operational disclosure."
    )


class VerificationSimulationRequest(BaseModel):
    """Payload for simulating out-of-band verification outcomes in demo and testing environments."""
    verification_type: str = Field(
        ...,
        description=(
            "Verification outcome to simulate: callback_passed, callback_failed, mfa_passed, "
            "mfa_failed, supervisor_approved, supervisor_rejected."
        ),
    )
    actor: Optional[str] = Field("security_operator", description="Operator, system, or supervisor identifier.")
    notes: Optional[str] = Field(None, description="Audit documentation notes.")


class PreventionResolutionRequest(BaseModel):
    """Payload for resolving an incident workflow following human review."""
    resolution_reason: str = Field(..., description="Audit rationale for resolving or closing the incident.")
    resolved_by: str = Field(..., description="Operator or supervisor badge/username.")
    final_action: PreventionAction = Field(
        default=PreventionAction.ALLOW, description="Final disposition: ALLOW or BLOCK_TRANSACTION."
    )


class DemoScenario(BaseModel):
    """Pre-configured hackathon demonstration scenario."""
    id: str
    name: str
    description: str
    sensitive_action: SensitiveActionType
    transaction_amount: Optional[float]
    policy_profile: PolicyProfile
    risk_score: int
    risk_level: str
    synthetic_score: float
    speaker_similarity: float
    context_risk_score: int
    caller_trust: str
    expected_status: PreventionStatus
    expected_action: PreventionAction
    key_evidence: str
