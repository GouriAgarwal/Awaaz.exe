"""
Pydantic schemas and enums for Contextual Risk Intelligence Engine.
Captures business, organizational, and transactional risk factors.
"""
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class CallType(str, Enum):
    NORMAL_CALL = "NORMAL_CALL"
    FINANCIAL_TRANSACTION = "FINANCIAL_TRANSACTION"
    PRIVILEGED_ACCESS = "PRIVILEGED_ACCESS"
    GOVERNMENT_INSTRUCTION = "GOVERNMENT_INSTRUCTION"
    ENTERPRISE_APPROVAL = "ENTERPRISE_APPROVAL"


class CallerTrust(str, Enum):
    KNOWN_CONTACT = "KNOWN_CONTACT"
    UNKNOWN_CALLER = "UNKNOWN_CALLER"
    VERIFIED_CONTACT = "VERIFIED_CONTACT"
    VIP_OR_EXECUTIVE = "VIP_OR_EXECUTIVE"


class RequestedAction(str, Enum):
    INFORMATION_ONLY = "INFORMATION_ONLY"
    PAYMENT_APPROVAL = "PAYMENT_APPROVAL"
    FUND_TRANSFER = "FUND_TRANSFER"
    CREDENTIAL_RESET = "CREDENTIAL_RESET"
    SENSITIVE_DATA_DISCLOSURE = "SENSITIVE_DATA_DISCLOSURE"
    PRIVILEGED_ACCESS_CHANGE = "PRIVILEGED_ACCESS_CHANGE"


class HistoricalRisk(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ContextRiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PolicySensitivity(str, Enum):
    STANDARD = "STANDARD"
    ELEVATED = "ELEVATED"
    HIGH_STAKES = "HIGH_STAKES"
    CRITICAL_DEFENSE = "CRITICAL_DEFENSE"


class ContextEvidenceSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ContextEvidenceItem(BaseModel):
    code: str = Field(..., description="Evidence code identifier, e.g. CONTEXT_HIGH_VALUE_TRANSFER.")
    severity: ContextEvidenceSeverity = Field(..., description="Severity level.")
    message: str = Field(..., description="Explanatory description of the risk factor.")


class ContextMetadata(BaseModel):
    """Business and interaction context parameters."""
    call_type: CallType = Field(default=CallType.NORMAL_CALL, description="Type of call or channel interaction.")
    caller_trust: CallerTrust = Field(default=CallerTrust.UNKNOWN_CALLER, description="Trust level of caller identity.")
    requested_action: RequestedAction = Field(
        default=RequestedAction.INFORMATION_ONLY, description="Specific action or authorization being requested."
    )
    transaction_amount: Optional[float] = Field(None, ge=0.0, description="Optional monetary value of requested transaction.")
    historical_risk: HistoricalRisk = Field(
        default=HistoricalRisk.NONE, description="Historical fraud incident record or account risk level."
    )


class ContextRiskResult(BaseModel):
    """Output of the contextual risk intelligence evaluation."""
    context_risk_score: int = Field(..., ge=0, le=100, description="Heuristic context risk score (0-100).")
    risk_level: ContextRiskLevel = Field(..., description="Categorical risk band.")
    policy_sensitivity: PolicySensitivity = Field(..., description="Recommended enforcement sensitivity.")
    sensitivity_multiplier: float = Field(..., ge=1.0, le=2.5, description="Non-linear fusion sensitivity weight.")
    evidence: List[ContextEvidenceItem] = Field(default_factory=list, description="Structured contextual evidence items.")
    score_type: str = Field(
        default="heuristic_contextual_risk",
        description="Explicit disclosure that the score is a heuristic decision-support metric."
    )
    disclaimer: str = Field(
        default=(
            "HEURISTIC CONTEXTUAL SCORING: Context risk scores reflect organizational threat modeling and policy rules. "
            "They do not represent statistically calibrated empirical probabilities."
        ),
        description="Scientific limitations disclosure."
    )
