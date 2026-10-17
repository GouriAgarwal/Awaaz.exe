"""
VoiceShield Impersonation Risk Fusion Package.
"""
from app.risk.schemas import (
    RiskLevel,
    RecommendedAction,
    EvidenceSeverity,
    EvidenceItem,
    SyntheticSignalSummary,
    SpeakerSignalSummary,
    ContextualSignals,
    RiskAnalysisResult,
    RiskSimulationRequest,
    RiskConfigResponse,
    RiskProvenanceResponse,
)
from app.risk.normalization import SignalNormalizer, SyntheticBand, SpeakerSimilarityBand
from app.risk.rules import FusionRuleMatrix
from app.risk.confidence import EvidenceConfidenceEvaluator
from app.risk.actions import ActionPolicyMapper
from app.risk.fusion import ControlledImpersonationRiskEngine
from app.risk.service import ImpersonationRiskService
from app.risk.provenance import get_risk_provenance

__all__ = [
    "RiskLevel",
    "RecommendedAction",
    "EvidenceSeverity",
    "EvidenceItem",
    "SyntheticSignalSummary",
    "SpeakerSignalSummary",
    "ContextualSignals",
    "RiskAnalysisResult",
    "RiskSimulationRequest",
    "RiskConfigResponse",
    "RiskProvenanceResponse",
    "SignalNormalizer",
    "SyntheticBand",
    "SpeakerSimilarityBand",
    "FusionRuleMatrix",
    "EvidenceConfidenceEvaluator",
    "ActionPolicyMapper",
    "ControlledImpersonationRiskEngine",
    "ImpersonationRiskService",
    "get_risk_provenance",
]
