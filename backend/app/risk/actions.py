"""
Operational Action Policy Mapper for VoiceShield Risk Decisions.
"""
from typing import Tuple, List
from app.risk.schemas import RiskLevel, RecommendedAction, EvidenceItem


class ActionPolicyMapper:
    """
    Maps evaluated risk levels and evidence items into recommended operational actions.
    """

    @staticmethod
    def map_action(
        risk_level: RiskLevel,
        evidence: List[EvidenceItem],
        audio_quality_status: str
    ) -> Tuple[RecommendedAction, str]:
        """
        Determines RecommendedAction and operational justification string.
        """
        has_clone_evidence = any(e.code == "POTENTIAL_VOICE_CLONE" for e in evidence)

        if audio_quality_status == "invalid":
            return (
                RecommendedAction.MONITOR,
                "Action held in monitor state due to degraded/invalid audio quality."
            )

        if risk_level == RiskLevel.CRITICAL or has_clone_evidence:
            return (
                RecommendedAction.BLOCK_OR_ESCALATE,
                "Strong acoustic evidence consistent with targeted voice-cloning attack. Immediate escalation recommended."
            )
        elif risk_level == RiskLevel.HIGH:
            return (
                RecommendedAction.STEP_UP_VERIFICATION,
                "Elevated synthetic voice or identity anomaly detected. Secondary out-of-band verification recommended."
            )
        elif risk_level == RiskLevel.MEDIUM:
            return (
                RecommendedAction.MONITOR,
                "Moderate acoustic anomaly or non-enrolled voice detected. Passive monitoring recommended."
            )
        else: # LOW
            return (
                RecommendedAction.ALLOW,
                "Acoustic and identity signals are consistent with genuine enrolled caller."
            )
