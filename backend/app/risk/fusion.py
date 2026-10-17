"""
Controlled Impersonation Risk Fusion Engine.
Orchestrates signal normalization, transparent rule matrix evaluation, confidence scoring, and action mapping.
"""
import math
import logging
from typing import Optional, Dict, Any, List

from app.risk.schemas import (
    RiskAnalysisResult,
    RiskLevel,
    RecommendedAction,
    EvidenceItem,
    SyntheticSignalSummary,
    SpeakerSignalSummary,
    ContextualSignals,
)
from app.risk.normalization import SignalNormalizer
from app.risk.rules import FusionRuleMatrix
from app.risk.confidence import EvidenceConfidenceEvaluator
from app.risk.actions import ActionPolicyMapper
from app.speaker_verification.privacy import BiometricPrivacyPolicy

logger = logging.getLogger(__name__)


class ControlledImpersonationRiskEngine:
    """
    Combines independent synthetic-detection and speaker-verification signals into an explainable risk assessment.
    """
    FUSION_VERSION = "v1.0-rule-matrix"

    @classmethod
    def analyze_signals(
        cls,
        synth_score: float,
        synth_classification: str,
        synth_confidence: str,
        synth_detector_id: str,
        spk_similarity: float,
        spk_decision: str,
        spk_confidence: str,
        spk_profile_id: str,
        spk_model_id: str,
        duration_seconds: float,
        audio_quality_status: str = "good",
        contextual_signals: Optional[ContextualSignals] = None,
        latency_ms: float = 0.0,
    ) -> RiskAnalysisResult:
        """
        Executes controlled fusion.
        """
        # Sanity validation
        if not math.isfinite(synth_score) or not math.isfinite(spk_similarity):
            logger.warning(f"Non-finite scores detected: synth={synth_score}, spk={spk_similarity}")
            synth_score = 0.5
            spk_similarity = 0.0

        synth_score = max(0.0, min(1.0, float(synth_score)))
        spk_similarity = max(-1.0, min(1.0, float(spk_similarity)))

        # 1. Normalize signals into documented evaluation bands
        synth_band = SignalNormalizer.categorize_synthetic_score(synth_score, synth_classification)
        spk_band = SignalNormalizer.categorize_speaker_similarity(spk_similarity, spk_decision)

        # 2. Evaluate Rule Matrix
        risk_score, risk_level, decision_code, evidence = FusionRuleMatrix.evaluate(
            synth_band=synth_band,
            spk_band=spk_band,
            synth_score=synth_score,
            spk_sim=spk_similarity,
            audio_quality_status=audio_quality_status,
        )

        # 3. Evaluate Evidence Confidence
        confidence_level = EvidenceConfidenceEvaluator.evaluate_confidence(
            audio_quality_status=audio_quality_status,
            duration_seconds=duration_seconds,
            synthetic_confidence=synth_confidence,
            speaker_confidence=spk_confidence,
        )

        # 4. Map Recommended Action
        recommended_action, action_reason = ActionPolicyMapper.map_action(
            risk_level=risk_level,
            evidence=evidence,
            audio_quality_status=audio_quality_status,
        )

        # 5. Build Signals Summary
        signals = {
            "synthetic_detection": SyntheticSignalSummary(
                synthetic_score=round(synth_score, 4),
                synthetic_score_type="uncalibrated_model_score",
                classification=synth_classification,
                detector_id=synth_detector_id,
                confidence_band=synth_confidence,
            ).model_dump(),
            "speaker_verification": SpeakerSignalSummary(
                speaker_similarity=round(spk_similarity, 4),
                speaker_score_type="cosine_similarity",
                decision=spk_decision,
                confidence_band=spk_confidence,
                profile_id=spk_profile_id,
                model_id=spk_model_id,
            ).model_dump(),
            "contextual_signals": contextual_signals.model_dump() if contextual_signals else None,
        }

        privacy_meta = BiometricPrivacyPolicy.get_privacy_metadata(in_memory_only=True).model_dump()

        return RiskAnalysisResult(
            risk_score=risk_score,
            risk_score_type="heuristic_fusion_score",
            risk_level=risk_level,
            decision=decision_code,
            evidence_confidence=confidence_level,
            signals=signals,
            evidence=evidence,
            recommended_action=recommended_action,
            action_reason=action_reason,
            calibration_status="NOT_CALIBRATED",
            fusion_version=cls.FUSION_VERSION,
            audio_quality_status=audio_quality_status,
            latency_ms=round(latency_ms, 2),
            privacy=privacy_meta,
        )
