"""
Multi-Layer Voice Integrity & Contextual Risk Fusion Engine.
Combines AASIST synthetic voice detection, ECAPA speaker verification,
prosody & behavioral dynamics, and business context risk into an explainable, non-linear assessment.
"""
import math
import logging
from typing import Dict, Any, List, Optional, Tuple

from app.risk.schemas import (
    RiskLevel,
    ExtendedRecommendedAction,
    MultiLayerEvidenceItem,
    MultiLayerRiskAnalysisResult,
)
from app.prosody.schemas import ProsodyAnalysisResult, ProsodyClassification
from app.context.schemas import (
    ContextMetadata,
    ContextRiskResult,
    CallType,
    RequestedAction,
    ContextRiskLevel,
)
from app.context.engine import ContextualRiskEngine

logger = logging.getLogger(__name__)


class MultiLayerImpersonationRiskEngine:
    """
    Step 13: Deterministic Multi-Layer Voice Cloning & Contextual Impersonation Defense Engine.
    """
    FUSION_VERSION = "v2.0-multi-layer-contextual"

    @classmethod
    def evaluate(
        cls,
        synth_score: float,
        synth_classification: str,
        synth_confidence: str,
        spk_similarity: float,
        spk_decision: str,
        spk_confidence: str,
        prosody_result: ProsodyAnalysisResult,
        context_result: ContextRiskResult,
        context_meta: ContextMetadata,
        profile_id: Optional[str] = None,
        synth_model_name: str = "AASIST",
        spk_model_name: str = "SpeechBrain ECAPA-TDNN",
        latency_ms: float = 0.0,
    ) -> MultiLayerRiskAnalysisResult:
        """
        Executes explainable, non-linear multi-layer fusion.
        """
        evidence: List[MultiLayerEvidenceItem] = []

        # Sanitize raw numeric inputs
        synth_val = max(0.0, min(1.0, float(synth_score))) if math.isfinite(synth_score) else 0.5
        spk_sim = max(-1.0, min(1.0, float(spk_similarity))) if math.isfinite(spk_similarity) else 0.0

        # -------------------------------------------------------------
        # 1. Base Acoustic / Biometric Impersonation Risk Calculation
        # -------------------------------------------------------------
        # Quadrant 1: Synthetic voice + Matches enrolled profile -> Targeted Voice Cloning Attack
        # Quadrant 2: Natural voice + Matches enrolled profile -> Legitimate Enrolled Speaker
        # Quadrant 3: Synthetic voice + Non-match -> Generic Synthetic / Spoofed Audio
        # Quadrant 4: Natural voice + Non-match -> Legitimate Non-enrolled Speaker / Impostor
        is_synth = synth_val >= 0.50
        is_spk_match = spk_sim >= 0.65

        if is_synth and is_spk_match:
            # High-confidence targeted voice clone
            base_risk = 75.0 + (synth_val - 0.50) * 30.0 + max(0.0, (spk_sim - 0.65)) * 25.0
            evidence.append(MultiLayerEvidenceItem(
                layer="FUSION",
                code="FUSION_TARGETED_CLONE_MATCH",
                severity="CRITICAL",
                message=f"High synthetic probability ({synth_val:.3f}) combined with enrolled speaker biometric match ({spk_sim:.3f}) indicates a targeted voice cloning attack."
            ))
        elif is_synth and not is_spk_match:
            # Synthetic speech from an un-enrolled or mismatched identity
            base_risk = 55.0 + (synth_val - 0.50) * 25.0
            evidence.append(MultiLayerEvidenceItem(
                layer="FUSION",
                code="FUSION_SYNTHETIC_SPEECH_NON_MATCH",
                severity="HIGH",
                message=f"Synthetic voice indicators detected ({synth_val:.3f}) with speaker identity mismatch ({spk_sim:.3f})."
            ))
        elif not is_synth and is_spk_match:
            # Natural speech matching enrolled speaker
            base_risk = max(5.0, 15.0 * synth_val - max(0.0, (spk_sim - 0.65)) * 20.0)
            evidence.append(MultiLayerEvidenceItem(
                layer="FUSION",
                code="FUSION_LEGITIMATE_SPEAKER_MATCH",
                severity="INFO",
                message=f"Natural acoustic spectral characteristics ({synth_val:.3f}) verified with enrolled speaker biometric match ({spk_sim:.3f})."
            ))
        else:
            # Natural speech, but non-matching speaker identity (e.g. impostor, un-enrolled caller)
            base_risk = 35.0 + max(0.0, (0.65 - spk_sim)) * 20.0
            evidence.append(MultiLayerEvidenceItem(
                layer="FUSION",
                code="FUSION_NATURAL_IDENTITY_MISMATCH",
                severity="MEDIUM",
                message=f"Natural speech detected, but biometric similarity ({spk_sim:.3f}) does not match claimed enrolled speaker profile."
            ))

        # Add Layer 1 (Synthetic) Evidence
        if synth_classification == "SYNTHETIC":
            evidence.append(MultiLayerEvidenceItem(
                layer="SYNTHETIC",
                code="SYNTHETIC_DETECTOR_POSITIVE",
                severity="HIGH",
                message=f"{synth_model_name} classified voice as SYNTHETIC (score={synth_val:.4f}, confidence={synth_confidence})."
            ))
        elif synth_classification == "NATURAL":
            evidence.append(MultiLayerEvidenceItem(
                layer="SYNTHETIC",
                code="SYNTHETIC_DETECTOR_NEGATIVE",
                severity="INFO",
                message=f"{synth_model_name} classified voice as NATURAL (score={synth_val:.4f}, confidence={synth_confidence})."
            ))
        else:
            evidence.append(MultiLayerEvidenceItem(
                layer="SYNTHETIC",
                code="SYNTHETIC_DETECTOR_UNCERTAIN",
                severity="MEDIUM",
                message=f"{synth_model_name} score is UNCERTAIN (score={synth_val:.4f})."
            ))

        # Add Layer 2 (Speaker) Evidence
        if spk_decision == "MATCH":
            evidence.append(MultiLayerEvidenceItem(
                layer="SPEAKER",
                code="SPEAKER_MATCH",
                severity="INFO",
                message=f"{spk_model_name} confirmed speaker match (cosine similarity={spk_sim:.4f} >= 0.65 threshold)."
            ))
        elif spk_decision == "NON_MATCH":
            evidence.append(MultiLayerEvidenceItem(
                layer="SPEAKER",
                code="SPEAKER_NON_MATCH",
                severity="HIGH",
                message=f"{spk_model_name} rejected speaker match (cosine similarity={spk_sim:.4f} < 0.65 threshold)."
            ))
        else:
            evidence.append(MultiLayerEvidenceItem(
                layer="SPEAKER",
                code="SPEAKER_PROVISIONAL",
                severity="MEDIUM",
                message=f"{spk_model_name} similarity is provisional ({spk_sim:.4f})."
            ))

        # -------------------------------------------------------------
        # 2. Prosody Anomaly Modifiers
        # -------------------------------------------------------------
        prosody_mod = 0.0
        p_class = prosody_result.classification

        if p_class == ProsodyClassification.LOW_VARIATION:
            # Monotone / robotic pitch and narrow energy dynamics
            prosody_mod = +12.0
            evidence.append(MultiLayerEvidenceItem(
                layer="PROSODY",
                code="PROSODY_LOW_VARIATION_SUSPICION",
                severity="MEDIUM",
                message="Monotone pitch contour and narrow dynamic energy range detected. Corroborates potential text-to-speech synthesis."
            ))
        elif p_class == ProsodyClassification.UNUSUAL_PROSODY:
            # Erratic jitter / pitch discontinuities
            prosody_mod = +8.0
            evidence.append(MultiLayerEvidenceItem(
                layer="PROSODY",
                code="PROSODY_UNUSUAL_DYNAMICS",
                severity="MEDIUM",
                message="Atypical pitch jitter or speech rhythm discontinuities observed."
            ))
        elif p_class == ProsodyClassification.NATURAL_VARIATION:
            # Healthy human pitch variation
            prosody_mod = -8.0
            evidence.append(MultiLayerEvidenceItem(
                layer="PROSODY",
                code="PROSODY_NATURAL_DYNAMICS",
                severity="INFO",
                message="Acoustic pitch variance and speaking cadence exhibit healthy conversational modulation."
            ))
        elif p_class == ProsodyClassification.INSUFFICIENT_AUDIO:
            evidence.append(MultiLayerEvidenceItem(
                layer="PROSODY",
                code="PROSODY_INSUFFICIENT_EVALUATION",
                severity="LOW",
                message="Prosodic analysis limited due to short audio length or low active speech frames."
            ))

        # Copy underlying prosodic evidence items
        for pe in prosody_result.evidence:
            if pe.code not in ("PROSODY_NATURAL_MODULATION",):
                evidence.append(MultiLayerEvidenceItem(
                    layer="PROSODY",
                    code=pe.code,
                    severity=pe.severity.value,
                    message=pe.message,
                ))

        # -------------------------------------------------------------
        # 3. Contextual Sensitivity Integration
        # -------------------------------------------------------------
        ctx_score = context_result.context_risk_score
        ctx_multiplier = context_result.sensitivity_multiplier

        # Copy context evidence items
        for ce in context_result.evidence:
            evidence.append(MultiLayerEvidenceItem(
                layer="CONTEXT",
                code=ce.code,
                severity=ce.severity.value,
                message=ce.message,
            ))

        # Non-linear Multi-Layer Fusion
        # Acoustic base score is nudged by prosody
        acoustic_prosodic_risk = max(0.0, min(100.0, base_risk + prosody_mod))

        # High stakes context lowers tolerance for acoustic uncertainty:
        # If acoustic signals are suspicious (>40), context multiplier amplifies risk.
        # If acoustic signals are clean/natural (<20), context risk adds a minor policy overhead
        # but does not falsely label a real customer as an attacker.
        if acoustic_prosodic_risk >= 45.0:
            overall_raw = acoustic_prosodic_risk * (1.0 + (ctx_score / 200.0))
        elif acoustic_prosodic_risk >= 25.0:
            overall_raw = acoustic_prosodic_risk + (ctx_score * 0.25)
        else:
            # Clean acoustic / natural verified voice: context raises awareness but doesn't fake high clone risk
            overall_raw = acoustic_prosodic_risk + (ctx_score * 0.10)

        overall_score = max(0, min(100, int(round(overall_raw))))

        # -------------------------------------------------------------
        # 4. Map Risk Level & Context-Aware Action Policy (Part D)
        # -------------------------------------------------------------
        if overall_score < 25:
            risk_level = RiskLevel.LOW
        elif overall_score < 50:
            risk_level = RiskLevel.MEDIUM
        elif overall_score < 75:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL

        recommended_action, action_reason = cls._determine_action(
            risk_level=risk_level,
            overall_score=overall_score,
            context_meta=context_meta,
            is_synth=is_synth,
            is_spk_match=is_spk_match,
        )

        # Build Primary Explanatory Rationale
        primary_rationale = cls._build_primary_rationale(
            risk_level=risk_level,
            overall_score=overall_score,
            synth_val=synth_val,
            spk_sim=spk_sim,
            p_class=p_class,
            context_meta=context_meta,
            recommended_action=recommended_action,
        )

        # Signals breakdown
        synthetic_signal = {
            "score": round(synth_val, 4),
            "classification": synth_classification,
            "confidence_band": synth_confidence,
            "detector_model": synth_model_name,
            "score_type": "uncalibrated_model_score",
        }

        speaker_signal = {
            "similarity_score": round(spk_sim, 4),
            "decision": spk_decision,
            "confidence_band": spk_confidence,
            "model": spk_model_name,
            "profile_id": profile_id,
            "score_type": "cosine_similarity",
        }

        prosody_signal = {
            "classification": prosody_result.classification.value,
            "quality_score": round(prosody_result.quality_score, 2),
            "confidence": prosody_result.confidence,
            "features": prosody_result.features.model_dump(),
        }

        context_signal = {
            "context_risk_score": context_result.context_risk_score,
            "context_risk_level": context_result.risk_level.value,
            "policy_sensitivity": context_result.policy_sensitivity.value,
            "sensitivity_multiplier": context_result.sensitivity_multiplier,
            "call_type": context_meta.call_type.value,
            "caller_trust": context_meta.caller_trust.value,
            "requested_action": context_meta.requested_action.value,
            "transaction_amount": context_meta.transaction_amount,
            "historical_risk": context_meta.historical_risk.value,
        }

        return MultiLayerRiskAnalysisResult(
            overall_risk_score=overall_score,
            risk_level=risk_level,
            risk_score_type="heuristic_contextual_impersonation_risk",
            recommended_action=recommended_action,
            action_reason=action_reason,
            synthetic_signal=synthetic_signal,
            speaker_signal=speaker_signal,
            prosody_signal=prosody_signal,
            context_signal=context_signal,
            evidence=evidence,
            primary_rationale=primary_rationale,
            latency_ms=round(latency_ms, 2),
        )

    @classmethod
    def _determine_action(
        cls,
        risk_level: RiskLevel,
        overall_score: int,
        context_meta: ContextMetadata,
        is_synth: bool,
        is_spk_match: bool,
    ) -> Tuple[ExtendedRecommendedAction, str]:
        """
        Implements Part D context-aware policy sensitivity logic.
        """
        is_financial = (
            context_meta.call_type == CallType.FINANCIAL_TRANSACTION
            or context_meta.requested_action in (RequestedAction.FUND_TRANSFER, RequestedAction.PAYMENT_APPROVAL)
        )
        is_privileged = (
            context_meta.call_type == CallType.PRIVILEGED_ACCESS
            or context_meta.requested_action in (
                RequestedAction.CREDENTIAL_RESET,
                RequestedAction.PRIVILEGED_ACCESS_CHANGE,
                RequestedAction.SENSITIVE_DATA_DISCLOSURE,
            )
        )
        is_high_value = context_meta.transaction_amount is not None and context_meta.transaction_amount >= 50000

        if risk_level == RiskLevel.LOW:
            return (
                ExtendedRecommendedAction.ALLOW,
                "Acoustic and biometric verification indicators are within normal parameters. Interaction permitted."
            )

        elif risk_level == RiskLevel.MEDIUM:
            return (
                ExtendedRecommendedAction.MONITOR,
                "Moderate risk or unverified speaker identity. Maintain continuous behavioral monitoring and log interaction."
            )

        elif risk_level == RiskLevel.HIGH:
            if is_financial:
                return (
                    ExtendedRecommendedAction.CALL_BACK_REQUIRED,
                    "High voice cloning / impersonation risk on financial transaction. Out-of-band telephone call-back to registered primary number required before authorization."
                )
            elif is_privileged:
                return (
                    ExtendedRecommendedAction.MFA_REQUIRED,
                    "Elevated risk on privileged access request. Step-up multi-factor authentication (hardware token / push auth) required."
                )
            else:
                return (
                    ExtendedRecommendedAction.STEP_UP_VERIFICATION,
                    "Elevated impersonation risk. Secondary verification challenge required before proceeding."
                )

        else:  # CRITICAL
            if context_meta.requested_action == RequestedAction.FUND_TRANSFER or is_high_value:
                return (
                    ExtendedRecommendedAction.BLOCK_TRANSACTION_AND_ESCALATE,
                    "Critical voice-cloning impersonation attack detected on high-stakes fund transfer. Immediate transaction hold and security operations center escalation recommended."
                )
            else:
                return (
                    ExtendedRecommendedAction.BLOCK_OR_ESCALATE,
                    "Critical impersonation risk. Immediate session termination and fraud escalation recommended."
                )

    @classmethod
    def _build_primary_rationale(
        cls,
        risk_level: RiskLevel,
        overall_score: int,
        synth_val: float,
        spk_sim: float,
        p_class: ProsodyClassification,
        context_meta: ContextMetadata,
        recommended_action: ExtendedRecommendedAction,
    ) -> str:
        """Constructs an executive, plain-language summary for enterprise security operators."""
        summary_parts = []

        if overall_score >= 75:
            summary_parts.append(
                f"CRITICAL RISK ({overall_score}/100): High probability of an active voice cloning impersonation attack."
            )
        elif overall_score >= 50:
            summary_parts.append(
                f"HIGH RISK ({overall_score}/100): Elevated impersonation risk detected."
            )
        elif overall_score >= 25:
            summary_parts.append(
                f"MEDIUM RISK ({overall_score}/100): Moderate risk profile requiring monitoring."
            )
        else:
            summary_parts.append(
                f"LOW RISK ({overall_score}/100): Voice interaction exhibits legitimate acoustic characteristics."
            )

        # Explain key drivers
        drivers = []
        if synth_val >= 0.50:
            drivers.append(f"synthetic voice cues (AASIST={synth_val:.2f})")
        else:
            drivers.append(f"natural spectral characteristics (AASIST={synth_val:.2f})")

        if spk_sim >= 0.65:
            drivers.append(f"enrolled biometric match (similarity={spk_sim:.2f})")
        else:
            drivers.append(f"biometric mismatch with claimed profile (similarity={spk_sim:.2f})")

        if p_class == ProsodyClassification.LOW_VARIATION:
            drivers.append("monotone prosodic delivery")
        elif p_class == ProsodyClassification.UNUSUAL_PROSODY:
            drivers.append("unusual pitch jitter")

        if context_meta.call_type != CallType.NORMAL_CALL or context_meta.requested_action != RequestedAction.INFORMATION_ONLY:
            drivers.append(f"elevated business stakes ({context_meta.call_type.value}, {context_meta.requested_action.value})")

        summary_parts.append(f"Driven by {', '.join(drivers)}.")
        summary_parts.append(f"Action Policy: {recommended_action.value}.")

        return " ".join(summary_parts)
