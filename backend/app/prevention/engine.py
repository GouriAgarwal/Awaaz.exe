"""
Prevention Decision Engine.
Deterministic, explainable policy execution converting voice cloning risk and business context
into concrete operational prevention decisions.
"""
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

from app.prevention.schemas import (
    PreventionAction,
    PreventionStatus,
    SensitiveActionType,
    VerificationMethod,
    PolicyProfile,
    PreventionEvaluationRequest,
    PreventionEvaluationResponse,
    TimelineEvent,
)
from app.prevention.policies import get_policy_config


class PreventionDecisionEngine:
    """
    Deterministic rule-based prevention policy engine.
    Evaluates risk signals and contextual parameters against organizational policy profiles.
    """

    @classmethod
    def evaluate(cls, req: PreventionEvaluationRequest) -> Tuple[PreventionStatus, PreventionAction, List[PreventionAction], bool, bool, bool, List[VerificationMethod], str, List[Dict[str, Any]]]:
        """
        Evaluates prevention decision based on risk scores, context, and policy profile.
        Returns:
            (status, primary_action, required_actions, is_blocked, is_paused, requires_verification, methods, explanation, evidence)
        """
        policy = get_policy_config(req.policy_profile)
        evidence: List[Dict[str, Any]] = []
        required_actions: List[PreventionAction] = []
        verification_methods: List[VerificationMethod] = []

        is_blocked = False
        is_paused = False
        requires_verification = False

        score = req.risk_score
        action_type = req.sensitive_action_type
        amount = req.transaction_amount or 0.0

        is_financial = action_type in (SensitiveActionType.FUND_TRANSFER, SensitiveActionType.PAYMENT_APPROVAL)
        is_privileged = action_type in (SensitiveActionType.PRIVILEGED_ACCESS, SensitiveActionType.ACCOUNT_CHANGE)
        is_confidential = action_type == SensitiveActionType.CONFIDENTIAL_DISCLOSURE
        is_executive = action_type == SensitiveActionType.EXECUTIVE_INSTRUCTION

        is_high_value = amount >= policy.high_value_transaction_limit
        is_moderate_value = amount >= policy.moderate_value_transaction_limit

        # Record input evidence
        evidence.append({
            "code": "INPUT_RISK_EVALUATION",
            "risk_score": score,
            "risk_level": req.risk_level,
            "sensitive_action": action_type.value,
            "policy_profile": policy.profile.value,
        })

        if req.synthetic_score is not None:
            evidence.append({
                "code": "AASIST_SYNTHETIC_SIGNAL",
                "synthetic_score": round(req.synthetic_score, 4),
                "is_synthetic": req.synthetic_score >= 0.50,
            })

        if req.speaker_similarity is not None:
            evidence.append({
                "code": "ECAPA_SPEAKER_SIGNAL",
                "similarity": round(req.speaker_similarity, 4),
                "is_match": req.speaker_similarity >= 0.65,
            })

        # -------------------------------------------------------------
        # Tier 1: CRITICAL RISK (score >= policy.critical_risk_threshold)
        # -------------------------------------------------------------
        if score >= policy.critical_risk_threshold:
            evidence.append({
                "code": "CRITICAL_IMPERSONATION_RISK_THRESHOLD_CROSSED",
                "threshold": policy.critical_risk_threshold,
                "score": score,
            })

            if is_financial or is_privileged or is_executive or is_high_value:
                status = PreventionStatus.BLOCKED
                primary_action = PreventionAction.BLOCK_TRANSACTION
                is_blocked = True
                required_actions = [
                    PreventionAction.BLOCK_TRANSACTION,
                    PreventionAction.ESCALATE_TO_SOC,
                    PreventionAction.REQUIRE_CALLBACK,
                ]
                verification_methods = [
                    VerificationMethod.CALLBACK_TO_REGISTERED_NUMBER,
                    VerificationMethod.SUPERVISOR_APPROVAL,
                ]
                explanation = (
                    f"CRITICAL impersonation risk ({score}/100) detected during {action_type.value} "
                    f"under {policy.profile.value} policy. Immediate transaction block and Security Operations "
                    f"Center (SOC) escalation enforced."
                )
            else:
                status = PreventionStatus.ESCALATED
                primary_action = PreventionAction.ESCALATE_TO_SOC
                is_paused = True
                required_actions = [
                    PreventionAction.ESCALATE_TO_SOC,
                    PreventionAction.PAUSE_SENSITIVE_ACTION,
                    PreventionAction.REQUIRE_SECONDARY_VERIFICATION,
                ]
                verification_methods = [
                    VerificationMethod.OUT_OF_BAND_VERIFICATION,
                    VerificationMethod.SUPERVISOR_APPROVAL,
                ]
                explanation = (
                    f"CRITICAL risk ({score}/100) on {action_type.value}. Call interaction escalated to SOC; "
                    f"sensitive actions suspended."
                )

        # -------------------------------------------------------------
        # Tier 2: HIGH RISK (score >= policy.high_risk_threshold)
        # -------------------------------------------------------------
        elif score >= policy.high_risk_threshold:
            evidence.append({
                "code": "HIGH_IMPERSONATION_RISK_THRESHOLD_CROSSED",
                "threshold": policy.high_risk_threshold,
                "score": score,
            })

            if is_financial:
                status = PreventionStatus.PAUSED
                primary_action = PreventionAction.REQUIRE_CALLBACK
                is_paused = True
                requires_verification = True
                required_actions = [
                    PreventionAction.PAUSE_SENSITIVE_ACTION,
                    PreventionAction.REQUIRE_CALLBACK,
                ]
                verification_methods = [VerificationMethod.CALLBACK_TO_REGISTERED_NUMBER]

                if is_high_value:
                    required_actions.append(PreventionAction.ESCALATE_TO_SUPERVISOR)
                    verification_methods.append(VerificationMethod.SUPERVISOR_APPROVAL)
                    explanation = (
                        f"High voice cloning risk ({score}/100) detected on high-value transfer (${amount:,.2f}). "
                        f"Transaction paused. Out-of-band callback and supervisor sign-off required."
                    )
                else:
                    explanation = (
                        f"High voice cloning risk ({score}/100) on financial transaction. Outbound funds paused. "
                        f"Mandatory callback to primary registered number required."
                    )

            elif is_privileged:
                status = PreventionStatus.PAUSED
                primary_action = PreventionAction.REQUIRE_MFA
                is_paused = True
                requires_verification = True
                required_actions = [
                    PreventionAction.PAUSE_SENSITIVE_ACTION,
                    PreventionAction.REQUIRE_MFA,
                    PreventionAction.ESCALATE_TO_SUPERVISOR,
                ]
                verification_methods = [
                    VerificationMethod.MULTI_FACTOR_AUTHENTICATION,
                    VerificationMethod.SUPERVISOR_APPROVAL,
                ]
                explanation = (
                    f"High impersonation risk ({score}/100) on privileged access. Credential modification paused; "
                    f"step-up MFA and supervisor authorization required."
                )

            elif is_confidential:
                status = PreventionStatus.PAUSED
                primary_action = PreventionAction.PAUSE_SENSITIVE_ACTION
                is_paused = True
                requires_verification = True
                required_actions = [
                    PreventionAction.PAUSE_SENSITIVE_ACTION,
                    PreventionAction.REQUIRE_SECONDARY_VERIFICATION,
                ]
                verification_methods = [VerificationMethod.OUT_OF_BAND_VERIFICATION]
                explanation = (
                    f"High risk ({score}/100) on confidential information disclosure. Data release suspended "
                    f"pending out-of-band verification."
                )

            else:
                # General call with high voice risk
                status = PreventionStatus.WARNING
                primary_action = PreventionAction.SHOW_WARNING
                required_actions = [
                    PreventionAction.SHOW_WARNING,
                    PreventionAction.MONITOR,
                ]
                verification_methods = [VerificationMethod.SECURITY_QUESTION]
                explanation = (
                    f"High voice cloning suspicion ({score}/100) during general call. Operator warning active; "
                    f"continuous monitoring engaged."
                )

        # -------------------------------------------------------------
        # Tier 3: MEDIUM RISK (score >= 25)
        # -------------------------------------------------------------
        elif score >= 25:
            evidence.append({
                "code": "MEDIUM_RISK_EVALUATION",
                "score": score,
            })

            if is_financial or is_privileged or is_confidential or is_moderate_value:
                status = PreventionStatus.VERIFICATION_REQUIRED
                primary_action = PreventionAction.REQUIRE_SECONDARY_VERIFICATION
                requires_verification = True
                required_actions = [
                    PreventionAction.SHOW_WARNING,
                    PreventionAction.REQUIRE_SECONDARY_VERIFICATION,
                    PreventionAction.MONITOR,
                ]
                verification_methods = [
                    VerificationMethod.SECURITY_QUESTION,
                    VerificationMethod.OUT_OF_BAND_VERIFICATION,
                ]
                explanation = (
                    f"Moderate voice uncertainty ({score}/100) on sensitive action ({action_type.value}). "
                    f"Secondary identity verification challenge required before execution."
                )
            else:
                status = PreventionStatus.MONITORING
                primary_action = PreventionAction.MONITOR
                required_actions = [PreventionAction.MONITOR]
                explanation = (
                    f"Moderate risk profile ({score}/100). Low-impact action permitted with behavioral monitoring."
                )

        # -------------------------------------------------------------
        # Tier 4: LOW RISK (score < 25)
        # -------------------------------------------------------------
        else:
            evidence.append({
                "code": "LOW_RISK_AUTHORIZED",
                "score": score,
            })
            status = PreventionStatus.ALLOWED
            primary_action = PreventionAction.ALLOW
            required_actions = [PreventionAction.ALLOW, PreventionAction.MONITOR]
            explanation = (
                f"Low impersonation risk ({score}/100). Acoustic integrity and claimed identity within normal "
                f"authorized parameters. Action permitted."
            )

        return (
            status,
            primary_action,
            required_actions,
            is_blocked,
            is_paused,
            requires_verification,
            verification_methods,
            explanation,
            evidence,
        )
