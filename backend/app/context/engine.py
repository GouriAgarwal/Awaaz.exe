"""
Contextual Risk Engine.
Evaluates transactional, organizational, and authorization risk to enrich acoustic voice signals.
"""
from typing import List, Tuple, Optional
import logging

from app.context.schemas import (
    CallType,
    CallerTrust,
    RequestedAction,
    HistoricalRisk,
    ContextRiskLevel,
    PolicySensitivity,
    ContextEvidenceSeverity,
    ContextEvidenceItem,
    ContextMetadata,
    ContextRiskResult,
)

logger = logging.getLogger(__name__)


class ContextualRiskEngine:
    """
    Deterministic rule-based contextual risk engine.
    Assesses operational and financial exposure of voice interactions.
    """

    # Baseline score contributions by CallType
    CALL_TYPE_BASE = {
        CallType.NORMAL_CALL: 10,
        CallType.ENTERPRISE_APPROVAL: 30,
        CallType.FINANCIAL_TRANSACTION: 45,
        CallType.GOVERNMENT_INSTRUCTION: 50,
        CallType.PRIVILEGED_ACCESS: 60,
    }

    # Modifiers by CallerTrust
    TRUST_MODIFIERS = {
        CallerTrust.VERIFIED_CONTACT: -15,
        CallerTrust.KNOWN_CONTACT: -5,
        CallerTrust.UNKNOWN_CALLER: +15,
        CallerTrust.VIP_OR_EXECUTIVE: +20,  # High-value target for spear-phishing / CEO cloning
    }

    # Modifiers by RequestedAction
    ACTION_MODIFIERS = {
        RequestedAction.INFORMATION_ONLY: 0,
        RequestedAction.PAYMENT_APPROVAL: +20,
        RequestedAction.FUND_TRANSFER: +25,
        RequestedAction.SENSITIVE_DATA_DISCLOSURE: +25,
        RequestedAction.CREDENTIAL_RESET: +30,
        RequestedAction.PRIVILEGED_ACCESS_CHANGE: +35,
    }

    # Modifiers by HistoricalRisk
    HISTORICAL_MODIFIERS = {
        HistoricalRisk.NONE: 0,
        HistoricalRisk.LOW: +5,
        HistoricalRisk.MEDIUM: +15,
        HistoricalRisk.HIGH: +25,
    }

    @classmethod
    def evaluate(cls, meta: ContextMetadata) -> ContextRiskResult:
        """
        Calculates contextual risk score, risk level, policy sensitivity, and structured evidence.
        """
        evidence: List[ContextEvidenceItem] = []

        # 1. Base score from CallType
        base_score = cls.CALL_TYPE_BASE.get(meta.call_type, 15)
        if meta.call_type == CallType.PRIVILEGED_ACCESS:
            evidence.append(ContextEvidenceItem(
                code="CONTEXT_PRIVILEGED_CHANNEL",
                severity=ContextEvidenceSeverity.HIGH,
                message="Privileged system access channel requested; heightened authentication standards apply."
            ))
        elif meta.call_type == CallType.FINANCIAL_TRANSACTION:
            evidence.append(ContextEvidenceItem(
                code="CONTEXT_FINANCIAL_CHANNEL",
                severity=ContextEvidenceSeverity.MEDIUM,
                message="Financial transaction channel active; exposure to monetary fraud."
            ))
        elif meta.call_type == CallType.GOVERNMENT_INSTRUCTION:
            evidence.append(ContextEvidenceItem(
                code="CONTEXT_GOVERNMENT_CHANNEL",
                severity=ContextEvidenceSeverity.HIGH,
                message="Government authority or legal compliance instruction claimed."
            ))

        # 2. Caller Trust modifier
        trust_mod = cls.TRUST_MODIFIERS.get(meta.caller_trust, 10)
        if meta.caller_trust == CallerTrust.UNKNOWN_CALLER:
            evidence.append(ContextEvidenceItem(
                code="CONTEXT_UNKNOWN_CALLER",
                severity=ContextEvidenceSeverity.MEDIUM,
                message="Caller originates from an unrecognized or unverified telephone/endpoint identifier."
            ))
        elif meta.caller_trust == CallerTrust.VIP_OR_EXECUTIVE:
            evidence.append(ContextEvidenceItem(
                code="CONTEXT_EXECUTIVE_TARGET",
                severity=ContextEvidenceSeverity.HIGH,
                message="Target identity is a VIP/C-level executive; primary target for CEO voice cloning attacks."
            ))
        elif meta.caller_trust == CallerTrust.VERIFIED_CONTACT:
            evidence.append(ContextEvidenceItem(
                code="CONTEXT_VERIFIED_CALLER",
                severity=ContextEvidenceSeverity.INFO,
                message="Caller originated from a previously verified biometric / telephone channel."
            ))

        # 3. Requested Action modifier
        action_mod = cls.ACTION_MODIFIERS.get(meta.requested_action, 10)
        if meta.requested_action in (RequestedAction.CREDENTIAL_RESET, RequestedAction.PRIVILEGED_ACCESS_CHANGE):
            evidence.append(ContextEvidenceItem(
                code="CONTEXT_IDENTITY_CREDENTIAL_TAKEOVER",
                severity=ContextEvidenceSeverity.CRITICAL,
                message=f"Critical authorization event: {meta.requested_action.value}. High risk of account takeover."
            ))
        elif meta.requested_action in (RequestedAction.FUND_TRANSFER, RequestedAction.PAYMENT_APPROVAL):
            evidence.append(ContextEvidenceItem(
                code="CONTEXT_MONETARY_ACTION",
                severity=ContextEvidenceSeverity.HIGH,
                message=f"Monetary movement requested: {meta.requested_action.value}."
            ))

        # 4. Monetary Amount modifier
        amount_mod = 0
        if meta.transaction_amount is not None and meta.transaction_amount > 0:
            amt = float(meta.transaction_amount)
            if amt >= 50000:
                amount_mod = 30
                evidence.append(ContextEvidenceItem(
                    code="CONTEXT_VERY_HIGH_VALUE_TRANSACTION",
                    severity=ContextEvidenceSeverity.CRITICAL,
                    message=f"High-value monetary transaction requested (${amt:,.2f} >= $50,000.00)."
                ))
            elif amt >= 10000:
                amount_mod = 20
                evidence.append(ContextEvidenceItem(
                    code="CONTEXT_HIGH_VALUE_TRANSACTION",
                    severity=ContextEvidenceSeverity.HIGH,
                    message=f"Substantial financial transfer requested (${amt:,.2f} >= $10,000.00)."
                ))
            elif amt >= 1000:
                amount_mod = 10
                evidence.append(ContextEvidenceItem(
                    code="CONTEXT_MODERATE_VALUE_TRANSACTION",
                    severity=ContextEvidenceSeverity.MEDIUM,
                    message=f"Standard financial transfer requested (${amt:,.2f} >= $1,000.00)."
                ))

        # 5. Historical Risk modifier
        hist_mod = cls.HISTORICAL_MODIFIERS.get(meta.historical_risk, 0)
        if meta.historical_risk == HistoricalRisk.HIGH:
            evidence.append(ContextEvidenceItem(
                code="CONTEXT_HIGH_HISTORICAL_FRAUD_RECORD",
                severity=ContextEvidenceSeverity.HIGH,
                message="Account or telephone identifier has previous high-risk security alerts or fraud flags."
            ))
        elif meta.historical_risk == HistoricalRisk.MEDIUM:
            evidence.append(ContextEvidenceItem(
                code="CONTEXT_ELEVATED_HISTORICAL_RISK",
                severity=ContextEvidenceSeverity.MEDIUM,
                message="Account has elevated historical risk indicators."
            ))

        # 6. Aggregate Raw Score & Clamp [0, 100]
        raw_score = base_score + trust_mod + action_mod + amount_mod + hist_mod
        bounded_score = max(0, min(100, int(raw_score)))

        # 7. Map to Categorical Risk Level
        if bounded_score < 25:
            risk_level = ContextRiskLevel.LOW
            policy_sensitivity = PolicySensitivity.STANDARD
            sensitivity_multiplier = 1.0
        elif bounded_score < 50:
            risk_level = ContextRiskLevel.MEDIUM
            policy_sensitivity = PolicySensitivity.ELEVATED
            sensitivity_multiplier = 1.25
        elif bounded_score < 75:
            risk_level = ContextRiskLevel.HIGH
            policy_sensitivity = PolicySensitivity.HIGH_STAKES
            sensitivity_multiplier = 1.50
        else:
            risk_level = ContextRiskLevel.CRITICAL
            policy_sensitivity = PolicySensitivity.CRITICAL_DEFENSE
            sensitivity_multiplier = 2.0

        if not evidence:
            evidence.append(ContextEvidenceItem(
                code="CONTEXT_BASELINE",
                severity=ContextEvidenceSeverity.INFO,
                message="Normal interaction parameters without elevated business exposure flags."
            ))

        return ContextRiskResult(
            context_risk_score=bounded_score,
            risk_level=risk_level,
            policy_sensitivity=policy_sensitivity,
            sensitivity_multiplier=sensitivity_multiplier,
            evidence=evidence,
        )
