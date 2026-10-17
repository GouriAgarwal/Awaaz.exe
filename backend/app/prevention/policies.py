"""
Prevention Policy Configurations and Threshold Rules.
Defines risk bounds, action triggers, and profile-specific sensitivity matrices.
"""
from typing import Dict, Any, List
from app.prevention.schemas import (
    PolicyProfile,
    PreventionAction,
    PreventionStatus,
    SensitiveActionType,
    VerificationMethod,
)


class PolicyRuleConfig:
    """Configurable prevention rules for an organizational profile."""

    def __init__(
        self,
        profile: PolicyProfile,
        name: str,
        description: str,
        high_risk_threshold: int = 50,
        critical_risk_threshold: int = 75,
        high_value_transaction_limit: float = 50000.0,
        moderate_value_transaction_limit: float = 10000.0,
        auto_block_on_critical: bool = True,
        mandatory_callback_for_financial: bool = True,
        mandatory_mfa_for_privileged: bool = True,
    ):
        self.profile = profile
        self.name = name
        self.description = description
        self.high_risk_threshold = high_risk_threshold
        self.critical_risk_threshold = critical_risk_threshold
        self.high_value_transaction_limit = high_value_transaction_limit
        self.moderate_value_transaction_limit = moderate_value_transaction_limit
        self.auto_block_on_critical = auto_block_on_critical
        self.mandatory_callback_for_financial = mandatory_callback_for_financial
        self.mandatory_mfa_for_privileged = mandatory_mfa_for_privileged

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.value,
            "name": self.name,
            "description": self.description,
            "high_risk_threshold": self.high_risk_threshold,
            "critical_risk_threshold": self.critical_risk_threshold,
            "high_value_transaction_limit": self.high_value_transaction_limit,
            "moderate_value_transaction_limit": self.moderate_value_transaction_limit,
            "auto_block_on_critical": self.auto_block_on_critical,
            "mandatory_callback_for_financial": self.mandatory_callback_for_financial,
            "mandatory_mfa_for_privileged": self.mandatory_mfa_for_privileged,
        }


# Policy Profile Registry
POLICY_PROFILES: Dict[PolicyProfile, PolicyRuleConfig] = {
    PolicyProfile.BANKING: PolicyRuleConfig(
        profile=PolicyProfile.BANKING,
        name="Banking & Financial Fraud Shield",
        description="High sensitivity for monetary movements, wire transfers, and payments. Zero-trust on high-value transfers.",
        high_risk_threshold=45,
        critical_risk_threshold=70,
        high_value_transaction_limit=50000.0,
        moderate_value_transaction_limit=10000.0,
        auto_block_on_critical=True,
        mandatory_callback_for_financial=True,
        mandatory_mfa_for_privileged=True,
    ),
    PolicyProfile.ENTERPRISE: PolicyRuleConfig(
        profile=PolicyProfile.ENTERPRISE,
        name="Enterprise IT & Privileged Access Policy",
        description="Focused on preventing executive spear-phishing, IT helpdesk takeover, and admin credential resets.",
        high_risk_threshold=50,
        critical_risk_threshold=75,
        high_value_transaction_limit=100000.0,
        moderate_value_transaction_limit=25000.0,
        auto_block_on_critical=True,
        mandatory_callback_for_financial=True,
        mandatory_mfa_for_privileged=True,
    ),
    PolicyProfile.GOVERNMENT: PolicyRuleConfig(
        profile=PolicyProfile.GOVERNMENT,
        name="Government & Critical Infrastructure Policy",
        description="Ultra-strict protection against social engineering, confidential data leaks, and emergency instructions.",
        high_risk_threshold=40,
        critical_risk_threshold=65,
        high_value_transaction_limit=25000.0,
        moderate_value_transaction_limit=5000.0,
        auto_block_on_critical=True,
        mandatory_callback_for_financial=True,
        mandatory_mfa_for_privileged=True,
    ),
    PolicyProfile.TELECOM: PolicyRuleConfig(
        profile=PolicyProfile.TELECOM,
        name="Telecommunications SIM & Account Policy",
        description="Specialized in thwarting SIM-swap fraud, account porting, and call forwarding manipulations.",
        high_risk_threshold=50,
        critical_risk_threshold=75,
        high_value_transaction_limit=10000.0,
        moderate_value_transaction_limit=2000.0,
        auto_block_on_critical=True,
        mandatory_callback_for_financial=True,
        mandatory_mfa_for_privileged=True,
    ),
    PolicyProfile.DEFAULT: PolicyRuleConfig(
        profile=PolicyProfile.DEFAULT,
        name="Standard Balanced Prevention Policy",
        description="Standard balanced threshold matrix suitable for general corporate and contact center deployments.",
        high_risk_threshold=50,
        critical_risk_threshold=75,
        high_value_transaction_limit=50000.0,
        moderate_value_transaction_limit=10000.0,
        auto_block_on_critical=True,
        mandatory_callback_for_financial=True,
        mandatory_mfa_for_privileged=True,
    ),
}


def get_policy_config(profile: PolicyProfile) -> PolicyRuleConfig:
    """Retrieves policy configuration for a profile with safe fallback."""
    return POLICY_PROFILES.get(profile, POLICY_PROFILES[PolicyProfile.DEFAULT])
