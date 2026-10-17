"""
Test Suite for Step 15: Final Hackathon Product Integration, End-to-End Demo & Presentation Readiness.
Validates all 6 hackathon demonstration scenarios, multi-layer risk evaluation,
prevention state machine responses, verification simulation workflows, and privacy guarantees.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import create_application
from app.prevention.schemas import (
    PreventionAction,
    PreventionStatus,
    SensitiveActionType,
    VerificationMethod,
    PolicyProfile,
    PreventionEvaluationRequest,
    VerificationSimulationRequest,
    PreventionResolutionRequest,
)
from app.prevention.engine import PreventionDecisionEngine
from app.prevention.workflows import InvalidStateTransitionError
from app.prevention.service import PreventionService


@pytest.fixture
def client():
    app = create_application()
    return TestClient(app)


@pytest.fixture
def service():
    return PreventionService.get_instance()


# 1. Verify all 6 hackathon demonstration scenarios are available
def test_all_6_hackathon_scenarios_available(service):
    scenarios = service.get_demo_scenarios()
    assert len(scenarios) == 6, f"Expected 6 hackathon scenarios, got {len(scenarios)}"
    
    scenario_ids = [s.id for s in scenarios]
    assert "scenario_1_ceo_wire_transfer_attack" in scenario_ids
    assert "scenario_2_bank_customer_legitimate" in scenario_ids
    assert "scenario_3_replay_attack_enrolled" in scenario_ids
    assert "scenario_4_customer_care_social_eng" in scenario_ids
    assert "scenario_5_family_emergency_upi" in scenario_ids
    assert "scenario_6_it_privileged_access" in scenario_ids

    for s in scenarios:
        assert s.name.startswith("Scenario ")
        assert s.risk_score >= 0 and s.risk_score <= 100
        assert s.expected_status in PreventionStatus
        assert s.expected_action in PreventionAction
        assert len(s.description) > 20
        assert len(s.key_evidence) > 10


# 2. Scenario 1: CEO Urgent Wire Transfer Attack -> Auto-Blocked + SOC Escalation
def test_scenario_1_ceo_wire_transfer_blocked(service):
    req = PreventionEvaluationRequest(
        risk_score=94,
        risk_level="CRITICAL",
        synthetic_score=0.96,
        speaker_similarity=0.89,
        context_risk_score=92,
        sensitive_action_type=SensitiveActionType.FUND_TRANSFER,
        transaction_amount=150000.0,
        policy_profile=PolicyProfile.BANKING,
        caller_trust="VIP_OR_EXECUTIVE",
    )
    resp = service.evaluate_and_create_workflow(req)
    assert resp.prevention_status == PreventionStatus.BLOCKED
    assert resp.primary_action == PreventionAction.BLOCK_TRANSACTION
    assert resp.is_blocked is True
    assert PreventionAction.ESCALATE_TO_SOC in resp.required_actions
    assert "CRITICAL" in resp.explanation


# 3. Scenario 2: Bank Customer Legitimate Transfer -> Permitted
def test_scenario_2_bank_customer_legitimate_allowed(service):
    req = PreventionEvaluationRequest(
        risk_score=12,
        risk_level="LOW",
        synthetic_score=0.04,
        speaker_similarity=0.88,
        context_risk_score=10,
        sensitive_action_type=SensitiveActionType.FUND_TRANSFER,
        transaction_amount=25000.0,
        policy_profile=PolicyProfile.BANKING,
        caller_trust="VERIFIED_CONTACT",
    )
    resp = service.evaluate_and_create_workflow(req)
    assert resp.prevention_status == PreventionStatus.ALLOWED
    assert resp.primary_action == PreventionAction.ALLOW
    assert resp.is_blocked is False
    assert resp.is_paused is False
    assert resp.requires_verification is False


# 4. Scenario 3: Replay Attack on Enrolled Account -> Verification Required
def test_scenario_3_replay_attack_verification_required(service):
    req = PreventionEvaluationRequest(
        risk_score=38,
        risk_level="MEDIUM",
        synthetic_score=0.35,
        speaker_similarity=0.82,
        context_risk_score=45,
        sensitive_action_type=SensitiveActionType.PAYMENT_APPROVAL,
        transaction_amount=50000.0,
        policy_profile=PolicyProfile.BANKING,
        caller_trust="UNKNOWN_CALLER",
    )
    resp = service.evaluate_and_create_workflow(req)
    assert resp.prevention_status == PreventionStatus.VERIFICATION_REQUIRED
    assert resp.primary_action == PreventionAction.REQUIRE_SECONDARY_VERIFICATION
    assert resp.requires_verification is True
    assert resp.is_blocked is False


# 5. Scenario 4: Customer Care Social Engineering -> Paused with MFA
def test_scenario_4_social_engineering_password_paused(service):
    req = PreventionEvaluationRequest(
        risk_score=72,
        risk_level="HIGH",
        synthetic_score=0.86,
        speaker_similarity=0.20,
        context_risk_score=75,
        sensitive_action_type=SensitiveActionType.ACCOUNT_CHANGE,
        transaction_amount=None,
        policy_profile=PolicyProfile.TELECOM,
        caller_trust="UNKNOWN_CALLER",
    )
    resp = service.evaluate_and_create_workflow(req)
    assert resp.prevention_status == PreventionStatus.PAUSED
    assert resp.primary_action == PreventionAction.REQUIRE_MFA
    assert resp.is_paused is True
    assert resp.requires_verification is True


# 6. Scenario 5: Family Emergency Impersonation -> Paused with Callback
def test_scenario_5_family_emergency_paused(service):
    req = PreventionEvaluationRequest(
        risk_score=68,
        risk_level="HIGH",
        synthetic_score=0.82,
        speaker_similarity=0.45,
        context_risk_score=88,
        sensitive_action_type=SensitiveActionType.FUND_TRANSFER,
        transaction_amount=150000.0,
        policy_profile=PolicyProfile.BANKING,
        caller_trust="UNKNOWN_CALLER",
    )
    resp = service.evaluate_and_create_workflow(req)
    assert resp.prevention_status == PreventionStatus.PAUSED
    assert resp.primary_action == PreventionAction.REQUIRE_CALLBACK
    assert resp.is_paused is True
    assert VerificationMethod.CALLBACK_TO_REGISTERED_NUMBER in resp.recommended_verification_methods


# 7. Scenario 6: IT Support Privileged Access Request -> Critical Block
def test_scenario_6_it_privileged_access_blocked(service):
    req = PreventionEvaluationRequest(
        risk_score=89,
        risk_level="CRITICAL",
        synthetic_score=0.92,
        speaker_similarity=0.35,
        context_risk_score=90,
        sensitive_action_type=SensitiveActionType.PRIVILEGED_ACCESS,
        transaction_amount=None,
        policy_profile=PolicyProfile.ENTERPRISE,
        caller_trust="UNKNOWN_CALLER",
    )
    resp = service.evaluate_and_create_workflow(req)
    assert resp.prevention_status == PreventionStatus.BLOCKED
    assert resp.primary_action == PreventionAction.BLOCK_TRANSACTION
    assert resp.is_blocked is True
    assert PreventionAction.ESCALATE_TO_SOC in resp.required_actions


# 8. Story Mode: Verification Challenge Callback Success Unpauses Workflow
def test_verification_callback_passed_resolves_flow(service):
    req = PreventionEvaluationRequest(
        risk_score=68,
        risk_level="HIGH",
        sensitive_action_type=SensitiveActionType.FUND_TRANSFER,
        transaction_amount=50000.0,
        policy_profile=PolicyProfile.BANKING,
    )
    resp = service.evaluate_and_create_workflow(req)
    assert resp.prevention_status == PreventionStatus.PAUSED

    # Process callback passed simulation
    verified = service.process_verification(
        workflow_id=resp.workflow_id,
        verification_type="callback_passed",
        actor="telephony_callback_gateway",
        notes="Customer confirmed voice authenticity via out-of-band PIN",
    )
    assert verified.prevention_status == PreventionStatus.RESOLVED
    assert verified.primary_action == PreventionAction.ALLOW
    assert verified.is_paused is False


# 9. Story Mode: Step-Up MFA Challenge Success Unpauses Workflow
def test_verification_mfa_passed_resolves_flow(service):
    req = PreventionEvaluationRequest(
        risk_score=70,
        risk_level="HIGH",
        sensitive_action_type=SensitiveActionType.ACCOUNT_CHANGE,
        policy_profile=PolicyProfile.TELECOM,
    )
    resp = service.evaluate_and_create_workflow(req)
    assert resp.prevention_status == PreventionStatus.PAUSED

    verified = service.process_verification(
        workflow_id=resp.workflow_id,
        verification_type="mfa_passed",
        actor="auth_authenticator_app",
        notes="FIDO2 WebAuthn biometric passkey verified",
    )
    assert verified.prevention_status == PreventionStatus.RESOLVED
    assert verified.primary_action == PreventionAction.ALLOW
    assert verified.is_paused is False


# 10. Security Guard: Blocked Action Cannot be Cleared by Callback or MFA
def test_blocked_action_cannot_be_unblocked_by_callback_or_mfa(service):
    req = PreventionEvaluationRequest(
        risk_score=95,
        risk_level="CRITICAL",
        sensitive_action_type=SensitiveActionType.FUND_TRANSFER,
        transaction_amount=200000.0,
        policy_profile=PolicyProfile.BANKING,
    )
    resp = service.evaluate_and_create_workflow(req)
    assert resp.prevention_status == PreventionStatus.BLOCKED

    with pytest.raises(InvalidStateTransitionError):
        service.process_verification(
            workflow_id=resp.workflow_id,
            verification_type="callback_passed",
            actor="attacker_spoof_attempt",
        )

    with pytest.raises(InvalidStateTransitionError):
        service.process_verification(
            workflow_id=resp.workflow_id,
            verification_type="mfa_passed",
            actor="attacker_spoof_attempt",
        )


# 11. Supervisor Override Can Clear Blocked Action with Audit Trail
def test_supervisor_override_unblocks_critical_transaction(service):
    req = PreventionEvaluationRequest(
        risk_score=92,
        risk_level="CRITICAL",
        sensitive_action_type=SensitiveActionType.FUND_TRANSFER,
        transaction_amount=150000.0,
        policy_profile=PolicyProfile.BANKING,
    )
    resp = service.evaluate_and_create_workflow(req)
    assert resp.prevention_status == PreventionStatus.BLOCKED

    unblocked = service.process_verification(
        workflow_id=resp.workflow_id,
        verification_type="supervisor_approved",
        actor="chief_risk_officer",
        notes="Verified in-person identity emergency exemption protocol.",
    )
    assert unblocked.prevention_status == PreventionStatus.RESOLVED
    assert unblocked.primary_action == PreventionAction.ALLOW
    assert unblocked.is_blocked is False

    # Check timeline has supervisor audit record
    timeline_texts = [e.details for e in unblocked.timeline]
    assert any("chief_risk_officer" in t for t in timeline_texts)


# 12. Privacy Audit Compliance: No Raw Audio, Base64, or Biometric Embeddings in Timeline
def test_privacy_safeguards_in_incident_timeline(service):
    req = PreventionEvaluationRequest(
        risk_score=88,
        risk_level="CRITICAL",
        synthetic_score=0.94,
        speaker_similarity=0.88,
        context_risk_score=85,
        sensitive_action_type=SensitiveActionType.FUND_TRANSFER,
        transaction_amount=50000.0,
        policy_profile=PolicyProfile.BANKING,
    )
    resp = service.evaluate_and_create_workflow(req)
    for event in resp.timeline:
        details_lower = event.details.lower()
        # Verify no raw payload indicators
        assert "base64" not in details_lower
        assert "embedding" not in details_lower
        assert "waveform" not in details_lower
        assert "pcm" not in details_lower
        assert "audio/wav" not in details_lower


# 13. REST API Integration: End-to-End Command Center API Calls
def test_command_center_api_integration(client):
    # 1. Fetch 6 scenarios
    res = client.get("/api/prevention/demo/scenarios")
    assert res.status_code == 200
    scenarios = res.json()
    assert len(scenarios) == 6

    # 2. Evaluate Scenario 1 payload
    eval_res = client.post("/api/prevention/evaluate", json={
        "risk_score": 94,
        "risk_level": "CRITICAL",
        "synthetic_score": 0.96,
        "speaker_similarity": 0.89,
        "context_risk_score": 92,
        "sensitive_action_type": "FUND_TRANSFER",
        "transaction_amount": 150000.0,
        "policy_profile": "BANKING",
    })
    assert eval_res.status_code == 200
    eval_data = eval_res.json()
    wf_id = eval_data["workflow_id"]
    assert eval_data["prevention_status"] == "BLOCKED"
    assert eval_data["is_blocked"] is True

    # 3. Verify workflow retrieval
    wf_res = client.get(f"/api/prevention/workflows/{wf_id}")
    assert wf_res.status_code == 200
    assert wf_res.json()["workflow_id"] == wf_id

    # 4. Resolve workflow
    res_res = client.post(f"/api/prevention/workflows/{wf_id}/resolve", json={
        "resolution_reason": "Hackathon Demonstration Authorized Resolution",
        "resolved_by": "demo_lead_analyst",
        "final_action": "ALLOW",
    })
    assert res_res.status_code == 200
    assert res_res.json()["prevention_status"] == "RESOLVED"
